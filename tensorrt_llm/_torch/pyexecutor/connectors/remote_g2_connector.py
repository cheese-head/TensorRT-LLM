# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from .remote_g2 import (
    RemoteG2BindingRecord,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
    TargetRemoteG2BindingStore,
    TargetRemotePlanStore,
    _normalize_request_id,
    target_remote_g2_plan_store,
)
from .remote_g2_group import LoadStatus, TPGroup
from .remote_g2_observability import (
    NullRemoteG2ObservabilitySink,
    RemoteG2LifecycleEvent,
    RemoteG2ObservabilitySink,
)
from .remote_g2_transfer import RemoteG2TransferError


@dataclass(frozen=True)
class RemoteG2ConnectorMetadata:
    bindings: tuple[RemoteG2BindingRecord, ...] = ()


def _missing_release_lease(lease_id: str, reason: str) -> bool:
    return False


def _assert_partial_reuse_disabled(llm_args: Any) -> None:
    """Hard-fail at construction if ``kv_cache_config.enable_partial_reuse``
    is left on the default ``True`` while the remote-G2 connector is active.

    Partial reuse silently stops remote-G2 fetch from triggering on
    subsequent requests, so misconfiguration must surface as a startup error
    rather than as a quiet drop in hit rate. The connector class being
    constructed at all means the user has selected remote_g2 — no further
    gate is needed.
    """
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    if kv_cache_config is None:
        return
    if getattr(kv_cache_config, "enable_partial_reuse", False):
        raise RuntimeError(
            "remote_g2: kv_cache_config.enable_partial_reuse must be set to "
            "False when the remote-G2 connector is enabled (currently True, "
            "the default). Leaving it on prevents remote-G2 fetch from "
            "triggering on subsequent requests; the failure mode is silent "
            "(no error, just no remote-G2 hits)."
        )


# Module-state slots for callables installed from outside (typically by
# remote_g2_target_setup.maybe_start_remote_g2_target_client). The
# connector scheduler/worker read these lazily at call time so that
# installation order vs. construction order doesn't matter.
_installed_resolve_and_lease: Optional[
    Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]
] = None
_installed_release_lease: Optional[Callable[[str, str], bool]] = None
_installed_transfer_adapter: Optional[Any] = None
_installed_mark_local_valid: Optional[Callable[[RemoteG2BindingRecord], None]] = None
_installed_publish_binding: Optional[Callable[[RemoteG2BindingRecord], None]] = None
# Maps engine block_ids → primary-pool slot indices. Installed by the
# target setup once the KV cache manager is available. The binding store
# calls this when binding so NIXL's local dlist gets the right
# dense per-slot index instead of the engine's globally-unique block_id.
_installed_block_id_to_slot_idx: Optional[
    Callable[[list[int]], list[int]]
] = None


def install_block_id_to_slot_idx(
    fn: Callable[[list[int]], list[int]]
) -> None:
    global _installed_block_id_to_slot_idx
    _installed_block_id_to_slot_idx = fn


def install_resolve_and_lease(
    fn: Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]
) -> None:
    global _installed_resolve_and_lease
    _installed_resolve_and_lease = fn


def install_release_lease(fn: Callable[[str, str], bool]) -> None:
    global _installed_release_lease
    _installed_release_lease = fn


def install_transfer_adapter(adapter: Any) -> None:
    """Install the NIXL transfer adapter that the connector worker uses
    in start_load_kv. Read lazily by the worker so installation order
    relative to worker construction doesn't matter."""
    global _installed_transfer_adapter
    _installed_transfer_adapter = adapter


def install_mark_local_valid(fn: Callable[[RemoteG2BindingRecord], None]) -> None:
    global _installed_mark_local_valid
    _installed_mark_local_valid = fn


def install_publish_binding(fn: Callable[[RemoteG2BindingRecord], None]) -> None:
    global _installed_publish_binding
    _installed_publish_binding = fn


def _resolve_release_lease(explicit: Optional[Callable[[str, str], bool]]):
    """Return a callable that defers lookup to call time so module-state
    installation that happens after the scheduler/worker is constructed
    is still picked up."""

    def _call(lease_id: str, reason: str) -> bool:
        fn = explicit if explicit is not None else _installed_release_lease
        if fn is None:
            return _missing_release_lease(lease_id, reason)
        return fn(lease_id, reason)

    return _call


class _BoundedKeySet:
    """Insertion-ordered set with a hard size cap; evicts the oldest entry on
    overflow.

    Used for the worker's per-request dedup state (``_completed_loads``,
    ``_released_leases``). The only consumers of these entries are requests
    still being emitted by the scheduler, which are always recent; request ids
    increase monotonically and a finished request is never re-emitted. Evicting
    ids far older than any in-flight load is therefore safe, and the cap keeps
    the state from growing without bound on a long-running server (the worker
    has no per-request finish hook of its own).
    """

    __slots__ = ("_max", "_items")

    def __init__(self, max_size: int = 16384) -> None:
        self._max = max_size
        self._items: "OrderedDict[Any, None]" = OrderedDict()

    def add(self, key: Any) -> None:
        if key in self._items:
            self._items.move_to_end(key)
            return
        self._items[key] = None
        if len(self._items) > self._max:
            self._items.popitem(last=False)

    def discard(self, key: Any) -> None:
        self._items.pop(key, None)

    def __contains__(self, key: Any) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


class RemoteG2KvCacheConnectorScheduler(KvCacheConnectorScheduler):
    requires_retryable_kv_admission = True
    # Attention-DP and non-uniform/linear attention windows are not validated
    # with retryable remote-G2 admission yet. Overlap scheduler is allowed by
    # the fine-grained KVCM admission path from PR #6.
    requires_disable_overlap_scheduler = False
    requires_disable_attention_dp = True
    requires_uniform_attention_window = True

    def __init__(
        self,
        llm_args: Any,
        *,
        plan_store: Optional[TargetRemotePlanStore] = None,
        binding_store: Optional[TargetRemoteG2BindingStore] = None,
        resolve_and_lease: Optional[
            Callable[[RemoteKvReusePlan], RemoteG2ResolveResult]
        ] = None,
        release_lease: Optional[Callable[[str, str], bool]] = None,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        super().__init__(llm_args)
        _assert_partial_reuse_disabled(llm_args)
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._plan_store = (
            plan_store if plan_store is not None else target_remote_g2_plan_store()
        )
        self._explicit_resolve_and_lease = resolve_and_lease
        self._binding_store = (
            binding_store
            if binding_store is not None
            else TargetRemoteG2BindingStore(
                _resolve_release_lease(release_lease),
                observability=self._observability,
            )
        )
        # Request ids whose transfer-ready binding has already been emitted in
        # connector metadata. A binding only needs to reach the worker once —
        # the worker tracks _active_loads / _completed_loads and never re-issues
        # a transfer for an id it has already seen — so re-emitting it every
        # tick only inflates the per-tick MPI broadcast of build_connector_meta.
        # Cleared in request_finished, so it is bounded by concurrent requests.
        self._emitted_transfer_ready: set[int | str] = set()

    @property
    def _resolve_and_lease(
        self,
    ) -> Optional[Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]]:
        """Prefer the explicit kwarg, fall back to module-state install
        at access time so late installation is still picked up."""
        if self._explicit_resolve_and_lease is not None:
            return self._explicit_resolve_and_lease
        return _installed_resolve_and_lease

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        plan = self._plan_store.get(request.request_id)
        resolver = self._resolve_and_lease
        if plan is None or resolver is None:
            return (0, False)

        record = self._binding_store.resolve_for_request(
            request.request_id,
            plan,
            num_computed_tokens,
            resolver,
        )
        if record is None:
            return (0, False)
        return (record.matched_tokens, True)

    def update_state_after_alloc(self, request: Any, block_ids: list[int]) -> None:
        self._binding_store.bind_target_blocks(request.request_id, block_ids)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> RemoteG2ConnectorMetadata:
        # The official contract is to filter records by what's in
        # scheduler_output. Empirically that input arrives with the
        # newly-allocated request missing during the same tick the
        # request was bound, so the connector would never emit any
        # transfer-ready record. Scan binding_store directly instead —
        # is_transfer_ready already gates on (state=BOUND and bound_blocks),
        # and the worker tracks per-request _active_loads / _completed_loads
        # to avoid double-start, so we don't actually need scheduler_output
        # to dedupe.
        #
        # Emit each transfer-ready binding exactly once: a binding stays BOUND
        # (and thus transfer-ready) until request_finished, so without this
        # guard every still-running request's record would be re-included and
        # re-broadcast over MPI on every tick for the life of the request.
        bindings: list[RemoteG2BindingRecord] = []
        for _request_id, record in self._binding_store.iter_records():
            if record.is_transfer_ready and (
                record.request_id not in self._emitted_transfer_ready
            ):
                bindings.append(record)
                self._emitted_transfer_ready.add(record.request_id)
        return RemoteG2ConnectorMetadata(tuple(bindings))

    def request_finished(self, request: Any, cache_block_ids: list[int]) -> bool:
        self._binding_store.discard(request.request_id, "request_finished")
        self._plan_store.discard(request.request_id)
        self._emitted_transfer_ready.discard(_normalize_request_id(request.request_id))
        return False


class RemoteG2KvCacheConnectorWorker(KvCacheConnectorWorker):
    requires_retryable_kv_admission = True
    # Keep scheduler and worker capability flags aligned.
    requires_disable_overlap_scheduler = False
    requires_disable_attention_dp = True
    requires_uniform_attention_window = True

    def __init__(
        self,
        llm_args: Any,
        *,
        transfer_adapter: Optional[Any] = None,
        release_lease: Optional[Callable[[str, str], bool]] = None,
        mark_local_valid: Optional[Callable[[RemoteG2BindingRecord], None]] = None,
        publish_binding: Optional[Callable[[RemoteG2BindingRecord], None]] = None,
        transfer_timeout_ms: int = 30_000,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        super().__init__(llm_args)
        _assert_partial_reuse_disabled(llm_args)
        # Worker is constructed by PyExecutor with just llm_args, before
        # maybe_start_remote_g2_target_client runs - none of the
        # adapter / hooks can be wired at that point. Stash whatever was
        # passed explicitly; the accessor properties below fall back to
        # module-state slots at call time.
        self._explicit_transfer_adapter = transfer_adapter
        self._release_lease = _resolve_release_lease(release_lease)
        self._explicit_mark_local_valid = mark_local_valid
        self._explicit_publish_binding = publish_binding
        self._transfer_timeout_ms = transfer_timeout_ms
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._active_loads: dict[int | str, _RemoteG2ActiveLoad] = {}
        # Bounded dedup state: a completed load must not be re-started if the
        # scheduler's (possibly stale) metadata still carries its binding, and a
        # lease must not be released twice. The worker has no per-request finish
        # hook, so these are size-capped rather than cleared per request — see
        # _BoundedKeySet. Only in-flight (recent) requests are ever looked up.
        self._completed_loads = _BoundedKeySet()
        self._released_leases = _BoundedKeySet()
        # TP rank group. Under TP>1 every rank issues its own NIXL transfer for
        # a request; get_finished uses this to make failure a group decision so
        # a transfer error on any rank tears the load down on all ranks.
        self._tp_group = TPGroup.detect(llm_args)
        # Requests that may still need group reconciliation. Seeded from the
        # scheduler's started_loading_req_ids and trimmed only by the group
        # collective's output, so it is identical on every rank — that lets
        # get_finished gate (skip) the collective on idle steps without risking
        # a one-sided skip that would deadlock MPI. See get_finished step 2.
        self._coord_pending: set[int | str] = set()
        # Target-pool block ids of loads that failed since the last drain.
        # Surfaced to the executor via get_block_ids_with_load_errors() so it
        # can fall back to local recompute instead of waiting for the timeout.
        self._failed_block_ids: list[int] = []

    @property
    def _transfer_adapter(self) -> Optional[Any]:
        if self._explicit_transfer_adapter is not None:
            return self._explicit_transfer_adapter
        return _installed_transfer_adapter

    @property
    def _mark_local_valid(self) -> Optional[Callable[[RemoteG2BindingRecord], None]]:
        if self._explicit_mark_local_valid is not None:
            return self._explicit_mark_local_valid
        return _installed_mark_local_valid

    @property
    def _publish_binding(self) -> Optional[Callable[[RemoteG2BindingRecord], None]]:
        if self._explicit_publish_binding is not None:
            return self._explicit_publish_binding
        return _installed_publish_binding

    def register_kv_caches(self, kv_cache_tensor: Any) -> None:
        self._kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: Any) -> None:
        metadata = self.get_connector_meta()
        if not isinstance(metadata, RemoteG2ConnectorMetadata) or not metadata.bindings:
            return
        # A missing transfer adapter is a supported, non-fatal state: the target
        # setup deliberately leaves it uninstalled when the NIXL build fails
        # ("resolve still works, transfer disabled"). Surface the affected
        # blocks for local recompute and return — never raise into the executor
        # loop, which would take the whole engine down.
        if self._transfer_adapter is None:
            for record in metadata.bindings:
                self._fallback_record(record, "transfer_adapter_missing")
            return

        # Hand the adapter this rank's TP-local index from the single
        # authoritative (DP-aware) TPGroup, so its per-rank descriptor
        # indexing matches the worker's topology instead of re-detecting it.
        bind_local_rank = getattr(self._transfer_adapter, "bind_local_rank", None)
        if bind_local_rank is not None:
            bind_local_rank(self._tp_group.local_rank)

        for record in metadata.bindings:
            request_id = record.request_id
            if request_id in self._active_loads or request_id in self._completed_loads:
                continue
            try:
                result = self._transfer_adapter.start_transfer(record)
            except Exception:
                # A failed start falls back to local recompute for THIS record
                # (blocks recorded, lease released) and continues with the rest;
                # it must not abort the whole batch or raise into the executor.
                import logging as _logging
                _logging.exception(
                    "remote_g2: transfer failed to start (request=%s)", request_id
                )
                self._fallback_record(record, "transfer_start_failed")
                continue
            self._active_loads[request_id] = _RemoteG2ActiveLoad(
                result=result, started_at_ms=_now_ms()
            )

    def _fallback_record(self, record: RemoteG2BindingRecord, reason: str) -> None:
        """Degrade a record to local recompute: surface its target blocks via
        get_block_ids_with_load_errors() and release the lease once. Used by the
        start_load_kv failure paths so a transfer that can't start never hangs
        the request (which is suspended awaiting get_finished) and never raises.
        """
        self._record_failed_blocks(record)
        self._emit_record_event(
            "fallback",
            record,
            reason=reason,
            outcome="local_recompute",
        )
        self._release_record_once(record, reason)

    def wait_for_layer_load(self, layer_idx: int, stream: Any) -> None:
        return

    def save_kv_layer(self, layer_idx: int, stream: Any) -> None:
        return

    def wait_for_save(self, stream: Any) -> None:
        return

    def get_finished(
        self, finished_gen_req_ids: list[int], started_loading_req_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        # Iterate self._active_loads (all in-flight transfers), not just
        # started_loading_req_ids (NEW this tick). The connector framework's
        # get_finished moves the request from new_async_requests to
        # pending_async_requests on the first tick, so subsequent ticks call
        # us with empty started_loading_req_ids — without this iteration we'd
        # only ever poll each transfer ONCE, and slow/in-progress transfers
        # would never get reported as finished.
        #
        # Step 1: poll every local transfer. A raised exception, an explicit
        # NIXL failed state, or a timeout are all treated as a failure. We do
        # NOT raise into the executor loop — a single failed transfer must not
        # take down the engine.
        local_done: list[int | str] = []
        local_failed: dict[int | str, str] = {}
        for request_id in list(self._active_loads.keys()):
            active = self._active_loads.get(request_id)
            if active is None:
                continue
            status, reason = self._poll_status(active)
            if status is LoadStatus.DONE:
                local_done.append(request_id)
            elif status is LoadStatus.FAILED:
                local_failed[request_id] = reason or "transfer_failed"

        # Step 2: make failure a group decision. Under TP>1 each rank runs its
        # own transfer for the same request; a failure on ANY rank must fail
        # the request on ALL ranks, otherwise the framework's all-ranks
        # completion intersection would hang (the failed rank never reports the
        # request finished) while the other ranks keep a doomed transfer alive.
        #
        # The allgather is GATED so steps with no remote-G2 load anywhere in the
        # group skip the collective entirely. _coord_pending is mutated only by
        # globally-consistent signals (the scheduler's started_loading_req_ids,
        # which is replicated across TP ranks, and the collective's own output),
        # so the gate predicate is identical on every rank — ranks always enter
        # or skip the collective together; a one-sided skip would deadlock MPI.
        # Invariant: _coord_pending ⊇ this rank's active loads, so the collective
        # runs whenever any rank still has work to reconcile.
        # Keep ids in their native type (request ids may be int or str); the
        # rest of this method indexes _active_loads with that same type.
        newly_started: set[int | str] = set(started_loading_req_ids)
        self._coord_pending |= newly_started

        globally_failed: set[int | str] = set(local_failed.keys())
        if self._tp_group.enabled and self._coord_pending:
            globally_active: set[int | str] = set()
            for entry in self._tp_group.allgather(
                {
                    "failed": list(local_failed.keys()),
                    "active": list(self._active_loads.keys()),
                }
            ):
                # .get() tolerates an entry from a rank that has not yet been
                # upgraded to the {"failed","active"} payload (defensive against
                # rolling restarts); a missing key just contributes nothing.
                globally_failed.update(entry.get("failed", ()))
                globally_active.update(entry.get("active", ()))
            # A request leaves coordination once no rank holds it active (failed
            # ones are torn down below). Ids started this step are kept for one
            # full step so we never trim ahead of start_load_kv.
            self._coord_pending = {
                rid
                for rid in self._coord_pending
                if rid in newly_started
                or (rid in globally_active and rid not in globally_failed)
            }
        else:
            # Single-rank, or nothing pending group-wide: the local view is the
            # whole truth, so no collective is needed.
            self._coord_pending = {
                rid
                for rid in self._coord_pending
                if rid in newly_started or rid in self._active_loads
            } - globally_failed

        # Step 3: tear down every globally-failed load this rank still holds —
        # including in-flight ones a sibling failed and ones that completed
        # locally but lost the group (their blocks must be recomputed, not
        # published).
        for request_id in list(globally_failed):
            active = self._active_loads.get(request_id)
            if active is None:
                continue
            reason = local_failed.get(request_id, "sibling_transfer_failed")
            self._fail_active_load(request_id, active, reason)

        # Step 4: complete the locally-done loads that survived the group
        # decision. _complete_success emits the appropriate event and releases
        # the record on its own failure paths, so here we just surface the
        # blocks for recompute and drop the load without raising.
        finished_loading: list[int | str] = []
        for request_id in local_done:
            if request_id in globally_failed:
                continue
            active = self._active_loads.get(request_id)
            if active is None:
                continue
            record = active.result.record
            self._release_transfer_result_once(active.result)
            self._emit_record_event(
                "transferred",
                record,
                reason="ok",
                outcome="completed",
            )
            try:
                self._complete_success(active)
            except Exception:
                self._active_loads.pop(request_id, None)
                self._record_failed_blocks(record)
                continue
            self._active_loads.pop(request_id, None)
            self._completed_loads.add(request_id)
            # Return the id with its native type. _active_loads is keyed by
            # int|str (request ids may be non-numeric strings); int()-casting
            # here would raise inside the one method that must never raise.
            finished_loading.append(request_id)
        return ([], finished_loading)

    def _poll_status(
        self, active: "_RemoteG2ActiveLoad"
    ) -> tuple[LoadStatus, Optional[str]]:
        """Classify a single in-flight load as DONE / FAILED / IN_FLIGHT.

        A poll exception or an explicit NIXL failed state is a hard failure;
        exceeding the transfer timeout is also a failure. Anything else is
        still in flight.
        """
        try:
            completed = bool(active.result.is_completed())
        except Exception:
            return LoadStatus.FAILED, "transfer_failed"
        if completed:
            return LoadStatus.DONE, None
        if self._result_failed(active.result):
            return LoadStatus.FAILED, "transfer_failed"
        if _now_ms() - active.started_at_ms > self._transfer_timeout_ms:
            return LoadStatus.FAILED, "transfer_timeout"
        return LoadStatus.IN_FLIGHT, None

    def get_block_ids_with_load_errors(self) -> list[int]:
        """Return target-pool block ids whose remote-G2 load failed since the
        last call, then clear the internal list (drain-once-per-step).

        The executor unions these across ranks and falls back to local
        recompute for the affected requests. Reporting and clearing in one
        call guarantees a given failure is surfaced exactly once.
        """
        failed = self._failed_block_ids
        self._failed_block_ids = []
        return failed

    def abort_request(self, request_id: int | str) -> None:
        """Tear down an in-flight load for a cancelled/preempted request.

        Cancels the NIXL transfer (via the result's abort hook) and releases
        the lease exactly once. No-op for unknown ids or loads that already
        completed (and were moved out of _active_loads).
        """
        active = self._active_loads.pop(request_id, None)
        self._completed_loads.discard(request_id)
        if active is None:
            return
        result = active.result
        abort = getattr(result, "abort", None)
        if abort is not None:
            try:
                abort()
            except Exception:
                import logging as _logging
                _logging.exception("remote_g2: transfer abort failed")
        else:
            self._release_transfer_result_once(result)
        record = result.record
        self._emit_record_event(
            "fallback",
            record,
            reason="cancelled",
            outcome="aborted",
        )
        self._release_record_once(record, "cancelled")

    def _result_failed(self, result: Any) -> bool:
        """True when the transfer result reports an explicit failed state.

        Results predating the failed-state protocol expose only
        is_completed(); for them we return False and rely on the timeout.
        """
        is_failed = getattr(result, "is_failed", None)
        if is_failed is None:
            return False
        try:
            return bool(is_failed())
        except Exception:
            return True

    def _record_failed_blocks(self, record: RemoteG2BindingRecord) -> None:
        for block in record.bound_blocks:
            self._failed_block_ids.append(int(block.target_block_id))

    def _fail_active_load(
        self,
        request_id: int | str,
        active: "_RemoteG2ActiveLoad",
        reason: str,
    ) -> None:
        """Common teardown for a failed in-flight load: drop it, release the
        NIXL handle, record its blocks for recompute, and release the lease."""
        self._active_loads.pop(request_id, None)
        try:
            self._release_transfer_result_once(active.result)
        except Exception:
            import logging as _logging
            _logging.exception("remote_g2: release after failed transfer raised")
        record = active.result.record
        self._record_failed_blocks(record)
        self._emit_record_event(
            "fallback",
            record,
            reason=reason,
            outcome="local_recompute",
        )
        self._release_record_once(record, reason)

    def _complete_success(self, active: "_RemoteG2ActiveLoad") -> None:
        record = active.result.record
        if self._mark_local_valid is None:
            self._emit_record_event(
                "fallback",
                record,
                reason="local_validity_missing",
                outcome="local_recompute",
            )
            self._release_record_once(record, "local_validity_missing")
            raise RemoteG2TransferError("remote G2 local validity hook is not configured")
        if self._publish_binding is None:
            self._emit_record_event(
                "fallback",
                record,
                reason="publication_missing",
                outcome="local_recompute",
            )
            self._release_record_once(record, "publication_missing")
            raise RemoteG2TransferError("remote G2 publication hook is not configured")
        try:
            self._mark_local_valid(record)
            active.local_valid_marked = True
        except Exception:
            self._emit_record_event(
                "fallback",
                record,
                reason="local_validity_failed",
                outcome="local_recompute",
            )
            self._release_record_once(record, "local_validity_failed")
            raise
        try:
            self._publish_binding(record)
            active.published = True
        except Exception:
            self._emit_record_event(
                "failed",
                record,
                reason="publication_failed",
                outcome="fail_closed",
            )
            self._release_record_once(record, "publication_failed")
            raise
        self._release_record_once(record, "transfer_succeeded")

    def _release_record_once(self, record: RemoteG2BindingRecord, reason: str) -> bool:
        lease_id = record.lease_id
        if lease_id is None:
            self._emit_record_event(
                "released", record, reason=reason, outcome="no_lease"
            )
            return False
        if lease_id in self._released_leases:
            self._emit_record_event(
                "released", record, reason=reason, outcome="already_released"
            )
            return False
        self._released_leases.add(lease_id)
        completed = self._release_lease(lease_id, reason)
        self._emit_record_event(
            "released",
            record,
            reason=reason,
            outcome="completed" if completed else "already_released",
        )
        return completed

    def _release_transfer_result_once(self, result: Any) -> None:
        release = getattr(result, "release", None)
        if release is not None:
            release()

    def _emit_record_event(
        self,
        event: str,
        record: RemoteG2BindingRecord,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        self._observability.emit(
            RemoteG2LifecycleEvent(
                event=event,
                reason=reason,
                tier=record.plan.source_tier,
                outcome=outcome,
                request_id=record.request_id,
                plan_id=record.plan.plan_id,
                lease_id=record.lease_id,
                source_worker_id=record.plan.source_worker_id,
                source_generation=record.source_generation,
                block_count=len(record.bound_blocks),
                byte_count=sum(
                    block.source_descriptor.byte_length for block in record.bound_blocks
                ),
                token_count=record.matched_tokens,
            )
        )


@dataclass
class _RemoteG2ActiveLoad:
    result: Any
    started_at_ms: int
    local_valid_marked: bool = False
    published: bool = False


def _now_ms() -> int:
    return int(time.time() * 1000)
