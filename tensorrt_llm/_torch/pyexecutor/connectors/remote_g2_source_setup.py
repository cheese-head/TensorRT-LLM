# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the source-side SourceG2DescriptorRegistry from inside the
engine subprocess.

The KV cache manager (and the C++ APIs we depend on - find_and_pin_blocks_by_hash,
get_secondary_pool_data) only exists in the engine
subprocess that PyExecutor runs in. So the registry has to be built there.
The connector worker's register_kv_caches hook is the natural anchor: it
runs in that subprocess, right after the KV cache pool is allocated.

Identity (source_worker_id, source_dp_rank) is read from environment
variables that the dynamo worker process sets before spawning the engine.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from .remote_g2 import SourceG2DescriptorRegistry
from .remote_g2_source_adapter import make_kv_pin_callbacks


@dataclass
class _NixlSourceBundle:
    """Source-side NIXL agent + metadata captured for the metadata RPC.

    `agent` is the live NixlTransferAgent (kept alive for the worker's
    lifetime — its memory registrations and the daemon polling thread
    expire when the agent is GCed). `agent_desc` is the opaque
    bytes-blob other workers' agents need to call `load_remote_agent`.
    `remote_name` is the agent's identifier used as the third argument
    to `TransferRequest(..., remote_name)` from a peer.
    """

    agent: Any
    remote_name: str
    agent_desc: bytes
    pool_base_ptr: int
    pool_size_bytes: int
    source_generation: int = 1


# Process-wide singleton — populated by maybe_start_remote_g2_service after
# the NIXL agent is constructed, read by the ZMQ REP loop when answering
# get_metadata RPCs (Stage T2).
_GLOBAL_NIXL_SOURCE_BUNDLE: Optional[_NixlSourceBundle] = None

# Per-rank NIXL bundles gathered from all TP siblings at startup (S1).
# Indexed by TP rank. TP=1 has a single entry at index 0.
# Populated by _gather_per_rank_nixl_metadata() after each rank builds
# its local NIXL agent.
_GLOBAL_PER_RANK_NIXL_BUNDLES: Optional[list[dict]] = None


def get_nixl_source_bundle() -> Optional[_NixlSourceBundle]:
    return _GLOBAL_NIXL_SOURCE_BUNDLE


def get_per_rank_nixl_bundles() -> Optional[list[dict]]:
    """Return the per-rank NIXL metadata list gathered at startup (S1).

    Each entry is a dict with keys: remote_name, agent_metadata_b64,
    pool_base_ptr, pool_size_bytes, source_generation, tp_rank.
    Returns None if the gather hasn't completed or TP=1 hasn't been
    initialized yet.
    """
    return _GLOBAL_PER_RANK_NIXL_BUNDLES


def _gather_per_rank_nixl_metadata(
    local_bundle: Optional[_NixlSourceBundle],
    tp_rank: int,
    tp_size: int,
) -> Optional[list[dict]]:
    """S1: symmetric exchange of per-rank NIXL agent metadata at startup.

    Every TP rank calls this unconditionally — *including* a rank whose NIXL
    bundle failed to build (``local_bundle is None``). A missing bundle is
    carried as an ``ok=False`` payload, never a skipped collective, so a rank
    that failed locally can never leave its siblings blocked in the allgather.

    Under TP>1, if *any* rank could not build a bundle the whole group raises
    identically — a loud, consistent startup failure rather than a half-enabled
    transfer group. Under TP=1 a missing bundle simply disables transfer
    (returns ``None``); there is no group to diverge from.

    Uses the MPI world communicator (safe when DP is off, i.e. world == TP
    group). For DP-on deployments this must be scoped to the TP subgroup to
    avoid cross-DP-group contamination.
    """
    import base64 as _b64

    local_entry: Optional[dict] = None
    if local_bundle is not None:
        local_entry = {
            "tp_rank": tp_rank,
            "remote_name": local_bundle.remote_name,
            "agent_metadata_b64": _b64.b64encode(
                local_bundle.agent_desc
            ).decode("ascii"),
            "pool_base_ptr": local_bundle.pool_base_ptr,
            "pool_size_bytes": local_bundle.pool_size_bytes,
            "source_generation": local_bundle.source_generation,
        }

    if tp_size <= 1:
        return [local_entry] if local_entry is not None else None

    from .remote_g2_group import TPGroup

    # Build the group from the live MPI world so the collective can scope itself
    # to this rank's TP subgroup under attention-DP (where world > tp_size). The
    # entry is keyed by the TP-local rank (the value the target worker indexes by
    # via its own DP-aware TPGroup), not the world rank.
    try:
        from tensorrt_llm._utils import mpi_rank, mpi_world_size

        world_rank = int(mpi_rank())
        world_size = int(mpi_world_size())
    except Exception:
        world_rank, world_size = tp_rank, tp_size
    group = TPGroup(
        world_rank=world_rank,
        world_size=max(world_size, tp_size),
        local_rank=(world_rank % tp_size) if tp_size > 0 else world_rank,
        tp_size=tp_size,
    )
    all_ok, entries = group.agree_tp_subgroup(
        local_entry, ok=local_entry is not None
    )
    if not all_ok:
        raise RuntimeError(
            "remote_g2: S1 startup aborted — not every TP rank could build a "
            "NIXL source bundle; remote-G2 transfer cannot be enabled "
            "consistently across the TP group"
        )
    # Every entry is a real contribution when all_ok; sort for deterministic
    # per-rank indexing.
    entries = sorted(entries, key=lambda e: e["tp_rank"])

    logging.warning(
        "remote_g2: S1 per-rank NIXL metadata gathered: tp_size=%d "
        "ranks=[%s]",
        tp_size,
        ", ".join(f"{e['tp_rank']}:{e['remote_name']}" for e in entries),
    )
    return entries


def _result_to_dict(result: Any) -> dict:
    """Convert a RemoteG2ResolveResult dataclass into a plain dict for
    wire transport. The dynamo parent and downstream consumers see only
    dicts and do not need to import RemoteG2ResolveResult / its nested
    types.
    """
    descriptors = [
        {
            "block_hash": d.block_hash,
            "descriptor_generation": d.descriptor_generation,
            "pool_id": d.pool_id,
            "byte_offset": d.byte_offset,
            "byte_length": d.byte_length,
            "metadata": dict(d.metadata or {}),
        }
        for d in (result.descriptors or ())
    ]
    per_block_status = [
        {
            "block_hash": s.block_hash,
            "status": s.status,
            "descriptor_generation": s.descriptor_generation,
        }
        for s in (result.per_block_status or ())
    ]
    out = {
        "lease_id": result.lease_id,
        "descriptors": descriptors,
        "num_tokens": result.num_tokens,
        "reason": result.reason,
        "source_generation": result.source_generation,
        "per_block_status": per_block_status,
    }
    # S4: Include per-rank data when available (TP>1).
    per_rank_descs = getattr(result, "per_rank_descriptors", {})
    if per_rank_descs:
        out["per_rank_descriptors"] = per_rank_descs
    per_rank_meta = getattr(result, "per_rank_source_metadata", {})
    if per_rank_meta:
        out["per_rank_source_metadata"] = per_rank_meta
    return out


def _ipc_socket_path(dynamo_pid: int, tp_rank: int = 0, tp_size: int = 1) -> str:
    """Return the ZMQ IPC socket path for a given TP rank.

    TP=1: /tmp/dynamo_remote_g2_ipc_{pid}.sock (backward-compatible)
    TP>1: /tmp/dynamo_remote_g2_ipc_{pid}_tp{rank}.sock (per-rank)
    """
    if tp_size <= 1:
        return f"/tmp/dynamo_remote_g2_ipc_{dynamo_pid}.sock"
    return f"/tmp/dynamo_remote_g2_ipc_{dynamo_pid}_tp{tp_rank}.sock"


# Intra-pod sibling resolve RPC timeout. Kept short so one slow/dead sibling
# can stall the single-threaded REP loop for at most this long per sibling
# rather than the previous 5s.
_SIBLING_RPC_TIMEOUT_MS = 2000


def _query_sibling_rank(
    dynamo_pid: int,
    sibling_rank: int,
    tp_size: int,
    block_hashes: list[int],
    lease_id: Optional[str] = None,
    socket_cache: Optional[dict] = None,
) -> list[dict]:
    """Query a sibling TP rank's ZMQ REP for descriptors via intra-pod IPC.

    Returns a list of descriptor dicts (one per block_hash, None entries
    for blocks not found on that rank). Sub-millisecond — Unix domain
    socket on the same pod.

    When ``socket_cache`` is provided, the REQ socket is reused across calls
    (keyed by sibling rank) instead of being created and torn down per query.
    A REQ socket that errors/times out is left in an unusable state by ZMQ's
    strict send/recv FSM, so on any failure the socket is closed and evicted
    from the cache; the next call rebuilds a fresh one.

    The sibling pins each block it resolves (find-and-pin lookup). The
    lease_id is threaded through so the sibling can register those pins
    against the lease and reclaim them by TTL (see register_sibling_pins).
    """
    import zmq

    sibling_path = _ipc_socket_path(dynamo_pid, sibling_rank, tp_size)
    req = socket_cache.get(sibling_rank) if socket_cache is not None else None
    try:
        if req is None:
            ctx = zmq.Context.instance()
            req = ctx.socket(zmq.REQ)
            req.RCVTIMEO = _SIBLING_RPC_TIMEOUT_MS
            req.SNDTIMEO = _SIBLING_RPC_TIMEOUT_MS
            req.setsockopt(zmq.LINGER, 0)
            req.connect(f"ipc://{sibling_path}")
            if socket_cache is not None:
                socket_cache[sibling_rank] = req
        req.send(pickle.dumps({
            "method": "resolve_hashes",
            "payload": {"block_hashes": block_hashes, "lease_id": lease_id},
        }))
        raw = req.recv()
        resp = pickle.loads(raw)
        if resp.get("ok"):
            return resp.get("result", [])
        logging.warning(
            "remote_g2: sibling rank %d resolve_hashes returned not-ok: %s",
            sibling_rank, resp.get("error"),
        )
        return []
    except Exception:
        logging.exception(
            "remote_g2: sibling rank %d IPC query failed (path=%s)",
            sibling_rank, sibling_path,
        )
        # The REQ socket is now in a broken state; drop it so the next query
        # rebuilds a fresh, reusable one.
        if req is not None:
            try:
                req.close(linger=0)
            except Exception:
                pass
        if socket_cache is not None:
            socket_cache.pop(sibling_rank, None)
        return []
    finally:
        # Only close per-call when not pooling; pooled sockets stay open for
        # reuse (and are closed above on error).
        if socket_cache is None and req is not None:
            req.close()


def _descs_from_records(records: list) -> list:
    """Serialize resolved descriptor records into the wire dict shape used by
    the per-rank gather (None entries pass through for misses)."""
    descs = []
    for record in records:
        if record is None:
            descs.append(None)
        else:
            descs.append({
                "block_hash": record.block_hash,
                "byte_offset": record.byte_offset,
                "byte_length": record.byte_length,
                "pool_id": record.pool_id,
                "metadata": dict(record.metadata or {}),
            })
    return descs


def _start_zmq_rep_service(
    registry: SourceG2DescriptorRegistry,
    dynamo_pid: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> str:
    """Start a ZMQ REP daemon thread bound to a Unix domain socket.

    Every TP rank binds its own socket (per-rank path). Rank 0's handler
    serves external resolve RPCs AND queries sibling ranks via intra-pod
    ZMQ IPC. Non-rank-0 handlers only serve intra-pod queries
    (resolve_hashes method) from rank 0.

    Returns the socket path for logging.
    """
    import zmq

    socket_path = _ipc_socket_path(dynamo_pid, tp_rank, tp_size)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"ipc://{socket_path}")

    # Sibling ranks pin blocks while serving rank 0's intra-pod resolve_hashes
    # and reclaim them purely by TTL (registry.register_sibling_pins records
    # them on the leader's lease clock). A background sweeper guarantees idle
    # ranks still reclaim; the request path also sweeps lazily. Harmless no-op
    # on the leader, which holds no sibling pins.
    if tp_size > 1:
        sweep_interval_s = max(1.0, min(registry.lease_ttl_ms / 2000.0, 10.0))

        def _sibling_pin_sweeper() -> None:
            while True:
                time.sleep(sweep_interval_s)
                try:
                    registry.expire_sibling_pins()
                except Exception:
                    logging.exception("remote_g2: sibling pin sweeper error")

        threading.Thread(
            target=_sibling_pin_sweeper,
            name=f"remote_g2_pin_sweeper_r{tp_rank}",
            daemon=True,
        ).start()

    # Persistent per-sibling REQ sockets reused across resolves. Only rank 0's
    # _loop thread touches this, so no lock is needed.
    sibling_socket_cache: dict = {}

    def _loop() -> None:
        while True:
            method = "<unparsed>"
            try:
                raw = rep.recv()
            except Exception:
                logging.exception("remote_g2: ZMQ REP recv failed; exiting loop")
                return
            try:
                req = pickle.loads(raw)
                method = req.get("method")
                payload = req.get("payload") or {}
                if method == "resolve_hashes":
                    # Lazy TTL sweep on the resolve path so active ranks reclaim
                    # promptly without waiting for the background sweeper tick.
                    # (Only the resolve methods touch sibling pins; skip it on
                    # get_metadata / release_lease.)
                    registry.expire_sibling_pins()
                    # Intra-pod query from rank 0: look up the whole prefix on
                    # THIS rank's registry in one batched find-and-pin call and
                    # return descriptors. Each hit is pinned; register the pins
                    # on the lease so they are reclaimed by TTL (no explicit
                    # release on the sibling path — see register_sibling_pins).
                    hashes = payload.get("block_hashes", [])
                    lease_id = payload.get("lease_id")
                    records = registry.find_and_pin_descriptor_records(hashes)
                    if lease_id is not None:
                        registry.register_sibling_pins(
                            lease_id,
                            [int(r.block_id) for r in records if r is not None],
                        )
                    response = {"ok": True, "result": _descs_from_records(records)}
                elif method == "resolve_and_lease":
                    registry.expire_sibling_pins()
                    result = registry.resolve_and_lease(payload.get("plan"))
                    result_dict = _result_to_dict(result)

                    # Intra-pod per-rank gather: query sibling ranks
                    # via ZMQ IPC (no MPI, no dynamo RPC).
                    if tp_size > 1 and result.reason == "ok" and result.descriptors:
                        block_hashes = [
                            d.block_hash for d in result.descriptors
                        ]
                        expected_ranks = set(range(tp_size))
                        expected_blocks = len(block_hashes)
                        per_rank_descs = {tp_rank: [
                            {
                                "block_hash": d.block_hash,
                                "byte_offset": d.byte_offset,
                                "byte_length": d.byte_length,
                                "pool_id": d.pool_id,
                                "metadata": dict(d.metadata or {}),
                            }
                            for d in result.descriptors
                        ]}
                        incomplete_reason: Optional[str] = None
                        # Query each sibling rank via local ZMQ IPC. The
                        # lease_id lets each sibling register the pins it takes
                        # under that lease so they expire on the lease's TTL.
                        for sibling in range(tp_size):
                            if sibling == tp_rank:
                                continue
                            sibling_descs = _query_sibling_rank(
                                dynamo_pid, sibling, tp_size, block_hashes,
                                lease_id=result.lease_id,
                                socket_cache=sibling_socket_cache,
                            )
                            if sibling_descs and len(sibling_descs) == expected_blocks:
                                per_rank_descs[sibling] = sibling_descs
                            else:
                                incomplete_reason = "incomplete_tp_rank_descriptors"
                                logging.warning(
                                    "remote_g2: TP resolve failed closed; "
                                    "rank %d returned %d/%d descriptors",
                                    sibling,
                                    len(sibling_descs or []),
                                    expected_blocks,
                                )
                                break
                        if incomplete_reason is None and set(per_rank_descs) != expected_ranks:
                            incomplete_reason = "incomplete_tp_rank_descriptors"

                        # Attach per-rank source metadata from S1. Every TP rank
                        # needs its matching source agent metadata; falling back
                        # to rank 0 would read the wrong KV shard.
                        per_rank_bundles = get_per_rank_nixl_bundles()
                        per_rank_meta = {
                            entry["tp_rank"]: entry
                            for entry in (per_rank_bundles or [])
                        }
                        if incomplete_reason is None and set(per_rank_meta) != expected_ranks:
                            incomplete_reason = "incomplete_tp_rank_metadata"
                            logging.warning(
                                "remote_g2: TP resolve failed closed; "
                                "metadata ranks=%s expected=%s",
                                sorted(per_rank_meta),
                                sorted(expected_ranks),
                            )

                        if incomplete_reason is not None:
                            if result.lease_id is not None:
                                registry.release_lease(result.lease_id, incomplete_reason)
                            result_dict.update({
                                "lease_id": None,
                                "descriptors": [],
                                "num_tokens": 0,
                                "reason": incomplete_reason,
                                "per_rank_descriptors": {},
                                "per_rank_source_metadata": {},
                            })
                            response = {"ok": True, "result": result_dict}
                            rep.send(pickle.dumps(response))
                            continue

                        result_dict["per_rank_descriptors"] = per_rank_descs
                        logging.info(
                            "remote_g2: intra-pod gather: %d ranks, "
                            "%d blocks each",
                            len(per_rank_descs),
                            len(block_hashes),
                        )

                        result_dict["per_rank_source_metadata"] = per_rank_meta

                    response = {"ok": True, "result": result_dict}
                elif method == "release_lease":
                    # Releases the leader's own lease pins immediately. Sibling
                    # pins are not released here — they expire by TTL on the same
                    # clock as this lease (see register_sibling_pins).
                    completed = registry.release_lease(
                        payload["lease_id"], payload.get("reason", "ack")
                    )
                    response = {"ok": True, "result": completed}
                elif method == "get_metadata":
                    bundle = get_nixl_source_bundle()
                    per_rank = get_per_rank_nixl_bundles()
                    if bundle is None:
                        response = {
                            "ok": False,
                            "error": "nixl_source_bundle_not_ready",
                        }
                    else:
                        # Bidirectional peer load (raw NIXL): if the
                        # caller sent its serialized agent metadata,
                        # call add_remote_agent on it BEFORE returning
                        # our own metadata.
                        import base64 as _b64
                        peer_metadata_b64 = payload.get("peer_metadata_b64")
                        if peer_metadata_b64:
                            try:
                                peer_bytes = _b64.b64decode(peer_metadata_b64)
                                loaded_name = bundle.agent.add_remote_agent(peer_bytes)
                                logging.warning(
                                    "remote_g2: source add_remote_agent "
                                    "loaded peer name=%s (bytes=%d)",
                                    loaded_name, len(peer_bytes),
                                )
                            except Exception:
                                logging.exception(
                                    "remote_g2: source add_remote_agent failed"
                                )
                        # Rank 0's own metadata (backward-compatible with
                        # TP=1 callers that don't read per_rank_metadata).
                        result = {
                            "source_worker_id": registry.source_worker_id,
                            "source_dp_rank": registry.source_dp_rank,
                            "source_generation": bundle.source_generation,
                            "remote_name": bundle.remote_name,
                            "agent_metadata_b64": _b64.b64encode(
                                bundle.agent_desc
                            ).decode("ascii"),
                            "pool_base_ptr": bundle.pool_base_ptr,
                            "pool_size_bytes": bundle.pool_size_bytes,
                        }
                        # S5 — per-rank metadata for TP>1 targets.
                        if per_rank is not None:
                            result["per_rank_metadata"] = per_rank
                        response = {"ok": True, "result": result}
                else:
                    response = {"ok": False, "error": f"unknown method: {method!r}"}
            except Exception as exc:
                logging.exception("remote_g2: ZMQ REP handler raised")
                response = {"ok": False, "error": repr(exc)}
            try:
                rep.send(pickle.dumps(response))
            except Exception:
                logging.exception("remote_g2: ZMQ REP send failed")

    thread = threading.Thread(target=_loop, name="remote_g2_zmq_rep", daemon=True)
    thread.start()
    return socket_path


def _walk_to_dynamo_worker_pid(max_depth: int = 10) -> Optional[int]:
    """Walk up the process tree from this process and return the first
    ancestor whose cmdline mentions 'dynamo.trtllm'. OpenMPI's orted
    strips arbitrary env vars when spawning ranks, so the engine
    subprocess can't read DYNAMO_REMOTE_G2_WORKER_ID directly; this
    helper finds the dynamo parent so we can read a sidecar file
    /tmp/dynamo_remote_g2_worker_<pid>.txt instead.

    When TP=1 (no MPI spawn), the engine runs inline in the dynamo
    process itself — there is no parent to walk to. In that case we
    check if the current process IS the dynamo process and return our
    own PID.
    """
    try:
        my_pid = os.getpid()
        # TP=1 fast path: check if *this* process is the dynamo worker
        # (no MPI subprocess when tensor_parallel_size == 1).
        try:
            with open(f"/proc/{my_pid}/cmdline") as f:
                my_cmdline = f.read().replace("\0", " ")
        except (FileNotFoundError, PermissionError):
            my_cmdline = ""
        if "dynamo.trtllm" in my_cmdline or "dynamo/trtllm" in my_cmdline:
            return my_pid

        # TP>1 path: walk ancestors to find the dynamo parent.
        pid = my_pid
        for _ in range(max_depth):
            try:
                with open(f"/proc/{pid}/status") as f:
                    status = f.read()
            except (FileNotFoundError, PermissionError):
                return None
            ppid = None
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid is None or ppid < 1:
                return None
            try:
                with open(f"/proc/{ppid}/cmdline") as f:
                    cmdline = f.read().replace("\0", " ")
            except (FileNotFoundError, PermissionError):
                cmdline = ""
            if "dynamo.trtllm" in cmdline or "dynamo/trtllm" in cmdline:
                return ppid
            pid = ppid
        return None
    except Exception:
        return None


def _resolve_source_identity() -> Optional[tuple[int, int]]:
    """Return (source_worker_id, dynamo_parent_pid) on success, None when
    the dynamo identity cannot be reached from the engine subprocess.

    Reads the worker_id from env var first, falling back to a sidecar
    file written by the dynamo parent (since OpenMPI orted strips env
    vars across the spawn boundary). The dynamo parent's PID is also
    needed so the ZMQ REP service can bind to a Unix domain socket the
    parent can find.
    """
    dynamo_pid = _walk_to_dynamo_worker_pid()
    env_value = os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID")
    if env_value and dynamo_pid is not None:
        try:
            return int(env_value), dynamo_pid
        except ValueError:
            pass
    if dynamo_pid is None:
        return None
    sidecar = f"/tmp/dynamo_remote_g2_worker_{dynamo_pid}.txt"
    try:
        with open(sidecar) as f:
            worker_id = int(f.read().strip())
        return worker_id, dynamo_pid
    except Exception:
        return None


def is_remote_g2_configured() -> bool:
    """Return whether this engine subprocess is expected to run remote-G2.

    MPI-spawned ranks may lose the original env var, so use the same env/sidecar
    identity resolution as source bootstrap instead of checking os.environ only.
    """
    return _resolve_source_identity() is not None


def _get_secondary_pool(kv: Any) -> Any:
    """Return the unsliced secondary pool tensor, falling back to
    get_secondary_pool_data(0) on older TRT-LLM builds that lack
    get_unique_secondary_pool().
    """
    if hasattr(kv, "get_unique_secondary_pool"):
        return kv.get_unique_secondary_pool()
    # Fallback: layer-0 slice — data_ptr coincides with allocation base
    # under block-major layout (the standard for transformer models).
    return kv.get_secondary_pool_data(0)


def _secondary_pool_base_ptr(kv: Any) -> int:
    """Return the secondary KV cache pool's base host address, or 0 when
    not available (no host pool allocated, exposure binding missing, etc.).
    """
    try:
        pool = _get_secondary_pool(kv)
        if pool is None or pool.numel() == 0:
            return 0
        return int(pool.data_ptr())
    except Exception as exc:
        logging.warning("remote_g2: _get_secondary_pool raised: %r", exc)
        return 0


def _derive_block_size_bytes(kv: Any) -> Optional[int]:
    """Compute per-LOGICAL-block byte size from the unsliced secondary
    pool tensor (parity with how the source-side NIXL agent registers
    its memory).

    Uses ``get_unique_secondary_pool()`` — the unsliced full secondary
    pool with shape ``(num_blocks, num_layers, kv_factor, blockSize)``
    for block-major layouts. One logical block occupies
    ``num_layers × kv_factor × blockSize × element_size`` bytes, which
    is the same as ``element_size × prod(shape[1:])``.

    Layout assumption: standard transformers run block-major. For
    recurrent-state / linear-attention models the first dim is
    num_layers instead of num_blocks and this derivation would be
    wrong; the modulo sanity check at the end catches that case as a
    fail-loud rather than a silent corruption.
    """
    try:
        pool = _get_secondary_pool(kv)
    except Exception:
        return None
    if pool is None or pool.numel() == 0 or pool.ndim < 2:
        return None

    used_fallback = not hasattr(kv, "get_unique_secondary_pool")
    per_block_elems = 1
    for d in pool.shape[1:]:
        per_block_elems *= int(d)
    if per_block_elems <= 0:
        return None

    per_block_bytes = int(pool.element_size()) * per_block_elems

    # When using get_secondary_pool_data(0) fallback, the tensor is a
    # per-layer slice — multiply by num_pools (== num_layers) to get
    # the full logical block size across all layers.
    if used_fallback:
        try:
            num_pools = int(kv.num_pools)
        except Exception:
            return None
        per_block_bytes *= num_pools

    total_bytes = int(pool.element_size()) * int(pool.numel())
    if used_fallback:
        total_bytes *= num_pools
    if total_bytes % per_block_bytes != 0:
        return None

    return per_block_bytes


def _derive_window_size(kv: Any) -> Optional[int]:
    """Return the attention window size the KV cache manager is configured
    with. Reads from KvCacheIterationStats keys; under the single-window-
    block-manager constraint this connector operates under there is
    exactly one entry."""
    try:
        iter_stats = kv.get_iteration_stats()
    except Exception:
        return None
    if not iter_stats:
        return None
    try:
        return int(next(iter(iter_stats.keys())))
    except (StopIteration, TypeError, ValueError):
        return None


def _pool_size_bytes(kv: Any) -> Optional[int]:
    """Total byte size of the host_pinned secondary pool covering all
    layers. We register this whole range with the NIXL agent so remote
    workers can issue READs against any slot in it.

    Uses ``get_unique_secondary_pool()`` — the unsliced full secondary
    pool tensor — so the total size is the direct
    ``element_size × numel`` of the underlying allocation, without
    needing to multiply by num_layers manually.
    """
    try:
        pool = _get_secondary_pool(kv)
    except Exception:
        return None
    if pool is None or pool.numel() == 0:
        return None
    size = int(pool.element_size() * pool.numel())
    # Fallback pool is per-layer; multiply by num_pools for full size.
    if not hasattr(kv, "get_unique_secondary_pool"):
        try:
            size *= int(kv.num_pools)
        except Exception:
            return None
    return size


def _setup_nixl_source_agent(
    pool_base_ptr: int,
    pool_size_bytes: int,
    source_worker_id: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Optional[_NixlSourceBundle]:
    """Build a raw nixl_agent on the source side and register the
    host_pinned secondary pool memory range so it can be read remotely.

    With TP>1, each rank builds its own agent with a rank-qualified
    name so peers can load multiple source agents (one per TP rank).
    """
    if tp_size > 1:
        agent_name = f"remote-g2-source-{source_worker_id}-tp{tp_rank}"
    else:
        agent_name = f"remote-g2-source-{source_worker_id}"
    from .remote_g2_raw_nixl_adapter import build_raw_nixl_source_agent

    handle = build_raw_nixl_source_agent(
        agent_name=agent_name,
        pool_base_ptr=pool_base_ptr,
        pool_size_bytes=pool_size_bytes,
    )
    if handle is None:
        return None
    # build_raw_nixl_source_agent has already constructed the agent,
    # registered the host_pinned pool, and captured the agent metadata
    # bytes. Mirror those values into _NixlSourceBundle so the existing
    # metadata RPC handler can read them via the same field names it
    # used to use with the TRT-LLM wrapper.
    return _NixlSourceBundle(
        agent=handle.agent,
        remote_name=handle.agent_name,
        agent_desc=handle.agent_metadata,
        pool_base_ptr=handle.pool_base_ptr,
        pool_size_bytes=handle.pool_size_bytes,
        source_generation=handle.source_generation,
    )


def maybe_start_remote_g2_service(
    kv: Any,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    lease_ttl_ms: int = 30_000,
    pool_id: str = "g2-host-pinned",
    tier: str = "host_pinned",
) -> Optional[SourceG2DescriptorRegistry]:
    """Start the source-side remote-G2 service against a live kv_cache_manager.

    Called from PyExecutor right after kv_cache_manager is constructed,
    inside the engine subprocess. Builds a SourceG2DescriptorRegistry
    and (in future iterations) spawns a daemon thread that exposes it
    over ZMQ for the dynamo parent process to forward RPC calls into.

    Returns None when the deployment isn't configured for remote-G2
    (env var missing), or when prerequisites aren't met (no secondary
    pool, no host_pinned blocks yet, etc.). Caller treats None as
    "remote-G2 service not started".

    Identity (source_worker_id, source_dp_rank) is read from env vars
    set by the dynamo parent process:
      - DYNAMO_REMOTE_G2_WORKER_ID  (required; matches dynamo
        endpoint.connection_id() for the owning dynamo worker process)
      - DYNAMO_REMOTE_G2_DP_RANK    (defaults to 0)

    TP>1 support (S1-S3):
      - tp_rank / tp_size control MPI gather of per-rank NIXL metadata.
      - Rank 0 runs the ZMQ REP server and answers resolve RPCs.
      - Non-rank-0 ranks participate in MPI gather during resolve (S3)
        but do NOT run a ZMQ server.
    """
    # Auto-detect tp_rank/tp_size from MPI when not passed explicitly.
    # This handles deployments where py_executor.py doesn't pass the
    # TP info (e.g. patched connectors without patched py_executor).
    if tp_rank == 0 and tp_size == 1:
        try:
            from tensorrt_llm._utils import mpi_rank, mpi_world_size
            detected_rank = mpi_rank()
            detected_size = mpi_world_size()
            if detected_size > 1:
                tp_rank = detected_rank
                tp_size = detected_size
                logging.warning(
                    "remote_g2: auto-detected TP from MPI: "
                    "tp_rank=%d tp_size=%d",
                    tp_rank, tp_size,
                )
        except Exception:
            pass

    identity = _resolve_source_identity()
    if identity is None:
        logging.info(
            "remote_g2: source registry skipped "
            "(DYNAMO_REMOTE_G2_WORKER_ID not reachable via env var or sidecar)"
        )
        return None
    source_worker_id, dynamo_pid = identity

    # PyExecutor.kv_cache_manager is a Python wrapper class
    # (resource_manager.KVCacheManager); the C++ binding with
    # get_secondary_pool_data / find_and_pin_blocks_by_hash
    # sits at .impl. Unwrap once so the rest of the code (and the
    # SourceG2DescriptorRegistry it builds) talks to the C++ object
    # directly.
    kv = getattr(kv, "impl", kv)
    try:
        source_dp_rank = int(os.environ.get("DYNAMO_REMOTE_G2_DP_RANK", "0"))
    except ValueError:
        source_dp_rank = 0

    pool_base_ptr = _secondary_pool_base_ptr(kv)
    if pool_base_ptr == 0:
        logging.info(
            "remote_g2: source registry skipped (secondary pool unavailable)"
        )
        return None

    block_size_bytes = _derive_block_size_bytes(kv)
    if not block_size_bytes or block_size_bytes <= 0:
        logging.warning(
            "remote_g2: source registry skipped (block_size_bytes unknown)"
        )
        return None

    window_size = _derive_window_size(kv)
    if window_size is None or window_size <= 0:
        logging.warning(
            "remote_g2: source registry skipped (window_size unknown)"
        )
        return None

    acquire_pin, release_pin = make_kv_pin_callbacks(
        kv,
        secondary_pool_base_ptr=pool_base_ptr,
        block_size_bytes=block_size_bytes,
    )

    registry = SourceG2DescriptorRegistry(
        source_worker_id=source_worker_id,
        source_dp_rank=source_dp_rank,
        lease_ttl_ms=lease_ttl_ms,
        acquire_pin=acquire_pin,
        release_pin=release_pin,
        require_trtllm_pin=True,
        kv=kv,
        window_size=window_size,
        pool_id=pool_id,
        pool_base_ptr=pool_base_ptr,
        block_size_bytes=block_size_bytes,
        tier=tier,
    )

    logging.warning(
        "remote_g2: service started "
        "(source_worker_id=%s dp_rank=%s window_size=%s "
        "block_size_bytes=%s pool_base_ptr=0x%x)",
        source_worker_id,
        source_dp_rank,
        window_size,
        block_size_bytes,
        pool_base_ptr,
    )

    # ZMQ REP server — every rank binds its own socket for intra-pod
    # IPC queries. Rank 0 serves external resolve RPCs from the dynamo
    # parent AND queries sibling ranks via their sockets. Non-rank-0
    # ranks serve only intra-pod resolve_hashes queries from rank 0.
    try:
        socket_path = _start_zmq_rep_service(
            registry, dynamo_pid,
            tp_rank=tp_rank, tp_size=tp_size,
        )
        logging.warning(
            "remote_g2: ZMQ REP service bound at %s "
            "(source_worker_id=%s tp_rank=%d tp_size=%d)",
            socket_path,
            source_worker_id,
            tp_rank,
            tp_size,
        )
    except Exception:
        logging.exception(
            "remote_g2: failed to start ZMQ REP service; registry built but "
            "not reachable"
        )
        raise

    # Stage T1 — bootstrap the NIXL agent and register the host_pinned
    # secondary pool. Each TP rank builds its own agent with a rank-
    # qualified name. The bundle is stashed as a process-wide singleton
    # so the ZMQ REP service can answer the get_metadata RPC (Stage T2).
    global _GLOBAL_NIXL_SOURCE_BUNDLE, _GLOBAL_PER_RANK_NIXL_BUNDLES
    pool_size_bytes = _pool_size_bytes(kv)
    bundle: Optional[_NixlSourceBundle] = None
    if not pool_size_bytes or pool_size_bytes <= 0:
        logging.warning(
            "remote_g2: NIXL agent skipped (pool_size_bytes unknown)"
        )
    else:
        # A bootstrap raise must not let one rank skip the S1 collective below
        # while its siblings enter it; collapse any failure to bundle=None so
        # the group decision stays symmetric.
        try:
            bundle = _setup_nixl_source_agent(
                pool_base_ptr=pool_base_ptr,
                pool_size_bytes=pool_size_bytes,
                source_worker_id=source_worker_id,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        except Exception:
            logging.exception("remote_g2: NIXL source agent bootstrap raised")
            bundle = None
        if bundle is not None:
            _GLOBAL_NIXL_SOURCE_BUNDLE = bundle
            logging.warning(
                "remote_g2: source NIXL agent built: agent_name=%s "
                "pool_base_ptr=0x%x pool_size=%d tp_rank=%d",
                bundle.remote_name,
                bundle.pool_base_ptr,
                bundle.pool_size_bytes,
                tp_rank,
            )
        else:
            logging.warning(
                "remote_g2: NIXL source agent bootstrap failed; "
                "resolve will still work, but transfer is disabled"
            )
            raise RuntimeError("remote_g2: NIXL source agent bootstrap failed")

    # S1 — exchange per-rank NIXL metadata symmetrically. Every rank reaches
    # this call regardless of whether its own bundle built, so a one-sided
    # skip can never deadlock the collective. Under TP>1 a missing bundle on
    # any rank raises identically on all ranks (hard-fail). Allowed to
    # propagate so a misconfigured transfer group fails loudly at startup.
    _GLOBAL_PER_RANK_NIXL_BUNDLES = _gather_per_rank_nixl_metadata(
        bundle, tp_rank=tp_rank, tp_size=tp_size,
    )

    # Per-rank resolve is now handled via intra-pod ZMQ IPC (no MPI).
    # Each rank's ZMQ REP serves resolve_hashes queries from rank 0.
    # No S3 sibling loop needed.

    return registry
