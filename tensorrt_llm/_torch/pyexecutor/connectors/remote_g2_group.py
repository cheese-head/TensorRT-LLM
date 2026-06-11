# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-group coordination primitives for the remote-G2 connector.

A remote-G2 KV load under tensor parallelism is a single *distributed*
operation: every TP rank issues its own NIXL transfer for the same request,
and the request only succeeds when all ranks succeed. Failure, abort, and
rank-to-peer indexing therefore have to be reasoned about at the level of the
rank group, not the individual rank.

`RankScope` is the one place that owns that group: which rank am I (TP-local,
DP-aware), how big is the group, and how do I reduce a per-rank value across
it. Keeping this in a single object means the worker's failure coordination,
the adapter's per-rank descriptor indexing, and any future sibling fan-out all
agree on the same topology instead of each re-deriving it from ``mpi_rank()``.

The collective used here (an allgather of *request ids*) is safe over the whole
MPI world even under attention-DP: request ids are globally unique, so a failed
id reported by one DP group is simply absent from another group's active loads
and tears down nothing there.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class LoadStatus(enum.Enum):
    """Status of a single rank's slice of a distributed KV load."""

    IN_FLIGHT = "in_flight"
    DONE = "done"
    FAILED = "failed"


def _tp_size_from_llm_args(llm_args: Any) -> Optional[int]:
    """Best-effort extraction of tensor_parallel_size from llm_args.

    Returns None when it cannot be determined; callers fall back to the MPI
    world size (correct when attention-DP is off, where world == TP group).
    """
    for attr in ("tensor_parallel_size", "tp_size"):
        val = getattr(llm_args, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    mapping = getattr(llm_args, "mapping", None) or getattr(
        llm_args, "parallel_config", None
    )
    if mapping is not None:
        for attr in ("tp_size", "tensor_parallel_size"):
            val = getattr(mapping, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    return None


@dataclass
class RankScope:
    """Topology + collective transport for one TP rank group.

    ``local_rank`` is the TP-local rank (equal to the MPI world rank when
    attention-DP is off) and is what indexes per-rank source descriptors.
    ``allgather`` reduces a per-rank value across the MPI world; it degrades to
    the identity ``[value]`` for a single-rank deployment so the TP=1 path is
    allocation- and collective-free.
    """

    world_rank: int
    world_size: int
    local_rank: int
    tp_size: int
    _allgather: Optional[Callable[[Any], list]] = field(default=None, repr=False)

    @property
    def enabled(self) -> bool:
        """True when there is more than one rank to coordinate with."""
        return self.world_size > 1

    @property
    def is_leader(self) -> bool:
        return self.local_rank == 0

    def allgather(self, value: Any) -> list:
        """Gather ``value`` from every rank into a list (one entry per rank).

        No-op (returns ``[value]``) for a single-rank deployment. An injected
        callable takes precedence so tests can drive the reduction without MPI.
        """
        if self._allgather is not None:
            return list(self._allgather(value))
        if not self.enabled:
            return [value]
        from tensorrt_llm._utils import mpi_allgather

        return list(mpi_allgather(value))

    @classmethod
    def detect(
        cls,
        llm_args: Any = None,
        *,
        allgather: Optional[Callable[[Any], list]] = None,
    ) -> "RankScope":
        """Build a RankScope from the live MPI environment.

        Falls back to a trivial single-rank scope if MPI is unavailable, so the
        connector never fails to construct in a non-distributed context.
        """
        try:
            from tensorrt_llm._utils import mpi_rank, mpi_world_size

            world_rank = int(mpi_rank())
            world_size = int(mpi_world_size())
        except Exception:
            logging.debug("remote_g2: RankScope.detect MPI unavailable; size=1")
            return cls(
                world_rank=0,
                world_size=1,
                local_rank=0,
                tp_size=1,
                _allgather=allgather,
            )

        attention_dp = (
            bool(getattr(llm_args, "enable_attention_dp", False))
            if llm_args is not None
            else False
        )
        tp_size = world_size
        if attention_dp:
            tp_size = _tp_size_from_llm_args(llm_args) or world_size
            tp_size = max(int(tp_size), 1)
        local_rank = world_rank % tp_size if tp_size > 0 else world_rank
        return cls(
            world_rank=world_rank,
            world_size=world_size,
            local_rank=local_rank,
            tp_size=tp_size,
            _allgather=allgather,
        )


# Cached process-wide scope for callers that only need the local rank (e.g. the
# transfer adapter indexing per-rank descriptors) and have no llm_args handle.
_cached_scope: Optional[RankScope] = None


def current_local_rank() -> int:
    """TP-local rank of this process, cached after first detection.

    Equivalent to ``mpi_rank()`` when attention-DP is off; routes through
    RankScope so per-rank indexing has a single source of truth.
    """
    global _cached_scope
    if _cached_scope is None:
        _cached_scope = RankScope.detect()
    return _cached_scope.local_rank
