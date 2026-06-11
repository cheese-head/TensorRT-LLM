# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Optional

from .remote_g2 import (
    SourceG2DescriptorRecord,
    SourceG2DescriptorRegistry,
    _is_remote_g2_tier,
)

_PRIMARY_CACHE_LEVEL = 0
_SECONDARY_CACHE_LEVEL = 1


class SourceG2PublisherEventAdapter:
    """Drives a SourceG2DescriptorRegistry from dynamo Publisher events.

    Only blocks landing on the secondary (host-pinned) tier are registered,
    matching the NIXL transfer adapter's G2->G1 constraint. Blocks moving off
    the secondary tier are removed from the registry; blocks moving onto it
    are upserted with their post-event slot.
    """

    def __init__(
        self,
        registry: SourceG2DescriptorRegistry,
        *,
        source_worker_id: int,
        source_dp_rank: int,
        block_size_bytes: int,
        secondary_pool_base_ptr: int,
        pool_id: str = "g2-host-pinned",
        tier: str = "host_pinned",
        device_id: int = 0,
    ) -> None:
        if block_size_bytes <= 0:
            raise ValueError("block_size_bytes must be positive")
        if secondary_pool_base_ptr <= 0:
            raise ValueError("secondary_pool_base_ptr must be a non-zero address")
        if not _is_remote_g2_tier(tier):
            raise ValueError(f"tier {tier!r} is not a remote G2 tier")
        if registry.source_worker_id != source_worker_id:
            raise ValueError("registry source_worker_id does not match adapter")
        if registry.source_dp_rank != source_dp_rank:
            raise ValueError("registry source_dp_rank does not match adapter")

        self._registry = registry
        self._source_worker_id = int(source_worker_id)
        self._source_dp_rank = int(source_dp_rank)
        self._block_size_bytes = int(block_size_bytes)
        self._secondary_pool_base_ptr = int(secondary_pool_base_ptr)
        self._pool_id = str(pool_id)
        self._tier = str(tier)
        self._device_id = int(device_id)
        self._lock = threading.Lock()
        self._descriptor_generation: dict[int, int] = {}

    def apply_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            return
        data = event.get("data")
        if not isinstance(data, Mapping):
            return
        kind = data.get("type")
        if kind == "stored":
            self._apply_stored(data)
        elif kind == "updated":
            self._apply_updated(data)
        elif kind == "removed":
            self._apply_removed(data)

    def _apply_stored(self, data: Mapping[str, Any]) -> None:
        blocks = data.get("blocks")
        if not blocks:
            return
        for block in blocks:
            cache_level = block.get("cache_level")
            if cache_level != _SECONDARY_CACHE_LEVEL:
                continue
            block_hash = block.get("block_hash")
            block_id = block.get("block_id")
            slot_idx = block.get("slot_idx")
            if block_hash is None or block_id is None or slot_idx is None:
                continue
            if int(block_id) < 0 or int(slot_idx) < 0:
                continue
            self._upsert(int(block_hash), int(block_id), int(slot_idx))

    def _apply_updated(self, data: Mapping[str, Any]) -> None:
        block_hash = data.get("block_hash")
        if block_hash is None:
            return
        cache_level = data.get("cache_level")
        new_level = (
            cache_level.get("new_value") if isinstance(cache_level, Mapping) else None
        )
        if new_level is None:
            return
        if int(new_level) == _SECONDARY_CACHE_LEVEL:
            block_id = data.get("block_id")
            new_slot_idx = data.get("new_slot_idx")
            if block_id is None or new_slot_idx is None:
                return
            if int(block_id) < 0 or int(new_slot_idx) < 0:
                return
            self._upsert(int(block_hash), int(block_id), int(new_slot_idx))
        else:
            self._forget(int(block_hash))

    def _apply_removed(self, data: Mapping[str, Any]) -> None:
        block_hashes = data.get("block_hashes")
        if not block_hashes:
            return
        for block_hash in block_hashes:
            self._forget(int(block_hash))

    def _upsert(self, block_hash: int, block_id: int, slot_idx: int) -> None:
        with self._lock:
            generation = self._descriptor_generation.get(block_hash, 0) + 1
            self._descriptor_generation[block_hash] = generation
        byte_offset = slot_idx * self._block_size_bytes
        ptr = self._secondary_pool_base_ptr + byte_offset
        record = SourceG2DescriptorRecord(
            block_hash=block_hash,
            source_worker_id=self._source_worker_id,
            source_dp_rank=self._source_dp_rank,
            tier=self._tier,
            descriptor_generation=generation,
            pool_id=self._pool_id,
            byte_offset=byte_offset,
            byte_length=self._block_size_bytes,
            block_id=block_id,
            metadata={
                "nixl_memory_desc": {
                    "ptr": ptr,
                    "size": self._block_size_bytes,
                    "device_id": self._device_id,
                    "memory_type": "DRAM",
                    "name": self._pool_id,
                }
            },
        )
        self._registry.upsert_descriptor(record)

    def _forget(self, block_hash: int) -> None:
        self._registry.remove_descriptor(block_hash)
        with self._lock:
            self._descriptor_generation.pop(block_hash, None)


def make_kv_pin_callbacks(
    kv_cache_manager: Any,
    *,
    secondary_pool_base_ptr: int,
    block_size_bytes: int,
) -> tuple[
    Callable[[SourceG2DescriptorRecord, str], int],
    Callable[[Optional[int]], None],
]:
    """Build (acquire_pin, release_pin) callbacks for SourceG2DescriptorRegistry.

    acquire_pin pins the block by id, validates the post-pin tier is the
    secondary pool (G2), and rewrites the record's byte_offset and
    nixl_memory_desc.ptr to reflect the authoritative post-pin slot. A block
    that has migrated off the secondary tier between event emission and pin
    acquisition is unpinned and rejected (caller's resolve attempt fails so
    the peer can fall back to local recompute).

    release_pin unpins the block by id. Sentinel pin_refs (None, negative)
    are no-ops to tolerate failed-acquire records that left a placeholder.
    """
    if block_size_bytes <= 0:
        raise ValueError("block_size_bytes must be positive")
    if secondary_pool_base_ptr <= 0:
        raise ValueError("secondary_pool_base_ptr must be a non-zero address")

    def acquire_pin(record: SourceG2DescriptorRecord, lease_id: str) -> int:
        block_id = int(record.block_id)
        if block_id < 0:
            raise ValueError(f"record block_id is invalid: {block_id}")

        # find_and_pin_blocks_by_hash already pinned the block atomically
        # under the lookup-tree mutex and waited for any in-flight offload DMA. We
        # only need to thread the block_id through to the lease so release_pin can
        # unpin it later. Pinning again would inflate the refcount and we'd have to
        # unpin twice on release, breaking the symmetry.
        if getattr(record, "_pinned_by_lookup", False):
            return block_id

        # Fallback path for records produced outside of
        # find_and_pin_descriptor_records (e.g. publisher-event-derived records
        # that weren't pinned at lookup). Pin here for host-pinned records only.
        locations = kv_cache_manager.pin_blocks_by_id([block_id])
        if not locations:
            raise RuntimeError(
                f"pin_blocks_by_id returned no locations for block_id={block_id}"
            )
        slot_idx, cache_level = locations[0]
        slot_idx = int(slot_idx)
        cache_level = int(cache_level)

        if cache_level != _SECONDARY_CACHE_LEVEL:
            kv_cache_manager.unpin_blocks_by_id([block_id])
            raise RuntimeError(
                f"block_id={block_id} pinned on cache_level={cache_level}, "
                f"expected {_SECONDARY_CACHE_LEVEL} (G2); refusing to serve"
            )

        record.byte_offset = slot_idx * block_size_bytes
        nixl_desc = record.metadata.get("nixl_memory_desc")
        if isinstance(nixl_desc, Mapping):
            new_ptr = secondary_pool_base_ptr + record.byte_offset
            record.metadata["nixl_memory_desc"] = {**nixl_desc, "ptr": new_ptr}
        return block_id

    def release_pin(pin_ref: Optional[int]) -> None:
        if pin_ref is None:
            return
        block_id = int(pin_ref)
        if block_id < 0:
            return
        kv_cache_manager.unpin_blocks_by_id([block_id])

    return acquire_pin, release_pin
