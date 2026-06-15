# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Sequence

from .remote_g2 import RemoteG2BindingRecord
from .remote_g2_transfer import RemoteG2TransferDescriptor, RemoteG2TransferError

def make_target_descriptor_resolver(
    kv_cache_manager: Any,
    *,
    primary_pool_base_ptr: int,
    block_size_bytes: int,
    window_size: int,
    device_id: int = 0,
    name_prefix: str = "g2-target",
) -> Callable[[RemoteG2BindingRecord], Sequence[RemoteG2TransferDescriptor]]:
    """Build a target_descriptor_resolver for RemoteG2NixlTransferAdapter.

    Reads the authoritative primary-pool slot of each target_block_id through
    the non-pinning KVCM accessor. Do not use pin_blocks_by_id here: by-id
    pinning can bypass radix-tree visibility and is intentionally disabled for
    remote-G2 on the PR #6 KVCM baseline.
    """
    if block_size_bytes <= 0:
        raise ValueError("block_size_bytes must be positive")
    if primary_pool_base_ptr <= 0:
        raise ValueError("primary_pool_base_ptr must be a non-zero address")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not hasattr(kv_cache_manager, "get_slot_idx_by_block_id"):
        raise ValueError("kv_cache_manager must expose get_slot_idx_by_block_id")

    def resolver(
        record: RemoteG2BindingRecord,
    ) -> Sequence[RemoteG2TransferDescriptor]:
        descriptors: list[RemoteG2TransferDescriptor] = []
        for bound_block in record.bound_blocks:
            block_id = int(bound_block.target_block_id)
            if block_id < 0:
                raise RemoteG2TransferError(
                    f"target block_id {block_id} is invalid"
                )

            try:
                slot_idx = int(
                    kv_cache_manager.get_slot_idx_by_block_id(
                        block_id, int(window_size)
                    )
                )
            except Exception as exc:
                raise RemoteG2TransferError(
                    f"target slot lookup failed for block_id={block_id}"
                ) from exc
            if slot_idx < 0:
                raise RemoteG2TransferError(
                    f"target slot lookup returned invalid slot {slot_idx} "
                    f"for block_id={block_id}"
                )

            ptr = primary_pool_base_ptr + slot_idx * block_size_bytes
            descriptors.append(
                RemoteG2TransferDescriptor(
                    ptr=ptr,
                    size=block_size_bytes,
                    device_id=device_id,
                    memory_type="VRAM",
                    name=f"{name_prefix}_{block_id}",
                )
            )
        return descriptors

    return resolver
