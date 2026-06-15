# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_CONNECTOR_DIR = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
)
_PATHS = {
    "remote_g2_observability": _CONNECTOR_DIR / "remote_g2_observability.py",
    "remote_g2": _CONNECTOR_DIR / "remote_g2.py",
    "remote_g2_transfer": _CONNECTOR_DIR / "remote_g2_transfer.py",
    "remote_g2_target_resolver": _CONNECTOR_DIR / "remote_g2_target_resolver.py",
}


def _install_package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)
    loaded = {}
    for name, path in _PATHS.items():
        loaded[name] = _load_module(f"{_CONNECTOR_PACKAGE}.{name}", path)
    return loaded


_MODS = _load_modules()
RemoteG2BoundBlock = _MODS["remote_g2"].RemoteG2BoundBlock
RemoteG2BindingRecord = _MODS["remote_g2"].RemoteG2BindingRecord
RemoteG2Descriptor = _MODS["remote_g2"].RemoteG2Descriptor
RemoteG2BindingState = _MODS["remote_g2"].RemoteG2BindingState
RemoteG2ResolveResult = _MODS["remote_g2"].RemoteG2ResolveResult
RemoteKvReusePlan = _MODS["remote_g2"].RemoteKvReusePlan
RemoteG2TransferError = _MODS["remote_g2_transfer"].RemoteG2TransferError
make_target_descriptor_resolver = _MODS[
    "remote_g2_target_resolver"
].make_target_descriptor_resolver

BLOCK_SIZE_BYTES = 4096
PRIMARY_POOL_BASE_PTR = 0x20_0000_0000


class _FakeKvCacheManager:
    def __init__(self, slot_locations):
        self._slot_locations = dict(slot_locations)
        self.slot_calls: list[tuple[int, int]] = []

    def get_slot_idx_by_block_id(self, block_id, window_size):
        self.slot_calls.append((int(block_id), int(window_size)))
        return self._slot_locations[int(block_id)]


def _make_plan(num_blocks):
    import time

    now_ms = int(time.time() * 1000)
    return RemoteKvReusePlan(
        plan_id="p",
        request_id="r",
        target_worker_id=99,
        target_dp_rank=1,
        source_worker_id=7,
        source_dp_rank=0,
        source_tier="host_pinned",
        block_hashes=tuple(100 + i for i in range(num_blocks)),
        start_block_index=0,
        planned_prefix_blocks=num_blocks,
        block_size_tokens=16,
        created_at_ms=now_ms,
        expires_at_ms=now_ms + 60_000,
    )


def _make_record(target_block_ids):
    plan = _make_plan(len(target_block_ids))
    descriptors = tuple(
        RemoteG2Descriptor(
            block_hash=h,
            descriptor_generation=1,
            pool_id="g2",
            byte_offset=0,
            byte_length=BLOCK_SIZE_BYTES,
        )
        for h in plan.block_hashes
    )
    resolve_result = RemoteG2ResolveResult(
        lease_id="lease-x",
        descriptors=descriptors,
        num_tokens=len(descriptors) * plan.block_size_tokens,
    )
    bound_blocks = tuple(
        RemoteG2BoundBlock(
            source_descriptor=descriptor,
            target_block_id=target_block_id,
            source_block_index=i,
            target_block_index=i,
        )
        for i, (descriptor, target_block_id) in enumerate(
            zip(descriptors, target_block_ids)
        )
    )
    return RemoteG2BindingRecord(
        request_id="r",
        plan=plan,
        resolve_result=resolve_result,
        matched_tokens=len(target_block_ids) * plan.block_size_tokens,
        num_computed_tokens=0,
        block_size_tokens=plan.block_size_tokens,
        state=RemoteG2BindingState.BOUND,
        bound_blocks=bound_blocks,
    )


def test_make_target_descriptor_resolver_rejects_invalid_inputs():
    fake_kv = _FakeKvCacheManager({})
    with pytest.raises(ValueError):
        make_target_descriptor_resolver(
            fake_kv,
            primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
            block_size_bytes=0,
            window_size=16,
        )
    with pytest.raises(ValueError):
        make_target_descriptor_resolver(
            fake_kv,
            primary_pool_base_ptr=0,
            block_size_bytes=BLOCK_SIZE_BYTES,
            window_size=16,
        )
    with pytest.raises(ValueError):
        make_target_descriptor_resolver(
            fake_kv,
            primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
            block_size_bytes=BLOCK_SIZE_BYTES,
            window_size=0,
        )
    with pytest.raises(ValueError):
        make_target_descriptor_resolver(
            object(),
            primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
            block_size_bytes=BLOCK_SIZE_BYTES,
            window_size=16,
        )


def test_resolver_builds_target_descriptors_from_non_pinning_slot_lookup():
    fake_kv = _FakeKvCacheManager({42: 5, 43: 6})
    resolver = make_target_descriptor_resolver(
        fake_kv,
        primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
        window_size=16,
    )
    record = _make_record([42, 43])
    descriptors = list(resolver(record))

    assert len(descriptors) == 2
    assert descriptors[0].ptr == PRIMARY_POOL_BASE_PTR + 5 * BLOCK_SIZE_BYTES
    assert descriptors[0].size == BLOCK_SIZE_BYTES
    assert descriptors[0].memory_type == "VRAM"
    assert descriptors[1].ptr == PRIMARY_POOL_BASE_PTR + 6 * BLOCK_SIZE_BYTES
    assert descriptors[1].memory_type == "VRAM"
    assert fake_kv.slot_calls == [(42, 16), (43, 16)]


def test_resolver_refuses_failed_slot_lookup():
    fake_kv = _FakeKvCacheManager({42: 5})
    resolver = make_target_descriptor_resolver(
        fake_kv,
        primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
        window_size=16,
    )
    record = _make_record([42, 43])
    with pytest.raises(RemoteG2TransferError):
        list(resolver(record))
    assert fake_kv.slot_calls == [(42, 16), (43, 16)]


def test_resolver_rejects_invalid_target_block_id():
    fake_kv = _FakeKvCacheManager({})
    resolver = make_target_descriptor_resolver(
        fake_kv,
        primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
        window_size=16,
    )
    record = _make_record([-1])
    with pytest.raises(RemoteG2TransferError):
        list(resolver(record))
    assert fake_kv.slot_calls == []


def test_resolver_slot_lookup_per_block():
    fake_kv = _FakeKvCacheManager({1: 1, 2: 2, 3: 3})
    resolver = make_target_descriptor_resolver(
        fake_kv,
        primary_pool_base_ptr=PRIMARY_POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
        window_size=16,
    )
    record = _make_record([1, 2, 3])
    list(resolver(record))
    assert fake_kv.slot_calls == [(1, 16), (2, 16), (3, 16)]
