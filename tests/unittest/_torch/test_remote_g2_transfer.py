# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_REMOTE_G2_PATH = (
    _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2.py"
)
_REMOTE_G2_TRANSFER_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_transfer.py"
)
_REMOTE_G2_RAW_NIXL_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_raw_nixl_adapter.py"
)
_REMOTE_G2_OBSERVABILITY_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_observability.py"
)


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


def _load_transfer_modules():
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)
    _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_observability",
        _REMOTE_G2_OBSERVABILITY_PATH,
    )
    remote_g2 = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2", _REMOTE_G2_PATH)
    transfer = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_transfer", _REMOTE_G2_TRANSFER_PATH
    )
    raw_nixl = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_raw_nixl_adapter",
        _REMOTE_G2_RAW_NIXL_PATH,
    )
    return remote_g2, transfer, raw_nixl


REMOTE_G2, TRANSFER, RAW_NIXL = _load_transfer_modules()


# bind_target_blocks does a lazy `from .remote_g2_connector import
# _installed_block_id_to_slot_idx`. Production installs the callable via
# maybe_start_remote_g2_target_client; these tests bypass that setup, so we
# preload a stub remote_g2_connector module in sys.modules with an identity
# callable so the lazy import resolves and binding can proceed.
_g2_connector_stub = types.ModuleType(
    f"{_CONNECTOR_PACKAGE}.remote_g2_connector"
)
_g2_connector_stub._installed_block_id_to_slot_idx = lambda ids: list(ids)
sys.modules[f"{_CONNECTOR_PACKAGE}.remote_g2_connector"] = _g2_connector_stub


@pytest.fixture(autouse=True)
def _identity_slot_lookup_for_lazy_import():
    """The lazy import in bind_target_blocks resolves to whatever module
    happens to be at sys.modules[...remote_g2_connector] when the test runs.
    Other test files (e.g. test_remote_g2_connector.py) load the real module
    and may leave `_installed_block_id_to_slot_idx` at None on teardown, so
    we re-stamp it before each test here and restore on exit."""
    module = sys.modules[f"{_CONNECTOR_PACKAGE}.remote_g2_connector"]
    saved = getattr(module, "_installed_block_id_to_slot_idx", None)
    module._installed_block_id_to_slot_idx = lambda ids: list(ids)
    yield
    module._installed_block_id_to_slot_idx = saved


RemoteG2Descriptor = REMOTE_G2.RemoteG2Descriptor
RemoteG2ResolveResult = REMOTE_G2.RemoteG2ResolveResult
RemoteKvReusePlan = REMOTE_G2.RemoteKvReusePlan
TargetRemoteG2BindingStore = REMOTE_G2.TargetRemoteG2BindingStore

RemoteG2NixlTransferAdapter = TRANSFER.RemoteG2NixlTransferAdapter
RemoteG2SourceMetadata = TRANSFER.RemoteG2SourceMetadata
RemoteG2SourceMetadataCache = TRANSFER.RemoteG2SourceMetadataCache
RemoteG2TransferDescriptor = TRANSFER.RemoteG2TransferDescriptor
RemoteG2TransferError = TRANSFER.RemoteG2TransferError
RawNixlRemoteG2Adapter = RAW_NIXL.RawNixlRemoteG2Adapter


class FakeMemoryDescs:
    def __init__(self, type, descs):
        self.type = type
        self.descs = descs


class FakeRegMemoryDescs(FakeMemoryDescs):
    pass


class FakeTransferOp:
    READ = "READ"


@dataclass
class FakeTransferRequest:
    op: str
    src_descs: FakeMemoryDescs
    dst_descs: FakeMemoryDescs
    remote_name: str
    sync_message: str | None = None


class FakeStatus:
    def __init__(self, completed=True):
        self.completed = completed

    def is_completed(self):
        return self.completed

    def wait(self, timeout_ms=None):
        return self.completed


class FakeAgent:
    def __init__(self):
        self.loaded = []
        self.registered = []
        self.deregistered = []
        self.requests = []

    def load_remote_agent(self, name, agent_desc):
        self.loaded.append((name, agent_desc))

    def register_memory(self, descs):
        self.registered.append(descs)

    def deregister_memory(self, descs):
        self.deregistered.append(descs)

    def submit_transfer_requests(self, request):
        self.requests.append(request)
        return FakeStatus()


FAKE_TRANSFER_TYPES = SimpleNamespace(
    MemoryDescs=FakeMemoryDescs,
    RegMemoryDescs=FakeRegMemoryDescs,
    TransferOp=FakeTransferOp,
    TransferRequest=FakeTransferRequest,
)


def _plan(**overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "dynamo-request-1",
        "target_worker_id": 42,
        "target_dp_rank": 2,
        "source_worker_id": 7,
        "source_dp_rank": 0,
        "source_tier": "host_pinned",
        "block_hashes": [11, 22, 33],
        "start_block_index": 0,
        "planned_prefix_blocks": 3,
        "block_size_tokens": 16,
        "created_at_ms": 100,
        "expires_at_ms": 10_000,
    }
    plan.update(overrides)
    return plan


def _source_descriptor(block_hash):
    return RemoteG2Descriptor(
        block_hash=block_hash,
        descriptor_generation=1,
        pool_id=f"source-{block_hash}",
        byte_offset=0,
        byte_length=4096,
        metadata={
            "nixl_memory_desc": {
                "ptr": block_hash * 8192,
                "size": 4096,
                "device_id": 0,
                "memory_type": "DRAM",
                "name": f"source-{block_hash}",
            }
        },
    )


def _binding_record(num_computed_tokens=16, resolve_result=None):
    store = TargetRemoteG2BindingStore(release_lease=lambda lease_id, reason: True)
    result = resolve_result or RemoteG2ResolveResult(
        lease_id="lease-1",
        descriptors=tuple(
            _source_descriptor(block_hash) for block_hash in (11, 22, 33)
        ),
        num_tokens=48,
        source_generation=99,
    )
    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        num_computed_tokens,
        lambda plan: result,
    )
    store.bind_target_blocks(1234, [100, 101, 102])
    return record


def _raw_adapter_for_fail_closed_tests():
    adapter = object.__new__(RawNixlRemoteG2Adapter)
    adapter._local_rank = 1
    adapter._source_metadata_fetcher = lambda worker_id, generation: {
        "remote_name": "rank0",
        "agent_metadata": b"rank0",
        "pool_base_ptr": 0,
        "pool_size_bytes": 4096,
        "source_generation": generation,
    }
    adapter._ensure_peer_loaded = lambda source_meta: ("local", "remote")
    adapter._agent = SimpleNamespace(
        make_prepped_xfer=lambda *args, **kwargs: "handle",
        transfer=lambda handle: "DONE",
    )
    adapter._block_size_bytes = 0
    return adapter


def _tp_resolve_result(*, per_rank_meta, per_rank_descs):
    return RemoteG2ResolveResult(
        lease_id="lease-1",
        descriptors=tuple(_source_descriptor(block_hash) for block_hash in (11, 22, 33)),
        num_tokens=48,
        source_generation=99,
        per_rank_source_metadata=per_rank_meta,
        per_rank_descriptors=per_rank_descs,
    )


def _source_metadata(worker_id=7, generation=99):
    return RemoteG2SourceMetadata(
        source_worker_id=worker_id,
        source_generation=generation,
        remote_name=f"source-{worker_id}",
        agent_desc=b"agent-desc",
    )


def test_remote_g2_source_metadata_cache_refreshes_by_generation():
    cache = RemoteG2SourceMetadataCache()
    calls = []

    def fetch(worker_id, generation):
        calls.append((worker_id, generation))
        return _source_metadata(worker_id, generation)

    assert cache.get_or_refresh(7, 99, fetch).source_generation == 99
    assert cache.get_or_refresh(7, 99, fetch).source_generation == 99
    assert cache.get_or_refresh(7, 100, fetch).source_generation == 100

    assert calls == [(7, 99), (7, 100)]


def test_raw_nixl_adapter_rejects_missing_tp_rank_metadata():
    record = _binding_record(
        resolve_result=_tp_resolve_result(
            per_rank_meta={
                0: {
                    "remote_name": "rank0",
                    "agent_metadata": b"rank0",
                    "pool_base_ptr": 0,
                    "pool_size_bytes": 4096,
                    "source_generation": 99,
                }
            },
            per_rank_descs={1: [{"byte_offset": 0}, {"byte_offset": 4096}]},
        )
    )
    adapter = _raw_adapter_for_fail_closed_tests()

    with pytest.raises(RuntimeError, match="missing per-rank source metadata"):
        adapter._start_transfer_impl(record, _nvtx=None)


def test_raw_nixl_adapter_rejects_missing_tp_rank_descriptors():
    record = _binding_record(
        resolve_result=_tp_resolve_result(
            per_rank_meta={
                1: {
                    "remote_name": "rank1",
                    "agent_metadata": b"rank1",
                    "pool_base_ptr": 0,
                    "pool_size_bytes": 4096,
                    "source_generation": 99,
                }
            },
            per_rank_descs={0: [{"byte_offset": 0}, {"byte_offset": 4096}]},
        )
    )
    adapter = _raw_adapter_for_fail_closed_tests()

    with pytest.raises(RuntimeError, match="missing per-rank descriptors"):
        adapter._start_transfer_impl(record, _nvtx=None)


def test_raw_nixl_adapter_rejects_incomplete_tp_rank_descriptors():
    record = _binding_record(
        resolve_result=_tp_resolve_result(
            per_rank_meta={
                1: {
                    "remote_name": "rank1",
                    "agent_metadata": b"rank1",
                    "pool_base_ptr": 0,
                    "pool_size_bytes": 4096,
                    "source_generation": 99,
                }
            },
            per_rank_descs={1: [{"byte_offset": 0}]},
        )
    )
    adapter = _raw_adapter_for_fail_closed_tests()

    with pytest.raises(RuntimeError, match="incomplete per-rank descriptors"):
        adapter._start_transfer_impl(record, _nvtx=None)


def test_remote_g2_transfer_adapter_submits_one_read_for_bound_prefix():
    agent = FakeAgent()
    record = _binding_record()
    adapter = RemoteG2NixlTransferAdapter(
        source_metadata_fetcher=lambda worker_id, generation: _source_metadata(
            worker_id, generation
        ),
        target_descriptor_resolver=lambda record: [
            RemoteG2TransferDescriptor(
                ptr=target_block.target_block_id * 16384,
                size=4096,
                device_id=0,
                memory_type="VRAM",
                name=f"target-{target_block.target_block_id}",
            )
            for target_block in record.bound_blocks
        ],
        agent_factory=lambda: agent,
        transfer_types=FAKE_TRANSFER_TYPES,
    )

    result = adapter.start_transfer(record)

    assert result.is_completed() is True
    assert agent.loaded == [("source-7", b"agent-desc")]
    assert len(agent.registered) == 1
    assert agent.registered[0].type == "VRAM"
    assert len(agent.requests) == 1
    request = agent.requests[0]
    assert request.op == "READ"
    assert request.src_descs.type == "DRAM"
    assert request.dst_descs.type == "VRAM"
    assert request.remote_name == "source-7"
    assert len(request.src_descs.descs) == len(record.bound_blocks)
    result.release()
    result.release()
    assert result.released is True
    assert agent.deregistered == [agent.registered[0]]


def test_remote_g2_transfer_adapter_rejects_descriptor_count_mismatch():
    agent = FakeAgent()
    adapter = RemoteG2NixlTransferAdapter(
        source_metadata_fetcher=lambda worker_id, generation: _source_metadata(
            worker_id, generation
        ),
        target_descriptor_resolver=lambda record: [
            RemoteG2TransferDescriptor(
                ptr=1000, size=4096, device_id=0, memory_type="VRAM"
            )
        ],
        agent_factory=lambda: agent,
        transfer_types=FAKE_TRANSFER_TYPES,
    )

    with pytest.raises(RemoteG2TransferError, match="descriptor counts differ"):
        adapter.start_transfer(_binding_record())

    assert agent.requests == []


def test_remote_g2_transfer_adapter_rejects_memory_type_mismatch():
    agent = FakeAgent()
    adapter = RemoteG2NixlTransferAdapter(
        source_metadata_fetcher=lambda worker_id, generation: _source_metadata(
            worker_id, generation
        ),
        target_descriptor_resolver=lambda record: [
            RemoteG2TransferDescriptor(
                ptr=target_block.target_block_id * 16384,
                size=4096,
                device_id=0,
                memory_type="DRAM",
            )
            for target_block in record.bound_blocks
        ],
        agent_factory=lambda: agent,
        transfer_types=FAKE_TRANSFER_TYPES,
    )

    with pytest.raises(RemoteG2TransferError, match="target descriptor must be VRAM"):
        adapter.start_transfer(_binding_record())

    assert agent.requests == []
