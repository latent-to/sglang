"""PD decode re-transfers the SWA tail, so tree SWA must not gate its prefix match."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv_cache_builder import swa_gates_tree_prefix_match
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _pool(cls, **attrs):
    pool = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(pool, name, value)
    return pool


def _router(*sub_pools):
    """A pool's layer router, reduced to the sub-pool lookup the gate reads."""
    return SimpleNamespace(
        sub_pool=lambda name: object() if name in sub_pools else None
    )


def _swa_pool():
    return _pool(
        SWAKVPool,
        layer_router=_router(IdSpaceKind.FULL, IdSpaceKind.SLIDING_WINDOW),
    )


def _dsv4_pool():
    # The DeepSeek-V4 pool routes through the plain one-sub-pool router, so the
    # gate takes it by type.
    return _pool(
        DeepSeekV4TokenToKVPool,
        request_window=None,
        layer_router=_router(IdSpaceKind.FULL),
    )


def _allocator(page_size, tail_prealloc=True):
    allocator = SimpleNamespace(page_size=page_size)
    if tail_prealloc:
        allocator.alloc_extend_swa_tail = lambda *args: None
    return allocator


class TestSwaGatesTreePrefixMatch(CustomTestCase):
    def _gates(self, mode, pool, allocator):
        with get_context().override_server_args(disaggregation_mode=mode):
            return swa_gates_tree_prefix_match(pool, allocator)

    def test_decode_with_swa_tail_prealloc_does_not_gate(self):
        for pool in (_swa_pool(), _dsv4_pool()):
            with self.subTest(pool=type(pool).__name__):
                self.assertFalse(self._gates("decode", pool, _allocator(256)))

    def test_prefill_and_colocated_still_gate(self):
        for mode in ("prefill", "null"):
            with self.subTest(mode=mode):
                self.assertTrue(self._gates(mode, _swa_pool(), _allocator(256)))

    def test_decode_without_swa_tail_prealloc_still_gates(self):
        self.assertTrue(self._gates("decode", _swa_pool(), _allocator(1)))
        self.assertTrue(
            self._gates("decode", _swa_pool(), _allocator(64, tail_prealloc=False))
        )


if __name__ == "__main__":
    unittest.main()
