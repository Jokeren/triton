import types

import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton.language.core import item as direct_item
from triton.language import static_range as direct_static_range
from triton.language import store as direct_store

from triton._C.libtriton import interpreter as _interpreter

_cross_globals_helper = None


def _cross_globals_helper_impl(output):
    direct_store(output, direct_item(tl.arange(0, 1)) + 7)


@triton.jit
def _call_cross_globals_helper(output):
    _cross_globals_helper(output)


def _element_ptrs(array: np.ndarray) -> np.ndarray:
    base = np.uint64(array.ctypes.data)
    offsets = np.arange(array.size, dtype=np.uint64) * np.uint64(array.itemsize)
    return (base + offsets).reshape(array.shape)


def test_load_accepts_non_contiguous_ndarray_views() -> None:
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    ptrs = _element_ptrs(data)[:, ::2]
    mask = np.array([[True, False, True, False], [False, True, False, True], [True, True, False, False]])[:, ::2]
    other = (np.arange(12, dtype=np.int32).reshape(3, 4) + 100)[:, ::2]

    loaded = _interpreter.load(ptrs, mask, other, np.int32)

    np.testing.assert_array_equal(loaded, np.where(mask, data[:, ::2], other))


def test_store_accepts_non_contiguous_ndarray_views() -> None:
    dst = np.zeros((3, 4), dtype=np.int32)
    ptrs = _element_ptrs(dst)[:, 1::2]
    values = (np.arange(12, dtype=np.int32).reshape(3, 4) + 10)[:, 1::2]
    mask = np.array([[True, False, False, True], [False, True, True, False], [True, False, True, False]])[:, 1::2]

    _interpreter.store(ptrs, values, mask)

    expected = np.zeros((3, 4), dtype=np.int32)
    expected[:, 1::2] = np.where(mask, values, expected[:, 1::2])
    np.testing.assert_array_equal(dst, expected)


def test_atomic_rmw_accepts_non_contiguous_ndarray_views() -> None:
    dst = np.arange(12, dtype=np.int32).reshape(3, 4)
    ptrs = _element_ptrs(dst)[:, ::2]
    values = (np.arange(12, dtype=np.int32).reshape(3, 4) + 1)[:, ::2]
    mask = np.ones((3, 4), dtype=bool)[:, ::2]

    old = _interpreter.atomic_rmw(_interpreter.RMW_OP.ADD, ptrs, values, mask, _interpreter.MEM_SEMANTIC.RELAXED)

    original = np.arange(12, dtype=np.int32).reshape(3, 4)
    np.testing.assert_array_equal(old, original[:, ::2])
    original[:, ::2] += values
    np.testing.assert_array_equal(dst, original)


def test_atomic_cas_accepts_non_contiguous_ndarray_views() -> None:
    dst = np.arange(12, dtype=np.int32).reshape(3, 4)
    ptrs = _element_ptrs(dst)[:, ::2]
    expected = dst.copy()[:, ::2]
    desired = (np.arange(12, dtype=np.int32).reshape(3, 4) + 200)[:, ::2]

    old = _interpreter.atomic_cas(ptrs, expected, desired, _interpreter.MEM_SEMANTIC.RELAXED)

    original = np.arange(12, dtype=np.int32).reshape(3, 4)
    np.testing.assert_array_equal(old, original[:, ::2])
    original[:, ::2] = desired
    np.testing.assert_array_equal(dst, original)


@pytest.mark.interpreter
def test_directly_imported_language_builtin(device) -> None:

    @triton.jit
    def kernel(output):
        value = 0
        for i in direct_static_range(4):
            value += i
        direct_store(output, value)

    original_store = direct_store
    original_static_range = direct_static_range
    output = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](output)

    assert output.item() == 6
    assert direct_store is original_store
    assert direct_static_range is original_static_range


@pytest.mark.interpreter
def test_nested_direct_builtin_from_other_globals(device) -> None:
    global _cross_globals_helper

    helper_globals = {
        "__builtins__": __builtins__,
        "direct_item": direct_item,
        "direct_store": direct_store,
        "tl": tl,
    }
    helper_fn = types.FunctionType(_cross_globals_helper_impl.__code__, helper_globals,
                                   _cross_globals_helper_impl.__name__)
    _cross_globals_helper = triton.jit(helper_fn)
    original_item = helper_globals["direct_item"]
    original_store = helper_globals["direct_store"]

    try:
        output = torch.empty(1, dtype=torch.int32, device=device)
        _call_cross_globals_helper[(1, )](output)

        assert output.item() == 7
        assert helper_globals["direct_item"] is original_item
        assert helper_globals["direct_store"] is original_store
    finally:
        helper_globals["direct_item"] = original_item
        helper_globals["direct_store"] = original_store
        _cross_globals_helper = None
