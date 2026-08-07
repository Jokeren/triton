import numpy as np
import pytest

import triton
import triton.language as tl
from triton.runtime.errors import InterpreterError

from triton._C.libtriton import interpreter as _interpreter


@triton.jit
def _device_patch_restore_helper(raise_error: tl.constexpr):
    if raise_error:
        raise RuntimeError("device helper failure")
    return 0


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
@pytest.mark.parametrize("raise_error", [False, True])
def test_device_function_restores_language_patches(raise_error) -> None:
    from triton.runtime.interpreter import _patch_lang

    missing = object()
    original_range = tl.range
    original_static_range = tl.static_range
    original_tensor_bool = getattr(tl.tensor, "__bool__", missing)
    outer_scope = _patch_lang(_device_patch_restore_helper.fn)
    outer_range = tl.range
    outer_static_range = tl.static_range
    outer_tensor_bool = getattr(tl.tensor, "__bool__", missing)
    try:
        if raise_error:
            with pytest.raises(InterpreterError, match="device helper failure"):
                _device_patch_restore_helper(raise_error)
        else:
            _device_patch_restore_helper(raise_error)

        assert tl.range is outer_range
        assert tl.static_range is outer_static_range
        assert getattr(tl.tensor, "__bool__", missing) is outer_tensor_bool
    finally:
        outer_scope.restore()
        assert tl.range is original_range
        assert tl.static_range is original_static_range
        assert getattr(tl.tensor, "__bool__", missing) is original_tensor_bool
