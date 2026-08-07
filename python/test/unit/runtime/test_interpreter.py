import numpy as np
import pytest
import triton.language as tl

from triton._C.libtriton import interpreter as _interpreter
from triton.runtime.interpreter import InterpreterBuilder, TensorHandle


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


@pytest.mark.parametrize(
    "input_bits, expected_bits",
    [(0x3F807FFF, 0x3F80),  # Below halfway.
     (0x3F808001, 0x3F81),  # Above halfway.
     (0x3F808000, 0x3F80),  # Halfway with an even retained significand.
     (0x3F818000, 0x3F82),  # Halfway with an odd retained significand.
     (0x3FFF8000, 0x4000),  # Significand carry into the exponent.
     (0xBF808000, 0xBF80),  # Negative halfway with an even retained significand.
     (0xBF818000, 0xBF82),  # Negative halfway with an odd retained significand.
     (0x7F800000, 0x7F80),  # Positive infinity.
     (0xFF800000, 0xFF80),  # Negative infinity.
     ],
)
def test_float32_to_bfloat16_cast_uses_rtne(input_bits, expected_bits) -> None:
    src = TensorHandle(np.array([input_bits], dtype=np.uint32).view(np.float32), tl.float32)

    result = InterpreterBuilder().cast_impl(src, tl.bfloat16)

    assert result.data.item() == expected_bits


@pytest.mark.parametrize("input_bits", [0x7F800001, 0x7FC01234, 0xFF800001])
def test_float32_to_bfloat16_cast_preserves_nan(input_bits) -> None:
    src = TensorHandle(np.array([input_bits], dtype=np.uint32).view(np.float32), tl.float32)

    result = InterpreterBuilder().cast_impl(src, tl.bfloat16)

    result_bits = result.data.item()
    assert result_bits & 0x7F80 == 0x7F80
    assert result_bits & 0x007F != 0
