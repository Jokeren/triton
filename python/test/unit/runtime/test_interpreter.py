import numpy as np
import pytest
import torch

from triton._C.libtriton import interpreter as _interpreter
from triton._C.libtriton import ir as _ir
import triton.language as tl
from triton.runtime.interpreter import (TensorHandle, _convert_float, _e8m0_to_f32, _mxfp_value_handle_to_float32)
from triton.tools.mxfp import MXScaleTensor


def _element_ptrs(array: np.ndarray) -> np.ndarray:
    base = np.uint64(array.ctypes.data)
    offsets = np.arange(array.size, dtype=np.uint64) * np.uint64(array.itemsize)
    return (base + offsets).reshape(array.shape)


def _assert_float32_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    not_nan = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[not_nan].view(np.uint32), expected[not_nan].view(np.uint32))


def _decode_e4m3b15_reference(bits: np.ndarray) -> np.ndarray:
    magnitude = bits & np.uint8(0x7F)
    exponent = magnitude >> np.uint8(3)
    mantissa = magnitude & np.uint8(0x7)
    # The CUDA helper maps the otherwise-special 0x7f/0xff encodings to
    # +/-1.75, making them aliases for the canonical 0x7e/0xfe encodings.
    mantissa = np.where(magnitude == np.uint8(0x7F), np.uint8(6), mantissa)
    subnormal = np.ldexp(mantissa.astype(np.float32), 1 - 15 - 3)
    normal = np.ldexp((mantissa + np.uint8(8)).astype(np.float32), exponent.astype(np.int32) - 15 - 3)
    value = np.where(exponent == 0, subnormal, normal)
    return np.where((bits & np.uint8(0x80)) != 0, -value, value)


@pytest.mark.parametrize(
    "triton_dtype, torch_dtype, reference_scale",
    [
        pytest.param(tl.float8e5, torch.float8_e5m2, 1.0, id="e5m2"),
        pytest.param(tl.float8e4nv, torch.float8_e4m3fn, 1.0, id="e4m3fn"),
        pytest.param(tl.float8e4b8, torch.float8_e4m3fnuz, 1.0, id="e4m3fnuz"),
        pytest.param(tl.float8e5b16, torch.float8_e5m2fnuz, 1.0, id="e5m2fnuz"),
    ],
)
def test_float8_decode_all_bit_patterns(triton_dtype, torch_dtype, reference_scale) -> None:
    bits = np.arange(256, dtype=np.uint8)
    expected = torch.arange(256, dtype=torch.uint8).view(torch_dtype).to(torch.float32).numpy()
    expected[np.isfinite(expected)] *= reference_scale

    converted = _convert_float(bits, triton_dtype, tl.float32, _ir.ROUNDING_MODE.RTNE).view(np.float32)
    _assert_float32_bits_equal(converted, expected)

    mxfp_converted = _mxfp_value_handle_to_float32(TensorHandle(bits, triton_dtype))
    _assert_float32_bits_equal(mxfp_converted, expected)


def test_float8e4b15_decode_all_bit_patterns() -> None:
    bits = np.arange(256, dtype=np.uint8)
    expected = _decode_e4m3b15_reference(bits)

    converted = _convert_float(bits, tl.float8e4b15, tl.float32, _ir.ROUNDING_MODE.RTNE).view(np.float32)
    _assert_float32_bits_equal(converted, expected)

    mxfp_converted = _mxfp_value_handle_to_float32(TensorHandle(bits, tl.float8e4b15))
    _assert_float32_bits_equal(mxfp_converted, expected)

    assert converted[0x7F] == np.float32(1.75)
    assert converted[0xFF] == np.float32(-1.75)
    assert converted[0x80].view(np.uint32) == np.uint32(0x80000000)


def test_float8e4b15_rtne_all_finite_values_and_ties() -> None:
    canonical_bits = np.arange(0x7F, dtype=np.uint8)
    positive_values = _decode_e4m3b15_reference(canonical_bits)

    converted = _convert_float(positive_values, tl.float32, tl.float8e4b15, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, canonical_bits)
    converted = _convert_float(-positive_values, tl.float32, tl.float8e4b15, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, canonical_bits | np.uint8(0x80))

    # Every midpoint is exact in float32.  RTNE selects the neighboring code
    # whose retained low bit is even, including at subnormal/exponent edges.
    midpoints = (positive_values[:-1] + positive_values[1:]) / np.float32(2)
    expected = np.where((canonical_bits[:-1] & np.uint8(1)) == 0, canonical_bits[:-1], canonical_bits[1:])
    converted = _convert_float(midpoints, tl.float32, tl.float8e4b15, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, expected)
    converted = _convert_float(-midpoints, tl.float32, tl.float8e4b15, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, expected | np.uint8(0x80))


def test_bfloat16_subnormal_decode() -> None:
    bits = np.array([0x0000, 0x0001, 0x0002, 0x007F, 0x8000, 0x8001, 0x807F], dtype=np.uint16)
    expected = torch.from_numpy(bits.copy()).view(torch.bfloat16).to(torch.float32).numpy()
    converted = _convert_float(bits, tl.bfloat16, tl.float32, _ir.ROUNDING_MODE.RTNE).view(np.float32)
    _assert_float32_bits_equal(converted, expected)


def test_float32_to_bfloat16_rtne_ties_and_carry() -> None:
    bits = np.array(
        [
            0x3F807FFF,  # Below an even tie.
            0x3F808000,  # Even tie rounds down.
            0x3F808001,  # Above the tie rounds up.
            0x3F817FFF,  # Below an odd tie.
            0x3F818000,  # Odd tie rounds up to even.
            0x3F818001,  # Above the tie rounds up.
            0x3FFFFFFF,  # Significand carry increments the exponent.
            0x7F7FFFFF,  # Carry overflows to positive infinity.
            0xFF7FFFFF,  # Carry overflows to negative infinity.
        ],
        dtype=np.uint32,
    )
    values = bits.view(np.float32)
    expected = torch.from_numpy(values.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    converted = _convert_float(values, tl.float32, tl.bfloat16, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, expected)


@pytest.mark.parametrize(
    "triton_dtype, torch_dtype, reference_scale",
    [
        pytest.param(tl.float8e5, torch.float8_e5m2, 1.0, id="e5m2"),
        pytest.param(tl.float8e4nv, torch.float8_e4m3fn, 1.0, id="e4m3fn"),
        pytest.param(tl.float8e4b8, torch.float8_e4m3fnuz, 1.0, id="e4m3fnuz"),
        pytest.param(tl.float8e5b16, torch.float8_e5m2fnuz, 1.0, id="e5m2fnuz"),
    ],
)
def test_float8_rtne_special_values(triton_dtype, torch_dtype, reference_scale) -> None:
    values = np.array(
        [
            0.0,
            -0.0,
            2.0**-30,
            -(2.0**-30),
            1.0,
            1.0625,
            1.1875,
            1.9999999,
            240.0,
            448.0,
            464.0,
            465.0,
            57344.0,
            61440.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    with np.errstate(invalid="ignore", over="ignore"):
        reference_values = values * np.float32(reference_scale)
    expected = torch.from_numpy(reference_values.copy()).to(torch_dtype).view(torch.uint8).numpy()
    converted = _convert_float(values, tl.float32, triton_dtype, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, expected)


def test_float8e4b15_rtne_special_values() -> None:
    values = np.array(
        [
            0.0,
            -0.0,
            2.0**-18,
            -(2.0**-18),
            2.0**-17,
            1.0,
            1.0625,
            1.1875,
            1.75,
            1.8125,
            1.8125001,
            -1.8125001,
            1.875,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    # FP8E4M3B15 is a signed-zero, all-finite format.  Its CUDA encoder
    # saturates to the canonical +/-1.75 encodings rather than emitting the
    # 0x7f/0xff aliases; unlike FNUZ, 0x80 is negative zero.
    expected = np.array(
        [
            0x00,
            0x80,
            0x00,
            0x80,
            0x01,
            0x78,
            0x78,
            0x7A,
            0x7E,
            0x7E,
            0x7E,
            0xFE,
            0x7E,
            0x7E,
            0xFE,
            0x7E,
        ],
        dtype=np.uint8,
    )
    converted = _convert_float(values, tl.float32, tl.float8e4b15, _ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(converted, expected)
    assert not np.isin(converted, np.array([0x7F, 0xFF], dtype=np.uint8)).any()


def test_e8m0_decode_all_bit_patterns() -> None:
    bits = np.arange(256, dtype=np.uint8)
    reference = MXScaleTensor(size=(256, ), device="cpu")
    reference.data = torch.arange(256, dtype=torch.uint8)
    expected = reference.to(torch.float32).numpy()
    _assert_float32_bits_equal(_e8m0_to_f32(bits), expected)


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
