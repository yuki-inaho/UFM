"""CPU tests for the deterministic bilinear resize helper (Step 19 attempt-9).

The helper under test is ``deterministic_bilinear_interpolate`` in
``uniflowmatch/training/rgbd_deterministic_ops.py``.  These tests use an
**independent** reference: ``torch.nn.functional.interpolate(..., mode=
"bilinear")``.  They never assert on the private implementation strategy
(index_select / separable sums); they assert on the *contract* -- output shape,
sampled values, gradients, and the fail-fast rejections -- so the helper can be
hooked into a forward pass by a later unit with confidence.

Only CPU tensors are used; no GPU, no model training, no dataset access.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO / "uniflowmatch" / "training" / "rgbd_deterministic_ops.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rgbd_deterministic_ops", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = _load_module()
bilinear = ops.deterministic_bilinear_interpolate
E = ops.DeterministicInterpolateError


# --------------------------------------------------------------------------- #
# value agreement with the F.interpolate reference
# --------------------------------------------------------------------------- #
_CASES = [
    ((2, 3, 5, 7), None, 2.0),
    ((2, 3, 5, 7), None, (2.0, 3.0)),
    ((2, 3, 5, 7), (9, 4), None),
    ((1, 2, 4, 6), (4, 6), None),
    ((1, 1, 3, 3), None, (1.0, 2.0)),
    ((2, 2, 8, 5), (3, 3), None),
    ((1, 1, 1, 5), (1, 9), None),
    ((1, 1, 6, 6), (3, 10), None),
    ((3, 2, 7, 3), None, 2.0),
    ((1, 4, 4, 4), (2, 2), None),
]


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,size,scale_factor", _CASES)
def test_values_match_reference(shape, size, scale_factor, dtype, align_corners):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)
    ref = F.interpolate(
        x, size=size, scale_factor=scale_factor,
        mode="bilinear", align_corners=align_corners,
    )
    got = bilinear(x, size=size, scale_factor=scale_factor, align_corners=align_corners)
    assert got.shape == ref.shape
    assert got.dtype == x.dtype
    assert got.device == x.device
    atol = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.allclose(got, ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_axis_resize(dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 5, 7, dtype=dtype)
    for size in [(9, 7), (5, 9), (5, 7)]:
        ref = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        got = bilinear(x, size=size, align_corners=False)
        atol = 1e-5 if dtype == torch.float32 else 1e-12
        assert torch.allclose(got, ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scale_factor_two(dtype):
    torch.manual_seed(2)
    for shape in [(1, 1, 3, 4), (2, 3, 5, 6), (1, 2, 2, 9)]:
        x = torch.randn(*shape, dtype=dtype)
        for ac in (False, True):
            ref = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=ac)
            got = bilinear(x, scale_factor=2, align_corners=ac)
            assert got.shape == (shape[0], shape[1], shape[2] * 2, shape[3] * 2)
            atol = 1e-5 if dtype == torch.float32 else 1e-12
            assert torch.allclose(got, ref, atol=atol, rtol=0.0)


def test_edge_clamping_identity_when_same_size():
    x = torch.randn(2, 3, 5, 7)
    got = bilinear(x, size=(5, 7), align_corners=False)
    assert torch.equal(got, x)


# --------------------------------------------------------------------------- #
# gradient agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_match_reference(dtype, align_corners):
    torch.manual_seed(3)
    x = torch.randn(2, 3, 5, 6, dtype=dtype, requires_grad=True)
    size = (9, 4)

    ref = F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)
    ref.sum().backward()
    grad_ref = x.grad.clone()
    x.grad = None

    got = bilinear(x, size=size, align_corners=align_corners)
    got.sum().backward()
    grad_got = x.grad.clone()

    atol = 1e-4 if dtype == torch.float32 else 1e-12
    assert torch.allclose(grad_got, grad_ref, atol=atol, rtol=0.0)


def test_gradcheck_float64():
    torch.manual_seed(4)

    def fn(x):
        return bilinear(x, size=(7, 5), align_corners=False)

    x = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-6)


def test_gradcheck_float64_align_corners():
    torch.manual_seed(5)

    def fn(x):
        return bilinear(x, size=(3, 8), align_corners=True)

    x = torch.randn(1, 1, 5, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# hand-computed pixel coordinates (align_corners=False, half-pixel rule)
# --------------------------------------------------------------------------- #
def test_hand_computed_half_pixel_1d_row():
    # 1x1x1x4 -> width 8, align_corners=False.
    # src = (dst + 0.5) * (4/8) - 0.5 = (dst + 0.5)/2 - 0.5
    # dst=0 -> -0.25 -> clamp 0.0 ; dst=1 -> 0.25 ; dst=3 -> 1.25 ; dst=7 -> 3.25
    x = torch.tensor([[[[0.0, 2.0, 4.0, 6.0]]]], dtype=torch.float64)
    got = bilinear(x, size=(1, 8), align_corners=False)
    expected = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.0]
    assert torch.allclose(got[0, 0, 0], torch.tensor(expected, dtype=torch.float64))


def test_hand_computed_align_corners_true():
    # align_corners=True, 1x1x1x3 -> width 5: scale = (3-1)/(5-1) = 0.5
    # src = dst*0.5 ; values 0,1,2 -> expected 0,0.5,1,1.5,2
    x = torch.tensor([[[[0.0, 1.0, 2.0]]]], dtype=torch.float64)
    got = bilinear(x, size=(1, 5), align_corners=True)
    expected = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert torch.allclose(got[0, 0, 0], torch.tensor(expected, dtype=torch.float64))


def test_hand_computed_2d_interior_pixel():
    # 2x2 -> 3x3, align_corners=False; center-ish pixel at (1,1):
    # src_h = (1.5)*(2/3) - 0.5 = 0.5 ; src_w = 0.5 -> average of 4 corners.
    x = torch.tensor([[[[0.0, 4.0], [8.0, 12.0]]]], dtype=torch.float64)
    got = bilinear(x, size=(3, 3), align_corners=False)
    assert math.isclose(got[0, 0, 1, 1].item(), 6.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# invalid-argument rejection
# --------------------------------------------------------------------------- #
def test_reject_antialias():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, size=(8, 8), antialias=True)


def test_reject_both_size_and_scale():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, size=(8, 8), scale_factor=2)


def test_reject_neither_size_nor_scale():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x)


def test_reject_non_4d():
    x = torch.randn(1, 1, 4)
    with pytest.raises(E):
        bilinear(x, size=(8,))


def test_reject_non_floating():
    x = torch.arange(16).reshape(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, size=(8, 8))


def test_reject_float64_grad_unsupported_dtype():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float16)
    with pytest.raises(E):
        bilinear(x, size=(8, 8))


def test_reject_bad_size_length():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, size=(8, 8, 8))


def test_reject_non_positive_size():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, size=(0, 8))


def test_reject_non_positive_scale():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, scale_factor=0)


def test_reject_bad_align_corners_type():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, size=(8, 8), align_corners="yes")


def test_reject_bad_recompute_scale_factor():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(E):
        bilinear(x, scale_factor=2, recompute_scale_factor="nope")


def test_recompute_scale_factor_true_matches_reference():
    torch.manual_seed(6)
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    ref = F.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=False,
        recompute_scale_factor=True,
    )
    got = bilinear(x, scale_factor=2, align_corners=False, recompute_scale_factor=True)
    assert torch.allclose(got, ref, atol=1e-12, rtol=0.0)
