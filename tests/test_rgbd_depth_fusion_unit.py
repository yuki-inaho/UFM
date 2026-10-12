"""Synthetic unit tests for the ADR-14 Option C ``DepthFusion`` module.

These exercise the *new* depth-fusion primitive directly (shape coercion,
zero-gate identity, mask semantics, fail-fast validation) without needing any
network.  They are public synthetic tests: no private data, calibration, or
absolute host paths are embedded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uniflowmatch.models.depth_fusion import (  # noqa: E402
    DepthFusion,
    DepthFusionDeviceError,
    DepthFusionDtypeError,
    DepthFusionMaskError,
    DepthFusionNonFiniteError,
    DepthFusionShapeError,
    prepare_metric_depth,
)

C = 4
H = W = 8
FH = FW = 3


def _feat(b=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, C, FH, FW)


def _masked_depth(value=0.5, mask=True, b=2, seed=1):
    torch.manual_seed(seed)
    d = torch.full((b, H, W), float(value))
    m = torch.full((b, H, W), bool(mask), dtype=torch.bool)
    return d, m


# --------------------------------------------------------------------------- #
# shape coercion: HW / 1HW / BHW / B1HW
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [(H, W), (1, H, W), (2, H, W), (1, 1, H, W), (2, 1, H, W)],
)
def test_prepare_metric_depth_accepts_documented_shapes(shape):
    d, m = prepare_metric_depth(torch.rand(*shape), ref=_feat())
    assert d.shape == (2, 1, H, W)
    assert m.shape == (2, 1, H, W)
    assert m.dtype == torch.bool


def test_prepare_metric_depth_default_mask_is_all_valid():
    d, m = prepare_metric_depth(torch.rand(1, H, W), ref=_feat())
    assert bool(m.all())


def test_prepare_metric_depth_rejects_non_tensor():
    with pytest.raises(DepthFusionDtypeError):
        prepare_metric_depth([0.1, 0.2], ref=_feat())


def test_prepare_metric_depth_rejects_non_floating():
    with pytest.raises(DepthFusionDtypeError):
        prepare_metric_depth(torch.ones(2, H, W, dtype=torch.int32), ref=_feat())


def test_prepare_metric_depth_rejects_bad_rank():
    with pytest.raises(DepthFusionShapeError):
        prepare_metric_depth(torch.rand(2, 1, H, W, 1), ref=_feat())


def test_prepare_metric_depth_rejects_bad_channel():
    with pytest.raises(DepthFusionShapeError):
        prepare_metric_depth(torch.rand(2, 3, H, W), ref=_feat())


def test_prepare_metric_depth_rejects_nonfinite():
    d = torch.full((2, H, W), float("nan"))
    with pytest.raises(DepthFusionNonFiniteError):
        prepare_metric_depth(d, ref=_feat())


def test_prepare_metric_depth_rejects_device_mismatch():
    d = torch.ones(2, H, W, device="meta")
    with pytest.raises(DepthFusionDeviceError):
        prepare_metric_depth(d, ref=_feat())


def test_prepare_metric_depth_rejects_mask_shape_mismatch():
    d = torch.ones(2, H, W)
    m = torch.ones(2, H + 1, W, dtype=torch.bool)
    with pytest.raises(DepthFusionShapeError):
        prepare_metric_depth(d, m, ref=_feat())


# --------------------------------------------------------------------------- #
# zero-gate identity / mask semantics
# --------------------------------------------------------------------------- #
def test_zero_gate_is_bit_exact_rgb_identity():
    df = DepthFusion(C, C, gate_init=0.0, depth_scale=1.0)
    feat = _feat()
    d, m = _masked_depth(0.7, True)
    out = df(feat, d, m, image_hw=(H, W))
    assert torch.equal(out, feat)


def test_all_invalid_is_identity_even_with_nonzero_gate_and_biased_adapter():
    df = DepthFusion(C, C, gate_init=0.7, depth_scale=1.0)
    with torch.no_grad():
        for p in df.parameters():
            p.add_(0.5)  # make the adapter strongly biased
    feat = _feat()
    d, m = _masked_depth(0.9, False)  # all invalid
    out = df(feat, d, m, image_hw=(H, W))
    assert torch.equal(out, feat)


def test_valid_depth_changes_output_when_gate_nonzero():
    df = DepthFusion(C, C, gate_init=0.0, depth_scale=1.0)
    with torch.no_grad():
        df.gate.fill_(0.5)
    feat = _feat()
    d, m = _masked_depth(0.9, True)
    out = df(feat, d, m, image_hw=(H, W))
    assert not torch.equal(out, feat)


def test_invalid_pixels_contribute_nothing_to_their_own_output_pixel():
    df = DepthFusion(C, C, gate_init=0.5, depth_scale=1.0)
    feat = _feat()
    d = torch.rand(2, H, W)
    m = torch.ones(2, H, W, dtype=torch.bool)
    m[:, : H // 2, :] = False  # left half invalid
    out = df(feat, d, m, image_hw=(H, W))
    # nearest H->FH maps feature rows {0,1} to invalid source rows, row 2 to valid.
    assert torch.equal(out[:, :, :2, :], feat[:, :, :2, :]), "invalid feature rows must be identity"
    assert not torch.equal(out[:, :, 2:, :], feat[:, :, 2:, :]), "valid feature rows must change"


def test_depth_scale_must_be_positive():
    with pytest.raises(Exception):
        DepthFusion(C, C, depth_scale=0.0)


def test_clamp_bounds_must_be_ordered():
    with pytest.raises(Exception):
        DepthFusion(C, C, clamp_min=1.0, clamp_max=0.0)


# --------------------------------------------------------------------------- #
# bool mask dtype contract: typed fail-fast, no implicit coercion
# --------------------------------------------------------------------------- #
def test_mask_dtype_error_is_typeerror():
    assert issubclass(DepthFusionDtypeError, TypeError)


def test_prepare_metric_depth_accepts_bool_mask():
    d = torch.ones(2, H, W)
    m = torch.ones(2, H, W, dtype=torch.bool)
    _, mm = prepare_metric_depth(d, m, ref=_feat())
    assert mm.dtype == torch.bool


def test_prepare_metric_depth_rejects_float_mask_zero_one():
    d = torch.ones(2, H, W)
    m = torch.zeros(2, H, W, dtype=torch.float32)
    m[:, :, : W // 2] = 1.0
    with pytest.raises(DepthFusionDtypeError):
        prepare_metric_depth(d, m, ref=_feat())


def test_prepare_metric_depth_rejects_nan_float_mask():
    d = torch.ones(2, H, W)
    m = torch.full((2, H, W), float("nan"))
    with pytest.raises(DepthFusionDtypeError):
        prepare_metric_depth(d, m, ref=_feat())


@pytest.mark.parametrize("bad_dtype", [torch.float64, torch.int32, torch.int64, torch.uint8])
def test_prepare_metric_depth_rejects_other_non_bool_mask(bad_dtype):
    d = torch.ones(2, H, W)
    m = torch.ones(2, H, W, dtype=bad_dtype)
    with pytest.raises(DepthFusionDtypeError):
        prepare_metric_depth(d, m, ref=_feat())


def test_prepare_metric_depth_rejects_non_tensor_mask():
    d = torch.ones(2, H, W)
    with pytest.raises(DepthFusionDtypeError):
        prepare_metric_depth(d, [[True] * W] * H, ref=_feat())


def test_forward_rejects_non_bool_mask_even_at_gate_zero():
    df = DepthFusion(C, C, gate_init=0.0, depth_scale=1.0)
    feat = _feat()
    d = torch.full((2, H, W), 0.5)
    m = torch.ones(2, H, W, dtype=torch.float32)
    with pytest.raises(DepthFusionDtypeError):
        df(feat, d, m, image_hw=(H, W))


def test_forward_direct_mask_none_fails_fast():
    df = DepthFusion(C, C, gate_init=0.0, depth_scale=1.0)
    feat = _feat()
    d = torch.full((2, H, W), 0.5)
    with pytest.raises(DepthFusionMaskError):
        df(feat, d, None, image_hw=(H, W))
