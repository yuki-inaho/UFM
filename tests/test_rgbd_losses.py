"""CPU contract tests for the RGB-D losses / augmentation helpers (procedure 17C).

Pure-functional: no model, no GPU, no private data.  The masks are synthetic and
bool; no real dataset identifier appears here.
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

from uniflowmatch.training.rgbd_losses import (  # noqa: E402
    LossContractError,
    NonFiniteLossError,
    RGBDLossComputer,
    UnsupportedLossError,
    augmentation_consistency_loss,
    consistency_mask,
    hflip_flow,
    hflip_intrinsics,
    masked_covis_bce,
    masked_flow_epe,
    photometric_augment,
    selfsup_weight_at_step,
)

N, H, W = 2, 6, 8


def _bool_mask(p=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(N, H, W, generator=g) < p


# --------------------------------------------------------------------------- #
# masked EPE
# --------------------------------------------------------------------------- #
def test_masked_epe_finite_no_mutation_and_grad():
    pred = torch.randn(N, 2, H, W, requires_grad=True)
    target = torch.randn(N, 2, H, W)
    target[0, :, 0, 0] = float("nan")  # unknown pixel
    target_clone = target.clone()
    mask = _bool_mask(0.7)
    loss, info = masked_flow_epe(pred, target, mask)
    assert torch.isfinite(loss)
    assert torch.allclose(target, target_clone, equal_nan=True), "teacher flow must not be mutated"
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert info["support"] >= 0 and info["n_pairs"] == N


def test_masked_epe_empty_mask_graph_connected_zero():
    pred = torch.randn(N, 2, H, W, requires_grad=True)
    target = torch.randn(N, 2, H, W)
    mask = torch.zeros(N, H, W, dtype=torch.bool)
    loss, info = masked_flow_epe(pred, target, mask)
    assert float(loss) == 0.0
    assert loss.requires_grad, "empty mask must return a graph-connected zero"
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))
    assert info["support"] == 0


def test_masked_epe_nonfinite_prediction_fails_fast():
    pred = torch.randn(N, 2, H, W)
    pred[1, 0, 2, 3] = float("inf")
    with pytest.raises(NonFiniteLossError):
        masked_flow_epe(pred, torch.randn(N, 2, H, W), _bool_mask())


def test_masked_epe_rejects_non_bool_mask():
    with pytest.raises(LossContractError):
        masked_flow_epe(torch.randn(N, 2, H, W), torch.randn(N, 2, H, W), torch.zeros(N, H, W))


def test_masked_epe_low_support_recorded_not_dropped():
    pred = torch.randn(N, 2, H, W)
    target = torch.randn(N, 2, H, W)
    mask = torch.zeros(N, H, W, dtype=torch.bool)
    mask[0, 0, 0] = True  # tiny support for pair 0
    mask[1, :, :] = True  # full support for pair 1
    loss, info = masked_flow_epe(pred, target, mask, min_support_threshold=0.5)
    assert info["n_pairs"] == N  # no pair silently dropped
    assert info["n_low_support_pairs"] == 1
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# covis BCE
# --------------------------------------------------------------------------- #
def test_covis_bce_squeezes_channel_and_respects_supervision():
    logits = torch.randn(N, 1, H, W)
    target = _bool_mask(0.5, seed=1)
    sup = _bool_mask(0.4, seed=2)
    loss, info = masked_covis_bce(logits, target, sup)
    assert torch.isfinite(loss)
    assert info["covis_support"] == int(sup.sum())


def test_covis_bce_empty_support_graph_connected():
    logits = torch.randn(N, 1, H, W, requires_grad=True)
    target = _bool_mask()
    sup = torch.zeros(N, H, W, dtype=torch.bool)
    loss, _ = masked_covis_bce(logits, target, sup)
    assert float(loss) == 0.0 and loss.requires_grad


# --------------------------------------------------------------------------- #
# hflip / photometric
# --------------------------------------------------------------------------- #
def test_hflip_flow_dx_negated_and_coords_flipped():
    flow = torch.zeros(1, 2, 2, 3)
    flow[0, 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    flow[0, 1] = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    out = hflip_flow(flow)
    # width flipped then dx negated
    assert torch.equal(out[0, 0], -torch.flip(flow[0, 0], dims=[-1]))
    assert torch.equal(out[0, 1], torch.flip(flow[0, 1], dims=[-1]))


def test_hflip_intrinsics_principal_point_and_fx():
    K = torch.tensor([[[100.0, 0.0, 3.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]]])
    out = hflip_intrinsics(K, W)
    assert float(out[0, 0, 0]) == -100.0  # fx -> -fx (matches loader convention)
    assert float(out[0, 0, 2]) == float(W - 1 - 3)
    assert float(out[0, 1, 1]) == 120.0


def test_photometric_augment_clamp_and_renorm():
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    img01 = torch.rand(1, 3, 4, 5)
    rgb = (img01 - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    out = photometric_augment(rgb, mean, std, brightness=0.5, contrast=2.0)
    expected = torch.clamp(img01 * 2.0 + 0.5, 0.0, 1.0)
    expected_norm = (expected - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    assert torch.allclose(out, expected_norm, atol=1e-6)


def test_consistency_mask_in_bounds_finite():
    valid = torch.ones(1, H, W, dtype=torch.bool)
    flow = torch.zeros(1, 2, H, W)
    flow[0, 0, 0, 0] = -5.0  # endpoint out of bounds
    mask = consistency_mask(valid, flow, H, W)
    assert not bool(mask[0, 0, 0])
    assert bool(mask[0, 1, 1])


# --------------------------------------------------------------------------- #
# selfsup warmup / consistency
# --------------------------------------------------------------------------- #
def test_selfsup_weight_positive_at_step_one():
    assert selfsup_weight_at_step(0.1, 100, 1) == pytest.approx(0.001)
    assert selfsup_weight_at_step(0.1, 100, 200) == pytest.approx(0.1)
    assert selfsup_weight_at_step(0.0, 100, 1) == 0.0
    assert selfsup_weight_at_step(0.1, 0, 1) == pytest.approx(0.1)


def test_augmentation_consistency_requires_detached_original():
    aug = torch.randn(N, 2, H, W, requires_grad=True)
    original = torch.randn(N, 2, H, W, requires_grad=True)  # NOT detached
    with pytest.raises(LossContractError):
        augmentation_consistency_loss(original, aug, _bool_mask())


def test_augmentation_consistency_gradient_only_on_augmented():
    aug = torch.randn(N, 2, H, W, requires_grad=True)
    original = torch.randn(N, 2, H, W).detach()
    loss, _ = augmentation_consistency_loss(original, aug, _bool_mask())
    loss.backward()
    assert aug.grad is not None and torch.isfinite(aug.grad).all()


# --------------------------------------------------------------------------- #
# config-driven computer
# --------------------------------------------------------------------------- #
def _selfsup_off_cfg():
    return {
        "name": "selfsup_off",
        "supervised": {
            "flow": {"class": "FlowEPELoss", "multiplier": 1.0, "supervision_range": "occlusion",
                     "minimum_mask_threshold": 0.1},
            "covis_bce": {"class": "CrossEntropyLoss", "multiplier": 1.0},
        },
        "selfsup": {"enabled": False, "weight": 0.0, "warmup_updates": 100,
                    "detach_original_prediction": True, "same_geometry": True,
                    "aug": {"hflip": True, "photometric": True, "crop": False, "complex_warp": False}},
        "homography": {"enabled": False},
        "laplace_nll": {"enabled": False, "detach_residual": True},
    }


def test_loss_computer_supervised_terms_finite():
    comp = RGBDLossComputer(_selfsup_off_cfg())
    pred = torch.randn(N, 2, H, W)
    target = torch.randn(N, 2, H, W)
    loss, info = comp.supervised_terms(
        pred, target, _bool_mask(),
        covis_logits=torch.randn(N, 1, H, W),
        covis_target=_bool_mask(seed=3),
        covis_supervision=_bool_mask(seed=4),
    )
    assert torch.isfinite(loss) and "supervised_loss" in info and "covis_loss" in info


def test_loss_computer_selfsup_off_returns_zero_weight():
    comp = RGBDLossComputer(_selfsup_off_cfg())
    assert comp.weight_at_step(1) == 0.0
    assert not comp.selfsup_enabled


def test_loss_computer_selfsup_on_weight_at_step_one_positive():
    cfg = _selfsup_off_cfg()
    cfg["name"] = "selfsup_on"
    cfg["selfsup"]["enabled"] = True
    cfg["selfsup"]["weight"] = 0.1
    comp = RGBDLossComputer(cfg)
    assert comp.weight_at_step(1) > 0.0


def test_loss_computer_rejects_unsupported_settings():
    bad = _selfsup_off_cfg()
    bad["homography"]["enabled"] = True
    with pytest.raises(UnsupportedLossError):
        RGBDLossComputer(bad)

    bad = _selfsup_off_cfg()
    bad["selfsup"]["aug"]["crop"] = True
    with pytest.raises(UnsupportedLossError):
        RGBDLossComputer(bad)

    bad = _selfsup_off_cfg()
    bad["supervised"]["flow"]["class"] = "SomeOtherLoss"
    with pytest.raises(UnsupportedLossError):
        RGBDLossComputer(bad)

    bad = _selfsup_off_cfg()
    bad["supervised"]["flow"]["supervision_range"] = "made_up"
    with pytest.raises(UnsupportedLossError):
        RGBDLossComputer(bad)

    bad = _selfsup_off_cfg()
    bad["laplace_nll"]["enabled"] = True
    with pytest.raises(UnsupportedLossError):
        RGBDLossComputer(bad, model_has_flow_covariance=False)
    # accepted only when the model really exposes a flow-covariance head
    RGBDLossComputer(bad, model_has_flow_covariance=True)


# --------------------------------------------------------------------------- #
# photometric augmentation must follow the RGB device (CPU stats + CUDA image)
# --------------------------------------------------------------------------- #
def test_photometric_augment_cpu_stats_cuda_image():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for the device-transfer oracle")
    mean = torch.tensor([0.5, 0.5, 0.5])  # CPU stats, as the loader/dataset yields
    std = torch.tensor([0.5, 0.5, 0.5])
    img01 = torch.rand(2, 3, 4, 5)
    rgb = (img01 - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    expected = photometric_augment(rgb, mean, std, brightness=0.1, contrast=1.2)
    got = photometric_augment(rgb.cuda(), mean, std, brightness=0.1, contrast=1.2)
    assert got.device.type == "cuda"
    assert torch.allclose(got.cpu(), expected, atol=1e-6)
