"""CPU contract tests for true RGB-only fine-tuning (no depth fusion).

These tests exercise the *real* tiny-network ``UniFlowMatchConfidence`` (the
same fixture pattern as ``tests/test_rgbd_trainer.py``) plus the real AMUSE
wrapper.  No GPU, no real data, no private identifiers.

The suite pins the RGB-only contract:

* ``model.use_depth_fusion`` is a strict bool; the RGB preset carries no depth
  kwargs and does not require a depth freeze.
* ``build_rgbd_model(..., rgb_only=True)`` disables fusion and omits
  ``depth_fusion_kwargs``; the default (RGB-D) path is byte-for-byte unchanged.
* the RGB-only model has zero ``depth_fusion`` state / parameters / groups and
  the optimizer plan measures no depth param.
* ``RGBDTrainer.forward`` strips depth / depth-mask / provenance keys from the
  view sent to an RGB model on every path (train / aug / validation) while the
  RGB-D default keeps the depth keys.
* the released RGB weights load strictly through the existing alias-aware
  ``load_released_state`` (never the RGB->RGBD migration helper).
* a resume identity mismatch is rejected *before* any state is copied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import uniflowmatch.models.ufm as U  # noqa: E402
from uniflowmatch.training.rgbd_config import (  # noqa: E402
    ShapeError,
    validate_config,
    without_hydra,
)
from uniflowmatch.training.rgbd_optimizer import (  # noqa: E402
    build_optimizer,
    build_parameter_plan,
)

DIM = 8
H = W = 16


# --------------------------------------------------------------------------- #
# tiny real model (mirrors tests/test_rgbd_trainer.py)
# --------------------------------------------------------------------------- #
class _FeatOut:
    def __init__(self, features):
        self.features = features


class _AdaptorOut:
    def __init__(self, value):
        self.value = value


class _MaskOut:
    def __init__(self, logits):
        self.logits = logits
        self.mask = torch.sigmoid(logits)


class TinyEncoder(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, 1)

    def forward(self, encoder_input):
        return [_FeatOut(self.proj(encoder_input.image))]


class TinyInfoSharing(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.lin = nn.Conv2d(DIM, DIM, 1)

    def forward(self, info_sharing_input):
        f1, f2 = info_sharing_input.features
        o1, o2 = self.lin(f1), self.lin(f2)
        return _FeatOut([o1, o2]), [_FeatOut([o1, o2]), _FeatOut([o1, o2])]


class TinyFeatureProcessor(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.lin = nn.Conv2d(DIM, DIM, 1)

    def forward(self, head_input):
        return self.lin(head_input.list_features[-1])


class TinyAdaptor(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class TinyAdaptorMap(nn.Module):
    def __init__(self, *adaptors):
        super().__init__()
        self.lin = nn.Conv2d(DIM, 2, 1)
        self.mask_lin = nn.Conv2d(DIM, 1, 1)

    def forward(self, feat):
        return {
            "flow": _AdaptorOut(self.lin(feat)),
            "non_occluded_mask": _MaskOut(self.mask_lin(feat)),
        }


def _build_tiny_model(*, use_depth_fusion=True):
    saved = {
        n: getattr(U, n)
        for n in (
            "feature_returner_encoder_factory", "INFO_SHARING_CLASSES",
            "CLASSNAME_TO_ADAPTOR_CLASS", "DPTFeature", "DPTRegressionProcessor", "AdaptorMap",
        )
    }
    U.feature_returner_encoder_factory = lambda *a, **k: TinyEncoder()
    U.INFO_SHARING_CLASSES = {"global_attention": (None, TinyInfoSharing)}
    U.CLASSNAME_TO_ADAPTOR_CLASS = {"FlowAdaptor": TinyAdaptor}
    U.DPTFeature = TinyFeatureProcessor
    U.DPTRegressionProcessor = lambda *a, **k: nn.Identity()
    U.AdaptorMap = TinyAdaptorMap
    try:
        model = U.UniFlowMatchConfidence(
            encoder_str="tiny",
            encoder_kwargs={},
            info_sharing_str="global_attention",
            info_sharing_kwargs={},
            head_type="dpt",
            feature_head_kwargs={"dpt_feature": {}, "dpt_processor": {}},
            adaptors_kwargs={"flow": {"class": "FlowAdaptor", "kwargs": {}}},
            uncertainty_head_kwargs={"dpt_feature": {}, "dpt_processor": {}},
            uncertainty_adaptors_kwargs={"flow": {"class": "FlowAdaptor", "kwargs": {}}},
            use_depth_fusion=use_depth_fusion,
            depth_fusion_kwargs={"feature_dim": DIM, "depth_hidden_dim": DIM,
                                 "gate_init": 0.0, "depth_scale": 1.0},
            inference_resolution=(64, 48),
        )
    finally:
        for n, v in saved.items():
            setattr(U, n, v)
    return model


def _view(seed, *, pair_id="p", flow_val=1.0, mask=True, instance="i"):
    g = torch.Generator().manual_seed(seed)
    return {
        "img": torch.randn(3, H, W, generator=g),
        "data_norm_type": "dinov2",
        "symmetrized": False,
        "flow": torch.full((2, H, W), float(flow_val)),
        "non_occluded_mask": torch.full((H, W), bool(mask), dtype=torch.bool),
        "occlusion_supervision_mask": torch.full((H, W), bool(mask), dtype=torch.bool),
        "fov_mask": torch.ones(H, W, dtype=torch.bool),
        "dilated_depth_m": torch.full((H, W), 0.5),
        "dilated_depth_window_mask": torch.ones(H, W, dtype=torch.bool),
        "raw_depth_m": torch.full((H, W), 0.5),
        "raw_depth_window_mask": torch.ones(H, W, dtype=torch.bool),
        "depth_provenance": torch.ones(H, W),
        "instance": instance,
        "pair_id": pair_id,
    }


def _pairs(n=2):
    return [[_view(i * 2, flow_val=1.0 + i, instance=f"a{i}", pair_id=f"p{i}"),
             _view(i * 2 + 1, flow_val=-1.0 - i, instance=f"b{i}", pair_id=f"p{i}")]
            for i in range(n)]


def _opt_cfg():
    return {
        "name": "amuse", "schedule_free": True, "external_scheduler": None,
        "lr": 1e-2, "sgd_lr": 1e-2, "weight_decay": 0.0,
        "constructor_args": {"beta1": 0.9, "weight_lr_power": 2.0, "warmup_steps": 2,
                             "rho": 1.0, "r": 0.0, "weight_decay_at_y": 0.0},
        "groups": {"muon": {"use_muon": True, "lr": 1e-2, "momentum": 0.95,
                            "aux_update_type": "sgd", "weight_decay": 0.0},
                   "aux": {"use_muon": False, "update_type": "sgd", "lr": 1e-2, "weight_decay": 0.0}},
        "train_eval": {"eval_returns_averaged_x": True, "save_training_y": True,
                       "save_training_z": True, "save_extra_attrs": ["train_mode"]},
        "gradient_clip": {"algorithm": "norm", "value": 1.0},
    }


def _loss_cfg(enabled=False, weight=0.0, hflip=True, photometric=True):
    return {
        "name": "selfsup_on" if enabled else "selfsup_off",
        "supervised": {"flow": {"class": "FlowEPELoss", "multiplier": 1.0,
                                "supervision_range": "occlusion", "minimum_mask_threshold": 0.1},
                       "covis_bce": {"class": "CrossEntropyLoss", "multiplier": 1.0}},
        "selfsup": {"enabled": enabled, "weight": weight, "warmup_updates": 100,
                    "detach_original_prediction": True, "same_geometry": True,
                    "aug": {"hflip": hflip, "photometric": photometric,
                            "crop": False, "complex_warp": False}},
        "homography": {"enabled": False},
        "laplace_nll": {"enabled": False, "detach_residual": True},
    }


def _trainer(*, rgb_only=True, selfsup=False, weight=0.0, seed=0):
    from uniflowmatch.training.rgbd_trainer import RGBDTrainer

    model = _build_tiny_model(use_depth_fusion=not rgb_only)
    cfg = {"seed": seed, "optimizer": _opt_cfg(), "loss": _loss_cfg(selfsup, weight),
           "model": {"use_depth_fusion": not rgb_only}}
    opt = build_optimizer(model, cfg)
    return model, opt, RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu", seed=seed)


# --------------------------------------------------------------------------- #
# config schema: strict bool + RGB preset
# --------------------------------------------------------------------------- #
def _compose(overrides):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    cfg_dir = _REPO / "configs" / "rgbd_training"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
        cfg = compose(config_name="default", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def _public_config():
    return without_hydra(_compose([]))


@pytest.mark.parametrize("bad", [0, 1, "true", "false", None, 0.0, []])
def test_config_rejects_non_bool_use_depth_fusion(bad):
    cfg = _public_config()
    cfg["model"]["use_depth_fusion"] = bad
    with pytest.raises(ShapeError):
        validate_config(cfg, require_external=False)


def test_rgb_only_preset_validates_without_depth_kwargs():
    cfg = without_hydra(_compose(["model=ufm_rgb_only"]))
    validate_config(cfg, require_external=False)
    assert cfg["model"]["use_depth_fusion"] is False
    assert "depth_fusion_kwargs" not in cfg["model"]
    # no mandatory depth freeze for the RGB-only model
    assert cfg["model"]["freeze"].get("depth_fusion") in (None, False)
    assert cfg["model"]["freeze"]["encoder"] is True


def test_true_branch_keeps_depth_guards():
    cfg = _public_config()
    # true without depth kwargs -> rejected
    cfg["model"].pop("depth_fusion_kwargs")
    with pytest.raises(ShapeError):
        validate_config(cfg, require_external=False)
    # true with a bad gate -> rejected
    cfg = _public_config()
    cfg["model"]["depth_fusion_kwargs"]["gate_init"] = 0.5
    with pytest.raises(ShapeError):
        validate_config(cfg, require_external=False)
    # true with freeze.depth_fusion=true -> rejected
    cfg = _public_config()
    cfg["model"]["freeze"]["depth_fusion"] = True
    with pytest.raises(ShapeError):
        validate_config(cfg, require_external=False)
    # false with leftover depth kwargs -> rejected (no unused depth kwargs)
    cfg = _public_config()
    cfg["model"]["use_depth_fusion"] = False
    with pytest.raises(ShapeError):
        validate_config(cfg, require_external=False)


# --------------------------------------------------------------------------- #
# builder / structure / optimizer coverage
# --------------------------------------------------------------------------- #
def test_build_rgb_model_disables_fusion_and_drops_depth_kwargs():
    from uniflowmatch.training.rgbd_trainer import build_rgbd_model

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    build_rgbd_model(
        {"encoder_str": "dinov2", "pretrained_checkpoint_path": "ignored"},
        rgb_only=True, model_class=FakeModel,
    )
    assert captured["use_depth_fusion"] is False
    assert "depth_fusion_kwargs" not in captured
    assert "pretrained_checkpoint_path" not in captured


def test_build_rgbd_model_default_still_forces_fusion():
    from uniflowmatch.training.rgbd_trainer import build_rgbd_model

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # no rgb_only keyword -> unchanged RGB-D behaviour (no `false` keyword).
    build_rgbd_model(
        {"encoder_str": "dinov2"},
        fusion_kwargs={"feature_dim": 1024, "depth_hidden_dim": 64,
                       "gate_init": 0.0, "depth_scale": 1.3},
        model_class=FakeModel,
    )
    assert captured["use_depth_fusion"] is True
    assert captured["depth_fusion_kwargs"]["gate_init"] == 0.0


def test_rgb_only_model_has_no_depth_state_or_group():
    model = _build_tiny_model(use_depth_fusion=False)
    assert model.depth_fusion is None
    assert not any(k.startswith("depth_fusion.") for k in model.state_dict())
    assert "depth_fusion" not in model.get_parameter_groups()


def test_rgb_only_optimizer_plan_measures_no_depth_param():
    rgb = _build_tiny_model(use_depth_fusion=False)
    rgbd = _build_tiny_model(use_depth_fusion=True)

    plan_rgb = build_parameter_plan(rgb)
    plan_rgbd = build_parameter_plan(rgbd)

    assert not any(n.startswith("depth_fusion.") for n, _ in plan_rgb.muon + plan_rgb.aux)
    assert all("encoder." not in n for n, _ in plan_rgb.muon + plan_rgb.aux)
    # every trainable param is covered exactly once (measured by the helper)
    assigned = {n for n, _ in plan_rgb.muon} | {n for n, _ in plan_rgb.aux}
    assert len(assigned) == plan_rgb.trainable_count
    # the only delta vs RGB-D is the depth-fusion module parameters.
    depth_names = {n for n, _ in plan_rgbd.muon + plan_rgbd.aux if n.startswith("depth_fusion.")}
    assert plan_rgbd.trainable_count - plan_rgb.trainable_count == len(depth_names)


# --------------------------------------------------------------------------- #
# forward depth-key stripping (spy fails if depth keys are present)
# --------------------------------------------------------------------------- #
def _depth_keys(view):
    return [k for k in ("dilated_depth_m", "raw_depth_m", "depthmap", "depth",
                        "dilated_depth_window_mask", "raw_depth_window_mask",
                        "valid_mask", "depth_provenance") if k in view]


def test_forward_strips_depth_keys_for_rgb_model():
    model, _opt, trainer = _trainer(rgb_only=True)
    seen = {"n": 0}
    real_forward = model.forward

    def spy(v1, v2):
        seen["n"] += 1
        assert _depth_keys(v1) == [], f"depth keys leaked to RGB forward: {_depth_keys(v1)}"
        assert _depth_keys(v2) == []
        assert "img" in v1 and "img" in v2
        return real_forward(v1, v2)

    model.forward = spy
    trainer.train_step(_pairs(2), global_step=1)
    assert seen["n"] == 1


def test_forward_spy_strips_depth_on_aug_and_validation_paths():
    model, _opt, trainer = _trainer(rgb_only=True, selfsup=True, weight=0.1)
    calls = []

    real_forward = model.forward

    def spy(v1, v2):
        calls.append(_depth_keys(v1) + _depth_keys(v2))
        return real_forward(v1, v2)

    model.forward = spy
    trainer.train_step(_pairs(2), global_step=1)  # original + augmented forwards
    trainer.validate([_pairs(2)], max_pairs=1)     # validation forward
    assert len(calls) >= 3
    assert all(keys == [] for keys in calls)


def test_rgbd_default_forward_keeps_depth_keys():
    model, _opt, trainer = _trainer(rgb_only=False)
    seen = {"n": 0}

    real_forward = model.forward

    def spy(v1, v2):
        seen["n"] += 1
        assert "dilated_depth_m" in v1  # RGB-D still receives depth
        return real_forward(v1, v2)

    model.forward = spy
    trainer.train_step(_pairs(2), global_step=1)
    assert seen["n"] == 1


def test_strip_depth_input_keys_removes_only_depth_and_copies():
    from uniflowmatch.training.rgbd_trainer import strip_depth_input_keys

    view = _view(1)
    stripped = strip_depth_input_keys(view)
    assert _depth_keys(stripped) == []
    assert "img" in stripped and "flow" in stripped and "camera_intrinsics" not in stripped
    # original view is untouched
    assert _depth_keys(view) != []


# --------------------------------------------------------------------------- #
# strict released RGB load (alias-aware, no migration helper)
# --------------------------------------------------------------------------- #
def test_load_released_rgb_weights_strict(tmp_path):
    from uniflowmatch.models.checkpoint import (
        CheckpointIdentityError,
        CheckpointStateError,
        sha256_file,
    )
    from uniflowmatch.training.rgbd_trainer import load_released_rgb_weights

    model = _build_tiny_model(use_depth_fusion=False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = tmp_path / "rgb.pt"
    torch.save(state, ckpt)
    sha = sha256_file(str(ckpt))

    report = load_released_rgb_weights(model, ckpt, expected_sha256=sha)
    assert report["sha256"] == sha

    with pytest.raises(CheckpointIdentityError):
        load_released_rgb_weights(model, ckpt, expected_sha256="0" * 64)

    bad = dict(state)
    bad.pop(next(iter(bad)))
    bad_ckpt = tmp_path / "bad.pt"
    torch.save(bad, bad_ckpt)
    with pytest.raises(CheckpointStateError):
        load_released_rgb_weights(model, bad_ckpt)


# --------------------------------------------------------------------------- #
# modality mismatch / resume identity before copy
# --------------------------------------------------------------------------- #
def test_rgb_state_rejected_into_rgbd_model_and_vice_versa():
    from uniflowmatch.models.checkpoint import CheckpointStateError, strict_load_state_dict

    rgb = _build_tiny_model(use_depth_fusion=False)
    rgbd = _build_tiny_model(use_depth_fusion=True)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(rgbd, rgb.state_dict(), require_fp32=True)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(rgb, rgbd.state_dict(), require_fp32=True)


def test_resume_identity_mismatch_rejected_before_copy(tmp_path):
    from uniflowmatch.training.rgbd_checkpoint import ResumeIdentityError

    _model, _opt, trainer = _trainer(rgb_only=True)
    trainer.identity = {"sha": "a" * 64}
    ckpt = tmp_path / "last_y.pt"
    trainer.save_last_y(ckpt, sampler_state=_sampler_state(len_=2))
    assert ckpt.is_file()

    _model2, _opt2, trainer2 = _trainer(rgb_only=True)
    trainer2.identity = {"sha": "b" * 64}
    copied = {"n": 0}
    real_load = trainer2.model.load_state_dict

    def counting_load(*a, **k):
        copied["n"] += 1
        return real_load(*a, **k)

    trainer2.model.load_state_dict = counting_load
    with pytest.raises(ResumeIdentityError):
        trainer2.load_last_y(ckpt)
    assert copied["n"] == 0  # rejected before any state copy


def _sampler_state(len_):
    from uniflowmatch.training.rgbd_checkpoint import StatefulPermutationSampler

    return StatefulPermutationSampler(len_, seed=0).state_dict()


# --------------------------------------------------------------------------- #
# entry modality decision (pure helper)
# --------------------------------------------------------------------------- #
def test_train_entry_modality_helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_rgbd_mod", str(_REPO / "scripts" / "train_rgbd.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_rgbd_mod"] = mod
    spec.loader.exec_module(mod)
    assert mod._modality_is_rgb({"model": {"use_depth_fusion": False}}) is True
    assert mod._modality_is_rgb({"model": {"use_depth_fusion": True}}) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
