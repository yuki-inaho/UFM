"""CPU contract tests for the RGB-D trainer (procedure 17C).

Uses a *real* ``UniFlowMatchConfidence`` on tiny network-free submodules (the
same fixture pattern as ``tests/test_rgbd_fusion_adr_red.py``) plus the real
AMUSE wrapper.  No GPU, no real data, no private identifiers.
"""

from __future__ import annotations

import json

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
from uniflowmatch.models.base import UFMOutputInterface  # noqa: E402
from uniflowmatch.training.rgbd_optimizer import build_optimizer  # noqa: E402
from uniflowmatch.training.rgbd_trainer import (  # noqa: E402
    RGBDTrainer,
    TrainerError,
    assert_no_half,
    freeze_encoder,
    pack_pair_batch,
    param_grad_norm,
)

DIM = 8
H = W = 16


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
    v = {
        "img": torch.randn(3, H, W, generator=g),
        "data_norm_type": "dinov2",
        "symmetrized": False,
        "flow": torch.full((2, H, W), float(flow_val)),
        "non_occluded_mask": torch.full((H, W), bool(mask), dtype=torch.bool),
        "occlusion_supervision_mask": torch.full((H, W), bool(mask), dtype=torch.bool),
        "fov_mask": torch.ones(H, W, dtype=torch.bool),
        "dilated_depth_m": torch.full((H, W), 0.5),
        "dilated_depth_window_mask": torch.ones(H, W, dtype=torch.bool),
        "instance": instance,
        "pair_id": pair_id,
    }
    return v


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


def _loss_cfg(enabled=False, weight=0.0):
    return {
        "name": "selfsup_on" if enabled else "selfsup_off",
        "supervised": {"flow": {"class": "FlowEPELoss", "multiplier": 1.0,
                                "supervision_range": "occlusion", "minimum_mask_threshold": 0.1},
                       "covis_bce": {"class": "CrossEntropyLoss", "multiplier": 1.0}},
        "selfsup": {"enabled": enabled, "weight": weight, "warmup_updates": 100,
                    "detach_original_prediction": True, "same_geometry": True,
                    "aug": {"hflip": True, "photometric": True, "crop": False, "complex_warp": False}},
        "homography": {"enabled": False},
        "laplace_nll": {"enabled": False, "detach_residual": True},
    }


def _trainer(*, selfsup=False, weight=0.0, seed=0):
    model = _build_tiny_model()
    cfg = {"seed": seed, "optimizer": _opt_cfg(), "loss": _loss_cfg(selfsup, weight)}
    opt = build_optimizer(model, cfg)
    return model, opt, RGBDTrainer(cfg, model=model, optimizer=opt, device="cpu", seed=seed)


# --------------------------------------------------------------------------- #
# packing / freeze / precision
# --------------------------------------------------------------------------- #
def test_pack_pair_batch_abba_alignment():
    pairs = _pairs(2)
    v1, v2, sup = pack_pair_batch(pairs)
    assert v1["symmetrized"] is True
    assert v1["img"].shape[0] == 4 and v2["img"].shape[0] == 4
    # flow rows alternate forward (a) then backward (b) target
    f = sup["flow"]
    assert float(f[0, 0, 0, 0]) == 1.0 and float(f[1, 0, 0, 0]) == -1.0
    assert float(f[2, 0, 0, 0]) == 2.0 and float(f[3, 0, 0, 0]) == -2.0
    # view1 img row0 == a0, row1 == b0 (ABBA)
    assert torch.equal(v1["img"][0], pairs[0][0]["img"])
    assert torch.equal(v1["img"][1], pairs[0][1]["img"])
    assert torch.equal(v2["img"][0], pairs[0][1]["img"])


def test_freeze_encoder_survives_model_train():
    model, _, trainer = _trainer()
    frozen = freeze_encoder(model)
    assert frozen and all(not p.requires_grad for p in model.encoder.parameters())
    trainer.set_train_mode()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert model.info_sharing.training  # trainable modules stay in train mode


def test_assert_no_half_rejects_half_param():
    model, _, _ = _trainer()
    assert_no_half(model)
    with torch.no_grad():
        model.depth_fusion.gate.data = model.depth_fusion.gate.data.half()
    with pytest.raises(TrainerError):
        assert_no_half(model)


# --------------------------------------------------------------------------- #
# two-step optimisation: both directions, freeze, gate->CNN gradient
# --------------------------------------------------------------------------- #
def test_two_steps_finite_both_directions_and_frozen_encoder():
    model, opt, trainer = _trainer()
    enc_before = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
    pairs = _pairs(2)
    i1 = trainer.train_step(pairs, global_step=1)
    i2 = trainer.train_step(pairs, global_step=2)
    assert i1["loss"] > 0 and i2["loss"] > 0
    assert i1["grad_norm"] > 0 and i2["grad_norm"] > 0
    assert i1["encoder_grad_norm"] == 0.0  # frozen encoder has no gradient
    for n, p in model.encoder.named_parameters():
        assert torch.equal(p, enc_before[n]), f"frozen encoder param {n} changed"


def test_gate_changes_first_then_cnn_gradient():
    model, opt, trainer = _trainer()
    pairs = _pairs(2)
    # step 1: gate gradient is non-zero, adapter (CNN) gradient is zero at gate==0
    trainer.set_train_mode()
    opt.zero_grad()
    v1, v2, sup = pack_pair_batch(pairs)
    result = trainer.forward(v1, v2)
    loss, _ = trainer.compute_supervised_loss(result, sup)
    loss.backward()
    assert param_grad_norm(model, "depth_fusion.gate") > 0.0
    assert param_grad_norm(model, "depth_fusion.depth_adapter") == 0.0
    trainer.train_step(pairs, global_step=1)
    assert trainer._gate_value() != 0.0
    # step 2: with a non-zero gate the CNN adapter now receives gradient
    trainer.set_train_mode()
    opt.zero_grad()
    result = trainer.forward(v1, v2)
    loss, _ = trainer.compute_supervised_loss(result, sup)
    loss.backward()
    assert param_grad_norm(model, "depth_fusion.depth_adapter") > 0.0


# --------------------------------------------------------------------------- #
# self-supervision
# --------------------------------------------------------------------------- #
def test_selfsup_on_augmented_gradient_and_positive_warmup():
    model, opt, trainer = _trainer(selfsup=True, weight=0.1)
    info = trainer.train_step(_pairs(2), global_step=1)
    assert info["selfsup_weight"] > 0.0  # warm-up step 1 is strictly positive
    assert info["selfsup_loss"] > 0.0
    assert info["gate_value"] != 0.0


def test_selfsup_off_does_no_extra_forward():
    model, opt, trainer = _trainer(selfsup=False)
    calls = {"n": 0}
    original = model.forward

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    model.forward = counting
    trainer.train_step(_pairs(2), global_step=1)
    assert calls["n"] == 1

    model2, opt2, trainer2 = _trainer(selfsup=True, weight=0.1)
    calls2 = {"n": 0}
    original2 = model2.forward

    def counting2(*a, **k):
        calls2["n"] += 1
        return original2(*a, **k)

    model2.forward = counting2
    trainer2.train_step(_pairs(2), global_step=1)
    assert calls2["n"] == 2  # original + augmented


def test_selfsup_empty_mask_is_finite_graph_connected_zero():
    model, opt, trainer = _trainer(selfsup=True, weight=0.1)
    pairs = _pairs(2)
    for pair in pairs:
        for view in pair:
            view["non_occluded_mask"] = torch.zeros(H, W, dtype=torch.bool)
            view["occlusion_supervision_mask"] = torch.zeros(H, W, dtype=torch.bool)
    info = trainer.train_step(pairs, global_step=1)
    assert torch.isfinite(torch.tensor(info["loss"]))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validation_restores_modes_and_uses_teacher_epe():
    model, opt, trainer = _trainer()
    trainer.train_step(_pairs(2), global_step=1)
    opt.train()
    metrics = trainer.validate([_pairs(2)], max_pairs=1)
    assert "val_teacher_epe" in metrics and metrics["n_pairs"] == 1
    assert opt.train_mode is True  # restored to training y
    assert not model.encoder.training  # encoder freeze re-applied
    assert trainer.best_monitor["name"] == "val_teacher_epe"


# --------------------------------------------------------------------------- #
# experiment contract / deadline / model kwargs
# --------------------------------------------------------------------------- #
def test_check_experiment_contract():
    from uniflowmatch.training.rgbd_trainer import (
        check_experiment_contract,
        final_validation_and_save_required,
        DeadlineGuard,
        DeadlineReached,
    )

    smoke = {"experiment": {"name": "smoke", "max_updates": 1, "validation_max_pairs": 2}}
    check_experiment_contract(smoke)
    assert final_validation_and_save_required(smoke)

    full = {"experiment": {"name": "full", "max_updates": None, "epochs": 12,
                           "validate_full_each_epoch": True, "early_stop": False,
                           "full_budget_fixed": False}}
    with pytest.raises(TrainerError):
        check_experiment_contract(full)  # budget not fixed yet
    full["experiment"]["full_budget_fixed"] = True
    check_experiment_contract(full)

    guard = DeadlineGuard(0.0)
    with pytest.raises(DeadlineReached):
        guard.check()


def test_build_rgbd_model_applies_bare_fusion_dict():
    from uniflowmatch.training.rgbd_trainer import build_rgbd_model

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    build_rgbd_model(
        {"encoder_str": "dinov2", "pretrained_checkpoint_path": "ignored"},
        fusion_kwargs={"feature_dim": 1024, "depth_hidden_dim": 64, "gate_init": 0.0, "depth_scale": 1.3},
        model_class=FakeModel,
    )
    assert captured["use_depth_fusion"] is True
    assert captured["depth_fusion_kwargs"] == {
        "feature_dim": 1024, "depth_hidden_dim": 64, "gate_init": 0.0, "depth_scale": 1.3}
    assert "pretrained_checkpoint_path" not in captured


def test_local_dino_hub_verifies_tree_sha(tmp_path):
    from uniflowmatch.training.rgbd_trainer import (
        local_dino_hub, dino_python_tree_sha256, LocalDinoHubError,
    )

    (tmp_path / "hubconf.py").write_text("# hub\n")
    good = dino_python_tree_sha256(tmp_path)
    with local_dino_hub(tmp_path, "dinov2_vitl14", good, pretrained=False) as d:
        assert d == str(tmp_path)
    with pytest.raises(LocalDinoHubError):
        with local_dino_hub(tmp_path, "dinov2_vitl14", "0" * 64, pretrained=False):
            pass


# --------------------------------------------------------------------------- #
# run loop (synthetic tiny dataset)
# --------------------------------------------------------------------------- #
def test_run_smoke_writes_last_y_and_best_x(tmp_path):
    from uniflowmatch.training.rgbd_trainer import EventsLogger

    model, opt, trainer = _trainer()
    trainer.events = EventsLogger(tmp_path / "events.jsonl")
    exp = {"name": "smoke", "max_updates": 1, "validation_max_pairs": 2,
           "validate_every_updates": 1, "save_every_updates": 1, "early_stop": False}
    res = trainer.run(_pairs(2), _pairs(2), experiment=exp, checkpoint_dir=tmp_path / "ckpt")
    assert res["status"] == "finished" and res["global_step"] == 1
    assert (tmp_path / "ckpt" / "last_y.pt").is_file()
    assert (tmp_path / "ckpt" / "best_averaged_x.pt").is_file()
    stages = {json.loads(l)["stage"] for l in (tmp_path / "events.jsonl").read_text().splitlines()}
    assert {"train", "val", "save", "best", "finish"} <= stages


def test_run_full_requires_budget_fixed(tmp_path):
    model, opt, trainer = _trainer()
    exp = {"name": "full", "max_updates": None, "epochs": 2,
           "validate_full_each_epoch": True, "early_stop": False, "full_budget_fixed": False}
    with pytest.raises(TrainerError):
        trainer.run(_pairs(2), _pairs(2), experiment=exp, checkpoint_dir=tmp_path / "ckpt")


def test_run_deadline_stops_and_keeps_last_y(tmp_path):
    model, opt, trainer = _trainer()
    exp = {"name": "short", "max_updates": 8, "validation_max_pairs": 2,
           "validate_every_updates": 4, "save_every_updates": 4, "early_stop": False}
    res = trainer.run(_pairs(2), _pairs(2), experiment=exp,
                      checkpoint_dir=tmp_path / "ckpt", deadline_s=0.0)
    assert res["status"] == "stopped"
    assert (tmp_path / "ckpt" / "last_y.pt").is_file()


# --------------------------------------------------------------------------- #
# packing metadata must interleave exactly like the tensors (2-pair oracle)
# --------------------------------------------------------------------------- #
def test_pack_pair_batch_metadata_interleaves_with_tensors():
    # two DIFFERENT pairs: distinct flow values, pair_ids and instances.
    pairs = [
        [_view(10, flow_val=3.0, instance="A0", pair_id="pid0"),
         _view(11, flow_val=-4.0, instance="B0", pair_id="pid0")],
        [_view(20, flow_val=5.0, instance="A1", pair_id="pid1"),
         _view(21, flow_val=-6.0, instance="B1", pair_id="pid1")],
    ]
    v1, v2, sup = pack_pair_batch(pairs)
    # tensor rows are A0,B0,A1,B1 (view1) / B0,A0,B1,A1 (view2)
    for row, (inst, fl) in enumerate([
        ("A0", 3.0), ("B0", -4.0), ("A1", 5.0), ("B1", -6.0)]):
        assert v1["instance"][row] == inst
        assert float(sup["flow"][row, 0, 0, 0]) == fl
    assert v2["instance"] == ["B0", "A0", "B1", "A1"]
    # pair_ids follow the same tensor interleave (pid0,pid0,pid1,pid1)
    assert sup["pair_ids"] == ["pid0", "pid0", "pid1", "pid1"]
    # view2 tensors are the BA ordering of the same pair
    assert torch.equal(v2["img"][0], pairs[0][1]["img"])
    assert torch.equal(v2["img"][1], pairs[0][0]["img"])


# --------------------------------------------------------------------------- #
# self-supervised augmentation must follow the hflip flag
# --------------------------------------------------------------------------- #
def _selfsup_trainer(hflip, photometric=False):
    model, opt, trainer = _trainer(selfsup=True, weight=0.1)
    trainer.loss.selfsup["hflip"] = hflip
    trainer.loss.selfsup["photometric"] = photometric
    return trainer


def test_augment_views_hflip_false_geometry_unchanged():
    trainer = _selfsup_trainer(hflip=False)
    v1, v2, _ = pack_pair_batch(_pairs(1))
    a1, a2 = trainer._augment_views(v1, v2)
    for key in ("img", "dilated_depth_m", "camera_intrinsics", "flow",
                "non_occluded_mask", "dilated_depth_window_mask"):
        if key in v1:
            assert torch.equal(a1[key], v1[key]), key
    # photometric-only (no geometry) still leaves flow/mask untouched
    t2 = _selfsup_trainer(hflip=False, photometric=True)
    b1, b2 = t2._augment_views(v1, v2)
    assert torch.equal(b1["flow"], v1["flow"])
    assert torch.equal(b1["non_occluded_mask"], v1["non_occluded_mask"])


def test_augment_views_hflip_true_flips_geometry_and_flow_dx():
    from uniflowmatch.training.rgbd_losses import hflip_flow
    trainer = _selfsup_trainer(hflip=True)
    v1, v2, _ = pack_pair_batch(_pairs(1))
    a1, a2 = trainer._augment_views(v1, v2)
    assert torch.equal(a1["img"], torch.flip(v1["img"], dims=[-1]))
    assert torch.allclose(a1["flow"], hflip_flow(v1["flow"]))


def test_selfsup_target_and_mask_follow_hflip_flag():
    import types
    from uniflowmatch.training.rgbd_losses import hflip_flow
    v1, v2, sup = pack_pair_batch(_pairs(2))
    flow = torch.randn_like(sup["flow"])

    for do_flip in (False, True):
        trainer = _selfsup_trainer(hflip=do_flip)
        captured = {}

        def rec(target, aug, mask, step, _c=captured):
            _c["target"] = target
            _c["mask"] = mask
            return aug.sum() * 0.0, {"selfsup_loss": 0.0, "selfsup_weight": 0.0}

        trainer.loss.selfsup_term = rec
        result = types.SimpleNamespace(flow=types.SimpleNamespace(flow_output=flow))
        trainer.compute_selfsup_loss(result, v1, v2, sup, global_step=1)
        expected = hflip_flow(flow) if do_flip else flow
        assert torch.allclose(captured["target"], expected)


# --------------------------------------------------------------------------- #
# autocast must consume the cfg train_precision
# --------------------------------------------------------------------------- #
def test_autocast_follows_train_precision():
    import contextlib

    def mkt(precision, device="cpu"):
        model, opt, _ = _trainer()
        cfg = {"model": {"train_precision": precision}, "optimizer": _opt_cfg(),
               "loss": _loss_cfg()}
        return RGBDTrainer(cfg, model=model, optimizer=opt, device=device)

    t32 = mkt("float32")
    assert t32._train_precision() == "float32"
    assert type(t32._autocast()).__name__ == "nullcontext"

    t16 = mkt("bf16")
    assert t16._train_precision() == "bf16"
    assert type(t16._autocast()).__name__ == "nullcontext"  # CPU stays float32

    if torch.cuda.is_available():
        tc = mkt("bf16", device="cuda")
        ctx = tc._autocast()
        assert type(ctx).__name__ != "nullcontext"
        with ctx:
            x = torch.randn(2, 2)
        assert x.dtype == torch.float32


# --------------------------------------------------------------------------- #
# unsupported runtime settings are rejected (never silently ignored)
# --------------------------------------------------------------------------- #
def test_run_rejects_unsupported_runtime(tmp_path):
    exp = {"name": "smoke", "max_updates": 1, "validation_max_pairs": 1}
    model, opt, trainer = _trainer()
    trainer.cfg["dataset"] = {"loader": {"batch_pairs": 2, "num_workers": 0}}
    with pytest.raises(TrainerError):
        trainer.run(_pairs(2), _pairs(2), experiment=exp, checkpoint_dir=tmp_path / "bp")

    model, opt, trainer = _trainer()
    trainer.cfg["dataset"] = {"loader": {"batch_pairs": 1, "num_workers": 4}}
    with pytest.raises(TrainerError):
        trainer.run(_pairs(2), _pairs(2), experiment=exp, checkpoint_dir=tmp_path / "nw")

    model, opt, trainer = _trainer()
    exp_acc = dict(exp, accumulate_grad_batches=2)
    with pytest.raises(TrainerError):
        trainer.run(_pairs(2), _pairs(2), experiment=exp_acc, checkpoint_dir=tmp_path / "ac")


# --------------------------------------------------------------------------- #
# FULL synthetic 2 epochs: all-population each epoch + full validation
# --------------------------------------------------------------------------- #
def test_run_full_synthetic_two_epochs_all_population(tmp_path):
    model, opt, trainer = _trainer()
    data = _pairs(4)
    exp = {"name": "full", "max_updates": None, "epochs": 2,
           "validate_full_each_epoch": True, "early_stop": False,
           "full_budget_fixed": True, "validation_max_pairs": None}
    res = trainer.run(data, _pairs(2), experiment=exp, checkpoint_dir=tmp_path / "ckpt")
    assert res["status"] == "finished" and res["global_step"] == 8
    assert res["train_population"] == 4
    assert res["counts"]["val_calls"] == 2  # one full validation per epoch
    assert sorted(res["index_sequence"][:4]) == [0, 1, 2, 3]
    assert sorted(res["index_sequence"][4:]) == [0, 1, 2, 3]
    assert (tmp_path / "ckpt" / "last_y.pt").is_file()


def test_run_receipt_written(tmp_path):
    model, opt, trainer = _trainer()
    exp = {"name": "short", "max_updates": 2, "validation_max_pairs": 2,
           "validate_every_updates": 2, "save_every_updates": 2, "early_stop": False}
    receipt = tmp_path / "run_receipt.json"
    res = trainer.run(_pairs(2), _pairs(2), experiment=exp,
                      checkpoint_dir=tmp_path / "ckpt", receipt_path=receipt)
    assert res["status"] == "finished"
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "finished" and payload["global_step"] == 2
    assert payload["population"]["train"] == 2
    assert payload["checkpoints"]["last_y"]["bytes"] > 0
    assert "identity" in payload and "timing_wall_s" in payload
