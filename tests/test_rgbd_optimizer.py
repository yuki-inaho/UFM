"""Synthetic tests for the procedure-17B AMUSE optimizer wrapper.

These tests are network-free, GPU-free and public: they embed no private dataset
identity, counts, absolute host path or credential.  They exercise the full RGB-D
optimizer contract on tiny mixed modules:

* parameter-plan coverage (every trainable parameter exactly once, no frozen
  parameters), shared-alias de-duplication by object identity and typed rejection
  of missing / duplicate / unknown modules,
* AMUSE construction with the *real* post-constructor (size-sorted) parameter
  order persisted into the checkpoint identity,
* one finite update including the zero-gate -> non-zero-gate gradient progression,
* train/eval mode switching with restore-even-on-exception semantics,
* uninterrupted vs fresh-resume numeric identity for the training ``y`` and the
  averaged evaluation ``x`` state (CPU round-off tolerance 1e-6, stated here),
* restoring the AMUSE instance hyper-parameters saved in extra state from a
  *different* constructor value to the saved one,
* strict-load rejection of a same-shape tensor-name swap / shape / dtype /
  config-identity mismatch before any state copy,
* fail-fast on a non-finite late-group gradient leaving every parameter
  unchanged, and post-step parameter finiteness checking,
* warmup 0 and non-matrix muon rejection,
* a basic AdamW comparison optimizer state.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uniflowmatch.training.rgbd_optimizer import (  # noqa: E402
    AMUSEControlledOptimizer,
    AMUSEModeError,
    DuplicateParameterError,
    MissingParameterError,
    MuonShapeError,
    NonFiniteGradientError,
    NonFiniteParameterError,
    StateDtypeMismatch,
    StateIdentityMismatch,
    StateNameOrderMismatch,
    StateShapeMismatch,
    UnknownModuleError,
    build_optimizer,
    build_parameter_plan,
    vendor_amuse_sha256,
)

C = 4

# Public upstream SHA-256 of the pinned AMUSE source (also recorded in NOTICE).
AMUSE_UPSTREAM_SHA256 = "84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f"


# --------------------------------------------------------------------------- #
# tiny mixed model
# --------------------------------------------------------------------------- #
class _TinyFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adapter = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, depth):
        return x + self.gate * self.adapter(depth)


class _TinyPlanModel(nn.Module):
    """Tiny UFM-like model exposing ``get_parameter_groups``.

    The encoder is frozen; info-sharing / depth-fusion host hidden 2D/conv4D
    weights (Muon candidates) plus embedding / norm / 3D / bias / gate fallbacks
    (aux SGD).  ``shared`` is deliberately referenced by two groups so its
    identity is de-duplicated.
    """

    def __init__(self, dim=C, big=False):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)

        self.info_sharing = nn.Module()
        self.info_sharing.proj = nn.Conv2d(dim, dim, 1)
        self.info_sharing.a = nn.Conv2d(dim, dim, 1)
        self.info_sharing.b = nn.Conv2d(dim, dim, 1)
        if big:
            self.info_sharing.big = nn.Conv2d(dim, 2 * dim, 3)
        self.info_sharing.embedding = nn.Parameter(torch.randn(dim, dim))
        self.info_sharing.norm = nn.LayerNorm(dim)
        self.info_sharing.scale3d = nn.Parameter(torch.randn(dim, 1, 1))

        self.output_head = nn.Linear(dim, dim)
        self.uncertainty_head = nn.Linear(dim, 1)
        self.depth_fusion = _TinyFusion(dim)
        self.shared = nn.Parameter(torch.randn(dim, dim))

    def get_parameter_groups(self):
        return {
            "encoder": nn.ParameterList(self.encoder.parameters()),
            "info_sharing": nn.ParameterList(self.info_sharing.parameters()),
            "output_head": nn.ParameterList(list(self.output_head.parameters()) + [self.shared]),
            "uncertainty_head": nn.ParameterList(self.uncertainty_head.parameters()),
            "depth_fusion": nn.ParameterList(list(self.depth_fusion.parameters()) + [self.shared]),
        }

    def forward(self, x):
        b = x.shape[0]
        f = self.encoder(x)
        g = f.view(b, C, 1, 1)
        g = self.info_sharing.proj(g) + self.info_sharing.a(g) + self.info_sharing.b(g)
        if hasattr(self.info_sharing, "big"):
            big = self.info_sharing.big(x.view(b, C, 1, 1))
            g = g + big.mean(dim=(2, 3)).view(b, C, 1, 1)
        g = g.mean(dim=(2, 3))
        g = g + x @ self.info_sharing.embedding.t()
        g = self.info_sharing.norm(g)
        g = g + self.info_sharing.scale3d.view(1, C)
        main = self.output_head(g)
        unc = self.uncertainty_head(g)
        depth = x.view(b, C, 1, 1)
        fused = main.view(b, C, 1, 1) + self.depth_fusion.gate * self.depth_fusion.adapter(depth)
        return fused.mean() + unc.mean()


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return _TinyPlanModel(**kw)


def _cfg(name="amuse", **over):
    cfg = {
        "name": name,
        "schedule_free": name == "amuse",
        "external_scheduler": None if name == "amuse" else "cosine",
        "lr": 2e-2,
        "sgd_lr": 2e-2,
        "weight_decay": 0.0,
        "constructor_args": {
            "beta1": 0.9,
            "weight_lr_power": 2.0,
            "warmup_steps": 2,
            "rho": 1.0,
            "r": 0.0,
            "weight_decay_at_y": 0.0,
        },
        "groups": {
            "muon": {
                "use_muon": True,
                "lr": 2e-2,
                "momentum": 0.95,
                "aux_update_type": "sgd",
                "weight_decay": 0.0,
            },
            "aux": {"use_muon": False, "update_type": "sgd", "lr": 2e-2, "weight_decay": 0.0},
        },
    }
    cfg.update(over)
    return {"optimizer": cfg}


def _full_cfg(**over):
    cfg = _cfg(**over)
    cfg["model"] = {
        "param_group_policy": {
            "frozen": ["encoder"],
            "muon_hidden_matrix": {
                "modules": ["info_sharing", "depth_fusion"],
                "exclude_name_patterns": [
                    "embedding",
                    "output_head",
                    "uncertainty_head",
                    "bias",
                    "norm",
                    "gate",
                ],
            },
            "aux_fallback": {"modules": ["output_head", "uncertainty_head"]},
        }
    }
    return cfg


def _step(opt, model, x, zero=True):
    if zero:
        model.zero_grad(set_to_none=True)
    loss = model(x)
    loss.backward()
    opt.step()
    return float(loss)


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _assert_same(a, b, atol=1e-6):
    assert set(a) == set(b)
    for n in a:
        assert torch.allclose(a[n], b[n], atol=atol, rtol=0.0), f"mismatch at {n}"


# --------------------------------------------------------------------------- #
# vendoring
# --------------------------------------------------------------------------- #
def test_vendor_amuse_sha_matches_upstream():
    assert vendor_amuse_sha256() == AMUSE_UPSTREAM_SHA256
    notice = (_REPO / "uniflowmatch/training/vendor/amuse/NOTICE").read_text()
    assert AMUSE_UPSTREAM_SHA256 in notice
    assert "Apache" in notice
    assert "48922743b32f33f919ab54edde3dbad0d0ce2dc7" in notice


# --------------------------------------------------------------------------- #
# plan coverage / classification
# --------------------------------------------------------------------------- #
def test_plan_covers_every_trainable_exactly_once_no_frozen():
    model = _model()
    plan = build_parameter_plan(model)
    named = dict(model.named_parameters())
    muon_names = {n for n, _ in plan.muon}
    aux_names = {n for n, _ in plan.aux}

    assert not (muon_names & aux_names), "a parameter appears in both groups"
    # encoder is frozen by policy even though its parameters still require grad
    assert set(plan.frozen) == {"encoder.weight", "encoder.bias"}
    trainable = {
        n for n, p in named.items() if p.requires_grad and n not in set(plan.frozen)
    }
    assert muon_names | aux_names == trainable
    assert len(muon_names) + len(aux_names) == len(trainable)

    # hidden conv4D weights from info_sharing + depth_fusion go to Muon
    assert "info_sharing.proj.weight" in muon_names
    assert "info_sharing.a.weight" in muon_names
    assert "info_sharing.b.weight" in muon_names
    assert "depth_fusion.adapter.weight" in muon_names
    # embedding (name-excluded), 3D, bias, norm, gate fall back to aux
    assert "info_sharing.embedding" in aux_names
    assert "info_sharing.scale3d" in aux_names
    assert "info_sharing.proj.bias" in aux_names
    assert "info_sharing.norm.weight" in aux_names
    assert "depth_fusion.gate" in aux_names
    # encoder frozen
    assert "encoder.weight" in plan.frozen and "encoder.weight" not in muon_names | aux_names


def test_optimizer_contains_only_trainable_and_no_frozen():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    plan_ids = {id(p) for _, p in opt.plan.muon} | {id(p) for _, p in opt.plan.aux}
    assert opt_ids == plan_ids
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    assert not (opt_ids & encoder_ids), "frozen encoder leaked into the optimizer"
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_shared_alias_deduped_by_identity():
    model = _model()
    plan = build_parameter_plan(model)
    all_names = [n for n, _ in plan.muon] + [n for n, _ in plan.aux]
    assert all_names.count("shared") == 1, "aliased parameter must appear exactly once"
    assert plan.aliases, "the shared alias must be reported"


def test_typed_reject_missing_module():
    model = _model()

    orig = model.get_parameter_groups

    def bad_groups():
        g = orig()
        g.pop("info_sharing")
        return g

    model.get_parameter_groups = bad_groups
    with pytest.raises(MissingParameterError):
        build_parameter_plan(model)


def test_typed_reject_duplicate_within_group():
    model = _model()
    orig = model.get_parameter_groups
    base = orig()

    def bad_groups():
        g = dict(base)
        plist = list(g["info_sharing"])
        g["info_sharing"] = nn.ParameterList(plist + [plist[0]])
        return g

    model.get_parameter_groups = bad_groups
    with pytest.raises(DuplicateParameterError):
        build_parameter_plan(model)


def test_typed_reject_unknown_module():
    model = _model()
    orig = model.get_parameter_groups
    base = orig()

    def bad_groups():
        g = dict(base)
        g["mystery"] = nn.ParameterList([nn.Parameter(torch.zeros(1))])
        return g

    model.get_parameter_groups = bad_groups
    with pytest.raises(UnknownModuleError):
        build_parameter_plan(model)


def test_reject_non_matrix_muon():
    plan = build_parameter_plan(_model())
    plan.muon.append(("forced_1d", nn.Parameter(torch.zeros(C))))
    with pytest.raises(MuonShapeError):
        from uniflowmatch.training.rgbd_optimizer import assert_muon_matrix

        assert_muon_matrix(plan)


# --------------------------------------------------------------------------- #
# constructor ordering
# --------------------------------------------------------------------------- #
def test_constructor_uses_real_post_init_sorted_order():
    model = _model(big=True)
    opt = build_optimizer(model, _full_cfg())
    named = dict(model.named_parameters())
    name_by_id = {id(p): n for n, p in named.items()}

    assert opt.ordered_names, "ordered_names must not be empty"
    # ordered_names is built from the real optimizer order
    real_order = [name_by_id[id(p)] for g in opt.optimizer.param_groups for p in g["params"]]
    assert opt.ordered_names == real_order

    # within the Muon group the official constructor sorts by size descending
    muon_group = next(g for g in opt.optimizer.param_groups if g.get("use_muon"))
    sizes = [p.numel() for p in muon_group["params"]]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == model.info_sharing.big.weight.numel()
    # pre-init name order is NOT used: big weight is not first in named_parameters
    assert list(named).index("info_sharing.big.weight") > 0


# --------------------------------------------------------------------------- #
# update / gate progression
# --------------------------------------------------------------------------- #
def test_one_update_finite_and_gate_grad_progression():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt.train()
    torch.manual_seed(1)
    x = torch.randn(3, C)

    # first step: gate is exactly zero -> depth adapter weight receives no grad
    assert float(model.depth_fusion.gate.detach()) == 0.0
    model.zero_grad(set_to_none=True)
    model(x).backward()
    adapter_grad_0 = model.depth_fusion.adapter.weight.grad
    assert adapter_grad_0 is not None and float(adapter_grad_0.abs().sum()) == 0.0
    assert model.depth_fusion.gate.grad is not None
    assert float(model.depth_fusion.gate.grad.abs().sum()) > 0.0
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert float(model.depth_fusion.gate.detach()) != 0.0

    # later step: non-zero gate -> depth adapter weight now receives gradient
    model.zero_grad(set_to_none=True)
    model(x).backward()
    adapter_grad_1 = model.depth_fusion.adapter.weight.grad
    assert adapter_grad_1 is not None and float(adapter_grad_1.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# mode contract
# --------------------------------------------------------------------------- #
def test_initial_train_mode_false_and_step_requires_train():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    assert opt.optimizer.train_mode is False
    x = torch.randn(2, C)
    model.zero_grad(set_to_none=True)
    model(x).backward()
    with pytest.raises(AMUSEModeError):
        opt.step()


def test_eval_mode_context_restores_and_restores_on_exception():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt.train()
    assert opt.optimizer.train_mode is True
    with opt.eval_mode():
        assert opt.optimizer.train_mode is False
    assert opt.optimizer.train_mode is True

    with pytest.raises(RuntimeError):
        with opt.eval_mode():
            assert opt.optimizer.train_mode is False
            raise RuntimeError("boom")
    assert opt.optimizer.train_mode is True


def test_mode_metadata_distinguishes_y_and_x():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt.train()
    sd_train = opt.state_dict()
    assert sd_train["amuse_extra"]["mode"]["param_state"] == "y"
    assert sd_train["amuse_extra"]["mode"]["train_mode"] is True

    opt.eval()
    sd_eval = opt.state_dict()
    assert sd_eval["amuse_extra"]["mode"]["param_state"] == "x"
    assert sd_eval["amuse_extra"]["mode"]["train_mode"] is False


# --------------------------------------------------------------------------- #
# numeric resume
# --------------------------------------------------------------------------- #
def test_uninterrupted_vs_fresh_resume_identical():
    x = torch.randn(4, C)
    model_a = _model(seed=0)
    opt_a = build_optimizer(model_a, _full_cfg())
    opt_a.train()
    for _ in range(6):
        _step(opt_a, model_a, x)
    final_a = _snapshot(model_a)

    model_b = _model(seed=0)
    opt_b = build_optimizer(model_b, _full_cfg())
    opt_b.train()
    for _ in range(3):
        _step(opt_b, model_b, x)
    sd3 = opt_b.state_dict()
    params_at_3 = _snapshot(model_b)

    # fresh resume on a freshly built optimizer (constructed from a different
    # instance) after restoring the step-3 parameter values
    model_c = _model(seed=0)
    with torch.no_grad():
        for n, p in model_c.named_parameters():
            p.copy_(params_at_3[n])
    opt_c = build_optimizer(model_c, _full_cfg())
    opt_c.load_state_dict(sd3)
    opt_c.train()
    for _ in range(3):
        _step(opt_c, model_c, x)
    final_c = _snapshot(model_c)

    # CPU determinism: identical parameters / states; allow 1e-6 round-off.
    _assert_same(final_a, final_c, atol=1e-6)

    # averaged evaluation x-state is also identical after a mode switch
    opt_a.eval()
    opt_c.eval()
    _assert_same(_snapshot(model_a), _snapshot(model_c), atol=1e-6)


def test_extra_attrs_restored_from_different_constructor_value():
    model = _model()
    opt_saved = build_optimizer(model, _full_cfg())
    opt_saved.train()
    x = torch.randn(2, C)
    _step(opt_saved, model, x)
    sd = opt_saved.state_dict()

    other = _full_cfg()
    other["optimizer"]["constructor_args"] = {
        "beta1": 0.5,
        "weight_lr_power": 1.0,
        "warmup_steps": 5,
        "rho": 0.3,
        "r": 0.2,
        "weight_decay_at_y": 0.1,
    }
    opt_new = build_optimizer(_model(seed=7), other)
    assert opt_new.optimizer.beta1_init == 0.5
    opt_new.load_state_dict(sd)
    assert opt_new.optimizer.beta1_init == 0.9
    assert opt_new.optimizer.weight_lr_power == 2.0
    assert opt_new.optimizer.warmup_steps == 2
    assert opt_new.optimizer.rho == 1.0
    assert opt_new.optimizer.r == 0.0
    assert opt_new.optimizer.weight_decay_at_y == 0.0


# --------------------------------------------------------------------------- #
# strict load
# --------------------------------------------------------------------------- #
def test_same_shape_name_swap_rejected_and_params_unchanged():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt.train()
    x = torch.randn(2, C)
    _step(opt, model, x)
    sd = opt.state_dict()

    # swap two equal-shaped Muon names in the saved order: must reject
    names = list(sd["amuse_extra"]["ordered_names"])
    swapped = list(names)
    idx = [
        i
        for i, n in enumerate(swapped)
        if n.endswith("weight") and n.startswith("info_sharing")
    ]
    assert len(idx) >= 2
    assert len({tuple(opt._param_by_name[names[k]].shape) for k in idx[:2]}) == 1
    i, j = idx[0], idx[1]
    swapped[i], swapped[j] = swapped[j], swapped[i]
    sd["amuse_extra"]["ordered_names"] = swapped

    fresh_model = _model(seed=9)
    fresh = build_optimizer(fresh_model, _full_cfg())
    before = _snapshot(fresh_model)
    with pytest.raises(StateNameOrderMismatch):
        fresh.load_state_dict(sd)
    # rejected before any state copy: the target model parameters are unchanged
    after = _snapshot(fresh_model)
    for n in before:
        assert torch.equal(before[n], after[n]), f"parameter {n} changed on rejected load"


def test_config_identity_mismatch_rejected():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    sd = opt.state_dict()
    sd["amuse_extra"]["config_identity"]["sha"] = "deadbeef"
    fresh = build_optimizer(_model(seed=3), _full_cfg())
    with pytest.raises(StateIdentityMismatch):
        fresh.load_state_dict(sd)


def test_shape_and_dtype_mismatch_rejected():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    sd = opt.state_dict()

    sd_shape = {k: v for k, v in sd.items()}
    import copy as _copy

    sd_shape["amuse_extra"] = _copy.deepcopy(sd["amuse_extra"])
    nm = list(sd_shape["amuse_extra"]["config_identity"]["shapes"])[0]
    sd_shape["amuse_extra"]["config_identity"]["shapes"][nm] = [999]
    with pytest.raises(StateShapeMismatch):
        build_optimizer(_model(seed=4), _full_cfg()).load_state_dict(sd_shape)

    sd_dtype = _copy.deepcopy(sd)
    sd_dtype["amuse_extra"]["config_identity"]["dtypes"][nm] = "torch.float16"
    with pytest.raises(StateDtypeMismatch):
        build_optimizer(_model(seed=5), _full_cfg()).load_state_dict(sd_dtype)


# --------------------------------------------------------------------------- #
# non-finite fail-fast
# --------------------------------------------------------------------------- #
def test_nonfinite_late_group_grad_leaves_all_params_unchanged():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt.train()
    x = torch.randn(2, C)
    model.zero_grad(set_to_none=True)
    model(x).backward()

    before = _snapshot(model)
    # corrupt the last (aux) group's gradient with a non-finite value
    model.uncertainty_head.weight.grad[0, 0] = float("inf")
    with pytest.raises(NonFiniteGradientError):
        opt.step()

    after = _snapshot(model)
    for n in before:
        assert torch.equal(before[n], after[n]), f"parameter {n} changed on failed step"


def test_post_step_nonfinite_parameter_fails_loudly():
    model = _model()
    opt = build_optimizer(model, _full_cfg())
    opt.train()
    x = torch.randn(2, C)
    model.zero_grad(set_to_none=True)
    model(x).backward()
    with torch.no_grad():
        model.uncertainty_head.bias.fill_(float("inf"))
    with pytest.raises(NonFiniteParameterError):
        opt.step()


# --------------------------------------------------------------------------- #
# config rejects
# --------------------------------------------------------------------------- #
def test_reject_warmup_zero():
    cfg = _full_cfg()
    cfg["optimizer"]["constructor_args"]["warmup_steps"] = 0
    with pytest.raises(ValueError):
        build_optimizer(_model(), cfg)


# --------------------------------------------------------------------------- #
# 17A config schema alignment (public, no Hydra needed)
# --------------------------------------------------------------------------- #
def test_optimizer_config_schema_aligns_with_17a():
    from uniflowmatch.training import rgbd_config

    schema = rgbd_config.SCHEMA["optimizer"]
    for k in (
        "name",
        "schedule_free",
        "external_scheduler",
        "lr",
        "sgd_lr",
        "weight_decay",
        "constructor_args",
        "groups",
        "train_eval",
        "gradient_clip",
    ):
        assert k in schema, f"optimizer schema missing {k!r}"
    for k in ("beta1", "weight_lr_power", "warmup_steps", "rho", "r", "weight_decay_at_y"):
        assert k in schema["constructor_args"], f"constructor_args missing {k!r}"
    for k in ("use_muon", "lr", "momentum", "aux_update_type", "weight_decay"):
        assert k in schema["groups"]["muon"], f"muon group missing {k!r}"
    for k in ("use_muon", "update_type", "lr", "weight_decay"):
        assert k in schema["groups"]["aux"], f"aux group missing {k!r}"


def test_public_amuse_preset_is_default_schedule_free():
    import yaml

    doc = yaml.safe_load(
        (_REPO / "configs/rgbd_training/optimizer/amuse.yaml").read_text()
    )
    assert doc["name"] == "amuse"
    assert doc["default"] is True
    assert doc["schedule_free"] is True
    assert doc["external_scheduler"] is None
    assert doc["constructor_args"]["warmup_steps"] > 0
    assert doc["groups"]["muon"]["use_muon"] is True
    assert doc["groups"]["aux"]["update_type"] == "sgd"


# --------------------------------------------------------------------------- #
# AdamW comparison
# --------------------------------------------------------------------------- #
def test_adamw_comparison_basic_state():
    model = _model()
    cfg = _cfg(name="adamw")
    opt = build_optimizer(model, cfg)
    assert opt.name == "adamw"
    opt.train()
    x = torch.randn(2, C)
    _step(opt, model, x)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    sd = opt.state_dict()
    fresh = build_optimizer(_model(seed=2), _cfg(name="adamw"))
    fresh.load_state_dict(sd)
    assert fresh.ordered_names == opt.ordered_names
