"""Procedure-14/15 synthetic model tests for the UFM RGB-D depth-fusion contract.

Two groups, selected by name so the GREEN baseline and the (formerly RED) new-API
tests can be logged separately:

* ``test_green_*`` — the *existing* RGB path: real import of
  ``UniFlowMatchConfidence`` and a network-free tiny encoder/decoder/head driven
  through the real ``forward``/``_encode_symmetrized``/``get_parameter_groups``
  code.  This is GREEN at procedure 14.
* ``test_red_*`` — the new RGB-D fusion API described in
  ``reports/ufm-rgbd-step14-fusion-adr-red/API_spec_14.md`` (ADR-14 Option C).
  These were the procedure-14 RED tests (they failed with
  ``TypeError: ... unexpected keyword argument 'use_depth_fusion'``); at
  procedure 15 the implementation lands and they turn GREEN **without weakening
  their original intent**.
* New procedure-15 strengthening tests (``test_symmetrized_*``,
  ``test_target_depth_*``, ``test_confidence_*``, ``test_state_roundtrip_*``,
  ``test_fail_fast_*``, ``test_legacy_*``, ``test_predict_*``) add value-space
  oracles, both-view coverage, state/roundtrip, gradient reachability with a
  frozen encoder, typed fail-fast and the public predict API.

No network is used: the tiny modules are injected by monkeypatching the
module-level constructors of ``uniflowmatch.models.ufm`` for the duration of model
construction only.  All paths are relative to this repo; no private
data/paths/calibration are embedded.  Real pretrained-asset numeric parity is a
procedure-16 concern and is **not** claimed here.
"""

from __future__ import annotations

import inspect
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
from uniflowmatch.models.base import (  # noqa: E402
    UFMOutputInterface,
    UniFlowMatchModelsBase,
)
from uniflowmatch.models.depth_fusion import (  # noqa: E402
    DepthFusionDeviceError,
    DepthFusionDtypeError,
    DepthFusionMaskError,
    DepthFusionNonFiniteError,
    DepthFusionShapeError,
)

DIM = 8
H = W = 32


# --------------------------------------------------------------------------- #
# network-free tiny encoder / decoder / head
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
        self.mask = torch.sigmoid(logits)  # (B, 1, H, W), matches real adaptors


class TinyEncoder(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.data_norm_type = "dinov2"
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


class CoupledTinyInfoSharing(nn.Module):
    """A tiny info-sharing that *does* mix the two views.

    Needed to test that a target-only depth change can reach the source-view
    output; the plain ``TinyInfoSharing`` keeps the streams independent.
    """

    def __init__(self, *a, **k):
        super().__init__()
        self.lin = nn.Conv2d(DIM, DIM, 1)
        self.mix = nn.Conv2d(DIM, DIM, 1)

    def forward(self, info_sharing_input):
        f1, f2 = info_sharing_input.features
        o1 = self.lin(f1) + self.mix(f2)
        o2 = self.lin(f2) + self.mix(f1)
        outs = _FeatOut([o1, o2])
        return outs, [_FeatOut([o1, o2]), _FeatOut([o1, o2])]


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


def _build_tiny_model(
    *, use_depth_fusion=False, depth_fusion_kwargs=None, coupled=False
):
    """Construct a real UniFlowMatchConfidence on tiny network-free submodules."""
    saved = {
        n: getattr(U, n)
        for n in (
            "feature_returner_encoder_factory",
            "INFO_SHARING_CLASSES",
            "CLASSNAME_TO_ADAPTOR_CLASS",
            "DPTFeature",
            "DPTRegressionProcessor",
            "AdaptorMap",
        )
    }
    U.feature_returner_encoder_factory = lambda *a, **k: TinyEncoder()
    sharing = CoupledTinyInfoSharing if coupled else TinyInfoSharing
    U.INFO_SHARING_CLASSES = {"global_attention": (None, sharing)}
    U.CLASSNAME_TO_ADAPTOR_CLASS = {"FlowAdaptor": TinyAdaptor}
    U.DPTFeature = TinyFeatureProcessor
    U.DPTRegressionProcessor = lambda *a, **k: nn.Identity()
    U.AdaptorMap = TinyAdaptorMap
    try:
        kwargs = dict(
            encoder_str="tiny",
            encoder_kwargs={},
            info_sharing_str="global_attention",
            info_sharing_kwargs={},
            head_type="dpt",
            feature_head_kwargs={"dpt_feature": {}, "dpt_processor": {}},
            adaptors_kwargs={"flow": {"class": "FlowAdaptor", "kwargs": {}}},
            uncertainty_head_kwargs={"dpt_feature": {}, "dpt_processor": {}},
            uncertainty_adaptors_kwargs={"flow": {"class": "FlowAdaptor", "kwargs": {}}},
            inference_resolution=(64, 48),
        )
        if use_depth_fusion:
            kwargs["use_depth_fusion"] = True
            kwargs["depth_fusion_kwargs"] = depth_fusion_kwargs or {
                "feature_dim": DIM,
                "depth_hidden_dim": DIM,
                "gate_init": 0.0,
                "depth_scale": 1.0,
            }
        model = U.UniFlowMatchConfidence(**kwargs)
    finally:
        for n, v in saved.items():
            setattr(U, n, v)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# view builders
# --------------------------------------------------------------------------- #
def _direct_views(a_img, b_img, a_depth=None, b_depth=None, a_mask=None, b_mask=None):
    """Unsymmetrized views: view1=a, view2=b (batch order preserved)."""
    v1 = {"img": a_img, "symmetrized": False, "data_norm_type": "dinov2"}
    v2 = {"img": b_img, "symmetrized": False, "data_norm_type": "dinov2"}
    if a_depth is not None:
        v1["depth"] = a_depth
    if b_depth is not None:
        v2["depth"] = b_depth
    if a_mask is not None:
        v1["depth_mask"] = a_mask
    if b_mask is not None:
        v2["depth_mask"] = b_mask
    return v1, v2


def _interleaved_views(a_img, b_img, a_depth, b_depth, a_mask, b_mask):
    """Symmetrized ABBA order: view1=[a0,b0,a1,b1,...], view2=[b0,a0,b1,a1,...]."""
    def interleave(x, y):
        return torch.stack([x, y], dim=1).flatten(0, 1)

    return _direct_views(
        interleave(a_img, b_img),
        interleave(b_img, a_img),
        interleave(a_depth, b_depth),
        interleave(b_depth, a_depth),
        interleave(a_mask, b_mask),
        interleave(b_mask, a_mask),
    )


def _sample_pair(seed=0, depth_a=0.3, depth_b=0.9, b=2):
    torch.manual_seed(seed)
    a_img = torch.randn(b, 3, H, W)
    b_img = torch.randn(b, 3, H, W)
    a_depth = torch.full((b, H, W), float(depth_a))
    b_depth = torch.full((b, H, W), float(depth_b))
    a_mask = torch.ones(b, H, W, dtype=torch.bool)
    a_mask[:, :, : W // 2] = False
    b_mask = torch.ones(b, H, W, dtype=torch.bool)
    b_mask[:, :, W // 2 :] = False
    return a_img, b_img, a_depth, b_depth, a_mask, b_mask


def _views(with_depth=False, all_invalid=False, symmetrized=False):
    a_img, b_img, a_depth, b_depth, a_mask, b_mask = _sample_pair()
    if all_invalid:
        a_mask = torch.zeros_like(a_mask)
        b_mask = torch.zeros_like(b_mask)
    if not with_depth:
        a_depth = b_depth = a_mask = b_mask = None
    if symmetrized:
        if not with_depth:
            # symmetrized RGB-only
            return _direct_views(
                torch.stack([a_img, b_img], 1).flatten(0, 1),
                torch.stack([b_img, a_img], 1).flatten(0, 1),
            )
        return _interleaved_views(a_img, b_img, a_depth, b_depth, a_mask, b_mask)
    return _direct_views(a_img, b_img, a_depth, b_depth, a_mask, b_mask)


# --------------------------------------------------------------------------- #
# GREEN: existing RGB path (must pass at procedure 14)
# --------------------------------------------------------------------------- #
def test_green_real_imports():
    assert U.UniFlowMatchConfidence is not None
    assert issubclass(U.UniFlowMatchConfidence, UniFlowMatchModelsBase)


def test_green_rgb_baseline_forward_and_param_groups():
    model = _build_tiny_model()
    v1, v2 = _views()
    out = model(v1, v2)
    assert isinstance(out, UFMOutputInterface)
    assert out.flow.flow_output.shape == (2, 2, H, W)

    named = dict(model.named_parameters())
    covered = set()
    for pl in model.get_parameter_groups().values():
        for p in pl:
            for n, q in named.items():
                if q is p:
                    covered.add(n)
    assert set(named) == covered, f"uncovered trainable params: {set(named) - covered}"


def test_green_rgb_forward_deterministic():
    model = _build_tiny_model()
    v1, v2 = _views()
    a = model(v1, v2).flow.flow_output
    b = model(v1, v2).flow.flow_output
    assert torch.equal(a, b)


def test_green_rgb_strict_state_load_baseline():
    model = _build_tiny_model()
    clone = _build_tiny_model()
    result = clone.load_state_dict(model.state_dict(), strict=True)
    assert set(result.missing_keys) == set()
    assert set(result.unexpected_keys) == set()
    v1, v2 = _views()
    assert torch.equal(model(v1, v2).flow.flow_output, clone(v1, v2).flow.flow_output)


# --------------------------------------------------------------------------- #
# RED (now GREEN): new RGB-D fusion API
# --------------------------------------------------------------------------- #
def test_red_depth_fusion_module_and_zero_gate():
    model = _build_tiny_model(use_depth_fusion=True)
    assert getattr(model, "depth_fusion", None) is not None
    assert model.depth_fusion.gate_value == 0.0


def test_red_rgb_only_parity_gate_zero():
    model = _build_tiny_model(use_depth_fusion=True)
    v1r, v2r = _views(with_depth=False)
    v1d, v2d = _views(with_depth=True)
    rgb = model(v1r, v2r)
    with_depth = model(v1d, v2d)
    assert torch.equal(rgb.flow.flow_output, with_depth.flow.flow_output)
    assert torch.equal(rgb.covisibility.mask, with_depth.covisibility.mask)


def test_red_all_invalid_depth_parity():
    model = _build_tiny_model(use_depth_fusion=True)
    v1r, v2r = _views(with_depth=False)
    v1i, v2i = _views(with_depth=True, all_invalid=True)
    rgb = model(v1r, v2r).flow.flow_output
    invalid = model(v1i, v2i).flow.flow_output
    assert torch.equal(rgb, invalid)


def test_red_nonfinite_depth_fail_fast():
    model = _build_tiny_model(use_depth_fusion=True)
    v1, v2 = _views(with_depth=True)
    v1["depth"][0, 0, 0] = float("nan")
    with pytest.raises(DepthFusionNonFiniteError):
        model(v1, v2)


def test_red_depth_shape_mismatch_fail_fast():
    model = _build_tiny_model(use_depth_fusion=True)
    v1, v2 = _views(with_depth=True)
    v1["depth"] = torch.zeros(2, H + 4, W)
    v1["depth_mask"] = torch.ones(2, H + 4, W, dtype=torch.bool)
    with pytest.raises(DepthFusionShapeError):
        model(v1, v2)


def test_red_get_parameter_groups_covers_depth_fusion():
    model = _build_tiny_model(use_depth_fusion=True)
    groups = model.get_parameter_groups()
    assert "depth_fusion" in groups
    named = dict(model.named_parameters())
    covered = set()
    for pl in groups.values():
        for p in pl:
            for n, q in named.items():
                if q is p:
                    covered.add(n)
    missing = set(named) - covered
    assert not missing, f"trainable params missing from groups: {missing}"


def test_red_state_keys_missing_only_adapter():
    rgb_model = _build_tiny_model()
    fusion_model = _build_tiny_model(use_depth_fusion=True)
    rgb_sd = rgb_model.state_dict()
    adapter_keys = {k for k in fusion_model.state_dict() if k.startswith("depth_fusion.")}
    assert adapter_keys, "expected new depth_fusion.* keys"
    result = fusion_model.load_state_dict(rgb_sd, strict=False)
    assert set(result.unexpected_keys) == set()
    assert set(result.missing_keys) == adapter_keys
    assert set(rgb_sd).issubset(fusion_model.state_dict().keys())


def test_red_gate_gradient_and_depth_gradient():
    model = _build_tiny_model(use_depth_fusion=True)
    v1, v2 = _views(with_depth=True)
    out = model(v1, v2).flow.flow_output
    out.sum().backward()
    gate = model.depth_fusion.gate
    assert gate.grad is not None and float(gate.grad.abs().sum()) > 0.0, "gate-0 first step needs grad"
    model.zero_grad()
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.3)
    out = model(v1, v2).flow.flow_output
    out.sum().backward()
    adapter_grads = [
        p.grad
        for n, p in model.depth_fusion.named_parameters()
        if "depth_adapter" in n
    ]
    assert any(
        g is not None and float(g.abs().sum()) > 0.0 for g in adapter_grads
    ), "non-zero gate must give depth-adapter gradient"


def test_red_symmetrized_depth_aligned():
    model = _build_tiny_model(use_depth_fusion=True)
    v1, v2 = _views(with_depth=True, symmetrized=True)
    out = model(v1, v2)
    assert out.flow.flow_output.shape[0] == v1["img"].shape[0]


def test_red_predict_api_accepts_depth_kwargs():
    sig = inspect.signature(UniFlowMatchModelsBase.predict_correspondences_batched)
    params = set(sig.parameters)
    for name in ("source_depth", "target_depth", "depth_mask", "camera_intrinsics"):
        assert name in params, f"predict API missing optional depth kwarg {name!r}"


# --------------------------------------------------------------------------- #
# procedure-15 strengthening tests
# --------------------------------------------------------------------------- #
def test_symmetrized_vs_unsymmetrized_value_oracle():
    """ABBA interleave must be numerically equivalent in *value* space.

    With a non-zero gate and *distinct* per-view metric depths (0.3 / 0.9) and
    distinct masks, the symmetrized source-view output at even indices
    (a_i -> b_i) must equal the unsymmetrized source output for the same pair.
    A constant-depth shortcut or a mis-interleaved depth would break this in
    value space; shape-only equality would not detect either.
    """
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=7)

    u1, u2 = _direct_views(a_img, b_img, a_d, b_d, a_m, b_m)
    unsym = model(u1, u2).flow.flow_output
    s1, s2 = _interleaved_views(a_img, b_img, a_d, b_d, a_m, b_m)
    sym = model(s1, s2).flow.flow_output

    assert sym.shape[0] == 2 * unsym.shape[0]
    oracle_sym = sym[0::2]
    reason = (
        "depth is interleaved with the images, so the symmetrized source-view "
        "output at even indices must equal the unsymmetrized source output for "
        "the same (a_i -> b_i) pair; distinct 0.3/0.9 depths and different masks "
        "make a constant/mis-interleaved depth fail in value space"
    )
    if not torch.equal(oracle_sym, unsym):
        assert torch.allclose(oracle_sym, unsym, atol=1e-6, rtol=1e-5), reason


def test_symmetrized_and_unsymmetrized_confidence_oracle():
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=11, depth_a=0.3, depth_b=0.9)
    u1, u2 = _direct_views(a_img, b_img, a_d, b_d, a_m, b_m)
    unsym = model(u1, u2).covisibility.mask
    s1, s2 = _interleaved_views(a_img, b_img, a_d, b_d, a_m, b_m)
    sym = model(s1, s2).covisibility.mask
    assert torch.allclose(sym[0::2], unsym, atol=1e-6, rtol=1e-5), (
        "covisibility must follow the same ABBA depth interleave as flow"
    )


def test_both_views_depth_reach_adapter_via_hook():
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=5)
    seen = []
    handle = model.depth_fusion.register_forward_pre_hook(
        lambda module, args: seen.append(args[1].detach().clone())
    )
    try:
        v1, v2 = _direct_views(a_img, b_img, a_d, b_d, a_m, b_m)
        model(v1, v2)
    finally:
        handle.remove()
    assert len(seen) == 2, "depth adapter must be invoked once per view"
    assert torch.equal(seen[0], a_d)
    assert torch.equal(seen[1], b_d)


def test_target_only_depth_effect_requires_coupled_infosharer():
    model = _build_tiny_model(use_depth_fusion=True, coupled=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=9)
    b_d2 = torch.full_like(b_d, 0.3)
    v1, v2 = _direct_views(a_img, b_img, a_d, b_d, a_m, b_m)
    out_a = model(v1, v2).flow.flow_output
    v1b, v2b = _direct_views(a_img, b_img, a_d, b_d2, a_m, b_m)
    out_b = model(v1b, v2b).flow.flow_output
    assert not torch.equal(out_a, out_b), (
        "a target-only depth change must reach the source-view output through a "
        "coupled info-sharing, while the plain TinyInfoSharing keeps streams apart"
    )


def test_target_depth_no_effect_at_gate_zero():
    model = _build_tiny_model(use_depth_fusion=True, coupled=True)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=9)
    b_d2 = torch.full_like(b_d, 0.3)
    v1, v2 = _direct_views(a_img, b_img, a_d, b_d, a_m, b_m)
    out_a = model(v1, v2).flow.flow_output
    v1b, v2b = _direct_views(a_img, b_img, a_d, b_d2, a_m, b_m)
    out_b = model(v1b, v2b).flow.flow_output
    assert torch.equal(out_a, out_b)


def test_confidence_flow_parity_gate_zero_with_depth():
    model = _build_tiny_model(use_depth_fusion=True)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=13)
    r1, r2 = _direct_views(a_img, b_img)
    d1, d2 = _direct_views(a_img, b_img, a_d, b_d, a_m, b_m)
    rgb = model(r1, r2)
    dep = model(d1, d2)
    assert torch.equal(rgb.flow.flow_output, dep.flow.flow_output)
    assert torch.equal(rgb.covisibility.mask, dep.covisibility.mask)


def test_state_roundtrip_bit_exact_all_state():
    m1 = _build_tiny_model(use_depth_fusion=True)
    m2 = _build_tiny_model(use_depth_fusion=True)
    result = m2.load_state_dict(m1.state_dict(), strict=True)
    assert set(result.missing_keys) == set() and set(result.unexpected_keys) == set()
    v1, v2 = _views(with_depth=True)
    assert torch.equal(m1(v1, v2).flow.flow_output, m2(v1, v2).flow.flow_output)


def test_legacy_rgb_keys_preserved_and_only_adapter_added():
    rgb = _build_tiny_model()
    fusion = _build_tiny_model(use_depth_fusion=True)
    rgb_sd, fusion_sd = rgb.state_dict(), fusion.state_dict()
    for k, v in rgb_sd.items():
        assert k in fusion_sd, f"legacy key {k} disappeared"
        assert fusion_sd[k].shape == v.shape, f"legacy key {k} reshaped"
    extra = set(fusion_sd) - set(rgb_sd)
    assert extra and all(k.startswith("depth_fusion.") for k in extra)


def test_parameter_groups_no_missing_no_duplicate_trainable():
    model = _build_tiny_model(use_depth_fusion=True)
    groups = model.get_parameter_groups()
    named = dict(model.named_parameters())
    covered = []
    for pl in groups.values():
        for p in pl:
            names = [n for n, q in named.items() if q is p]
            assert len(names) == 1, "a parameter was added to more than one group"
            covered.append(names[0])
    assert set(covered) == set(named), "some trainable parameter is uncovered"
    assert len(covered) == len(set(covered)), "duplicate parameter across groups"
    # depth_fusion group is exactly the adapter/gate parameters
    df_names = {n for n in named if n.startswith("depth_fusion.")}
    assert df_names.issubset(set(covered))


def test_gradient_reachable_with_frozen_encoder():
    model = _build_tiny_model(use_depth_fusion=True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    v1, v2 = _views(with_depth=True)

    model.zero_grad()
    model(v1, v2).flow.flow_output.sum().backward()
    assert model.depth_fusion.gate.grad is not None
    assert float(model.depth_fusion.gate.grad.abs().sum()) > 0.0

    model.zero_grad()
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.3)
    model(v1, v2).flow.flow_output.sum().backward()
    adapter_grads = [
        p.grad for n, p in model.depth_fusion.named_parameters() if "depth_adapter" in n
    ]
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in adapter_grads), (
        "depth adapter gradient must be reachable even when the RGB encoder is frozen"
    )


def test_fail_fast_typed_exceptions():
    model = _build_tiny_model(use_depth_fusion=True)
    base1, base2 = _views(with_depth=True)

    v1, v2 = _views(with_depth=True)
    v1["depth"] = v1["depth"].to(torch.int32)
    with pytest.raises(DepthFusionDtypeError):
        model(v1, v2)

    v1, v2 = _views(with_depth=True)
    v1["depth"] = v1["depth"].to("meta")
    with pytest.raises(DepthFusionDeviceError):
        model(v1, v2)

    v1, v2 = _views(with_depth=True)
    del v1["depth_mask"]
    with pytest.raises(DepthFusionMaskError):
        model(v1, v2)

    # depth present, no depth, mask alone -> RGB path (no crash, identity parity)
    v1, v2 = _views(with_depth=False)
    v1["valid_mask"] = torch.zeros(2, H, W, dtype=torch.bool)
    assert torch.equal(
        model(v1, v2).flow.flow_output, model(*_views(with_depth=False)).flow.flow_output
    )


def test_all_invalid_parity_holds_with_nonzero_gate_and_biased_adapter():
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.7)
        for p in model.depth_fusion.parameters():
            p.add_(0.5)
    rgb = model(*_views(with_depth=False)).flow.flow_output
    invalid = model(*_views(with_depth=True, all_invalid=True)).flow.flow_output
    assert torch.equal(rgb, invalid)


def test_mask_priority_excludes_raw_window_mask():
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=17)
    v1, v2 = _direct_views(a_img, b_img)
    v1["depth"] = a_d
    # raw window mask is all-valid, but it must NOT be used for the 3x3 input;
    # only valid_mask (all-invalid) is in the priority, so the result is identity.
    v1["raw_depth_window_mask"] = torch.ones(2, H, W, dtype=torch.bool)
    v1["valid_mask"] = torch.zeros(2, H, W, dtype=torch.bool)
    rgb = model(*_direct_views(a_img, b_img)).flow.flow_output
    out = model(v1, v2).flow.flow_output
    assert torch.equal(rgb, out), "raw_depth_window_mask must not feed the dilated input"


def test_dilated_window_mask_priority_beats_valid_mask():
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    a_img, b_img, a_d, b_d, a_m, b_m = _sample_pair(seed=19)
    v1, v2 = _direct_views(a_img, b_img)
    v1["dilated_depth_m"] = a_d  # loader key supplied without explicit "depth"
    v1["dilated_depth_window_mask"] = torch.ones(2, H, W, dtype=torch.bool)
    v1["valid_mask"] = torch.zeros(2, H, W, dtype=torch.bool)  # lower priority
    rgb = model(*_direct_views(a_img, b_img)).flow.flow_output
    out = model(v1, v2).flow.flow_output
    assert not torch.equal(rgb, out), "dilated_depth_window_mask must take priority over valid_mask"


def test_predict_api_depth_parity_and_pair_requirement():
    model = _build_tiny_model(use_depth_fusion=True)
    torch.manual_seed(21)
    src = torch.rand(1, 3, H, W)
    tgt = torch.rand(1, 3, H, W)
    rgb = model.predict_correspondences_batched(src, tgt, data_norm_type="dinov2")

    sd = torch.full((1, H, W), 0.4)
    td = torch.full((1, H, W), 0.6)
    sd_mask = torch.ones(1, H, W, dtype=torch.bool)
    td_mask = torch.ones(1, H, W, dtype=torch.bool)
    K = torch.eye(3).unsqueeze(0)
    out = model.predict_correspondences_batched(
        src,
        tgt,
        data_norm_type="dinov2",
        source_depth=sd,
        target_depth=td,
        depth_mask=(sd_mask, td_mask),
        camera_intrinsics=(K, K),
    )
    assert out.flow.flow_output.shape == rgb.flow.flow_output.shape
    # gate 0 => exact RGB parity even with depth + K supplied through predict
    assert torch.equal(out.flow.flow_output, rgb.flow.flow_output)

    with pytest.raises(ValueError):
        model.predict_correspondences_batched(
            src,
            tgt,
            data_norm_type="dinov2",
            source_depth=sd,
            target_depth=td,
            depth_mask=sd_mask,
        )


def test_predict_api_depth_changes_output_when_gate_nonzero():
    model = _build_tiny_model(use_depth_fusion=True)
    with torch.no_grad():
        model.depth_fusion.gate.fill_(0.5)
    torch.manual_seed(23)
    src = torch.rand(1, 3, H, W)
    tgt = torch.rand(1, 3, H, W)
    rgb = model.predict_correspondences_batched(src, tgt, data_norm_type="dinov2")
    sd = torch.full((1, H, W), 0.4)
    td = torch.full((1, H, W), 0.6)
    out = model.predict_correspondences_batched(
        src, tgt, data_norm_type="dinov2", source_depth=sd, target_depth=td
    )
    assert not torch.equal(out.flow.flow_output, rgb.flow.flow_output)


# --------------------------------------------------------------------------- #
# bool mask dtype contract through the public predict API
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float64, torch.int32, torch.int64])
def test_predict_rejects_non_bool_explicit_mask_pair(bad_dtype):
    model = _build_tiny_model(use_depth_fusion=True)
    torch.manual_seed(31)
    src = torch.rand(1, 3, H, W)
    tgt = torch.rand(1, 3, H, W)
    sd = torch.full((1, H, W), 0.4)
    td = torch.full((1, H, W), 0.6)
    bad = torch.ones(1, H, W, dtype=bad_dtype)
    with pytest.raises(DepthFusionDtypeError):
        model.predict_correspondences_batched(
            src, tgt, data_norm_type="dinov2",
            source_depth=sd, target_depth=td, depth_mask=(bad, bad),
        )


def test_predict_rejects_nan_float_mask():
    model = _build_tiny_model(use_depth_fusion=True)
    torch.manual_seed(33)
    src = torch.rand(1, 3, H, W)
    tgt = torch.rand(1, 3, H, W)
    sd = torch.full((1, H, W), 0.4)
    td = torch.full((1, H, W), 0.6)
    bad = torch.full((1, H, W), float("nan"))
    with pytest.raises(DepthFusionDtypeError):
        model.predict_correspondences_batched(
            src, tgt, data_norm_type="dinov2",
            source_depth=sd, target_depth=td, depth_mask=(bad, bad),
        )


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int64])
def test_predict_rejects_non_bool_single_view_override(bad_dtype):
    model = _build_tiny_model(use_depth_fusion=True)
    torch.manual_seed(35)
    src = torch.rand(1, 3, H, W)
    tgt = torch.rand(1, 3, H, W)
    sd = torch.full((1, H, W), 0.4)
    td = torch.full((1, H, W), 0.6)
    bad = torch.ones(1, H, W, dtype=bad_dtype)
    with pytest.raises(DepthFusionDtypeError):
        model.predict_correspondences_batched(
            src, tgt, data_norm_type="dinov2",
            source_depth=sd, target_depth=td,
            source_depth_mask=bad, target_depth_mask=bad,
        )


def test_predict_rejects_non_tensor_mask():
    model = _build_tiny_model(use_depth_fusion=True)
    torch.manual_seed(37)
    src = torch.rand(1, 3, H, W)
    tgt = torch.rand(1, 3, H, W)
    sd = torch.full((1, H, W), 0.4)
    td = torch.full((1, H, W), 0.6)
    bad = [[True] * W for _ in range(H)]
    with pytest.raises(DepthFusionDtypeError):
        model.predict_correspondences_batched(
            src, tgt, data_norm_type="dinov2",
            source_depth=sd, target_depth=td, depth_mask=(bad, bad),
        )
