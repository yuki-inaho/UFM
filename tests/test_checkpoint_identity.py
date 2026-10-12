"""Synthetic tests for the checkpoint identity / RGB->RGB-D migration helpers.

These are network-free and embed no private data, calibration, credentials or
absolute host paths.  They exercise:

* file identity (bytes / SHA256),
* safe loading (``weights_only``) and format rejection,
* exact key/shape/finiteness validation,
* the released-checkpoint *alias* case (shared parameters stored under one name),
* the RGB->RGB-D migration delta (exactly the new module's keys) and zero-gate
  preservation,
* a save -> reload strict roundtrip with a forward comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uniflowmatch.models.checkpoint import (  # noqa: E402
    CheckpointFormatError,
    CheckpointIdentityError,
    CheckpointMigrationError,
    CheckpointNonFiniteError,
    CheckpointStateError,
    alias_groups,
    check_state_finite,
    classify_missing,
    expected_new_keys,
    extract_state_dict,
    load_checkpoint,
    load_released_state,
    migrate_rgb_to_rgbd,
    sha256_file,
    state_dict_diff,
    strict_load_state_dict,
    verify_file_identity,
)
from uniflowmatch.models.depth_fusion import DepthFusion  # noqa: E402

C = 4
FH = FW = 6


class _Evil:
    """A module-level, non-allowlisted class used to test safe loading."""

    pass


# --------------------------------------------------------------------------- #
# tiny synthetic models (with an aliased parameter pair, like the real DPT head)
# --------------------------------------------------------------------------- #
class _TinyRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, C)
        # the same Conv2d object referenced twice => shared storage, like the
        # real DPT `scratch.layer*_rn` / `input_process.*.1` alias.
        conv = nn.Conv2d(3, C, kernel_size=1)
        self.conv = conv
        self.alias = conv

    def forward(self, x):
        return self.conv(x)


class _TinyRGBD(_TinyRGB):
    def __init__(self):
        super().__init__()
        self.depth_fusion = DepthFusion(C, C, gate_init=0.0, depth_scale=1.0)

    def forward(self, x, depth=None, mask=None):
        feat = self.conv(x)
        if depth is not None:
            return self.depth_fusion(feat, depth, mask, image_hw=(int(x.shape[-2]), int(x.shape[-1])))
        return feat


def _seeded_rgb(seed=0):
    torch.manual_seed(seed)
    m = _TinyRGB()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return m


def _dedup_state(model):
    """Drop alias duplicates (keep the canonical name), like the released file."""
    groups = alias_groups(model.state_dict())
    drop = set()
    for keys in groups.values():
        # keep the canonical (non-"alias") name, drop the alias like the release
        canonical = [k for k in keys if not k.startswith("alias.")]
        keep = canonical[0] if canonical else keys[0]
        drop.update(k for k in keys if k != keep)
    return {k: v for k, v in model.state_dict().items() if k not in drop}


# --------------------------------------------------------------------------- #
# file identity
# --------------------------------------------------------------------------- #
def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"uniflowmatch")
    import hashlib

    assert sha256_file(str(p)) == hashlib.sha256(b"uniflowmatch").hexdigest()


def test_verify_file_identity_ok(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    info = verify_file_identity(str(p), expected_sha256=sha256_file(str(p)), expected_bytes=3)
    assert info["bytes"] == 3


def test_verify_file_identity_rejects_wrong_sha(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    with pytest.raises(CheckpointIdentityError):
        verify_file_identity(str(p), expected_sha256="0" * 64)


def test_verify_file_identity_rejects_truncated(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    with pytest.raises(CheckpointIdentityError):
        verify_file_identity(str(p), expected_bytes=999)


def test_verify_file_identity_rejects_missing(tmp_path):
    with pytest.raises(CheckpointIdentityError):
        verify_file_identity(str(tmp_path / "nope.bin"))


# --------------------------------------------------------------------------- #
# safe loading / format
# --------------------------------------------------------------------------- #
def test_extract_state_dict_accepts_variants():
    sd = {"a": torch.zeros(1)}
    assert extract_state_dict(sd) == sd
    assert extract_state_dict({"state_dict": sd}) == sd
    assert extract_state_dict({"model": sd}) == sd


def test_extract_state_dict_rejects_garbage():
    with pytest.raises(CheckpointFormatError):
        extract_state_dict({"foo": 1, "bar": [2]})


def test_load_checkpoint_weights_only_rejects_unsafe_pickle(tmp_path):
    p = tmp_path / "evil.pt"
    torch.save({"_evil": _Evil()}, str(p))
    with pytest.raises(CheckpointFormatError):
        load_checkpoint(str(p), weights_only=True)


def test_load_checkpoint_reads_plain_state_dict(tmp_path):
    p = tmp_path / "ok.pt"
    torch.save({"a": torch.ones(2)}, str(p))
    loaded = extract_state_dict(load_checkpoint(str(p), weights_only=True))
    assert torch.equal(loaded["a"], torch.ones(2))


def test_load_checkpoint_rejects_truncated_file(tmp_path):
    p = tmp_path / "trunc.pt"
    torch.save({"a": torch.ones(4)}, str(p))
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(CheckpointFormatError):
        load_checkpoint(str(p), weights_only=True)


# --------------------------------------------------------------------------- #
# diff / finiteness
# --------------------------------------------------------------------------- #
def test_state_dict_diff_reports_all_three():
    model = {"a": torch.zeros(2), "b": torch.zeros(3)}
    loaded = {"a": torch.zeros(2), "b": torch.zeros(9), "c": torch.zeros(1)}
    missing, unexpected, mismatch = state_dict_diff(model, loaded)
    assert missing == [] and unexpected == ["c"]
    assert mismatch == [("b", (3,), (9,))]


def test_state_dict_diff_missing():
    missing, unexpected, mismatch = state_dict_diff({"a": torch.zeros(1)}, {})
    assert missing == ["a"] and unexpected == [] and mismatch == []


def test_check_state_finite_rejects_nan():
    with pytest.raises(CheckpointNonFiniteError):
        check_state_finite({"a": torch.tensor([1.0, float("nan")])})


def test_check_state_finite_rejects_non_fp32():
    with pytest.raises(CheckpointNonFiniteError):
        check_state_finite({"a": torch.zeros(2, dtype=torch.float64)})


def test_check_state_finite_allows_int_buffers():
    check_state_finite({"n": torch.zeros(2, dtype=torch.long)})


# --------------------------------------------------------------------------- #
# strict load (rejects the mixin's silent missing behaviour)
# --------------------------------------------------------------------------- #
def test_strict_load_state_dict_exact():
    m = _seeded_rgb()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    m2 = _TinyRGB()
    report = strict_load_state_dict(m2, sd)
    assert report["unexpected"] == []
    for k in sd:
        assert torch.equal(m2.state_dict()[k], sd[k])


def test_strict_load_rejects_genuine_missing():
    m = _seeded_rgb()
    sd = {k: v for k, v in m.state_dict().items() if not k.endswith("stem.weight")}
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(_TinyRGB(), sd)


def test_strict_load_rejects_unexpected():
    sd = dict(_seeded_rgb().state_dict())
    sd["bogus"] = torch.zeros(1)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(_TinyRGB(), sd)


def test_strict_load_rejects_shape_mismatch():
    sd = dict(_seeded_rgb().state_dict())
    sd["stem.weight"] = torch.zeros(3, 2)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(_TinyRGB(), sd)


def test_strict_load_rejects_nonfinite():
    sd = dict(_seeded_rgb().state_dict())
    sd["stem.bias"] = torch.full_like(sd["stem.bias"], float("inf"))
    with pytest.raises(CheckpointNonFiniteError):
        strict_load_state_dict(_TinyRGB(), sd)


# --------------------------------------------------------------------------- #
# alias handling
# --------------------------------------------------------------------------- #
def test_alias_groups_detects_shared_storage():
    groups = alias_groups(_seeded_rgb().state_dict())
    merged = [sorted(v) for v in groups.values()]
    assert ["alias.weight", "conv.weight"] in merged


def test_classify_missing_benign_alias():
    m = _seeded_rgb()
    dedup = _dedup_state(m)
    benign, genuine = classify_missing(m, dedup)
    assert genuine == []
    assert set(benign) == {"alias.weight", "alias.bias"}
    assert benign["alias.weight"] == ["conv.weight"]
    assert benign["alias.bias"] == ["conv.bias"]


def test_classify_missing_genuine():
    m = _seeded_rgb()
    sd = {k: v for k, v in m.state_dict().items() if not k.endswith("stem.weight")}
    benign, genuine = classify_missing(m, sd)
    assert genuine == ["stem.weight"]


def test_load_released_state_accepts_alias_and_fills_values():
    m = _seeded_rgb()
    dedup = _dedup_state(m)
    target = _TinyRGB()
    report = load_released_state(target, dedup)
    assert set(report["benign_aliases"]) == {"alias.weight", "alias.bias"}
    # both alias names carry the canonical value
    assert torch.equal(target.state_dict()["alias.weight"], target.state_dict()["conv.weight"])
    assert torch.equal(target.state_dict()["alias.bias"], target.state_dict()["conv.bias"])


def test_load_released_state_rejects_genuine_missing():
    m = _seeded_rgb()
    dedup = _dedup_state(m)
    del dedup["stem.weight"]
    with pytest.raises(CheckpointStateError):
        load_released_state(_TinyRGB(), dedup)


def test_load_released_state_rejects_unexpected():
    dedup = _dedup_state(_seeded_rgb())
    dedup["ghost"] = torch.zeros(1)
    with pytest.raises(CheckpointStateError):
        load_released_state(_TinyRGB(), dedup)


# --------------------------------------------------------------------------- #
# RGB -> RGB-D migration
# --------------------------------------------------------------------------- #
def test_expected_new_keys_is_exactly_depth_fusion():
    keys = expected_new_keys(_TinyRGBD(), "depth_fusion")
    assert keys == [
        "depth_fusion.depth_adapter.0.bias",
        "depth_fusion.depth_adapter.0.weight",
        "depth_fusion.depth_adapter.2.bias",
        "depth_fusion.depth_adapter.2.weight",
        "depth_fusion.gate",
    ]


def test_migrate_materialised_state_has_exactly_five_genuine_missing():
    rgb = _seeded_rgb()
    rgb_state = {k: v.clone() for k, v in rgb.state_dict().items()}
    rgbd = _TinyRGBD()
    report = migrate_rgb_to_rgbd(rgbd, rgb_state)
    assert report["genuine_missing"] == expected_new_keys(rgbd, "depth_fusion")
    assert len(report["genuine_missing"]) == 5
    assert report["unexpected"] == []
    assert report["shape_mismatch"] == []
    assert report["gate_value"] == 0.0


def test_migrate_raw_dedup_state_resolves_aliases():
    rgb = _seeded_rgb()
    dedup = _dedup_state(rgb)
    rgbd = _TinyRGBD()
    report = migrate_rgb_to_rgbd(rgbd, dedup)
    assert report["genuine_missing"] == expected_new_keys(rgbd, "depth_fusion")
    assert set(report["benign_aliases"]) == {"alias.weight", "alias.bias"}


def test_migrate_preserves_legacy_values():
    rgb = _seeded_rgb()
    rgb_state = {k: v.clone() for k, v in rgb.state_dict().items()}
    rgbd = _TinyRGBD()
    migrate_rgb_to_rgbd(rgbd, rgb_state)
    for k, v in rgb_state.items():
        assert torch.equal(rgbd.state_dict()[k], v)


def test_migrate_preserves_zero_gate():
    rgb_state = _seeded_rgb().state_dict()
    rgbd = _TinyRGBD()
    migrate_rgb_to_rgbd(rgbd, rgb_state)
    assert float(rgbd.depth_fusion.gate) == 0.0


def test_migrate_rejects_incomplete_rgb_state():
    rgb_state = {k: v for k, v in _seeded_rgb().state_dict().items() if not k.endswith("stem.weight")}
    with pytest.raises(CheckpointMigrationError):
        migrate_rgb_to_rgbd(_TinyRGBD(), rgb_state)


def test_migrate_rejects_unexpected_key():
    rgb_state = dict(_seeded_rgb().state_dict())
    rgb_state["ghost"] = torch.zeros(1)
    with pytest.raises(CheckpointMigrationError):
        migrate_rgb_to_rgbd(_TinyRGBD(), rgb_state)


def test_migrate_rejects_shape_mismatch():
    rgb_state = dict(_seeded_rgb().state_dict())
    rgb_state["stem.weight"] = torch.zeros(1, 1)
    with pytest.raises(CheckpointMigrationError):
        migrate_rgb_to_rgbd(_TinyRGBD(), rgb_state)


def test_migrate_rejects_nonfinite_legacy():
    rgb_state = dict(_seeded_rgb().state_dict())
    rgb_state["stem.bias"] = torch.full_like(rgb_state["stem.bias"], float("nan"))
    with pytest.raises(CheckpointNonFiniteError):
        migrate_rgb_to_rgbd(_TinyRGBD(), rgb_state)


# --------------------------------------------------------------------------- #
# save -> reload strict roundtrip + forward comparison
# --------------------------------------------------------------------------- #
def test_save_reload_strict_roundtrip_and_forward(tmp_path):
    from safetensors.torch import save_file

    rgb_state = _seeded_rgb().state_dict()
    rgbd = _TinyRGBD()
    migrate_rgb_to_rgbd(rgbd, rgb_state)
    with torch.no_grad():  # make the new module non-trivial for the forward check
        rgbd.depth_fusion.gate.fill_(0.7)
        for p in rgbd.depth_fusion.depth_adapter.parameters():
            p.add_(0.3)
    rgbd.eval()

    path = tmp_path / "rgbd.safetensors"
    save_file(_dedup_state(rgbd), str(path))

    reloaded = _TinyRGBD()
    loaded = load_checkpoint(str(path))
    load_released_state(reloaded, loaded)
    reloaded.eval()

    x = torch.randn(1, 3, FH, FW)
    d = torch.rand(1, FH, FW)
    m = torch.ones(1, FH, FW, dtype=torch.bool)
    with torch.no_grad():
        a = rgbd(x, d, m)
        b = reloaded(x, d, m)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# NEW (16B): typed dtype model + contradictory-alias / dtype-mismatch fixtures
# --------------------------------------------------------------------------- #
class _TinyTyped(nn.Module):
    """A tiny model with a float parameter and a genuine integer buffer."""

    def __init__(self, param_dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(C, dtype=param_dtype))
        self.bias = nn.Parameter(torch.randn(C, dtype=param_dtype))
        self.register_buffer("counter", torch.zeros(3, dtype=torch.int64))

    def forward(self, x):
        return x + self.weight.sum() + self.bias.sum()


def _contradictory_alias_state(model):
    """Full state where the alias name carries a *different* value than canonical."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["alias.weight"] = sd["conv.weight"] + 1.0
    sd["alias.bias"] = sd["conv.bias"] + 1.0
    return sd


def _clone_state(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def _assert_unchanged(module, before):
    for k, v in before.items():
        assert torch.equal(module.state_dict()[k], v), f"{k} mutated after rejection"


# --- contradictory shared-alias values (reject before any copy) ------------- #
def test_strict_load_rejects_contradictory_alias():
    sd = _contradictory_alias_state(_seeded_rgb())
    target = _TinyRGB()
    before = _clone_state(target)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(target, sd)
    _assert_unchanged(target, before)


def test_load_released_state_rejects_contradictory_alias():
    sd = _contradictory_alias_state(_seeded_rgb())
    target = _TinyRGB()
    before = _clone_state(target)
    with pytest.raises(CheckpointStateError):
        load_released_state(target, sd)
    _assert_unchanged(target, before)


def test_migrate_rejects_contradictory_alias():
    sd = _contradictory_alias_state(_seeded_rgb())
    rgbd = _TinyRGBD()
    before = _clone_state(rgbd)
    with pytest.raises(CheckpointStateError):
        migrate_rgb_to_rgbd(rgbd, sd)
    _assert_unchanged(rgbd, before)


# --- dtype mismatch on float parameters ------------------------------------- #
@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.int32, torch.float64, torch.bool])
def test_strict_load_rejects_param_dtype_mismatch(bad_dtype):
    sd = _clone_state(_TinyTyped())
    sd["weight"] = torch.zeros(C, dtype=bad_dtype)
    target = _TinyTyped()
    before = _clone_state(target)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(target, sd)
    _assert_unchanged(target, before)


# --- dtype mismatch on integer buffers -------------------------------------- #
@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float64, torch.int32, torch.bool])
def test_strict_load_rejects_buffer_dtype_mismatch(bad_dtype):
    sd = _clone_state(_TinyTyped())
    sd["counter"] = torch.zeros(3, dtype=bad_dtype)
    target = _TinyTyped()
    before = _clone_state(target)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(target, sd)
    _assert_unchanged(target, before)


def test_load_released_state_rejects_dtype_mismatch():
    dedup = {k: v.clone() for k, v in _dedup_state(_TinyRGB()).items()}
    dedup["stem.weight"] = dedup["stem.weight"].to(torch.float64)
    target = _TinyRGB()
    before = _clone_state(target)
    with pytest.raises(CheckpointStateError):
        load_released_state(target, dedup)
    _assert_unchanged(target, before)


def test_migrate_rejects_dtype_mismatch():
    rgb = _seeded_rgb()
    sd = {k: v.clone() for k, v in rgb.state_dict().items()}
    sd["stem.weight"] = sd["stem.weight"].to(torch.int64)
    rgbd = _TinyRGBD()
    before = _clone_state(rgbd)
    with pytest.raises(CheckpointStateError):
        migrate_rgb_to_rgbd(rgbd, sd)
    _assert_unchanged(rgbd, before)


# --- GREEN: valid cases that must keep passing ------------------------------ #
def test_strict_load_allows_matching_int_buffer():
    sd = _clone_state(_TinyTyped())
    sd["counter"] = torch.arange(3, dtype=torch.int64)
    target = _TinyTyped()
    strict_load_state_dict(target, sd)
    assert target.counter.dtype == torch.int64
    assert torch.equal(target.counter, torch.arange(3, dtype=torch.int64))


def test_strict_load_allows_shared_aliases_with_same_values():
    m = _seeded_rgb()
    sd = _clone_state(m)
    target = _TinyRGB()
    strict_load_state_dict(target, sd)
    assert torch.equal(target.state_dict()["alias.weight"], target.state_dict()["conv.weight"])


def test_strict_load_fp16_match_allowed_when_not_requiring_fp32():
    sd = _clone_state(_TinyTyped(param_dtype=torch.float16))
    target = _TinyTyped(param_dtype=torch.float16)
    strict_load_state_dict(target, sd, require_fp32=False)
    assert target.weight.dtype == torch.float16


def test_strict_load_fp16_model_rejects_fp32_state_even_without_require_fp32():
    sd = _clone_state(_TinyTyped(param_dtype=torch.float32))
    target = _TinyTyped(param_dtype=torch.float16)
    with pytest.raises(CheckpointStateError):
        strict_load_state_dict(target, sd, require_fp32=False)


def test_alias_groups_distinguishes_dtype_sharing_storage():
    f = torch.zeros(C, dtype=torch.float32)
    i = f.view(torch.int32)  # same storage, different dtype
    assert f.data_ptr() == i.data_ptr()
    assert alias_groups({"f": f, "i": i}) == {}
