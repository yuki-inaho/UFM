"""CPU dependency-injection tests for ``scripts/evaluate_rgbd.py`` (step 21B1).

These tests drive the **real CLI path** (``main`` -> ``run`` -> real
``RGBDPairDataset`` loader -> ``pack_pair_batch`` -> metric core -> persistence)
with tiny **synthetic, generic** staged inputs and an injected fake model.  No
private dataset identifier, id, path, count, calibration or result is embedded:
every synthetic value below is a made-up constant.  No GPU work is performed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_rgbd", str(REPO / "scripts" / "evaluate_rgbd.py")
)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["evaluate_rgbd"] = mod
_SPEC.loader.exec_module(mod)

H, W = 6, 8


# --------------------------------------------------------------------------- #
# synthetic staged index (generic names only)
# --------------------------------------------------------------------------- #
def _write_png(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.ascontiguousarray(arr))


def _frame(root: Path, fid: int):
    d = root / "frames" / "capture" / f"{fid:08d}"
    rng = np.random.default_rng(fid)
    rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    depth = rng.integers(100, 1200, size=(H, W)).astype(np.uint16)
    dil = depth.copy()
    _write_png(d / "rgb.jpg", rgb[:, :, ::-1])
    _write_png(d / "raw_depth_mm.png", depth)
    _write_png(d / "dilated3_depth_mm.png", dil)
    _write_png(d / "raw_depth_mask.png", (depth > 0).astype(np.uint8) * 255)
    _write_png(d / "dilated3_depth_mask.png", (dil > 0).astype(np.uint8) * 255)
    _write_png(d / "depth_provenance.png", np.ones((H, W), np.uint8))
    return d


def _direction(root: Path, src: int, dst: int, tag: str, *, flow_scale: float,
               empty_covis: bool, occ_empty: bool):
    d = root / "teacher" / f"pair_{src:03d}_{dst:03d}"
    d.mkdir(parents=True, exist_ok=True)
    flow = np.full((2, H, W), flow_scale, dtype=np.float32)
    with (d / f"{tag}.npz").open("wb") as fh:
        np.savez(fh, flow=flow)
    valid = np.zeros((H, W), np.uint8) if occ_empty else np.full((H, W), 255, np.uint8)
    covis = np.zeros((H, W), np.uint8) if empty_covis else np.full((H, W), 255, np.uint8)
    _write_png(d / f"{tag}_valid.png", valid)
    _write_png(d / f"{tag}_covis.png", covis)
    _write_png(d / f"{tag}_occlusion.png", valid)
    return {
        "flow_path": f"teacher/pair_{src:03d}_{dst:03d}/{tag}.npz",
        "valid_path": f"teacher/pair_{src:03d}_{dst:03d}/{tag}_valid.png",
        "covis_path": f"teacher/pair_{src:03d}_{dst:03d}/{tag}_covis.png",
        "reference_hw": [H, W],
        "occlusion_supervision_path": f"teacher/pair_{src:03d}_{dst:03d}/{tag}_occlusion.png",
    }


def make_staging(root: Path, pairs, *, ref_hw=(H, W), empty_covis_pair=None,
                 occ_empty_pair=None, split="val"):
    root.mkdir(parents=True, exist_ok=True)
    K = [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]
    idx_pairs = []
    for src, dst in pairs:
        _frame(root, src)
        _frame(root, dst)
        empty_covis = empty_covis_pair == (src, dst)
        occ_empty = occ_empty_pair == (src, dst)
        fwd = _direction(root, src, dst, "fwd", flow_scale=1.0,
                         empty_covis=empty_covis, occ_empty=occ_empty)
        bwd = _direction(root, src, dst, "bwd", flow_scale=-1.0,
                         empty_covis=empty_covis, occ_empty=occ_empty)
        views = []
        for fid in (src, dst):
            views.append({
                "frame_index": fid,
                "rgb_path": f"frames/capture/{fid:08d}/rgb.jpg",
                "raw_depth_mm_path": f"frames/capture/{fid:08d}/raw_depth_mm.png",
                "dilated3_depth_mm_path": f"frames/capture/{fid:08d}/dilated3_depth_mm.png",
                "raw_depth_mask_path": f"frames/capture/{fid:08d}/raw_depth_mask.png",
                "dilated3_depth_mask_path": f"frames/capture/{fid:08d}/dilated3_depth_mask.png",
                "depth_provenance_path": f"frames/capture/{fid:08d}/depth_provenance.png",
                "K": K, "K_known": True,
            })
        idx_pairs.append({
            "pair_id": f"cap_{src:07d}_{dst:07d}",
            "capture": "capture",
            "split": split,
            "src_frame_index": src,
            "dst_frame_index": dst,
            "views": views,
            "flow_fwd": fwd,
            "flow_bwd": bwd,
            "sha256": {},
            "is_synthetic": True,
        })
    index = {
        "schema_version": 1,
        "dataset_name": "synthetic",
        "already_dilated": True,
        "mm_to_m": 0.001,
        "depth_units": "millimetres",
        "depth_window_m": [0.0, 1.3],
        "native_hw": [H, W],
        "teacher_reference_hw": list(ref_hw),
        "occlusion_supervision_source": "path",
        "pairs": idx_pairs,
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root / "index.json"


def make_descriptor(path: Path, index_path: Path, *, role="val",
                    split_note=None, depth_units="metre", variants_extra=None,
                    monitor=None):
    desc = {
        "role": role,
        "dataset": {
            "index_path": str(index_path),
            "resolution": [W, H],
            "data_norm_type": "dinov2",
            "transform": "imgnorm",
            "depth_window_m": [0.0, 1.3],
            "depth_units": depth_units,
            "mm_to_m": 0.001,
            "already_dilated": True,
            "occlusion_supervision_source": "path",
            "verify_hashes": False,
        },
        "assets": {
            "asset_path": "/nonexistent/asset",
            "asset_sha256": "0" * 64,
            "config_path": "/nonexistent/config",
            "config_sha256": "0" * 64,
            "dino_hub_dir": "/nonexistent/hub",
            "dino_python_tree_sha256": "0" * 64,
        },
        "model": {"depth_fusion_kwargs": {"feature_dim": 4, "depth_hidden_dim": 2,
                                          "gate_init": 0.0, "depth_scale": 1.3}},
        "variants": {
            "rgb_pretrained": {"kind": "rgb"},
            "rgbd_pretrained_init": {"kind": "rgbd_init"},
            "rgbd_off": {"kind": "checkpoint", "checkpoint_path": "/nonexistent/off.pt"},
            "rgbd_on": {"kind": "checkpoint", "checkpoint_path": "/nonexistent/on.pt"},
        },
        "determinism": {
            "enabled": True, "warn_only": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True, "allow_tf32": False,
            "cublas_workspace_config": ":4096:8", "sdpa_backend": "math",
            "bilinear": "deterministic",
        },
    }
    if monitor is not None:
        desc["monitor"] = monitor
    if variants_extra:
        desc["variants"].update(variants_extra)
    path.write_text(json.dumps(desc), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# fake model
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, flow, logits):
        self.flow = type("F", (), {"flow_output": flow})()
        self.covisibility = type("C", (), {"logits": logits})()


class _FakeModel(torch.nn.Module):
    """Deterministic fake: flow depends only on RGB so fusion is a no-op."""

    def __init__(self, hw, *, bias=0.0, nan=False, ignores_depth=True):
        super().__init__()
        self.register_buffer("_dummy", torch.zeros(1))
        self.hw = hw
        self.bias = bias
        self.nan = nan

    def forward(self, view1, view2):
        img = view1["img"].float()
        b = img.shape[0]
        h, w = self.hw
        base = img.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        flow = base + self.bias
        assert flow.shape == (b, 2, h, w), flow.shape
        logits = torch.zeros(b, 1, h, w)
        if self.nan:
            flow = flow.clone()
            flow[0, :, 0, 0] = float("nan")
        return _Out(flow, logits)


def _patch_models(monkeypatch, hw=(H, W), nan_variant=None):
    def factory(name, desc, device):
        if name == nan_variant:
            return _FakeModel(hw, bias=1.0, nan=True)
        bias = {"rgb_pretrained": 0.0, "rgbd_pretrained_init": 0.0,
                "rgbd_off": 2.0, "rgbd_on": 3.0}.get(name, 0.0)
        return _FakeModel(hw, bias=bias)
    monkeypatch.setattr(mod, "_build_variant_model", factory)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_all_four_variants_all_pairs_abba_and_persistence(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    pairs = [(0, 1), (1, 2), (2, 3)]
    index = make_staging(root, pairs)
    desc = make_descriptor(tmp_path / "desc.json", index)
    out = tmp_path / "out"
    _patch_models(monkeypatch)

    rc = mod.main([
        "--descriptor", str(desc), "--dataset-index", str(index),
        "--split", "val", "--output-dir", str(out),
        "--variants", ",".join(mod.VARIANTS), "--device", "cpu",
    ])
    assert rc == 0
    comparison = json.loads((out / "comparison.json").read_text())
    assert set(comparison["variants"]) == set(mod.VARIANTS)
    assert len(comparison["common_pair_ids"]) == len(pairs)
    # per-pair NPZ, both directions, for every variant
    for name in mod.VARIANTS:
        npzs = sorted((out / name / "pairs").glob("*.npz"))
        assert len(npzs) == len(pairs)
        with np.load(npzs[0], allow_pickle=True) as z:
            assert z["pred_flow"].shape == (2, 2, H, W)
            assert list(z["direction"]) == ["fwd", "bwd"]
            assert z["input_sha"].shape == ()
    # events log stages
    events = (out / "events.jsonl").read_text().strip().splitlines()
    stages = {json.loads(e)["stage"] for e in events}
    for name in mod.VARIANTS:
        assert f"eval_{name}" in stages
    assert (out / "run_receipt.json").is_file()
    # gate-0 parity (rgb vs rgbd_init share the same function)
    assert comparison["rgb_rgbd_init_parity"]["checked"] is True
    assert comparison["rgb_rgbd_init_parity"]["ok"] is True


def test_rgb_rgbd_init_parity_fails_when_fusion_changes_output(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    desc = make_descriptor(tmp_path / "desc.json", index)
    def factory(name, d, device):
        return _FakeModel((H, W), bias=(0.0 if name != "rgbd_pretrained_init" else 5.0))
    monkeypatch.setattr(mod, "_build_variant_model", factory)
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained,rgbd_pretrained_init",
                   "--device", "cpu"])
    assert rc == 0
    comparison = json.loads((out / "comparison.json").read_text())
    assert comparison["rgb_rgbd_init_parity"]["ok"] is False


def test_nonfinite_prediction_retained_as_failure(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1), (1, 2)])
    desc = make_descriptor(tmp_path / "desc.json", index)
    _patch_models(monkeypatch, nan_variant="rgbd_on")
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgbd_on", "--device", "cpu"])
    assert rc == 0
    comparison = json.loads((out / "comparison.json").read_text())
    v = comparison["variants"]["rgbd_on"]
    assert v["full_population"]["n_failure_pairs"] >= 1
    assert v["full_population"]["total_nonfinite_pred"] >= 1


def test_empty_covis_support_is_no_data(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1), (1, 2)], empty_covis_pair=(1, 2))
    desc = make_descriptor(tmp_path / "desc.json", index)
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained,rgbd_on", "--device", "cpu"])
    assert rc == 0
    comparison = json.loads((out / "comparison.json").read_text())
    assert comparison["variants"]["rgb_pretrained"]["full_population"]["n_missing"] >= 1
    # the empty pair is excluded from the common normal population
    assert len(comparison["common_pair_ids"]) == 1


def test_test_role_selection_rejected(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)], split="test")
    desc = make_descriptor(tmp_path / "desc.json", index, role="test")
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "test", "--output-dir", str(out),
                   "--variants", "rgb_pretrained", "--device", "cpu", "--select"])
    assert rc == 2


def test_val_role_selection_allowed(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    desc = make_descriptor(tmp_path / "desc.json", index, role="val")
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained,rgbd_on", "--device", "cpu", "--select"])
    assert rc == 0
    receipt = json.loads((out / "run_receipt.json").read_text())
    assert receipt["selection"] == "rgb_pretrained"


def test_in_repo_output_rejected(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    desc = make_descriptor(tmp_path / "desc.json", index)
    _patch_models(monkeypatch)
    in_repo = REPO / "temp" / "should_not_be_written_eval"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(in_repo),
                   "--variants", "rgb_pretrained", "--device", "cpu"])
    assert rc == 2
    assert not in_repo.exists()


def test_checkpoint_kind_and_sha_rejected(tmp_path):
    from uniflowmatch.training.rgbd_checkpoint import ResumeKindError
    from uniflowmatch.models.checkpoint import CheckpointIdentityError

    ckpt = tmp_path / "last_y.pt"
    torch.save({"format": "ufm_rgbd_training_checkpoint", "format_version": 1,
                "kind": "last_y", "model": {}}, ckpt)
    with pytest.raises(ResumeKindError):
        mod._load_checkpoint_state(ckpt)
    # a wrong SHA is rejected before the kind/state is even read
    with pytest.raises(CheckpointIdentityError):
        mod._load_checkpoint_state(ckpt, expected_sha256="0" * 64)


def test_depth_units_and_resize_rejected(tmp_path):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    bad_units = make_descriptor(tmp_path / "du.json", index, depth_units="mm")
    with pytest.raises(mod.EvaluationRunError):
        mod.validate_descriptor(json.loads(bad_units.read_text()))
    # resize: reference grid != model target grid
    root2 = tmp_path / "stage2"
    index2 = make_staging(root2, [(0, 1)], ref_hw=(H + 2, W))
    desc2 = make_descriptor(tmp_path / "d2.json", index2)
    with pytest.raises(mod.EvaluationRunError):
        mod.check_grid(json.loads(desc2.read_text()),
                       json.loads(index2.read_text()))


def test_population_mismatch_rejected():
    with pytest.raises(mod.EvaluationRunError):
        mod._check_population({"a": ["p", "q"], "b": ["q", "p"]})
    mod._check_population({"a": ["p", "q"], "b": ["p", "q"]})


# --------------------------------------------------------------------------- #
# determinism is APPLIED to the whole run (not merely recorded)
# --------------------------------------------------------------------------- #
def test_determinism_context_applied_to_whole_run(tmp_path, monkeypatch):
    import os

    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    desc = make_descriptor(tmp_path / "desc.json", index)
    out = tmp_path / "out"

    captured = {}

    class _Probe(_FakeModel):
        def forward(self, view1, view2):
            captured.setdefault("det", torch.are_deterministic_algorithms_enabled())
            captured.setdefault("warn_only", torch.is_deterministic_algorithms_warn_only_enabled())
            captured.setdefault("tf32_matmul", torch.backends.cuda.matmul.allow_tf32)
            captured.setdefault("tf32_cudnn", torch.backends.cudnn.allow_tf32)
            captured.setdefault("cudnn_benchmark", torch.backends.cudnn.benchmark)
            captured.setdefault("cudnn_deterministic", torch.backends.cudnn.deterministic)
            captured.setdefault("cublas", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
            return super().forward(view1, view2)

    monkeypatch.setattr(mod, "_build_variant_model", lambda n, d, dev: _Probe((H, W)))

    # pre-existing state that must be restored after the run
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.benchmark = True
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained", "--device", "cpu"])
    assert rc == 0
    # the flags were actually set while the run executed
    assert captured["det"] is True
    assert captured["warn_only"] is False
    assert captured["tf32_matmul"] is False
    assert captured["tf32_cudnn"] is False
    assert captured["cudnn_benchmark"] is False
    assert captured["cudnn_deterministic"] is True
    assert captured["cublas"] == ":4096:8"
    # ... and restored on exit
    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.backends.cudnn.benchmark is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    # the applied settings are also recorded truthfully
    receipt = json.loads((out / "run_receipt.json").read_text())
    assert receipt["determinism_applied"]["cublas_workspace_config"] == ":4096:8"
    assert receipt["determinism_applied"]["enabled"] is True


# --------------------------------------------------------------------------- #
# selection guard: role/split mismatch + real-role keying
# --------------------------------------------------------------------------- #
def test_role_split_mismatch_rejected_before_heavy_work(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)], split="val")
    # descriptor says test, CLI says val -> rejected
    desc = make_descriptor(tmp_path / "desc.json", index, role="test")
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained", "--device", "cpu"])
    assert rc == 2
    assert not (out / "comparison.json").exists()

    # descriptor says val, CLI says test -> also rejected
    root2 = tmp_path / "stage2"
    index2 = make_staging(root2, [(0, 1)], split="test")
    desc2 = make_descriptor(tmp_path / "d2.json", index2, role="val")
    out2 = tmp_path / "out2"
    rc = mod.main(["--descriptor", str(desc2), "--dataset-index", str(index2),
                   "--split", "test", "--output-dir", str(out2),
                   "--variants", "rgb_pretrained", "--device", "cpu"])
    assert rc == 2


def test_role_test_split_val_select_bypass_rejected(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)], split="val")
    desc = make_descriptor(tmp_path / "desc.json", index, role="test")
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    rc = mod.main(["--descriptor", str(desc), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained", "--device", "cpu", "--select"])
    assert rc == 2
    assert not (out / "run_receipt.json").exists()


# --------------------------------------------------------------------------- #
# dataset index identity (declared path + expected SHA)
# --------------------------------------------------------------------------- #
def test_index_identity_mismatch_rejected(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    desc_path = make_descriptor(tmp_path / "desc.json", index)
    _patch_models(monkeypatch)

    # wrong declared path
    d = json.loads(desc_path.read_text())
    d["dataset"]["index_path"] = str(tmp_path / "elsewhere.json")
    (tmp_path / "elsewhere.json").write_text("{}")
    desc_bad_path = tmp_path / "bad_path.json"
    desc_bad_path.write_text(json.dumps(d))
    out = tmp_path / "out1"
    rc = mod.main(["--descriptor", str(desc_bad_path), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out),
                   "--variants", "rgb_pretrained", "--device", "cpu"])
    assert rc == 2

    # wrong expected SHA
    d = json.loads(desc_path.read_text())
    d["dataset"]["index_sha256"] = "0" * 64
    desc_bad_sha = tmp_path / "bad_sha.json"
    desc_bad_sha.write_text(json.dumps(d))
    out2 = tmp_path / "out2"
    rc = mod.main(["--descriptor", str(desc_bad_sha), "--dataset-index", str(index),
                   "--split", "val", "--output-dir", str(out2),
                   "--variants", "rgb_pretrained", "--device", "cpu"])
    assert rc == 2


# --------------------------------------------------------------------------- #
# variant set: empty / duplicate rejection
# --------------------------------------------------------------------------- #
def test_empty_and_duplicate_variants_rejected(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    index = make_staging(root, [(0, 1)])
    desc = make_descriptor(tmp_path / "desc.json", index)
    _patch_models(monkeypatch)
    with pytest.raises(mod.EvaluationRunError):
        mod.run(descriptor=json.loads(desc.read_text()), index_path=str(index),
                split="val", output_dir=str(tmp_path / "o0"), variants=[])
    with pytest.raises(mod.EvaluationRunError):
        mod.run(descriptor=json.loads(desc.read_text()), index_path=str(index),
                split="val", output_dir=str(tmp_path / "o1"),
                variants=["rgb_pretrained", "rgb_pretrained"])


# --------------------------------------------------------------------------- #
# released-asset identity verified for ALL variants (including RGB)
# --------------------------------------------------------------------------- #
def test_asset_identity_verified_including_rgb(tmp_path):
    from uniflowmatch.models.checkpoint import CheckpointIdentityError

    asset = tmp_path / "model.safetensors"
    asset.write_bytes(b"released-weights-not-real")
    good = mod._sha256_file(asset)
    ok = mod._verify_asset_identity(
        {"asset_path": str(asset), "asset_sha256": good, "asset_bytes": asset.stat().st_size}
    )
    assert ok["sha256"] == good
    with pytest.raises(CheckpointIdentityError):
        mod._verify_asset_identity({"asset_path": str(asset), "asset_sha256": "0" * 64})
    with pytest.raises(CheckpointIdentityError):
        mod._verify_asset_identity(
            {"asset_path": str(asset), "asset_sha256": good, "asset_bytes": 1}
        )


def test_fp32_guard_rejects_half_model():
    class _Half(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.zeros(1, dtype=torch.float16))

    class _F32(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.zeros(1, dtype=torch.float32))

    mod._assert_fp32_model(_F32(), "rgb_pretrained")
    with pytest.raises(mod.EvaluationRunError):
        mod._assert_fp32_model(_Half(), "rgb_pretrained")
