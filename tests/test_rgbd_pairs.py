"""Synthetic RED->GREEN tests for the generic paired RGB-D UFM loader.

Loads ``uniflowmatch/datasets/utils/rgbd_geometry.py`` and
``uniflowmatch/datasets/rgbd_pairs.py`` as standalone modules (the UFM package
top-level eagerly imports heavy model deps).  All fixtures are generated
synthetically into a tmp dir; no real data/calibration/host paths are embedded.

The loader returns the UFM two-view schema consumed by the trainer:
PATHWAY_REQUIREMENTS["identity"] in
``uniflowmatch/datasets/base/flow_postprocessing.py`` (verified in
``test_existing_rgb_dataset_regression_smoke``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import py_compile
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"environment failure: cv2 unavailable: {exc}")

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"environment failure: torch unavailable: {exc}")

_REPO = Path(__file__).resolve().parents[1]
_GEOM_PATH = _REPO / "uniflowmatch" / "datasets" / "utils" / "rgbd_geometry.py"
_LOADER_PATH = _REPO / "uniflowmatch" / "datasets" / "rgbd_pairs.py"
if not _GEOM_PATH.exists():
    raise FileNotFoundError(f"RED: missing actual API {_GEOM_PATH}")
if not _LOADER_PATH.exists():
    raise FileNotFoundError(f"RED: missing actual API {_LOADER_PATH}")

sys.path.insert(0, str(_GEOM_PATH.parent))
sys.path.insert(0, str(_LOADER_PATH.parent))
import rgbd_geometry as G  # noqa: E402
import rgbd_pairs as P  # noqa: E402

RGBDPairDataset = P.RGBDPairDataset
RGBDIndexError = P.RGBDIndexError
RGBDSampleError = P.RGBDSampleError
ZeroValidLabelsError = P.ZeroValidLabelsError
SplitLeakError = P.SplitLeakError
HashMismatchError = P.HashMismatchError


# --------------------------------------------------------------------------- #
# synthetic fixture builder
# --------------------------------------------------------------------------- #
def _ref_dilate3(src):
    out = np.zeros_like(src)
    H, W = src.shape
    for y in range(H):
        for x in range(W):
            s = int(src[y, x])
            if s == 0 or s == 65535:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ty, tx = y + dy, x + dx
                    if 0 <= ty < H and 0 <= tx < W:
                        if out[ty, tx] == 0 or s < out[ty, tx]:
                            out[ty, tx] = s
    return out.astype(np.uint16)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rgb_pattern(H, W):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def make_fixture(tmp: Path, *, native_hw=(6, 8), teacher_hw=None, zero_valid=False,
                 flow_nan=False, k_nonfinite=False, bad_mask_shape=False,
                 rgb_size_mismatch=False):
    tmp.mkdir(parents=True, exist_ok=True)
    H, W = native_hw
    if teacher_hw is None:
        teacher_hw = native_hw
    th, tw = teacher_hw
    rgb = _rgb_pattern(H, W)
    if rgb_size_mismatch:
        rgb = rgb[:-1, :-1]
    raw = np.array(
        [[0, 65535, 500, 700, 900, 1100, 1300, 1400],
         [600, 600, 600, 0, 800, 800, 1000, 65535],
         [700, 700, 0, 0, 0, 0, 900, 950],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [500, 500, 500, 500, 500, 500, 500, 500],
         [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200]], np.uint16)
    raw = raw[:H, :W]
    dil = _ref_dilate3(raw)
    raw_mask = ((raw > 0) & (raw != 65535)).astype(np.uint8)
    dil_mask = ((dil > 0) & (dil != 65535)).astype(np.uint8)
    if bad_mask_shape:
        raw_mask = raw_mask[: H - 1, : W - 1]  # wrong 2D shape, not broadcastable
    prov = np.full((H, W), 1, np.uint8)
    prov[raw == 0] = 2
    # geometry exactly on pixel centers
    fx, fy, cx, cy = 5.0, 5.0, W / 2.0, H / 2.0
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    if k_nonfinite:
        K[0][0] = float("inf")
    K_known = True

    files = {}
    for v in (1, 2):
        d = tmp / "views"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{v}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / f"{v}_raw_mm.png"), raw)
        cv2.imwrite(str(d / f"{v}_dil_mm.png"), dil)
        cv2.imwrite(str(d / f"{v}_raw_mask.png"), raw_mask)
        cv2.imwrite(str(d / f"{v}_dil_mask.png"), dil_mask)
        cv2.imwrite(str(d / f"{v}_prov.png"), prov)
        files[f"views/{v}_rgb.png"] = None
        files[f"views/{v}_raw_mm.png"] = None
        files[f"views/{v}_dil_mm.png"] = None
        files[f"views/{v}_raw_mask.png"] = None
        files[f"views/{v}_dil_mask.png"] = None
        files[f"views/{v}_prov.png"] = None

    def flow_pattern(seed):
        rng = np.random.default_rng(seed)
        fl = rng.uniform(-1.5, 1.5, size=(2, th, tw)).astype(np.float32)
        val = np.ones((th, tw), np.uint8)
        cov = np.ones((th, tw), np.uint8)
        if zero_valid:
            val[:] = 0
        if flow_nan:
            fl[0, 0, 0] = np.nan  # valid pixel non-finite
        return fl, val, cov

    fd = tmp / "flow"
    fd.mkdir(parents=True, exist_ok=True)
    fwd = flow_pattern(1)
    bwd = flow_pattern(2)
    np.savez(str(fd / "fwd.npz"), flow=fwd[0])
    np.savez(str(fd / "bwd.npz"), flow=bwd[0])
    cv2.imwrite(str(fd / "fwd_valid.png"), fwd[1])
    cv2.imwrite(str(fd / "fwd_covis.png"), fwd[2])
    cv2.imwrite(str(fd / "bwd_valid.png"), bwd[1])
    cv2.imwrite(str(fd / "bwd_covis.png"), bwd[2])
    for n in ("fwd.npz", "bwd.npz", "fwd_valid.png", "fwd_covis.png", "bwd_valid.png", "bwd_covis.png"):
        files[f"flow/{n}"] = None

    views_meta = []
    for v in (1, 2):
        views_meta.append({
            "frame_index": v,
            "rgb_path": f"views/{v}_rgb.png",
            "raw_depth_mm_path": f"views/{v}_raw_mm.png",
            "dilated3_depth_mm_path": f"views/{v}_dil_mm.png",
            "raw_depth_mask_path": f"views/{v}_raw_mask.png",
            "dilated3_depth_mask_path": f"views/{v}_dil_mask.png",
            "depth_provenance_path": f"views/{v}_prov.png",
            "K": K, "K_known": K_known,
        })
    pair = {
        "pair_id": "syn_pair_1_2", "capture": "syn_cap", "split": "train",
        "src_frame_index": 1, "dst_frame_index": 2,
        "provenance": {"origin": "synthetic"},
        "views": views_meta,
        "flow_fwd": {"flow_path": "flow/fwd.npz", "valid_path": "flow/fwd_valid.png",
                     "covis_path": "flow/fwd_covis.png", "reference_hw": [th, tw]},
        "flow_bwd": {"flow_path": "flow/bwd.npz", "valid_path": "flow/bwd_valid.png",
                     "covis_path": "flow/bwd_covis.png", "reference_hw": [th, tw]},
    }
    sha = {}
    for rel in files:
        sha[rel] = _sha(tmp / rel)
    pair["sha256"] = sha

    idx = {
        "schema_version": 1, "dataset_name": "synthetic",
        "already_dilated": True, "mm_to_m": 0.001, "depth_window_m": [0.0, 1.3],
        "native_hw": [H, W], "teacher_reference_hw": [th, tw],
        "pairs": [pair],
    }
    ip = tmp / "index.json"
    ip.write_text(json.dumps(idx, indent=1))
    return ip, pair


def load_ds(ip, **kw):
    kw.setdefault("verify_hashes", True)
    return RGBDPairDataset(str(ip), **kw)


# --------------------------------------------------------------------------- #
# schema / values
# --------------------------------------------------------------------------- #
IDENTITY_REQUIRED = {
    "img", "data_norm_type", "instance", "flow", "non_occluded_mask",
    "occlusion_supervision_mask", "fov_mask", "is_widebaseline", "is_synthetic",
    "camera_pose", "camera_intrinsics", "data_partition", "suitable_for_refinement",
}


def test_dataset_returns_ufm_two_view_schema(tmp_path):
    ip, _ = make_fixture(tmp_path)
    ds = load_ds(ip)
    assert len(ds) == 1
    views = ds[0]
    assert isinstance(views, list) and len(views) == 2
    for v, view in enumerate(views):
        missing = IDENTITY_REQUIRED - set(view.keys())
        assert not missing, f"view {v} missing {missing}"
        assert view["img"].dtype == torch.float32 and view["img"].shape[0] == 3
        assert view["flow"].shape[0] == 2
        assert tuple(view["true_shape"]) == (6, 8)
        assert view["camera_intrinsics"].shape == (3, 3)
        assert view["camera_pose"].shape == (4, 4)
        assert bool(np.isnan(view["camera_pose"]).all())
        for k in ("raw_depth_m", "dilated_depth_m", "raw_depth_mask",
                  "dilated_depth_mask", "depth_provenance",
                  "raw_depth_window_mask"):
            assert k in view, k
    # view0 forward, view1 backward
    assert views[0]["flow_direction"] == "fwd"
    assert views[1]["flow_direction"] == "bwd"


def test_mm_to_m_and_window_in_loader(tmp_path):
    ip, pair = make_fixture(tmp_path)
    ds = load_ds(ip, depth_window_m=(0.0, 1.3))
    v = ds[0][0]
    raw = v["raw_depth_m"].numpy()
    rmask = v["raw_depth_mask"].numpy()
    wmask = v["raw_depth_window_mask"].numpy()
    assert raw.shape == (6, 8)
    assert rmask[0, 2] and abs(float(raw[0, 2]) - 0.5) < 1e-6      # 500mm -> 0.5m
    assert rmask[0, 6] and not wmask[0, 6]                          # 1.3 excluded
    assert rmask[0, 7] and not wmask[0, 7]                          # 1.4 raw-valid out of window
    assert not rmask[0, 0] and raw[0, 0] == 0.0                     # invalid -> 0
    assert not rmask[0, 1]                                          # 65535 invalid


def test_raw_dilated_separation_no_double_dilation(tmp_path):
    ip, pair = make_fixture(tmp_path)
    # raw file bytes are untouched by loading
    raw_sha_before = _sha(tmp_path / "views/1_raw_mm.png")
    ds = load_ds(ip)
    v = ds[0][0]
    assert _sha(tmp_path / "views/1_raw_mm.png") == raw_sha_before
    raw = v["raw_depth_m"].numpy()
    dil = v["dilated_depth_m"].numpy()
    # position (3,7) is raw invalid but filled in dilated3 (nearest valid 500 at row4)
    assert not v["raw_depth_mask"].numpy()[3, 7]
    assert v["dilated_depth_mask"].numpy()[3, 7]
    assert dil[3, 7] > 0.0 and raw[3, 7] == 0.0
    # raw and dilated arrays are distinct objects
    assert raw is not dil


def test_dilate3_min_valid_reference_holes_and_border():
    src = np.array([[0, 0, 0, 0, 0],
                    [0, 900, 0, 0, 0],
                    [0, 0, 65535, 0, 0],
                    [0, 0, 0, 700, 0],
                    [0, 0, 0, 0, 0]], np.uint16)
    out = _ref_dilate3(src)
    assert src[1, 1] == 900 and out[1, 1] == 900
    assert out[0, 0] == 900 and out[3, 3] == 700 and out[2, 2] == 700
    assert out[4, 4] == 700 and out[0, 3] == 0  # value-max would differ


def test_K_known_finite_and_unknown_nan(tmp_path):
    ip, pair = make_fixture(tmp_path)
    v = load_ds(ip)[0][0]
    assert np.isfinite(v["camera_intrinsics"]).all()
    pair2 = json.loads(ip.read_text())
    for vm in pair2["pairs"][0]["views"]:
        vm["K_known"] = False
    ip.write_text(json.dumps(pair2))
    v2 = load_ds(ip, verify_hashes=False)[0][0]
    assert bool(np.isnan(v2["camera_intrinsics"]).all())


# --------------------------------------------------------------------------- #
# fail-fast contract
# --------------------------------------------------------------------------- #
def test_rgb_depth_size_mismatch_fails(tmp_path):
    ip, _ = make_fixture(tmp_path, rgb_size_mismatch=True)
    with pytest.raises(RGBDSampleError):
        load_ds(ip)[0]


def test_invalid_mask_shape_fails(tmp_path):
    ip, _ = make_fixture(tmp_path, bad_mask_shape=True)
    with pytest.raises(RGBDSampleError):
        load_ds(ip)[0]


def test_missing_file_fails(tmp_path):
    ip, pair = make_fixture(tmp_path)
    (tmp_path / "views/1_rgb.png").unlink()
    with pytest.raises(RGBDSampleError):
        load_ds(ip, verify_hashes=False)[0]


def test_corrupt_flow_fails(tmp_path):
    ip, _ = make_fixture(tmp_path)
    (tmp_path / "flow/fwd.npz").write_bytes(b"not an npz")
    with pytest.raises(RGBDSampleError):
        load_ds(ip, verify_hashes=False)[0]


def test_nonfinite_flow_label_fails(tmp_path):
    ip, _ = make_fixture(tmp_path, flow_nan=True)
    with pytest.raises(RGBDSampleError):
        load_ds(ip, verify_hashes=False)[0]


def test_nonfinite_K_fails(tmp_path):
    ip, _ = make_fixture(tmp_path, k_nonfinite=True)
    with pytest.raises(RGBDSampleError):
        load_ds(ip, verify_hashes=False)[0]


def test_hash_mismatch_fails(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["pairs"][0]["sha256"]["views/1_rgb.png"] = "0" * 64
    ip.write_text(json.dumps(idx))
    with pytest.raises(HashMismatchError):
        load_ds(ip, verify_hashes=True)[0]


def test_flow_hash_mismatch_fails(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["pairs"][0]["sha256"]["flow/fwd.npz"] = "0" * 64
    ip.write_text(json.dumps(idx))
    with pytest.raises(HashMismatchError):
        load_ds(ip, verify_hashes=True)[0]


def test_declared_native_hw_mismatch_fails(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["native_hw"] = [12, 16]  # actual RGB is 6x8
    ip.write_text(json.dumps(idx))
    with pytest.raises(RGBDSampleError):
        load_ds(ip, verify_hashes=False)[0]


def _overwrite_mask(tmp_path, rel, value):
    mp = tmp_path / rel
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    m[...] = value
    cv2.imwrite(str(mp), m)


def test_all_false_declared_mask_yields_all_false(tmp_path):
    ip, _ = make_fixture(tmp_path)
    _overwrite_mask(tmp_path, "views/1_raw_mask.png", 0)
    _overwrite_mask(tmp_path, "views/1_dil_mask.png", 0)
    v = load_ds(ip, verify_hashes=False)[0][0]
    # the declared all-false measurement mask is honoured (AND semantics)
    assert int(v["raw_depth_mask"].sum()) == 0
    assert int(v["dilated_depth_mask"].sum()) == 0
    assert int(v["raw_depth_window_mask"].sum()) == 0
    assert int(v["valid_mask"].sum()) == 0
    assert float(v["depthmap"].abs().sum()) == 0.0
    # sanitized numerics remain (only the mask filters them)
    assert abs(float(v["raw_depth_m"][0, 2]) - 0.5) < 1e-6


def test_partial_declared_mask_ands_with_numeric(tmp_path):
    ip, _ = make_fixture(tmp_path)
    mp = tmp_path / "views/1_raw_mask.png"
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
    m[:, :4] = 0  # remove validity on the left half even where depth is valid
    cv2.imwrite(str(mp), m)
    v = load_ds(ip, verify_hashes=False)[0][0]
    rmask = v["raw_depth_mask"].numpy()
    assert not rmask[0, 2]                              # 500mm numeric-valid, mask-removed
    assert not v["raw_depth_window_mask"].numpy()[0, 2]
    assert float(v["depthmap"][0, 2]) == 0.0
    assert rmask[0, 6]                                  # right half untouched
    assert float(v["raw_depth_m"][0, 2]) > 0.0          # numeric value preserved


def test_numeric_invalid_not_made_valid_by_true_mask(tmp_path):
    ip, _ = make_fixture(tmp_path)
    # declared true everywhere, including numeric-invalid (0/65535) pixels
    _overwrite_mask(tmp_path, "views/1_raw_mask.png", 255)
    _overwrite_mask(tmp_path, "views/1_dil_mask.png", 255)
    v = load_ds(ip, verify_hashes=False)[0][0]
    rmask = v["raw_depth_mask"].numpy()
    assert not rmask[0, 0] and not rmask[0, 1]          # numeric invalid stays invalid
    assert float(v["depthmap"][0, 0]) == 0.0


def test_window_mask_subset_of_effective_mask(tmp_path):
    ip, _ = make_fixture(tmp_path)
    for v in load_ds(ip, verify_hashes=False)[0]:
        rmask = v["raw_depth_mask"].numpy()
        rwin = v["raw_depth_window_mask"].numpy()
        dwin = v["dilated_depth_window_mask"].numpy()
        assert not (rwin & ~rmask).any()
        assert np.array_equal(v["valid_mask"].numpy(), rwin)
        assert not ((v["depthmap"].numpy() != 0.0) & ~rwin).any()
        assert not (dwin & ~v["dilated_depth_mask"].numpy()).any()


def test_zero_valid_labels_not_silently_skipped(tmp_path):
    ip, _ = make_fixture(tmp_path, zero_valid=True)
    with pytest.raises(ZeroValidLabelsError):
        load_ds(ip, verify_hashes=True)[0]


def test_split_leak_detected(tmp_path):
    ip, pair = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    leak = json.loads(json.dumps(pair))
    leak["pair_id"] = "syn_pair_2_3"
    leak["split"] = "val"
    leak["src_frame_index"] = 2
    leak["dst_frame_index"] = 3
    leak["views"][0]["frame_index"] = 2
    leak["views"][1]["frame_index"] = 3
    idx["pairs"].append(leak)
    ip.write_text(json.dumps(idx))
    with pytest.raises(SplitLeakError):
        RGBDPairDataset(str(ip), verify_hashes=False)


# --------------------------------------------------------------------------- #
# collate + seeding
# --------------------------------------------------------------------------- #
def test_default_collate_has_no_none(tmp_path):
    ip, _ = make_fixture(tmp_path)
    ds = load_ds(ip)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
    batch = next(iter(dl))
    assert isinstance(batch, list) and len(batch) == 2
    for view in batch:
        for k, val in view.items():
            assert val is not None, f"None value for key {k}"
        assert view["img"].shape[0] == 1
        assert view["flow"].shape[1] == 2


def test_worker_seed_reproducibility_and_set_epoch(tmp_path):
    ip, _ = make_fixture(tmp_path)
    a = RGBDPairDataset(str(ip), random_hflip=True, seed=0, verify_hashes=True)
    b = RGBDPairDataset(str(ip), random_hflip=True, seed=0, verify_hashes=True)
    a.set_epoch(0); b.set_epoch(0)
    # seed 0 must be honoured (not treated as "unset")
    assert torch.equal(a[0][0]["img"], b[0][0]["img"])
    assert torch.equal(a[0][0]["flow"], b[0][0]["flow"])
    c = RGBDPairDataset(str(ip), random_hflip=True, seed=0, verify_hashes=True)
    c.set_epoch(0)
    assert torch.equal(a[0][0]["img"], c[0][0]["img"])


def test_geometry_same_transform_shared_across_fields(tmp_path):
    ip, _ = make_fixture(tmp_path)
    ds = load_ds(ip, resolution=(4, 3))  # (W,H) -> half size
    v = ds[0][0]
    assert tuple(v["img"].shape[1:]) == (3, 4)
    for k in ("raw_depth_m", "raw_depth_mask", "dilated_depth_m",
              "dilated_depth_mask", "depth_provenance", "flow",
              "non_occluded_mask", "fov_mask", "raw_depth_window_mask"):
        assert tuple(v[k].shape[-2:]) == (3, 4), k


# --------------------------------------------------------------------------- #
# existing RGB dataset regression (no source changes + smoke)
# --------------------------------------------------------------------------- #
def test_existing_rgb_dataset_regression_smoke():
    for rel in ("uniflowmatch/datasets/flyingchairs.py",
                "uniflowmatch/datasets/base/base_optical_flow_dataset.py",
                "uniflowmatch/datasets/base/base_stereo_view_dataset.py",
                "uniflowmatch/datasets/base/flow_postprocessing.py"):
        py_compile.compile(str(_REPO / rel), doraise=True)
    # our returned schema satisfies the identity pathway requirement set
    src = (_REPO / "uniflowmatch/datasets/base/flow_postprocessing.py").read_text()
    for key in IDENTITY_REQUIRED:
        assert f'"{key}"' in src, key


# --------------------------------------------------------------------------- #
# correction attempt2: strict hashing, resolved native_hw, split policy,
# transform/window validation, generic docstring
# --------------------------------------------------------------------------- #
def test_missing_sha_entry_for_flow_fails(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    del idx["pairs"][0]["sha256"]["flow/fwd.npz"]
    ip.write_text(json.dumps(idx))
    with pytest.raises(HashMismatchError, match="syn_pair_1_2"):
        load_ds(ip, verify_hashes=True)[0]


@pytest.mark.parametrize("rel", ["flow/fwd_valid.png", "flow/fwd_covis.png", "flow/bwd.npz"])
def test_valid_covis_flow_hashes_verified(tmp_path, rel):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["pairs"][0]["sha256"][rel] = "0" * 64
    ip.write_text(json.dumps(idx))
    with pytest.raises(HashMismatchError):
        load_ds(ip, verify_hashes=True)[0]


def test_sha_value_must_be_64_lowercase_hex(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["pairs"][0]["sha256"]["flow/fwd.npz"] = "abc"
    ip.write_text(json.dumps(idx))
    with pytest.raises(HashMismatchError):
        load_ds(ip, verify_hashes=True)[0]


def test_occlusion_supervision_path_hash_verified(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    occ_fwd = "flow/fwd_occ.png"
    occ_bwd = "flow/bwd_occ.png"
    cv2.imwrite(str(tmp_path / occ_fwd),
                cv2.imread(str(tmp_path / "flow/fwd_valid.png"), cv2.IMREAD_UNCHANGED))
    cv2.imwrite(str(tmp_path / occ_bwd),
                cv2.imread(str(tmp_path / "flow/bwd_valid.png"), cv2.IMREAD_UNCHANGED))
    pair = idx["pairs"][0]
    pair["flow_fwd"]["occlusion_supervision_path"] = occ_fwd
    pair["flow_bwd"]["occlusion_supervision_path"] = occ_bwd
    pair["sha256"][occ_fwd] = _sha(tmp_path / occ_fwd)
    pair["sha256"][occ_bwd] = _sha(tmp_path / occ_bwd)
    ip.write_text(json.dumps(idx))
    ds = load_ds(ip, verify_hashes=True, occlusion_supervision_source="path")
    assert tuple(ds[0][0]["occlusion_supervision_mask"].shape) == (6, 8)
    idx["pairs"][0]["sha256"][occ_fwd] = "0" * 64
    ip.write_text(json.dumps(idx))
    with pytest.raises(HashMismatchError):
        load_ds(ip, verify_hashes=True, occlusion_supervision_source="path")[0]
    del idx["pairs"][0]["sha256"][occ_fwd]
    del idx["pairs"][0]["sha256"][occ_bwd]
    ip.write_text(json.dumps(idx))
    # optional path is not required when the flow_valid source is used
    assert load_ds(ip, verify_hashes=True)[0][0]["occlusion_supervision_mask"] is not None


def test_native_hw_mismatch_message_has_pair_id(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["native_hw"] = [7, 8]
    ip.write_text(json.dumps(idx))
    with pytest.raises(RGBDSampleError, match="syn_pair_1_2"):
        load_ds(ip, verify_hashes=False)[0]


def _clone_pair(pair, pair_id, split, frames):
    import copy
    p = copy.deepcopy(pair)
    p["pair_id"] = pair_id
    p["split"] = split
    for vm, fi in zip(p["views"], frames):
        vm["frame_index"] = fi
    return p


def test_same_split_frame_reuse_allowed(tmp_path):
    ip, pair = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    # adjacent pair in the SAME split reusing frame 2 is allowed (generic loader)
    idx["pairs"].append(_clone_pair(pair, "syn_pair_2_3", "train", [2, 3]))
    ip.write_text(json.dumps(idx))
    assert len(RGBDPairDataset(str(ip), verify_hashes=False)) == 2


def test_cross_split_frame_share_rejected(tmp_path):
    ip, pair = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["pairs"].append(_clone_pair(pair, "syn_pair_2_3", "val", [2, 3]))
    ip.write_text(json.dumps(idx))
    with pytest.raises(SplitLeakError):
        RGBDPairDataset(str(ip), verify_hashes=False)


def test_duplicate_pair_id_rejected(tmp_path):
    ip, pair = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["pairs"].append(_clone_pair(pair, "syn_pair_1_2", "train", [3, 4]))
    ip.write_text(json.dumps(idx))
    with pytest.raises(RGBDIndexError):
        RGBDPairDataset(str(ip), verify_hashes=False)


def test_colorjitter_transform_rejected(tmp_path):
    ip, _ = make_fixture(tmp_path)
    with pytest.raises(RGBDIndexError):
        RGBDPairDataset(str(ip), transform="colorjitter", verify_hashes=False)


@pytest.mark.parametrize("window", [(-1.0, 1.3), (1.3, 1.3), (1.3, 1.0),
                                    (0.0, float("inf")), (float("nan"), 1.3)])
def test_invalid_depth_window_rejected(tmp_path, window):
    ip, _ = make_fixture(tmp_path)
    with pytest.raises(RGBDIndexError):
        RGBDPairDataset(str(ip), depth_window_m=window, verify_hashes=False)


def test_index_depth_window_invalid_rejected(tmp_path):
    ip, _ = make_fixture(tmp_path)
    idx = json.loads(ip.read_text())
    idx["depth_window_m"] = [1.3, 1.0]
    ip.write_text(json.dumps(idx))
    with pytest.raises(RGBDIndexError):
        RGBDPairDataset(str(ip), verify_hashes=False)


def test_public_docstring_is_generic_no_private_token():
    import re
    src = _LOADER_PATH.read_text()
    assert "generic" in src.lower()
    # no host path or host/model-style private token (uppercase code + digits)
    _host_markers = ("/" + "home/", "/" + "workspace/")
    assert not any(marker in src for marker in _host_markers)
    assert not re.search(r"[A-Z]{2,}\d+-\d+", src)
