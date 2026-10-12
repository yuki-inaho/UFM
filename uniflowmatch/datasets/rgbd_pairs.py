"""Generic paired RGB-D dataset returning the UFM two-view schema.

This is a **standalone** ``torch.utils.data.Dataset``.  It deliberately does NOT
inherit :class:`~uniflowmatch.datasets.base.base_optical_flow_dataset.BaseOpticalFlowDataset`,
because that base overwrites ``camera_intrinsics`` with all-NaN at the end of
``_get_item``; RGB-D must keep the *known finite* intrinsics.

The dataset reads an **external JSON index** (see ``schema.json`` / the example
under ``example/``) describing generic paired RGB-D captures: RGB, raw mapped uint16 mm
depth, an *already computed* 3x3 min-valid dilated3 uint16 mm depth, both masks,
depth provenance, intrinsics, and CoW forward/backward teacher flow / valid /
covisibility.  It returns a list of two view dicts whose keys are a superset of
``PATHWAY_REQUIREMENTS["identity"]`` in
``uniflowmatch/datasets/base/flow_postprocessing.py`` and of the fields the
existing model/loss read (``img``, ``data_norm_type``, ``flow``,
``non_occluded_mask``, ``fov_mask``, ``occlusion_supervision_mask``, ...).

Design rules enforced here:

* the raw mapped array and the dilated3 array are separate; the loader never
  dilates (`already_dilated` MUST be ``true``) so depth is never dilated twice;
* raw invalid ``0``/``65535`` are sanitized to ``0.0`` m; the input window
  ``0 < z < 1.3`` is an explicit configurable and is preserved as a mask that is
  distinct from the raw measurement mask;
* one pixel affine is applied to RGB / raw depth / dilated depth / masks /
  provenance / K, and the flow affine to flow / valid / covisibility; masks and
  depth use nearest sampling (no fabricated interpolated depth), RGB bilinear;
* the native image frame and the teacher flow reference frame are handled as a
  two-stage mapping so an already-resized flow is never scaled twice;
* malformed or non-finite samples fail fast (never silently skipped), with the
  ``pair_id`` and reason in the message.

No dataset-specific absolute paths, calibration arrays, or images are embedded:
all locations come from the external index at runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

try:  # imported as part of the package
    from .utils.rgbd_geometry import (
        INVALID_U16,
        MM_TO_M,
        apply_affine_to_K,
        depth_window_mask,
        flow_endpoint_in_bounds,
        plan_affines,
        u16_mm_to_meters,
        warp_depth_u16,
        warp_flow,
        warp_image_rgb,
        warp_mask_nearest,
        warp_scalar_nearest,
    )
except ImportError:  # loaded standalone (tests use an importlib path loader)
    from rgbd_geometry import (  # type: ignore
        INVALID_U16,
        MM_TO_M,
        apply_affine_to_K,
        depth_window_mask,
        flow_endpoint_in_bounds,
        plan_affines,
        u16_mm_to_meters,
        warp_depth_u16,
        warp_flow,
        warp_image_rgb,
        warp_mask_nearest,
        warp_scalar_nearest,
    )


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class RGBDIndexError(ValueError):
    """The external JSON index is malformed or violates the contract."""


class RGBDSampleError(RuntimeError):
    """A sample is missing, corrupt, mismatched, or non-finite (fail-fast)."""


class ZeroValidLabelsError(RGBDSampleError):
    """A pair has zero valid flow labels; silent skipping is forbidden."""


class SplitLeakError(RGBDIndexError):
    """A frame is shared across splits (train/val/test leakage)."""


class HashMismatchError(RGBDSampleError):
    """A file's sha256 does not match the external index."""


#: Fallback normalizations if the heavy UniCeption import is unavailable; values
#: mirror ``uniception.models.encoders.image_normalizations``.
_FALLBACK_NORM = {
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "croco": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "dust3r": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def _norm_stats(data_norm_type: str):
    try:  # pragma: no cover - exercised when the real package is importable
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

        entry = IMAGE_NORMALIZATION_DICT[data_norm_type]
        mean = np.array([float(x) for x in entry.mean], dtype=np.float32)
        std = np.array([float(x) for x in entry.std], dtype=np.float32)
        return mean, std
    except Exception:
        if data_norm_type not in _FALLBACK_NORM:
            raise RGBDIndexError(
                f"unknown data_norm_type {data_norm_type!r}; provide UniCeption "
                f"or one of {sorted(_FALLBACK_NORM)}"
            )
        mean, std = _FALLBACK_NORM[data_norm_type]
        return np.array(mean, np.float32), np.array(std, np.float32)


# --------------------------------------------------------------------------- #
# index loading / validation
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: a lowercase 64-hex sha256 digest (the only accepted hash format).
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _validate_window(window_m) -> tuple:
    """Validate an explicit input window ``(near, far)`` in metres.

    Requires finite values with ``near >= 0`` and ``far > near``.
    """
    try:
        near = float(window_m[0])
        far = float(window_m[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise RGBDIndexError(
            f"depth_window_m must be a (near, far) pair, got {window_m!r}"
        ) from exc
    if not (bool(np.isfinite(near)) and bool(np.isfinite(far))):
        raise RGBDIndexError(f"depth_window_m must be finite, got {(near, far)}")
    if near < 0.0:
        raise RGBDIndexError(f"depth_window_m near must be >= 0, got {near}")
    if not far > near:
        raise RGBDIndexError(f"depth_window_m far must be > near, got {(near, far)}")
    return near, far


def load_index(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RGBDIndexError(f"index not found: {path}")
    try:
        with open(path) as f:
            index = json.load(f)
    except json.JSONDecodeError as exc:
        raise RGBDIndexError(f"index is not valid JSON: {path}: {exc}") from exc
    if not isinstance(index, dict):
        raise RGBDIndexError("index root must be an object")
    return index


_REQUIRED_PAIR_KEYS = ("pair_id", "capture", "split", "views", "flow_fwd", "flow_bwd")
_REQUIRED_VIEW_KEYS = (
    "frame_index", "rgb_path", "raw_depth_mm_path", "dilated3_depth_mm_path",
    "raw_depth_mask_path", "dilated3_depth_mask_path", "depth_provenance_path", "K",
)
_REQUIRED_FLOW_KEYS = ("flow_path", "valid_path", "covis_path", "reference_hw")


def validate_index(index: Dict[str, Any]) -> None:
    if index.get("schema_version") != 1:
        raise RGBDIndexError(f"unsupported schema_version {index.get('schema_version')!r}")
    if index.get("already_dilated") is not True:
        raise RGBDIndexError(
            "index must declare already_dilated=true; the loader never dilates "
            "depth (double dilation is forbidden)"
        )
    if "depth_window_m" in index:
        _validate_window(index["depth_window_m"])
    pairs = index.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise RGBDIndexError("index must contain a non-empty 'pairs' list")
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise RGBDIndexError(f"pairs[{i}] must be an object")
        missing = [k for k in _REQUIRED_PAIR_KEYS if k not in pair]
        if missing:
            raise RGBDIndexError(f"pairs[{i}] missing keys {missing}")
        if pair["split"] not in ("train", "val", "test"):
            raise RGBDIndexError(f"pairs[{i}].split invalid: {pair['split']!r}")
        if len(pair["views"]) != 2:
            raise RGBDIndexError(f"pairs[{i}] must have exactly 2 views")
        for j, view in enumerate(pair["views"]):
            mv = [k for k in _REQUIRED_VIEW_KEYS if k not in view]
            if mv:
                raise RGBDIndexError(f"pairs[{i}].views[{j}] missing keys {mv}")
        for direction in ("flow_fwd", "flow_bwd"):
            entry = pair[direction]
            mf = [k for k in _REQUIRED_FLOW_KEYS if k not in entry]
            if mf:
                raise RGBDIndexError(f"pairs[{i}].{direction} missing keys {mf}")
            ref = entry["reference_hw"]
            if not (isinstance(ref, (list, tuple)) and len(ref) == 2):
                raise RGBDIndexError(f"pairs[{i}].{direction}.reference_hw must be [H, W]")


def check_no_split_leak(index: Dict[str, Any]) -> Dict[str, Any]:
    """Reject only CROSS-split frame sharing; return a small report.

    A frame reused by several pairs **within the same split** is allowed: this is
    a generic loader and adjacent pairs / sliding windows legitimately share
    frames.  Forbidden is a frame appearing in more than one split
    (train/val/test leakage).  Duplicate ``pair_id`` values are rejected
    separately as an index error.
    """
    frames: Dict[Any, set] = {}
    seen_ids: set = set()
    duplicate_ids: List[str] = []
    for pair in index["pairs"]:
        pid = pair["pair_id"]
        if pid in seen_ids:
            duplicate_ids.append(pid)
        seen_ids.add(pid)
        capture = pair["capture"]
        split = pair["split"]
        for view in pair["views"]:
            key = (capture, int(view["frame_index"]))
            frames.setdefault(key, set()).add(split)
    leaks = {k: sorted(v) for k, v in frames.items() if len(v) > 1}
    if leaks:
        raise SplitLeakError(f"frame shared across splits: {leaks}")
    if duplicate_ids:
        raise RGBDIndexError(
            f"duplicate pair_id(s) (rejected separately): {sorted(set(duplicate_ids))}"
        )
    return {
        "n_frames": len(frames),
        "n_pairs": len(index["pairs"]),
        "n_unique_pair_ids": len(seen_ids),
        "splits": sorted({p["split"] for p in index["pairs"]}),
    }


# --------------------------------------------------------------------------- #
# low-level file reads
# --------------------------------------------------------------------------- #
def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RGBDSampleError(f"cannot read RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_u16(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RGBDSampleError(f"cannot read depth image: {path}")
    if arr.ndim == 3:
        raise RGBDSampleError(f"depth image must be single channel: {path} shape={arr.shape}")
    if arr.dtype != np.uint16:
        raise RGBDSampleError(f"depth image must be uint16 mm: {path} dtype={arr.dtype}")
    return arr


def _read_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RGBDSampleError(f"cannot read mask image: {path}")
    if arr.ndim == 3:
        raise RGBDSampleError(f"mask image must be single channel: {path} shape={arr.shape}")
    return arr > 0


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RGBDSampleError(f"cannot read provenance image: {path}")
    if arr.ndim == 3:
        raise RGBDSampleError(f"provenance image must be single channel: {path} shape={arr.shape}")
    return arr.astype(np.uint8)


def _load_flow(entry: Dict[str, Any], flow_path: Path, valid_path: Path, covis_path: Path, pair_id: str):
    if not flow_path.is_file():
        raise RGBDSampleError(f"[{pair_id}] missing flow npz: {flow_path}")
    try:
        with np.load(str(flow_path)) as data:
            if "flow" not in data:
                raise RGBDSampleError(f"[{pair_id}] flow npz lacks 'flow': {flow_path}")
            flow = np.asarray(data["flow"])
    except RGBDSampleError:
        raise
    except Exception as exc:
        raise RGBDSampleError(f"[{pair_id}] corrupt flow npz {flow_path}: {exc!r}") from exc
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise RGBDSampleError(f"[{pair_id}] flow must be (2, H, W), got {flow.shape}")
    flow = flow.astype(np.float32)
    ref_h, ref_w = int(entry["reference_hw"][0]), int(entry["reference_hw"][1])
    if tuple(flow.shape[1:]) != (ref_h, ref_w):
        raise RGBDSampleError(
            f"[{pair_id}] flow shape {flow.shape[1:]} != reference_hw {(ref_h, ref_w)}"
        )
    valid = _read_mask(valid_path)
    covis = _read_mask(covis_path)
    if valid.shape != (ref_h, ref_w) or covis.shape != (ref_h, ref_w):
        raise RGBDSampleError(
            f"[{pair_id}] valid/covis shape {valid.shape}/{covis.shape} != reference_hw"
        )
    if valid.sum() == 0:
        raise ZeroValidLabelsError(
            f"[{pair_id}] zero valid flow labels for this direction; silent skip is forbidden"
        )
    if not np.isfinite(flow[:, valid]).all():
        raise RGBDSampleError(f"[{pair_id}] non-finite flow value at a valid label")
    flow[:, ~valid] = 0.0
    return flow, valid, covis


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class RGBDPairDataset(torch.utils.data.Dataset):
    """Generic paired RGB-D dataset producing the UFM two-view schema."""

    def __init__(
        self,
        index_path,
        *,
        root=None,
        split: Optional[str] = None,
        data_norm_type: str = "dinov2",
        transform: str = "imgnorm",
        resolution: Optional[Sequence[int]] = None,
        output_hw: Optional[Sequence[int]] = None,
        native_hw: Optional[Sequence[int]] = None,
        crop_xy: Optional[Sequence[float]] = None,
        crop_hw: Optional[Sequence[int]] = None,
        hflip: bool = False,
        vflip: bool = False,
        random_hflip: bool = False,
        depth_window_m: Sequence[float] = (0.0, 1.3),
        mm_to_m: float = MM_TO_M,
        already_dilated: bool = True,
        verify_hashes: bool = True,
        occlusion_supervision_source: str = "flow_valid",
        seed: Optional[int] = None,
        on_zero_valid: str = "error",
    ):
        if transform != "imgnorm":
            raise RGBDIndexError(
                f"unsupported transform {transform!r}; only 'imgnorm' is implemented "
                f"(colorjitter is rejected because it is not implemented)"
            )
        if already_dilated is not True:
            raise RGBDIndexError("already_dilated must be True; the loader never dilates depth")
        if on_zero_valid != "error":
            raise RGBDIndexError("on_zero_valid must be 'error'; silent skipping is forbidden")
        if occlusion_supervision_source not in ("flow_valid", "path"):
            raise RGBDIndexError("occlusion_supervision_source must be 'flow_valid' or 'path'")

        self.index_path = Path(index_path)
        self.root = Path(root) if root is not None else self.index_path.parent
        self.index = load_index(self.index_path)
        validate_index(self.index)
        self.split_report = check_no_split_leak(self.index)

        if mm_to_m != MM_TO_M:
            raise RGBDIndexError(
                f"mm_to_m must be {MM_TO_M} for uint16 millimetre depth, got {mm_to_m}"
            )
        self.mm_to_m = float(mm_to_m)
        self.data_norm_type = data_norm_type
        self.transform = transform
        self.verify_hashes = bool(verify_hashes)
        near, far = _validate_window(depth_window_m)
        self.depth_window_m = (near, far)
        self.occlusion_supervision_source = occlusion_supervision_source
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        self.random_hflip = bool(random_hflip)
        self.crop_xy = tuple(crop_xy) if crop_xy is not None else None
        self.crop_hw = tuple(crop_hw) if crop_hw is not None else None
        self.seed = seed
        self.epoch = 0

        if output_hw is not None:
            self.target_hw = (int(output_hw[0]), int(output_hw[1]))
        elif resolution is not None:
            # UFM resolution is (W, H)
            self.target_hw = (int(resolution[1]), int(resolution[0]))
        else:
            self.target_hw = None

        if native_hw is not None:
            self.native_hw = (int(native_hw[0]), int(native_hw[1]))
        elif "native_hw" in self.index:
            self.native_hw = (int(self.index["native_hw"][0]), int(self.index["native_hw"][1]))
        else:
            self.native_hw = None  # derived per pair from the RGB shape

        self._norm_mean, self._norm_std = _norm_stats(data_norm_type)

        pairs = self.index["pairs"]
        if split is not None and split != "all":
            pairs = [p for p in pairs if p["split"] == split]
        self._pairs = pairs
        self._partition = {"train": 0, "val": 1, "test": 2}

    # -- torch Dataset protocol ------------------------------------------- #
    def __len__(self) -> int:
        return len(self._pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def get_stats(self) -> str:
        return f"{len(self)} pairs ({self.split_report['splits']})"

    def __repr__(self) -> str:
        return (
            f"RGBDPairDataset({len(self)} pairs, split={self.index.get('dataset_name')!r}, "
            f"norm={self.data_norm_type!r}, target_hw={self.target_hw})"
        )

    # -- helpers ----------------------------------------------------------- #
    def _verify_hash(self, pair: Dict[str, Any], rel: str, path: Path) -> None:
        if not self.verify_hashes:
            return
        table = pair.get("sha256") or {}
        expected = table.get(rel)
        if expected is None:
            raise HashMismatchError(
                f"[{pair['pair_id']}] missing sha256 for used file {rel} "
                f"(verify_hashes=True requires a 64-hex digest for every used path)"
            )
        if not (isinstance(expected, str) and _HEX64.match(expected)):
            raise HashMismatchError(
                f"[{pair['pair_id']}] sha256 for {rel} is not 64 lowercase hex: "
                f"{expected!r}"
            )
        got = _sha256_file(path)
        if got != expected:
            raise HashMismatchError(
                f"[{pair['pair_id']}] sha256 mismatch for {rel}: {got} != {expected}"
            )

    def _resolve(self, pair: Dict[str, Any], rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not path.is_file():
            raise RGBDSampleError(f"[{pair['pair_id']}] missing file {rel} -> {path}")
        self._verify_hash(pair, rel, path)
        return path

    def _check_known_K(self, pair: Dict[str, Any], view: Dict[str, Any]):
        K_known = bool(view.get("K_known", True))
        K = np.asarray(view.get("K"), dtype=np.float64)
        if K.shape != (3, 3):
            raise RGBDSampleError(f"[{pair['pair_id']}] K must be 3x3, got {K.shape}")
        if K_known:
            if not np.isfinite(K).all():
                raise RGBDSampleError(f"[{pair['pair_id']}] K marked known but non-finite")
            return K.astype(np.float32), True
        return np.full((3, 3), np.nan, np.float32), False

    # -- item -------------------------------------------------------------- #
    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        pair = self._pairs[idx]
        pair_id = pair["pair_id"]

        fwd_ref = (int(pair["flow_fwd"]["reference_hw"][0]), int(pair["flow_fwd"]["reference_hw"][1]))
        bwd_ref = (int(pair["flow_bwd"]["reference_hw"][0]), int(pair["flow_bwd"]["reference_hw"][1]))
        if fwd_ref != bwd_ref:
            raise RGBDSampleError(
                f"[{pair_id}] fwd/bwd flow reference_hw differ: {fwd_ref} vs {bwd_ref}"
            )
        base_hw = fwd_ref

        if self.random_hflip:
            rng = np.random.default_rng([self.seed if self.seed is not None else 0, self.epoch, idx])
            do_hflip = bool(rng.random() < 0.5)
        else:
            do_hflip = self.hflip

        native_hw = self.native_hw
        if native_hw is None:
            rgb0 = _read_rgb(self._resolve(pair, pair["views"][0]["rgb_path"]))
            native_hw = (rgb0.shape[0], rgb0.shape[1])

        plan = plan_affines(
            native_hw=native_hw,
            flow_ref_hw=base_hw,
            target_hw=self.target_hw,
            crop_xy=self.crop_xy,
            crop_hw=self.crop_hw,
            hflip=do_hflip,
            vflip=self.vflip,
        )
        out_h, out_w = plan["out_hw"]
        A_img = plan["A_img"]
        A_flow = plan["A_flow"]

        views: List[Dict[str, Any]] = []
        for v in (0, 1):
            view_meta = pair["views"][v]
            direction = "fwd" if v == 0 else "bwd"
            flow_entry = pair["flow_fwd"] if v == 0 else pair["flow_bwd"]

            rgb = _read_rgb(self._resolve(pair, view_meta["rgb_path"]))
            raw = _read_u16(self._resolve(pair, view_meta["raw_depth_mm_path"]))
            dil = _read_u16(self._resolve(pair, view_meta["dilated3_depth_mm_path"]))
            raw_mask_in = _read_mask(self._resolve(pair, view_meta["raw_depth_mask_path"]))
            dil_mask_in = _read_mask(self._resolve(pair, view_meta["dilated3_depth_mask_path"]))
            prov = _read_u8(self._resolve(pair, view_meta["depth_provenance_path"]))

            rgb_hw = (rgb.shape[0], rgb.shape[1])
            if rgb_hw != native_hw:
                raise RGBDSampleError(
                    f"[{pair_id}] view {v} actual RGB {rgb_hw} != resolved native_hw "
                    f"{native_hw}"
                )
            if raw.shape != native_hw:
                raise RGBDSampleError(
                    f"[{pair_id}] view {v} raw depth {raw.shape} != resolved native_hw "
                    f"{native_hw}"
                )
            for name, arr in (("dilated3", dil), ("raw_mask", raw_mask_in),
                              ("dilated_mask", dil_mask_in), ("provenance", prov)):
                if arr.shape != raw.shape:
                    raise RGBDSampleError(
                        f"[{pair_id}] view {v} {name} shape {arr.shape} != depth {raw.shape}"
                    )

            K, K_known = self._check_known_K(pair, view_meta)

            # -- geometry: one affine for RGB/depth/masks/provenance/K -- #
            rgb_w = warp_image_rgb(rgb, A_img, (out_h, out_w)).astype(np.float32)
            raw_w = warp_depth_u16(raw, A_img, (out_h, out_w))
            dil_w = warp_depth_u16(dil, A_img, (out_h, out_w))
            raw_mask_w = warp_mask_nearest(raw_mask_in, A_img, (out_h, out_w))
            dil_mask_w = warp_mask_nearest(dil_mask_in, A_img, (out_h, out_w))
            prov_w = warp_scalar_nearest(prov, A_img, (out_h, out_w), fill=0)

            raw_depth_m, raw_numeric_valid = u16_mm_to_meters(raw_w)
            dil_depth_m, dil_numeric_valid = u16_mm_to_meters(dil_w)
            # The declared measurement masks are authoritative filters: they are
            # ANDed with numeric validity, so a true declared mask can never
            # resurrect a numeric-invalid pixel, and a false declared pixel is
            # excluded even where the numeric depth looks valid.  The raw numeric
            # values (raw_depth_m) are left as sanitized.
            raw_valid = raw_mask_w & raw_numeric_valid
            dil_valid = dil_mask_w & dil_numeric_valid
            near, far = self.depth_window_m
            # window evaluated in float64 mm->m so an exact 1.3 boundary is excluded
            raw_window = (
                depth_window_mask(raw_w.astype(np.float64) * MM_TO_M, near, far) & raw_valid
            )
            dil_window = (
                depth_window_mask(dil_w.astype(np.float64) * MM_TO_M, near, far) & dil_valid
            )

            if K_known:
                Kp = apply_affine_to_K(K, A_img).astype(np.float32)
            else:
                Kp = np.full((3, 3), np.nan, np.float32)

            # -- flow (already in the reference frame; stage2 only) -- #
            flow_p = self._resolve(pair, flow_entry["flow_path"])
            valid_p = self._resolve(pair, flow_entry["valid_path"])
            covis_p = self._resolve(pair, flow_entry["covis_path"])
            flow, valid, covis = _load_flow(flow_entry, flow_p, valid_p, covis_p, pair_id)
            flow_w = warp_flow(flow, A_flow, (out_h, out_w))
            valid_w = warp_mask_nearest(valid, A_flow, (out_h, out_w))
            covis_w = warp_mask_nearest(covis, A_flow, (out_h, out_w))
            in_bounds = flow_endpoint_in_bounds(flow_w, out_h, out_w)
            fov = valid_w & in_bounds
            non_occluded = covis_w & in_bounds
            if fov.sum() == 0:
                raise ZeroValidLabelsError(
                    f"[{pair_id}] view {v} has zero valid flow endpoints after geometry"
                )
            if not np.isfinite(flow_w[:, fov]).all():
                raise RGBDSampleError(f"[{pair_id}] view {v} non-finite flow after geometry")
            flow_w = flow_w.copy()
            flow_w[:, ~valid_w] = 0.0

            if self.occlusion_supervision_source == "flow_valid":
                occ_sup = fov
            else:  # 'path'
                occ_rel = flow_entry.get("occlusion_supervision_path")
                if not occ_rel:
                    raise RGBDSampleError(
                        f"[{pair_id}] view {v} occlusion_supervision_source='path' "
                        f"requires occlusion_supervision_path"
                    )
                occ_sup = _read_mask(self._resolve(pair, occ_rel))
                occ_sup = warp_mask_nearest(occ_sup, A_flow, (out_h, out_w)) & in_bounds

            # -- image normalization (existing UFM norm) -- #
            mean255 = (self._norm_mean * 255.0).reshape(1, 1, 3)
            std255 = (self._norm_std * 255.0).reshape(1, 1, 3)
            img = ((rgb_w - mean255) / std255).astype(np.float32).transpose(2, 0, 1)

            depthmap = np.where(raw_window, raw_depth_m, 0.0).astype(np.float32)

            view: Dict[str, Any] = {
                # UFM identity-pathway / model requirements
                "img": torch.from_numpy(np.ascontiguousarray(img)),
                "data_norm_type": self.data_norm_type,
                "true_shape": np.int32((out_h, out_w)),
                "camera_intrinsics": Kp,
                "camera_pose": np.full((4, 4), np.nan, np.float32),
                "flow": torch.from_numpy(np.ascontiguousarray(flow_w)),
                "non_occluded_mask": torch.from_numpy(np.ascontiguousarray(non_occluded)),
                "occlusion_supervision_mask": torch.from_numpy(np.ascontiguousarray(occ_sup)),
                "fov_mask": torch.from_numpy(np.ascontiguousarray(fov)),
                "flow_valid": torch.from_numpy(np.ascontiguousarray(fov)),
                "dataset": self.index.get("dataset_name", "rgbd_pairs"),
                "label": pair_id,
                "instance": f"{pair_id}_{v}",
                "is_widebaseline": False,
                "is_synthetic": bool(pair.get("is_synthetic", False)),
                "suitable_for_refinement": False,
                "data_partition": self._partition[pair["split"]],
                # loss/geometry-purpose depth inputs (raw vs dilated kept distinct)
                "raw_depth_m": torch.from_numpy(np.ascontiguousarray(raw_depth_m)),
                "raw_depth_mask": torch.from_numpy(np.ascontiguousarray(raw_valid)),
                "raw_depth_window_mask": torch.from_numpy(np.ascontiguousarray(raw_window)),
                "dilated_depth_m": torch.from_numpy(np.ascontiguousarray(dil_depth_m)),
                "dilated_depth_mask": torch.from_numpy(np.ascontiguousarray(dil_valid)),
                "dilated_depth_window_mask": torch.from_numpy(np.ascontiguousarray(dil_window)),
                "depth_provenance": torch.from_numpy(np.ascontiguousarray(prov_w.astype(np.uint8))),
                "depth_window_m": torch.tensor([near, far], dtype=torch.float32),
                # UFM-compatible depth aliases
                "depthmap": torch.from_numpy(np.ascontiguousarray(depthmap)),
                "valid_mask": torch.from_numpy(np.ascontiguousarray(raw_window)),
                "sky_mask": torch.zeros((out_h, out_w), dtype=torch.bool),
                # collatable metadata
                "pair_id": pair_id,
                "capture": pair["capture"],
                "split": pair["split"],
                "frame_index": int(view_meta["frame_index"]),
                "flow_direction": direction,
                "already_dilated": True,
            }
            views.append(view)

        return views


__all__ = [
    "RGBDPairDataset",
    "load_index",
    "validate_index",
    "check_no_split_leak",
    "RGBDIndexError",
    "RGBDSampleError",
    "ZeroValidLabelsError",
    "SplitLeakError",
    "HashMismatchError",
]
