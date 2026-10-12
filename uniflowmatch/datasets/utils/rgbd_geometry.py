"""Generic RGB-D spatial geometry for the paired UFM dataset.

Pure, dataset-independent pixel geometry helpers shared by the generic RGB-D
loader.  No dataset-specific paths, calibration or private identifiers are
referenced.  Only ``numpy``/``cv2`` are required at import time; ``torch`` is not
imported here.

Conventions (approved design contract, all machine-checked in
``tests/test_rgbd_geometry.py``):

* A 3x3 affine ``A`` acts on homogeneous **source** pixel coords ``(u, v, 1)``:
  ``dst = A @ (u, v, 1)``.  The intended intrinsic of the transformed image is
  ``K' = A @ K``.
* resize (``align_corners=False`` half-pixel):
  ``A = [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]``.
* crop with positive origin ``(x0, y0)`` **subtracts**:
  ``A = [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]``.
* horizontal flip reflects RGB pixels while keeping camera coordinates
  unchanged: ``F = [[-1, 0, W-1], [0, 1, 0], [0, 0, 1]]`` so ``fx`` becomes
  negative and ``cx' = (W-1) - cx``.
* flow transforms as ``flow' = A(target) - A(source)``; equivalently only the
  linear part acts on the flow vector (the translation cancels).  hflip negates
  ``dx`` and keeps ``dy``; the source/target **roles are never swapped**.
* depth/masks/provenance use nearest sampling (never fabricated interpolated
  depth); RGB alone may use bilinear.

Two-stage native -> teacher mapping: RGB/raw-depth live at the native capture
resolution while the teacher flow already lives at ``reference_hw``.  The
loader first resizes images/depth/K from native to the flow reference frame and
then applies the augmentation affine to everything -- so an already-resized
flow is never scaled a second time.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

#: uint16 millimetre invalid sentinel (raw mapped / dilated depth).
INVALID_U16 = 65535

#: default millimetres -> metres factor.
MM_TO_M = 0.001


# --------------------------------------------------------------------------- #
# depth units and masks
# --------------------------------------------------------------------------- #
def u16_mm_to_meters(mm, sentinel: int = INVALID_U16):
    """Convert uint16 millimetres to metres, sanitizing invalid values.

    Invalid values (``0`` and ``sentinel``) become ``0.0`` metres and are marked
    False in the returned mask.  The invalid values are **not** propagated.

    Returns
    -------
    (depth_m, valid) : (np.ndarray float32, np.ndarray bool)
    """
    mm = np.asarray(mm)
    if mm.ndim < 1:
        raise ValueError(f"depth must be at least 1D, got shape {mm.shape}")
    valid = (mm != 0) & (mm != sentinel)
    depth_m = np.where(valid, mm.astype(np.float64) * MM_TO_M, 0.0).astype(np.float32)
    return depth_m, valid.astype(bool)


def depth_window_mask(depth_m, near: float = 0.0, far: float = 1.3):
    """Explicit input window mask ``near < z < far`` (exclusive both ends).

    This is deliberately distinct from the raw measurement mask so the loader
    can preserve both.  NaN/``0`` depths are outside the window.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    return (depth_m > near) & (depth_m < far)


# --------------------------------------------------------------------------- #
# affine construction / composition
# --------------------------------------------------------------------------- #
def _affine_3x3(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3x3 affine, got {m.shape}")
    return m


def resize_affine(sx: float, sy: float) -> np.ndarray:
    """Half-pixel (``align_corners=False``) resize affine for scale ``sx, sy``."""
    return np.array(
        [[sx, 0.0, (sx - 1.0) / 2.0], [0.0, sy, (sy - 1.0) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resize_affine_from_hw(src_hw: Sequence[int], dst_hw: Sequence[int]) -> np.ndarray:
    """Resize affine mapping a ``(H, W)`` source to a ``(H, W)`` destination."""
    sh, sw = int(src_hw[0]), int(src_hw[1])
    dh, dw = int(dst_hw[0]), int(dst_hw[1])
    return resize_affine(dw / float(sw), dh / float(sh))


def crop_affine(x0: float, y0: float) -> np.ndarray:
    """Crop affine: a positive origin subtracts from pixel coordinates."""
    return np.array([[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]], dtype=np.float64)


def hflip_affine(width: int) -> np.ndarray:
    """Horizontal pixel reflection; camera coords preserved (``fx -> -fx``)."""
    return np.array([[-1.0, 0.0, float(width) - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def vflip_affine(height: int) -> np.ndarray:
    """Vertical pixel reflection; camera coords preserved (``fy -> -fy``)."""
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(height) - 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def compose_affines(*affines) -> np.ndarray:
    """Compose affines applied **left to right**.

    ``compose_affines(A, B, C) == C @ B @ A`` (i.e. apply ``A`` first).
    """
    out = np.eye(3, dtype=np.float64)
    for a in affines:
        out = _affine_3x3(a) @ out
    return out


def apply_affine_to_K(K, A) -> np.ndarray:
    """Intrinsic of the transformed image: ``K' = A @ K``."""
    return _affine_3x3(A) @ np.asarray(K, dtype=np.float64)


# --------------------------------------------------------------------------- #
# point / flow transforms
# --------------------------------------------------------------------------- #
def transform_points(A, pts) -> np.ndarray:
    """Apply a 3x3 affine to ``(N, 2)`` Cartesian points -> ``(N, 2)``."""
    A = _affine_3x3(A)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {pts.shape}")
    return (A[:2, :2] @ pts.T).T + A[:2, 2]


def transform_flow(flow, A) -> np.ndarray:
    """Transform a ``(2, H, W)`` flow field's vectors by ``A`` (linear part)."""
    A = _affine_3x3(A)
    flow = np.asarray(flow, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow must be (2, H, W), got {flow.shape}")
    out = np.einsum("ij,jhw->ihw", A[:2, :2], flow)
    return out.astype(np.float32)


def flow_endpoint_in_bounds(flow, height: int, width: int) -> np.ndarray:
    """``True`` where the target endpoint ``grid + flow`` lies in the image.

    Uses the same convention as UFM's fov mask: valid coordinates satisfy
    ``0 <= u < W`` and ``0 <= v < H``.
    """
    flow = np.asarray(flow, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"flow must be (2, H, W), got {flow.shape}")
    if tuple(flow.shape[1:]) != (height, width):
        raise ValueError(f"flow spatial shape {flow.shape[1:]} != {(height, width)}")
    u = np.arange(width, dtype=np.float64)[None, :] + flow[0]
    v = np.arange(height, dtype=np.float64)[:, None] + flow[1]
    return (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)


# --------------------------------------------------------------------------- #
# image warps (same pixel affine for every field)
# --------------------------------------------------------------------------- #
def _M(A) -> np.ndarray:
    return _affine_3x3(A)[:2, :].astype(np.float64)


def warp_image_rgb(img_hwc, A, out_hw: Sequence[int]) -> np.ndarray:
    """Bilinear RGB warp.  ``img_hwc`` may be uint8 or float; output is float32."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    src = np.asarray(img_hwc)
    is_u8 = src.dtype == np.uint8
    src_f = src if is_u8 else src.astype(np.float32)
    out = cv2.warpAffine(
        src_f, _M(A), (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return out.astype(np.uint8) if is_u8 else out.astype(np.float32)


def warp_depth_u16(depth_u16, A, out_hw: Sequence[int]) -> np.ndarray:
    """Nearest uint16 depth warp.  No interpolated (fake) depth is created."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    src = np.asarray(depth_u16)
    if src.dtype != np.uint16:
        raise ValueError(f"depth must be uint16 millimetres, got {src.dtype}")
    out = cv2.warpAffine(
        src, _M(A), (out_w, out_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return out.astype(np.uint16)


def warp_mask_nearest(mask, A, out_hw: Sequence[int]) -> np.ndarray:
    """Nearest bool mask warp (0 fill outside)."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    src = np.asarray(mask).astype(np.uint8)
    out = cv2.warpAffine(
        src, _M(A), (out_w, out_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return out.astype(bool)


def warp_scalar_nearest(arr, A, out_hw: Sequence[int], fill: int = 0) -> np.ndarray:
    """Nearest warp for an integer scalar map (e.g. provenance), preserving dtype."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    src = np.asarray(arr)
    out = cv2.warpAffine(
        src, _M(A), (out_w, out_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=int(fill),
    )
    return out.astype(src.dtype)


def warp_flow(flow, A, out_hw: Sequence[int]) -> np.ndarray:
    """Spatially nearest-resample a ``(2, H, W)`` flow then transform vectors.

    The spatial resampling matches the nearest policy used for depth/masks; the
    vector transform uses only the linear part (translation cancels).
    """
    flow = np.asarray(flow, dtype=np.float32)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    chans = []
    for c in range(2):
        chans.append(
            cv2.warpAffine(
                flow[c], _M(A), (out_w, out_h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        )
    sampled = np.stack(chans, axis=0).astype(np.float32)
    return transform_flow(sampled, A)


# --------------------------------------------------------------------------- #
# affine planning (two-stage native -> flow reference -> output)
# --------------------------------------------------------------------------- #
def augment_affine(
    base_hw: Sequence[int],
    target_hw: Optional[Sequence[int]] = None,
    crop_xy: Optional[Sequence[float]] = None,
    crop_hw: Optional[Sequence[int]] = None,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Augmentation affine in the order resize -> crop -> hflip -> vflip.

    Returns ``(A, out_hw)``.  Flip translations use the *current* (post-resize,
    post-crop) dimensions, never the original ones.
    """
    A = np.eye(3, dtype=np.float64)
    cur_h, cur_w = int(base_hw[0]), int(base_hw[1])
    if target_hw is not None and crop_hw is None:
        th, tw = int(target_hw[0]), int(target_hw[1])
        if (th, tw) != (cur_h, cur_w):
            A = resize_affine(tw / float(cur_w), th / float(cur_h)) @ A
            cur_h, cur_w = th, tw
    if crop_xy is not None:
        if crop_hw is None:
            raise ValueError("crop_hw is required when crop_xy is given")
        ch, cw = int(crop_hw[0]), int(crop_hw[1])
        A = crop_affine(float(crop_xy[0]), float(crop_xy[1])) @ A
        cur_h, cur_w = ch, cw
    if hflip:
        A = hflip_affine(cur_w) @ A
    if vflip:
        A = vflip_affine(cur_h) @ A
    return A, (cur_h, cur_w)


def plan_affines(
    native_hw: Sequence[int],
    flow_ref_hw: Sequence[int],
    target_hw: Optional[Sequence[int]] = None,
    crop_xy: Optional[Sequence[float]] = None,
    crop_hw: Optional[Sequence[int]] = None,
    hflip: bool = False,
    vflip: bool = False,
):
    """Plan the two-stage affines.

    * ``stage1`` resizes images/depth/K from ``native_hw`` to the flow reference
      frame ``flow_ref_hw`` (identity when equal).
    * ``stage2`` is the augmentation in the reference frame applied to
      everything (including the flow).

    Flow already lives in ``flow_ref_hw``, so it is transformed by ``stage2``
    only -- never double-scaled.
    """
    native_hw = (int(native_hw[0]), int(native_hw[1]))
    flow_ref_hw = (int(flow_ref_hw[0]), int(flow_ref_hw[1]))
    if native_hw == flow_ref_hw:
        stage1 = np.eye(3, dtype=np.float64)
    else:
        stage1 = resize_affine_from_hw(native_hw, flow_ref_hw)
    stage2, out_hw = augment_affine(
        flow_ref_hw, target_hw=target_hw, crop_xy=crop_xy, crop_hw=crop_hw,
        hflip=hflip, vflip=vflip,
    )
    if target_hw is not None and crop_hw is not None:
        # resize to target first, then crop to crop_hw within that frame
        th, tw = int(target_hw[0]), int(target_hw[1])
        resize2 = resize_affine(tw / float(flow_ref_hw[1]), th / float(flow_ref_hw[0]))
        A2, out_hw = augment_affine(
            (th, tw), target_hw=None, crop_xy=crop_xy, crop_hw=crop_hw, hflip=hflip, vflip=vflip
        )
        stage2 = A2 @ resize2
    return {
        "stage1": stage1,
        "stage2": stage2,
        "A_img": stage2 @ stage1,
        "A_flow": stage2,
        "native_hw": native_hw,
        "flow_ref_hw": flow_ref_hw,
        "out_hw": (int(out_hw[0]), int(out_hw[1])),
    }


__all__ = [
    "INVALID_U16",
    "MM_TO_M",
    "u16_mm_to_meters",
    "depth_window_mask",
    "resize_affine",
    "resize_affine_from_hw",
    "crop_affine",
    "hflip_affine",
    "vflip_affine",
    "compose_affines",
    "apply_affine_to_K",
    "transform_points",
    "transform_flow",
    "flow_endpoint_in_bounds",
    "warp_image_rgb",
    "warp_depth_u16",
    "warp_mask_nearest",
    "warp_scalar_nearest",
    "warp_flow",
    "augment_affine",
    "plan_affines",
]
