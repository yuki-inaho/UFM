"""Pure UFM-flow -> native scene-flow track adapter (step 22A1).

Purpose
-------
The UFM model predicts dense flow on a *model grid* (e.g. ``Hm x Wm``), while
the downstream scene-flow / registration contract works in *native* image
pixels (``Hn x Wn``) with a native ``K`` and camera->world poses.  This module
converts one model-grid flow field into the native track-like record the
existing contract consumes:

* the query grid ``q`` (model pixel centres) and the target grid ``q + flow``
  are both mapped back to native UV through ``Ainv`` (the inverse of the
  native->model affine ``A``), using the same align-corners-free convention as
  the UFM preprocess;
* tracks are stored in **chronological frame order** ``[src, dst]`` for both
  directions, while the *query* frame differs per direction (``src`` for
  ``forward``, ``dst`` for ``backward``), matching the existing CoW-derived
  contract;
* depth ``mm -> m`` and the ``K' / K`` and camera->world contracts are exposed
  as small pure helpers.

What this module is NOT
-----------------------
Every identifier (frame ids, query id, timestamp, calibration) arrives from the
caller; nothing here embeds a private dataset name, id, path, count,
calibration or result.  ``confidence`` is a **unit weight** (binary
eligibility), never a predicted model confidence: the caller supplies the
evaluation support (e.g. a teacher-derived fixed support) explicitly and this
module never manufactures one from the model output.

Origins: the flow consumed here is a **UFM** prediction.  It is not a
CoW-inference result; any CoW-derived quantity is the caller's evaluation
support, not model output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DIRECTIONS",
    "NativeTrackDirection",
    "SceneFlowAdapterError",
    "adapt_direction",
    "apply_camera_to_world",
    "camera_to_world_contract",
    "depth_mm_to_m",
    "invert_affine",
    "model_query_grid",
    "model_uv_to_native",
    "native_to_model_affine",
    "scale_intrinsics",
    "transform_uv",
]

#: Supported temporal directions, in the existing contract's naming.
DIRECTIONS: tuple[str, str] = ("forward", "backward")

#: Endpoint reason codes (geometry only; never model confidence).
REASON_OK = 0
REASON_NONFINITE = 1
REASON_OUT_OF_BOUNDS = 2


class SceneFlowAdapterError(ValueError):
    """An adapter input violates the pure-conversion contract."""


def _positive_hw(hw: Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(hw, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise SceneFlowAdapterError(f"{label} must be two positive finite numbers")
    return array


def native_to_model_affine(native_hw: Sequence[int], model_hw: Sequence[int]) -> np.ndarray:
    """Affine (3x3) mapping native pixel UV -> model pixel UV.

    Align-corners-free: ``x_model = sx * x_native + (sx - 1) / 2`` with
    ``sx = Wm / Wn`` (and ``sy = Hm / Hn`` in ``y``).  This is the inverse of
    :func:`model_uv_to_native`.
    """
    native = _positive_hw(native_hw, "native_hw")
    model = _positive_hw(model_hw, "model_hw")
    sy, sx = model / native
    return np.array(
        [[sx, 0.0, (sx - 1.0) / 2.0], [0.0, sy, (sy - 1.0) / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def invert_affine(affine: np.ndarray) -> np.ndarray:
    """Invert a 3x3 finite affine; reject anything degenerate."""
    array = np.asarray(affine, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise SceneFlowAdapterError("Affine must be a finite 3x3 matrix")
    if not np.allclose(array[2], [0.0, 0.0, 1.0], atol=1e-12):
        raise SceneFlowAdapterError("Affine bottom row must be [0, 0, 1]")
    if abs(float(np.linalg.det(array))) <= np.finfo(float).eps:
        raise SceneFlowAdapterError("Affine is singular")
    inverse = np.linalg.inv(array)
    if not np.isfinite(inverse).all():
        raise SceneFlowAdapterError("Inverse affine is non-finite")
    return inverse


def transform_uv(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine to ``(N, 2)`` UV; propagate non-finite values."""
    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise SceneFlowAdapterError("UV must have shape (N, 2)")
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise SceneFlowAdapterError("Affine must be a finite 3x3 matrix")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ matrix.T
    return projected[:, :2]


def model_uv_to_native(uv_model: np.ndarray, a_native_to_model: np.ndarray) -> np.ndarray:
    """Map model pixel UV back to native pixel UV through ``Ainv``."""
    return transform_uv(uv_model, invert_affine(a_native_to_model))


def model_query_grid(model_hw: Sequence[int]) -> np.ndarray:
    """Row-major (``y`` outer, ``x`` inner) integer-centre query grid, ``(N, 2)``."""
    model = _positive_hw(model_hw, "model_hw").astype(int)
    height, width = int(model[0]), int(model[1])
    y, x = np.indices((height, width))
    return np.column_stack((x.ravel(), y.ravel())).astype(float)


def depth_mm_to_m(
    depth_mm: np.ndarray,
    scale_m_per_unit: float = 0.001,
    invalid_values: Sequence[int] = (0, 65535),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert integer millimetre depth to metres + a validity mask.

    The documented contract input is **integer millimetres** (``uint16`` in the
    capture files): a non-integer dtype is rejected rather than silently
    reinterpreted, so ``NaN`` / ``inf`` can never reach the conversion.  Within
    the integer domain, ``invalid_values`` (depth holes / saturation) and any
    negative sentinel are returned as ``0.0`` metre samples and flagged
    ``valid=False``; nothing is clipped or repaired.
    """
    raw = np.asarray(depth_mm)
    if raw.ndim != 2:
        raise SceneFlowAdapterError("Depth must be a 2D array")
    if not np.issubdtype(raw.dtype, np.integer):
        raise SceneFlowAdapterError(
            "Depth must be integer millimetres; float/NaN/inf input is rejected, not cast"
        )
    if not np.isfinite(float(scale_m_per_unit)) or scale_m_per_unit <= 0:
        raise SceneFlowAdapterError("Depth scale must be finite and positive")
    invalid = np.isin(raw, np.asarray(list(invalid_values))) | (raw < 0)
    valid = ~invalid
    metres = np.zeros(raw.shape, dtype=float)
    metres[valid] = raw[valid].astype(float) * float(scale_m_per_unit)
    return metres, valid


def scale_intrinsics(
    k_native: np.ndarray, native_hw: Sequence[int], model_hw: Sequence[int]
) -> np.ndarray:
    """Native intrinsics ``K`` -> model-resolution intrinsics ``K'``.

    ``K' = A K`` with ``A = native_to_model_affine`` so that a native
    back-projection ray re-expressed on the model grid stays consistent with the
    UV mapping used to build the tracks.
    """
    k = np.asarray(k_native, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise SceneFlowAdapterError("K must be a finite 3x3 matrix")
    return native_to_model_affine(native_hw, model_hw) @ k


def camera_to_world_contract(pose: np.ndarray) -> np.ndarray:
    """Validate and copy a rigid camera->world transform ``(4, 4)``.

    Enforces a finite proper rigid transform: bottom row ``[0, 0, 1]``-style,
    orthonormal ``SO(3)`` rotation with determinant ``+1``.  Nothing is
    regularised or filled in: an unknown pose must be rejected by the caller,
    never replaced by the identity.
    """
    array = np.asarray(pose, dtype=float)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise SceneFlowAdapterError("Pose must be a finite 4x4 matrix")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise SceneFlowAdapterError("Pose bottom row must be [0, 0, 0, 1]")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise SceneFlowAdapterError("Pose rotation is not orthonormal")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-6):
        raise SceneFlowAdapterError("Pose rotation determinant must be +1")
    return array.copy()


def apply_camera_to_world(points_cam: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a validated camera->world pose to ``(N, 3)`` camera points."""
    points = np.asarray(points_cam, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SceneFlowAdapterError("Camera points must have shape (N, 3)")
    rigid = camera_to_world_contract(pose)
    return points @ rigid[:3, :3].T + rigid[:3, 3]


@dataclass(frozen=True)
class NativeTrackDirection:
    """One temporal direction of a native scene-flow track record.

    ``tracks_uv_native`` is ``(2, N, 2)`` in **chronological** frame order
    ``[src, dst]``.  ``valid_2d`` / ``visibility`` / ``confidence`` /
    ``reason_2d`` are ``(2, N)`` aligned to that order.  ``visibility`` is the
    caller's evaluation support (binary); ``confidence`` is a unit weight, NOT a
    model confidence.  ``query_frame_id`` is ``src`` for ``forward`` and ``dst``
    for ``backward``.
    """

    frame_ids: tuple[int, int]
    query_frame_id: int
    query_array_index: int
    direction: str
    tracks_uv_native: np.ndarray
    valid_2d: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    reason_2d: np.ndarray
    track_id: np.ndarray
    timestamp_s: float

    def as_arrays(self) -> dict[str, Any]:
        """Serialisable mapping compatible with the existing track contract."""
        return {
            "tracks_uv_native": self.tracks_uv_native,
            "frame_ids": np.asarray(self.frame_ids, dtype=np.int64),
            "query_frame": np.asarray(self.query_frame_id, dtype=np.int64),
            "query_array_index": np.asarray(self.query_array_index, dtype=np.int64),
            "direction": np.asarray(self.direction),
            "valid_2d": self.valid_2d,
            "visibility": self.visibility,
            "confidence": self.confidence,
            "reason_2d": self.reason_2d,
            "track_id": self.track_id,
            "timestamp_s": np.asarray(self.timestamp_s, dtype=float),
        }


def _endpoint_reason(uv: np.ndarray, native_hw: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-point ``(valid, reason)`` for a ``(N, 2)`` native UV set."""
    native = _positive_hw(native_hw, "native_hw")
    width, height = float(native[1]), float(native[0])
    finite = np.isfinite(uv).all(axis=1)
    reason = np.where(finite, REASON_OK, REASON_NONFINITE).astype(np.uint8)
    inside = (
        finite
        & (uv >= 0.0).all(axis=1)
        & (uv[:, 0] < width)
        & (uv[:, 1] < height)
    )
    reason[(reason == REASON_OK) & ~inside] = REASON_OUT_OF_BOUNDS
    return reason == REASON_OK, reason


def _as_flow_field(flow_model_uv: np.ndarray, model_hw: Sequence[int]) -> np.ndarray:
    """Normalise a model-grid flow field to ``(N, 2)`` ``(dx, dy)`` row-major."""
    model = _positive_hw(model_hw, "model_hw").astype(int)
    height, width = int(model[0]), int(model[1])
    field = np.asarray(flow_model_uv, dtype=float)
    if field.shape == (height, width, 2):
        pass
    elif field.shape == (2, height, width):
        field = np.transpose(field, (1, 2, 0))
    else:
        raise SceneFlowAdapterError(
            f"flow_model_uv must be (H, W, 2) or (2, H, W) for model_hw={model.tolist()}"
        )
    return field.reshape(-1, 2)


def adapt_direction(
    flow_model_uv: np.ndarray,
    native_hw: Sequence[int],
    model_hw: Sequence[int],
    a_native_to_model: np.ndarray,
    src_frame_id: int,
    dst_frame_id: int,
    direction: str,
    timestamp_s: float,
    query_uv_model: np.ndarray | None = None,
    eligibility: np.ndarray | None = None,
) -> NativeTrackDirection:
    """Convert one model-grid flow field into a native track direction.

    Parameters
    ----------
    flow_model_uv:
        Model-grid flow ``(H, W, 2)`` or ``(2, H, W)`` in model pixels.
    native_hw, model_hw:
        ``(height, width)`` for the native image and model grid.
    a_native_to_model:
        Affine ``A`` mapping native -> model; its inverse maps the query and
        target grids back to native UV.
    src_frame_id, dst_frame_id:
        Chronological pair ids.  Tracks are stored ``[src, dst]``.
    direction:
        ``"forward"`` (query at ``src``) or ``"backward"`` (query at ``dst``,
        i.e. the flow maps ``dst -> src``).
    timestamp_s:
        Pair timestamp carried through unchanged.
    query_uv_model:
        Optional ``(N, 2)`` query grid in model pixels.  Defaults to the full
        integer-centre model grid; pass an empty ``(0, 2)`` array for the empty
        case.
    eligibility:
        Optional ``(H, W)`` or ``(N,)`` **bool** evaluation support, broadcast to
        both frames as ``visibility``.  It is an *evaluation* mask, never a
        model confidence; a non-bool dtype is rejected (no implicit cast).
        ``None`` means "no eligibility filtering" (all points eligible), not a
        fabricated confidence.

    Returns a :class:`NativeTrackDirection`.
    """
    if direction not in DIRECTIONS:
        raise SceneFlowAdapterError(f"direction must be one of {DIRECTIONS}")
    flow = _as_flow_field(flow_model_uv, model_hw)
    model = _positive_hw(model_hw, "model_hw").astype(int)
    height, width = int(model[0]), int(model[1])

    if query_uv_model is None:
        query_model = model_query_grid((height, width))
        default_ids = np.arange(height * width, dtype=np.int64)
    else:
        query_model = np.asarray(query_uv_model, dtype=float)
        if query_model.ndim != 2 or query_model.shape[1] != 2:
            raise SceneFlowAdapterError("query_uv_model must have shape (N, 2)")
        default_ids = np.arange(len(query_model), dtype=np.int64)
        if len(query_model) != len(flow):
            # A caller-provided query grid is only valid with a matching flow field.
            raise SceneFlowAdapterError("query_uv_model length must match the flow field")

    if len(query_model) != len(flow):
        raise SceneFlowAdapterError("query grid length must match the flow field")

    ainv = invert_affine(a_native_to_model)
    target_model = query_model + flow
    query_native = transform_uv(query_model, ainv)
    target_native = transform_uv(target_model, ainv)

    query_valid, query_reason = _endpoint_reason(query_native, native_hw)
    target_valid, target_reason = _endpoint_reason(target_native, native_hw)

    n = len(query_model)
    tracks = np.empty((2, n, 2), dtype=float)
    valid = np.zeros((2, n), dtype=bool)
    reason = np.zeros((2, n), dtype=np.uint8)
    if direction == "forward":
        sources, query_id, query_index = (query_native, target_native), int(src_frame_id), 0
        valid[0], valid[1] = query_valid, target_valid
        reason[0], reason[1] = query_reason, target_reason
    else:
        sources, query_id, query_index = (target_native, query_native), int(dst_frame_id), 1
        valid[0], valid[1] = target_valid, query_valid
        reason[0], reason[1] = target_reason, query_reason
    tracks[0], tracks[1] = sources

    if eligibility is None:
        eligible = np.ones(n, dtype=bool)
    else:
        support = np.asarray(eligibility)
        if support.dtype != np.bool_:
            # An evaluation support must be an explicit boolean mask: implicit
            # ``float``/``int``/``NaN`` -> bool coercion is forbidden so a
            # numeric confidence can never masquerade as eligibility.
            raise SceneFlowAdapterError(
                "eligibility must be a bool array; implicit float/int/NaN->bool cast is forbidden"
            )
        if support.shape == (height, width):
            support = support.reshape(-1)
        if support.shape != (n,):
            raise SceneFlowAdapterError("eligibility must be (H, W) or (N,) matching queries")
        eligible = support

    visibility = np.broadcast_to(eligible, (2, n)).copy()
    confidence = np.ones((2, n), dtype=np.float64)  # unit weight, NOT model confidence

    return NativeTrackDirection(
        frame_ids=(int(src_frame_id), int(dst_frame_id)),
        query_frame_id=query_id,
        query_array_index=query_index,
        direction=str(direction),
        tracks_uv_native=tracks,
        valid_2d=valid,
        visibility=visibility,
        confidence=confidence,
        reason_2d=reason,
        track_id=default_ids,
        timestamp_s=float(timestamp_s),
    )
