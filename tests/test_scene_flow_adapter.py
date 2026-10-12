"""CPU hand-oracle tests for the pure UFM-flow -> native track adapter (step 22A).

Pure / synthetic: no model, no GPU, no real captures, no private identifiers.
Every expected value is computable by hand from the affine and the constant
flow fields.  Two contract hardenings are covered explicitly:

* eligibility must be an explicit ``bool`` array (float/int/NaN must not be
  implicitly coerced);
* ``depth_mm_to_m`` requires the documented integer-millimetre input and never
  reports ``NaN`` / ``inf`` / negative samples as valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uniflowmatch.evaluation.scene_flow_adapter import (
    SceneFlowAdapterError,
    adapt_direction,
    apply_camera_to_world,
    camera_to_world_contract,
    depth_mm_to_m,
    invert_affine,
    model_query_grid,
    model_uv_to_native,
    native_to_model_affine,
    scale_intrinsics,
    transform_uv,
)


# --------------------------------------------------------------------------- #
# affine hand-oracle (native 4x4 -> model 2x2)
# --------------------------------------------------------------------------- #
def _affine_4x4_to_2x2():
    return native_to_model_affine((4, 4), (2, 2))


def test_native_to_model_affine_hand_values():
    a = _affine_4x4_to_2x2()
    # sx = sy = 0.5, offset = (0.5 - 1) / 2 = -0.25
    expected = np.array([[0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(a, expected)


def test_invert_affine_hand_values_and_roundtrip():
    ainv = invert_affine(_affine_4x4_to_2x2())
    expected = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(ainv, expected)
    np.testing.assert_allclose(ainv @ _affine_4x4_to_2x2(), np.eye(3), atol=1e-12)


def test_model_query_grid_row_major():
    grid = model_query_grid((2, 3))
    np.testing.assert_array_equal(
        grid, np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    )


def test_model_uv_to_native_hand_oracle():
    # native = 2 * model + 0.5: (0,0) -> (0.5, 0.5); (1,1) -> (2.5, 2.5)
    out = model_uv_to_native(np.array([[0.0, 0.0], [1.0, 1.0]]), _affine_4x4_to_2x2())
    np.testing.assert_allclose(out, np.array([[0.5, 0.5], [2.5, 2.5]]))


def test_scale_intrinsics_matches_affine_product():
    k = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(k, (4, 4), (2, 2))
    np.testing.assert_allclose(scaled, _affine_4x4_to_2x2() @ k)


def test_transform_uv_propagates_nonfinite():
    uv = np.array([[1.0, np.inf], [np.nan, 0.0], [1.0, 1.0]])
    moved = transform_uv(uv, _affine_4x4_to_2x2())
    assert not np.isfinite(moved[0]).all()
    assert np.isnan(moved[1, 0])
    np.testing.assert_allclose(moved[2], [0.25, 0.25])


# --------------------------------------------------------------------------- #
# adapt_direction: chronological order + query slot + hand-computed targets
# --------------------------------------------------------------------------- #
def test_adapt_direction_hand_oracle_forward_and_backward():
    native_hw, model_hw = (4, 4), (2, 2)
    affine = _affine_4x4_to_2x2()
    # constant model flow (dx, dy) = (1, 0)
    flow = np.zeros((2, 2, 2))
    flow[:, :, 0] = 1.0

    fw = adapt_direction(
        flow,
        native_hw,
        model_hw,
        affine,
        src_frame_id=10,
        dst_frame_id=11,
        direction="forward",
        timestamp_s=1.5,
    )
    # query grid native = 2 * model + 0.5 -> (0,0) maps to (0.5, 0.5).
    # target model = q + (1, 0) -> native (1,0) maps to (2.5, 0.5).
    assert fw.frame_ids == (10, 11)
    assert fw.query_frame_id == 10 and fw.query_array_index == 0
    np.testing.assert_allclose(fw.tracks_uv_native[0, 0], [0.5, 0.5])
    np.testing.assert_allclose(fw.tracks_uv_native[1, 0], [2.5, 0.5])
    assert fw.timestamp_s == 1.5
    np.testing.assert_array_equal(fw.track_id, np.arange(4))

    bw = adapt_direction(
        flow,
        native_hw,
        model_hw,
        affine,
        src_frame_id=10,
        dst_frame_id=11,
        direction="backward",
        timestamp_s=1.5,
    )
    assert bw.frame_ids == (10, 11)
    assert bw.query_frame_id == 11 and bw.query_array_index == 1
    # backward keeps chronological [src, dst]: slot0 still frame 10 (target),
    # slot1 frame 11 (query).
    np.testing.assert_allclose(bw.tracks_uv_native[0, 0], [2.5, 0.5])
    np.testing.assert_allclose(bw.tracks_uv_native[1, 0], [0.5, 0.5])


def test_adapt_direction_bounds_and_confidence_unit_weight():
    native_hw, model_hw = (2, 2), (4, 4)
    affine = native_to_model_affine(native_hw, model_hw)
    flow = np.zeros((4, 4, 2))  # identity: model (0,0) -> native (-0.25, -0.25) out of bounds
    out = adapt_direction(flow, native_hw, model_hw, affine, 0, 1, "forward", 0.25)
    assert not out.valid_2d[:, 0].any()  # first query maps outside native bounds
    np.testing.assert_allclose(out.confidence, 1.0)  # unit weight, not a model score


# --------------------------------------------------------------------------- #
# eligibility: explicit bool required, no implicit float/int/NaN -> bool
# --------------------------------------------------------------------------- #
def test_eligibility_requires_bool_dtype():
    native_hw, model_hw = (4, 4), (2, 2)
    affine = _affine_4x4_to_2x2()
    flow = np.zeros((2, 2, 2))
    for bad in (
        np.ones((2, 2), dtype=float),
        np.ones((2, 2), dtype=np.int64),
        np.full((2, 2), np.nan),
    ):
        with pytest.raises(SceneFlowAdapterError):
            adapt_direction(flow, native_hw, model_hw, affine, 0, 1, "forward", 0.1, eligibility=bad)


def test_eligibility_bool_hw_and_flat_shapes_agree():
    native_hw, model_hw = (4, 4), (2, 2)
    affine = _affine_4x4_to_2x2()
    flow = np.zeros((2, 2, 2))
    mask_hw = np.array([[True, False], [False, True]])
    out_hw = adapt_direction(flow, native_hw, model_hw, affine, 0, 1, "forward", 0.1, eligibility=mask_hw)
    out_flat = adapt_direction(
        flow, native_hw, model_hw, affine, 0, 1, "forward", 0.1, eligibility=mask_hw.reshape(-1)
    )
    np.testing.assert_array_equal(out_hw.visibility, out_flat.visibility)
    np.testing.assert_array_equal(out_hw.visibility, np.array([[True, False, False, True]] * 2))


def test_empty_support_is_all_false_not_fabricated():
    """An all-False (empty) evaluation support must stay empty, never imputed."""
    native_hw, model_hw = (4, 4), (2, 2)
    affine = _affine_4x4_to_2x2()
    flow = np.zeros((2, 2, 2))
    support = np.zeros((2, 2), dtype=bool)
    out = adapt_direction(flow, native_hw, model_hw, affine, 0, 1, "forward", 0.1, eligibility=support)
    assert not out.visibility.any()
    # the caller combines geometry validity with eligibility; that conjunction
    # is empty here but the geometry-only flag is preserved for diagnostics.
    assert out.valid_2d.dtype == bool
    np.testing.assert_allclose(out.confidence, 1.0)


def test_none_eligibility_means_unfiltered_not_a_confidence():
    native_hw, model_hw = (4, 4), (2, 2)
    affine = _affine_4x4_to_2x2()
    flow = np.zeros((2, 2, 2))
    out = adapt_direction(flow, native_hw, model_hw, affine, 0, 1, "forward", 0.1)
    assert out.visibility.all()
    np.testing.assert_allclose(out.confidence, 1.0)


# --------------------------------------------------------------------------- #
# depth_mm_to_m: integer millimetres only; NaN/inf/negative never valid
# --------------------------------------------------------------------------- #
def test_depth_mm_to_m_hand_values_and_mask():
    depth = np.array([[1000, 0], [65535, 2500]], dtype=np.uint16)
    metres, valid = depth_mm_to_m(depth)
    np.testing.assert_allclose(metres, np.array([[1.0, 0.0], [0.0, 2.5]]))
    np.testing.assert_array_equal(valid, np.array([[True, False], [False, True]]))


def test_depth_mm_to_m_rejects_float_nan_and_inf():
    for bad in (
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
    ):
        with pytest.raises(SceneFlowAdapterError):
            depth_mm_to_m(bad)


def test_depth_mm_to_m_negative_never_valid():
    depth = np.array([[1000, -5], [0, 42]], dtype=np.int32)
    metres, valid = depth_mm_to_m(depth)
    assert valid[0, 0] and valid[1, 1]
    assert not valid[0, 1] and not valid[1, 0]
    assert metres[0, 1] == 0.0


# --------------------------------------------------------------------------- #
# unknown pose: rejected, never replaced by identity
# --------------------------------------------------------------------------- #
def test_unknown_pose_rejected_never_identity():
    for bad in (
        np.full((4, 4), np.nan),  # unknown / missing pose
        np.eye(4),                # zero-translation identity is still a valid pose...
    ):
        if np.isfinite(bad).all():
            # a genuine identity is a legal rigid pose (not an "unknown" marker)
            out = camera_to_world_contract(bad)
            np.testing.assert_allclose(out, np.eye(4))
        else:
            with pytest.raises(SceneFlowAdapterError):
                camera_to_world_contract(bad)


def test_unknown_pose_near_miss_rejected():
    broken = np.eye(4)
    broken[0, 1] = 0.5  # non-orthonormal rotation
    with pytest.raises(SceneFlowAdapterError):
        camera_to_world_contract(broken)
    with pytest.raises(SceneFlowAdapterError):
        apply_camera_to_world(np.zeros((1, 3)), broken)


def test_apply_camera_to_world_hand_oracle():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    out = apply_camera_to_world(np.array([[0.0, 0.0, 0.0]]), pose)
    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 3.0]]))
