"""Synthetic RED->GREEN tests for the generic RGB-D spatial geometry utilities.

The UFM package top-level eagerly imports heavy model dependencies, so this
lightweight utility test loads ``uniflowmatch/datasets/utils/rgbd_geometry.py``
directly with an importlib file-loader instead of importing the package.

All fixtures are generated synthetically in-test. No real data, calibration,
host absolute paths, or tokens are embedded (a synthetic pinhole K is used).

Geometry follows the approved design contract exactly:
  resize: A=[sx,0,(sx-1)/2; 0,sy,(sy-1)/2; 0,0,1]  (align_corners=False)
  crop:   A=[1,0,-x0; 0,1,-y0; 0,0,1]              (origin SUBTRACTS)
  hflip:  F=[-1,0,W-1; 0,1,0; 0,0,1]               (camera coords preserved, fx<0)
  flow:   flow' = A(target) - A(source); roles unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_GEOM_PATH = Path(__file__).resolve().parents[1] / "uniflowmatch" / "datasets" / "utils" / "rgbd_geometry.py"
if not _GEOM_PATH.exists():
    raise FileNotFoundError(
        "RED: missing actual API uniflowmatch/datasets/utils/rgbd_geometry.py "
        f"(expected at {_GEOM_PATH})"
    )

_spec = importlib.util.spec_from_file_location("ufm_rgbd_geometry_under_test", _GEOM_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

u16_mm_to_meters = _mod.u16_mm_to_meters
depth_window_mask = _mod.depth_window_mask
resize_affine = _mod.resize_affine
resize_affine_from_hw = _mod.resize_affine_from_hw
crop_affine = _mod.crop_affine
hflip_affine = _mod.hflip_affine
vflip_affine = _mod.vflip_affine
compose_affines = _mod.compose_affines
apply_affine_to_K = _mod.apply_affine_to_K
transform_points = _mod.transform_points
transform_flow = _mod.transform_flow
flow_endpoint_in_bounds = _mod.flow_endpoint_in_bounds
warp_image_rgb = _mod.warp_image_rgb
warp_depth_u16 = _mod.warp_depth_u16
warp_mask_nearest = _mod.warp_mask_nearest
plan_affines = _mod.plan_affines
INVALID_U16 = _mod.INVALID_U16


def _project(X, K):
    uv = K @ np.asarray(X, float)
    return uv[:2] / uv[2]


def _apply_pt(A, uv):
    h = A @ np.array([uv[0], uv[1], 1.0])
    return h[:2] / h[2]


# --------------------------------------------------------------------------- #
# depth units / masks
# --------------------------------------------------------------------------- #
def test_mm_to_m_and_invalid_sentinel():
    mm = np.array([[0, 1000], [65535, 1234]], np.uint16)
    m, valid = u16_mm_to_meters(mm)
    assert m.dtype == np.float32
    assert valid.dtype == np.bool_
    assert m[0, 0] == 0.0 and not valid[0, 0]
    assert m[1, 0] == 0.0 and not valid[1, 0]
    assert abs(float(m[0, 1]) - 1.0) < 1e-12 and valid[0, 1]
    assert abs(float(m[1, 1]) - 1.234) < 1e-6 and valid[1, 1]


def test_depth_window_mask_distinct_from_raw_mask():
    mm = np.array([500, 1000, 1300, 1400, 65535, 0], np.uint16)
    m, raw_valid = u16_mm_to_meters(mm)
    m64 = mm.astype(np.float64) * 0.001  # window evaluated in float64 (exact 1.3)
    win = depth_window_mask(m64, near=0.0, far=1.3)
    assert m.dtype == np.float32
    # raw valid at 1.4m but OUTSIDE the input window
    assert raw_valid[3] and not win[3]
    # 1.3 exactly is excluded (strict far <)
    assert raw_valid[2] and not win[2]
    # 0.5 and 1.0 inside window
    assert win[0] and win[1]
    # invalid never inside
    assert not win[4] and not win[5]


# --------------------------------------------------------------------------- #
# affine rules
# --------------------------------------------------------------------------- #
def test_resize_half_pixel_affine_and_numeric():
    sx, sy = 0.7, 1.0
    A = resize_affine(sx, sy)
    expected = np.array([[sx, 0.0, (sx - 1) / 2.0], [0.0, sy, (sy - 1) / 2.0], [0.0, 0.0, 1.0]])
    assert np.allclose(A, expected)
    dst = _apply_pt(A, (420.0, 240.0))
    assert abs(dst[0] - 293.85) < 1e-9
    assert abs(dst[1] - 240.0) < 1e-9
    assert abs(sx * 420.0 - 294.0) < 1e-9  # no-offset value is different


def test_resize_from_hw_uses_half_pixel():
    A = resize_affine_from_hw((600, 800), (300, 400))
    assert np.allclose(A, resize_affine(0.5, 0.5))


def test_crop_origin_subtracts():
    A = crop_affine(20.0, 15.0)
    assert np.allclose(A, np.array([[1.0, 0.0, -20.0], [0.0, 1.0, -15.0], [0.0, 0.0, 1.0]]))
    dst = _apply_pt(A, (420.0, 240.0))
    assert abs(dst[0] - 400.0) < 1e-12
    assert abs(dst[1] - 225.0) < 1e-12
    assert abs((420.0 + 20.0) - 440.0) < 1e-12  # wrong plus-origin is different


def test_hflip_camera_ray_preserved_fx_negative():
    W = 800
    K = np.array([[500.0, 0.0, 400.0], [0.0, 500.0, 300.0], [0.0, 0.0, 1.0]])
    A = hflip_affine(W)
    assert np.allclose(A, np.array([[-1.0, 0.0, 799.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    K2 = apply_affine_to_K(K, A)
    assert K2[0, 0] < 0 and abs(K2[0, 0] + K[0, 0]) < 1e-9
    X = np.array([0.13, -0.07, 1.0])
    uv = _project(X, K)
    uv2 = _project(X, K2)
    assert np.allclose(uv2, [(W - 1.0) - uv[0], uv[1]], atol=1e-9)
    assert abs(_apply_pt(A, (420.0, 240.0))[0] - 379.0) < 1e-12


def test_vflip_camera_ray_preserved():
    H = 600
    K = np.array([[500.0, 0.0, 400.0], [0.0, 500.0, 300.0], [0.0, 0.0, 1.0]])
    A = vflip_affine(H)
    K2 = apply_affine_to_K(K, A)
    assert K2[1, 1] < 0
    X = np.array([0.13, -0.07, 1.0])
    uv = _project(X, K)
    uv2 = _project(X, K2)
    assert np.allclose(uv2, [uv[0], (H - 1.0) - uv[1]], atol=1e-9)


def test_composition_resize_crop_flip_numeric_and_kprime():
    sx, sy = 0.7, 1.0
    Wp = 560.0
    A = compose_affines(
        resize_affine(sx, sy), crop_affine(20.0, 15.0), hflip_affine(int(Wp))
    )
    dst = _apply_pt(A, (420.0, 240.0))
    assert abs(dst[0] - 285.15) < 1e-9 and abs(dst[1] - 225.0) < 1e-9
    seq = _apply_pt(resize_affine(sx, sy), (420.0, 240.0))
    seq = _apply_pt(crop_affine(20.0, 15.0), seq)
    seq = _apply_pt(hflip_affine(int(Wp)), seq)
    assert np.allclose(dst, seq, atol=1e-9)
    K = np.array([[500.0, 0.0, 400.0], [0.0, 500.0, 300.0], [0.0, 0.0, 1.0]])
    X = np.array([0.13, -0.07, 1.0])
    assert np.allclose(_project(X, apply_affine_to_K(K, A)), _apply_pt(A, _project(X, K)), atol=1e-9)


# --------------------------------------------------------------------------- #
# flow
# --------------------------------------------------------------------------- #
def test_flow_transform_nontautological_oracle():
    rng = np.random.default_rng(11)
    src = np.stack([rng.uniform(60, 740, 256), rng.uniform(60, 540, 256)], 1)
    flw = np.stack([rng.uniform(-6, 6, 256), rng.uniform(-6, 6, 256)], 1)
    tgt = src + flw
    for name, A in (
        ("resize", resize_affine(0.7, 1.0)),
        ("crop", crop_affine(20.0, 15.0)),
        ("hflip", hflip_affine(800)),
        ("vflip", vflip_affine(600)),
    ):
        got = transform_points(A, tgt) - transform_points(A, src)
        exp = (A[:2, :2] @ (tgt - src).T).T
        assert np.allclose(got, exp, atol=1e-9), name
    hf = hflip_affine(800)
    hfv = transform_points(hf, tgt) - transform_points(hf, src)
    assert np.allclose(hfv[:, 0], -flw[:, 0], atol=1e-9)
    assert np.allclose(hfv[:, 1], flw[:, 1], atol=1e-9)
    # roles are NOT swapped: x spatial order reverses, y stays
    order_ok = (np.argsort(transform_points(hf, src)[:, 0])[::-1] == np.argsort(src[:, 0])).all()
    assert bool(order_ok)


def test_transform_flow_2hw_matches_pointwise_oracle():
    rng = np.random.default_rng(3)
    flow = rng.uniform(-5, 5, size=(2, 4, 5)).astype(np.float32)
    for A in (resize_affine(0.5, 0.75), hflip_affine(5), vflip_affine(4), crop_affine(1.0, 2.0)):
        got = transform_flow(flow, A)
        exp = np.einsum("ij,jhw->ihw", A[:2, :2], flow.astype(np.float64))
        assert np.allclose(got, exp, atol=1e-6)


def test_flow_endpoint_out_of_bounds_invalid():
    H, W = 4, 5
    flow = np.zeros((2, H, W), np.float32)
    flow[0, 0, 0] = -1.0   # endpoint u=-1 out
    flow[0, -1, -1] = 5.0  # endpoint u = (W-1)+5 out
    flow[1, 2, 2] = 1.0    # in bounds
    valid = flow_endpoint_in_bounds(flow, H, W)
    assert not valid[0, 0]
    assert not valid[-1, -1]
    assert valid[2, 2]


# --------------------------------------------------------------------------- #
# random camera rays
# --------------------------------------------------------------------------- #
def test_random_camera_rays_reprojection():
    K = np.array([[520.0, 0.0, 400.0], [0.0, 505.0, 300.0], [0.0, 0.0, 1.0]])
    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(7)
    for _ in range(300):
        u, v = rng.uniform(0, 800), rng.uniform(0, 600)
        z = rng.uniform(0.2, 1.3)
        Xc = Kinv @ np.array([u, v, 1.0]) * z
        assert np.allclose(_project(Xc, K), [u, v], atol=1e-6)


# --------------------------------------------------------------------------- #
# warps
# --------------------------------------------------------------------------- #
def test_warp_rgb_bilinear_vs_depth_nearest():
    # RGB ramp: bilinear on 2x downscale may produce new values
    img = np.zeros((4, 4, 3), np.float32)
    img[..., 0] = np.arange(4)[None, :] * 10.0
    A = resize_affine(0.5, 0.5)
    out_rgb = warp_image_rgb(img, A, (2, 2))
    assert out_rgb.shape == (2, 2, 3)
    # depth is uint16: nearest keeps only original integer values (no fake depth)
    depth = np.array([[100, 200, 300, 400]] * 4, np.uint16)
    out_d = warp_depth_u16(depth, A, (2, 2))
    assert out_d.dtype == np.uint16
    assert set(np.unique(out_d)).issubset(set(np.unique(depth)))


def test_warp_mask_nearest_bool():
    mask = np.zeros((4, 4), bool)
    mask[1, 1] = True
    out = warp_mask_nearest(mask, resize_affine(1.0, 1.0), (4, 4))
    assert out.dtype == np.bool_
    assert out[1, 1] and out.sum() == 1


# --------------------------------------------------------------------------- #
# two-stage native -> teacher ; no double scaling of already-resized flow
# --------------------------------------------------------------------------- #
def test_plan_affines_two_stage_flow_not_double_scaled():
    native = (600, 800)
    teacher = (336, 448)
    plan = plan_affines(native_hw=native, flow_ref_hw=teacher, target_hw=teacher)
    # images/depth get native -> teacher resize
    got = _apply_pt(plan["A_img"], (800.0 - 1, 600.0 - 1))
    exp = _apply_pt(resize_affine_from_hw(native, teacher), (799.0, 599.0))
    assert np.allclose(got, exp, atol=1e-9)
    # flow already at teacher and target==teacher -> NO scaling
    assert np.allclose(plan["A_flow"], np.eye(3), atol=1e-12)
    assert tuple(plan["out_hw"]) == teacher
    # a flow vector at teacher is unchanged
    flow = np.full((2, teacher[0], teacher[1]), 2.5, np.float32)
    assert np.allclose(transform_flow(flow, plan["A_flow"]), flow)


def test_plan_affines_flow_ref_native_single_scale():
    native = (600, 800)
    target = (300, 400)  # exactly half
    plan = plan_affines(native_hw=native, flow_ref_hw=native, target_hw=target)
    assert np.allclose(plan["A_flow"], resize_affine(0.5, 0.5))
    assert np.allclose(plan["A_img"], plan["A_flow"])
    assert tuple(plan["out_hw"]) == target


def test_plan_affines_crop_and_flip_post_resize():
    native = (600, 800)
    plan = plan_affines(
        native_hw=native, flow_ref_hw=native, target_hw=(560, 780), crop_xy=(20.0, 15.0),
        crop_hw=(560, 560), hflip=True,
    )
    # flip translation must use the POST-resize/post-crop width (560), not native
    # verify a point maps as documented composition
    A = plan["A_flow"]
    seq = _apply_pt(resize_affine(780.0 / 800.0, 560.0 / 600.0), (420.0, 240.0))
    seq = _apply_pt(crop_affine(20.0, 15.0), seq)
    seq = _apply_pt(hflip_affine(560), seq)
    assert np.allclose(_apply_pt(A, (420.0, 240.0)), seq, atol=1e-9)
