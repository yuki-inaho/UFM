"""Synthetic RED->GREEN tests for the generic 3x3 min-valid depth dilation.

The UFM package top-level (and ``uniflowmatch.utils.__init__``) eagerly imports
heavy model dependencies, so this lightweight utility test loads
``uniflowmatch/utils/depth.py`` directly with an importlib file-loader instead of
importing the package.

All fixtures are generated synthetically in-test. No real data, calibration,
host absolute paths, or tokens are embedded.
"""

from __future__ import annotations

import importlib.util
import sys
import math
from pathlib import Path

import numpy as np
import pytest

_DEPTH_PATH = Path(__file__).resolve().parents[1] / "uniflowmatch" / "utils" / "depth.py"
if not _DEPTH_PATH.exists():
    raise FileNotFoundError(
        "min-valid depth utility is not implemented yet: "
        f"{_DEPTH_PATH.relative_to(_DEPTH_PATH.parents[2])}"
    )

_spec = importlib.util.spec_from_file_location("ufm_utils_depth_under_test", _DEPTH_PATH)
_depth_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _depth_mod
_spec.loader.exec_module(_depth_mod)

min_valid_depth_dilate = _depth_mod.min_valid_depth_dilate
PROVENANCE_NONE = _depth_mod.PROVENANCE_NONE
PROVENANCE_UNCHANGED = _depth_mod.PROVENANCE_UNCHANGED
PROVENANCE_FILLED = _depth_mod.PROVENANCE_FILLED
PROVENANCE_MODIFIED = _depth_mod.PROVENANCE_MODIFIED


try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is expected in the runtime
    torch = None
    _HAS_TORCH = False


def _torch_device_report():
    if not _HAS_TORCH:
        return {"torch": False}
    rep = {"torch": True, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        rep["device_name"] = torch.cuda.get_device_name(0)
    return rep


def _scalar_reference(depth, mask, radius=1, sentinel=65535):
    """Independent slow scalar reference for values and provenance."""
    depth = np.asarray(depth)
    h, w = depth.shape
    is_u16 = depth.dtype == np.uint16
    d64 = depth.astype(np.float64)

    def valid_at(y, x):
        v = depth[y, x]
        if is_u16:
            ok = (v != 0) and (v != sentinel)
        else:
            f = d64[y, x]
            ok = (v != 0) and math.isfinite(f) and (f >= 0)
        if mask is not None and not bool(mask[y, x]):
            ok = False
        return ok

    out = np.zeros((h, w), dtype=np.float64)
    prov = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            measured = valid_at(y, x)
            best = None
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if not valid_at(ny, nx):
                        continue
                    v = float(depth[ny, nx])
                    if best is None or v < best:
                        best = v
            if best is None:
                out[y, x] = 0.0
                prov[y, x] = PROVENANCE_UNCHANGED if measured else PROVENANCE_NONE
            else:
                out[y, x] = best
                if measured:
                    prov[y, x] = (
                        PROVENANCE_UNCHANGED
                        if best == float(depth[y, x])
                        else PROVENANCE_MODIFIED
                    )
                else:
                    prov[y, x] = PROVENANCE_FILLED
    return out, prov


# --------------------------------------------------------------------------- #
# holes / border
# --------------------------------------------------------------------------- #
def test_hole_is_filled_by_3x3_support_numpy():
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    raw = depth.copy()
    res = min_valid_depth_dilate(depth)
    assert res.depth.dtype == np.float32
    expected_block = np.zeros_like(depth)
    expected_block[1:4, 1:4] = 1.0
    np.testing.assert_array_equal(res.depth, expected_block)
    assert int(res.valid_mask.sum()) == 9
    assert int(res.added_support.sum()) == 8
    assert int(res.changed_measured.sum()) == 0
    assert int(res.provenance[2, 2]) == PROVENANCE_UNCHANGED
    np.testing.assert_array_equal(depth, raw)  # immutability


def test_corner_border_clips_support_numpy():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[0, 0] = 2.0
    res = min_valid_depth_dilate(depth)
    expected = np.zeros((3, 3), dtype=np.float32)
    expected[0:2, 0:2] = 2.0
    np.testing.assert_array_equal(res.depth, expected)
    assert int(res.valid_mask.sum()) == 4
    assert int(res.added_support.sum()) == 3


# --------------------------------------------------------------------------- #
# closest (min) tie-breaking -- value-max is forbidden
# --------------------------------------------------------------------------- #
def test_closest_minimum_wins_over_farther_value():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 0] = 10.0
    depth[1, 2] = 5.0
    res = min_valid_depth_dilate(depth)
    assert res.depth[1, 1] == pytest.approx(5.0)  # closest, NOT max 10
    assert res.depth[1, 1] != pytest.approx(10.0)


def test_equal_distance_equal_value_tie_is_stable():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[0, 1] = 7.0
    depth[2, 1] = 7.0
    res = min_valid_depth_dilate(depth)
    assert res.depth[1, 1] == pytest.approx(7.0)


def test_measured_neighbour_can_become_smaller():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1] = 10.0
    depth[1, 2] = 5.0
    res = min_valid_depth_dilate(depth)
    # center measured pixel is now reduced by its closer neighbour
    assert res.depth[1, 1] == pytest.approx(5.0)
    assert int(res.provenance[1, 1]) == PROVENANCE_MODIFIED
    assert int(res.changed_measured.sum()) >= 1


# --------------------------------------------------------------------------- #
# invalid values 0 / 65535 / non-finite / negative
# --------------------------------------------------------------------------- #
def test_uint16_zero_and_sentinel_are_invalid():
    depth = np.zeros((3, 3), dtype=np.uint16)
    depth[1, 1] = 100
    depth[0, 0] = 65535
    res = min_valid_depth_dilate(depth)
    assert res.depth.dtype == np.uint16
    assert int((res.depth == 65535).sum()) == 0  # sentinel never propagates
    expected = np.zeros((3, 3), dtype=np.uint16)
    expected[0:3, 0:3] = 100
    np.testing.assert_array_equal(res.depth, expected)


def test_float_65535_is_a_normal_metric_value_not_a_sentinel():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1] = 65535.0  # metric float: NOT the uint16 mm sentinel
    res = min_valid_depth_dilate(depth)
    assert res.depth[1, 1] == pytest.approx(65535.0)
    assert int(res.valid_mask.sum()) == 9


def test_non_finite_and_negative_are_invalid():
    depth = np.full((3, 3), 0.0, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 2] = np.inf
    depth[2, 0] = -np.inf
    depth[2, 2] = -3.0
    depth[1, 1] = 4.0
    res = min_valid_depth_dilate(depth)
    np.testing.assert_array_equal(np.isfinite(res.depth), res.valid_mask)
    assert int(res.valid_mask.sum()) == 9
    assert res.depth[1, 1] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# empty / all-valid
# --------------------------------------------------------------------------- #
def test_all_empty_yields_zeros_and_none_provenance():
    depth = np.zeros((4, 4), dtype=np.uint16)
    res = min_valid_depth_dilate(depth)
    assert int(res.valid_mask.sum()) == 0
    assert int(res.added_support.sum()) == 0
    assert int(res.changed_measured.sum()) == 0
    assert int((res.provenance != PROVENANCE_NONE).sum()) == 0
    np.testing.assert_array_equal(res.depth, depth)


def test_all_valid_uniform_is_unchanged():
    depth = np.full((4, 5), 3.0, dtype=np.float32)
    res = min_valid_depth_dilate(depth)
    np.testing.assert_array_equal(res.depth, depth)
    assert int(res.valid_mask.sum()) == depth.size
    assert int(res.added_support.sum()) == 0
    assert int(res.changed_measured.sum()) == 0
    assert int((res.provenance != PROVENANCE_UNCHANGED).sum()) == 0


# --------------------------------------------------------------------------- #
# immutability + mask independence + counts
# --------------------------------------------------------------------------- #
def test_raw_depth_and_mask_are_immutable():
    depth = np.array([[0, 5, 0], [0, 9, 0], [0, 0, 0]], dtype=np.uint16)
    mask = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    depth_raw = depth.copy()
    mask_raw = mask.copy()
    res = min_valid_depth_dilate(depth, mask)
    np.testing.assert_array_equal(depth, depth_raw)
    np.testing.assert_array_equal(mask, mask_raw)
    assert int(res.depth.sum()) > 0


def test_mask_independently_rejects_otherwise_valid_depth():
    depth = np.zeros((3, 3), dtype=np.uint16)
    depth[1, 1] = 50
    mask = np.zeros((3, 3), dtype=bool)  # mask rejects the valid value
    res = min_valid_depth_dilate(depth, mask)
    assert int(res.valid_mask.sum()) == 0
    assert int((res.depth != 0).sum()) == 0


def test_mask_provenance_and_counts_are_consistent():
    rng = np.random.default_rng(0)
    depth = rng.integers(0, 200, size=(7, 8)).astype(np.uint16)
    depth[depth % 5 == 0] = 0
    mask = rng.random((7, 8)) > 0.3
    res = min_valid_depth_dilate(depth, mask)
    valid = res.valid_mask.astype(bool)
    added = res.added_support.astype(bool)
    changed = res.changed_measured.astype(bool)
    prov = res.provenance
    assert set(np.unique(prov).tolist()) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(added, prov == PROVENANCE_FILLED)
    np.testing.assert_array_equal(changed, prov == PROVENANCE_MODIFIED)
    np.testing.assert_array_equal(valid, prov != PROVENANCE_NONE)
    assert int(added.sum()) == int((prov == PROVENANCE_FILLED).sum())
    assert int(changed.sum()) == int((prov == PROVENANCE_MODIFIED).sum())


# --------------------------------------------------------------------------- #
# float mm -> m parity
# --------------------------------------------------------------------------- #
def test_uint16_mm_to_metric_m_parity():
    rng = np.random.default_rng(7)
    depth_mm = rng.choice(np.arange(1, 9000), size=(6, 6), replace=False).astype(np.uint16)
    depth_m = (depth_mm.astype(np.float64) * 0.001).astype(np.float64)
    res_mm = min_valid_depth_dilate(depth_mm)
    res_m = min_valid_depth_dilate(depth_m)
    np.testing.assert_allclose(
        res_mm.depth.astype(np.float64) * 0.001, res_m.depth, rtol=1e-9, atol=0.0
    )
    np.testing.assert_array_equal(res_mm.valid_mask, res_m.valid_mask)
    np.testing.assert_array_equal(res_mm.provenance, res_m.provenance)


# --------------------------------------------------------------------------- #
# random independent scalar/NumPy reference
# --------------------------------------------------------------------------- #
def test_matches_independent_scalar_reference_uint16():
    rng = np.random.default_rng(3)
    depth = rng.integers(0, 300, size=(7, 6)).astype(np.uint16)
    depth[depth % 4 == 0] = 0
    depth[0, 0] = 65535
    mask = rng.random((7, 6)) > 0.25
    res = min_valid_depth_dilate(depth, mask)
    ref_out, ref_prov = _scalar_reference(depth, mask)
    np.testing.assert_array_equal(res.depth.astype(np.float64), ref_out)
    np.testing.assert_array_equal(res.provenance, ref_prov)


def test_matches_independent_scalar_reference_float():
    rng = np.random.default_rng(11)
    depth = rng.random((5, 7), dtype=np.float64) * 9.0
    depth[depth < 1.0] = 0.0
    depth[4, 6] = np.nan
    res = min_valid_depth_dilate(depth)
    ref_out, ref_prov = _scalar_reference(depth, None)
    np.testing.assert_allclose(res.depth, ref_out, rtol=0, atol=0)
    np.testing.assert_array_equal(res.provenance, ref_prov)


# --------------------------------------------------------------------------- #
# torch CPU / CUDA + dtype/device parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_cpu_dtype_and_device_preserved():
    depth = torch.zeros((4, 4), dtype=torch.uint16)
    depth[1, 1] = 123
    res = min_valid_depth_dilate(depth)
    assert isinstance(res.depth, torch.Tensor)
    assert res.depth.dtype == torch.uint16
    assert res.depth.device == depth.device
    assert res.provenance.dtype == torch.uint8
    assert int(res.valid_mask.sum().item()) == 9


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_numpy_parity_cpu():
    rng = np.random.default_rng(5)
    depth_np = rng.integers(0, 500, size=(6, 5)).astype(np.uint16)
    depth_np[depth_np % 3 == 0] = 0
    res_np = min_valid_depth_dilate(depth_np)
    res_t = min_valid_depth_dilate(torch.from_numpy(depth_np))
    np.testing.assert_array_equal(res_t.depth.cpu().numpy(), res_np.depth)
    np.testing.assert_array_equal(res_t.provenance.cpu().numpy(), res_np.provenance)
    np.testing.assert_array_equal(res_t.valid_mask.cpu().numpy(), res_np.valid_mask)


@pytest.mark.skipif(
    not _HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available"
)
def test_torch_cuda_matches_cpu_and_records_device():
    rng = np.random.default_rng(9)
    depth_np = rng.integers(0, 800, size=(8, 6)).astype(np.uint16)
    depth_np[depth_np % 4 == 0] = 0
    res_cpu = min_valid_depth_dilate(torch.from_numpy(depth_np))
    dev = torch.device("cuda:0")
    res_cuda = min_valid_depth_dilate(torch.from_numpy(depth_np).to(dev))
    assert res_cuda.depth.device.type == "cuda"
    assert torch.cuda.get_device_name(0)  # actual CUDA device used
    np.testing.assert_array_equal(
        res_cuda.depth.cpu().numpy(), res_cpu.depth.numpy()
    )
    np.testing.assert_array_equal(
        res_cuda.provenance.cpu().numpy(), res_cpu.provenance.numpy()
    )




# --------------------------------------------------------------------------- #
# mask / depth fail-fast validation
#   Regression for the previously silent broadcast of a malformed mask:
#   ``min_valid_depth_dilate`` must validate mask shape, backend and device
#   against ``depth`` and raise ``ValueError``/``TypeError`` instead of
#   broadcasting.
# --------------------------------------------------------------------------- #
def test_numpy_mask_shape_mismatch_raises_value_error():
    depth = np.ones((3, 3), dtype=np.uint16)
    for bad_shape in [(1, 3), (3,), (3, 1), (1, 1), (4, 4)]:
        bad_mask = np.ones(bad_shape, dtype=bool)
        with pytest.raises(ValueError):
            min_valid_depth_dilate(depth, bad_mask)


def test_numpy_float_depth_mask_shape_mismatch_raises_value_error():
    depth = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        min_valid_depth_dilate(depth, np.ones((3,), dtype=bool))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_mask_shape_mismatch_raises_value_error():
    depth = torch.ones((3, 3), dtype=torch.uint16)
    for bad_shape in [(1, 3), (3,), (4, 4)]:
        bad_mask = torch.ones(bad_shape, dtype=torch.bool)
        with pytest.raises(ValueError):
            min_valid_depth_dilate(depth, bad_mask)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_depth_rejects_numpy_mask_backend_mix():
    depth = torch.ones((3, 3), dtype=torch.uint16)
    with pytest.raises(TypeError):
        min_valid_depth_dilate(depth, np.ones((3, 3), dtype=bool))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_numpy_depth_rejects_torch_mask_backend_mix():
    depth = np.ones((3, 3), dtype=np.uint16)
    with pytest.raises(TypeError):
        min_valid_depth_dilate(depth, torch.ones((3, 3), dtype=torch.bool))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_depth_rejects_python_list_mask():
    depth = torch.zeros((3, 3), dtype=torch.uint16)
    with pytest.raises(TypeError):
        min_valid_depth_dilate(depth, [[True, True, True]] * 3)


@pytest.mark.skipif(
    not _HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available"
)
def test_torch_mask_device_mismatch_raises_value_error():
    depth_cpu = torch.ones((3, 3), dtype=torch.uint16)
    mask_cuda = torch.ones((3, 3), dtype=torch.bool, device=torch.device("cuda:0"))
    with pytest.raises(ValueError):
        min_valid_depth_dilate(depth_cpu, mask_cuda)


@pytest.mark.skipif(
    not _HAS_TORCH or not torch.cuda.is_available(), reason="CUDA not available"
)
def test_torch_cuda_depth_rejects_cpu_mask_device_mismatch():
    depth_cuda = torch.ones((3, 3), dtype=torch.uint16, device=torch.device("cuda:0"))
    mask_cpu = torch.ones((3, 3), dtype=torch.bool)
    with pytest.raises(ValueError):
        min_valid_depth_dilate(depth_cuda, mask_cpu)


def test_numpy_unsupported_depth_dtype_rejected():
    for bad_dtype in (np.int32, np.int64, np.uint8, np.uint32, np.bool_):
        depth = np.ones((3, 3), dtype=bad_dtype)
        with pytest.raises(TypeError):
            min_valid_depth_dilate(depth)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_torch_unsupported_depth_dtype_rejected():
    for bad_dtype in (torch.int32, torch.int64, torch.uint8, torch.bool):
        depth = torch.ones((3, 3), dtype=bad_dtype)
        with pytest.raises(TypeError):
            min_valid_depth_dilate(depth)


def test_numpy_python_list_mask_is_accepted_and_shape_validated():
    depth = np.zeros((3, 3), dtype=np.uint16)
    depth[1, 1] = 42
    mask = [[False, False, False], [False, True, False], [False, False, False]]
    res = min_valid_depth_dilate(depth, mask)
    assert int(res.valid_mask.sum()) == 9
    np.testing.assert_array_equal(depth, np.array([[0, 0, 0], [0, 42, 0], [0, 0, 0]], dtype=np.uint16))
    with pytest.raises(ValueError):
        min_valid_depth_dilate(depth, [[True, True], [True, True]])


def test_device_report(capsys):
    with capsys.disabled():
        print("TORCH_DEVICE_REPORT", _torch_device_report())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
