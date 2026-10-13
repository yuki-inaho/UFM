"""End-to-end ONNX tests for the released UFM checkpoint.

Exports the verified release model to ONNX, optionally simplifies the graph with
onnxsim, runs both graphs through ONNX Runtime (CUDA execution provider when
available) and checks the full pipeline (preprocessing + core network +
postprocessing) against the PyTorch output.

Run with:

    uv run --no-sync pytest -m e2e tests/test_onnx_e2e.py -v -s
"""

import pytest
import torch

from tests.e2e_common import FIRE_PAIR, ONNX_INPUT_SHAPE, PAIR_DIR, RELEASE_INFERENCE_RESOLUTION, load_rgb
from uniflowmatch.utils.onnx import UFMOnnxRunner, export_onnx, onnx_size_report, simplify_onnx

pytestmark = pytest.mark.e2e

FLOW_MEAN_ATOL = 0.05
FLOW_MAX_ATOL = 1.0
COV_MEAN_ATOL = 0.01
COV_MAX_ATOL = 0.05


@pytest.fixture(scope="module")
def exported_onnx(release_model, tmp_path_factory):
    """Export (and simplify) the release model once for the whole module."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is not available in this onnxruntime build")

    model, _info = release_model
    model = model.eval()
    model.inference_resolution = RELEASE_INFERENCE_RESOLUTION

    export_dir = tmp_path_factory.mktemp("onnx")
    onnx_path = export_dir / "ufm_release_core.onnx"
    export_onnx(model, onnx_path, input_shape=ONNX_INPUT_SHAPE)

    report = onnx_size_report(onnx_path)
    assert report["nodes"] > 0
    print(
        f"\n[e2e-onnx] exported {onnx_path.name}: {report['total_megabytes']:.1f} MB, "
        f"{report['nodes']} nodes, {report['initializers']} initializers"
    )

    simplified_path = simplify_onnx(onnx_path)
    simplified_report = onnx_size_report(simplified_path)
    assert simplified_report["nodes"] <= report["nodes"]
    print(
        f"[e2e-onnx] simplified: {simplified_report['total_megabytes']:.1f} MB, "
        f"{simplified_report['nodes']} nodes, {simplified_report['initializers']} initializers"
    )

    return model, onnx_path, simplified_path


def _assert_parity(model, onnx_path):
    runner = UFMOnnxRunner(onnx_path)
    assert "CUDAExecutionProvider" in runner.providers, f"CUDA provider not active: {runner.providers}"

    source = load_rgb(PAIR_DIR / FIRE_PAIR[0])
    target = load_rgb(PAIR_DIR / FIRE_PAIR[1])

    with torch.no_grad():
        reference = model.predict_correspondences_batched(
            source_image=torch.from_numpy(source), target_image=torch.from_numpy(target)
        )

    output = runner.predict_correspondences(model, source, target)
    valid = output.valid[0]

    assert output.flow.shape == reference.flow.flow_output.shape
    assert output.covisibility.shape == reference.covisibility.mask.shape
    assert torch.isfinite(output.flow).all()
    assert valid.any(), "no valid output region"

    flow_diff = (output.flow[0].float() - reference.flow.flow_output[0].float().cpu())[:, valid].abs()
    cov_diff = (output.covisibility[0].float() - reference.covisibility.mask[0].float().cpu())[valid].abs()

    assert flow_diff.mean() < FLOW_MEAN_ATOL, f"flow mean abs diff too high: {flow_diff.mean():.4f}"
    assert flow_diff.max() < FLOW_MAX_ATOL, f"flow max abs diff too high: {flow_diff.max():.4f}"
    assert cov_diff.mean() < COV_MEAN_ATOL, f"covisibility mean abs diff too high: {cov_diff.mean():.4f}"
    assert cov_diff.max() < COV_MAX_ATOL, f"covisibility max abs diff too high: {cov_diff.max():.4f}"

    print(
        f"[e2e-onnx] {onnx_path.name} providers={runner.providers} "
        f"flow mean/max diff={flow_diff.mean():.5f}/{flow_diff.max():.5f} "
        f"covisibility mean/max diff={cov_diff.mean():.5f}/{cov_diff.max():.5f}"
    )


def test_onnxruntime_gpu_matches_pytorch(exported_onnx):
    model, onnx_path, _simplified_path = exported_onnx
    _assert_parity(model, onnx_path)


def test_simplified_onnx_still_matches_pytorch(exported_onnx):
    model, _onnx_path, simplified_path = exported_onnx
    _assert_parity(model, simplified_path)
