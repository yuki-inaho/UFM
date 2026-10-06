"""
End-to-end ONNX tests for UFM.

Exports the real UFM checkpoint to ONNX, runs it through ONNX Runtime (CUDA
Execution Provider when available) and checks that the full pipeline
(preprocessing + core network + postprocessing) matches the PyTorch output.

Run with:

    uv sync
    uv run pytest -m e2e -v -s
"""

import pytest
import torch

from tests.test_e2e import BASE_MODEL_ID, PAIR_DIR, _load_rgb
from uniflowmatch.models.ufm import UniFlowMatchConfidence
from uniflowmatch.utils.onnx import UFMOnnxRunner, export_onnx

pytestmark = pytest.mark.e2e

ONNX_MODEL_FILENAME = "ufm_core_420x560.onnx"


@pytest.fixture(scope="module")
def onnx_export(tmp_path_factory):
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is not available in this onnxruntime build")

    model = UniFlowMatchConfidence.from_pretrained(BASE_MODEL_ID).eval()
    output_path = tmp_path_factory.mktemp("onnx") / ONNX_MODEL_FILENAME
    export_onnx(model, output_path)
    assert output_path.is_file()
    return model, output_path


def test_onnxruntime_gpu_matches_pytorch(onnx_export):
    model, onnx_path = onnx_export

    runner = UFMOnnxRunner(onnx_path)
    assert "CUDAExecutionProvider" in runner.providers, f"CUDA provider not active: {runner.providers}"

    source = _load_rgb(PAIR_DIR / "fire_academy_0.png")
    target = _load_rgb(PAIR_DIR / "fire_academy_1.png")

    with torch.no_grad():
        reference = model.predict_correspondences_batched(
            source_image=torch.from_numpy(source), target_image=torch.from_numpy(target)
        )

    output = runner.predict_correspondences(model, source, target)

    assert output.flow.shape == reference.flow.flow_output.shape
    assert output.covisibility.shape == reference.covisibility.mask.shape
    assert torch.isfinite(output.flow).all()

    valid = output.valid[0]
    assert valid.any(), "no valid output region"

    flow_diff = (output.flow[0].float() - reference.flow.flow_output[0].float())[:, valid].abs()
    cov_diff = (output.covisibility[0].float() - reference.covisibility.mask[0].float())[valid].abs()

    assert flow_diff.mean() < 0.05, f"flow mean abs diff too high: {flow_diff.mean():.4f}"
    assert flow_diff.max() < 1.0, f"flow max abs diff too high: {flow_diff.max():.4f}"
    assert cov_diff.mean() < 0.01, f"covisibility mean abs diff too high: {cov_diff.mean():.4f}"
    assert cov_diff.max() < 0.05, f"covisibility max abs diff too high: {cov_diff.max():.4f}"

    print(
        f"\n[e2e-onnx] providers={runner.providers} "
        f"flow mean/max diff={flow_diff.mean():.5f}/{flow_diff.max():.5f} "
        f"covisibility mean/max diff={cov_diff.mean():.5f}/{cov_diff.max():.5f}"
    )
