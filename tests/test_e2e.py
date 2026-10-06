"""
End-to-end tests for UFM.

These tests download the real UFM checkpoints from Hugging Face and run the
full pipeline (image loading -> normalization -> model forward -> flow and
covisibility outputs -> visualization / CLI) on the bundled example pairs.

Run with:

    uv sync
    uv run pytest -m e2e -v -s
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from uniflowmatch.models.ufm import UniFlowMatchConfidence
from uniflowmatch.utils.viz import warp_image_with_flow

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIR_DIR = REPO_ROOT / "examples" / "image_pairs"
BASE_MODEL_ID = os.environ.get("UFM_E2E_BASE_MODEL", "infinity1096/UFM-Base")
FIRE_PAIR = ("fire_academy_0.png", "fire_academy_1.png")

pytestmark = pytest.mark.e2e


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None, f"Could not read image: {path}"
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def base_model() -> UniFlowMatchConfidence:
    model = UniFlowMatchConfidence.from_pretrained(BASE_MODEL_ID).eval().to(_device())
    yield model
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_real_checkpoint_loads(base_model):
    assert isinstance(base_model, UniFlowMatchConfidence)
    assert not base_model.training

    num_params = sum(p.numel() for p in base_model.parameters())
    assert 0.3e9 < num_params < 0.6e9, f"unexpected parameter count: {num_params}"

    param_devices = {p.device.type for p in base_model.parameters()}
    assert param_devices == {_device().type}, f"model not fully on {_device()}: {param_devices}"


def test_end_to_end_flow_and_covisibility(base_model, tmp_path):
    source = _load_rgb(PAIR_DIR / FIRE_PAIR[0])
    target = _load_rgb(PAIR_DIR / FIRE_PAIR[1])
    height, width = source.shape[:2]

    source_t = torch.from_numpy(source).to(_device())
    target_t = torch.from_numpy(target).to(_device())

    start = time.time()
    with torch.no_grad():
        result = base_model.predict_correspondences_batched(source_image=source_t, target_image=target_t)
    if _device().type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    flow = result.flow.flow_output
    covisibility = result.covisibility.mask

    assert flow.shape == (1, 2, height, width), f"unexpected flow shape: {flow.shape}"
    assert covisibility.shape == (1, height, width), f"unexpected covisibility shape: {covisibility.shape}"
    assert torch.isfinite(flow).all(), "flow contains non-finite values"
    assert torch.isfinite(covisibility).all(), "covisibility contains non-finite values"
    assert covisibility.min() >= -1e-4 and covisibility.max() <= 1 + 1e-4, "covisibility outside [0, 1]"

    flow_np = flow[0].float().cpu().numpy()
    cov_np = covisibility[0].float().cpu().numpy()
    cov_mask = cov_np > 0.5
    covisible_fraction = float(cov_mask.mean())
    mean_flow_magnitude = float(np.linalg.norm(flow_np, axis=0).mean())

    assert 0.05 < covisible_fraction < 0.99, f"unexpected covisible fraction: {covisible_fraction:.3f}"
    assert 5.0 < mean_flow_magnitude < 1000.0, f"unexpected mean flow magnitude: {mean_flow_magnitude:.2f}"

    warped = warp_image_with_flow(
        source.astype(np.float32), None, target.astype(np.float32), flow_np.transpose(1, 2, 0)
    )
    warped = np.clip(warped, 0, 255)
    photometric_mae = float(np.abs(warped - source.astype(np.float32))[cov_mask].mean())
    assert photometric_mae < 70.0, f"photometric consistency too low, MAE={photometric_mae:.2f}"

    flow_vis = cv2.applyColorMap(
        np.clip(np.linalg.norm(flow_np, axis=0) / max(mean_flow_magnitude * 2, 1e-6) * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    cv2.imwrite(str(tmp_path / "flow_magnitude.png"), flow_vis)
    cv2.imwrite(str(tmp_path / "covisibility_mask.png"), (cov_mask * 255).astype(np.uint8))
    cv2.imwrite(str(tmp_path / "warped_source.png"), warped.astype(np.uint8))

    print(
        f"\n[e2e] device={_device()} inference={elapsed:.2f}s "
        f"covisible_fraction={covisible_fraction:.3f} "
        f"mean_flow_magnitude={mean_flow_magnitude:.2f} "
        f"photometric_mae={photometric_mae:.2f}"
    )


def test_cli_inference_end_to_end(tmp_path):
    source = PAIR_DIR / FIRE_PAIR[0]
    target = PAIR_DIR / FIRE_PAIR[1]

    ufm_bin = shutil.which("ufm")
    if ufm_bin:
        cmd = [ufm_bin, "infer", str(source), str(target), "-o", str(tmp_path)]
    else:
        cmd = [sys.executable, "-m", "uniflowmatch.cli", "infer", str(source), str(target), "-o", str(tmp_path)]

    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800)
    assert completed.returncode == 0, f"CLI failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"

    for name in ["flow_visualization.png", "covisibility_mask.png", "warped_source.png"]:
        output = tmp_path / name
        assert output.is_file(), f"missing CLI output: {name}"
        assert output.stat().st_size > 1024, f"CLI output looks empty: {name}"

    print(f"\n[e2e] CLI outputs: {sorted(p.name for p in tmp_path.iterdir())}")
