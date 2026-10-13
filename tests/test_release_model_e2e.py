"""End-to-end test of the published RGB-only release checkpoint.

Downloads/verifies the public release package, strict-loads the checkpoint, and
runs the full correspondence pipeline (including the "CPU uint8 images + CUDA
model" usage the release model card documents) on the bundled example pair.

Run with:

    uv run --no-sync pytest -m e2e tests/test_release_model_e2e.py -v -s
"""

import time

import pytest
import torch

from tests.e2e_common import FIRE_PAIR, PAIR_DIR, RELEASE_INFERENCE_RESOLUTION, device, load_rgb

pytestmark = pytest.mark.e2e


def test_release_checkpoint_loads_strictly(release_model):
    model, info = release_model

    assert not model.training
    assert model.depth_fusion is None, "the released RGB-only package must not fuse depth"

    load_report = info["load_report"]
    assert load_report["unexpected"] == []
    assert load_report["shape_mismatch"] == []
    assert load_report["loaded"] == 629, f"unexpected tensor count: {load_report['loaded']}"
    assert info["weights_sha256"] == "d9aba61da7fb7f8465924cea10cf201a65eb1db97926308297942a6a7ad5ccfb"

    num_params = sum(p.numel() for p in model.parameters())
    assert 0.3e9 < num_params < 0.6e9, f"unexpected parameter count: {num_params}"

    # the released model must be fp32 end to end
    assert all(p.dtype == torch.float32 for p in model.parameters())

    print(
        f"\n[e2e-release] loaded {load_report['loaded']} tensors "
        f"({num_params / 1e6:.1f}M params) from {info['repo']}@{info['tag']}"
    )


def test_release_inference_from_cpu_uint8_inputs(release_model):
    """The documented model-card call: CPU uint8 tensors, model on the GPU."""
    model, _info = release_model
    target_device = device()

    model.inference_resolution = RELEASE_INFERENCE_RESOLUTION
    model = model.eval().to(target_device)

    source = load_rgb(PAIR_DIR / FIRE_PAIR[0])
    target = load_rgb(PAIR_DIR / FIRE_PAIR[1])
    height, width = source.shape[:2]

    # deliberately keep the images on the CPU: predict_* must follow the model device
    source_tensor = torch.from_numpy(source)
    target_tensor = torch.from_numpy(target)

    if target_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.predict_correspondences_batched(source_image=source_tensor, target_image=target_tensor)
    if target_device.type == "cuda":
        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
    else:
        peak_gib = 0.0
    elapsed = time.perf_counter() - start

    flow = result.flow.flow_output
    covisibility = result.covisibility.mask

    assert flow.shape == (1, 2, height, width), f"unexpected flow shape: {flow.shape}"
    assert covisibility.shape == (1, height, width), f"unexpected covisibility shape: {covisibility.shape}"
    assert torch.isfinite(flow).all(), "flow contains non-finite values"
    assert torch.isfinite(covisibility).all(), "covisibility contains non-finite values"
    assert covisibility.min() >= -1e-4 and covisibility.max() <= 1 + 1e-4

    covisible_fraction = float((covisibility > 0.5).float().mean())
    assert 0.05 < covisible_fraction < 0.999, f"unexpected covisible fraction: {covisible_fraction:.3f}"

    print(
        f"\n[e2e-release] device={target_device} inference={elapsed:.2f}s "
        f"peak_gpu={peak_gib:.2f} GiB covisible_fraction={covisible_fraction:.3f}"
    )

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
