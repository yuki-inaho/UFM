#!/usr/bin/env python3
"""
Run the published RGB-only UFM release end to end on a single image pair.

The release assets (public GitHub release) are downloaded and verified on first
use, the strict checkpoint is loaded, and one inference pass is executed either
through PyTorch or through ONNX Runtime (CUDA provider when available).  Flow /
covisibility / warped-image visualizations and a machine-readable run report are
written to ``--output``.

Examples:
    # PyTorch backend on the GPU
    python scripts/release_infer.py -o outputs/release_torch

    # ONNX Runtime backend (requires scripts/export_onnx.py output)
    python scripts/release_infer.py --backend onnx --onnx-model exports/ufm_core.onnx -o outputs/release_onnx
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from uniflowmatch.models.release_assets import (
    DEFAULT_RELEASE_REPO,
    DEFAULT_RELEASE_TAG,
    load_release_model,
)
from uniflowmatch.utils.viz import warp_image_with_flow

DEFAULT_SOURCE = "examples/image_pairs/fire_academy_0.png"
DEFAULT_TARGET = "examples/image_pairs/fire_academy_1.png"


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_outputs(output_dir: Path, source: np.ndarray, target: np.ndarray, flow: np.ndarray, covisibility: np.ndarray):
    """Save the same three visualizations the CLI demo writes."""
    import flow_vis

    output_dir.mkdir(parents=True, exist_ok=True)
    flow_vis_img = flow_vis.flow_to_color(flow.transpose(1, 2, 0))
    cv2.imwrite(str(output_dir / "flow_visualization.png"), cv2.cvtColor(flow_vis_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "covisibility_mask.png"), (np.clip(covisibility, 0, 1) * 255).astype(np.uint8))

    warped = warp_image_with_flow(source, None, target, flow.transpose(1, 2, 0))
    warped = covisibility[..., None] * warped + (1 - covisibility[..., None]) * 255 * np.ones_like(warped)
    cv2.imwrite(str(output_dir / "warped_source.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))


def run_torch_backend(model, source: np.ndarray, target: np.ndarray, device: str):
    model = model.eval().to(device)
    source_tensor = torch.from_numpy(source)
    target_tensor = torch.from_numpy(target)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.predict_correspondences_batched(source_image=source_tensor, target_image=target_tensor)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
    else:
        peak_gib = None
    elapsed = time.perf_counter() - start

    flow = result.flow.flow_output[0].float().cpu().numpy()
    covisibility = result.covisibility.mask[0].float().cpu().numpy()
    return flow, covisibility, {"elapsed_s": elapsed, "peak_gpu_gib": peak_gib, "device": device}


def run_onnx_backend(model, onnx_path: Path, source: np.ndarray, target: np.ndarray):
    from uniflowmatch.utils.onnx import UFMOnnxRunner

    runner = UFMOnnxRunner(onnx_path)
    print(f"[onnx] providers: {runner.providers}")

    start = time.perf_counter()
    output = runner.predict_correspondences(model, source, target)
    elapsed = time.perf_counter() - start

    flow = output.flow[0].float().numpy()
    covisibility = output.covisibility[0].float().numpy()
    return (
        flow,
        covisibility,
        {
            "elapsed_s": elapsed,
            "providers": runner.providers,
            "onnx_model": str(onnx_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the published UFM RGB-only release on one image pair")
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG, help="Public release tag")
    parser.add_argument("--release-repo", default=DEFAULT_RELEASE_REPO, help="GitHub repository of the release")
    parser.add_argument("--release-dir", default=None, help="Local release asset directory (default: cache dir)")
    parser.add_argument(
        "--release-cache-dir",
        default="~/.cache/ufm_release",
        help="Cache root for downloaded release assets (one subdirectory per tag)",
    )

    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch", help="Inference backend")
    parser.add_argument("--onnx-model", default=None, help="Exported .onnx core model (required for --backend onnx)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for the torch backend"
    )

    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source image path")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target image path")
    parser.add_argument("--output", "-o", default="outputs/release_infer", help="Output directory")
    parser.add_argument("--width", type=int, default=None, help="Inference width override (default: release config)")
    parser.add_argument("--height", type=int, default=None, help="Inference height override (default: release config)")
    args = parser.parse_args()

    if args.backend == "onnx" and not args.onnx_model:
        parser.error("--backend onnx requires --onnx-model (export it with scripts/export_onnx.py)")

    cache_dir = (
        Path(args.release_dir).expanduser()
        if args.release_dir
        else (Path(args.release_cache_dir).expanduser() / args.release_tag)
    )
    print(f"[release] assets: {cache_dir}")
    model, info = load_release_model(cache_dir, tag=args.release_tag, repo=args.release_repo, device="cpu")
    print(
        f"[release] {info['repo']}@{info['tag']} weights_sha256={info['weights_sha256']} "
        f"({info['weights_bytes'] / 1e9:.3f} GB), {info['load_report']['loaded']} tensors loaded strictly"
    )

    if args.width and args.height:
        model.inference_resolution = (int(args.width), int(args.height))
    print(f"[release] inference_resolution(W,H)={model.inference_resolution}")

    source = load_rgb(args.source)
    target = load_rgb(args.target)
    print(f"[input] source={args.source} {source.shape}, target={args.target} {target.shape}")

    if args.backend == "torch":
        flow, covisibility, run_info = run_torch_backend(model, source, target, args.device)
    else:
        flow, covisibility, run_info = run_onnx_backend(model, Path(args.onnx_model), source, target)
    run_info["backend"] = args.backend

    output_dir = Path(args.output)
    save_outputs(output_dir, source, target, flow, covisibility)

    report = {
        "release": {
            "repo": info["repo"],
            "tag": info["tag"],
            "weights": info["weights"],
            "weights_sha256": info["weights_sha256"],
            "weights_bytes": info["weights_bytes"],
            "load_report": {
                "loaded": info["load_report"]["loaded"],
                "benign_aliases": sorted(info["load_report"]["benign_aliases"]),
                "unexpected": info["load_report"]["unexpected"],
                "shape_mismatch": info["load_report"]["shape_mismatch"],
            },
        },
        "inference": {
            **run_info,
            "inference_resolution": model.inference_resolution,
            "source": args.source,
            "target": args.target,
            "flow_shape": list(flow.shape),
            "covisibility_shape": list(covisibility.shape),
            "flow_finite": bool(np.isfinite(flow).all()),
            "covisible_fraction": float((covisibility > 0.5).mean()),
        },
        "outputs": ["flow_visualization.png", "covisibility_mask.png", "warped_source.png"],
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(
        f"[run] backend={args.backend} time={run_info['elapsed_s']:.2f}s "
        f"peak_gpu={run_info.get('peak_gpu_gib')} GiB "
        f"flow={tuple(flow.shape)} cov>0.5={(covisibility > 0.5).mean():.3f}"
    )
    print(f"[run] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
