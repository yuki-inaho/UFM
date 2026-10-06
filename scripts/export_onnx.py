#!/usr/bin/env python3
"""
Export a UFM checkpoint to ONNX and optionally verify it with ONNX Runtime.

Examples:
    uv run python scripts/export_onnx.py --model infinity1096/UFM-Base --output exports/ufm_base_420x560.onnx
    uv run python scripts/export_onnx.py --model infinity1096/UFM-Refine --verify
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence
from uniflowmatch.utils.onnx import DEFAULT_INPUT_SHAPE, UFMOnnxRunner, export_onnx


def load_model(model_id: str):
    if "refine" in model_id:
        return UniFlowMatchClassificationRefinement.from_pretrained(model_id)
    return UniFlowMatchConfidence.from_pretrained(model_id)


def load_rgb(path: str):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def verify(model, onnx_path: Path, source_path: str, target_path: str):
    runner = UFMOnnxRunner(onnx_path)
    print(f"ONNX Runtime providers: {runner.providers}")

    source = load_rgb(source_path)
    target = load_rgb(target_path)

    with torch.no_grad():
        reference = model.predict_correspondences_batched(
            source_image=torch.from_numpy(source), target_image=torch.from_numpy(target)
        )

    output = runner.predict_correspondences(model, source, target)
    valid = output.valid[0]

    flow_diff = (output.flow[0] - reference.flow.flow_output[0])[:, valid].abs()
    cov_diff = (output.covisibility[0] - reference.covisibility.mask[0]).abs()[valid]

    print(f"flow shape: {tuple(output.flow.shape)}, covisibility shape: {tuple(output.covisibility.shape)}")
    print(f"flow  | diff vs PyTorch: max={flow_diff.max():.4f} mean={flow_diff.mean():.4f}")
    print(f"covis | diff vs PyTorch: max={cov_diff.max():.4f} mean={cov_diff.mean():.4f}")

    if not (flow_diff.max() < 1.0 and flow_diff.mean() < 0.05 and cov_diff.max() < 0.05):
        print("Verification FAILED: ONNX outputs deviate from PyTorch")
        return 1

    print("Verification PASSED: ONNX Runtime matches the PyTorch pipeline")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Export a UFM checkpoint to ONNX")
    parser.add_argument("--model", default="infinity1096/UFM-Base", help="HuggingFace model id or local checkpoint")
    parser.add_argument("--output", "-o", default="exports/ufm_base_420x560.onnx", help="Output .onnx path")
    parser.add_argument("--height", type=int, default=DEFAULT_INPUT_SHAPE[0], help="Static input height")
    parser.add_argument("--width", type=int, default=DEFAULT_INPUT_SHAPE[1], help="Static input width")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Compare ONNX Runtime outputs against PyTorch")
    parser.add_argument("--source", default="examples/image_pairs/fire_academy_0.png")
    parser.add_argument("--target", default="examples/image_pairs/fire_academy_1.png")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = load_model(args.model).eval()

    output_path = export_onnx(model, args.output, (args.height, args.width), args.opset)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported ONNX model to {output_path} ({size_mb:.1f} MB plus external data if present)")

    if args.verify:
        return verify(model, output_path, args.source, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
