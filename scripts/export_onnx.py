#!/usr/bin/env python3
"""
Export a UFM checkpoint to ONNX, optionally simplify it with onnxsim and verify
it with ONNX Runtime (CUDA provider when available).

Examples:
    # Public research-prerelease package (assets are downloaded on first use)
    python scripts/export_onnx.py --release-tag rgb-only-off-20261013 --simplify --verify

    # Hugging Face checkpoint
    python scripts/export_onnx.py --model infinity1096/UFM-Base --simplify --verify

    # Existing local release assets, custom resolution and output path
    python scripts/export_onnx.py \
        --release-dir ~/.cache/ufm_release/rgb-only-off-20261013 \
        --height 336 --width 448 -o exports/ufm_rgb_only_336x448.onnx --simplify

The exported graph is the fixed-resolution core network (encoder + info sharing
+ prediction heads); normalization/resizing and the mapping of the predictions
back to the original image space stay in Python, see ``uniflowmatch.utils.onnx``.
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

from uniflowmatch.models.release_assets import DEFAULT_RELEASE_REPO, DEFAULT_RELEASE_TAG, load_release_model
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence
from uniflowmatch.utils.onnx import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_OPSET_VERSION,
    UFMOnnxRunner,
    export_onnx,
    onnx_size_report,
)

DEFAULT_SOURCE = "examples/image_pairs/fire_academy_0.png"
DEFAULT_TARGET = "examples/image_pairs/fire_academy_1.png"

FLOW_MEAN_ATOL = 0.05
FLOW_MAX_ATOL = 1.0
COV_MEAN_ATOL = 0.01
COV_MAX_ATOL = 0.05


def load_hf_model(model_id: str):
    if "refine" in model_id:
        return UniFlowMatchClassificationRefinement.from_pretrained(model_id)
    return UniFlowMatchConfidence.from_pretrained(model_id)


def load_rgb(path: str):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resolve_model(args):
    """Return ``(model, identity)`` for the requested checkpoint source."""
    if args.release_tag or args.release_dir:
        cache_dir = args.release_dir or (Path(args.release_cache_dir).expanduser() / args.release_tag)
        release_tag = args.release_tag or Path(cache_dir).name
        model, info = load_release_model(cache_dir, tag=release_tag, repo=args.release_repo, device="cpu")
        print(f"[release] {info['repo']}@{info['tag']} sha256={info['weights_sha256']}")
        print(
            f"[release] load report: {info['load_report']['loaded']} tensors, "
            f"benign aliases: {len(info['load_report']['benign_aliases'])}"
        )
        return model, f"release:{info['tag']}"

    model_id = args.model
    model = load_hf_model(model_id).eval()
    print(f"[hf] loaded {model_id}")
    return model, f"hf:{model_id}"


def resolve_input_shape(args, model) -> tuple:
    if args.height and args.width:
        return (args.height, args.width)
    resolution = model.inference_resolution
    if resolution:
        width, height = resolution[0]
        return (int(height), int(width))
    return DEFAULT_INPUT_SHAPE


def verify(model, onnx_path: Path, source_path: str, target_path: str, device: str) -> int:
    runner = UFMOnnxRunner(onnx_path)
    print(f"[verify] ONNX Runtime providers: {runner.providers}")
    if "CUDAExecutionProvider" in runner.providers:
        print("[verify] CUDA execution provider is active")

    source = load_rgb(source_path)
    target = load_rgb(target_path)

    reference_model = model.to(device).eval()
    with torch.no_grad():
        reference = reference_model.predict_correspondences_batched(
            source_image=torch.from_numpy(source), target_image=torch.from_numpy(target)
        )

    output = runner.predict_correspondences(reference_model, source, target)
    valid = output.valid[0]

    flow_diff = (output.flow[0].float() - reference.flow.flow_output[0].float().cpu())[:, valid].abs()
    cov_diff = (output.covisibility[0].float() - reference.covisibility.mask[0].float().cpu())[valid].abs()

    print(f"[verify] flow shape: {tuple(output.flow.shape)}, covisibility shape: {tuple(output.covisibility.shape)}")
    print(f"[verify] flow  | diff vs PyTorch: max={flow_diff.max():.6f} mean={flow_diff.mean():.6f}")
    print(f"[verify] covis | diff vs PyTorch: max={cov_diff.max():.6f} mean={cov_diff.mean():.6f}")

    if not (
        flow_diff.max() < FLOW_MAX_ATOL
        and flow_diff.mean() < FLOW_MEAN_ATOL
        and cov_diff.max() < COV_MAX_ATOL
        and cov_diff.mean() < COV_MEAN_ATOL
    ):
        print("[verify] FAILED: ONNX outputs deviate from PyTorch")
        return 1

    print("[verify] PASSED: ONNX Runtime matches the PyTorch pipeline")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a UFM checkpoint to ONNX")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--model", default=None, help="HuggingFace model id (e.g. infinity1096/UFM-Base)")
    source_group.add_argument("--release-dir", default=None, help="Local release asset directory")
    parser.add_argument(
        "--release-tag",
        default=None,
        help=f"Public release tag (default when no source is given: " f"{DEFAULT_RELEASE_TAG})",
    )
    parser.add_argument("--release-repo", default=DEFAULT_RELEASE_REPO, help="GitHub repository of the release")
    parser.add_argument(
        "--release-cache-dir",
        default="~/.cache/ufm_release",
        help="Cache root for downloaded release assets (one subdirectory per tag)",
    )

    parser.add_argument("--output", "-o", default="exports/ufm_core.onnx", help="Output .onnx path")
    parser.add_argument("--height", type=int, default=None, help="Static input height (default: model resolution)")
    parser.add_argument("--width", type=int, default=None, help="Static input width (default: model resolution)")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION, help="ONNX opset version")

    parser.add_argument("--simplify", action="store_true", help="Optimize the graph size with onnxsim")
    parser.add_argument(
        "--simplify-output", default=None, help="Simplified .onnx path (default: <output>_simplified.onnx)"
    )
    parser.add_argument(
        "--simplify-check-n",
        type=int,
        default=0,
        help="onnxsim verification evaluations through ONNX Runtime (0 = skip)",
    )

    parser.add_argument("--verify", action="store_true", help="Compare ONNX Runtime outputs against PyTorch")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source image for --verify")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target image for --verify")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the PyTorch reference during --verify",
    )
    args = parser.parse_args()

    if args.model is None and args.release_dir is None and args.release_tag is None:
        args.release_tag = DEFAULT_RELEASE_TAG

    model, identity = resolve_model(args)
    model = model.eval()

    if args.height and args.width:
        # keep the runtime preprocessing (image scaler) consistent with the export
        model.inference_resolution = (int(args.width), int(args.height))
    input_shape = resolve_input_shape(args, model)
    print(f"[export] checkpoint={identity} input_shape(H,W)={input_shape} opset={args.opset}")

    output_path = Path(args.output)
    export_onnx(model, output_path, input_shape=input_shape, opset_version=args.opset)
    report = onnx_size_report(output_path)
    print(
        f"[export] wrote {output_path} ({report['total_megabytes']:.1f} MB total"
        f"{', external data' if report['external_bytes'] else ''}, "
        f"{report['nodes']} nodes, {report['initializers']} initializers)"
    )

    verify_path = output_path
    if args.simplify:
        from uniflowmatch.utils.onnx import simplify_onnx

        simplified_path = Path(args.simplify_output) if args.simplify_output else None
        simplify_onnx(output_path, simplified_path, check_n=args.simplify_check_n)
        verify_path = (
            simplified_path
            if simplified_path is not None
            else output_path.with_name(output_path.stem + "_simplified" + output_path.suffix)
        )
        simplified = onnx_size_report(verify_path)
        delta_mb = simplified["total_megabytes"] - report["total_megabytes"]
        print(
            f"[simplify] wrote {simplified['path']} ({simplified['total_megabytes']:.1f} MB total, "
            f"{simplified['nodes']} nodes, {simplified['initializers']} initializers)"
        )
        print(
            f"[simplify] size delta: {delta_mb:+.1f} MB "
            f"({(simplified['total_bytes'] / report['total_bytes'] - 1) * 100:+.2f}%)"
        )

    if args.verify:
        return verify(model, verify_path, args.source, args.target, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
