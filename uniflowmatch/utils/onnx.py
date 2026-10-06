"""
ONNX export and ONNX Runtime inference helpers for UFM core networks.

The exported graph covers the fixed-resolution core network (encoder + info
sharing + prediction heads) and returns the flow and covisibility tensors in the
scaled model space (420x560 for the 560-resolution checkpoints). Preprocessing
(normalization and resizing) and postprocessing (mapping the predictions back to
the original image space) are handled in Python so that the ONNX graph stays
static and portable.
"""

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from uniflowmatch.utils.flow_resizing import unmap_predicted_channels, unmap_predicted_flow

DEFAULT_INPUT_SHAPE: Tuple[int, int] = (420, 560)  # (height, width) of the 560-resolution checkpoints


class UFMOnnxWrapper(nn.Module):
    """Thin wrapper exposing the UFM core network as (source, target) -> (flow, covisibility)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.data_norm_type = model.encoder.data_norm_type

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor):
        result = self.model(
            {"img": source_image, "symmetrized": False, "data_norm_type": self.data_norm_type},
            {"img": target_image, "symmetrized": False, "data_norm_type": self.data_norm_type},
        )
        return result.flow.flow_output, result.covisibility.mask


def export_onnx(
    model,
    output_path: Path,
    input_shape: Tuple[int, int] = DEFAULT_INPUT_SHAPE,
    opset_version: int = 18,
) -> Path:
    """
    Export the UFM core network to ONNX with fixed input shape.

    Args:
        model: A UFM model instance (e.g. ``UniFlowMatchConfidence``).
        output_path: Destination ``.onnx`` file. Weights larger than 2 GB are stored next to it as external data.
        input_shape: ``(height, width)`` of the model input used for the static export.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    height, width = input_shape
    wrapper = UFMOnnxWrapper(model).eval()
    dummy_source = torch.zeros(1, 3, height, width)
    dummy_target = torch.zeros(1, 3, height, width)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["source_image", "target_image"],
        output_names=["flow", "covisibility"],
        opset_version=opset_version,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = True

    torch.onnx.export(wrapper, (dummy_source, dummy_target), str(output_path), **export_kwargs)
    return output_path


@dataclass
class UFMOnnxOutput:
    """Outputs of :meth:`UFMOnnxRunner.predict_correspondences`, in the original image space."""

    flow: torch.Tensor  # (1, 2, H, W)
    covisibility: torch.Tensor  # (1, H, W)
    valid: torch.Tensor  # (1, H, W) bool


class UFMOnnxRunner:
    """Runs an exported UFM ONNX model with the same preprocessing/postprocessing as the PyTorch pipeline."""

    def __init__(self, onnx_path: Path, providers: Optional[Sequence[str]] = None):
        import onnxruntime as ort

        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            raise RuntimeError(f"No usable ONNX Runtime providers found, available: {ort.get_available_providers()}")

        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers))

    @property
    def providers(self) -> List[str]:
        return self.session.get_providers()

    def predict_correspondences(self, model, source_image: np.ndarray, target_image: np.ndarray) -> UFMOnnxOutput:
        """
        Predict flow and covisibility for a single uint8 RGB image pair.

        Args:
            model: The UFM model the ONNX graph was exported from. Used for its image scaler and normalization.
            source_image: ``(H, W, 3)`` uint8 RGB source image.
            target_image: ``(H, W, 3)`` uint8 RGB target image.

        Returns:
            UFMOnnxOutput with flow and covisibility mapped back to the original image space.
        """
        assert source_image.ndim == 3 and source_image.shape[-1] == 3, "source_image must be (H, W, 3)"
        assert target_image.ndim == 3 and target_image.shape[-1] == 3, "target_image must be (H, W, 3)"

        data_norm_type = model.encoder.data_norm_type
        normalization = IMAGE_NORMALIZATION_DICT[data_norm_type]
        mean = normalization.mean.view(1, 3, 1, 1)
        std = normalization.std.view(1, 3, 1, 1)

        source = torch.from_numpy(np.ascontiguousarray(source_image)).float()[None].permute(0, 3, 1, 2)
        target = torch.from_numpy(np.ascontiguousarray(target_image)).float()[None].permute(0, 3, 1, 2)
        source = (source / 255.0 - mean) / std
        target = (target / 255.0 - mean) / std

        scaled_source, scaled_target, src_region_source, tgt_region_source, src_region_repr, tgt_region_repr = (
            model.image_scaler(source.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1))
        )

        feeds = {
            "source_image": scaled_source.permute(0, 3, 1, 2).contiguous().numpy().astype(np.float32),
            "target_image": scaled_target.permute(0, 3, 1, 2).contiguous().numpy().astype(np.float32),
        }
        flow_scaled, covisibility_scaled = self.session.run(None, feeds)

        flow, valid = unmap_predicted_flow(
            torch.from_numpy(flow_scaled),
            src_region_repr,
            tgt_region_repr,
            src_region_source,
            tgt_region_source,
            source_image.shape[:2],
            target_image.shape[:2],
        )
        covisibility, _ = unmap_predicted_channels(
            torch.from_numpy(covisibility_scaled),
            src_region_repr,
            src_region_source,
            source_image.shape[:2],
        )

        return UFMOnnxOutput(flow=flow, covisibility=covisibility.squeeze(1), valid=valid)
