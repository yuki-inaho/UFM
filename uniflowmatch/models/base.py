"""
Base class of the UniFlowMatch training system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from uniflowmatch.models.depth_fusion import DepthFusionShapeError, prepare_metric_depth


@dataclass
class UFMFlowFieldOutput:
    """
    Output interface of the flow field prediction network.
    """

    flow_output: torch.Tensor
    flow_covariance: Optional[torch.Tensor] = None
    flow_covariance_inv: Optional[torch.Tensor] = None
    flow_covariance_log_det: Optional[torch.Tensor] = None


@dataclass
class UFMMaskFieldOutput:
    """
    Output interface of the mask prediction network.
    """

    mask: torch.Tensor
    logits: torch.Tensor


@dataclass
class UFMClassificationRefinementOutput:
    """
    Output interface of the classification refinement network.
    """

    # the flow output of the regression step, with shape [B, 2, H, W].
    # it is the initial flow output, which is used to get the first local feature maps for the residual.
    regression_flow_output: torch.Tensor

    # residual is the output of the refinement step, with shape [B, 2, H, W].
    # it is added to the initial flow output to get the final flow output.
    residual: torch.Tensor

    # log_softmax is
    #   the logarithm of
    #   the softmax of
    #   similarity of the pixel's feature
    #       to that of its neighborhood of the flow prediction
    #       in the other image.
    # it have shape [B, H, W, P, P], the similarity of pixel at [b, h, w] to its neighborhood [P, P] centered at regression_flow_output[b, h, w]
    log_softmax: torch.Tensor

    feature_map_0: torch.Tensor
    feature_map_1: torch.Tensor

    temperature: Optional[torch.Tensor] = None
    attention_bias: Optional[torch.Tensor] = None

@dataclass
class UFMOutputInterface:
    """
    Output interface of the UniFlowMatch training system.
    """

    flow: Optional[UFMFlowFieldOutput] = None

    # Refinement output (for training and visualization)
    classification_refinement: Optional[UFMClassificationRefinementOutput] = None

    # auxiliary ouputs
    covisibility: Optional[UFMMaskFieldOutput] = None


from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from uniflowmatch.utils.flow_resizing import (
    AutomaticShapeSelection,
    ResizeToFixedManipulation,
    unmap_predicted_channels,
    unmap_predicted_flow,
)


class UniFlowMatchModelsBase(torch.nn.Module):
    def __init__(self, inference_resolution: Optional[Union[List[Tuple[int, int]], Tuple[int, int]]] = None):
        super().__init__()

        if inference_resolution is None:
            inference_resolution = [(560, 420)]

        if isinstance(inference_resolution[0], int):  # Handle the case for single resolution
            inference_resolution = [inference_resolution]

        self.inference_resolution = inference_resolution

        self.image_scaler = AutomaticShapeSelection(
            *[ResizeToFixedManipulation((resolution[1], resolution[0])) for resolution in inference_resolution],
            strategy="closest_aspect",  # will inference on the trained aspect ratio that is closest to the input image 1
        )

    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks.

        Args:
        - view1 (Dict[str, Any]): Input view 1
          - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
          - instance (List[int]): List of instance indices, or id of the input image
          - data_norm_type (str): Data normalization type, see uniception.models.encoders.IMAGE_NORMALIZATION_DICT
        - view2 (Dict[str, Any]): Input view 2
          - (same structure as view1)
        Returns:
        - Dict[str, Any]: Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - occlusion [Optional] (Dict[str, torch.Tensor]): Occlusion output
            - [Optional] mask
            - [Optional] logits
        """
        raise NotImplementedError("Implement this method in derived classes")

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This methods guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
        - Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer
        """

        raise NotImplementedError("Implement this method in derived classes")

    def predict_correspondences_batched(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        data_norm_type: Optional[str] = None,
        source_depth: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        depth_mask=None,
        camera_intrinsics=None,
        source_depth_mask: Optional[torch.Tensor] = None,
        target_depth_mask: Optional[torch.Tensor] = None,
    ) -> UFMOutputInterface:
        """
        Predict correspondences between source and target images.

        This method generates random correspondences for demonstration purposes.

        Args:
            source_image (torch.Tensor): Tensor of shape BCHW/BHWC/CHW/HWC, dtype of uint8 or float32 The source image tensor.
            target_image (torch.Tensor): Tensor of shape BCHW/BHWC/CHW/HWC, dtype of uint8 or float32 The target image tensor.
            data_norm_type (Optional[str]): normalization of float inputs (see IMAGE_NORMALIZATION_DICT).
            source_depth (Optional[torch.Tensor]): metric depth (metres) for the source view,
                shape HW / 1HW / BHW. ``None`` keeps the RGB-only path.
            target_depth (Optional[torch.Tensor]): metric depth (metres) for the target view.
            depth_mask: optional explicit **(source, target)** pair of bool valid masks
                (HW / 1HW / BHW).  A single tensor is rejected: never implicitly reuse
                one mask for both views.  Defaults per view to ``depth > 0``.
            camera_intrinsics: optional explicit **(source, target)** pair of (3, 3)
                intrinsics; may contain NaN when K is unknown.  The depth stem does
                not unproject, so K is carried for the loader/trainer only.
            source_depth_mask / target_depth_mask (Optional[torch.Tensor]): per-view
                explicit masks that override ``depth_mask``.

        Returns:
            UFMOutputInterface:
                - flow
                    - flow_output       (torch.Tensor): Tensor of shape (B, 2, H, W) representing the flow output in the original image space.
                - occlusion
                    - mask              (torch.Tensor): Tensor of shape (B, H, W) representing the covisibility in range [0, 1]. 1 = fully covisible, 0 = fully occluded or out of range.
        """

        assert isinstance(source_image, torch.Tensor) and isinstance(
            target_image, torch.Tensor
        ), "source_image and target_image must be torch.Tensors"
        assert source_image.dim() in [3, 4], "source_image must have dimensions 3 or 4"
        assert target_image.dim() in [3, 4], "target_image must have dimensions 3 or 4"

        batched = source_image.dim() == 4

        if not batched:
            # add batch dimension
            source_image = source_image.unsqueeze(0)
            target_image = target_image.unsqueeze(0)

        # check the channel
        if source_image.shape[1] == 3 and target_image.shape[1] == 3:
            pass  # do nothing because the image is in BCHW format
        elif source_image.shape[-1] == 3 and target_image.shape[-1] == 3:
            # convert to BCHW
            source_image = source_image.permute(0, 3, 1, 2)
            target_image = target_image.permute(0, 3, 1, 2)
        else:
            raise ValueError("source_image and target_image must have 3 channels in either BCHW or BHWC format")

        required_data_norm_type = self.encoder.data_norm_type

        image_device = source_image.device

        if source_image.dtype == torch.float32:
            assert data_norm_type is not None, "data_norm_type must be provided for float32 images"
            assert (
                data_norm_type in IMAGE_NORMALIZATION_DICT
            ), f"data_norm_type must be one of {list(IMAGE_NORMALIZATION_DICT.keys())}"

            if data_norm_type != required_data_norm_type:
                # apply transformation to the correct from the old normalization
                prev_mean = (
                    IMAGE_NORMALIZATION_DICT[data_norm_type].mean.view(1, 3, 1, 1).to(image_device, non_blocking=True)
                )
                prev_std = (
                    IMAGE_NORMALIZATION_DICT[data_norm_type].std.view(1, 3, 1, 1).to(image_device, non_blocking=True)
                )
                mean = (
                    IMAGE_NORMALIZATION_DICT[required_data_norm_type]
                    .mean.view(1, 3, 1, 1)
                    .to(image_device, non_blocking=True)
                )
                std = (
                    IMAGE_NORMALIZATION_DICT[required_data_norm_type]
                    .std.view(1, 3, 1, 1)
                    .to(image_device, non_blocking=True)
                )

                source_image = source_image * (prev_std / std) + (prev_mean - mean) / std
                target_image = target_image * (prev_std / std) + (prev_mean - mean) / std

        elif source_image.dtype == torch.uint8:
            # convert into float32 and apply normalization
            mean = (
                IMAGE_NORMALIZATION_DICT[required_data_norm_type]
                .mean.view(1, 3, 1, 1)
                .to(image_device, non_blocking=True)
            )
            std = (
                IMAGE_NORMALIZATION_DICT[required_data_norm_type]
                .std.view(1, 3, 1, 1)
                .to(image_device, non_blocking=True)
            )

            source_image = (source_image.float() / 255.0 - mean) / std
            target_image = (target_image.float() / 255.0 - mean) / std
        else:
            raise ValueError("source_image and target_image must be of type torch.float32 or torch.uint8")

        # Now all the inputs are normalized according to the model's encoder and organized in BCHW format

        # ----- optional metric depth (RGB-D), explicit source/target pairs ----- #
        # depth is coerced first; explicit per-view masks and the (source, target)
        # ``depth_mask`` pair then override the ``depth > 0`` default (never the
        # other way round, and one tensor is never reused for both views).
        src_d = self._coerce_predict_depth(source_depth, "source_depth")
        tgt_d = self._coerce_predict_depth(target_depth, "target_depth")
        src_m = source_depth_mask
        tgt_m = target_depth_mask
        if depth_mask is not None:
            m_src, m_tgt = self._as_pair(depth_mask, "depth_mask")
            if src_m is None:
                src_m = m_src
            if tgt_m is None:
                tgt_m = m_tgt
        src_m = self._resolve_predict_mask(src_d, src_m, "source_depth")
        tgt_m = self._resolve_predict_mask(tgt_d, tgt_m, "target_depth")
        src_K, tgt_K = self._as_pair(camera_intrinsics, "camera_intrinsics")

        return self._predict_correspondences_batched(
            source_image,
            target_image,
            source_depth=src_d,
            target_depth=tgt_d,
            source_depth_mask=src_m,
            target_depth_mask=tgt_m,
            source_camera_intrinsics=src_K,
            target_camera_intrinsics=tgt_K,
        )

    # ------------------------------------------------------------------ #
    # RGB-D predict helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_pair(value, name):
        """Split an explicit ``(source, target)`` pair; reject a single tensor."""
        if value is None:
            return None, None
        if isinstance(value, torch.Tensor):
            raise ValueError(
                f"{name} must be an explicit (source, target) pair, not a single tensor; "
                f"never reuse one tensor for both views"
            )
        try:
            src, tgt = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a (source, target) pair, got {type(value).__name__}") from exc
        return src, tgt

    @staticmethod
    def _coerce_predict_depth(depth, name):
        """Coerce an optional metric depth to ``(B, 1, H, W)`` float metres."""
        if depth is None:
            return None
        d, _ = prepare_metric_depth(depth)
        return d

    @staticmethod
    def _resolve_predict_mask(depth, mask, name):
        """Resolve the per-view mask: explicit mask wins, else ``depth > 0``."""
        if depth is None:
            if mask is not None:
                raise ValueError(f"{name}_mask was given without {name}")
            return None
        if mask is None:
            return depth > 0
        _, m = prepare_metric_depth(depth, mask)
        return m

    @staticmethod
    def _resize_intrinsics_halfpixel(K, src_hw, dst_hw):
        """K' = A K for an align_corners=False (half-pixel) resize; NaN K stays NaN."""
        K = K.to(torch.float32)
        src_h, src_w = int(src_hw[0]), int(src_hw[1])
        dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
        sx, sy = dst_w / src_w, dst_h / src_h
        A = torch.tensor(
            [[sx, 0.0, 0.5 * (sx - 1.0)], [0.0, sy, 0.5 * (sy - 1.0)], [0.0, 0.0, 1.0]],
            device=K.device,
            dtype=K.dtype,
        )
        return A @ K

    def _predict_correspondences_batched(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        source_depth: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        source_depth_mask: Optional[torch.Tensor] = None,
        target_depth_mask: Optional[torch.Tensor] = None,
        source_camera_intrinsics: Optional[torch.Tensor] = None,
        target_camera_intrinsics: Optional[torch.Tensor] = None,
    ) -> UFMOutputInterface:
        assert isinstance(source_image, torch.Tensor), "source_image must be a torch.Tensor"
        assert isinstance(target_image, torch.Tensor), "target_image must be a torch.Tensor"

        assert source_image.dim() == 4, "source_image must be of shape (B, 3, H, W)"
        assert target_image.dim() == 4, "target_image must be of shape (B, 3, H, W)"
        assert source_image.shape[1] == 3, "source_image must be of shape (B, 3, H, W)"
        assert target_image.shape[1] == 3, "target_image must be of shape (B, 3, H, W)"

        assert source_image.dtype == torch.float32, "source_image must be of dtype torch.float32"
        assert target_image.dtype == torch.float32, "target_image must be of dtype torch.float32"

        source_shape_hw = source_image.shape[2:]
        target_shape_hw = target_image.shape[2:]

        # Scale images to one of the model's trained resolution.
        (
            scaled_img0,  # The scaled source image
            scaled_img1,  # The scaled target image
            img0_region_source,  # Where in the source image is captured in the scaled image
            img1_region_source,  # Where in the target image is captured in the scaled image
            img0_region_representation,  # Region in the source image is captured in this region in the scaled image
            img1_region_representation,  # same as above, but for the target image
        ) = self.image_scaler(source_image.permute(0, 2, 3, 1), target_image.permute(0, 2, 3, 1))

        scaled_img0 = scaled_img0.permute(0, 3, 1, 2)
        scaled_img1 = scaled_img1.permute(0, 3, 1, 2)

        # Run a forward pass
        view1 = {"img": scaled_img0, "symmetrized": False, "data_norm_type": self.encoder.data_norm_type}
        view2 = {"img": scaled_img1, "symmetrized": False, "data_norm_type": self.encoder.data_norm_type}

        # depth/mask are resized nearest alongside the native->model resize; K is
        # carried as K'=AK (half-pixel) for the loader/trainer (the depth stem does
        # not unproject, so K is informational here).
        scaled_hw0 = scaled_img0.shape[2:]
        scaled_hw1 = scaled_img1.shape[2:]
        if source_depth is not None and tuple(source_depth.shape[-2:]) != tuple(source_image.shape[2:]):
            raise DepthFusionShapeError(
                f"source_depth spatial {tuple(source_depth.shape[-2:])} != source image "
                f"{tuple(source_image.shape[2:])}: metric depth must match the native "
                f"image resolution before the model resize"
            )
        if target_depth is not None and tuple(target_depth.shape[-2:]) != tuple(target_image.shape[2:]):
            raise DepthFusionShapeError(
                f"target_depth spatial {tuple(target_depth.shape[-2:])} != target image "
                f"{tuple(target_image.shape[2:])}"
            )
        if source_depth is not None:
            view1["depth"] = F.interpolate(source_depth, size=scaled_hw0, mode="nearest")
            if source_depth_mask is not None:
                view1["depth_mask"] = (
                    F.interpolate(source_depth_mask.to(torch.float32), size=scaled_hw0, mode="nearest") > 0.5
                )
        if target_depth is not None:
            view2["depth"] = F.interpolate(target_depth, size=scaled_hw1, mode="nearest")
            if target_depth_mask is not None:
                view2["depth_mask"] = (
                    F.interpolate(target_depth_mask.to(torch.float32), size=scaled_hw1, mode="nearest") > 0.5
                )
        if source_camera_intrinsics is not None:
            view1["camera_intrinsics"] = self._resize_intrinsics_halfpixel(
                source_camera_intrinsics, source_image.shape[2:], scaled_hw0
            )
        if target_camera_intrinsics is not None:
            view2["camera_intrinsics"] = self._resize_intrinsics_halfpixel(
                target_camera_intrinsics, target_image.shape[2:], scaled_hw1
            )

        with torch.no_grad():
            with torch.autocast("cuda", torch.float16):
                result = self(view1, view2)

        rescaled_ufm_result = UFMOutputInterface()

        # rescale flow
        flow_output = result.flow.flow_output
        flow_unmapped, flow_unmap_validity = unmap_predicted_flow(
            flow_output,
            img0_region_representation,
            img1_region_representation,
            img0_region_source,
            img1_region_source,
            source_shape_hw,
            target_shape_hw,
        )

        rescaled_ufm_result.flow = UFMFlowFieldOutput(
            flow_output=flow_unmapped,
        )

        # rescale covariance if it exists
        if result.flow.flow_covariance is not None:
            flow_covariance = result.flow.flow_covariance
            flow_covariance_unmapped, _ = unmap_predicted_channels(
                flow_covariance,
                img0_region_representation,
                img0_region_source,
                source_shape_hw,
            )

            # scale covariance in the correct way
            w_pred = scaled_img0.shape[3]
            h_pred = scaled_img0.shape[2]

            w_final = source_shape_hw[1]
            h_final = source_shape_hw[0]

            w_ratio, h_ratio = w_final / w_pred, h_final / h_pred

            flow_covariance_unmapped *= (
                torch.tensor([w_ratio**2, h_ratio**2, w_ratio * h_ratio])
                .view(1, 3, 1, 1)
                .to(flow_covariance_unmapped.device)
            )

            rescaled_ufm_result.flow.flow_covariance = flow_covariance_unmapped

        # rescale occlusion if it exists
        if result.covisibility is not None:
            occlusion_mask = result.covisibility.mask
            covisibility_unmapped, _ = unmap_predicted_channels(
                occlusion_mask,
                img0_region_representation,
                img0_region_source,
                source_shape_hw,
            )

            covisibility_unmapped = covisibility_unmapped.squeeze(1)
            rescaled_ufm_result.covisibility = UFMMaskFieldOutput(mask=covisibility_unmapped, logits=None)

        return rescaled_ufm_result
