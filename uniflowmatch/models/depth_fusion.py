"""Metric-depth adapter and zero-gated residual fusion for UFM (ADR-14 Option C).

This module implements the *selected* RGB-D fusion design:

    h_fused = h_rgb + g * (masked_depth_adapter(depth, depth_mask))

where ``g`` is a learnable scalar ``gate`` whose value is initialised to exactly
``0.0``.  Consequences that the contract in
``reports/ufm-rgbd-step14-fusion-adr-red/API_spec_14.md`` binds us to:

* with ``g == 0`` the fusion is the exact identity on ``h_rgb`` (bit-exact
  parity, since ``0 * finite == 0``);
* with an all-invalid depth mask the masked adapter output is exactly ``0`` for
  *every* output feature pixel, so parity holds even when ``g != 0`` (the mask is
  applied *after* the adapter, so an adapter bias cannot leak);
* invalid pixels cannot contribute to their own output feature pixel;
* non-finite depth, wrong shape, wrong dtype, wrong device, or a missing mask
  when depth is supplied all fail fast with typed exceptions.

The module is intentionally small and dependency-light.  It does **not** unproject
depth to 3D and does **not** consume camera intrinsics; ``camera_intrinsics`` is
carried through the forward API for the loader/trainer but the depth stem treats
depth as a per-pixel scalar feature map.  See the procedure-15 addendum.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DepthFusion",
    "DepthFusionError",
    "DepthFusionShapeError",
    "DepthFusionDtypeError",
    "DepthFusionDeviceError",
    "DepthFusionNonFiniteError",
    "DepthFusionMaskError",
]


# --------------------------------------------------------------------------- #
# typed fail-fast exceptions
# --------------------------------------------------------------------------- #
class DepthFusionError(Exception):
    """Base class for all depth-fusion contract violations."""


class DepthFusionShapeError(DepthFusionError, ValueError):
    """Depth/mask/feature shape is not contract-compliant."""


class DepthFusionDtypeError(DepthFusionError, TypeError):
    """Depth/mask dtype is not contract-compliant."""


class DepthFusionDeviceError(DepthFusionError, ValueError):
    """Depth/mask/feature device mismatch."""


class DepthFusionNonFiniteError(DepthFusionError, ValueError):
    """Depth contains NaN/inf (validated even at gate 0)."""


class DepthFusionMaskError(DepthFusionError, ValueError):
    """Depth was supplied without a usable valid mask."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def prepare_metric_depth(depth, depth_mask=None, *, ref=None):
    """Coerce a metric-depth tensor to ``(B, 1, H, W)`` float32 with a bool mask.

    Accepted depth shapes: ``(H, W)``, ``(1, H, W)``, ``(B, H, W)``,
    ``(1, 1, H, W)`` or ``(B, 1, H, W)``.  Depth units are **metres**.

    ``ref`` (optional) is a reference ``(B, H, W)``/``(B, C, H, W)`` feature or
    image tensor used to validate batch, device and (where meaningful) HxW.
    """
    if not isinstance(depth, torch.Tensor):
        raise DepthFusionDtypeError(
            f"depth must be a torch.Tensor in metric metres, got {type(depth).__name__}"
        )
    if not depth.is_floating_point():
        raise DepthFusionDtypeError(f"depth must be a floating tensor, got {depth.dtype}")

    d = depth
    if d.dim() == 2:
        d = d.unsqueeze(0).unsqueeze(0)
    elif d.dim() == 3:
        d = d.unsqueeze(1)
    elif d.dim() == 4:
        if d.shape[1] != 1:
            raise DepthFusionShapeError(
                f"depth with 4 dims must be (B, 1, H, W), got {tuple(d.shape)}"
            )
    else:
        raise DepthFusionShapeError(
            "depth must have shape (H, W), (B, H, W), (1, H, W), (B, 1, H, W) or (1, 1, H, W); "
            f"got {tuple(d.shape)}"
        )

    if ref is not None:
        ref_b = int(ref.shape[0])
        if d.shape[0] not in (1, ref_b):
            raise DepthFusionShapeError(
                f"depth batch {d.shape[0]} incompatible with reference batch {ref_b}"
            )
        if d.device != ref.device:
            raise DepthFusionDeviceError(
                f"depth on {d.device} but reference on {ref.device}"
            )

    if not bool(torch.isfinite(d).all()):
        raise DepthFusionNonFiniteError(
            "depth contains NaN/inf; non-finite metric depth is rejected (even at gate 0)"
        )

    if depth_mask is None:
        mask = torch.ones_like(d, dtype=torch.bool)
    else:
        if not isinstance(depth_mask, torch.Tensor):
            raise DepthFusionDtypeError(
                f"depth_mask must be a bool torch.Tensor, got {type(depth_mask).__name__}"
            )
        # bool dtype is mandatory: a non-bool mask (float 0/1, float NaN, int, ...)
        # is rejected *before* any cast, so NaN/float masks can never silently
        # become a valid True.  See the procedure-15 addendum / contract.
        if depth_mask.dtype != torch.bool:
            raise DepthFusionDtypeError(
                f"depth_mask must be a bool tensor, got {depth_mask.dtype}"
            )
        m = depth_mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(1)
        elif m.dim() == 4:
            if m.shape[1] != 1:
                raise DepthFusionShapeError(
                    f"depth_mask with 4 dims must be (B, 1, H, W), got {tuple(m.shape)}"
                )
        else:
            raise DepthFusionShapeError(
                "depth_mask must have shape (H, W), (B, H, W), (1, H, W), (B, 1, H, W) or (1, 1, H, W); "
                f"got {tuple(m.shape)}"
            )
        if m.shape[-2:] != d.shape[-2:]:
            raise DepthFusionShapeError(
                f"depth_mask spatial {tuple(m.shape[-2:])} != depth {tuple(d.shape[-2:])}"
            )
        if m.shape[0] not in (1, d.shape[0]):
            raise DepthFusionShapeError(
                f"depth_mask batch {m.shape[0]} incompatible with depth batch {d.shape[0]}"
            )
        if m.device != d.device:
            raise DepthFusionDeviceError(
                f"depth_mask on {m.device} but depth on {d.device}"
            )
        mask = m

    if ref is not None:
        ref_b = int(ref.shape[0])
        if d.shape[0] == 1 and ref_b > 1:
            d = d.expand(ref_b, *d.shape[1:])
            mask = mask.expand(ref_b, *mask.shape[1:])

    return d, mask


# --------------------------------------------------------------------------- #
# module
# --------------------------------------------------------------------------- #
class DepthFusion(nn.Module):
    """Small CNN depth adapter with a zero-initialised gated residual.

    Args:
        feature_dim: channel count of the RGB feature map the residual is added to.
        depth_hidden_dim: hidden width of the depth stem.
        gate_init: initial value of the gate (must default to ``0.0``).
        depth_scale: positive metres scale used to normalise metric depth into a
            bounded range before the stem (documented, not learned).
        clamp_min / clamp_max: clamp bounds applied after scaling (defaults 0..1).
    """

    def __init__(
        self,
        feature_dim: int,
        depth_hidden_dim: int,
        gate_init: float = 0.0,
        depth_scale: float = 1.0,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        depth_hidden_dim = int(depth_hidden_dim)
        if feature_dim <= 0 or depth_hidden_dim <= 0:
            raise DepthFusionShapeError("feature_dim and depth_hidden_dim must be positive")
        if not (depth_scale > 0.0):
            raise DepthFusionError(f"depth_scale must be positive, got {depth_scale}")
        if not (clamp_max > clamp_min):
            raise DepthFusionError(
                f"clamp_max must be > clamp_min, got {clamp_min}/{clamp_max}"
            )

        self.feature_dim = feature_dim
        self.depth_hidden_dim = depth_hidden_dim
        self.depth_scale = float(depth_scale)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        self.depth_adapter = nn.Sequential(
            nn.Conv2d(1, depth_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(depth_hidden_dim, feature_dim, kernel_size=3, padding=1),
        )
        # zero-initialised gate => exact RGB identity at init
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    # -- introspection ----------------------------------------------------- #
    @property
    def gate_value(self) -> float:
        return float(self.gate.detach())

    # -- internals --------------------------------------------------------- #
    def _depth_contribution(
        self, rgb_feat: torch.Tensor, depth, depth_mask, image_hw: Tuple[int, int]
    ) -> torch.Tensor:
        # A valid mask is mandatory whenever depth is supplied: the module contract
        # makes a missing mask fail fast (the public predict API is the only place
        # allowed to default the mask, and it does so with ``depth > 0``).
        if depth_mask is None:
            raise DepthFusionMaskError(
                "depth was supplied without a valid mask; DepthFusion.forward requires "
                "an explicit bool depth_mask (a mask is mandatory)"
            )
        d, m = prepare_metric_depth(depth, depth_mask, ref=rgb_feat)

        img_h, img_w = int(image_hw[0]), int(image_hw[1])
        if tuple(d.shape[-2:]) != (img_h, img_w):
            raise DepthFusionShapeError(
                f"depth spatial {(int(d.shape[-2]), int(d.shape[-1]))} != image HW {(img_h, img_w)}"
            )
        if d.dtype != rgb_feat.dtype:
            d = d.to(rgb_feat.dtype)
        m = m.to(rgb_feat.dtype)

        out_h, out_w = int(rgb_feat.shape[-2]), int(rgb_feat.shape[-1])
        # zero the invalid depth values *before* the stem so no NaN/junk leaks.
        d = torch.where(m > 0, d, torch.zeros_like(d))
        d = (d / self.depth_scale).clamp(self.clamp_min, self.clamp_max)

        d_feat = F.interpolate(d, size=(out_h, out_w), mode="nearest")
        m_feat = F.interpolate(m, size=(out_h, out_w), mode="nearest")

        contrib = self.depth_adapter(d_feat)
        # mask nearest to the *output* feature grid too, blocking invalid-pixel
        # contribution (and making all-invalid exactly zero even with adapter bias).
        contrib = contrib * m_feat
        return contrib

    def forward(self, rgb_feat: torch.Tensor, depth, depth_mask=None, image_hw=None) -> torch.Tensor:
        """Fuse metric depth into ``rgb_feat`` via the zero-gated residual."""
        if not isinstance(rgb_feat, torch.Tensor) or rgb_feat.dim() != 4:
            raise DepthFusionShapeError("rgb_feat must be a (B, C, H, W) torch.Tensor")
        if image_hw is None:
            image_hw = (int(rgb_feat.shape[-2]), int(rgb_feat.shape[-1]))

        contrib = self._depth_contribution(rgb_feat, depth, depth_mask, image_hw)
        if tuple(contrib.shape) != tuple(rgb_feat.shape):
            raise DepthFusionShapeError(
                f"depth contribution shape {tuple(contrib.shape)} != rgb_feat {tuple(rgb_feat.shape)}"
            )
        gate = self.gate.to(rgb_feat.dtype).view(1, 1, 1, 1)
        return rgb_feat + gate * contrib
