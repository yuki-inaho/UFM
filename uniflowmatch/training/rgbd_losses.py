"""Pure masked losses + augmentation consistency for the UFM RGB-D trainer (17C).

This module deliberately does **not** reuse
``uniflowmatch.loss.epe.FlowEPELoss.compute_loss_with_gathered_info``: that
existing implementation replaces non-finite *predictions* in-place with ``0`` and
rewrites the teacher flow, which would silently hide a broken forward pass.  The
losses here instead:

* check that the *prediction* is finite and **fail fast** (typed error, no
  quiet ``NaN -> 0``) when it is not;
* never mutate their inputs (the teacher is sanitised into a *new* tensor);
* normalise by the **sum of valid support** so a pair with small support
  contributes proportionally instead of being dropped;
* return a **graph-connected zero** (``pred.sum() * 0``) when the mask is empty,
  so an empty-support step still has a valid backward graph instead of a
  detached constant.

The teacher flow / covisibility targets are *pseudo* labels (never ground truth).

Variable-support schema policy
------------------------------
``minimum_mask_threshold`` is the per-pair valid-support ratio below which a pair
is *reported* as low-support (for the numeric log).  It is **never** used to
silently drop a pair from the population: the batch loss is always normalised by
the total valid support across every pair, and low-support pairs are counted in
the returned ``info`` dict so the FULL population cannot shrink unnoticed.

Nothing here embeds private data, host paths, or dataset identifiers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "LossContractError",
    "NonFiniteLossError",
    "UnsupportedLossError",
    "SUPPORTED_SUPERVISION",
    "require_bool_mask",
    "masked_flow_epe",
    "masked_covis_bce",
    "photometric_augment",
    "hflip_flow",
    "hflip_intrinsics",
    "consistency_mask",
    "augmentation_consistency_loss",
    "selfsup_weight_at_step",
    "RGBDLossComputer",
]


# --------------------------------------------------------------------------- #
# typed fail-fast exceptions
# --------------------------------------------------------------------------- #
class LossContractError(ValueError):
    """A loss input / config violates the RGB-D loss contract."""


class NonFiniteLossError(FloatingPointError):
    """The prediction (or loss input) contains NaN/inf; no quiet zeroing."""


class UnsupportedLossError(LossContractError):
    """A loss setting is not supported and must not be silently ignored."""


#: supervision ranges we can honour with the real loader masks.
SUPPORTED_SUPERVISION = ("occlusion", "fov_mask", "all")


def require_bool_mask(name: str, mask: torch.Tensor) -> None:
    """Require a genuine ``torch.bool`` mask (never coerce float/int masks)."""
    if not isinstance(mask, torch.Tensor):
        raise LossContractError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise LossContractError(
            f"{name} must be a bool tensor (got {mask.dtype}); a float/int mask is "
            "rejected so a depth mask can never be silently used as a flow mask"
        )


# --------------------------------------------------------------------------- #
# masked EPE (pure)
# --------------------------------------------------------------------------- #
def masked_flow_epe(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_support_threshold: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Masked end-point-error between a prediction and a pseudo teacher flow.

    ``pred_flow`` / ``target_flow`` are ``(N, 2, H, W)``; ``mask`` is a bool
    ``(N, H, W)``.  The prediction must be finite (else :class:`NonFiniteLossError`).
    Non-finite teacher pixels are masked out.  Returns ``(loss, info)`` where the
    loss is normalised by the summed valid support and is graph-connected to
    ``pred_flow`` even for an empty mask.
    """
    if pred_flow.dim() != 4 or pred_flow.shape[1] != 2:
        raise LossContractError(f"pred_flow must be (N,2,H,W), got {tuple(pred_flow.shape)}")
    if target_flow.shape != pred_flow.shape:
        raise LossContractError(
            f"target_flow {tuple(target_flow.shape)} != pred_flow {tuple(pred_flow.shape)}"
        )
    require_bool_mask("mask", mask)
    if mask.shape != (pred_flow.shape[0], pred_flow.shape[2], pred_flow.shape[3]):
        raise LossContractError(
            f"mask {tuple(mask.shape)} != (N,H,W) {tuple(pred_flow.shape[:1] + pred_flow.shape[2:])}"
        )
    if not bool(torch.isfinite(pred_flow).all()):
        n_bad = int((~torch.isfinite(pred_flow)).sum())
        raise NonFiniteLossError(
            f"prediction flow contains {n_bad} non-finite value(s); failing fast "
            "(no NaN->0 replacement)"
        )

    target_finite = torch.isfinite(target_flow).all(dim=1)
    # new tensor: never mutate the caller's teacher flow
    target_safe = torch.where(target_finite.unsqueeze(1), target_flow, torch.zeros_like(target_flow))
    valid = mask & target_finite

    epe = torch.linalg.norm(pred_flow - target_safe, dim=1)  # (N,H,W), finite
    support = valid.float()
    support_sum = support.sum()
    total = int(valid.shape[0])
    per_pair_ratio = support.reshape(total, -1).mean(dim=1)
    low_support = int((per_pair_ratio < float(min_support_threshold)).sum())

    if float(support_sum) <= 0.0:
        loss = pred_flow.sum() * 0.0  # graph-connected zero, not a detached const
    else:
        loss = (epe * support).sum() / support_sum

    info: Dict[str, Any] = {
        "epe": float(loss.detach()),
        "support": int(support_sum),
        "n_pairs": total,
        "n_low_support_pairs": low_support,
        "min_support_threshold": float(min_support_threshold),
        "mean_pair_support_ratio": float(per_pair_ratio.mean()) if total else 0.0,
    }
    return loss, info


# --------------------------------------------------------------------------- #
# covisibility BCE (pseudo teacher)
# --------------------------------------------------------------------------- #
def masked_covis_bce(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Binary cross-entropy of covisibility logits on the supervision mask only.

    ``logits`` is ``(N, 1, H, W)`` (channel squeezed) or ``(N, H, W)``; the target
    and supervision mask are bool ``(N, H, W)``.  Only pixels that are both in the
    occlusion-supervision mask and finite are used.  Empty support -> graph-connected
    zero.
    """
    if logits.dim() == 4:
        if logits.shape[1] != 1:
            raise LossContractError(f"covis logits must be (N,1,H,W), got {tuple(logits.shape)}")
        logits = logits.squeeze(1)
    if logits.dim() != 3:
        raise LossContractError(f"covis logits must be (N,1,H,W) or (N,H,W), got {tuple(logits.shape)}")
    require_bool_mask("target_mask", target_mask)
    require_bool_mask("supervision_mask", supervision_mask)
    if target_mask.shape != logits.shape or supervision_mask.shape != logits.shape:
        raise LossContractError("covis target/supervision mask shape must match squeezed logits")
    if not bool(torch.isfinite(logits).all()):
        raise NonFiniteLossError("covisibility logits contain NaN/inf; failing fast")

    bce = F.binary_cross_entropy_with_logits(
        logits, target_mask.to(logits.dtype), reduction="none"
    )
    support = supervision_mask.float()
    support_sum = support.sum()
    if float(support_sum) <= 0.0:
        loss = logits.sum() * 0.0
    else:
        loss = (bce * support).sum() / support_sum
    return loss, {"covis_bce": float(loss.detach()), "covis_support": int(support_sum)}


# --------------------------------------------------------------------------- #
# augmentation geometry / photometric (pure)
# --------------------------------------------------------------------------- #
def photometric_augment(
    rgb: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> torch.Tensor:
    """DINOv2-normalised RGB -> denorm -> brightness/contrast (clamp 0..1) -> renorm.

    Depth / intrinsics are *not* touched by a photometric transform.
    """
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise LossContractError(f"rgb must be (N,3,H,W), got {tuple(rgb.shape)}")
    # mean / std must follow BOTH the rgb dtype AND device (a CPU stats tensor
    # feeding a CUDA image would otherwise raise / broadcast on the wrong device).
    m = mean.reshape(1, 3, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
    s = std.reshape(1, 3, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
    img = rgb * s + m
    img = torch.clamp(img * float(contrast) + float(brightness), 0.0, 1.0)
    return (img - m) / s


def hflip_flow(flow: torch.Tensor) -> torch.Tensor:
    """Horizontal flip of a ``(N, 2, H, W)`` flow: flip W, negate dx."""
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise LossContractError(f"flow must be (N,2,H,W), got {tuple(flow.shape)}")
    out = torch.flip(flow, dims=[-1]).clone()
    out[:, 0] = -out[:, 0]
    return out


def hflip_intrinsics(K: torch.Tensor, width: int) -> torch.Tensor:
    """Transform ``(N,3,3)`` intrinsics under a horizontal image flip of ``width`` px.

    ``K' = T K T^{-1}`` with ``T = [[-1,0,W-1],[0,1,0],[0,0,1]]``.  NaN-unknown
    intrinsics are carried through unchanged.
    """
    if K.dim() != 3 or K.shape[-2:] != (3, 3):
        raise LossContractError(f"K must be (N,3,3), got {tuple(K.shape)}")
    w = float(width)
    # Matches uniflowmatch.datasets.utils.rgbd_geometry.hflip_affine / apply_affine_to_K:
    # K' = A @ K with A = [[-1, 0, W-1], [0, 1, 0], [0, 0, 1]].
    T = torch.tensor(
        [[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=K.dtype,
        device=K.device,
    )
    T = T.unsqueeze(0).expand(K.shape[0], -1, -1)
    return T @ K


def consistency_mask(
    valid_mask: torch.Tensor,
    flow: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Transformed-original valid mask x transformed-endpoint in-bounds x finite.

    All inputs are the *transformed* (e.g. hflipped) tensors.  Returns a bool
    ``(N, H, W)`` mask suitable as the augmentation-consistency support.
    """
    require_bool_mask("valid_mask", valid_mask)
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise LossContractError(f"flow must be (N,2,H,W), got {tuple(flow.shape)}")
    if flow.shape[0] != valid_mask.shape[0] or tuple(flow.shape[2:]) != tuple(valid_mask.shape[1:]):
        raise LossContractError("flow / valid_mask batch and spatial dims must agree")

    finite = torch.isfinite(flow).all(dim=1)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=flow.device),
        torch.arange(width, device=flow.device),
        indexing="ij",
    )
    xs = xs.to(flow.dtype).unsqueeze(0)
    ys = ys.to(flow.dtype).unsqueeze(0)
    tx = xs + flow[:, 0]
    ty = ys + flow[:, 1]
    in_bounds = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
    return valid_mask & in_bounds & finite


def augmentation_consistency_loss(
    original_detached_flow: torch.Tensor,
    augmented_flow: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_support_threshold: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Consistency between a detached original prediction and an augmented one.

    The *original* prediction must already be detached by the caller
    (``pred.flow.detach()``); only the augmented prediction carries gradient.
    """
    if original_detached_flow.requires_grad:
        raise LossContractError(
            "original prediction must be detached before the consistency loss "
            "(detach_original_prediction=true)"
        )
    return masked_flow_epe(
        augmented_flow,
        original_detached_flow,
        mask,
        min_support_threshold=min_support_threshold,
    )


def selfsup_weight_at_step(
    weight: float, warmup_updates: int, global_step: int
) -> float:
    """Linear warm-up ramp ``weight * min(1, step/warmup)`` (strictly > 0 at step 1)."""
    w = float(weight)
    if w <= 0.0:
        return 0.0
    warmup = int(warmup_updates)
    if warmup <= 0:
        return w
    step = max(1, int(global_step))
    return w * min(1.0, float(step) / float(warmup))


# --------------------------------------------------------------------------- #
# config-driven loss computer
# --------------------------------------------------------------------------- #
class RGBDLossComputer:
    """Consume the resolved ``loss`` config and compute supervised + selfsup terms.

    Fail-fast on unsupported settings (unknown loss class, unsupported
    ``supervision_range``, Laplace NLL ON without a flow-covariance head, crop /
    complex-warp / homography ON).
    """

    def __init__(self, loss_cfg: Mapping[str, Any], *, model_has_flow_covariance: bool = False):
        if not isinstance(loss_cfg, Mapping):
            raise LossContractError("loss config must be a mapping")
        self.cfg = dict(loss_cfg)
        self.supervised = self._parse_supervised(self.cfg.get("supervised", {}) or {})
        self.selfsup = self._parse_selfsup(self.cfg.get("selfsup", {}) or {})
        self._reject_unsupported(self.cfg)
        if bool((self.cfg.get("laplace_nll") or {}).get("enabled", False)):
            if not model_has_flow_covariance:
                raise UnsupportedLossError(
                    "loss.laplace_nll.enabled=true but the model head has no flow "
                    "covariance; sigma must not be faked (rejected)"
                )
        self._last_selfsup_info: Dict[str, Any] = {}

    # -- parsing ----------------------------------------------------------- #
    @staticmethod
    def _parse_supervised(sup: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"flow": None, "covis_bce": None}
        flow = sup.get("flow")
        if flow is not None:
            if flow.get("class") != "FlowEPELoss":
                raise UnsupportedLossError(
                    f"unsupported supervised flow loss class {flow.get('class')!r}; "
                    "only FlowEPELoss is supported"
                )
            rng = flow.get("supervision_range", "occlusion")
            if rng not in SUPPORTED_SUPERVISION:
                raise UnsupportedLossError(
                    f"unsupported supervision_range {rng!r}; expected one of "
                    f"{SUPPORTED_SUPERVISION}"
                )
            out["flow"] = {
                "multiplier": float(flow.get("multiplier", 1.0)),
                "supervision_range": rng,
                "minimum_mask_threshold": float(flow.get("minimum_mask_threshold", 0.0)),
            }
        covis = sup.get("covis_bce")
        if covis is not None:
            if covis.get("class") != "CrossEntropyLoss":
                raise UnsupportedLossError(
                    f"unsupported covis loss class {covis.get('class')!r}; "
                    "only CrossEntropyLoss is supported"
                )
            out["covis_bce"] = {"multiplier": float(covis.get("multiplier", 1.0))}
        return out

    @staticmethod
    def _parse_selfsup(ss: Mapping[str, Any]) -> Dict[str, Any]:
        aug = ss.get("aug", {}) or {}
        return {
            "enabled": bool(ss.get("enabled", False)),
            "weight": float(ss.get("weight", 0.0)),
            "warmup_updates": int(ss.get("warmup_updates", 0)),
            "detach_original_prediction": bool(ss.get("detach_original_prediction", True)),
            "same_geometry": bool(ss.get("same_geometry", True)),
            "hflip": bool(aug.get("hflip", False)),
            "photometric": bool(aug.get("photometric", False)),
            "crop": bool(aug.get("crop", False)),
            "complex_warp": bool(aug.get("complex_warp", False)),
        }

    @staticmethod
    def _reject_unsupported(cfg: Mapping[str, Any]) -> None:
        if bool((cfg.get("homography") or {}).get("enabled", False)):
            raise UnsupportedLossError("planar homography loss must stay OFF for non-rigid scenes")
        ss = cfg.get("selfsup", {}) or {}
        aug = ss.get("aug", {}) or {}
        if bool(aug.get("crop", False)):
            raise UnsupportedLossError("selfsup.aug.crop=true is unsupported (crop OFF by default)")
        if bool(aug.get("complex_warp", False)):
            raise UnsupportedLossError(
                "selfsup.aug.complex_warp=true is unsupported (complex warp OFF by default)"
            )

    # -- weight ------------------------------------------------------------ #
    def weight_at_step(self, global_step: int) -> float:
        if not self.selfsup["enabled"]:
            return 0.0
        return selfsup_weight_at_step(
            self.selfsup["weight"], self.selfsup["warmup_updates"], global_step
        )

    @property
    def selfsup_enabled(self) -> bool:
        return self.selfsup["enabled"]

    @property
    def last_selfsup_info(self) -> Dict[str, Any]:
        return dict(self._last_selfsup_info)

    # -- terms ------------------------------------------------------------- #
    def supervised_terms(
        self,
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        flow_mask: torch.Tensor,
        *,
        covis_logits: Optional[torch.Tensor] = None,
        covis_target: Optional[torch.Tensor] = None,
        covis_supervision: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sum of the supervised flow EPE and optional covis BCE (multiplier applied)."""
        info: Dict[str, Any] = {}
        loss = pred_flow.sum() * 0.0
        flow_cfg = self.supervised.get("flow")
        if flow_cfg is not None:
            flow_loss, flow_info = masked_flow_epe(
                pred_flow,
                target_flow,
                flow_mask,
                min_support_threshold=flow_cfg["minimum_mask_threshold"],
            )
            loss = loss + flow_cfg["multiplier"] * flow_loss
            info["supervised_loss"] = float(flow_loss.detach())
            info.update(flow_info)
        covis_cfg = self.supervised.get("covis_bce")
        if covis_cfg is not None and covis_logits is not None and covis_target is not None:
            if covis_supervision is None:
                raise LossContractError("covis BCE requires the occlusion supervision mask")
            covis_loss, covis_info = masked_covis_bce(covis_logits, covis_target, covis_supervision)
            loss = loss + covis_cfg["multiplier"] * covis_loss
            info["covis_loss"] = float(covis_loss.detach())
            info.update(covis_info)
        info["loss"] = float(loss.detach())
        return loss, info

    def selfsup_term(
        self,
        original_detached_flow: torch.Tensor,
        augmented_flow: torch.Tensor,
        mask: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Weighted augmentation-consistency term (0.0 and no work when OFF)."""
        if not self.selfsup["enabled"]:
            self._last_selfsup_info = {"selfsup_loss": 0.0, "selfsup_weight": 0.0}
            return augmented_flow.sum() * 0.0, dict(self._last_selfsup_info)
        weight = self.weight_at_step(global_step)
        loss, info = augmentation_consistency_loss(
            original_detached_flow,
            augmented_flow,
            mask,
            min_support_threshold=self.supervised.get("flow", {}).get("minimum_mask_threshold", 0.0),
        )
        weighted = weight * loss
        info["selfsup_weight"] = float(weight)
        info["selfsup_loss"] = float(weighted.detach())
        self._last_selfsup_info = dict(info)
        return weighted, info
