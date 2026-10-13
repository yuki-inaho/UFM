"""RGB-D self-supervised fine-tuning trainer for UFM (procedure 17C).

Scope of this unit
------------------
This module provides the *trainer implementation and the CPU contract*.  It is
not a claim that the real GPU run (procedure 18), the short gate (19) or the FULL
run (20) have been executed.  ``scripts/train_rgbd.py`` runs a CPU
resolve/preflight/synthetic smoke; the real CUDA run is launched on the host via
the command recorded in the worker report.

Key contracts implemented here
------------------------------
* **Local DINO hub**: the released UFM encoder is built from a *pinned local*
  ``facebookresearch/dinov2`` hub tree (verified aggregate python-tree SHA) with
  ``pretrained=False``; the released weights are then loaded strictly via the
  existing :mod:`uniflowmatch.models.checkpoint` helper.  A moving
  ``github/main`` download is never used.
* **Precision**: master weights stay float32; the forward runs under bf16
  autocast on CUDA and float32 on CPU.  ``model.half()`` is forbidden.
* **Freeze**: the encoder is ``requires_grad_(False)`` + ``eval()``, re-applied
  after every ``model.train()`` (``model.train()`` would otherwise flip it back).
* **Symmetry**: pairs are packed ``A0B0A1B1...`` / ``B0A0B1A1...`` so the
  existing ``encode-sym`` fastpath is exercised with the same RGB / depth /
  dilated-window mask / K / flow / covis / occl-sup alignment.
* **Self-supervision**: augmentation consistency (hflip + photometric) with the
  original prediction detached and only the augmented prediction carrying
  gradient; warm-up ramp ``step/warmup`` (strictly > 0 at step 1); OFF performs no
  extra forward.
* **Checkpointing**: strict resume identity, atomic last-good writes, last-y
  resume-only vs best-x evaluation-only.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from itertools import islice
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from uniflowmatch.training import rgbd_checkpoint as C
from uniflowmatch.training.rgbd_deterministic_ops import (
    deterministic_forward_context,
)
from uniflowmatch.training.rgbd_losses import (
    LossContractError,
    NonFiniteLossError,
    RGBDLossComputer,
    UnsupportedLossError,
    hflip_flow,
    hflip_intrinsics,
    photometric_augment,
    consistency_mask,
)

__all__ = [
    "TrainerError",
    "NonFiniteTrainingError",
    "ModelBuildError",
    "LocalDinoHubError",
    "DINO_TREE_SHA_ALGORITHM",
    "assert_no_half",
    "interleave_batch",
    "pack_pair_batch",
    "hflip_view_batch",
    "local_dino_hub",
    "dino_python_tree_sha256",
    "freeze_encoder",
    "build_rgbd_model",
    "load_released_rgbd_weights",
    "load_released_rgb_weights",
    "RGB_DEPTH_STRIP_KEYS",
    "strip_depth_input_keys",
    "param_grad_norm",
    "EventsLogger",
    "RGBDTrainer",
    "check_experiment_contract",
    "final_validation_and_save_required",
    "DeadlineReached",
    "DeadlineGuard",
]


class TrainerError(RuntimeError):
    """Base class for RGB-D trainer contract violations."""


class NonFiniteTrainingError(FloatingPointError):
    """A non-finite loss/gradient was detected; the update is aborted."""


class ModelBuildError(TrainerError):
    """The RGB-D model could not be built from the resolved config."""


class LocalDinoHubError(TrainerError):
    """The pinned local DINO hub tree failed verification / loading."""


class DeadlineReached(TrainerError):
    """The host wall-clock deadline was reached; stop gracefully and keep last-y."""


#: aggregation algorithm from the approved descriptor (step16).
DINO_TREE_SHA_ALGORITHM = "sha256(json.dumps(relativepath_to_fileSHA,sort_keys=True,separators=(comma,colon)).encode())"

#: image tensors / masks that share one geometry and must be packed together.
_PACK_TENSOR_KEYS = (
    "img",
    "dilated_depth_m",
    "raw_depth_m",
    "depthmap",
    "dilated_depth_window_mask",
    "raw_depth_window_mask",
    "valid_mask",
    "camera_intrinsics",
    "flow",
    "non_occluded_mask",
    "occlusion_supervision_mask",
    "fov_mask",
    "flow_valid",
    "depth_provenance",
)
_PACK_SCALAR_KEYS = ("data_norm_type", "pair_id", "capture", "split", "frame_index", "flow_direction")

#: depth / depth-mask / provenance keys removed from the view sent to a *true
#: RGB-only* model (``depth_fusion is None``).  These are exactly the depth
#: inputs the UFM depth-fusion path would consume; the packed supervision
#: (``flow`` / ``non_occluded_mask`` / ...) is never touched.
RGB_DEPTH_STRIP_KEYS = (
    "depth",
    "dilated_depth_m",
    "depthmap",
    "depth_mask",
    "dilated_depth_window_mask",
    "valid_mask",
    "raw_depth_m",
    "raw_depth_window_mask",
    "depth_provenance",
)


# --------------------------------------------------------------------------- #
# precision / freeze guards
# --------------------------------------------------------------------------- #
def assert_no_half(model: nn.Module) -> None:
    """Forbid any half-precision parameter/buffer (``model.half`` is forbidden)."""
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if isinstance(t, torch.Tensor) and t.dtype in (torch.float16, torch.bfloat16):
            raise TrainerError(
                f"{name!r} is {t.dtype}; master weights must stay float32 "
                "(model.half is forbidden; use autocast for the forward)"
            )


def freeze_encoder(model: nn.Module) -> List[str]:
    """Set ``requires_grad_(False)`` on the encoder and put it in ``eval`` mode.

    Returns the sorted list of frozen parameter names.  Must be re-applied after
    any ``model.train()`` call.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise TrainerError("model has no 'encoder' module to freeze")
    frozen: List[str] = []
    for n, p in encoder.named_parameters():
        p.requires_grad_(False)
        frozen.append(f"encoder.{n}")
    encoder.eval()
    return sorted(frozen)


def param_grad_norm(model: nn.Module, prefix: str) -> float:
    total = 0.0
    for n, p in model.named_parameters():
        if n.startswith(prefix) and p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


def strip_depth_input_keys(view: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of ``view`` without any depth / mask / provenance key.

    Applied ONLY to the model-forward input of a true RGB-only model
    (``depth_fusion is None``).  The packed supervision and the augmentation
    geometry keep their depth keys; this merely guarantees the RGB model never
    sees a depth input.
    """
    return {k: v for k, v in view.items() if k not in RGB_DEPTH_STRIP_KEYS}


# --------------------------------------------------------------------------- #
# symmetry packing
# --------------------------------------------------------------------------- #
def _interleave_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack((x, y), dim=1).flatten(0, 1)


def _interleave_list(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    """Interleave two equal-length sequences as ``A0B0A1B1...`` (metadata)."""
    if len(a) != len(b):
        raise TrainerError("interleave_list requires equal-length sequences")
    out: List[Any] = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return out


def interleave_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``A0B0A1B1...`` for tensors whose leading dim is the pair batch."""
    if a.shape[0] != b.shape[0]:
        raise TrainerError("interleave_batch requires equal leading dims")
    return _interleave_pair(a, b)


def pack_pair_batch(views: Sequence[Sequence[Mapping[str, Any]]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Pack ``N`` pairs into the symmetrized ``A0B0A1B1...`` / ``B0A0B1A1...``.

    ``views`` is a sequence of pairs ``[view_a, view_b]`` where ``view_a`` is the
    forward-direction source (its ``flow`` is the fwd teacher) and ``view_b`` is
    the backward-direction source.  Returns ``(view1, view2, supervision)`` where
    ``supervision`` carries the per-row teacher flow / masks / occlusion support in
    the same ABBA order.
    """
    if not views:
        raise TrainerError("pack_pair_batch received no pairs")
    a_views = [pair[0] for pair in views]
    b_views = [pair[1] for pair in views]

    def stack(key: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if key not in a_views[0]:
            return None, None
        a = torch.stack([torch.as_tensor(v[key]) for v in a_views], dim=0)
        b = torch.stack([torch.as_tensor(v[key]) for v in b_views], dim=0)
        return a, b

    view1: Dict[str, Any] = {"symmetrized": True, "data_norm_type": a_views[0]["data_norm_type"]}
    view2: Dict[str, Any] = {"symmetrized": True, "data_norm_type": b_views[0]["data_norm_type"]}
    for key in _PACK_TENSOR_KEYS:
        a, b = stack(key)
        if a is None:
            continue
        view1[key] = _interleave_pair(a, b)
        view2[key] = _interleave_pair(b, a)
    # metadata MUST interleave exactly like the tensors: view1 rows are
    # A0,B0,A1,B1... and view2 rows are B0,A0,B1,A1...
    a_instances = [str(v["instance"]) for v in a_views]
    b_instances = [str(v["instance"]) for v in b_views]
    view1["instance"] = _interleave_list(a_instances, b_instances)
    view2["instance"] = _interleave_list(b_instances, a_instances)

    a_flow, b_flow = stack("flow")
    if a_flow is None:
        raise TrainerError("pack_pair_batch requires per-view teacher flow")
    a_mask, b_mask = stack("non_occluded_mask")
    if a_mask is None:
        a_mask, b_mask = stack("flow_valid")
    a_occl, b_occl = stack("occlusion_supervision_mask")
    a_pair_ids = [str(v["pair_id"]) for v in a_views]
    b_pair_ids = [str(v["pair_id"]) for v in b_views]
    supervision = {
        "flow": _interleave_pair(a_flow, b_flow),
        "mask": _interleave_pair(a_mask, b_mask) if a_mask is not None else None,
        "covis_target": _interleave_pair(a_mask, b_mask) if a_mask is not None else None,
        "covis_supervision": _interleave_pair(a_occl, b_occl) if a_occl is not None else None,
        # pair_ids interleave to match the tensor rows A0,B0,A1,B1...
        "pair_ids": _interleave_list(a_pair_ids, b_pair_ids),
    }
    return view1, view2, supervision


def hflip_view_batch(views: Mapping[str, Any], width: int) -> Dict[str, Any]:
    """Horizontal-flip a packed view dict (RGB / depth / masks / provenance / K / flow).

    Flow has its ``dx`` negated; intrinsics are transformed; depth is unchanged in
    value but flipped spatially.
    """
    out: Dict[str, Any] = dict(views)
    for key, value in views.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key == "camera_intrinsics":
            out[key] = hflip_intrinsics(value, width)
        elif key == "flow":
            out[key] = hflip_flow(value)
        elif key in ("img", "dilated_depth_m", "raw_depth_m", "depthmap",
                     "dilated_depth_window_mask", "raw_depth_window_mask",
                     "valid_mask", "non_occluded_mask",
                     "occlusion_supervision_mask", "fov_mask", "flow_valid",
                     "depth_provenance"):
            out[key] = torch.flip(value, dims=[-1]).clone()
    return out


# --------------------------------------------------------------------------- #
# local DINO hub
# --------------------------------------------------------------------------- #
def dino_python_tree_sha256(hub_dir: Any) -> str:
    """Aggregate SHA of a local DINO hub tree's ``*.py`` files (descriptor algo)."""
    root = Path(str(hub_dir))
    table: Dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        table[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def local_dino_hub(hub_dir: Any, entry: str, expected_tree_sha: str, *, pretrained: bool = False):
    """Bounded context that forces ``torch.hub.load`` onto the pinned local tree.

    Verifies the aggregate python-tree SHA *before* patching, then forces
    ``source="local"`` and ``pretrained=False`` for the duration of model
    construction.  UniCeption itself is never modified.
    """
    if expected_tree_sha is None:
        raise LocalDinoHubError("expected local DINO tree SHA is required")
    got = dino_python_tree_sha256(hub_dir)
    if got != str(expected_tree_sha):
        raise LocalDinoHubError(
            f"local DINO hub tree SHA mismatch: expected {expected_tree_sha}, got {got}"
        )
    original = torch.hub.load

    def _local_load(repo_or_dir, model, *args, **kwargs):
        kwargs.pop("force_reload", None)
        kwargs["source"] = "local"
        kwargs["pretrained"] = bool(pretrained)
        return original(str(hub_dir), entry, *args, **kwargs)

    torch.hub.load = _local_load
    try:
        yield str(hub_dir)
    finally:
        torch.hub.load = original


# --------------------------------------------------------------------------- #
# model construction / weight migration
# --------------------------------------------------------------------------- #
def build_rgbd_model(
    model_kwargs: Mapping[str, Any],
    *,
    fusion_kwargs: Optional[Mapping[str, Any]] = None,
    model_class: Callable[..., nn.Module],
    rgb_only: bool = False,
) -> nn.Module:
    """Construct the model, applying the fusion kwargs correctly.

    ``fusion_kwargs`` is a **bare** dict (``feature_dim`` / ``depth_hidden_dim`` /
    ``gate_init`` / ``depth_scale``).  It must be nested under
    ``depth_fusion_kwargs`` together with ``use_depth_fusion=True``; updating
    ``model_kwargs`` with the bare dict would raise an unexpected ``feature_dim``.

    ``rgb_only=True`` builds the *true* RGB-only model: ``use_depth_fusion=False``
    and NO ``depth_fusion_kwargs`` (the fusion kwargs are deliberately not passed).
    ``rgb_only`` defaults to ``False`` so the existing RGB-D path is unchanged.
    """
    kwargs = dict(model_kwargs)
    if rgb_only:
        kwargs.update(use_depth_fusion=False)
        kwargs.pop("depth_fusion_kwargs", None)
    else:
        kwargs.update(use_depth_fusion=True, depth_fusion_kwargs=dict(fusion_kwargs or {}))
    kwargs.pop("pretrained_checkpoint_path", None)
    kwargs.pop("pretrained_backbone_checkpoint_path", None)
    return model_class(**kwargs)


def load_released_rgbd_weights(
    model: nn.Module,
    asset_path: Any,
    *,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Strictly load the released UFM weights and migrate RGB -> RGB-D in place."""
    from uniflowmatch.models.checkpoint import (
        load_checkpoint,
        migrate_rgb_to_rgbd,
        verify_file_identity,
    )

    verify_file_identity(
        str(asset_path), expected_sha256=expected_sha256, expected_bytes=expected_bytes
    )
    state = load_checkpoint(str(asset_path), weights_only=True, map_location="cpu")
    return migrate_rgb_to_rgbd(model, state)


def load_released_rgb_weights(
    model: nn.Module,
    asset_path: Any,
    *,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Strictly load the released UFM RGB weights into a true RGB-only model.

    Uses the existing alias-aware :func:`load_released_state` (never the
    RGB->RGB-D migration helper, which *requires* the depth module).  Verifies
    the asset SHA256 / byte size before any weight bytes are read.
    """
    from uniflowmatch.models.checkpoint import (
        extract_state_dict,
        load_checkpoint,
        load_released_state,
        verify_file_identity,
    )

    report = verify_file_identity(
        str(asset_path), expected_sha256=expected_sha256, expected_bytes=expected_bytes
    )
    state = load_checkpoint(str(asset_path), weights_only=True, map_location="cpu")
    out = dict(report)
    out["state"] = load_released_state(model, extract_state_dict(state), require_fp32=True)
    return out


# --------------------------------------------------------------------------- #
# experiment contract / deadline
# --------------------------------------------------------------------------- #
def check_experiment_contract(cfg: Mapping[str, Any]) -> None:
    """Enforce the per-preset run contract before any training starts.

    FULL must have ``full_budget_fixed=true`` (fixed after the procedure-18
    capacity measurement), no ``max_updates`` cap, full validation each epoch and
    early stop OFF.  smoke/short must carry an explicit positive
    ``validation_max_pairs`` and ``max_updates``.
    """
    exp = cfg.get("experiment", {}) or {}
    name = exp.get("name")
    if name == "full":
        if exp.get("full_budget_fixed") is not True:
            raise TrainerError(
                "FULL requires experiment.full_budget_fixed=true (fix the budget from the "
                "procedure-18 capacity measurement before starting FULL)"
            )
        if exp.get("max_updates") is not None:
            raise TrainerError("FULL must not set max_updates")
        if exp.get("early_stop") is not False:
            raise TrainerError("FULL requires early_stop=false")
        if exp.get("validate_full_each_epoch") is not True:
            raise TrainerError("FULL requires validate_full_each_epoch=true")
    elif name in ("smoke", "short"):
        if int(exp.get("max_updates") or 0) <= 0:
            raise TrainerError(f"{name} requires a positive max_updates")
        if int(exp.get("validation_max_pairs") or 0) <= 0:
            raise TrainerError(f"{name} requires an explicit validation_max_pairs")
    else:
        raise TrainerError(f"unknown experiment preset {name!r}")


def final_validation_and_save_required(cfg: Mapping[str, Any]) -> bool:
    """smoke/short must still run a final validation + save even at the last step."""
    return (cfg.get("experiment", {}) or {}).get("name") in ("smoke", "short")


class DeadlineGuard:
    """Wall-clock stop/pause protection: raises :class:`DeadlineReached` and keeps last-y.

    This is *not* a substitute for a completed FULL run; the caller records the
    ``stopped`` status and the reason, and the last-y checkpoint remains resumable.
    """

    def __init__(self, max_wall_s: Optional[float]):
        self.max_wall_s = None if max_wall_s is None else float(max_wall_s)
        self.start = time.monotonic()

    def check(self) -> None:
        if self.max_wall_s is not None and (time.monotonic() - self.start) >= self.max_wall_s:
            raise DeadlineReached(
                f"wall-clock deadline ({self.max_wall_s}s) reached; stop and keep last-y"
            )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start


# --------------------------------------------------------------------------- #
# events log
# --------------------------------------------------------------------------- #
class EventsLogger:
    """Append+flush Git-external ``events.jsonl`` numeric records."""

    _FIELDS = (
        "stage", "event", "processed", "total", "epoch", "global_step", "loss",
        "supervised_loss", "selfsup_loss", "covis_loss", "val_loss", "grad_norm",
        "elapsed_wall_s", "eta_s", "gpu_memory_bytes", "status",
    )

    def __init__(self, path: Any):
        self.path = Path(str(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, *, stage: str, event: str, status: str = "running", **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {"stage": stage, "event": event, "status": status}
        for key in self._FIELDS:
            if key in fields and fields[key] is not None:
                value = fields[key]
                if isinstance(value, float) and not math.isfinite(value):
                    raise NonFiniteTrainingError(
                        f"refusing to log non-finite {key}={value} for {stage}/{event}"
                    )
                record[key] = value
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        return record


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class RGBDTrainer:
    """Bounded RGB-D trainer (CPU contract + host entry)."""

    def __init__(
        self,
        resolved_cfg: Mapping[str, Any],
        *,
        model: Optional[nn.Module] = None,
        optimizer: Any = None,
        dataset: Any = None,
        loss_computer: Optional[RGBDLossComputer] = None,
        events_path: Optional[Any] = None,
        identity: Optional[Mapping[str, Any]] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        val_teacher_epe_key: str = "val_teacher_epe",
    ):
        self.cfg = dict(resolved_cfg)
        self.device = torch.device(device)
        self.seed = int(seed if seed is not None else self.cfg.get("seed", 0))
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.val_teacher_epe_key = val_teacher_epe_key
        self.identity = dict(identity or {})
        self.loss = loss_computer or RGBDLossComputer(
            self.cfg.get("loss", {}),
            model_has_flow_covariance=self._model_has_covariance(),
        )
        self.events = EventsLogger(events_path) if events_path is not None else None
        self.global_step = 0
        self.epoch = 0
        self.best_monitor: Dict[str, Any] = {"value": None, "global_step": None, "name": self.val_teacher_epe_key}
        self._aug_generator = torch.Generator().manual_seed(self.seed + 1)
        # dedicated DataLoader generator: iterator creation must not consume the
        # global RNG, so resume with the same sampler order is deterministic.
        self._dataloader_generator = torch.Generator().manual_seed(self.seed + 2)
        self._start_time: Optional[float] = None
        self.trainable: Dict[str, bool] = {}

    # -- introspection ----------------------------------------------------- #
    def _model_has_covariance(self) -> bool:
        return False

    def _encoder_params(self) -> List[nn.Parameter]:
        return list(self.model.encoder.parameters()) if self.model is not None else []

    # -- mode -------------------------------------------------------------- #
    def set_train_mode(self) -> None:
        """``model.train()`` followed by re-applying the encoder freeze."""
        if self.model is None:
            raise TrainerError("no model attached")
        self.model.train()
        freeze_encoder(self.model)
        assert_no_half(self.model)

    def set_eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()

    # -- precision --------------------------------------------------------- #
    def _train_precision(self) -> str:
        """Resolve the configured forward autocast precision.

        The value lives under ``model.train_precision`` in the resolved config
        (with a bare top-level fallback for unit fixtures); the default is
        float32 so a missing setting can never silently enable bf16.
        """
        model_cfg = self.cfg.get("model")
        precision = None
        if isinstance(model_cfg, Mapping):
            precision = model_cfg.get("train_precision")
        if precision is None:
            precision = self.cfg.get("train_precision")
        return str(precision if precision is not None else "float32").lower()

    def _autocast(self):
        """bf16 autocast ONLY when the cfg asks for it on CUDA; else float32."""
        if self.device.type == "cuda" and self._train_precision() in ("bf16", "bfloat16"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # -- device transfer --------------------------------------------------- #
    @staticmethod
    def _move_tensors_inplace(mapping: Mapping[str, Any], device: torch.device) -> Mapping[str, Any]:
        """Move every torch.Tensor value to ``device``; metadata (str/list) stays."""
        for key, value in list(mapping.items()):
            if isinstance(value, torch.Tensor):
                mapping[key] = value.to(device)
        return mapping

    def _to_device(self, view1, view2, supervision=None):
        self._move_tensors_inplace(view1, self.device)
        self._move_tensors_inplace(view2, self.device)
        if supervision is not None:
            for key in ("flow", "mask", "covis_target", "covis_supervision"):
                value = supervision.get(key)
                if isinstance(value, torch.Tensor):
                    supervision[key] = value.to(self.device)
        return view1, view2, supervision

    # -- forward ----------------------------------------------------------- #
    def forward(self, view1: Mapping[str, Any], view2: Mapping[str, Any]):
        # determinism hook: math-only SDPA and/or deterministic bilinear are
        # applied for the duration of this single forward scope only, and
        # restored in `finally`.  It wraps `_autocast` (outside it).
        with deterministic_forward_context(self.cfg):
            with self._autocast():
                # A true RGB-only model (depth_fusion is None) must never receive
                # a depth input: strip depth / depth-mask / provenance keys from
                # the forwarded view.  The RGB-D default is unchanged.
                if getattr(self.model, "depth_fusion", None) is None:
                    view1 = strip_depth_input_keys(view1)
                    view2 = strip_depth_input_keys(view2)
                return self.model(view1, view2)

    # -- loss -------------------------------------------------------------- #
    def compute_supervised_loss(self, model_result, supervision: Mapping[str, Any]):
        pred_flow = model_result.flow.flow_output
        covis_logits = None
        if getattr(model_result, "covisibility", None) is not None:
            covis_logits = model_result.covisibility.logits
        return self.loss.supervised_terms(
            pred_flow,
            supervision["flow"],
            supervision["mask"],
            covis_logits=covis_logits,
            covis_target=supervision.get("covis_target"),
            covis_supervision=supervision.get("covis_supervision"),
        )

    # -- augmentation consistency ----------------------------------------- #
    def compute_selfsup_loss(
        self,
        original_result,
        view1: Mapping[str, Any],
        view2: Mapping[str, Any],
        supervision: Mapping[str, Any],
        global_step: int,
    ):
        """Augmented forward + consistency term (OFF does no extra forward).

        The *same* geometry flag (``selfsup.aug.hflip``) drives the augmented
        image / depth / K / flow / mask AND the transformed target / consistency
        mask.  When ``hflip=false`` the geometry is unchanged (photometric-only
        augmentation), so the target flow and its support mask are used as-is.
        """
        if not self.loss.selfsup_enabled:
            return original_result.flow.flow_output.sum() * 0.0, {"selfsup_loss": 0.0, "selfsup_weight": 0.0}
        do_hflip = bool(self.loss.selfsup.get("hflip"))
        aug1, aug2 = self._augment_views(view1, view2)
        aug_result = self.forward(aug1, aug2)
        original_detached = original_result.flow.flow_output.detach()
        valid = supervision["mask"]
        if do_hflip:
            target = hflip_flow(original_detached)
            valid_t = torch.flip(valid, dims=[-1]) if isinstance(valid, torch.Tensor) else valid
        else:
            target = original_detached
            valid_t = valid
        mask = consistency_mask(valid_t, target, target.shape[2], target.shape[3])
        return self.loss.selfsup_term(target, aug_result.flow.flow_output, mask, global_step)

    def _augment_views(self, view1, view2):
        """Build augmented views honoring ``selfsup.aug.hflip`` / ``photometric``.

        The geometry transform is applied ONLY when ``hflip`` is true so an
        ``hflip=false`` run leaves RGB / depth / K / flow / mask geometry exactly
        unchanged; photometric-only augmentation is therefore supported.
        """
        ss = self.loss.selfsup
        do_hflip = bool(ss.get("hflip"))
        if do_hflip:
            width = int(view1["img"].shape[-1])
            aug1 = hflip_view_batch(view1, width)
            aug2 = hflip_view_batch(view2, width)
        else:
            aug1 = dict(view1)
            aug2 = dict(view2)
        if ss.get("photometric"):
            mean, std = self._norm_stats(view1)
            for v in (aug1, aug2):
                if "img" in v and isinstance(v["img"], torch.Tensor):
                    b = float((torch.rand(1, generator=self._aug_generator).item() - 0.5) * 0.2)
                    c = float(1.0 + (torch.rand(1, generator=self._aug_generator).item() - 0.5) * 0.2)
                    v["img"] = photometric_augment(v["img"], mean, std, brightness=b, contrast=c)
        return aug1, aug2

    def _norm_stats(self, view1):
        from uniflowmatch.training.rgbd_losses import LossContractError as _E

        entry_mean = view1.get("norm_mean")
        entry_std = view1.get("norm_std")
        if entry_mean is None or entry_std is None:
            # dinov2 fallback (public, non-private constants)
            entry_mean = torch.tensor([0.485, 0.456, 0.406])
            entry_std = torch.tensor([0.229, 0.224, 0.225])
        return torch.as_tensor(entry_mean).float(), torch.as_tensor(entry_std).float()

    # -- one optimizer update --------------------------------------------- #
    def train_step(self, views, *, global_step: Optional[int] = None, accumulate: int = 1) -> Dict[str, Any]:
        """One optimizer update over ``views`` (a list of pairs). Returns info.

        A non-finite loss aborts the update *before* the optimizer step and raises
        :class:`NonFiniteTrainingError`; the last-good checkpoint is left intact.
        """
        if self.model is None or self.optimizer is None:
            raise TrainerError("trainer requires a model and an optimizer")
        step = int(global_step if global_step is not None else self.global_step + 1)
        self.set_train_mode()
        # AMUSE allocates its momentum buffers on ``train()`` for the *current*
        # parameter device, so this must run after the model has been moved to
        # its device (CUDA buffers from the start).
        self.optimizer.train()
        self.optimizer.zero_grad()

        view1, view2, supervision = pack_pair_batch(views)
        self._to_device(view1, view2, supervision)
        result = self.forward(view1, view2)
        sup_loss, sup_info = self.compute_supervised_loss(result, supervision)
        selfsup_loss, selfsup_info = self.compute_selfsup_loss(result, view1, view2, supervision, step)
        loss = sup_loss + selfsup_loss
        if not bool(torch.isfinite(loss)):
            raise NonFiniteTrainingError(
                f"non-finite total loss at step {step}; aborting before optimizer step "
                f"(supervised={sup_info.get('supervised_loss')}, "
                f"selfsup={selfsup_info.get('selfsup_loss')})"
            )
        loss.backward()
        grad_norm = float(self.optimizer.grad_norm())
        if not math.isfinite(grad_norm):
            raise NonFiniteTrainingError(
                f"non-finite grad_norm {grad_norm} at step {step}; aborting before optimizer step"
            )
        self.optimizer.clip_grad_norm_(float((self.cfg.get("optimizer", {}).get("gradient_clip", {}) or {}).get("value", 1.0)))
        self.optimizer.train()
        self.optimizer.step()

        info: Dict[str, Any] = {
            "global_step": step,
            "loss": float(loss.detach()),
            "supervised_loss": float(sup_info.get("supervised_loss", 0.0)),
            "selfsup_loss": float(selfsup_info.get("selfsup_loss", 0.0)),
            "covis_loss": float(sup_info.get("covis_loss", 0.0)),
            "grad_norm": grad_norm,
            "selfsup_weight": float(selfsup_info.get("selfsup_weight", 0.0)),
            "encoder_grad_norm": param_grad_norm(self.model, "encoder."),
            "info_sharing_grad_norm": param_grad_norm(self.model, "info_sharing."),
            "depth_fusion_grad_norm": param_grad_norm(self.model, "depth_fusion."),
            "gate_value": self._gate_value(),
        }
        self.global_step = step
        return info

    def _gate_value(self) -> Optional[float]:
        df = getattr(self.model, "depth_fusion", None)
        if df is None or not hasattr(df, "gate"):
            return None
        return float(df.gate.detach().reshape(-1)[0])

    # -- batching (deterministic, dedicated generator) --------------------- #
    def dataloader_generator(self) -> torch.Generator:
        return self._dataloader_generator

    def iter_train_batches(self, dataset, sampler, *, max_updates: Optional[int] = None):
        """Yield ``dataset[idx]`` in the sampler's consumed-cursor order.

        The sampler cursor is *not* advanced here; the caller advances it via
        ``sampler.commit()`` after each completed optimizer update.  Iterator
        creation uses the dedicated generator, never the global RNG.
        """
        produced = 0
        while max_updates is None or produced < max_updates:
            exhausted = True
            for idx in sampler:
                exhausted = False
                yield dataset[idx]
                produced += 1
                if max_updates is not None and produced >= max_updates:
                    return
            if exhausted:
                sampler.advance_epoch()

    # -- validation (no selfsup) ------------------------------------------ #
    @torch.no_grad()
    def validate(self, val_views, *, max_pairs: Optional[int] = None) -> Dict[str, Any]:
        """Validation uses the averaged ``x`` state and the same teacher flow/mask.

        No self-supervised augmentation is applied.  The best monitor uses the
        common ``val_teacher_epe`` metric.
        """
        self.set_eval_mode()
        if self.optimizer is None:
            raise TrainerError("validation requires an optimizer")
        pairs = list(val_views)
        if max_pairs is not None:
            pairs = pairs[: int(max_pairs)]
        epe_losses: List[float] = []
        covis_losses: List[float] = []
        with self.optimizer.eval_mode():
            for views in pairs:
                view1, view2, supervision = pack_pair_batch(views)
                self._to_device(view1, view2, supervision)
                result = self.forward(view1, view2)
                covis_logits = None
                if getattr(result, "covisibility", None) is not None:
                    covis_logits = result.covisibility.logits
                # self-supervised augmentation is OFF during validation.
                _, info = self.loss.supervised_terms(
                    result.flow.flow_output,
                    supervision["flow"],
                    supervision["mask"],
                    covis_logits=covis_logits,
                    covis_target=supervision.get("covis_target"),
                    covis_supervision=supervision.get("covis_supervision"),
                )
                epe_losses.append(float(info["supervised_loss"]))
                if "covis_bce" in info:
                    covis_losses.append(float(info["covis_bce"]))
        value = float(sum(epe_losses) / len(epe_losses)) if epe_losses else 0.0
        covis_value = float(sum(covis_losses) / len(covis_losses)) if covis_losses else None
        is_best = self.best_monitor["value"] is None or value < self.best_monitor["value"]
        if is_best:
            self.best_monitor = {"value": value, "global_step": self.global_step, "name": self.val_teacher_epe_key}
        self.set_train_mode()
        return {
            self.val_teacher_epe_key: value,
            "val_covis_bce": covis_value,
            "n_pairs": len(pairs),
            "is_best": is_best,
        }

    # -- checkpoint -------------------------------------------------------- #
    def _generator_extra(self) -> Dict[str, Any]:
        return {
            "aug_generator_state": self._aug_generator.get_state(),
            "dataloader_generator_state": self._dataloader_generator.get_state(),
        }

    def save_last_y(self, path: Any, *, sampler_state: Mapping[str, Any]) -> str:
        if self.optimizer is None or self.model is None:
            raise TrainerError("checkpoint requires model + optimizer")
        return C.save_training_checkpoint(
            path,
            kind="last_y",
            model=self.model,
            optimizer=self.optimizer,
            identity=self.identity,
            resolved_config=self.cfg,
            resume_config_sha=C.semantic_config_sha(self.cfg),
            global_step=self.global_step,
            epoch=self.epoch,
            sampler_state=sampler_state,
            rng_state=C.capture_rng_state(),
            best_monitor=self.best_monitor,
            trainable=self.trainable or C.trainable_mask(self.model),
            extra=self._generator_extra(),
        )

    def save_best_x(self, path: Any, *, sampler_state: Mapping[str, Any]) -> str:
        """Save the averaged ``x`` (evaluation-only) while in eval mode."""
        if self.optimizer is None or self.model is None:
            raise TrainerError("checkpoint requires model + optimizer")
        with self.optimizer.eval_mode():
            return C.save_training_checkpoint(
                path,
                kind="best_averaged_x",
                model=self.model,
                optimizer=self.optimizer,
                identity=self.identity,
                resolved_config=self.cfg,
                resume_config_sha=C.semantic_config_sha(self.cfg),
                global_step=self.global_step,
                epoch=self.epoch,
                sampler_state=sampler_state,
                rng_state=C.capture_rng_state(),
                best_monitor=self.best_monitor,
                trainable=self.trainable or C.trainable_mask(self.model),
            )

    # -- run loop ---------------------------------------------------------- #
    def _val_batches(self, val_dataset, max_pairs: Optional[int]):
        """Read only the needed validation items.

        A sized/indexable dataset is indexed directly so ``max_pairs`` bounds
        the number of ``__getitem__`` calls; an unsized iterable falls back to
        ``islice``.  ``max_pairs is None`` means FULL (read everything).
        """
        if hasattr(val_dataset, "__len__") and hasattr(val_dataset, "__getitem__"):
            count = len(val_dataset)
            if max_pairs is not None:
                count = min(count, int(max_pairs))
            items = [val_dataset[i] for i in range(count)]
        else:
            stream = iter(val_dataset)
            if max_pairs is not None:
                stream = islice(stream, int(max_pairs))
            items = list(stream)
        return [[item] for item in items]

    def _emit(self, *, stage, event, status="running", **fields):
        if self.events is not None:
            self.events.emit(stage=stage, event=event, status=status, **fields)

    def _iter_pairs(self, dataset, sampler, *, limit: int):
        """Yield ``dataset[idx]`` straight from the sampler's consumed cursor.

        The sampler cursor is advanced by the run loop via ``commit()`` after
        each completed optimizer update, so the *actual* permutation indices,
        epoch and cursor are shared with checkpoint resume.  ``limit`` bounds the
        number of yielded pairs (the remaining budget for this invocation).
        """
        produced = 0
        if limit <= 0:
            return
        while produced < limit:
            got = False
            for idx in sampler:
                got = True
                yield dataset[idx]
                produced += 1
                if produced >= limit:
                    return
            if not got:
                raise TrainerError("sampler yielded no indices; cannot continue")

    def _peak_vram_bytes(self) -> int:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(self.device))
        return 0

    def _reject_unsupported_runtime(self, experiment: Mapping[str, Any]) -> None:
        """Refuse settings this bounded trainer cannot honour (never silent)."""
        ds = self.cfg.get("dataset")
        loader = ds.get("loader") if isinstance(ds, Mapping) else None
        if isinstance(loader, Mapping):
            batch_pairs = loader.get("batch_pairs", 1)
            if batch_pairs is not None and int(batch_pairs) > 1:
                raise TrainerError(
                    "dataset.loader.batch_pairs>1 is not supported (one pair per update); "
                    "refusing to silently ignore it"
                )
            workers = loader.get("num_workers", 0)
            if workers is not None and int(workers) != 0:
                raise TrainerError(
                    "dataset.loader.num_workers>0 (parallel workers) is not supported; "
                    "refusing to silently ignore it"
                )
        for key in ("accumulate_grad_batches", "accumulation", "grad_accumulation"):
            value = experiment.get(key) if isinstance(experiment, Mapping) else None
            if value is None:
                value = self.cfg.get(key)
            if value is not None and int(value) > 1:
                raise TrainerError(
                    f"{key}>1 (gradient accumulation) is not supported; "
                    "refusing to silently ignore it"
                )

    def write_run_receipt(self, path: Any, result: Mapping[str, Any]) -> str:
        """Atomically write the Git-external run receipt (atomic tmp + replace)."""
        payload: Dict[str, Any] = {
            "unit": "UFM_STEP17C",
            "status": result.get("status"),
            "reason": result.get("reason"),
            "experiment": result.get("experiment"),
            "global_step": result.get("global_step"),
            "epoch": self.epoch,
            "total_updates": result.get("total_updates"),
            "remaining_at_start": result.get("remaining_at_start"),
            "population": {
                "train": result.get("train_population"),
                "val": result.get("val_population"),
            },
            "counts": result.get("counts"),
            "last_losses": result.get("last_losses"),
            "val_teacher_epe": result.get("val_teacher_epe"),
            "best_monitor": dict(self.best_monitor),
            "checkpoints": result.get("checkpoints"),
            "identity": dict(self.identity),
            "timing_wall_s": result.get("timing"),
            "peak_vram_bytes": result.get("peak_vram_bytes"),
            "gate_value": result.get("gate_value"),
            "gate_delta": result.get("gate_delta"),
            "adapter_grad_norm": result.get("adapter_grad_norm"),
            # reproducibility contract actually applied by the entry point
            # (env + torch backend flags); recorded verbatim for audit.
            "determinism": self.cfg.get("determinism"),
        }
        target = Path(str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.flush()
        os.replace(tmp, target)
        return str(target)

    def run(
        self,
        train_dataset,
        val_dataset,
        *,
        experiment: Mapping[str, Any],
        checkpoint_dir: Any,
        sampler: Optional[C.StatefulPermutationSampler] = None,
        deadline_s: Optional[float] = None,
        resume_path: Optional[Any] = None,
        pause_after_updates: Optional[int] = None,
        receipt_path: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run smoke / short / FULL per the experiment preset.

        The remaining update budget is computed from ``self.global_step`` (the
        value restored by resume), so an uninterrupted run and a pause+resume run
        execute exactly the same sampler order / augmentation RNG / updates.  A
        host wall-clock deadline stops gracefully (keeps last-y, records
        ``stopped``); it is never reported as a completed FULL run.  A final
        validation + save always runs for smoke/short.
        """
        from pathlib import Path as _P

        check_experiment_contract({"experiment": dict(experiment)})
        self._reject_unsupported_runtime(experiment)
        ckpt_dir = _P(str(checkpoint_dir))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_path = ckpt_dir / "last_y.pt"
        best_path = ckpt_dir / "best_averaged_x.pt"
        if sampler is None:
            sampler = C.StatefulPermutationSampler(len(train_dataset), seed=self.seed)
        if resume_path is not None:
            payload = self.load_last_y(resume_path)
            sampler.load_state_dict(payload["sampler"])
            self._emit(stage="resume", event="resumed", global_step=self.global_step)

        name = str(experiment.get("name"))
        max_updates = experiment.get("max_updates")
        epochs = experiment.get("epochs")
        val_every = int(experiment.get("validate_every_updates", 0) or 0)
        save_every = int(experiment.get("save_every_updates", 0) or 0)
        val_max = experiment.get("validation_max_pairs")
        train_population = len(train_dataset)
        if name == "full":
            total_updates = train_population * int(epochs)
            full_val = True
        else:
            total_updates = int(max_updates)
            full_val = False
        start_step = int(self.global_step)
        remaining = max(0, total_updates - start_step)
        run_limit = remaining
        if pause_after_updates is not None:
            if name != "short":
                raise TrainerError("--pause-after-updates is only supported for the short preset")
            run_limit = min(remaining, int(pause_after_updates))

        guard = DeadlineGuard(deadline_s)
        start = time.monotonic()
        heartbeat_s = float(os.environ.get("UFM_RGBD_HEARTBEAT_S", "300") or 0)
        last_beat = start
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        updates = start_step
        gate_before = self._gate_value()
        done = 0
        status = "finished"
        index_sequence: List[int] = []
        loss_sequence: List[float] = []
        last_info: Dict[str, Any] = {}
        last_metrics: Optional[Dict[str, Any]] = None
        timing: Dict[str, Any] = {}
        checkpoints: Dict[str, Any] = {}
        counts: Dict[str, Any] = {"train_updates": 0, "val_calls": 0, "save_calls": 0, "best_calls": 0}

        def _val() -> Dict[str, Any]:
            metrics = self.validate(self._val_batches(val_dataset, val_max))
            counts["val_calls"] += 1
            return metrics

        def _save_last() -> None:
            _t = time.monotonic()
            self.save_last_y(last_path, sampler_state=sampler.state_dict())
            timing["save_wall_s"] = timing.get("save_wall_s", 0.0) + (time.monotonic() - _t)
            counts["save_calls"] += 1
            checkpoints["last_y"] = {
                "path": str(last_path),
                "bytes": last_path.stat().st_size if last_path.is_file() else 0,
            }

        def _save_best() -> None:
            _t = time.monotonic()
            self.save_best_x(best_path, sampler_state=sampler.state_dict())
            timing["save_wall_s"] = timing.get("save_wall_s", 0.0) + (time.monotonic() - _t)
            counts["best_calls"] += 1
            checkpoints["best_averaged_x"] = {
                "path": str(best_path),
                "bytes": best_path.stat().st_size if best_path.is_file() else 0,
            }

        try:
            for pair in self._iter_pairs(train_dataset, sampler, limit=run_limit):
                guard.check()
                idx = sampler.perm[sampler.cursor]
                step = updates + 1
                t0 = time.monotonic()
                info = self.train_step([pair], global_step=step)
                sampler.commit()
                updates = step
                done += 1
                self.epoch = sampler.epoch
                index_sequence.append(int(idx))
                loss_sequence.append(float(info["loss"]))
                last_info = info
                counts["train_updates"] = done
                timing["update_wall_s"] = timing.get("update_wall_s", 0.0) + (time.monotonic() - t0)
                self._emit(stage="train", event="update", processed=step, total=total_updates,
                           epoch=self.epoch, global_step=step, loss=info["loss"],
                           supervised_loss=info["supervised_loss"], selfsup_loss=info["selfsup_loss"],
                           covis_loss=info["covis_loss"], grad_norm=info["grad_norm"],
                           gpu_memory_bytes=self._peak_vram_bytes(),
                           elapsed_wall_s=time.monotonic() - start)
                _now = time.monotonic()
                if heartbeat_s > 0 and (_now - last_beat) >= heartbeat_s:
                    last_beat = _now
                    # finite numeric heartbeat (findings/tips/struggles cadence)
                    self._emit(stage="heartbeat", event="progress", processed=done,
                               total=total_updates, global_step=updates,
                               gpu_memory_bytes=self._peak_vram_bytes(),
                               elapsed_wall_s=_now - start)
                if full_val:
                    if updates % train_population == 0:
                        tv = time.monotonic()
                        metrics = _val()
                        timing["val_wall_s"] = timing.get("val_wall_s", 0.0) + (time.monotonic() - tv)
                        last_metrics = metrics
                        self._emit(stage="val", event="epoch_end", epoch=self.epoch,
                                   val_loss=metrics[self.val_teacher_epe_key], status="ok")
                        _save_last()
                        if metrics["is_best"]:
                            _save_best()
                            self._emit(stage="best", event="saved", epoch=self.epoch, status="ok")
                else:
                    if val_every and updates % val_every == 0:
                        metrics = _val()
                        last_metrics = metrics
                        self._emit(stage="val", event="update_end", global_step=updates,
                                   val_loss=metrics[self.val_teacher_epe_key], status="ok")
                        if metrics["is_best"]:
                            _save_best()
                            self._emit(stage="best", event="saved", global_step=updates, status="ok")
                    if save_every and updates % save_every == 0:
                        _save_last()
                        self._emit(stage="save", event="last_y", global_step=updates, status="ok")

            if done < remaining:
                # paused: keep last-y and report `paused` (never `finished`).
                status = "paused"
                _save_last()
                timing["overall_wall_s"] = time.monotonic() - start
                self._emit(stage="pause", event="paused", global_step=updates, status="paused",
                           elapsed_wall_s=timing["overall_wall_s"])
            else:
                if not full_val:
                    tv = time.monotonic()
                    last_metrics = _val()
                    timing["val_wall_s"] = timing.get("val_wall_s", 0.0) + (time.monotonic() - tv)
                    self._emit(stage="val", event="final", global_step=updates,
                               val_loss=last_metrics[self.val_teacher_epe_key], status="ok")
                    _save_last()
                    if last_metrics["is_best"]:
                        _save_best()
                        self._emit(stage="best", event="saved", global_step=updates, status="ok")
                timing["overall_wall_s"] = time.monotonic() - start
                self._emit(stage="finish", event="done", global_step=updates, status="ok")
        except DeadlineReached as exc:
            status = "stopped"
            _save_last()
            timing["overall_wall_s"] = time.monotonic() - start
            self._emit(stage="stop", event="deadline", global_step=updates, status="stopped",
                       elapsed_wall_s=timing["overall_wall_s"])
            result = {
                "status": "stopped", "reason": str(exc), "global_step": updates,
                "checkpoint": str(last_path), "experiment": name,
                "total_updates": total_updates, "remaining_at_start": remaining,
                "train_population": train_population, "val_population": len(val_dataset),
                "counts": counts, "last_losses": last_info, "val_teacher_epe": None,
                "checkpoints": checkpoints, "timing": timing,
                "peak_vram_bytes": self._peak_vram_bytes(), "gate_value": self._gate_value(),
                "gate_delta": None, "index_sequence": index_sequence,
                "loss_sequence": loss_sequence,
            }
            if receipt_path is not None:
                self.write_run_receipt(receipt_path, result)
            return result
        except NonFiniteTrainingError:
            self._emit(stage="failed", event="non_finite", global_step=updates, status="failed",
                       elapsed_wall_s=time.monotonic() - start)
            raise
        except Exception as exc:  # noqa: BLE001
            self._emit(stage="failed", event=type(exc).__name__, global_step=updates,
                       status="failed", elapsed_wall_s=time.monotonic() - start)
            raise

        gate_after = self._gate_value()
        result = {
            "status": status,
            "experiment": name,
            "global_step": updates,
            "checkpoint": str(last_path),
            "total_updates": total_updates,
            "remaining_at_start": remaining,
            "train_population": train_population,
            "val_population": len(val_dataset),
            "counts": counts,
            "last_losses": last_info,
            "val_teacher_epe": (last_metrics[self.val_teacher_epe_key] if last_metrics else None),
            "checkpoints": checkpoints,
            "timing": timing,
            "peak_vram_bytes": self._peak_vram_bytes(),
            "gate_value": gate_after,
            "gate_delta": (None if (gate_before is None or gate_after is None)
                           else float(gate_after - gate_before)),
            "adapter_grad_norm": last_info.get("depth_fusion_grad_norm"),
            "index_sequence": index_sequence,
            "loss_sequence": loss_sequence,
        }
        if receipt_path is not None:
            self.write_run_receipt(receipt_path, result)
        return result

    def load_last_y(self, path: Any, *, strict_identity: bool = True) -> Dict[str, Any]:
        payload = C.load_training_checkpoint(
            path,
            expected_identity=self.identity if strict_identity else None,
            require_kind="last_y",
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"], strict=True)
        self.global_step = int(payload["global_step"])
        self.epoch = int(payload["epoch"])
        self.best_monitor = dict(payload["best_monitor"])
        extra = payload.get("extra") or {}
        if extra.get("aug_generator_state") is not None:
            self._aug_generator.set_state(extra["aug_generator_state"])
        if extra.get("dataloader_generator_state") is not None:
            self._dataloader_generator.set_state(extra["dataloader_generator_state"])
        C.restore_rng_state(payload["rng"])
        return payload
