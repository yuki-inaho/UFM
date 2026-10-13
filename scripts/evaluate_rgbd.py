"""Public generic four-variant RGB-D evaluation runner (step 21B1).

This entry point scores the four UFM RGB-D variants --

    ``rgb_pretrained``        released UFM-Base, RGB encoder only (no fusion)
    ``rgbd_pretrained_init``  released UFM-Base migrated to RGB-D (gate == 0)
    ``rgbd_off``              trained RGB-D checkpoint (self-supervision OFF)
    ``rgbd_on``               trained RGB-D checkpoint (self-supervision ON)

-- over **all** pairs of one split with an identical pair order / input / mask
population, then writes per-pair predictions, per-pair and pooled metrics, a
common-normal-population comparison, an events log and an exit receipt.

Design rules enforced here
--------------------------
* Not a single private dataset identifier, id, path, count, calibration array or
  result is embedded: every value (asset / index / checkpoint / split / output /
  variants) is supplied through a Git-external JSON *descriptor* on the CLI.
* ``model`` / ``loader`` / ``trainer`` / ``config`` / ``UniCeption`` / the metric
  core are imported read-only and never modified.
* The released asset / config / local DINO tree / trained checkpoints are
  verified by SHA256 before use; construction is pinned local hub
  ``pretrained=False`` -> strict released load -> RGB-D averaged-``x`` strict
  state.
* Each model runs on one GPU, ``model.eval()`` + ``torch.inference_mode`` under
  the existing determinism contexts (FP32, TF32 off, ``CUBLAS_WORKSPACE_CONFIG``
  ``:4096:8``, math SDPA, deterministic bilinear).
* Metrics come from the existing pure core (pseudo-teacher, never ground truth)
  and keep every pair / direction and every failure; empty support is ``None``,
  never a fabricated zero.  Checkpoint selection on a test-only role is refused.

The module is deliberately importable for CPU dependency-injection tests: the
model factory :func:`_build_variant_model` and the dataset factory
:func:`_load_dataset` are the only seams, and the whole ``main`` path is exercised
with a fake model.

Usage:
    python scripts/evaluate_rgbd.py \
        --descriptor /git/external/eval_descriptor.json \
        --dataset-index /git/external/index.json \
        --split test \
        --output-dir /git/external/eval_out \
        --variants rgb_pretrained,rgbd_pretrained_init,rgbd_off,rgbd_on \
        --device cuda --gpu 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from uniflowmatch.evaluation.rgbd_evaluation import (
    EvaluationContractError,
    MaskedFlowEPEResult,
    ReportPathError,
    SelectionPolicyError,
    assert_report_path_external,
    compare_variants,
    covis_metrics,
    masked_teacher_flow_epe,
    select_best_variant,
)
from uniflowmatch.training.rgbd_config import determinism_context
from uniflowmatch.training.rgbd_deterministic_ops import deterministic_forward_context
from uniflowmatch.training.rgbd_trainer import (
    EventsLogger,
    build_rgbd_model,
    load_released_rgbd_weights,
    local_dino_hub,
    pack_pair_batch,
    strip_depth_input_keys,
)

#: the four variants this runner evaluates, in canonical order.
VARIANTS: Tuple[str, ...] = (
    "rgb_pretrained",
    "rgbd_pretrained_init",
    "rgbd_off",
    "rgbd_on",
)

#: optional, additive variants for the RGB-only comparison.  They are NOT part
#: of the canonical tuple (whose semantics are unchanged); a descriptor may
#: define them and a caller may explicitly request them.
OPTIONAL_VARIANTS: Tuple[str, ...] = (
    "rgb_off",
    "rgb_on",
)

#: every variant name the runner accepts (canonical + optional).
ALL_VARIANTS: Tuple[str, ...] = VARIANTS + OPTIONAL_VARIANTS

#: variant kinds the descriptor may declare.  ``checkpoint`` keeps its RGB-D
#: meaning; ``rgb_checkpoint`` is the additive RGB-trained (depth-free) kind.
ALLOWED_KINDS: Tuple[str, ...] = ("rgb", "rgbd_init", "checkpoint", "rgb_checkpoint")

#: splits the CLI accepts; a test-only role forbids checkpoint selection.
ALLOWED_SPLITS: Tuple[str, ...] = ("val", "test")

#: tolerance for the RGB vs RGB-D-init gate-0 parity check (fused residual is
#: ``0 * finite == 0`` so the two encodings must be numerically identical).
PARITY_ATOL = 0.0

#: tiny helper to make a failed run explicit.
_PROGRESS_UNITS = ("eval", "processed", "total", "save", "finish", "fail")


class EvaluationRunError(RuntimeError):
    """The evaluation descriptor / run violates a contract before/while running."""


# --------------------------------------------------------------------------- #
# source identity (public modules only, never private data)
# --------------------------------------------------------------------------- #
def _sha256_file(path: Any) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def public_source_sha() -> str:
    """Aggregate SHA256 of the public evaluation source (runner + metric core)."""
    repo = Path(__file__).resolve().parents[1]
    table: Dict[str, str] = {}
    for rel in (
        "scripts/evaluate_rgbd.py",
        "uniflowmatch/evaluation/rgbd_evaluation.py",
        "uniflowmatch/training/rgbd_trainer.py",
    ):
        p = repo / rel
        if p.is_file():
            table[rel] = _sha256_file(p)
    payload = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# descriptor loading / validation
# --------------------------------------------------------------------------- #
def load_descriptor(path: Any) -> Dict[str, Any]:
    p = Path(str(path))
    if not p.is_file():
        raise EvaluationRunError(f"descriptor not found: {p}")
    try:
        desc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise EvaluationRunError(f"descriptor is not valid JSON: {exc}") from exc
    if not isinstance(desc, dict):
        raise EvaluationRunError("descriptor root must be an object")
    return desc


_DEPTH_UNIT_ALIASES = ("metre", "meter", "metres", "meters", "m", "mm", "millimetre", "millimetre")


def validate_descriptor(desc: Mapping[str, Any]) -> None:
    """Validate the descriptor shape / units / variants before any heavy work."""
    for key in ("dataset", "assets", "model", "variants", "determinism"):
        if key not in desc or not isinstance(desc[key], Mapping):
            raise EvaluationRunError(f"descriptor.{key} is required and must be an object")
    role = str(desc.get("role", "")).strip().lower()
    if role not in ("train", "val", "validation", "test", "tune"):
        raise EvaluationRunError(f"descriptor.role {role!r} is not a recognised data role")

    ds = desc["dataset"]
    if not ds.get("index_path"):
        raise EvaluationRunError("descriptor.dataset.index_path is required")
    res = ds.get("resolution")
    if not (isinstance(res, (list, tuple)) and len(res) == 2):
        raise EvaluationRunError("descriptor.dataset.resolution must be [W, H]")
    units = str(ds.get("depth_units", "metre")).strip().lower()
    if units not in _DEPTH_UNIT_ALIASES:
        raise EvaluationRunError(f"descriptor.dataset.depth_units {units!r} is not recognised")
    if units in ("mm", "millimetre"):
        raise EvaluationRunError(
            "descriptor.dataset.depth_units must be metres (the loader converts uint16 mm "
            "to metres with a fixed factor); a millimetre unit is rejected"
        )
    if "mm_to_m" in ds and float(ds["mm_to_m"]) != 0.001:
        raise EvaluationRunError("descriptor.dataset.mm_to_m must be 0.001 (uint16 millimetres)")

    assets = desc["assets"]
    for key in ("asset_path", "asset_sha256", "config_path", "config_sha256",
                "dino_hub_dir", "dino_python_tree_sha256"):
        if not assets.get(key):
            raise EvaluationRunError(f"descriptor.assets.{key} is required")

    variants = desc["variants"]
    for name in ALL_VARIANTS:
        if name not in variants:
            if name in VARIANTS:
                raise EvaluationRunError(f"descriptor.variants must define {name!r}")
            continue
        v = variants[name]
        if not isinstance(v, Mapping):
            raise EvaluationRunError(f"descriptor.variants.{name} must be an object")
        kind = str(v.get("kind", ""))
        if kind not in ALLOWED_KINDS:
            raise EvaluationRunError(
                f"variant {name!r} kind must be one of {list(ALLOWED_KINDS)}, got {kind!r}"
            )
        if kind in ("checkpoint", "rgb_checkpoint") and not v.get("checkpoint_path"):
            raise EvaluationRunError(f"variant {name!r} requires checkpoint_path")


def _model_target_hw(desc: Mapping[str, Any]) -> Tuple[int, int]:
    res = desc["dataset"]["resolution"]
    # UFM resolution is (W, H)
    return int(res[1]), int(res[0])


def check_grid(desc: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    """Assert the staged index grid matches the model grid so nothing re-scales.

    The flow reference grid and every pair's ``reference_hw`` must equal the model
    target ``(H, W)``: otherwise the loader would rescale the already-model-grid
    teacher flow a second time.
    """
    target_hw = _model_target_hw(desc)
    ref = index.get("teacher_reference_hw")
    if ref is not None and (int(ref[0]), int(ref[1])) != target_hw:
        raise EvaluationRunError(
            f"index teacher_reference_hw {tuple(ref)} != model target {target_hw} "
            "(teacher flow would be re-scaled)"
        )
    pairs = index.get("pairs") or []
    for i, pair in enumerate(pairs):
        for direction in ("flow_fwd", "flow_bwd"):
            entry = pair.get(direction) or {}
            rhw = entry.get("reference_hw")
            if rhw is None:
                continue
            if (int(rhw[0]), int(rhw[1])) != target_hw:
                raise EvaluationRunError(
                    f"pairs[{i}].{direction}.reference_hw {tuple(rhw)} != model target "
                    f"{target_hw} (flow grid would be re-scaled)"
                )


def _check_population(per_variant_pair_ids: Mapping[str, Sequence[Any]]) -> None:
    """Reject a variant whose pair population / order differs from the reference."""
    names = list(per_variant_pair_ids)
    if not names:
        raise EvaluationRunError("no variants to compare")
    ref = list(per_variant_pair_ids[names[0]])
    for name in names[1:]:
        got = list(per_variant_pair_ids[name])
        if got != ref:
            raise EvaluationRunError(
                f"population mismatch for variant {name!r}: pair id/order differs "
                "from the reference variant"
            )


#: canonical data role -> the split the CLI must be run with.  A descriptor
#: whose declared role does not match ``--split`` is rejected *before* any heavy
#: work, so a ``role=test`` descriptor can never be evaluated (and then selected)
#: under ``--split val``.
_ROLE_TO_SPLIT: Dict[str, str] = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
    "tune": "tune",
}


def _canonical_role(role: Any) -> str:
    """Normalise a descriptor role (``validation`` -> ``val``)."""
    r = str(role).strip().lower()
    return "val" if r == "validation" else r


def check_role_split(role: Any, split: str) -> str:
    """Reject a descriptor role / ``--split`` mismatch before any heavy work.

    Returns the canonical role.  The **real** role (not the CLI split) is what
    the selection guard must be keyed on: matching them here removes the
    ``role=test`` + ``--split val`` + ``--select`` bypass entirely.
    """
    canonical = _canonical_role(role)
    expected = _ROLE_TO_SPLIT.get(canonical)
    if expected is None:
        raise EvaluationRunError(
            f"descriptor role {role!r} is not a recognised data role"
        )
    if expected != split:
        raise EvaluationRunError(
            f"descriptor role {role!r} resolves to split {expected!r} but --split is "
            f"{split!r}; refusing (role/split mismatch would bypass the selection guard)"
        )
    return canonical


def check_dataset_index_identity(desc: Mapping[str, Any], index_path: str) -> str:
    """Verify the resolved CLI index matches the descriptor's declared identity.

    The descriptor's ``dataset.index_path`` (resolved, symlinks followed) must be
    the same file as ``--dataset-index`` and, when supplied, its
    ``dataset.index_sha256`` must match the resolved bytes.  Returns the resolved
    index SHA256.
    """
    ds = desc["dataset"]
    declared = ds.get("index_path")
    if declared:
        if Path(str(declared)).resolve() != Path(str(index_path)).resolve():
            raise EvaluationRunError(
                "descriptor.dataset.index_path does not resolve to the --dataset-index "
                "file; refusing to score an index the descriptor did not pin"
            )
    resolved_sha = _sha256_file(index_path)
    expected = ds.get("index_sha256")
    if expected:
        if resolved_sha.lower() != str(expected).lower():
            raise EvaluationRunError(
                "resolved index sha256 does not match descriptor.dataset.index_sha256 "
                "(the staged index is not the pinned one)"
            )
    return resolved_sha


def _verify_asset_identity(assets: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the released asset SHA256 / byte size *before* it is loaded.

    Shared by **every** variant (RGB included): the released asset identity must
    be checked before any weight bytes are read.
    """
    from uniflowmatch.models.checkpoint import verify_file_identity

    return verify_file_identity(
        str(assets["asset_path"]),
        expected_sha256=assets.get("asset_sha256"),
        expected_bytes=assets.get("asset_bytes"),
    )


def _assert_fp32_model(model: torch.nn.Module, name: str) -> None:
    """Guarantee the evaluated model stays FP32 (no half / bf16 converted state)."""
    for key, value in model.state_dict().items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise EvaluationRunError(
                f"variant {name!r} state {key!r} is {value.dtype}, expected float32 "
                "(FP32 is required for the deterministic evaluation run)"
            )


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def _resolve_class(dotted: str):
    if ":" in dotted:
        mod, _, attr = dotted.partition(":")
    else:
        mod, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


def _load_model_kwargs(assets: Mapping[str, Any]) -> Dict[str, Any]:
    from uniflowmatch.models.checkpoint import verify_file_identity

    expected = assets.get("config_sha256")
    verify_file_identity(str(assets["config_path"]), expected_sha256=expected)
    with open(str(assets["config_path"])) as fh:
        kwargs = json.load(fh)
    if not isinstance(kwargs, dict):
        raise EvaluationRunError("asset config must be a JSON object of model kwargs")
    return kwargs


def _load_checkpoint_state(path: Any, *, expected_sha256: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Verify + strictly load a trained RGB-D averaged-``x`` checkpoint state.

    Rejects a wrong byte/SHA identity and any non-``best_averaged_x`` kind (a
    ``last_y`` resume checkpoint is never an evaluation checkpoint).
    """
    from uniflowmatch.models.checkpoint import verify_file_identity
    from uniflowmatch.training import rgbd_checkpoint as C

    verify_file_identity(str(path), expected_sha256=expected_sha256)
    payload = C.load_training_checkpoint(path, require_kind="best_averaged_x")
    return dict(payload["model"])


def _build_rgbd(name: str, desc: Mapping[str, Any], device: torch.device):
    """Construct one variant model.  CPU tests monkeypatch this factory."""
    from uniflowmatch.models.checkpoint import load_checkpoint, load_released_state, strict_load_state_dict
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    assets = desc["assets"]
    model_spec = desc["model"]
    model_kwargs = _load_model_kwargs(assets)
    fusion = dict(model_spec.get("depth_fusion_kwargs") or {})
    kind = str(desc["variants"][name]["kind"])
    entry = str(assets.get("dino_entry", "dinov2_vitl14"))
    is_rgb = kind in ("rgb", "rgb_checkpoint")

    with local_dino_hub(
        assets["dino_hub_dir"], entry, str(assets["dino_python_tree_sha256"]), pretrained=False
    ):
        if is_rgb:
            model = _resolve_class(str(model_spec.get("rgb_class", "uniflowmatch.models.ufm:UniFlowMatchConfidence")))(
                **_strip_fusion_kwargs(model_kwargs)
            )
        else:
            model = build_rgbd_model(
                model_kwargs, fusion_kwargs=fusion, model_class=UniFlowMatchConfidence
            )

    if kind == "rgb":
        # the RGB variant consumes the released asset directly: verify SHA256 /
        # byte size BEFORE the weight bytes are read.
        _verify_asset_identity(assets)
        from uniflowmatch.models.checkpoint import extract_state_dict

        released = load_checkpoint(
            str(assets["asset_path"]), weights_only=True, map_location="cpu"
        )
        load_released_state(model, extract_state_dict(released), require_fp32=True)
    elif kind == "rgb_checkpoint":
        # an RGB-trained checkpoint is depth-free: it is loaded *only* from the
        # trained best-averaged-x state below (never the RGB-D migration helper).
        # The released asset identity is still verified (the fixed-comparison
        # guard is kept for every variant).
        _verify_asset_identity(assets)
    else:
        # load_released_rgbd_weights verifies SHA256 / byte size before loading.
        load_released_rgbd_weights(
            model,
            assets["asset_path"],
            expected_sha256=assets.get("asset_sha256"),
            expected_bytes=assets.get("asset_bytes"),
        )

    if kind in ("checkpoint", "rgb_checkpoint"):
        v = desc["variants"][name]
        state = _load_checkpoint_state(v["checkpoint_path"], expected_sha256=v.get("checkpoint_sha256"))
        strict_load_state_dict(model, state, require_fp32=True, check_finite=True)

    model.to(device)
    model.eval()
    return model


def _strip_fusion_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    out.pop("use_depth_fusion", None)
    out.pop("depth_fusion_kwargs", None)
    return out


def _build_variant_model(name: str, desc: Mapping[str, Any], device: torch.device):
    """Public seam used by the CLI; monkeypatched by the CPU dependency tests."""
    return _build_rgbd(name, desc, device)


def _load_dataset(desc: Mapping[str, Any], split: str):
    """Load the split dataset through the existing read-only loader."""
    from uniflowmatch.datasets.rgbd_pairs import RGBDPairDataset

    ds = desc["dataset"]
    return RGBDPairDataset(
        ds["index_path"],
        split=split,
        data_norm_type=str(ds.get("data_norm_type", "dinov2")),
        transform=str(ds.get("transform", "imgnorm")),
        resolution=list(ds["resolution"]),
        depth_window_m=list(ds.get("depth_window_m", (0.0, 1.3))),
        already_dilated=bool(ds.get("already_dilated", True)),
        verify_hashes=bool(ds.get("verify_hashes", True)),
        occlusion_supervision_source=str(ds.get("occlusion_supervision_source", "path")),
    )


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _extract_covis_logits(result: Any) -> Optional[torch.Tensor]:
    covis = getattr(result, "covisibility", None)
    if covis is None:
        return None
    logits = getattr(covis, "logits", None)
    if logits is None:
        return None
    return logits


def evaluate_variant(
    model: torch.nn.Module,
    desc: Mapping[str, Any],
    pairs: Sequence[Any],
    *,
    name: str,
    device: torch.device,
    out_dir: Path,
    events: EventsLogger,
    input_sha: str,
    checkpoint_sha: str,
    source_sha: str,
) -> Dict[str, Any]:
    """Run one variant over every pair; persist per-pair NPZ + pooled metrics."""
    det_cfg = {"determinism": dict(desc["determinism"])}
    H, W = _model_target_hw(desc)
    variant_dir = out_dir / name / "pairs"
    variant_dir.mkdir(parents=True, exist_ok=True)

    flow_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    logit_rows: List[torch.Tensor] = []
    covis_target_rows: List[torch.Tensor] = []
    covis_sup_rows: List[torch.Tensor] = []
    pair_ids: List[Any] = []
    directions: List[str] = []
    per_pair: List[Dict[str, Any]] = []
    total = len(pairs)

    model.eval()
    for i, pair in enumerate(pairs):
        view1, view2, sup = pack_pair_batch([pair])
        view1 = _to_device(view1, device)
        view2 = _to_device(view2, device)
        sup = _to_device(sup, device)

        # a true RGB-only model (depth_fusion is None) must never see depth.
        if getattr(model, "depth_fusion", None) is None:
            view1 = strip_depth_input_keys(view1)
            view2 = strip_depth_input_keys(view2)

        with torch.inference_mode(), deterministic_forward_context(det_cfg):
            result = model(view1, view2)
        pred_flow = result.flow.flow_output.detach().float().cpu()
        logits = _extract_covis_logits(result)
        logits = None if logits is None else logits.detach().float().cpu()

        target = sup["flow"].detach().float().cpu()
        mask = sup["mask"].detach().cpu()
        pids = list(sup["pair_ids"])
        dirs = ["fwd", "bwd"][: pred_flow.shape[0]]
        covis_target = sup.get("covis_target")
        covis_sup = sup.get("covis_supervision")
        covis_target = None if covis_target is None else covis_target.detach().cpu()
        covis_sup = None if covis_sup is None else covis_sup.detach().cpu()

        res = masked_teacher_flow_epe(pred_flow, target, mask, pair_ids=pids, directions=dirs)
        covis_out = None
        if logits is not None and covis_target is not None and covis_sup is not None:
            covis_out = covis_metrics(logits, covis_target, covis_sup)

        npz_path = variant_dir / f"{_safe_id(pids[0])}.npz"
        payload = {
            "pred_flow": pred_flow.numpy(),
            "pair_id": _as_str_array(pids),
            "direction": _as_str_array(dirs),
            "input_sha": input_sha,
            "checkpoint_sha": checkpoint_sha,
            "source_sha": source_sha,
        }
        if logits is not None:
            payload["pred_covis_logits"] = logits.numpy()
        _save_npz(npz_path, payload)

        flow_rows.append(pred_flow)
        target_rows.append(target)
        mask_rows.append(mask)
        if logits is not None:
            logit_rows.append(logits)
            covis_target_rows.append(covis_target)
            covis_sup_rows.append(covis_sup)
        pair_ids.extend(pids)
        directions.extend(dirs)
        per_pair.append(
            {
                "pair_id": str(pids[0]),
                "teacher_epe": res.teacher_epe,
                "n_missing": res.n_missing,
                "n_failure_pairs": res.n_failure_pairs,
                "covis_bce": None if covis_out is None else covis_out.bce,
                "covis_f1": None if covis_out is None else covis_out.f1,
                "npz": str(npz_path),
            }
        )
        events.emit(
            stage=f"eval_{name}",
            event="processed",
            status="running",
            processed=i + 1,
            total=total,
        )
        events.emit(
            stage=f"eval_{name}",
            event="save",
            status="ok",
            processed=i + 1,
            total=total,
        )

    flow_all = torch.cat(flow_rows, dim=0)
    target_all = torch.cat(target_rows, dim=0)
    mask_all = torch.cat(mask_rows, dim=0)
    logit_all = torch.cat(logit_rows, dim=0) if logit_rows else None
    covis_target_all = torch.cat(covis_target_rows, dim=0) if covis_target_rows else None
    covis_sup_all = torch.cat(covis_sup_rows, dim=0) if covis_sup_rows else None

    covis_full = None
    if logit_all is not None:
        covis_full = covis_metrics(logit_all, covis_target_all, covis_sup_all)

    events.emit(
        stage=f"eval_{name}",
        event="finish",
        status="ok",
        processed=total,
        total=total,
        val_loss=(None if covis_full is None or covis_full.bce is None else float(covis_full.bce)),
    )

    return {
        "name": name,
        "flow": flow_all,
        "target": target_all,
        "mask": mask_all,
        "pair_ids": pair_ids,
        "directions": directions,
        "covis": covis_full,
        "per_pair": per_pair,
    }


def _as_str_array(values: Sequence[Any]):
    import numpy as np

    return np.array([str(v) for v in values], dtype=object)


def _safe_id(value: Any) -> str:
    s = str(value)
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in s]
    out = "".join(keep)
    return out or "pair"


def _save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez(fh, **payload)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(
    *,
    descriptor: Mapping[str, Any],
    index_path: str,
    split: str,
    output_dir: str,
    variants: Sequence[str],
    device_str: str = "cuda",
    gpu: Optional[int] = None,
    select: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    validate_descriptor(descriptor)
    split = str(split).strip().lower()
    if split not in ALLOWED_SPLITS:
        raise EvaluationRunError(f"split {split!r} must be one of {ALLOWED_SPLITS}")

    # The descriptor's *real* role must match the CLI split, and the selection
    # guard is keyed on that role (never on the caller-supplied split): a
    # ``role=test`` descriptor can no longer be run with ``--split val`` +
    # ``--select`` to slip past the guard.
    role = check_role_split(descriptor.get("role"), split)

    requested = [v for v in variants if v]
    if not requested:
        raise EvaluationRunError("no variants requested; refusing an empty variant set")
    duplicates = sorted({v for v in requested if requested.count(v) > 1})
    if duplicates:
        raise EvaluationRunError(f"duplicate variants requested: {duplicates}")
    unknown = [v for v in requested if v not in ALL_VARIANTS]
    if unknown:
        raise EvaluationRunError(f"unknown variants {unknown}; allowed {list(ALL_VARIANTS)}")

    # the output directory must be Git-external (symlinks followed) BEFORE any
    # directory is created or any byte is written.
    out_dir = assert_report_path_external(output_dir, what="evaluation output")

    index = json.loads(Path(str(index_path)).read_text(encoding="utf-8"))
    check_dataset_index_identity(descriptor, index_path)
    check_grid(descriptor, index)
    input_sha = _sha256_file(index_path)
    source_sha = public_source_sha()

    # Apply the configured determinism contract to the WHOLE run -- from before
    # the first CUDA use until after the last model is scored -- and restore every
    # global flag / env var on exit (including on error).  Recording the settings
    # without applying them is not acceptable; determinism_context both applies
    # and restores them here.
    with determinism_context(descriptor) as det_settings:
        if device_str == "cuda" and torch.cuda.is_available() and gpu is not None:
            torch.cuda.set_device(int(gpu))
        device = torch.device(device_str)

        ds = _load_dataset(descriptor, split)
        n = len(ds)
        pairs = [ds[i] for i in range(n if limit is None else min(n, int(limit)))]
        events = EventsLogger(out_dir / "events.jsonl")

        t0 = time.monotonic()
        per_variant: Dict[str, Dict[str, Any]] = {}
        identity: Dict[str, Any] = {}
        for name in requested:
            checkpoint_sha = str(descriptor["variants"][name].get("checkpoint_sha256") or "")
            model = _build_variant_model(name, descriptor, device)
            _assert_fp32_model(model, name)
            info = evaluate_variant(
                model, descriptor, pairs,
                name=name, device=device, out_dir=out_dir, events=events,
                input_sha=input_sha, checkpoint_sha=checkpoint_sha, source_sha=source_sha,
            )
            # record model identity (gate value + a state checksum, never private paths)
            identity[name] = _model_identity(model, name, descriptor)
            per_variant[name] = info
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _check_population({name: info["pair_ids"] for name, info in per_variant.items()})

        ref = requested[0]
        variant_flows = {name: info["flow"] for name, info in per_variant.items()}
        target = per_variant[ref]["target"]
        mask = per_variant[ref]["mask"]
        comparison = compare_variants(
            variant_flows, target, mask, pair_ids=per_variant[ref]["pair_ids"], reference=ref
        )
        for name, info in per_variant.items():
            covis = info["covis"]
            comparison["variants"][name]["covis_population"] = (
                None if covis is None else covis.as_dict()
            )
            comparison["variants"][name]["per_pair"] = info["per_pair"]

        # RGB vs RGB-D-init gate-0 parity
        parity = _parity_check(per_variant)
        comparison["rgb_rgbd_init_parity"] = parity

        # optional sealed-monitor comparability (validation full-population EPE)
        monitor = descriptor.get("monitor")
        monitor_check = None
        if isinstance(monitor, Mapping) and split == "val":
            monitor_check = []
            for name, expected in monitor.items():
                if name not in comparison["variants"]:
                    continue
                got = comparison["variants"][name].get("teacher_epe")
                monitor_check.append(
                    {
                        "variant": name,
                        "sealed_teacher_epe": expected,
                        "measured_full_population_teacher_epe": got,
                        "abs_diff": (None if got is None else abs(float(got) - float(expected))),
                    }
                )
        comparison["monitor_check"] = monitor_check
        comparison["determinism_applied"] = dict(det_settings)
        comparison["identity"] = {
            "input_index_sha256": input_sha,
            "public_source_sha256": source_sha,
            "role": role,
            "split": split,
            "n_pairs": len(pairs),
            "device": str(device),
            "variants": identity,
            "model_identity": identity,
        }

        # selection guard: measuring on test is fine, *selecting* is not.  The
        # guard is keyed on the descriptor's REAL role, never the CLI split.
        selection = None
        if select:
            results = {
                name: _result_from_comparison(comparison["variants"][name])
                for name in per_variant
            }
            selection = select_best_variant(results, data_role=role)

        comparison_path = out_dir / "comparison.json"
        _write_json(comparison_path, comparison)
        timing = {"overall_wall_s": time.monotonic() - t0}

        events.emit(stage="finish", event="done", status="ok", elapsed_wall_s=timing["overall_wall_s"])
        receipt = {
            "unit": "UFM_STEP21B1_RGBD_EVALUATION",
            "status": "finished",
            "role": role,
            "split": split,
            "n_pairs": len(pairs),
            "variants": list(requested),
            "selection": selection,
            "parity_ok": parity["ok"],
            "monitor_check": monitor_check,
            "determinism_applied": dict(det_settings),
            "input_index_sha256": input_sha,
            "public_source_sha256": source_sha,
            "comparison_json": str(comparison_path),
            "timing": timing,
        }
        _write_json(out_dir / "run_receipt.json", receipt)
    return receipt


def _model_identity(model: torch.nn.Module, name: str, desc: Mapping[str, Any]) -> Dict[str, Any]:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    df = getattr(model, "depth_fusion", None)
    gate = None
    if df is not None and hasattr(df, "gate"):
        gate = float(df.gate.detach().reshape(-1)[0])
    return {
        "variant": name,
        "kind": str(desc["variants"][name]["kind"]),
        "modality": "rgb" if df is None else "rgbd",
        "state_sha256": h.hexdigest(),
        "gate": gate,
    }


def _parity_check(per_variant: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    a = per_variant.get("rgb_pretrained")
    b = per_variant.get("rgbd_pretrained_init")
    if a is None or b is None:
        return {"checked": False, "ok": None, "reason": "both variants not evaluated"}
    fa, fb = a["flow"], b["flow"]
    if fa.shape != fb.shape:
        return {"checked": True, "ok": False, "reason": "flow shape mismatch"}
    diff = (fa - fb).abs().max().item()
    return {
        "checked": True,
        "ok": bool(diff <= PARITY_ATOL),
        "max_abs_diff": float(diff),
        "atol": PARITY_ATOL,
    }


def _result_from_comparison(v: Mapping[str, Any]) -> MaskedFlowEPEResult:
    """Rehydrate a MaskedFlowEPEResult-like object for the selection guard.

    ``select_best_variant`` only reads ``teacher_epe`` + the two population
    counts, so a light shim is sufficient and keeps the guard honest.
    """
    from dataclasses import dataclass

    full = v.get("full_population") or {}

    @dataclass(frozen=True)
    class _Pop:
        n_missing: int
        n_failure_pairs: int

    @dataclass(frozen=True)
    class _Shim:
        teacher_epe: Optional[float]
        full_population: _Pop

    return _Shim(
        teacher_epe=v.get("teacher_epe"),
        full_population=_Pop(
            n_missing=int(full.get("n_missing", 0)),
            n_failure_pairs=int(full.get("n_failure_pairs", 0)),
        ),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generic four-variant RGB-D UFM evaluation")
    ap.add_argument("--descriptor", required=True, help="Git-external JSON descriptor")
    ap.add_argument("--dataset-index", required=True, help="Git-external loader index.json")
    ap.add_argument("--split", required=True, choices=list(ALLOWED_SPLITS))
    ap.add_argument("--output-dir", required=True, help="Git-external output directory")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of the four variants")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--select", action="store_true",
                    help="attempt a best-variant selection (refused on a test-only role)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N pairs")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    desc = load_descriptor(args.descriptor)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    try:
        receipt = run(
            descriptor=desc,
            index_path=args.dataset_index,
            split=args.split,
            output_dir=args.output_dir,
            variants=variants,
            device_str=args.device,
            gpu=args.gpu,
            select=bool(args.select),
            limit=args.limit,
        )
    except (SelectionPolicyError, EvaluationRunError, ReportPathError, EvaluationContractError) as exc:
        print(f"EVAL_ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print(
        "eval_ok split={split} n_pairs={n} variants={v} parity={p}".format(
            split=receipt["split"], n=receipt["n_pairs"],
            v=",".join(receipt["variants"]), p=receipt["parity_ok"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
