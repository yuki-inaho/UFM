"""Generic pseudo-teacher evaluation for the RGB-D UFM variants (step 21A).

Purpose
-------
Score a *prediction* flow / covisibility tensor against the dataset's own
"teacher" labels, with no model, loader, trainer or GPU import.  This module is
intentionally pure: it consumes already-computed tensors (for the real UFM runs
that means ``result.flow.flow_output`` ``B2HW`` and ``result.covisibility.logits``
``B1HW`` plus the packed ``ABBA`` supervision produced by ``pack_pair_batch``) and
returns plain, serialisable statistics.

These are **pseudo-teacher metrics, NOT ground truth** (``PSEUDO_TEACHER_NOT_GROUND_TRUTH``).
The teacher flow / covisibility targets are derived from the capture's own
reconstruction, so a low EPE means "agrees with the pseudo teacher", never
"correct".

Contracts enforced here
-----------------------
* The **flow mask is supplied by the dataset** and must be a genuine ``bool``
  tensor; a float/int mask (e.g. a predicted-confidence heatmap) is rejected, so
  evaluation support is never selected by predicted confidence.
* The covisibility supervision mask comes from ``occlusion_supervision_mask`` and
  is **independent of flow confidence**; the covisibility target comes from
  ``non_occluded_mask``.
* **Non-finite predictions inside the valid region are retained as failures**:
  they are counted per pair and never silently dropped, zeroed, or replaced.
* **Empty support -> ``no_data`` / ``None``**; a mean of ``0`` is never fabricated.
* The full population and the finite (complete, uncontaminated) population are
  aggregated separately for every variant, and every per-pair / per-direction
  value is kept.  :func:`compare_variants` intersects the *normal* pair ids across
  all variants and reports each metric / delta on that common cohort; an empty
  intersection yields ``None`` rather than a fabricated ranking.
* ``teacher_epe`` groups the two ABBA directions of a pair and pools by valid
  support *before* averaging over pairs, exactly mirroring the trainer's
  ``val_teacher_epe`` (``supervised_terms`` pooled over the packed rows, then
  mean over pairs), so the two numbers are directly comparable on the same
  pair / order / mask set.

Nothing here embeds private dataset identifiers, host paths, calibration or
counts: every identifier arrives from the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    "ALLOWED_SELECTION_ROLES",
    "PSEUDO_TEACHER_NOT_GROUND_TRUTH",
    "CovisResult",
    "EvaluationContractError",
    "MaskedFlowEPEResult",
    "PairFlowEPE",
    "PopulationStats",
    "ReportPathError",
    "SelectionPolicyError",
    "assert_report_path_external",
    "assert_selection_allowed",
    "compare_variants",
    "covis_metrics",
    "masked_teacher_flow_epe",
    "select_best_variant",
    "write_evaluation_report",
]

#: Every metric in this module is a pseudo-teacher metric, never ground truth.
PSEUDO_TEACHER_NOT_GROUND_TRUTH = True

#: Data roles on which a checkpoint / hyper-parameter selection is permitted.
#: A test-only role (e.g. ``"test"``) is intentionally absent: selection on
#: test-only data is rejected by :func:`assert_selection_allowed`.
ALLOWED_SELECTION_ROLES = frozenset({"train", "val", "validation", "tune"})


# --------------------------------------------------------------------------- #
# typed errors
# --------------------------------------------------------------------------- #
class EvaluationContractError(ValueError):
    """An evaluation input violates the pure-metric contract."""


class SelectionPolicyError(RuntimeError):
    """Checkpoint / variant selection was attempted on non-selection data."""


class ReportPathError(RuntimeError):
    """An evaluation report path is not Git-external (or otherwise unsafe)."""


# --------------------------------------------------------------------------- #
# per-row / per-group record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairFlowEPE:
    """Masked teacher-flow EPE for one row (pair-direction) or one grouped pair.

    ``epe`` is the support-normalised mean EPE (``epe_sum / n_valid``) and is
    ``None`` whenever the row/group is not a complete, uncontaminated
    observation: no usable data at all (``no_data``), a non-finite prediction
    inside the valid mask, or a non-finite target inside the valid mask.  A
    NaN-dropped partial mean is never promoted to the primary metric; the raw
    ``n_valid`` / ``n_nonfinite_pred`` / ``n_nonfinite_target`` / ``epe_sum``
    and the failure counts are still retained.
    """

    pair_id: Any
    direction: Optional[str]
    support: int          # dataset flow-mask pixels
    usable: int           # support pixels with a finite teacher target
    n_valid: int          # usable pixels with a finite prediction
    n_nonfinite_pred: int # failures: finite target, non-finite prediction
    n_nonfinite_target: int
    epe_sum: float        # sum of EPE over the n_valid finite pixels
    epe: Optional[float]  # per-row/group support-normalised EPE (None if no_data)

    @property
    def missing(self) -> bool:
        """True when the row has no usable data at all (``no_data``)."""
        return self.usable == 0

    @property
    def has_failure(self) -> bool:
        """True when a non-finite prediction inside the valid region was kept."""
        return self.n_nonfinite_pred > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "direction": self.direction,
            "support": int(self.support),
            "usable": int(self.usable),
            "n_valid": int(self.n_valid),
            "n_nonfinite_pred": int(self.n_nonfinite_pred),
            "n_nonfinite_target": int(self.n_nonfinite_target),
            "epe_sum": float(self.epe_sum),
            "epe": None if self.epe is None else float(self.epe),
            "missing": bool(self.missing),
            "has_failure": bool(self.has_failure),
        }


@dataclass(frozen=True)
class PopulationStats:
    """Aggregate over a population of :class:`PairFlowEPE` records."""

    n_pairs: int
    n_missing: int
    n_failure_pairs: int
    n_finite_pairs: int
    total_support: int
    total_valid: int
    total_nonfinite_pred: int
    mean_pair_epe: Optional[float]
    support_weighted_epe: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_pairs": int(self.n_pairs),
            "n_missing": int(self.n_missing),
            "n_failure_pairs": int(self.n_failure_pairs),
            "n_finite_pairs": int(self.n_finite_pairs),
            "total_support": int(self.total_support),
            "total_valid": int(self.total_valid),
            "total_nonfinite_pred": int(self.total_nonfinite_pred),
            "mean_pair_epe": None if self.mean_pair_epe is None else float(self.mean_pair_epe),
            "support_weighted_epe": (
                None if self.support_weighted_epe is None else float(self.support_weighted_epe)
            ),
        }


def _aggregate(records: Sequence[PairFlowEPE]) -> PopulationStats:
    epe_sum = sum(r.epe_sum for r in records)
    total_valid = sum(r.n_valid for r in records)
    finite = [r.epe for r in records if r.epe is not None]
    mean_pair = (sum(finite) / len(finite)) if finite else None
    pooled = (epe_sum / total_valid) if total_valid > 0 else None
    return PopulationStats(
        n_pairs=len(records),
        n_missing=sum(1 for r in records if r.missing),
        n_failure_pairs=sum(1 for r in records if r.has_failure),
        n_finite_pairs=len(finite),
        total_support=sum(r.support for r in records),
        total_valid=total_valid,
        total_nonfinite_pred=sum(r.n_nonfinite_pred for r in records),
        mean_pair_epe=mean_pair,
        support_weighted_epe=pooled,
    )


@dataclass(frozen=True)
class MaskedFlowEPEResult:
    """Result of :func:`masked_teacher_flow_epe`.

    ``rows`` keeps every pair-direction value (ABBA order preserved).  ``groups``
    pools the two directions of one ``pair_id`` by valid support; when no
    ``pair_ids`` are supplied each row is its own group.
    """

    rows: Tuple[PairFlowEPE, ...]
    groups: Tuple[PairFlowEPE, ...]
    row_full_population: PopulationStats
    row_finite_population: PopulationStats
    full_population: PopulationStats
    finite_population: PopulationStats

    @property
    def teacher_epe(self) -> Optional[float]:
        """Comparable to the trainer's ``val_teacher_epe`` (group-pooled mean)."""
        return self.finite_population.mean_pair_epe

    @property
    def n_missing(self) -> int:
        return self.full_population.n_missing

    @property
    def n_failure_pairs(self) -> int:
        return self.full_population.n_failure_pairs

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pseudo_teacher_not_ground_truth": PSEUDO_TEACHER_NOT_GROUND_TRUTH,
            "teacher_epe": None if self.teacher_epe is None else float(self.teacher_epe),
            "rows": [r.as_dict() for r in self.rows],
            "groups": [r.as_dict() for r in self.groups],
            "row_full_population": self.row_full_population.as_dict(),
            "row_finite_population": self.row_finite_population.as_dict(),
            "full_population": self.full_population.as_dict(),
            "finite_population": self.finite_population.as_dict(),
        }


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _require_bool_mask(name: str, mask: torch.Tensor) -> None:
    if not isinstance(mask, torch.Tensor):
        raise EvaluationContractError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise EvaluationContractError(
            f"{name} must be a bool tensor (got {mask.dtype}); a float/int mask is "
            "rejected so flow support can never be selected by predicted confidence"
        )


def _require_flow(name: str, flow: torch.Tensor) -> None:
    if not isinstance(flow, torch.Tensor):
        raise EvaluationContractError(f"{name} must be a torch.Tensor, got {type(flow).__name__}")
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise EvaluationContractError(f"{name} must be (N,2,H,W), got {tuple(flow.shape)}")


def _check_meta(name: str, values: Optional[Sequence[Any]], n: int) -> Optional[List[Any]]:
    if values is None:
        return None
    seq = list(values)
    if len(seq) != n:
        raise EvaluationContractError(f"{name} length {len(seq)} != batch {n}")
    return seq


# --------------------------------------------------------------------------- #
# masked teacher-flow EPE
# --------------------------------------------------------------------------- #
def masked_teacher_flow_epe(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    flow_mask: torch.Tensor,
    *,
    pair_ids: Optional[Sequence[Any]] = None,
    directions: Optional[Sequence[str]] = None,
) -> MaskedFlowEPEResult:
    """Masked teacher-flow EPE preserving every pair / direction.

    ``pred_flow`` / ``target_flow`` are ``(N, 2, H, W)`` and ``flow_mask`` is a
    bool ``(N, H, W)`` straight from the dataset's flow label.  The prediction is
    allowed to be non-finite: such pixels inside the valid region are **counted as
    failures** (retained, never dropped or zeroed).  Empty support yields
    ``epe=None`` (``no_data``), never a fabricated ``0``.
    """
    _require_flow("pred_flow", pred_flow)
    _require_flow("target_flow", target_flow)
    if target_flow.shape != pred_flow.shape:
        raise EvaluationContractError(
            f"target_flow {tuple(target_flow.shape)} != pred_flow {tuple(pred_flow.shape)}"
        )
    _require_bool_mask("flow_mask", flow_mask)
    n, _, h, w = pred_flow.shape
    if flow_mask.shape != (n, h, w):
        raise EvaluationContractError(
            f"flow_mask {tuple(flow_mask.shape)} != (N,H,W) {(n, h, w)}"
        )
    pids = _check_meta("pair_ids", pair_ids, n)
    dirs = _check_meta("directions", directions, n)

    finite_pred = torch.isfinite(pred_flow).all(dim=1)      # (N,H,W)
    finite_target = torch.isfinite(target_flow).all(dim=1)  # (N,H,W)
    usable = flow_mask & finite_target
    valid_pred = usable & finite_pred

    diff = pred_flow - target_flow
    epe_map = torch.linalg.norm(diff, dim=1)                # (N,H,W); nan/inf where bad
    epe_map = torch.where(valid_pred, epe_map, torch.zeros_like(epe_map))

    rows: List[PairFlowEPE] = []
    for i in range(n):
        support = int(flow_mask[i].sum())
        usable_i = int(usable[i].sum())
        n_nonfinite_pred = int((usable[i] & ~finite_pred[i]).sum())
        n_nonfinite_target = int((flow_mask[i] & ~finite_target[i]).sum())
        n_valid = int(valid_pred[i].sum())
        epe_sum = float(epe_map[i].sum())
        # Primary EPE is only reported for a complete, uncontaminated row: any
        # non-finite prediction/target inside the valid mask makes it None so a
        # NaN-dropped partial mean is never used as the primary metric.
        if n_valid > 0 and n_nonfinite_pred == 0 and n_nonfinite_target == 0:
            epe = epe_sum / n_valid
        else:
            epe = None
        rows.append(
            PairFlowEPE(
                pair_id=(pids[i] if pids is not None else i),
                direction=(dirs[i] if dirs is not None else None),
                support=support,
                usable=usable_i,
                n_valid=n_valid,
                n_nonfinite_pred=n_nonfinite_pred,
                n_nonfinite_target=n_nonfinite_target,
                epe_sum=epe_sum,
                epe=epe,
            )
        )

    groups = _group_rows(rows)

    row_finite = [r for r in rows if r.epe is not None]
    group_finite = [g for g in groups if g.epe is not None]
    return MaskedFlowEPEResult(
        rows=tuple(rows),
        groups=tuple(groups),
        row_full_population=_aggregate(rows),
        row_finite_population=_aggregate(row_finite),
        full_population=_aggregate(groups),
        finite_population=_aggregate(group_finite),
    )


def _group_rows(rows: Sequence[PairFlowEPE]) -> List[PairFlowEPE]:
    """Pool rows sharing a ``pair_id`` by valid support, preserving first order."""
    order: List[Any] = []
    buckets: Dict[Any, List[PairFlowEPE]] = {}
    for r in rows:
        key = r.pair_id
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(r)
    groups: List[PairFlowEPE] = []
    for key in order:
        bucket = buckets[key]
        support = sum(r.support for r in bucket)
        usable = sum(r.usable for r in bucket)
        n_valid = sum(r.n_valid for r in bucket)
        n_nonfinite_pred = sum(r.n_nonfinite_pred for r in bucket)
        n_nonfinite_target = sum(r.n_nonfinite_target for r in bucket)
        epe_sum = sum(r.epe_sum for r in bucket)
        if n_valid > 0 and n_nonfinite_pred == 0 and n_nonfinite_target == 0:
            epe = epe_sum / n_valid
        else:
            epe = None
        groups.append(
            PairFlowEPE(
                pair_id=key,
                direction=None,
                support=support,
                usable=usable,
                n_valid=n_valid,
                n_nonfinite_pred=n_nonfinite_pred,
                n_nonfinite_target=n_nonfinite_target,
                epe_sum=epe_sum,
                epe=epe,
            )
        )
    return groups


# --------------------------------------------------------------------------- #
# covisibility metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CovisResult:
    """Covisibility BCE + precision / recall / F1 + support / positive / negative.

    The target is ``non_occluded_mask`` and the supervision mask is
    ``occlusion_supervision_mask``; both are supplied by the dataset and the
    supervision mask is independent of flow confidence.  ``None`` metrics mean the
    corresponding denominator was zero (``no_data``), never a fabricated ``0``.

    ``support`` is the **original supervision support** (``supervision_mask``
    pixels) and is kept separate from ``usable``, the **finite usable** subset
    (supervision pixels with a finite logit).  When the valid supervision region
    contains non-finite logits the primary ``bce`` / ``precision`` / ``recall`` /
    ``f1`` are all ``None`` (a NaN-dropped partial metric is never primary),
    ``n_nonfinite_logits`` retains the failures, and both support counts remain
    available.
    """

    bce: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    support: int          # original supervision support (supervision_mask pixels)
    usable: int           # finite usable subset (supervision & finite logits)
    positives: int
    negatives: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    n_nonfinite_logits: int

    @property
    def missing(self) -> bool:
        return self.support == 0

    @property
    def has_failure(self) -> bool:
        """True when a non-finite logit was found inside the supervision region."""
        return self.n_nonfinite_logits > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pseudo_teacher_not_ground_truth": PSEUDO_TEACHER_NOT_GROUND_TRUTH,
            "bce": None if self.bce is None else float(self.bce),
            "precision": None if self.precision is None else float(self.precision),
            "recall": None if self.recall is None else float(self.recall),
            "f1": None if self.f1 is None else float(self.f1),
            "support": int(self.support),
            "usable": int(self.usable),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "true_positive": int(self.true_positive),
            "false_positive": int(self.false_positive),
            "false_negative": int(self.false_negative),
            "true_negative": int(self.true_negative),
            "n_nonfinite_logits": int(self.n_nonfinite_logits),
            "missing": bool(self.missing),
            "has_failure": bool(self.has_failure),
        }


def covis_metrics(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> CovisResult:
    """Covisibility BCE / precision / recall / F1 on the supervision mask only.

    ``logits`` is ``(N,1,H,W)`` (single channel, e.g. ``covisibility.logits``) or
    ``(N,H,W)``.  ``target_mask`` (``non_occluded_mask``) and ``supervision_mask``
    (``occlusion_supervision_mask``) are bool ``(N,H,W)``.  Non-finite logits
    inside the supervision region are retained/counted as failures and force the
    primary BCE / precision / recall / F1 to ``None`` (no NaN-dropped partial
    metric is used); ``support`` keeps the original supervision count while
    ``usable`` records the finite subset.  Empty support -> all metrics ``None``.
    """
    if logits.dim() == 4:
        if logits.shape[1] != 1:
            raise EvaluationContractError(f"covis logits must be (N,1,H,W), got {tuple(logits.shape)}")
        logits = logits.squeeze(1)
    if logits.dim() != 3:
        raise EvaluationContractError(
            f"covis logits must be (N,1,H,W) or (N,H,W), got {tuple(logits.shape)}"
        )
    _require_bool_mask("target_mask", target_mask)
    _require_bool_mask("supervision_mask", supervision_mask)
    if target_mask.shape != logits.shape or supervision_mask.shape != logits.shape:
        raise EvaluationContractError("covis target/supervision mask shape must match squeezed logits")

    finite = torch.isfinite(logits)
    supervised = supervision_mask & finite
    n_nonfinite = int((supervision_mask & ~finite).sum())
    support = int(supervision_mask.sum())   # original supervision support
    usable = int(supervised.sum())          # finite usable subset

    pos = supervised & target_mask
    neg = supervised & (~target_mask)
    positives = int(pos.sum())
    negatives = int(neg.sum())

    pred = logits > float(threshold)
    tp = int((pos & pred).sum())
    fp = int((neg & pred).sum())
    fn = int((pos & ~pred).sum())
    tn = int((neg & ~pred).sum())

    # A non-finite logit inside the supervision region, or an entirely empty
    # finite subset, means no complete metric is available: the primary metrics
    # are None and the failures / support counts are retained below.
    if usable == 0 or n_nonfinite > 0:
        bce = precision = recall = f1 = None
    else:
        target_f = target_mask.to(logits.dtype)
        bce_t = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[supervised], target_f[supervised], reduction="mean"
        )
        bce = float(bce_t)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
        if precision is None or recall is None:
            f1 = None
        elif (precision + recall) == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

    return CovisResult(
        bce=bce,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        usable=usable,
        positives=positives,
        negatives=negatives,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        n_nonfinite_logits=n_nonfinite,
    )


# --------------------------------------------------------------------------- #
# variant comparison (identical pair / order / mask conditions)
# --------------------------------------------------------------------------- #
def compare_variants(
    variant_flows: Mapping[str, torch.Tensor],
    target_flow: torch.Tensor,
    flow_mask: torch.Tensor,
    *,
    pair_ids: Optional[Sequence[Any]] = None,
    directions: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """Score several variants under *identical* pair / order / mask conditions.

    Every variant prediction must share the batch size and the exact ``pair_ids``
    ordering of the shared supervision; a mismatch is rejected (a variant must
    never be scored on a different population).  Returns per-variant full results
    and, in addition, the **common population**: the intersection of the normal
    (finite, uncontaminated) pair ids across *all* variants, on which each metric
    and delta is computed.  An empty intersection yields ``None`` (no fabricated
    ranking); every variant's own full / finite / missing / failure populations
    are still retained.
    """
    if not isinstance(variant_flows, Mapping) or not variant_flows:
        raise EvaluationContractError("variant_flows must be a non-empty mapping name -> (N,2,H,W)")

    n = int(target_flow.shape[0]) if target_flow.dim() == 4 else -1
    pids = _check_meta("pair_ids", pair_ids, n) if n >= 0 else None

    variants: Dict[str, MaskedFlowEPEResult] = {}
    for name, pred in variant_flows.items():
        _require_flow(f"variant {name!r}", pred)
        if pred.shape != target_flow.shape:
            raise EvaluationContractError(
                f"variant {name!r} shape {tuple(pred.shape)} != target {tuple(target_flow.shape)} "
                "(variant population mismatch)"
            )
        variants[str(name)] = masked_teacher_flow_epe(
            pred, target_flow, flow_mask, pair_ids=pids, directions=directions
        )

    keys = list(variants.keys())
    ref_name = str(reference) if reference is not None else keys[0]
    if ref_name not in variants:
        raise EvaluationContractError(f"reference variant {ref_name!r} not among {keys}")

    # population identity check: every variant must expose the same group ids/order.
    ref_ids = [g.pair_id for g in variants[ref_name].groups]
    for name, res in variants.items():
        ids = [g.pair_id for g in res.groups]
        if ids != ref_ids:
            raise EvaluationContractError(
                f"variant {name!r} population mismatch: pair id/order differs from reference"
            )

    # Normal (finite, uncontaminated) pair ids per variant: excluded pairs are
    # missing (no_data) or carry a failure inside the valid mask.  The common
    # cohort is the intersection across *all* variants, so a variant that does
    # not observe a pair normally can never widen or shrink the comparison.
    normal_ids: Dict[str, List[Any]] = {
        name: [g.pair_id for g in res.groups if g.epe is not None]
        for name, res in variants.items()
    }
    common = set(normal_ids[ref_name])
    for name in keys:
        common &= set(normal_ids[name])
    common_ids = [g.pair_id for g in variants[ref_name].groups if g.pair_id in common]

    common_records: Dict[str, List[PairFlowEPE]] = {
        name: [g for g in res.groups if g.pair_id in common]
        for name, res in variants.items()
    }
    common_population = {name: _aggregate(recs) for name, recs in common_records.items()}
    common_epe: Dict[str, Optional[float]] = {
        name: stats.mean_pair_epe for name, stats in common_population.items()
    }

    ref_epe = common_epe[ref_name]
    deltas: Dict[str, Optional[float]] = {}
    for name in keys:
        epe = common_epe[name]
        deltas[name] = float(epe - ref_epe) if (epe is not None and ref_epe is not None) else None

    return {
        "pseudo_teacher_not_ground_truth": PSEUDO_TEACHER_NOT_GROUND_TRUTH,
        "reference": ref_name,
        "variants": {name: res.as_dict() for name, res in variants.items()},
        "common_pair_ids": common_ids,
        "common_epe": common_epe,
        "common_population": {name: stats.as_dict() for name, stats in common_population.items()},
        "teacher_epe_delta_vs_reference": deltas,
    }


# --------------------------------------------------------------------------- #
# checkpoint / variant selection policy
# --------------------------------------------------------------------------- #
def assert_selection_allowed(data_role: Any, *, purpose: str = "checkpoint selection") -> None:
    """Refuse selection when the data role is not a selection role.

    Test-only / held-out roles (e.g. ``"test"``) are never permitted; this stops a
    checkpoint from being chosen on test-only data.
    """
    role = str(data_role).strip().lower()
    if role in ALLOWED_SELECTION_ROLES:
        return
    raise SelectionPolicyError(
        f"{purpose} is not allowed on data role {data_role!r}; choosing a checkpoint on "
        f"test-only data is forbidden (allowed roles: {sorted(ALLOWED_SELECTION_ROLES)})"
    )


def select_best_variant(
    results: Mapping[str, MaskedFlowEPEResult],
    *,
    data_role: Any,
    lower_is_better: bool = True,
    purpose: str = "checkpoint selection",
) -> Optional[str]:
    """Return the best variant name by ``teacher_epe``, subject to the selection guard.

    Raises :class:`SelectionPolicyError` on a test-only role and returns ``None``
    when no *eligible* variant exists (no fabricated winner).

    A variant is eligible only when it is complete and failure-free: its
    ``teacher_epe`` is defined **and** it has neither missing (``no_data``) pairs
    nor pairs carrying a non-finite-prediction failure.  An incomplete / failed
    candidate is therefore never selected just because it happens to have a low
    EPE.
    """
    assert_selection_allowed(data_role, purpose=purpose)
    scored = [
        (name, res.teacher_epe)
        for name, res in results.items()
        if res.teacher_epe is not None
        and res.full_population.n_missing == 0
        and res.full_population.n_failure_pairs == 0
    ]
    if not scored:
        return None
    scored.sort(key=lambda kv: kv[1], reverse=not lower_is_better)
    return scored[0][0]


# --------------------------------------------------------------------------- #
# safe, Git-external report writing
# --------------------------------------------------------------------------- #
def assert_report_path_external(path: Any, *, what: str = "evaluation report") -> Path:
    """Resolve ``path`` and refuse any location inside the public repository.

    Delegates to the repository's Git-external guard, which follows symlinks, so a
    link pointing back into the repo is rejected too.  The raised message never
    echoes the offending path.
    """
    try:
        from uniflowmatch.training.rgbd_config import assert_git_external
    except Exception as exc:  # pragma: no cover - import environment failure
        raise ReportPathError(
            f"cannot verify that the {what} path is Git-external (guard unavailable)"
        ) from exc
    try:
        return assert_git_external(path, what=what)
    except ReportPathError:
        raise
    except Exception as exc:
        raise ReportPathError(
            f"{what} path is not Git-external (symlinks followed); refusing to write"
        ) from exc


def write_evaluation_report(
    path: Any,
    payload: Mapping[str, Any],
    *,
    what: str = "evaluation report",
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    """Write a JSON evaluation report to a verified Git-external location.

    Safety: the target is checked Git-external (symlinks followed) *before* any
    directory is created or any byte is written, and an existing file is not
    overwritten unless ``overwrite=True``.
    """
    target = assert_report_path_external(path, what=what)
    if target.exists() and not overwrite:
        raise ReportPathError(f"{what} already exists and overwrite=False")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=int(indent), sort_keys=True, default=str)
    target.write_text(text + "\n", encoding="utf-8")
    return target
