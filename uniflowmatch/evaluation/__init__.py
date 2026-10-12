"""Generic, inference-free RGB-D evaluation metrics (step 21A).

The public surface is :mod:`uniflowmatch.evaluation.rgbd_evaluation`.  Every
number produced here is a **pseudo-teacher metric, NOT ground truth**: the
"teacher" flow / covisibility labels are the dataset's own self/covisibility
targets, not an independent GT annotation.  The metric code is kept strictly
separate from model inference so a prediction tensor can be scored without any
model import.
"""

from uniflowmatch.evaluation.rgbd_evaluation import (
    ALLOWED_SELECTION_ROLES,
    PSEUDO_TEACHER_NOT_GROUND_TRUTH,
    CovisResult,
    EvaluationContractError,
    MaskedFlowEPEResult,
    PairFlowEPE,
    PopulationStats,
    ReportPathError,
    SelectionPolicyError,
    assert_report_path_external,
    assert_selection_allowed,
    compare_variants,
    covis_metrics,
    masked_teacher_flow_epe,
    select_best_variant,
    write_evaluation_report,
)

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
