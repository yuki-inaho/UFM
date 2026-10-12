"""CPU contract tests for the generic RGB-D pseudo-teacher evaluation (step 21A).

Pure / synthetic: no model, no GPU, no real data, no private identifiers.  The
"teacher" tensors are hand-built constants so every expectation is computable by
hand.  Every metric here is a *pseudo-teacher* metric, not ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "UniCeption"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uniflowmatch.evaluation.rgbd_evaluation import (  # noqa: E402
    PSEUDO_TEACHER_NOT_GROUND_TRUTH,
    EvaluationContractError,
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

N, H, W = 2, 3, 4


def _bool(shape, value=True):
    return torch.full(shape, bool(value), dtype=torch.bool)


def _const_flow(n, h, w, dx, dy):
    f = torch.zeros(n, 2, h, w)
    f[:, 0] = float(dx)
    f[:, 1] = float(dy)
    return f


# --------------------------------------------------------------------------- #
# hand-computed constant-flow EPE
# --------------------------------------------------------------------------- #
def test_constant_flow_epe_hand_computed():
    pred = torch.zeros(N, 2, H, W)
    target = _const_flow(N, H, W, 3.0, 4.0)  # |(3,4)| = 5 exactly
    mask = _bool((N, H, W))
    res = masked_teacher_flow_epe(pred, target, mask)
    assert res.teacher_epe == pytest.approx(5.0)
    assert res.full_population.total_support == N * H * W
    assert res.full_population.n_missing == 0
    for r in res.rows:
        assert r.epe == pytest.approx(5.0)
        assert r.n_valid == H * W


def test_grouping_pools_directions_then_averages_over_pairs():
    # pair p0: fwd EPE 5 over 2 px, bwd EPE 0 over 4 px -> (10+0)/6
    # pair p1: fwd EPE 2 over 4 px, bwd EPE 2 over 4 px -> 16/8 = 2
    pred = torch.zeros(4, 2, 2, 2)
    target = torch.zeros(4, 2, 2, 2)
    target[0, 0] = 5.0
    target[2, 0] = 2.0
    target[3, 0] = 2.0
    mask = torch.zeros(4, 2, 2, dtype=torch.bool)
    mask[0, 0, :] = True   # 2 px
    mask[1] = True         # 4 px
    mask[2] = True         # 4 px
    mask[3] = True         # 4 px
    res = masked_teacher_flow_epe(
        pred, target, mask,
        pair_ids=["p0", "p0", "p1", "p1"],
        directions=["fwd", "bwd", "fwd", "bwd"],
    )
    p0 = next(g for g in res.groups if g.pair_id == "p0")
    p1 = next(g for g in res.groups if g.pair_id == "p1")
    assert p0.epe == pytest.approx(10.0 / 6.0)
    assert p0.n_valid == 6
    assert p1.epe == pytest.approx(2.0)
    assert p1.n_valid == 8
    assert [r.direction for r in res.rows] == ["fwd", "bwd", "fwd", "bwd"]
    assert [g.pair_id for g in res.groups] == ["p0", "p1"]
    assert res.teacher_epe == pytest.approx((10.0 / 6.0 + 2.0) / 2.0)


# --------------------------------------------------------------------------- #
# bidirectional ABBA handling via pack_pair_batch
# --------------------------------------------------------------------------- #
def test_abba_bidirectional_handling_with_pack_pair_batch():
    from uniflowmatch.training.rgbd_trainer import pack_pair_batch

    def view(pair_id, dx, instance):
        return {
            "pair_id": pair_id,
            "instance": instance,
            "data_norm_type": "dinov2",
            "flow": _const_flow(1, 1, 1, dx, 0.0)[0],  # (2,1,1)
            "non_occluded_mask": _bool((1, 1), True),
            "occlusion_supervision_mask": _bool((1, 1), True),
        }

    views = [
        [view("p0", 3.0, "p0_a"), view("p0", 7.0, "p0_b")],
        [view("p1", 1.0, "p1_a"), view("p1", 2.0, "p1_b")],
    ]
    _v1, _v2, sup = pack_pair_batch(views)
    assert sup["pair_ids"] == ["p0", "p0", "p1", "p1"]  # A0 B0 A1 B1 order
    assert [float(sup["flow"][i, 0, 0, 0]) for i in range(4)] == [3.0, 7.0, 1.0, 2.0]

    res = masked_teacher_flow_epe(
        sup["flow"].clone(), sup["flow"], sup["mask"],
        pair_ids=sup["pair_ids"],
        directions=["fwd", "bwd", "fwd", "bwd"],
    )
    assert res.teacher_epe == pytest.approx(0.0)
    assert [r.pair_id for r in res.rows] == ["p0", "p0", "p1", "p1"]
    assert [r.direction for r in res.rows] == ["fwd", "bwd", "fwd", "bwd"]


# --------------------------------------------------------------------------- #
# covisibility BCE / precision / recall / F1 with B1HW logits
# --------------------------------------------------------------------------- #
def test_covis_b1hw_hand_computed():
    logits = torch.zeros(1, 1, 1, 3)      # B1HW, single row
    logits[0, 0, 0] = torch.tensor([10.0, 10.0, -10.0])
    target = _bool((1, 1, 3), False)
    target[0, 0, 0] = True
    target[0, 0, 2] = True                # targets T F T
    sup = _bool((1, 1, 3), True)
    r = covis_metrics(logits, target, sup)
    assert (r.true_positive, r.false_positive, r.false_negative, r.true_negative) == (1, 1, 1, 0)
    assert r.positives == 2 and r.negatives == 1 and r.support == 3
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(0.5)
    assert r.f1 == pytest.approx(0.5)
    assert r.bce is not None and r.bce > 0.0


def test_covis_4d_and_3d_equivalent():
    torch.manual_seed(0)
    lg = torch.randn(2, 1, 3, 4)
    tgt = torch.rand(2, 3, 4) > 0.5
    sup = torch.rand(2, 3, 4) > 0.4
    a = covis_metrics(lg, tgt, sup)
    b = covis_metrics(lg.squeeze(1), tgt, sup)
    assert a.as_dict()["f1"] == b.as_dict()["f1"]
    assert a.support == b.support


def test_covis_empty_support_is_none_not_zero():
    logits = torch.randn(1, 1, 2, 2)
    r = covis_metrics(logits, _bool((1, 2, 2), True), _bool((1, 2, 2), False))
    assert r.support == 0 and r.missing
    assert r.bce is None and r.precision is None and r.recall is None and r.f1 is None
    assert r.positives == 0 and r.negatives == 0


# --------------------------------------------------------------------------- #
# mask independence: dataset mask, bool-only, never predicted confidence
# --------------------------------------------------------------------------- #
def test_flow_mask_must_be_bool_not_confidence():
    with pytest.raises(EvaluationContractError):
        masked_teacher_flow_epe(
            torch.zeros(N, 2, H, W), torch.zeros(N, 2, H, W), torch.rand(N, H, W)
        )


def test_covis_supervision_must_be_bool_and_drives_support():
    logits = torch.zeros(1, 1, 1, 2)
    logits[0, 0, 0] = torch.tensor([20.0, -20.0])
    target = _bool((1, 1, 2), False)
    sup = _bool((1, 1, 2), False)
    sup[0, 0, 1] = True  # only pixel 1 supervised: target F, pred F -> TN
    r = covis_metrics(logits, target, sup)
    assert r.support == 1
    assert r.true_negative == 1 and r.false_positive == 0 and r.positives == 0
    with pytest.raises(EvaluationContractError):
        covis_metrics(logits, target, torch.rand(1, 1, 2))


# --------------------------------------------------------------------------- #
# empty / all-invalid
# --------------------------------------------------------------------------- #
def test_empty_support_no_fabricated_zero():
    res = masked_teacher_flow_epe(
        torch.zeros(N, 2, H, W), torch.zeros(N, 2, H, W), _bool((N, H, W), False)
    )
    assert res.teacher_epe is None
    assert res.full_population.n_missing == N
    assert res.full_population.mean_pair_epe is None
    assert res.full_population.support_weighted_epe is None


def test_all_invalid_target_is_no_data_but_not_failure():
    target = _const_flow(N, H, W, 1.0, 0.0)
    target[0] = float("nan")
    target[1] = float("nan")
    res = masked_teacher_flow_epe(torch.zeros(N, 2, H, W), target, _bool((N, H, W)))
    assert res.teacher_epe is None
    assert res.full_population.n_missing == N
    assert res.full_population.total_nonfinite_pred == 0  # non-finite target, not failure


def test_population_full_vs_finite_separated():
    target = _const_flow(2, 1, 1, 3.0, 4.0)
    mask = torch.zeros(2, 1, 1, dtype=torch.bool)
    mask[1] = True  # pair0 empty, pair1 present
    res = masked_teacher_flow_epe(
        torch.zeros(2, 2, 1, 1), target, mask, pair_ids=["p0", "p1"]
    )
    assert res.full_population.n_pairs == 2
    assert res.full_population.n_missing == 1
    assert res.finite_population.n_pairs == 1
    assert res.teacher_epe == pytest.approx(5.0)  # only the finite pair contributes


# --------------------------------------------------------------------------- #
# non-finite prediction retention (never dropped / zeroed)
# --------------------------------------------------------------------------- #
def test_nonfinite_prediction_retained_as_failure():
    pred = _const_flow(1, 1, 2, 3.0, 4.0).reshape(1, 2, 1, 2)  # EPE 5 everywhere
    pred[0, 0, 0, 1] = float("nan")                            # one failure
    target = _const_flow(1, 1, 2, 0.0, 0.0).reshape(1, 2, 1, 2)
    mask = _bool((1, 1, 2))
    res = masked_teacher_flow_epe(pred, target, mask)
    r = res.rows[0]
    assert r.support == 2
    assert r.n_nonfinite_pred == 1
    assert r.n_valid == 1
    assert r.epe_sum == pytest.approx(5.0)      # raw partial sum is retained ...
    assert r.epe is None                        # ... but never the primary metric
    assert r.has_failure
    assert res.full_population.total_nonfinite_pred == 1
    assert res.full_population.n_failure_pairs == 1
    assert res.teacher_epe is None              # no complete observation here


def test_nan_target_inside_support_does_not_pollute_sum():
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)
    target[0, 0, 0, 0] = float("nan")
    res = masked_teacher_flow_epe(pred, target, _bool((1, 1, 1)))
    assert res.rows[0].missing            # target un-usable -> no_data
    assert res.teacher_epe is None


# --------------------------------------------------------------------------- #
# shape / dtype errors
# --------------------------------------------------------------------------- #
def test_shape_and_dtype_errors():
    with pytest.raises(EvaluationContractError):
        masked_teacher_flow_epe(torch.zeros(N, 3, H, W), torch.zeros(N, 2, H, W), _bool((N, H, W)))
    with pytest.raises(EvaluationContractError):
        masked_teacher_flow_epe(torch.zeros(N, 2, H, W), torch.zeros(N, 2, H, W), _bool((N, H, W + 1)))
    with pytest.raises(EvaluationContractError):
        masked_teacher_flow_epe(
            torch.zeros(N, 2, H, W), torch.zeros(N, 2, H, W), _bool((N, H, W)), pair_ids=["only-one"]
        )
    with pytest.raises(EvaluationContractError):
        covis_metrics(torch.zeros(N, 2, H, W), _bool((N, H, W)), _bool((N, H, W)))


# --------------------------------------------------------------------------- #
# variant population identity (four variants under identical conditions)
# --------------------------------------------------------------------------- #
def test_compare_four_variants_identical_conditions():
    target = _const_flow(N, H, W, 1.0, 0.0)
    mask = _bool((N, H, W))
    variants = {
        "rgb": torch.zeros(N, 2, H, W),
        "rgbd_init": _const_flow(N, H, W, 0.5, 0.0),
        "off_trained": _const_flow(N, H, W, 1.0, 0.0),
        "on_trained": _const_flow(N, H, W, 1.0, 0.0),
    }
    out = compare_variants(variants, target, mask, reference="rgb")
    assert out["reference"] == "rgb"
    assert set(out["variants"]) == set(variants)
    assert out["variants"]["rgb"]["teacher_epe"] == pytest.approx(1.0)
    assert out["variants"]["on_trained"]["teacher_epe"] == pytest.approx(0.0)
    assert out["teacher_epe_delta_vs_reference"]["on_trained"] == pytest.approx(-1.0)
    assert out["pseudo_teacher_not_ground_truth"] is True


def test_variant_population_mismatch_rejected():
    target = _const_flow(N, H, W, 1.0, 0.0)
    mask = _bool((N, H, W))
    variants = {
        "ok": torch.zeros(N, 2, H, W),
        "wrong_shape": torch.zeros(N, 2, H, W + 1),
    }
    with pytest.raises(EvaluationContractError, match="mismatch"):
        compare_variants(variants, target, mask)


# --------------------------------------------------------------------------- #
# selection policy: never select on test-only data
# --------------------------------------------------------------------------- #
def test_selection_guard_rejects_test_only_role():
    assert_selection_allowed("validation")
    with pytest.raises(SelectionPolicyError):
        assert_selection_allowed("test")
    with pytest.raises(SelectionPolicyError):
        assert_selection_allowed("Test_Only")


def test_select_best_variant_respects_role_and_no_data():
    target = _const_flow(N, H, W, 1.0, 0.0)
    mask = _bool((N, H, W))
    res = {
        "a": masked_teacher_flow_epe(torch.zeros(N, 2, H, W), target, mask),
        "b": masked_teacher_flow_epe(_const_flow(N, H, W, 1.0, 0.0), target, mask),
    }
    assert select_best_variant(res, data_role="validation") == "b"
    with pytest.raises(SelectionPolicyError):
        select_best_variant(res, data_role="test")
    empty = {"a": masked_teacher_flow_epe(torch.zeros(N, 2, H, W), target, _bool((N, H, W), False))}
    assert select_best_variant(empty, data_role="validation") is None


# --------------------------------------------------------------------------- #
# safe Git-external report writing
# --------------------------------------------------------------------------- #
def test_report_writer_rejects_in_repo_output():
    with pytest.raises(ReportPathError):
        write_evaluation_report(_REPO / "temp" / "should_not_write.json", {"x": 1})


def test_report_writer_rejects_symlink_into_repo(tmp_path):
    link = tmp_path / "link_into_repo"
    link.symlink_to(_REPO / "temp", target_is_directory=True)
    with pytest.raises(ReportPathError):
        assert_report_path_external(link / "report.json")
    with pytest.raises(ReportPathError):
        write_evaluation_report(link / "report.json", {"x": 1})


def test_report_writer_writes_git_external(tmp_path):
    out = write_evaluation_report(tmp_path / "report.json", {"metric": 1.25})
    assert out.exists()
    assert json.loads(out.read_text())["metric"] == pytest.approx(1.25)
    with pytest.raises(ReportPathError):
        write_evaluation_report(tmp_path / "report.json", {"metric": 2.0})  # no overwrite
    write_evaluation_report(tmp_path / "report.json", {"metric": 2.0}, overwrite=True)


def test_pseudo_teacher_label_present_in_payloads():
    target = _const_flow(1, 1, 1, 1.0, 0.0)
    res = masked_teacher_flow_epe(torch.zeros(1, 2, 1, 1), target, _bool((1, 1, 1)))
    assert res.as_dict()["pseudo_teacher_not_ground_truth"] is True
    assert PSEUDO_TEACHER_NOT_GROUND_TRUTH is True
    c = covis_metrics(torch.zeros(1, 1, 1, 1), _bool((1, 1, 1)), _bool((1, 1, 1)))
    assert c.as_dict()["pseudo_teacher_not_ground_truth"] is True


# --------------------------------------------------------------------------- #
# common population: intersection of normal pair ids across ALL variants
# --------------------------------------------------------------------------- #
def _pairs_6_target_mask():
    target = torch.zeros(6, 2, 1, 2)
    mask = _bool((6, 1, 2))
    pair_ids = ["p", "p", "q", "q", "r", "r"]
    directions = ["fwd", "bwd", "fwd", "bwd", "fwd", "bwd"]
    return target, mask, pair_ids, directions


def test_common_population_intersection_golden():
    target, mask, pair_ids, directions = _pairs_6_target_mask()
    good = torch.zeros(6, 2, 1, 2)
    good[:, 0] = torch.tensor([1.0, 1.0, 50.0, 50.0, 3.0, 3.0]).view(6, 1, 1)
    bad = torch.zeros(6, 2, 1, 2)
    bad[:, 0] = torch.tensor([2.0, 2.0, 0.0, 0.0, 4.0, 4.0]).view(6, 1, 1)
    bad[2, 0, 0, 0] = float("nan")  # pair q fwd: one non-finite prediction
    out = compare_variants(
        {"good": good, "bad": bad}, target, mask,
        pair_ids=pair_ids, directions=directions, reference="good",
    )
    # q is failed in bad -> normal cohort is only p and r
    assert out["common_pair_ids"] == ["p", "r"]
    assert out["common_epe"]["good"] == pytest.approx(2.0)
    assert out["common_epe"]["bad"] == pytest.approx(3.0)
    assert out["teacher_epe_delta_vs_reference"]["bad"] == pytest.approx(1.0)
    # each variant's own populations / failures are still retained
    assert out["variants"]["bad"]["finite_population"]["n_pairs"] == 2
    assert out["variants"]["bad"]["full_population"]["n_failure_pairs"] == 1
    assert out["variants"]["good"]["finite_population"]["n_pairs"] == 3


def test_compare_one_variant_all_nan_common_empty():
    target = torch.zeros(4, 2, 1, 2)
    mask = _bool((4, 1, 2))
    pair_ids = ["p", "p", "q", "q"]
    good = torch.ones(4, 2, 1, 2)
    bad = torch.full((4, 2, 1, 2), float("nan"))
    out = compare_variants({"good": good, "bad": bad}, target, mask, pair_ids=pair_ids)
    assert out["common_pair_ids"] == []
    assert out["common_epe"]["good"] is None
    assert out["common_epe"]["bad"] is None
    assert out["teacher_epe_delta_vs_reference"]["bad"] is None


def test_compare_all_no_data_common_none():
    target = torch.zeros(4, 2, 1, 2)
    mask = _bool((4, 1, 2), False)
    out = compare_variants(
        {"good": torch.zeros(4, 2, 1, 2), "bad": torch.zeros(4, 2, 1, 2)},
        target, mask, pair_ids=["p", "p", "q", "q"],
    )
    assert out["common_pair_ids"] == []
    assert out["common_epe"]["good"] is None
    assert out["teacher_epe_delta_vs_reference"]["good"] is None
    assert out["variants"]["good"]["full_population"]["n_missing"] == 2


def test_group_one_direction_failure_primary_none():
    target = torch.zeros(2, 2, 1, 2)
    mask = _bool((2, 1, 2))
    pred = torch.zeros(2, 2, 1, 2)
    pred[0, 0, 0, 0] = float("nan")  # fwd direction fails inside the group
    res = masked_teacher_flow_epe(
        pred, target, mask, pair_ids=["q", "q"], directions=["fwd", "bwd"]
    )
    assert res.rows[0].epe is None and res.rows[0].has_failure
    assert res.rows[1].epe == pytest.approx(0.0)  # the healthy direction is retained
    group = res.groups[0]
    assert group.epe is None
    assert group.n_nonfinite_pred == 1
    assert res.teacher_epe is None


def test_masked_out_nan_is_irrelevant():
    target = torch.zeros(1, 2, 1, 2)
    mask = torch.zeros(1, 1, 2, dtype=torch.bool)
    mask[0, 0, 0] = True               # only pixel 0 is supervised
    pred = torch.zeros(1, 2, 1, 2)
    pred[0, 0, 0, 1] = float("nan")    # NaN outside the valid mask
    res = masked_teacher_flow_epe(pred, target, mask)
    assert res.rows[0].n_nonfinite_pred == 0
    assert res.rows[0].epe == pytest.approx(0.0)
    assert res.teacher_epe == pytest.approx(0.0)


def test_select_best_variant_rejects_failed_candidate():
    target = torch.zeros(4, 2, 1, 2)
    mask = _bool((4, 1, 2))
    pair_ids = ["p", "p", "q", "q"]
    clean = torch.full((4, 2, 1, 2), 1.0)
    cheat = torch.zeros(4, 2, 1, 2)
    cheat[2, 0, 0, 0] = float("nan")  # low EPE but one failed pair
    res = {
        "clean": masked_teacher_flow_epe(clean, target, mask, pair_ids=pair_ids),
        "cheat": masked_teacher_flow_epe(cheat, target, mask, pair_ids=pair_ids),
    }
    assert res["cheat"].teacher_epe == pytest.approx(0.0)  # deceptively low
    assert res["cheat"].full_population.n_failure_pairs == 1
    assert select_best_variant(res, data_role="validation") == "clean"
    with pytest.raises(SelectionPolicyError):
        select_best_variant(res, data_role="test")


def test_covis_partial_invalid_support_counts():
    logits = torch.zeros(1, 1, 2, 2)
    logits[0, 0, 0, 0] = float("nan")  # one non-finite logit inside supervision
    target = _bool((1, 2, 2), True)
    sup = _bool((1, 2, 2), True)
    r = covis_metrics(logits, target, sup)
    assert r.support == 4          # original supervision support
    assert r.usable == 3           # finite usable subset
    assert r.n_nonfinite_logits == 1
    assert r.has_failure
    assert r.bce is None and r.precision is None and r.recall is None and r.f1 is None
    d = r.as_dict()
    assert d["support"] == 4 and d["usable"] == 3
