"""Metric-adapter tests against PUBLISHED reference values.

The point of these tests is to prove the Inspect port is faithful: each metric,
fed unreduced SampleScores that encode a published table, must reproduce the
published number. The underlying statistics are validated inside the
``rater-agreement`` library; here we verify the adapter (reshape + call + return)
does not distort them.

References
----------
Feinstein, A. R. & Cicchetti, D. V. (1990). High agreement but low kappa.
    J Clin Epidemiol 43(6), 543-549.
Byrt, T., Bishop, J. & Carlin, J. B. (1993). Bias, prevalence and kappa.
    J Clin Epidemiol 46(5), 423-429.
"""

from __future__ import annotations

import math

import pytest

from grader_variance import (
    flip_rate,
    grader_variance_share,
    icc_1_1,
    pabak,
    prevalence_index,
)
from grader_variance import test_retest as retest_metric

from ._helpers import sample_scores


def _pair_rows(a: int, b: int, c: int, d: int) -> list[list[str]]:
    """Rows of (first, second) grader decisions reproducing a 2x2 table.

    Table [[a, b], [c, d]] with C=pass, I=fail: a both-pass, b pass-then-fail,
    c fail-then-pass, d both-fail.
    """
    rows: list[list[str]] = []
    rows += [["C", "C"]] * a
    rows += [["C", "I"]] * b
    rows += [["I", "C"]] * c
    rows += [["I", "I"]] * d
    return rows


# --- flip_rate --------------------------------------------------------------


def test_flip_rate_labels_hand_computed() -> None:
    # (A,A) stable, (A,B) flips, (C,C) stable -> 1/3
    scores = sample_scores([["A", "A"], ["A", "B"], ["C", "C"]])
    assert flip_rate()(scores) == pytest.approx(1 / 3)


def test_flip_rate_threshold_decision_vs_label() -> None:
    # Grades (3,4),(4,5),(1,2) with pass mark 3.5: only (3,4) straddles it.
    scores = sample_scores([[3, 4], [4, 5], [1, 2]])
    assert flip_rate()(scores) == pytest.approx(1.0)  # every label changes
    assert flip_rate(threshold=3.5)(scores) == pytest.approx(1 / 3)


def test_flip_rate_binary_grades() -> None:
    # C/I decisions: item flips iff it is not all-C or all-I.
    scores = sample_scores([["C", "I", "I"], ["C", "C", "C"], ["I", "C", "I"]])
    assert flip_rate()(scores) == pytest.approx(2 / 3)


# --- test_retest ------------------------------------------------------------


def test_test_retest_hand_computed() -> None:
    # Item1 (A,A,B) -> 1/3 of pairs agree; Item2 (B,B,B) -> 1. Mean = 2/3.
    scores = sample_scores([["A", "A", "B"], ["B", "B", "B"]])
    assert retest_metric()(scores) == pytest.approx(2 / 3)


# --- icc_1_1 ----------------------------------------------------------------


def test_icc_1_1_hand_computed() -> None:
    # rows (1,2),(3,4),(5,6) -> ICC(1,1) = 15/17 (see rater-agreement test_intra).
    scores = sample_scores([[1, 2], [3, 4], [5, 6]])
    assert icc_1_1()(scores) == pytest.approx(15 / 17)


def test_icc_1_1_undefined_is_nan() -> None:
    # Every grade identical -> zero total variance -> undefined -> NaN, not 1.0.
    scores = sample_scores([[3, 3, 3], [3, 3, 3]])
    assert math.isnan(icc_1_1()(scores))


# --- pabak / prevalence_index (Feinstein & Cicchetti 1990) ------------------
#
# The metrics aggregate over all unordered repeat-pairs. With exactly 2 repeats
# per item there is one pair per item, so _pair_rows reproduces the published
# 2x2 tables and PABAK / prevalence index match the published values. (The bias
# index is not an intra-grader metric — it lives in compare_judges, tested in
# test_analysis.py — because repeated draws of one grader are exchangeable.)

# FC 1990 Table 1 (balanced) and Table 2 (skewed); same raw agreement 0.85.
FC_TABLE_1 = (40, 9, 6, 45)
FC_TABLE_2 = (80, 10, 5, 5)


def test_pabak_fc1990_both_tables_identical() -> None:
    # Both tables agree on 85/100 -> PABAK = 2*0.85 - 1 = 0.70 for BOTH.
    assert pabak()(sample_scores(_pair_rows(*FC_TABLE_1))) == pytest.approx(0.70)
    assert pabak()(sample_scores(_pair_rows(*FC_TABLE_2))) == pytest.approx(0.70)


def test_prevalence_index_fc1990() -> None:
    # PI = |both-pass - both-fail| / n_pairs : Table1 = 0.05, Table2 = 0.75.
    assert prevalence_index()(sample_scores(_pair_rows(*FC_TABLE_1))) == pytest.approx(
        0.05
    )
    assert prevalence_index()(sample_scores(_pair_rows(*FC_TABLE_2))) == pytest.approx(
        0.75
    )


def test_pabak_uses_all_repeat_pairs_not_just_first_two() -> None:
    # 3 repeats per item, so C(3,2)=3 pairs per item -- the metric must use all of
    # them, order-independently. One item all-pass, one item 2-pass/1-fail:
    #   item A (C,C,C): pairs pp,pp,pp -> 3 agree
    #   item B (C,C,I): pairs pp, pf, pf -> 1 agree, 2 mixed
    # total pairs = 6, agreeing = 4 -> p_o = 4/6, PABAK = 2*(4/6) - 1 = 1/3.
    scores = sample_scores([["C", "C", "C"], ["C", "C", "I"]])
    assert pabak()(scores) == pytest.approx(1 / 3)


# --- grader_variance_share (random-effects component share) -----------------


def test_grader_variance_share_pure_grader_noise() -> None:
    # Items share the same mean but disagree internally: no between-item signal,
    # all variance is grader -> share = 1.0.
    scores = sample_scores([[1, 0], [0, 1]])
    assert grader_variance_share()(scores) == pytest.approx(1.0)


def test_grader_variance_share_pure_question_variance() -> None:
    # Each item perfectly self-consistent but items differ -> share = 0.0.
    scores = sample_scores([[1, 1, 1], [0, 0, 0]])
    assert grader_variance_share()(scores) == pytest.approx(0.0)


def test_grader_variance_share_hand_computed() -> None:
    # rows [[0,2],[4,6]], k=2 (one-way random-effects components):
    #   grader (MS_error)  = mean(var([0,2]), var([4,6])) = mean(2, 2) = 2
    #   item means 1, 5 (grand 3): SS_item = k*((1-3)^2+(5-3)^2) = 2*8 = 16
    #     MS_item = 16/1 = 16 ; question = (MS_item - MS_error)/k = (16-2)/2 = 7
    #   share = grader/(grader+question) = 2/(2+7) = 2/9.
    scores = sample_scores([[0, 2], [4, 6]])
    assert grader_variance_share()(scores) == pytest.approx(2 / 9)


def test_grader_variance_share_no_variance_is_nan() -> None:
    scores = sample_scores([[1, 1], [1, 1]])
    assert math.isnan(grader_variance_share()(scores))


def test_metrics_reject_multi_completion_logs() -> None:
    # When model_epochs > 1 the harness tags samples with the model-sampling
    # index; intra-grader metrics must refuse rather than silently conflate
    # model variance into the grader term.
    from inspect_ai.scorer import SampleScore, Score

    from grader_variance._reshape import MODEL_SAMPLE_KEY

    scores = [
        SampleScore(
            score=Score(value="C"), sample_id=0, sample_metadata={MODEL_SAMPLE_KEY: 1}
        ),
        SampleScore(
            score=Score(value="I"), sample_id=0, sample_metadata={MODEL_SAMPLE_KEY: 2}
        ),
    ]
    with pytest.raises(ValueError, match="one frozen completion per item"):
        flip_rate()(scores)
