"""Hand-computed tests for the variance decomposition and stopping rule.

The decomposition uses nested random-effects (method-of-moments) variance
components, so the question/model terms are the LATENT between-group components,
not the observed variances of group means (which would double-count grader
noise). The hand calculations below follow the ANOVA estimators in the module.
"""

from __future__ import annotations

import numpy as np
import pytest

from grader_variance import (
    decompose,
    k_selection_curve,
    stopping_rule_text,
)
from grader_variance.decomposition import MAX_CURVE_K

# --- decompose: 2D frozen single-completion case ---------------------------


def test_decompose_2d_hand_computed() -> None:
    # rows [[1,0],[1,1]], one-way ANOVA (n=2 items, k=2 reps):
    #   grader (MS_error) = mean(var([1,0]), var([1,1])) = mean(0.5, 0.0) = 0.25
    #   item means 1.0, 1.0? no: means are 0.5 and 1.0, grand 0.75.
    #   SS_item = k*((0.5-0.75)^2 + (1.0-0.75)^2) = 2*0.125 = 0.25 ; MS_item = 0.25
    #   question = (MS_item - MS_error)/k = (0.25 - 0.25)/2 = 0  (NOT 0.125!)
    # The observed variance of item means (0.125) would wrongly attribute grader
    # noise to the question term; the component estimate correctly reports 0.
    comp = decompose([[1, 0], [1, 1]])
    assert comp.grader == pytest.approx(0.25)
    assert comp.model == pytest.approx(0.0)
    assert comp.question == pytest.approx(0.0)
    assert comp.total == pytest.approx(0.25)
    assert comp.share("grader") == pytest.approx(1.0)


def test_decompose_3d_hand_computed() -> None:
    # x[0] = [[1,0],[1,0]], x[1] = [[1,1],[0,0]] (n=2, m=2, k=2). Nested ANOVA:
    #   grader (MS_error)                  = mean(0.5,0.5,0,0) = 0.25
    #   MS_model = 0.5 ; model = (0.5-0.25)/k = 0.125
    #   MS_item  = 0   ; question = (0 - 0.5)/(m*k) = -0.125 -> clamped to 0
    x = np.array([[[1, 0], [1, 0]], [[1, 1], [0, 0]]], dtype=float)
    comp = decompose(x)
    assert comp.grader == pytest.approx(0.25)
    assert comp.model == pytest.approx(0.125)
    assert comp.question == pytest.approx(0.0)
    assert comp.total == pytest.approx(0.375)


def test_decompose_clean_signal_dominates() -> None:
    # Items truly differ, grader nearly reproducible -> question dominates.
    x = np.array([[[0.9, 1.0], [0.95, 1.0]], [[0.1, 0.0], [0.05, 0.0]]], dtype=float)
    comp = decompose(x)
    assert comp.question > comp.grader
    assert comp.share("question") > 0.9


def test_decompose_shares_sum_to_one() -> None:
    x = np.array([[[1, 0], [0.5, 1]], [[1, 1], [0, 0.5]]], dtype=float)
    comp = decompose(x)
    assert comp.total > 0
    total_share = comp.share("grader") + comp.share("model") + comp.share("question")
    assert total_share == pytest.approx(1.0)


def test_grader_share_equals_one_minus_icc11() -> None:
    # Cross-validation against the independently-validated ICC(1,1): for the
    # single-completion case the one-way random-effects identity gives
    #   question / (question + grader) == ICC(1,1)  ->  grader_share == 1 - ICC(1,1)
    # This ties the new component estimator to rater-agreement's ICC(1,1), which
    # is itself tested against a hand-computed reference (15/17).
    from raters import intra

    for rows in ([[0, 2], [4, 6]], [[1, 2, 3], [7, 8, 6]], [[0.2, 0.9], [0.1, 0.0]]):
        comp = decompose(rows)
        icc = intra.icc_1_1(rows)
        assert comp.share("question") == pytest.approx(icc)
        assert comp.share("grader") == pytest.approx(1.0 - icc)


def test_decompose_requires_two_items() -> None:
    with pytest.raises(ValueError, match="2 items"):
        decompose([[1, 0]])


def test_decompose_requires_two_repeats() -> None:
    with pytest.raises(ValueError, match="2 grader repeats"):
        decompose([[1], [0]])


# --- k-selection curve + stopping rule -------------------------------------


def test_k_selection_stopping_rule_hand_computed() -> None:
    # rows [[0.0, 0.5], [0.5, 1.0]], m=1, k=2:
    #   grader (MS_error) = mean(var([0,0.5]), var([0.5,1])) = mean(0.125,0.125)=0.125
    #   item means 0.25, 0.75 grand 0.5: SS_item = 2*((0.25-0.5)^2+(0.75-0.5)^2)=0.25
    #     MS_item = 0.25 ; question = (0.25 - 0.125)/2 = 0.0625  (the component)
    #   tol=0.05 -> threshold = 0.05*0.0625 = 0.003125 ; k >= 0.125/0.003125 = 40
    curve = k_selection_curve([[0.0, 0.5], [0.5, 1.0]], tol=0.05)
    assert curve.grader_variance == pytest.approx(0.125)
    assert curve.item_variance == pytest.approx(0.0625)
    assert curve.recommended_k == 40
    assert curve.capped is False


def test_k_selection_curve_se_decreases_like_one_over_k() -> None:
    curve = k_selection_curve([[0.0, 0.5], [0.5, 1.0]], tol=0.05, k_max=8)
    ses = [p.score_se for p in curve.points]
    assert all(ses[i] > ses[i + 1] for i in range(len(ses) - 1))
    # grader VARIANCE contribution halves from k=1 to k=2 (it is a variance term)
    c1 = curve.points[0].grader_variance_contribution
    c2 = curve.points[1].grader_variance_contribution
    assert c2 == pytest.approx(c1 / 2)
    # and it is a summand of score_se ** 2, not of score_se
    floor = curve.points[0].score_se ** 2 - c1
    assert curve.points[1].score_se ** 2 == pytest.approx(floor + c2)


def test_null_result_grader_variance_zero_recommends_one() -> None:
    # Each item perfectly self-consistent, items differ: grader term is zero,
    # so no repeats are needed. This is the null result to publish as-is.
    curve = k_selection_curve([[1, 1, 1], [0, 0, 0]], tol=0.05)
    assert curve.grader_variance == pytest.approx(0.0)
    assert curve.recommended_k == 1


def test_no_between_item_signal_recommends_none() -> None:
    # Items share the same mean, grader noisy: no between-item signal exists, so
    # no finite k makes the grader negligible relative to it.
    curve = k_selection_curve([[1, 0], [0, 1]], tol=0.05)
    assert curve.item_variance == pytest.approx(0.0)
    assert curve.grader_variance > 0
    assert curve.recommended_k is None
    assert "cannot distinguish" in stopping_rule_text(curve)


def test_k_selection_is_bounded_when_recommended_k_huge() -> None:
    # Tiny-but-positive between-item signal with substantial grader noise pushes
    # the recommended k above the cap; it must clamp, never allocate unboundedly.
    # item within var 1 each; item means differ by ~1.01 so MS_item ~ 1.02 > 1.
    rows = [[-0.70710678, 0.70710678], [0.30289322, 1.71710678]]
    curve = k_selection_curve(rows, tol=0.05)
    assert curve.item_variance > 0
    assert curve.capped is True
    assert curve.recommended_k == MAX_CURVE_K
    assert len(curve.points) <= MAX_CURVE_K


def test_k_max_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k_max"):
        k_selection_curve([[0.0, 0.5], [0.5, 1.0]], k_max=0)


def test_stopping_rule_text_mentions_miller_and_k() -> None:
    curve = k_selection_curve([[0.0, 0.5], [0.5, 1.0]], tol=0.05)
    text = stopping_rule_text(curve)
    assert "Miller" in text
    assert "k = 40" in text
    assert "sigma2_grader" in text
