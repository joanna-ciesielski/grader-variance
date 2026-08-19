"""Inspect metrics for intra-grader (judge self-consistency) reliability.

These are thin adapters. Every statistic is computed by the ``rater-agreement``
library (DOI 10.5281/zenodo.21983269), which is itself tested against published
reference values (Feinstein & Cicchetti 1990; Byrt, Bishop & Carlin 1993;
Krippendorff 2011). This module only reshapes Inspect's unreduced epoch scores
into the (n_items, k_repeats) layout that library expects and returns the result
as an Inspect :data:`~inspect_ai.scorer.Value`.

Each metric is declared ``@metric(scores="unreduced")`` so it receives one score
per sample per epoch. In the fixed-completion harness (see
:mod:`grader_variance.harness`) each epoch is an independent grader re-scoring of
the **same frozen completion**, so these measure grader self-inconsistency in
isolation from model-sampling variance.

Prior art. That LLM judges are self-inconsistent across repeated runs is
established, not a finding here — see Rating Roulette (arXiv 2510.27106) and
Reliability without Validity (arXiv 2606.19544). These metrics quantify that
instability inside Inspect; they do not claim to discover it.

Relation to Inspect core. Inspect already ships ``krippendorff_alpha()`` for
agreement *across multiple judges*. The metrics here are for *repeats of one
judge* (intra-rater), which core does not provide.
"""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Value,
    metric,
    value_to_float,
)
from raters import intra

from ._reshape import label_matrix, numeric_matrix
from .decomposition import decompose

__all__ = [
    "flip_rate",
    "icc_1_1",
    "test_retest",
    "pabak",
    "prevalence_index",
    "grader_variance_share",
]


@metric(scores="unreduced")
def flip_rate(threshold: float | None = None) -> Metric:
    """Proportion of items whose grader decision changes across repeats.

    With ``threshold=None`` (default) grades are compared as raw labels: an item
    flips if the grader did not assign the identical label on every repeat. For a
    binary model-graded scorer (C/I) this is exactly the decision flip rate.

    With a numeric ``threshold`` grades are first mapped to floats
    (Inspect's :func:`value_to_float`) and binarized as pass (>= threshold) /
    fail (< threshold); an item flips only if both outcomes occur across its
    repeats. Use this for graded/continuous scorers where the wobble that matters
    is the one that crosses the pass mark.

    Args:
        threshold: Pass mark for decision-level flipping, or None for label-level
            flipping.
    """

    def compute(scores: list[SampleScore]) -> Value:
        if threshold is None:
            return float(intra.flip_rate(label_matrix(scores)))
        return float(intra.flip_rate(numeric_matrix(scores), threshold=threshold))

    return compute


@metric(scores="unreduced")
def icc_1_1(to_float: Callable[[Value], float] | None = None) -> Metric:
    """ICC(1,1): one-way intraclass correlation across a grader's own repeats.

    High values mean item identity, not run-to-run grader noise, drives the
    scores. The natural intra-rater form: repeats of an item are interchangeable
    draws from the same grader. Grades are mapped to floats before computation.

    Returns NaN when the statistic is mathematically undefined (zero total
    variance — e.g. the grader gave every item the same grade on every repeat),
    rather than a misleading 1.0.

    Args:
        to_float: Optional grade-to-float mapping; defaults to Inspect's
            :func:`value_to_float` (C/I/P -> 1.0/0.0/0.5).
    """
    convert = to_float or value_to_float()

    def compute(scores: list[SampleScore]) -> Value:
        matrix = numeric_matrix(scores, convert)
        try:
            return float(intra.icc_1_1(matrix))
        except intra.UndefinedStatistic:
            return float("nan")

    return compute


@metric(scores="unreduced")
def test_retest() -> Metric:
    """Mean pairwise agreement of the grader with itself across repeats.

    For each item, the fraction of unordered pairs of repeats that assigned the
    same label, averaged over items. 1.0 is perfect self-consistency. This is raw
    (not chance-corrected) agreement on the grader's own labels — the first thing
    to look at for a stability check, alongside :func:`icc_1_1`.
    """

    def compute(scores: list[SampleScore]) -> Value:
        return float(intra.test_retest_agreement(label_matrix(scores)))

    return compute


@metric(scores="unreduced")
def grader_variance_share(to_float: Callable[[Value], float] | None = None) -> Metric:
    """Fraction of score variance that is grader re-scoring noise (headline).

    On a fixed-completion regrade log, the one-way random-effects share of the
    grader component in the between-item + grader variance of a single grade:

        share = sigma2_grader / (sigma2_grader + sigma2_question)

    where ``sigma2_grader`` and ``sigma2_question`` are the **latent** variance
    components estimated by :func:`grader_variance.decompose` (method-of-moments
    ANOVA). Using the latent between-item component — rather than the observed
    variance of item means, which equals ``sigma2_question + sigma2_grader / k``
    and would double-count grader noise — is what makes this the grader's true
    share. Equivalently, ``share = 1 - ICC(1,1)``: a share near 0 means the grader
    is stable and the spread reflects real item differences; a share near 1 means
    the reported spread is mostly the grader disagreeing with itself.

    Scope. Assumes one frozen completion per item; for the full
    question/model/grader split with ``model_epochs > 1`` use
    :func:`grader_variance.grades_array` + :func:`grader_variance.decompose`.
    Returns NaN if there is no variance at all or fewer than two items/repeats.

    Args:
        to_float: Optional grade-to-float mapping; defaults to
            :func:`value_to_float`.
    """
    convert = to_float or value_to_float()

    def compute(scores: list[SampleScore]) -> Value:
        matrix = numeric_matrix(scores, convert)
        try:
            return decompose(matrix).share("grader")
        except ValueError:
            # Fewer than 2 items or 2 repeats: variance components undefined.
            return float("nan")

    return compute


# --- kappa-paradox diagnostics (order-independent, all repeat-pairs) --------
#
# These diagnose whether a raw self-agreement percentage is trustworthy or is
# being inflated by skewed pass/fail prevalence (the kappa paradox). They are
# computed over ALL unordered pairs of the grader's repeats per item, summed
# across items, so they use every repeat and do not depend on epoch order
# (repeated draws of one grader on a frozen completion are exchangeable — there
# is no "first" vs "second" pass). The resulting co-occurrence table is
# symmetric by construction; the *bias* index (asymmetry between two raters) is
# therefore identically zero here and is intentionally not offered as an
# intra-grader metric — use :func:`grader_variance.compare_judges` for bias
# between two distinct judges.


def _pair_counts(
    scores: list[SampleScore], convert: Callable[[Value], float], threshold: float
) -> tuple[float, float, float]:
    """Return (both-pass, both-fail, mixed) counts over all repeat-pairs per item."""
    matrix = numeric_matrix(scores, convert)
    both_pass = both_fail = mixed = 0.0
    for row in matrix:
        if len(row) < 2:
            raise ValueError("paradox diagnostics need at least 2 repeats per item")
        n_pass = sum(1 for v in row if v >= threshold)
        n_fail = len(row) - n_pass
        both_pass += n_pass * (n_pass - 1) / 2.0
        both_fail += n_fail * (n_fail - 1) / 2.0
        mixed += n_pass * n_fail
    return both_pass, both_fail, mixed


@metric(scores="unreduced")
def pabak(
    threshold: float = 0.5, to_float: Callable[[Value], float] | None = None
) -> Metric:
    """Prevalence-adjusted bias-adjusted kappa for the grader's pass/fail decisions.

    Computes ``2 * p_o - 1`` (Byrt, Bishop & Carlin 1993) where ``p_o`` is the
    fraction of the grader's repeat-pairs that agree, over all unordered pairs of
    repeats per item. Comparing this with a raw self-agreement percentage
    separates "the grader disagrees with itself" from "the pass/fail marginals
    make chance agreement large".

    Args:
        threshold: Pass mark used to binarize grades.
        to_float: Optional grade-to-float mapping; defaults to
            :func:`value_to_float`.
    """
    convert = to_float or value_to_float()

    def compute(scores: list[SampleScore]) -> Value:
        both_pass, both_fail, mixed = _pair_counts(scores, convert, threshold)
        total = both_pass + both_fail + mixed
        if total == 0.0:
            return float("nan")
        p_o = (both_pass + both_fail) / total
        return 2.0 * p_o - 1.0

    return compute


@metric(scores="unreduced")
def prevalence_index(
    threshold: float = 0.5, to_float: Callable[[Value], float] | None = None
) -> Metric:
    """Prevalence index of the grader's pass/fail decisions (Byrt et al. 1993).

    ``|both-pass - both-fail| / n_pairs`` over all unordered repeat-pairs. Near 1
    means one outcome dominates — the condition under which kappa collapses while
    raw agreement stays high.

    Args:
        threshold: Pass mark used to binarize grades.
        to_float: Optional grade-to-float mapping.
    """
    convert = to_float or value_to_float()

    def compute(scores: list[SampleScore]) -> Value:
        both_pass, both_fail, mixed = _pair_counts(scores, convert, threshold)
        total = both_pass + both_fail + mixed
        if total == 0.0:
            return float("nan")
        return abs(both_pass - both_fail) / total

    return compute
