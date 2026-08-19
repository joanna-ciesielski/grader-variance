"""Variance decomposition and the grader-repeat stopping rule.

This is the open contribution of the project. The intra-grader *metrics* quantify
judge self-inconsistency (established prior art: Rating Roulette arXiv 2510.27106;
Reliability without Validity arXiv 2606.19544). What is not established is:

1. How much of the spread in a benchmark's headline score is the grader, versus
   which questions were sampled, versus model sampling on a fixed question. Miller
   ("Adding Error Bars to Evals", arXiv 2411.00640) decomposes the question and
   model-sampling components and explicitly assumes scores are already computed —
   i.e. excludes the grader component. This module adds it.
2. How many grader repeats are enough. Miller gives a stopping rule for model
   resamples: once the per-item sampling variance divided by the number of
   resamples is small relative to the between-item variance, more resamples stop
   helping. There is no published analogue for graders. We state the grader
   stopping rule in exactly that form.

Design. We work from a nested array of grades shaped
``(n_items, n_model_samples, n_grader_repeats)`` — items on the outside, model
completions per item, grader re-scorings per completion. The harness produces
exactly this when run with ``model_epochs`` model completions and ``k`` grader
repeats.

The components are estimated as **nested random-effects variance components** by
the method of moments (balanced ANOVA), NOT as observed variances of group means.
This distinction matters: the observed variance of item means already contains a
grader-noise term (sigma2_grader / repeats), so decomposing with observed
variances would leak grader noise into the question and model components and
overstate them. The ANOVA estimators isolate the latent components so that
question + model + grader is a genuine additive partition of the variance of a
single grade. Method-of-moments component estimates can come out negative when a
true component is near zero; following standard practice (Searle, Casella &
McCulloch, *Variance Components*) such estimates are truncated at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Hard cap on how many points the k-selection curve will enumerate, so a
# near-zero between-item variance (which drives the recommended k towards
# infinity) can never produce an unbounded loop / allocation.
MAX_CURVE_K = 1000


@dataclass(frozen=True)
class VarianceComponents:
    """Additive random-effects variance components of a single grade.

    All components are latent variance components (method-of-moments, truncated at
    zero) in squared score units, and estimate the variance of one individual
    grade of one completion: ``question + model + grader``.

    Attributes:
        question: Between-question variance component — the spread attributable to
            which questions are in the benchmark. Dominates the uncertainty of the
            headline score when the item set is a sample of a larger pool.
        model: Model-sampling variance component — spread attributable to which
            completion the model produced for a fixed question. Zero when there is
            one frozen completion per item.
        grader: Grader variance component — the same completion re-scored by the
            grader, the term Miller assumes away.
    """

    question: float
    model: float
    grader: float

    @property
    def total(self) -> float:
        """Sum of the (non-negative) components: estimated variance of one grade."""
        return self.question + self.model + self.grader

    def share(self, component: str) -> float:
        """Fraction of total component variance for 'question', 'model', 'grader'."""
        total = self.total
        if total == 0.0:
            return float("nan")
        return float(getattr(self, component) / total)

    def as_dict(self) -> dict[str, float]:
        return {
            "question": self.question,
            "model": self.model,
            "grader": self.grader,
            "total": self.total,
            "grader_share": self.share("grader"),
            "model_share": self.share("model"),
            "question_share": self.share("question"),
        }


def _as_3d(grades: object) -> Array:
    x = np.asarray(grades, dtype=float)
    if x.ndim == 2:
        # (n_items, n_grader_repeats) -> insert a singleton model axis.
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError(
            "grades must be shaped (n_items, n_model_samples, n_grader_repeats) "
            f"or (n_items, n_grader_repeats); got ndim={x.ndim}"
        )
    n_items, _n_model, n_grader = x.shape
    if n_items < 2:
        raise ValueError("need at least 2 items to estimate between-question variance")
    if n_grader < 2:
        raise ValueError("need at least 2 grader repeats to estimate grader variance")
    return x


def decompose(grades: object) -> VarianceComponents:
    """Decompose grade variance into question / model / grader components.

    Balanced nested random-effects ANOVA (method of moments). With ``n`` items,
    ``m`` model completions per item, and ``k`` grader repeats per completion, and
    mean squares ``MS_item``, ``MS_model``, ``MS_error``:

    * ``grader``   = ``MS_error``                          (= sigma2_epsilon)
    * ``model``    = ``(MS_model - MS_error) / k``         (= sigma2_beta, m > 1)
    * ``question`` = ``(MS_item - MS_model) / (m * k)``    (= sigma2_alpha)

    For the single-completion case (``m == 1``) this reduces to one-way ANOVA:
    ``question = (MS_item - MS_error) / k``. Negative component estimates are
    truncated at zero.

    Args:
        grades: Array shaped ``(n_items, n_model_samples, n_grader_repeats)`` of
            numeric grades, or ``(n_items, n_grader_repeats)`` for the frozen
            single-completion case.

    Returns:
        The :class:`VarianceComponents`.
    """
    x = _as_3d(grades)
    n, m, k = x.shape

    grand = float(x.mean())
    item_means = x.mean(axis=(1, 2))  # (n,)
    comp_means = x.mean(axis=2)  # (n, m)

    ss_item = m * k * float(((item_means - grand) ** 2).sum())
    ss_error = float(((x - comp_means[:, :, None]) ** 2).sum())

    ms_item = ss_item / (n - 1)
    ms_error = ss_error / (n * m * (k - 1))

    grader = ms_error
    if m >= 2:
        ss_model = k * float(((comp_means - item_means[:, None]) ** 2).sum())
        ms_model = ss_model / (n * (m - 1))
        model = (ms_model - ms_error) / k
        question = (ms_item - ms_model) / (m * k)
    else:
        model = 0.0
        question = (ms_item - ms_error) / k

    return VarianceComponents(
        question=max(question, 0.0),
        model=max(model, 0.0),
        grader=max(grader, 0.0),
    )


@dataclass(frozen=True)
class KSelectionPoint:
    """One point on the grader-repeat k-selection curve.

    Attributes:
        k: Number of grader repeats.
        score_se: Standard error of the benchmark score at this k.
        grader_variance_contribution: The grader term of the score *variance* at
            this k (a variance, not an SE — it is a summand of ``score_se ** 2``).
    """

    k: int
    score_se: float
    grader_variance_contribution: float


@dataclass(frozen=True)
class KSelectionCurve:
    """The k-selection curve and the resulting stopping rule.

    Attributes:
        points: Standard error of the benchmark score as a function of the number
            of grader repeats k.
        recommended_k: Smallest k satisfying the stopping rule, or None when the
            between-item (question) variance is estimated at zero and the grader
            term is non-zero — in that case no finite number of grader repeats
            makes the grader negligible relative to a signal that is statistically
            absent (see :func:`stopping_rule_text`).
        tol: The negligibility tolerance used.
        grader_variance: Grader variance component (sigma2_grader).
        item_variance: Between-item (question) variance component (sigma2_between).
        capped: True if the recommended k was clamped to :data:`MAX_CURVE_K`.
    """

    points: list[KSelectionPoint]
    recommended_k: int | None
    tol: float
    grader_variance: float
    item_variance: float
    capped: bool


def k_selection_curve(
    grades: object,
    *,
    k_max: int | None = None,
    tol: float = 0.05,
) -> KSelectionCurve:
    """Standard error of the benchmark score vs. number of grader repeats k.

    The benchmark score is the mean over items of each item's mean grade. Its
    sampling variance has an item term that k cannot reduce and a grader term that
    shrinks like 1/k. With ``n`` items, ``m`` model completions per item, grader
    variance component ``sigma2_grader``, model component ``sigma2_model`` and
    between-item component ``sigma2_between``:

        Var(score)(k) = [sigma2_between + sigma2_model / m + sigma2_grader / (m*k)] / n

    The curve reports ``sqrt(Var(score)(k))`` for k = 1..k_max.

    Stopping rule (Miller's form, for graders). Increasing k stops helping once the
    grader term is negligible relative to the between-item signal k cannot touch:

        sigma2_grader / (m * k)  <=  tol * sigma2_between

    ``recommended_k`` is the smallest k satisfying this. If the grader term is
    already negligible at k = 1 (or is zero), the recommendation is 1 — the null
    result the build plan says to publish as-is: the grader does not need repeating
    for this benchmark. If ``sigma2_between`` is estimated at zero while the grader
    term is positive, no finite k satisfies the rule and ``recommended_k`` is None:
    the benchmark cannot distinguish items beyond grader noise, which is itself the
    finding.

    Args:
        grades: Array as accepted by :func:`decompose`.
        k_max: Largest k to plot; defaults to ``min(MAX_CURVE_K, max(10, 2 * k*))``
            where k* is the recommended k (or 10 when that is None).
        tol: Negligibility tolerance for the grader term (default 0.05).

    Returns:
        A :class:`KSelectionCurve`.
    """
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must be in (0, 1)")
    x = _as_3d(grades)
    n_items, m, _ = x.shape

    comp = decompose(x)
    sigma2_g = comp.grader
    sigma2_model = comp.model
    sigma2_between = comp.question

    capped = False
    if sigma2_g == 0.0:
        recommended_k: int | None = 1
    elif sigma2_between <= 0.0:
        # No measurable between-item signal: the grader term can never be made
        # negligible relative to zero. Report None rather than an unbounded k.
        recommended_k = None
    else:
        raw = _smallest_k(sigma2_g / m, tol, sigma2_between)
        if raw > MAX_CURVE_K:
            recommended_k = MAX_CURVE_K
            capped = True
        else:
            recommended_k = raw

    if k_max is None:
        anchor = recommended_k if recommended_k is not None else 10
        k_max = min(MAX_CURVE_K, max(10, 2 * anchor))
    elif k_max < 1:
        raise ValueError("k_max must be >= 1")

    floor = (sigma2_between + sigma2_model / m) / n_items
    points: list[KSelectionPoint] = []
    for k in range(1, k_max + 1):
        grader_contrib = sigma2_g / (m * k * n_items)
        var_score = floor + grader_contrib
        points.append(
            KSelectionPoint(
                k=k,
                score_se=float(np.sqrt(var_score)),
                grader_variance_contribution=float(grader_contrib),
            )
        )

    return KSelectionCurve(
        points=points,
        recommended_k=recommended_k,
        tol=tol,
        grader_variance=sigma2_g,
        item_variance=sigma2_between,
        capped=capped,
    )


def _smallest_k(reducible: float, tol: float, sigma2_between: float) -> int:
    """Smallest integer k with reducible / k <= tol * sigma2_between.

    Returns 1 when there is nothing to reduce. When the reducible term is
    positive but the threshold is non-positive (e.g. a denormal
    ``sigma2_between`` underflowing ``tol * sigma2_between`` to 0.0), no finite k
    satisfies the rule; returns ``MAX_CURVE_K + 1`` so the caller clamps rather
    than falsely reporting k = 1.
    """
    if reducible <= 0.0:
        return 1
    threshold = tol * sigma2_between
    if threshold <= 0.0:
        return MAX_CURVE_K + 1
    k = int(np.ceil(reducible / threshold))
    return max(k, 1)


def stopping_rule_text(curve: KSelectionCurve) -> str:
    """One-paragraph statement of the stopping rule in Miller's form."""
    if curve.recommended_k is None:
        return (
            "The between-item (question) variance is estimated at zero, so no "
            "finite number of grader repeats makes the grader term negligible "
            "relative to it: the benchmark cannot distinguish items beyond grader "
            f"noise (sigma2_grader = {curve.grader_variance:.5g}). That is itself "
            "the finding. This is the grader analogue of Miller's (arXiv "
            "2411.00640) rule for model resamples, which assumes the grader term "
            "is zero."
        )
    capped_note = (
        f" (clamped to the MAX_CURVE_K cap of {MAX_CURVE_K})" if curve.capped else ""
    )
    return (
        "Grader repeats stop helping once the grader variance term is negligible "
        "relative to the between-item variance the item sample fixes: "
        f"sigma2_grader / (m*k) <= {curve.tol:g} * sigma2_between. Here "
        f"sigma2_grader = {curve.grader_variance:.5g} and "
        f"sigma2_between = {curve.item_variance:.5g}, giving a recommended "
        f"k = {curve.recommended_k}{capped_note}. This is the grader analogue of "
        "Miller's (arXiv 2411.00640) rule for model resamples; Miller assumes the "
        "grader term is zero, so it does not appear in his stopping condition."
    )
