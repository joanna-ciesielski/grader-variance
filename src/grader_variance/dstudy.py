"""D-study utilities: dependability under k grader repeats, and REML cross-check.

Complements :mod:`grader_variance.decomposition`, which states the stopping rule
in Miller's form (grader term negligible relative to between-item variance).
This module states the same planning question in generalizability-theory form
(Shavelson & Webb 1991): the **dependability** of a completion's mean grade
under k grader repeats, and the smallest k reaching a target dependability.
The two rules answer different questions — Miller's form bounds the grader's
contribution to the *benchmark score's* SE; dependability bounds the grader's
noise share in a *single completion's* mean grade — and the demo study reports
both.

Honesty note (state this in the paper): under the nested design
``(item, model-sample, grader-repeat)`` with a single grader whose repeats are
exchangeable, the grader component IS the within-cell residual. There is no
separable fourth "residual" component; ``sigma2_grader`` and the residual are
the same number by construction.

REML cross-check. :func:`grader_variance.decompose` estimates components by
balanced method-of-moments ANOVA. :func:`estimate_components_reml` fits the
same nested random-effects model by REML (statsmodels), as an independent check
that the reported components are not an artifact of the estimator. statsmodels
is an optional dependency (``pip install grader-variance[analysis]``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .decomposition import VarianceComponents

Array = NDArray[np.float64]

__all__ = [
    "dependability",
    "estimate_components_reml",
    "repeats_needed",
]

MAX_REPEATS = 1000


def dependability(components: VarianceComponents, k: int) -> float:
    """Dependability of a completion's mean grade under ``k`` grader repeats.

    ``Phi(k) = (q + m) / (q + m + g / k)`` where ``q``, ``m``, ``g`` are the
    question, model-sampling, and grader variance components: the proportion of
    variance in a k-repeat mean grade attributable to real differences between
    completions rather than grader re-scoring noise.

    Args:
        components: Estimated components (from
            :func:`grader_variance.decompose` or
            :func:`estimate_components_reml`).
        k: Number of grader repeats (>= 1).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    true_var = components.question + components.model
    denom = true_var + components.grader / k
    if denom == 0.0:
        raise ValueError("dependability undefined: all components are zero")
    return float(true_var / denom)


def repeats_needed(
    components: VarianceComponents,
    target: float,
    *,
    max_repeats: int = MAX_REPEATS,
) -> int | None:
    """Smallest k with ``dependability(k) >= target``; None if unreachable.

    Closed form: ``Phi(k) >= t  <=>  k >= t * g / ((1 - t) * (q + m))``.
    Returns None when ``q + m == 0`` (no signal for any number of grader
    repeats to recover — itself the finding) or when the required k exceeds
    ``max_repeats``.

    Args:
        components: Estimated variance components.
        target: Target dependability in (0, 1).
        max_repeats: Cap on the search (default 1000).
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    true_var = components.question + components.model
    if true_var == 0.0:
        return None
    if components.grader == 0.0:
        return 1
    k = int(np.ceil(target * components.grader / ((1.0 - target) * true_var)))
    k = max(k, 1)
    if k > max_repeats:
        return None
    return k


def estimate_components_reml(grades: object) -> VarianceComponents:
    """REML estimate of the question / model / grader components.

    Fits the nested random-effects model
    ``y = mu + a_item + b_completion(item) + e`` by REML via statsmodels
    ``MixedLM`` (item random intercept; completion-within-item variance
    component; the residual is the grader component — see the module honesty
    note). Requires the balanced 3-D array of
    :func:`grader_variance.decompose` with ``n_model_samples >= 2``; negative
    estimates cannot occur under REML, but tiny components may be returned at
    the optimizer boundary.

    Raises:
        ImportError: statsmodels is not installed (``pip install
            grader-variance[analysis]``).
        ValueError: input is not a balanced 3-D design with m >= 2.
    """
    try:
        import pandas as pd
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError as err:  # pragma: no cover - exercised without extra
        raise ImportError(
            "estimate_components_reml requires statsmodels and pandas: "
            "pip install 'grader-variance[analysis]'"
        ) from err

    x = np.asarray(grades, dtype=float)
    if x.ndim != 3:
        raise ValueError(
            "grades must be shaped (n_items, n_model_samples, n_grader_repeats)"
        )
    n_items, m, k = x.shape
    if n_items < 2 or k < 2:
        raise ValueError("need >= 2 items and >= 2 grader repeats")
    if m < 2:
        raise ValueError(
            "REML needs n_model_samples >= 2 to separate the model component; "
            "use decompose() for the single-completion case"
        )
    if not np.isfinite(x).all():
        raise ValueError("grades contain non-finite values")

    item_idx, comp_idx, _ = np.meshgrid(
        np.arange(n_items), np.arange(m), np.arange(k), indexing="ij"
    )
    frame = pd.DataFrame(
        {
            "y": x.ravel(),
            "item": item_idx.ravel().astype(str),
            "completion": (item_idx * m + comp_idx).ravel().astype(str),
        }
    )
    model = MixedLM.from_formula(
        "y ~ 1",
        groups="item",
        re_formula="1",
        vc_formula={"completion": "0 + C(completion)"},
        data=frame,
    )
    fit = model.fit(reml=True, method="lbfgs")
    question = float(np.asarray(fit.cov_re)[0, 0])
    model_var = float(np.asarray(fit.vcomp).ravel()[0])
    grader = float(fit.scale)
    return VarianceComponents(
        question=max(question, 0.0),
        model=max(model_var, 0.0),
        grader=max(grader, 0.0),
    )
