"""Reassemble grades from a regraded log, and compare two judges.

The fixed-completion harness stores k grader repeats as k epochs of each item and
records the model-sampling completion index in sample metadata under
:data:`grader_variance._reshape.MODEL_SAMPLE_KEY`. :func:`grades_array` uses those
to rebuild the nested ``(n_items, n_model_samples, n_grader_repeats)`` array that
:func:`grader_variance.decompose` and :func:`grader_variance.k_selection_curve`
consume.

:func:`compare_judges` covers the genuinely two-rater question — does judge A pass
systematically more often than judge B on the same completions — using the
kappa-paradox diagnostics (prevalence index, bias index, PABAK) from the
``rater-agreement`` library. Bias between two distinct judges is where the bias
index is meaningful; it is deliberately *not* offered as an intra-grader metric,
because repeated draws of a single judge on a frozen completion are exchangeable
and have no "first vs second" direction for a bias to point in.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from inspect_ai.log import EvalLog
from inspect_ai.scorer import Value, value_to_float
from numpy.typing import NDArray
from raters import paradox

from ._reshape import MODEL_SAMPLE_KEY


def grades_array(
    log: EvalLog,
    scorer: str | None = None,
    to_float: Callable[[Value], float] | None = None,
) -> NDArray[np.float64]:
    """Assemble a ``(n_items, n_model_samples, n_grader_repeats)`` grade array.

    Args:
        log: A regraded log from :func:`grader_variance.harness.regrade_frozen`.
        scorer: Name of the score key to read (defaults to the sole score key if
            there is exactly one).
        to_float: Grade-to-float mapping; defaults to :func:`value_to_float`.

    Returns:
        A rectangular float array. Raises if the log is ragged (items with
        differing numbers of model samples or grader repeats), since the
        decomposition requires a balanced design.
    """
    if log.samples is None:
        raise ValueError("log has no samples")
    convert = to_float or value_to_float()

    # base_item -> model_sample_index -> list[grade]
    nested: OrderedDict[str, OrderedDict[Any, list[float]]] = OrderedDict()
    for s in log.samples:
        base = str(s.id)
        model_sample = (s.metadata or {}).get(MODEL_SAMPLE_KEY, 1)
        scores = s.scores or {}
        if scorer is not None:
            key = scorer
        elif len(scores) == 1:
            key = next(iter(scores))
        else:
            raise ValueError(
                "multiple score keys present; pass scorer= to select one "
                f"(available: {sorted(scores)})"
            )
        if key not in scores:
            raise ValueError(f"score {key!r} not found on sample {s.id!r}")
        nested.setdefault(base, OrderedDict()).setdefault(model_sample, []).append(
            convert(scores[key].value)
        )

    n_items = len(nested)
    if n_items == 0:
        raise ValueError("no samples to assemble")
    model_counts = {len(m) for m in nested.values()}
    if len(model_counts) != 1:
        raise ValueError(f"ragged model-sample counts across items: {model_counts}")
    n_model = model_counts.pop()
    repeat_counts = {len(reps) for m in nested.values() for reps in m.values()}
    if len(repeat_counts) != 1:
        raise ValueError(f"ragged grader-repeat counts: {repeat_counts}")
    n_grader = repeat_counts.pop()

    arr = np.empty((n_items, n_model, n_grader), dtype=float)
    for i, model_map in enumerate(nested.values()):
        for j, reps in enumerate(model_map.values()):
            arr[i, j, :] = reps
    return arr


def compare_judges(
    grades_a: object,
    grades_b: object,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Kappa-paradox diagnostics for two judges on the same items.

    Each judge's grades for an item are reduced to a single pass/fail decision by
    thresholding the item's mean grade (``mean >= threshold``). The two judges'
    decisions form an ordered 2x2 table
    ``[[A-pass & B-pass, A-pass & B-fail], [A-fail & B-pass, A-fail & B-fail]]``,
    on which :func:`raters.paradox.diagnose` reports percent agreement, Cohen's
    kappa, the prevalence and bias indices, and PABAK. The bias index here is
    meaningful: it measures whether one judge passes systematically more often
    than the other.

    Args:
        grades_a: Judge A grades, shaped ``(n_items,)`` or ``(n_items, ...)`` (any
            trailing axes are averaged to a per-item mean before thresholding).
        grades_b: Judge B grades, same items and shape convention.
        threshold: Pass mark applied to each judge's per-item mean grade.

    Returns:
        The dict returned by :func:`raters.paradox.diagnose`.
    """
    a = np.asarray(grades_a, dtype=float)
    b = np.asarray(grades_b, dtype=float)
    a_mean = a if a.ndim == 1 else a.reshape(a.shape[0], -1).mean(axis=1)
    b_mean = b if b.ndim == 1 else b.reshape(b.shape[0], -1).mean(axis=1)
    if a_mean.shape != b_mean.shape:
        raise ValueError(
            f"judges must cover the same items (got {a_mean.shape} vs {b_mean.shape})"
        )
    if a_mean.size == 0:
        raise ValueError("no items to compare")

    a_pass = a_mean >= threshold
    b_pass = b_mean >= threshold
    table = np.array(
        [
            [int((a_pass & b_pass).sum()), int((a_pass & ~b_pass).sum())],
            [int((~a_pass & b_pass).sum()), int((~a_pass & ~b_pass).sum())],
        ],
        dtype=float,
    )
    result: dict[str, Any] = paradox.diagnose(table)
    return result
