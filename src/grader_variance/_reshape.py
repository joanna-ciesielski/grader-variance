"""Turn Inspect unreduced epoch scores into a (n_items, k_repeats) matrix.

A metric declared ``@metric(scores="unreduced")`` receives one
:class:`~inspect_ai.scorer.SampleScore` per sample per epoch. In the
fixed-completion harness each *epoch* of a given sample is an independent grader
re-scoring of the **same frozen completion**, so grouping the unreduced scores by
``sample_id`` yields exactly the intra-rater layout the ``rater-agreement``
library expects: one row per item, one column per grader repeat.

All intra-rater statistics used here (flip rate, test-retest agreement, ICC(1,1),
within-item variance) are invariant to the ordering of repeats within an item, so
grouping by ``sample_id`` alone is sufficient; we do not need to recover the
original epoch order.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from inspect_ai.scorer import SampleScore, Value, value_to_float

Grade = str | int | float | bool

# Reserved sample-metadata key recording which model-sampling completion a
# regrade replicate belongs to (1-based). Set by the harness; read here only to
# reject multi-completion logs from the intra-grader metrics (which assume one
# frozen completion per item — use grader_variance.decompose for the multi-model
# case).
MODEL_SAMPLE_KEY = "_gv_model_sample"


class RaggedRepeatsError(ValueError):
    """Raised when items do not all have the same number of grader repeats.

    The intra-rater matrix must be rectangular. In the fixed-completion harness k
    is controlled exactly, so a ragged result signals dropped/errored samples
    rather than an expected condition, and is surfaced rather than silently
    padded.
    """


def group_by_item(scores: list[SampleScore]) -> OrderedDict[str, list[Grade]]:
    """Group unreduced scores by ``sample_id``, preserving first-seen item order.

    Returns a mapping ``item_id -> [grade, grade, ...]`` where each grade is the
    raw scalar :attr:`Score.value` of one grader repeat. ``sample_id`` is
    stringified so that mixed int/str ids group stably.
    """
    grouped: OrderedDict[str, list[Grade]] = OrderedDict()
    model_samples: dict[str, set[object]] = {}
    for s in scores:
        key = str(s.sample_id)
        value = s.score.value
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                "grader-variance metrics require scalar score values, but got "
                f"{type(value).__name__} for sample {key!r}. Use a scorer that "
                "emits one scalar grade per epoch (e.g. model_graded_fact)."
            )
        meta = s.sample_metadata or {}
        if MODEL_SAMPLE_KEY in meta:
            model_samples.setdefault(key, set()).add(meta[MODEL_SAMPLE_KEY])
        grouped.setdefault(key, []).append(value)
    multi = sorted(k for k, ms in model_samples.items() if len(ms) > 1)
    if multi:
        raise ValueError(
            "intra-grader metrics assume one frozen completion per item, but "
            f"items {multi} have multiple model-sampling completions "
            f"(model_epochs > 1). Use grader_variance.grades_array + decompose "
            "for the question/model/grader decomposition instead."
        )
    return grouped


def label_matrix(scores: list[SampleScore]) -> list[list[Grade]]:
    """Rectangular list-of-rows of raw grade labels, one row per item.

    Suitable for label-based statistics (decision flip rate, test-retest
    agreement) that operate on categorical grades without numeric conversion.
    """
    grouped = group_by_item(scores)
    rows = list(grouped.values())
    _check_rectangular(rows, grouped)
    return rows


def numeric_matrix(
    scores: list[SampleScore],
    to_float: Callable[[Value], float] | None = None,
) -> list[list[float]]:
    """Rectangular matrix of grades mapped to floats, one row per item.

    ``to_float`` defaults to Inspect's :func:`value_to_float`, which maps the
    standard model-graded letters (C/I/P) to 1.0 / 0.0 / 0.5 and passes numeric
    scores through. Required by variance-based statistics (ICC(1,1),
    within-item variance, grader variance share).
    """
    convert = to_float or value_to_float()
    grouped = group_by_item(scores)
    rows = [[convert(v) for v in row] for row in grouped.values()]
    _check_rectangular(rows, grouped)
    return rows


def _check_rectangular(
    rows: list[list[Grade]] | list[list[float]],
    grouped: OrderedDict[str, list[Grade]],
) -> None:
    if not rows:
        raise ValueError("no scores to reshape")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        counts = {k: len(v) for k, v in grouped.items()}
        raise RaggedRepeatsError(
            "items have differing numbers of grader repeats "
            f"(counts={counts}); the intra-rater matrix must be rectangular. "
            "This usually means some samples errored during grading."
        )
