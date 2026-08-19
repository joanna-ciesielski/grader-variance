"""Shared test helpers: build Inspect SampleScores for metric tests."""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.scorer import SampleScore, Score

Grade = str | int | float | bool


def sample_scores(rows: Sequence[Sequence[Grade]]) -> list[SampleScore]:
    """Flatten a (n_items x k_repeats) grid into unreduced SampleScores.

    Each inner list is one item's k grader repeats; every grade becomes a
    SampleScore tagged with that item's id, exactly as Inspect delivers unreduced
    epoch scores to a ``scores="unreduced"`` metric.
    """
    out: list[SampleScore] = []
    for i, row in enumerate(rows):
        for grade in row:
            out.append(SampleScore(score=Score(value=grade), sample_id=i))
    return out
