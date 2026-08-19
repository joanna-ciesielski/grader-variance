"""End-to-end harness tests using Inspect's mockllm provider.

These prove the fixed-completion mechanism itself: completions are generated once
and are byte-identical across grader repeats, while grader draws vary. No real
model API is needed — the model under test emits a fixed completion and the grader
cycles grades deterministically.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import pytest
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import Model, ModelOutput, get_model
from inspect_ai.scorer import Metric, model_graded_fact

from grader_variance import (
    completion_digest,
    flip_rate,
    freeze_completions,
    grader_variance_share,
    regrade_frozen,
    verify_frozen,
)
from grader_variance import test_retest as retest_metric


def gv_metrics() -> list[Metric]:
    return [flip_rate(), retest_metric(), grader_variance_share()]


@task
def tiny_task() -> Task:
    return Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="Capital of France?", target="Paris"),
            Sample(input="Color of the sky?", target="blue"),
        ],
        scorer=model_graded_fact(),
    )


def _fixed_model() -> Model:
    def outputs() -> Iterator[ModelOutput]:
        while True:
            yield ModelOutput.from_content(
                model="mockllm/model", content="A fixed, frozen completion."
            )

    return get_model("mockllm/model", custom_outputs=outputs())


def _wobbly_grader() -> Model:
    def outputs() -> Iterator[ModelOutput]:
        # Deterministic but varying grades so repeats disagree.
        for g in itertools.cycle(["C", "C", "I", "C", "I", "I"]):
            yield ModelOutput.from_content(
                model="mockllm/model", content=f"Reasoning.\nGRADE: {g}"
            )

    return get_model("mockllm/model", custom_outputs=outputs(), memoize=False)


def test_freeze_then_regrade_completions_are_byte_identical(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "frozen")
    )
    assert frozen.status == "success"

    scored = regrade_frozen(
        frozen,
        model_graded_fact(),
        grader=_wobbly_grader(),
        k=5,
        metrics=gv_metrics(),
    )
    check = verify_frozen(scored)
    assert check.ok
    assert check.n_items == 3
    assert check.n_repeats == 5
    assert check.offending_items == []


def test_regrade_produces_k_epochs_per_item(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "frozen")
    )
    scored = regrade_frozen(
        frozen,
        model_graded_fact(),
        grader=_wobbly_grader(),
        k=4,
        metrics=[flip_rate()],
    )
    assert scored.samples is not None
    assert len(scored.samples) == 3 * 4  # 3 items x 4 repeats
    epochs_per_item: dict[str, set[int]] = {}
    for s in scored.samples:
        epochs_per_item.setdefault(str(s.id), set()).add(s.epoch)
    assert all(epochs == {1, 2, 3, 4} for epochs in epochs_per_item.values())


def test_grader_variance_metric_is_reported(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "frozen")
    )
    scored = regrade_frozen(
        frozen,
        model_graded_fact(),
        grader=_wobbly_grader(),
        k=6,
        metrics=gv_metrics(),
    )
    assert scored.results is not None
    reported: dict[str, float] = {}
    for sc in scored.results.scores:
        for name, m in sc.metrics.items():
            reported[name] = m.value
    assert "flip_rate" in reported
    assert "test_retest" in reported
    assert "grader_variance_share" in reported
    # With a wobbling grader on identical completions, some flipping is expected.
    assert 0.0 <= reported["flip_rate"] <= 1.0


def test_completion_digest_detects_a_changed_completion(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "frozen")
    )
    assert frozen.samples is not None
    a = frozen.samples[0]
    d0 = completion_digest(a)
    # same bytes -> same digest
    assert completion_digest(a) == d0
    # mutate the completion -> digest must change (guards the freeze check)
    a.output.completion = a.output.completion + " tampered"
    assert completion_digest(a) != d0


def test_regrade_requires_k_at_least_two(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "frozen")
    )
    with pytest.raises(ValueError, match="k must be >= 2"):
        regrade_frozen(
            frozen,
            model_graded_fact(),
            grader=_wobbly_grader(),
            k=1,
            metrics=[flip_rate()],
        )
