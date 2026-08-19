"""Cross-process byte-identity: manifest digests vs (re)graded logs.

Uses a custom cycling scorer rather than ``model_graded_fact`` so these tests
run fully offline (no tokenizer downloads); byte-identity verification is about
the frozen completions, not about which scorer re-graded them.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import pytest
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import Model, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from grader_variance import (
    freeze_completions,
    frozen_digests,
    regrade_frozen,
    verify_manifest_digests,
)
from grader_variance.manifest import CANARY, RunManifest, write_run_manifest


@task
def tiny_task() -> Task:
    return Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="Capital of France?", target="Paris"),
        ],
        scorer=cycling_scorer(),
    )


@scorer(metrics=[accuracy()])
def cycling_scorer() -> Scorer:
    """Deterministic-but-varying stand-in for a stochastic grader (offline)."""
    grades = itertools.cycle(["C", "I", "C"])

    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=next(grades))

    return score


def _fixed_model(content: str = "A fixed, frozen completion.") -> Model:
    def outputs() -> Iterator[ModelOutput]:
        while True:
            yield ModelOutput.from_content(model="mockllm/model", content=content)

    return get_model("mockllm/model", custom_outputs=outputs())


def _write(tmp_path: Path, digests: dict[str, str]) -> Path:
    return write_run_manifest(
        tmp_path / "run.manifest.json",
        RunManifest(
            canary=CANARY,
            created_utc="2026-08-19T00:00:00Z",
            inspect_ai_version="test",
            grader_variance_version="0.1.0",
            dataset_name="tiny",
            dataset_revision="rev-test",
            model_alias="mockllm/model",
            model_resolved="mockllm/model",
            completion_digests=digests,
        ),
    )


def test_regraded_log_verifies_against_manifest(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "logs")
    )
    manifest = _write(tmp_path, frozen_digests(frozen))
    scored = regrade_frozen(
        frozen, cycling_scorer(), grader=_fixed_model("grader"), k=3
    )
    verify_manifest_digests(manifest, scored)  # must not raise


def test_mutated_manifest_digest_fails(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "logs")
    )
    digests = frozen_digests(frozen)
    first = next(iter(digests))
    digests[first] = "0" * 64
    manifest = _write(tmp_path, digests)
    scored = regrade_frozen(
        frozen, cycling_scorer(), grader=_fixed_model("grader"), k=2
    )
    with pytest.raises(ValueError, match="differ from the frozen manifest"):
        verify_manifest_digests(manifest, scored)


def test_tampered_canary_fails(tmp_path: Path) -> None:
    frozen = freeze_completions(
        tiny_task(), _fixed_model(), log_dir=str(tmp_path / "logs")
    )
    manifest = _write(tmp_path, frozen_digests(frozen))
    text = manifest.read_text().replace("DO NOT TRAIN", "do not train")
    manifest.write_text(text)
    with pytest.raises(ValueError, match="canary"):
        verify_manifest_digests(manifest, frozen)
