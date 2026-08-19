"""Demonstration study: grader-variance decomposition + k-selection curve.

Runs the full pipeline on ONE existing Inspect eval (``theory_of_mind``, which
ships with Inspect and uses a model-graded scorer) with a model under test and
two judge model families:

1. Freeze completions once (the model under test).
2. Re-grade the frozen completions k times with each judge (grader variance
   isolated; byte-identical completions verified).
3. Decompose the score variance into question / model / grader components.
4. Compute the k-selection curve and state the stopping rule in Miller's form.

Real run (requires provider API keys)::

    python examples/demo_study.py \
        --model anthropic/claude-3-5-sonnet-latest \
        --graders anthropic/claude-3-5-haiku-latest openai/gpt-4o-mini \
        --k 12 --limit 50 --grader-temperature 1.0

Mechanism check (no API keys; NOT a scientific result)::

    python examples/demo_study.py --mock

The ``--mock`` path substitutes deterministic mock models with a stochastic
grader so the end-to-end pipeline — freeze, verify, regrade, decompose, k-curve —
can be exercised without network access. Its numbers are meaningless as evidence
about real judges; they exist only to prove the plumbing computes.

This script does not author a dataset (it uses Inspect's bundled
``theory_of_mind`` example) and prints prior-art citations with its output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate

from grader_variance import (
    compare_judges,
    decompose,
    flip_rate,
    freeze_completions,
    grader_variance_share,
    grades_array,
    k_selection_curve,
    regrade_frozen,
    stopping_rule_text,
    test_retest,
    verify_frozen,
)

PRIOR_ART = (
    "Prior art (grader self-inconsistency is established, not discovered here): "
    "Rating Roulette (arXiv 2510.27106); Reliability without Validity "
    "(arXiv 2606.19544). Stopping-rule form follows Miller, Adding Error Bars to "
    "Evals (arXiv 2411.00640), which adds the question and model-sampling terms "
    "and assumes the grader term is zero. Statistics: rater-agreement "
    "(DOI 10.5281/zenodo.21983269)."
)


@task
def theory_of_mind_graded() -> Task:
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[generate()],
        scorer=model_graded_fact(),
    )


def _mock_model(content: str) -> Model:
    def outputs() -> Iterator[ModelOutput]:
        return itertools.repeat(
            ModelOutput.from_content(model="mockllm/model", content=content)
        )

    return get_model("mockllm/model", custom_outputs=outputs())


def _mock_grader(p_correct: float, seed: int) -> Model:
    rng = random.Random(seed)

    def outputs() -> Iterator[ModelOutput]:
        while True:
            grade = "C" if rng.random() < p_correct else "I"
            yield ModelOutput.from_content(
                model="mockllm/model", content=f"Reasoning.\nGRADE: {grade}"
            )

    return get_model("mockllm/model", custom_outputs=outputs(), memoize=False)


def run_for_grader(
    frozen: EvalLog,
    grader: str | Model,
    *,
    k: int,
    label: str,
    single_completion: bool,
) -> tuple[dict[str, object], np.ndarray]:
    # The intra-grader metrics assume one frozen completion per item, so they are
    # only attached when model_epochs == 1. With multiple model completions the
    # question/model/grader split comes from grades_array + decompose below.
    metrics = (
        [flip_rate(), test_retest(), grader_variance_share()]
        if single_completion
        else None
    )
    scored = regrade_frozen(
        frozen,
        model_graded_fact(),
        grader=grader,
        k=k,
        metrics=metrics,
    )
    check = verify_frozen(scored)
    if not check.ok:
        raise RuntimeError(f"[{label}] frozen-completion check failed: {check}")

    arr = grades_array(scored)
    comp = decompose(arr)
    curve = k_selection_curve(arr, k_max=k)

    reported: dict[str, float] = {}
    if scored.results is not None:
        for sc in scored.results.scores:
            for name, m in sc.metrics.items():
                reported[name] = m.value

    summary: dict[str, object] = {
        "grader": label,
        "frozen_check": {
            "ok": check.ok,
            "n_items": check.n_items,
            "n_repeats": check.n_repeats,
        },
        "metrics": reported,
        "decomposition": comp.as_dict(),
        "recommended_k": curve.recommended_k,
        "grader_variance": curve.grader_variance,
        "item_variance": curve.item_variance,
        "k_curve": [{"k": p.k, "score_se": p.score_se} for p in curve.points],
        "stopping_rule": stopping_rule_text(curve),
    }
    return summary, arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="model under test")
    ap.add_argument("--graders", nargs="+", default=None, help="judge models")
    ap.add_argument("--k", type=int, default=12, help="grader repeats")
    ap.add_argument("--limit", type=int, default=50, help="dataset limit")
    ap.add_argument(
        "--model-epochs",
        type=int,
        default=1,
        help="model-sampling epochs (>1 also measures model variance)",
    )
    ap.add_argument("--grader-temperature", type=float, default=1.0)
    ap.add_argument("--out", default="demo_results.json")
    ap.add_argument(
        "--mock",
        action="store_true",
        help="run with mock models (mechanism check, NOT a scientific result)",
    )
    args = ap.parse_args()

    print(PRIOR_ART, file=sys.stderr)

    if args.mock:
        model: str | Model = _mock_model("A fixed, frozen answer: the bathtub.")
        graders: list[tuple[str, str | Model]] = [
            ("mock-judge-A (p=0.80)", _mock_grader(0.80, seed=1)),
            ("mock-judge-B (p=0.55)", _mock_grader(0.55, seed=2)),
        ]
        limit = args.limit
    else:
        if not args.model or not args.graders:
            ap.error("--model and --graders are required unless --mock is set")
        model = args.model
        graders = [
            (
                g,
                get_model(
                    g,
                    config=GenerateConfig(temperature=args.grader_temperature),
                ),
            )
            for g in args.graders
        ]
        limit = args.limit

    frozen = freeze_completions(
        theory_of_mind_graded(),
        model,
        model_epochs=args.model_epochs,
        limit=limit,
        log_dir="./logs/demo_frozen",
    )
    print(f"Froze {len(frozen.samples or [])} completions.", file=sys.stderr)

    results = []
    arrays: list[tuple[str, np.ndarray]] = []
    for label, grader in graders:
        summary, arr = run_for_grader(
            frozen,
            grader,
            k=args.k,
            label=label,
            single_completion=(args.model_epochs == 1),
        )
        results.append(summary)
        arrays.append((label, arr))

    # Inter-judge comparison (where the bias index is meaningful): per-item mean
    # grade for each judge, thresholded, on the same frozen completions.
    judge_comparison: dict[str, object] | None = None
    if len(arrays) == 2:
        (label_a, arr_a), (label_b, arr_b) = arrays
        judge_comparison = {
            "judge_a": label_a,
            "judge_b": label_b,
            "diagnostics": compare_judges(
                arr_a.mean(axis=(1, 2)), arr_b.mean(axis=(1, 2))
            ),
        }

    out = {
        "note": (
            "MOCK mechanism check — not a scientific result."
            if args.mock
            else "Demonstration study."
        ),
        "prior_art": PRIOR_ART,
        "config": {
            "model": str(getattr(model, "name", model)),
            "k": args.k,
            "limit": limit,
            "model_epochs": args.model_epochs,
            "grader_temperature": args.grader_temperature if not args.mock else None,
        },
        "results": results,
        "judge_comparison": judge_comparison,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
