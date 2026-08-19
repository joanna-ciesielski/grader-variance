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
    RunManifest,
    compare_judges,
    dataset_revision_for_bundled_example,
    decompose,
    dependability,
    flip_rate,
    freeze_completions,
    frozen_digests,
    grader_variance_share,
    grades_array,
    icc_2_1,
    k_selection_curve,
    krippendorff_alpha_repeats,
    regrade_frozen,
    repeats_needed,
    resolve_model_id,
    stopping_rule_text,
    test_retest,
    verify_frozen,
    verify_manifest_digests,
)
from grader_variance.manifest import utc_now_iso, write_run_manifest

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
    manifest_path: Path | None = None,
    dependability_target: float = 0.9,
) -> tuple[dict[str, object], np.ndarray]:
    # The intra-grader metrics assume one frozen completion per item, so they are
    # only attached when model_epochs == 1. With multiple model completions the
    # question/model/grader split comes from grades_array + decompose below.
    metrics = (
        [
            flip_rate(),
            test_retest(),
            grader_variance_share(),
            icc_2_1(),
            krippendorff_alpha_repeats(),
        ]
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
    if manifest_path is not None:
        # Cross-process byte-identity: the regraded log must carry exactly the
        # completion digests recorded in the run manifest at freeze time.
        verify_manifest_digests(manifest_path, scored)

    arr = grades_array(scored)
    comp = decompose(arr)
    curve = k_selection_curve(arr, k_max=k)

    # Independent estimator cross-check (REML), when the design supports it
    # and the optional analysis extra is installed.
    comp_reml_dict: dict[str, float] | None = None
    if arr.shape[1] >= 2:
        try:
            from grader_variance import estimate_components_reml

            comp_reml_dict = estimate_components_reml(arr).as_dict()
        except ImportError:
            comp_reml_dict = None

    # G-theory dependability view of the same planning question.
    phi_needed = repeats_needed(comp, dependability_target)
    dep_curve = [{"k": kk, "phi": dependability(comp, kk)} for kk in range(1, k + 1)]

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
        "decomposition_reml": comp_reml_dict,
        "recommended_k": curve.recommended_k,
        "grader_variance": curve.grader_variance,
        "item_variance": curve.item_variance,
        "k_curve": [{"k": p.k, "score_se": p.score_se} for p in curve.points],
        "stopping_rule": stopping_rule_text(curve),
        "dependability_target": dependability_target,
        "repeats_needed_for_target": phi_needed,
        "dependability_curve": dep_curve,
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
        "--dataset-revision",
        default=None,
        help=(
            "dataset revision pin recorded in the run manifest; defaults to "
            "the SHA-256 of the bundled theory_of_mind.jsonl plus the "
            "installed inspect_ai version"
        ),
    )
    ap.add_argument(
        "--dependability-target",
        type=float,
        default=0.9,
        help="target dependability for the G-theory repeats_needed report",
    )
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
        model_alias = "mockllm/model"
        model_resolved = "mockllm/model"
        grader_aliases = [label for label, _ in graders]
        graders_resolved = ["mockllm/model"] * len(graders)
    else:
        if not args.model or not args.graders:
            ap.error("--model and --graders are required unless --mock is set")
        # Resolve provider "-latest"/rolling aliases to the concrete snapshot
        # each serves right now, and RUN with the resolved IDs so the study is
        # pinned, not just documented (see manifest.resolve_model_id).
        model_alias = args.model
        model_resolved = resolve_model_id(args.model)
        grader_aliases = list(args.graders)
        graders_resolved = [resolve_model_id(g) for g in args.graders]
        for alias, resolved in zip(
            [model_alias, *grader_aliases],
            [model_resolved, *graders_resolved],
            strict=True,
        ):
            print(f"Resolved {alias} -> {resolved}", file=sys.stderr)
        model = model_resolved
        graders = [
            (
                resolved,
                get_model(
                    resolved,
                    config=GenerateConfig(temperature=args.grader_temperature),
                ),
            )
            for resolved in graders_resolved
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

    # Write the run manifest at freeze time: canary, dataset revision pin,
    # resolved model snapshots, and per-completion SHA-256 digests. Every
    # subsequent regrade is verified against this file.
    dataset_revision = args.dataset_revision or dataset_revision_for_bundled_example(
        "theory_of_mind"
    )
    import inspect_ai as _inspect_ai

    import grader_variance as _gv

    manifest_path = Path(args.out).with_suffix(".manifest.json")
    write_run_manifest(
        manifest_path,
        RunManifest(
            canary=_gv.CANARY,
            created_utc=utc_now_iso(),
            inspect_ai_version=getattr(_inspect_ai, "__version__", "unknown"),
            grader_variance_version=_gv.__version__,
            dataset_name="theory_of_mind",
            dataset_revision=dataset_revision,
            model_alias=model_alias,
            model_resolved=model_resolved,
            grader_aliases=grader_aliases,
            graders_resolved=graders_resolved,
            k=args.k,
            limit=limit,
            model_epochs=args.model_epochs,
            grader_temperature=(args.grader_temperature if not args.mock else None),
            completion_digests=frozen_digests(frozen),
            out_file=str(args.out),
        ),
    )
    print(f"Wrote run manifest: {manifest_path}", file=sys.stderr)

    results = []
    arrays: list[tuple[str, np.ndarray]] = []
    for label, grader in graders:
        summary, arr = run_for_grader(
            frozen,
            grader,
            k=args.k,
            label=label,
            single_completion=(args.model_epochs == 1),
            manifest_path=manifest_path,
            dependability_target=args.dependability_target,
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
            "model_alias": model_alias,
            "model_resolved": model_resolved,
            "graders_resolved": graders_resolved,
            "dataset_revision": dataset_revision,
            "manifest_file": str(manifest_path),
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
