"""Fixed-completion harness: freeze model completions, re-grade them N times.

Inspect's ``epochs`` re-runs the **model**, so repeated epochs mix model-sampling
variance with grader variance. This harness does the opposite: generate each
completion once, freeze it, then re-run only the **scorer** N times against the
identical bytes. Grader variance is thereby isolated from model variance.

The mechanism relies on Inspect's re-scoring path
(:func:`inspect_ai.scorer.score`): it reconstructs the task state from the stored
``sample.messages`` and ``sample.output`` and re-runs the scorer without ever
invoking the model. To obtain N independent grader draws per item we replicate
each frozen sample into N epochs — byte-identical completions, independent grader
calls — and re-score the expanded log with the grader bound to the ``grader``
model role.

Trust nothing downstream until :func:`verify_frozen` confirms the completions are
byte-identical across the N repeats. This is a real check, not a formality: if a
future Inspect change caused re-scoring to regenerate completions, every variance
number below would silently become a model+grader mixture again.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass

from inspect_ai import Epochs, Task, score
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import Model
from inspect_ai.scorer import Metric, Scorer

from ._reshape import MODEL_SAMPLE_KEY

DEFAULT_REDUCERS = ["mean"]

__all__ = [
    "freeze_completions",
    "regrade_frozen",
    "verify_frozen",
    "completion_digest",
    "FrozenCheck",
    "MODEL_SAMPLE_KEY",
]


def freeze_completions(
    task: Task,
    model: str | Model,
    *,
    model_epochs: int = 1,
    log_dir: str | None = None,
    **eval_kwargs: object,
) -> EvalLog:
    """Run ``task`` once to generate and freeze completions.

    The returned log holds the completions that every subsequent regrade re-scores
    without re-invoking the model. Any scorer attached to ``task`` still runs here,
    but its scores are irrelevant to the frozen regrade and are overwritten by
    :func:`regrade_frozen`.

    Args:
        task: The Inspect task to run (an existing eval — this project does not
            author datasets).
        model: The model under test (produces the completions).
        model_epochs: Number of model-sampling epochs. Keep at 1 to freeze a
            single completion per item; set > 1 to also capture model-sampling
            variance for the full question/model/grader decomposition (each
            model epoch becomes an independent frozen completion).
        log_dir: Optional log directory.
        **eval_kwargs: Forwarded to :func:`inspect_ai.eval` (e.g. ``limit``).

    Returns:
        The completed :class:`EvalLog` with frozen completions.
    """
    logs = inspect_eval(
        task,
        model=model,
        epochs=Epochs(model_epochs) if model_epochs > 1 else None,
        log_dir=log_dir or "./logs/frozen",
        **eval_kwargs,  # type: ignore[arg-type]
    )
    if not logs:
        raise RuntimeError("freeze_completions: eval returned no logs")
    log = logs[0]
    if log.status != "success":
        raise RuntimeError(
            f"freeze_completions: eval did not succeed (status={log.status!r})"
        )
    return log


def completion_digest(sample: EvalSample) -> str:
    """Stable SHA-256 over the exact bytes a grader would see for this sample.

    Covers the assistant completion text and the full message transcript, which
    together are what the scorer reconstructs its task state from. Two samples
    with the same digest present a byte-identical completion to the grader.
    """
    h = hashlib.sha256()
    h.update(b"completion\x00")
    h.update(sample.output.completion.encode("utf-8"))
    for msg in sample.messages:
        h.update(b"\x00msg\x00")
        h.update(str(msg.role).encode("utf-8"))
        h.update(b"\x00")
        h.update(msg.text.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class FrozenCheck:
    """Result of verifying that regrade repeats saw identical completions.

    Attributes:
        ok: True iff every (item, model-sample) presented a single unique
            completion digest across all its grader repeats.
        n_items: Number of distinct (item, model-sample) units checked.
        n_repeats: Grader repeats per unit (None if ragged — itself a failure).
        offending_items: Unit keys ("<item id>\\x1e<model-sample>") whose
            completion bytes changed across repeats.
    """

    ok: bool
    n_items: int
    n_repeats: int | None
    offending_items: list[str]


def verify_frozen(log: EvalLog) -> FrozenCheck:
    """Assert every item's completion is byte-identical across its repeats.

    Groups the log's samples by ``id`` and hashes each replicate's completion. An
    item passes only if all its replicates share one digest. Call this on the
    output of :func:`regrade_frozen` before trusting any variance metric.
    """
    if log.samples is None:
        raise ValueError("log has no samples to verify")
    digests: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for s in log.samples:
        # Group by (item, model-sample): distinct model completions of the same
        # item legitimately differ, so each must be checked for freezing on its
        # own, not merged with the item's other completions.
        model_sample = (s.metadata or {}).get(MODEL_SAMPLE_KEY, 1)
        key = f"{s.id}\x1e{model_sample}"
        digests.setdefault(key, set()).add(completion_digest(s))
        counts[key] = counts.get(key, 0) + 1
    offending = sorted(k for k, ds in digests.items() if len(ds) != 1)
    repeat_counts = set(counts.values())
    n_repeats = repeat_counts.pop() if len(repeat_counts) == 1 else None
    return FrozenCheck(
        ok=(not offending and n_repeats is not None),
        n_items=len(digests),
        n_repeats=n_repeats,
        offending_items=offending,
    )


def _expand_to_epochs(frozen: EvalLog, k: int) -> EvalLog:
    """Replicate each frozen sample into k epochs with identical completions.

    The original sample id is preserved unchanged; the model-sampling completion
    index (the frozen log's epoch) is recorded in sample metadata under
    :data:`MODEL_SAMPLE_KEY` so that grader repeats nest within
    (item, model-sample) without mangling the id.
    """
    if frozen.samples is None:
        raise ValueError("frozen log has no samples")
    expanded = deepcopy(frozen)
    new_samples: list[EvalSample] = []
    for base in frozen.samples:
        for repeat in range(1, k + 1):
            replicate = deepcopy(base)
            replicate.epoch = repeat
            replicate.scores = None
            replicate.metadata = {
                **(replicate.metadata or {}),
                MODEL_SAMPLE_KEY: base.epoch,
            }
            new_samples.append(replicate)
    expanded.samples = new_samples
    return expanded


def regrade_frozen(
    frozen: EvalLog,
    scorer: Scorer | list[Scorer],
    grader: str | Model,
    *,
    k: int,
    metrics: list[Metric] | None = None,
    reducers: list[str] | None = None,
    verify: bool = True,
) -> EvalLog:
    """Re-grade frozen completions k times, isolating grader variance.

    Replicates each frozen completion into k epochs and re-scores the expanded log
    with ``grader`` bound to the ``grader`` model role. The model is never
    re-invoked, so all k grades for an item are draws of the same grader on the
    identical completion. The grader-variance metrics are passed via ``metrics``
    and evaluate the k epochs through their ``scores="unreduced"`` contract.

    Args:
        frozen: Output of :func:`freeze_completions`.
        scorer: The model-graded scorer(s) to repeat, e.g. ``model_graded_fact()``.
        grader: The judge model to bind to the ``grader`` role.
        k: Number of grader repeats per item.
        metrics: Metrics to compute over the k repeats, e.g.
            ``[flip_rate(), test_retest(), grader_variance_share()]``. Overrides
            the scorer's own metrics for this pass (Inspect's ``score(metrics=...)``
            path).
        reducers: Epoch reducers; defaults to ``["mean", "collect"]``.
        verify: If True (default), raise unless every item's completion is
            byte-identical across the k repeats.

    Returns:
        The re-scored :class:`EvalLog` (k epochs per item).
    """
    if k < 2:
        raise ValueError("k must be >= 2 to measure grader variance")
    expanded = _expand_to_epochs(frozen, k)
    scored = score(
        expanded,
        scorers=scorer,
        metrics=list(metrics) if metrics is not None else None,
        epochs_reducer=reducers or DEFAULT_REDUCERS,
        model_roles={"grader": grader},
        action="overwrite",
        display="none",
        copy=False,
    )
    if verify:
        check = verify_frozen(scored)
        if not check.ok:
            raise RuntimeError(
                "regrade_frozen: completions were NOT byte-identical across "
                f"repeats (offending items: {check.offending_items or 'ragged'}). "
                "Grader variance cannot be trusted; downstream numbers would mix "
                "model and grader variance."
            )
    return scored
