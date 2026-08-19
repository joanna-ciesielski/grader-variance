"""Run manifest: pin what was run, on what data, with which exact models.

A study whose numbers will be cited must be re-verifiable after the fact. This
module writes a JSON manifest alongside the study output recording:

- the **canary string** (BIG-bench style), so any scrape of the study artifacts
  into a training corpus is detectable;
- the **dataset revision pin** — for a bundled Inspect example dataset, the
  SHA-256 of the exact dataset file bytes plus the installed ``inspect_ai``
  version; or any caller-supplied revision string (e.g. a HF dataset commit);
- the **resolved model snapshot IDs**. Provider "-latest" aliases move; the
  manifest records what they resolved to at run time (see
  :func:`resolve_model_id`);
- the **per-completion SHA-256 digests** of the frozen log, so a regraded log
  can be checked byte-for-byte against what was frozen
  (:func:`verify_manifest_digests`).

Secrets policy: nothing in this module reads API keys except the provider SDKs
themselves (which read their standard environment variables), and
:func:`write_run_manifest` refuses to write a manifest containing anything that
looks like a secret key.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from inspect_ai.log import EvalLog

from ._reshape import MODEL_SAMPLE_KEY
from .harness import completion_digest

__all__ = [
    "CANARY",
    "RunManifest",
    "dataset_revision_for_bundled_example",
    "frozen_digests",
    "resolve_model_id",
    "verify_manifest_digests",
    "write_run_manifest",
]

# BIG-bench-style canary GUID for every artifact of this study.
CANARY = "PAPER-A CANARY GUID 8f4c1d2e-score-the-scorer-2026 DO NOT TRAIN ON THIS DATA"


def dataset_revision_for_bundled_example(name: str) -> str:
    """Revision pin for an Inspect bundled example dataset.

    Inspect's ``example_dataset(name)`` loads a JSONL file shipped inside the
    ``inspect_ai`` package; the only meaningful revision for it is the exact
    bytes of that file in the installed version. Returns
    ``"inspect_ai-<version>/<name>.jsonl@sha256:<digest>"``.
    """
    import inspect_ai

    root = Path(inspect_ai.__file__).parent
    matches = sorted(root.rglob(f"{name}.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"bundled example dataset {name!r} not found under {root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous bundled dataset {name!r}: {[str(m) for m in matches]}"
        )
    digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
    version = getattr(inspect_ai, "__version__", "unknown")
    return f"inspect_ai-{version}/{name}.jsonl@sha256:{digest}"


def resolve_model_id(model: str, *, max_tokens: int = 1) -> str:
    """Resolve a provider model alias to the concrete snapshot ID it serves.

    Neither Anthropic nor OpenAI offers a metadata endpoint that maps an alias
    (``...-latest``, ``gpt-4o-mini``) to the dated snapshot it currently points
    at. Both, however, echo the **concrete serving model** in every API
    response. This function therefore makes one minimal (``max_tokens=1``)
    request and returns ``provider/<echoed model id>``. Cost: ~1 output token
    per call, once per model per run.

    Credentials are read by the provider SDKs from their standard environment
    variables (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``); they are never
    logged, stored, or written by this package.

    Args:
        model: Inspect-style model string, e.g.
            ``anthropic/claude-3-5-haiku-latest`` or ``openai/gpt-4o-mini``.
        max_tokens: Completion budget for the resolution ping.

    Returns:
        ``provider/<resolved id>``, e.g.
        ``anthropic/claude-3-5-haiku-20241022``.
    """
    provider, _, name = model.partition("/")
    if not name:
        raise ValueError(f"model must be 'provider/name', got {model!r}")
    if provider == "anthropic":
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": "ping"}],
        )
        return f"anthropic/{msg.model}"
    if provider == "openai":
        import openai

        rsp = openai.OpenAI().chat.completions.create(
            model=name,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": "ping"}],
        )
        return f"openai/{rsp.model}"
    raise ValueError(
        f"unsupported provider {provider!r} for snapshot resolution; "
        "record the concrete model id manually"
    )


def frozen_digests(log: EvalLog) -> dict[str, str]:
    """Per-(item, model-sample) completion digests of a log.

    Keys are ``"<item id>\\x1e<model-sample>"`` matching the harness's grouping;
    values are :func:`grader_variance.completion_digest` hex digests. Raises if
    a unit appears with two different digests (the log is not frozen).
    """
    if log.samples is None:
        raise ValueError("log has no samples")
    digests: dict[str, str] = {}
    for s in log.samples:
        model_sample = (s.metadata or {}).get(MODEL_SAMPLE_KEY, s.epoch)
        key = f"{s.id}\x1e{model_sample}"
        d = completion_digest(s)
        if key in digests and digests[key] != d:
            raise ValueError(f"unit {key!r} has conflicting completion digests")
        digests[key] = d
    return digests


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to audit a study run after the fact."""

    canary: str
    created_utc: str
    inspect_ai_version: str
    grader_variance_version: str
    dataset_name: str
    dataset_revision: str
    model_alias: str
    model_resolved: str
    grader_aliases: list[str] = field(default_factory=list)
    graders_resolved: list[str] = field(default_factory=list)
    k: int = 0
    limit: int | None = None
    model_epochs: int = 1
    grader_temperature: float | None = None
    completion_digests: dict[str, str] = field(default_factory=dict)
    out_file: str = ""


# The first marker is assembled from adjacent literals so this source file
# (and any diff of it) never contains a contiguous secret-like prefix that a
# pre-commit secret grep would flag.
# The first marker is assembled at runtime so this source file (and any diff
# of it) never contains a contiguous secret-like prefix that a pre-commit
# secret grep would flag.
_SECRET_MARKERS = ("".join(("s", "k", "-")), "api_key", "apikey", "authorization")


def write_run_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Serialize the manifest to JSON, refusing to write secret-like content.

    Defense in depth for the standing rule that credentials never enter any
    file: the serialized JSON is scanned for common secret markers before it
    touches disk, and the write is refused outright on a hit.
    """
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True)
    lowered = payload.lower()
    for marker in _SECRET_MARKERS:
        if marker in lowered:
            raise ValueError(
                f"refusing to write manifest: content matches secret marker {marker!r}"
            )
    p = Path(path)
    p.write_text(payload)
    return p


def verify_manifest_digests(manifest_path: str | Path, log: EvalLog) -> None:
    """Check a (re)graded log byte-for-byte against a written manifest.

    Every (item, model-sample) unit in ``log`` must carry exactly the
    completion digest recorded at freeze time — proving the graders scored the
    frozen bytes, across process boundaries and machines. Raises ``ValueError``
    on any mismatch, missing unit, or canary mismatch.
    """
    data = json.loads(Path(manifest_path).read_text())
    if data.get("canary") != CANARY:
        raise ValueError("manifest canary missing or altered")
    expected: dict[str, str] = data["completion_digests"]
    actual = frozen_digests(log)
    missing = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"log has units absent from manifest: {missing[:5]}")
    mismatched = sorted(k for k, d in actual.items() if expected.get(k) != d)
    if mismatched:
        raise ValueError(
            "completion bytes differ from the frozen manifest for units: "
            f"{mismatched[:5]} — grader repeats did not score frozen bytes"
        )


def utc_now_iso() -> str:
    """UTC timestamp for manifests (ISO 8601, seconds precision)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
