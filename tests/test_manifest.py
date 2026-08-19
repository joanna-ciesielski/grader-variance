"""Run-manifest: digests, canary, secret refusal, dataset pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grader_variance.manifest import (
    CANARY,
    RunManifest,
    dataset_revision_for_bundled_example,
    write_run_manifest,
)


def make_manifest(**overrides: object) -> RunManifest:
    base: dict[str, object] = {
        "canary": CANARY,
        "created_utc": "2026-08-19T00:00:00Z",
        "inspect_ai_version": "0.0.0-test",
        "grader_variance_version": "0.1.0",
        "dataset_name": "theory_of_mind",
        "dataset_revision": "test-revision",
        "model_alias": "anthropic/claude-3-5-sonnet-latest",
        "model_resolved": "anthropic/claude-3-5-sonnet-20241022",
        "grader_aliases": ["openai/gpt-4o-mini"],
        "graders_resolved": ["openai/gpt-4o-mini-2024-07-18"],
        "k": 12,
        "limit": 50,
        "model_epochs": 3,
        "grader_temperature": 1.0,
        "completion_digests": {"q1\x1e1": "ab" * 32},
        "out_file": "demo_results.json",
    }
    base.update(overrides)
    return RunManifest(**base)  # type: ignore[arg-type]


class TestWriteManifest:
    def test_roundtrip(self, tmp_path: Path) -> None:
        p = write_run_manifest(tmp_path / "m.json", make_manifest())
        data = json.loads(p.read_text())
        assert data["canary"] == CANARY
        assert "DO NOT TRAIN ON THIS DATA" in data["canary"]
        assert data["model_resolved"].endswith("20241022")
        assert data["completion_digests"] == {"q1\x1e1": "ab" * 32}

    def test_refuses_secret_content(self, tmp_path: Path) -> None:
        # Marker assembled at runtime so no secret-like literal ever appears
        # in the source tree or in a staged diff.
        fake_marker = "".join(["s", "k", "-"]) + "oops-not-a-real-key"
        bad = make_manifest(dataset_revision=fake_marker)
        with pytest.raises(ValueError, match="secret marker"):
            write_run_manifest(tmp_path / "m.json", bad)
        assert not (tmp_path / "m.json").exists()


class TestDatasetPin:
    def test_bundled_theory_of_mind_pin(self) -> None:
        rev = dataset_revision_for_bundled_example("theory_of_mind")
        assert rev.startswith("inspect_ai-")
        assert "theory_of_mind.jsonl@sha256:" in rev
        digest = rev.rsplit("sha256:", 1)[1]
        assert len(digest) == 64
        # Deterministic: same installed file, same pin.
        assert rev == dataset_revision_for_bundled_example("theory_of_mind")

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            dataset_revision_for_bundled_example("no_such_dataset_xyz")
