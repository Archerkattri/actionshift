"""Tests for evaluation.runner.summarize_file (the actionshift-evaluate summarize path)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from actionshift.evaluation.metrics import EpisodeMetrics, summarize
from actionshift.evaluation.runner import load_episodes, summarize_file


def _episode(*, success: bool, recovery_steps: int | None) -> EpisodeMetrics:
    return EpisodeMetrics(
        success=success,
        episode_steps=50,
        recovery_steps=recovery_steps,
        unintended_displacement=0.01,
        safety_violations=0,
        cumulative_action_cost=1.5,
        task_return=0.8 if success else 0.1,
        posterior_true_probability=0.9 if success else 0.2,
        posterior_entropy=0.3,
    )


def _write_jsonl(path: Path, episodes: list[EpisodeMetrics]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(episode), sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )


def test_summarize_file_matches_direct_summary(tmp_path: Path) -> None:
    episodes = [
        _episode(success=True, recovery_steps=4),
        _episode(success=False, recovery_steps=None),
        _episode(success=True, recovery_steps=7),
    ]
    source = tmp_path / "episodes.jsonl"
    destination = tmp_path / "summary.json"
    _write_jsonl(source, episodes)

    report = summarize_file(source, destination)

    assert report == summarize(load_episodes(source))
    assert report["episode_count"] == 3
    assert report["success_rate"]["mean"] == pytest.approx(2 / 3)
    assert report["recovery_steps"]["observed_count"] == 2
    assert report["recovery_steps"]["censored_count"] == 1
    # File is written atomically with the same content it returns.
    assert json.loads(destination.read_text(encoding="utf-8")) == report


def test_load_episodes_rejects_schema_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"success": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_episodes(source)
