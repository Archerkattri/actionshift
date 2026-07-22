"""Tests for actionshift-sprint status/run plumbing that need no GPU or scheduler."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from actionshift.benchmarking.schema import SprintJob, SprintResult
from actionshift.benchmarking.store import ArtifactStore
from actionshift.cli import sprint


def _jobs() -> tuple[SprintJob, ...]:
    return (
        SprintJob(
            gate="gate1",
            task="pick_cube",
            method="dualabi",
            seed=1,
            gpu=0,
            budget_steps=1000,
            condition="seen",
        ),
        SprintJob(
            gate="gate1",
            task="push_cube",
            method="entropy",
            seed=2,
            gpu=0,
            budget_steps=1000,
            condition="seen",
        ),
    )


def test_status_counts_terminal_and_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = _jobs()
    jobs_path = tmp_path / "jobs.jsonl"
    sprint._write_jobs(jobs_path, jobs)
    store = ArtifactStore(tmp_path / "artifacts")
    store.finish(
        SprintResult(jobs[0].job_id, "completed", 0, 1.0, None, None)
    )  # terminal
    assert store.claim(jobs[1], attempt=0) is True  # running

    exit_code = sprint._status(
        argparse.Namespace(jobs=jobs_path, artifacts=tmp_path / "artifacts")
    )
    assert exit_code == 0
    import json

    printed = json.loads(capsys.readouterr().out)
    assert printed == {"total": 2, "terminal": 1, "running": 1}


def test_run_rejects_job_gpu_outside_allowed_set(tmp_path: Path) -> None:
    job = SprintJob(
        gate="gate1",
        task="pick_cube",
        method="dualabi",
        seed=1,
        gpu=3,
        budget_steps=1000,
        condition="seen",
    )
    jobs_path = tmp_path / "jobs.jsonl"
    sprint._write_jobs(jobs_path, (job,))
    with pytest.raises(ValueError, match="outside --gpus"):
        sprint._run(argparse.Namespace(jobs=jobs_path, gpus="0,1"))


def test_load_jobs_detects_job_id_tampering(tmp_path: Path) -> None:
    jobs_path = tmp_path / "jobs.jsonl"
    sprint._write_jobs(jobs_path, _jobs()[:1])
    import json

    line = json.loads(jobs_path.read_text(encoding="utf-8"))
    line["job_id"] = "tampered"
    jobs_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="job ID mismatch"):
        sprint._load_jobs(jobs_path)
