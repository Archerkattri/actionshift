from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from actionshift.benchmarking.schema import SprintJob, SprintResult
from actionshift.benchmarking.store import ArtifactStore


def _job(task: str = "pick_cube") -> SprintJob:
    return SprintJob("gate0", task, "ppo", 20260718, 0, 100_000, "identity")


def test_claim_is_exclusive_and_finish_is_atomic(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    job = _job()

    assert store.claim(job, attempt=0)
    assert not store.claim(job, attempt=0)
    claim = json.loads((tmp_path / "claims" / f"{job.job_id}.json").read_text())
    assert claim["job"] == job.to_dict()
    assert claim["attempt"] == 0

    result = SprintResult(job.job_id, "completed", 0, 2.5, "model.pt", None)
    store.finish(result)

    destination = tmp_path / "results" / f"{job.job_id}.json"
    assert json.loads(destination.read_text()) == result.to_dict()
    assert not (tmp_path / "claims" / f"{job.job_id}.json").exists()
    events = (tmp_path / "events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in events] == [result.to_dict()]
    assert not list((tmp_path / "results").glob("*.tmp"))


def test_terminal_result_cannot_be_overwritten(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    job = _job()
    assert store.claim(job, attempt=0)
    store.finish(SprintResult(job.job_id, "failed", 0, 1.0, None, "oom"))

    with pytest.raises(FileExistsError, match="terminal"):
        store.finish(SprintResult(job.job_id, "completed", 1, 2.0, "model.pt", None))

    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


def test_pending_excludes_terminal_and_active_but_recovers_stale_claims(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path, now=lambda: 1_000.0)
    completed = _job("pick_cube")
    active = _job("push_cube")
    stale = _job("peg_insertion_side")
    assert store.claim(completed, attempt=0)
    store.finish(SprintResult(completed.job_id, "completed", 0, 1.0, None, None))
    assert store.claim(active, attempt=0)
    assert store.claim(stale, attempt=1)
    stale_path = tmp_path / "claims" / f"{stale.job_id}.json"
    os.utime(stale_path, (800.0, 800.0))

    assert store.pending(
        [completed, active, stale], stale_after_seconds=100.0
    ) == [stale]
    assert not stale_path.exists()
    assert (tmp_path / "claims" / f"{active.job_id}.json").exists()


def test_store_rejects_malformed_existing_result(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    job = _job()
    result_path = tmp_path / "results" / f"{job.job_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        store.pending([job], stale_after_seconds=10.0)
