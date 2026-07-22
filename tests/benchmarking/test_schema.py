from __future__ import annotations

import json
import math

import pytest

from actionshift.benchmarking.schema import SprintJob, SprintResult


def _job(**changes: object) -> SprintJob:
    values: dict[str, object] = {
        "gate": "gate0",
        "task": "pick_cube",
        "method": "ppo",
        "seed": 20260718,
        "gpu": 0,
        "budget_steps": 100_000,
        "condition": "identity",
    }
    values.update(changes)
    return SprintJob(**values)  # type: ignore[arg-type]


def test_job_id_is_stable_and_sensitive_to_scientific_inputs() -> None:
    base = _job()

    assert base.job_id == _job().job_id
    assert base.job_id != _job(budget_steps=200_000).job_id
    assert len(base.job_id) == 16
    assert base.to_dict() == {
        "job_id": base.job_id,
        "schema_version": "1.0",
        "gate": "gate0",
        "task": "pick_cube",
        "method": "ppo",
        "seed": 20260718,
        "gpu": 0,
        "budget_steps": 100_000,
        "condition": "identity",
    }
    json.dumps(base.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gate", "gate2", "gate"),
        ("task", "", "task"),
        ("method", "", "method"),
        ("gpu", -1, "gpu"),
        ("budget_steps", 0, "budget"),
        ("condition", "", "condition"),
    ],
)
def test_job_rejects_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _job(**{field: value})


def test_result_accepts_only_terminal_finite_outcomes() -> None:
    job = _job()
    result = SprintResult(job.job_id, "completed", 0, 1.25, "checkpoint.pt", None)

    assert result.to_dict() == {
        "schema_version": "1.0",
        "job_id": job.job_id,
        "status": "completed",
        "attempt": 0,
        "elapsed_seconds": 1.25,
        "checkpoint": "checkpoint.pt",
        "error": None,
    }

    with pytest.raises(ValueError, match="terminal"):
        SprintResult(job.job_id, "running", 0, 1.0, None, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        SprintResult(job.job_id, "failed", 0, math.inf, None, "oom")
    with pytest.raises(ValueError, match="attempt"):
        SprintResult(job.job_id, "failed", -1, 1.0, None, "oom")
