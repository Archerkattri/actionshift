from __future__ import annotations

import json
import sys
from pathlib import Path

from actionshift.benchmarking.scheduler import (
    CommandSpec,
    GpuSnapshot,
    SprintScheduler,
    query_gpus,
)
from actionshift.benchmarking.schema import SprintJob
from actionshift.benchmarking.store import ArtifactStore


def _job(task: str, gpu: int) -> SprintJob:
    return SprintJob("gate0", task, "ppo", 20260718, gpu, 100, "identity")


def _available(gpu: int) -> list[GpuSnapshot]:
    return [GpuSnapshot(gpu, "RTX 5090", 30_000, 32_607, 0.0)]


def test_scheduler_exposes_exactly_one_assigned_gpu_to_each_child(tmp_path: Path) -> None:
    jobs = [_job("pick_cube", 0), _job("push_cube", 1)]
    specs = [
        CommandSpec(
            job,
            (
                sys.executable,
                "-c",
                "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
            ),
        )
        for job in jobs
    ]
    scheduler = SprintScheduler(
        ArtifactStore(tmp_path),
        gpu_query=lambda gpu: _available(gpu),
    )

    results = scheduler.run(specs, deadline_seconds=10.0)

    assert {result.status for result in results.values()} == {"completed"}
    for job in jobs:
        log = tmp_path / "logs" / f"{job.job_id}.attempt-0.stdout"
        assert log.read_text(encoding="utf-8").strip() == str(job.gpu)


def test_scheduler_retries_transient_exit_and_keeps_other_jobs_running(
    tmp_path: Path,
) -> None:
    failed = _job("pick_cube", 0)
    passed = _job("push_cube", 1)
    specs = [
        CommandSpec(failed, (sys.executable, "-c", "raise SystemExit(75)")),
        CommandSpec(passed, (sys.executable, "-c", "print('ok')")),
    ]
    scheduler = SprintScheduler(
        ArtifactStore(tmp_path),
        gpu_query=lambda gpu: _available(gpu),
    )

    results = scheduler.run(specs, deadline_seconds=10.0, max_retries=1)

    assert results[failed.job_id].status == "failed"
    assert results[failed.job_id].attempt == 1
    assert results[failed.job_id].error == "process exited 75 after 2 attempts"
    assert results[passed.job_id].status == "completed"
    assert (tmp_path / "logs" / f"{failed.job_id}.attempt-1.stderr").exists()


def test_scheduler_defers_launch_until_memory_guard_passes(tmp_path: Path) -> None:
    job = _job("pick_cube", 0)
    snapshots = [
        [GpuSnapshot(0, "RTX 5090", 100, 32_607, 95.0)],
        _available(0),
    ]
    sleeps: list[float] = []

    def gpu_query(gpu: int) -> list[GpuSnapshot]:
        assert gpu == 0
        return snapshots.pop(0)

    scheduler = SprintScheduler(
        ArtifactStore(tmp_path),
        gpu_query=gpu_query,
        sleep=sleeps.append,
    )
    spec = CommandSpec(job, (sys.executable, "-c", "print('launched')"), min_free_memory_mib=500)

    result = scheduler.run([spec], deadline_seconds=10.0)[job.job_id]

    assert result.status == "completed"
    assert sleeps == [1.0]


def test_scheduler_does_not_launch_after_deadline(tmp_path: Path) -> None:
    job = _job("pick_cube", 0)
    marker = tmp_path / "should-not-exist"
    spec = CommandSpec(
        job,
        (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
    )
    scheduler = SprintScheduler(
        ArtifactStore(tmp_path),
        gpu_query=lambda gpu: _available(gpu),
    )

    result = scheduler.run([spec], deadline_seconds=0.0)[job.job_id]

    assert result.status == "deadline"
    assert not marker.exists()


def test_query_gpus_parses_inventory_and_preserves_command_failure() -> None:
    def success(command: list[str]) -> tuple[int, str, str]:
        assert command[0] == "nvidia-smi"
        return 0, "0, RTX 5090, 30000, 32607, 12\n", ""

    assert query_gpus(success) == [GpuSnapshot(0, "RTX 5090", 30_000, 32_607, 12.0)]

    def failure(command: list[str]) -> tuple[int, str, str]:
        return 9, "", "driver unavailable"

    try:
        query_gpus(failure)
    except RuntimeError as error:
        assert json.loads(str(error))["stderr"] == "driver unavailable"
    else:
        raise AssertionError("query_gpus must preserve nvidia-smi failure")
