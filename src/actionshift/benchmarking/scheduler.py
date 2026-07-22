"""Deadline-aware subprocess scheduling with explicit single-GPU visibility."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from actionshift.benchmarking.schema import SprintJob, SprintResult
from actionshift.benchmarking.store import ArtifactStore

QueryRunner = Callable[[list[str]], tuple[int, str, str]]


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    index: int
    name: str
    memory_free_mib: int
    memory_total_mib: int
    utilization_percent: float


@dataclass(frozen=True, slots=True)
class CommandSpec:
    job: SprintJob
    argv: tuple[str, ...]
    cwd: Path | None = None
    checkpoint: Path | None = None
    min_free_memory_mib: int = 1024
    transient_exit_codes: frozenset[int] = frozenset({75})

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("argv must be nonempty")
        if self.min_free_memory_mib < 0:
            raise ValueError("min_free_memory_mib must be nonnegative")


def _query_command(command: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def query_gpus(command_runner: QueryRunner = _query_command) -> list[GpuSnapshot]:
    """Read free memory and utilization without importing or initializing CUDA."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    returncode, stdout, stderr = command_runner(command)
    if returncode != 0:
        raise RuntimeError(
            json.dumps(
                {"command": command, "returncode": returncode, "stderr": stderr.strip()},
                sort_keys=True,
            )
        )
    snapshots: list[GpuSnapshot] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=4)]
        if len(fields) != 5:
            raise RuntimeError(json.dumps({"malformed_nvidia_smi_row": line}))
        try:
            snapshots.append(
                GpuSnapshot(
                    index=int(fields[0]),
                    name=fields[1],
                    memory_free_mib=int(fields[2]),
                    memory_total_mib=int(fields[3]),
                    utilization_percent=float(fields[4]),
                )
            )
        except ValueError as error:
            raise RuntimeError(json.dumps({"malformed_nvidia_smi_row": line})) from error
    return snapshots


def _query_assigned_gpu(index: int) -> list[GpuSnapshot]:
    return [snapshot for snapshot in query_gpus() if snapshot.index == index]


class SprintScheduler:
    """Run one process per GPU at a time and preserve every terminal outcome."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        gpu_query: Callable[[int], list[GpuSnapshot]] = _query_assigned_gpu,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self._gpu_query = gpu_query
        self._monotonic = monotonic
        self._sleep = sleep
        self.logs = store.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        specs: Iterable[CommandSpec],
        *,
        deadline_seconds: float,
        max_retries: int = 0,
    ) -> dict[str, SprintResult]:
        if deadline_seconds < 0:
            raise ValueError("deadline_seconds must be nonnegative")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        deadline = self._monotonic() + deadline_seconds
        by_gpu: dict[int, list[CommandSpec]] = defaultdict(list)
        for spec in specs:
            by_gpu[spec.job.gpu].append(spec)
        results: dict[str, SprintResult] = {}

        def run_gpu(queue: list[CommandSpec]) -> None:
            for spec in queue:
                result = self._run_one(spec, deadline=deadline, max_retries=max_retries)
                if result is not None:
                    results[result.job_id] = result

        if not by_gpu:
            return results
        with ThreadPoolExecutor(max_workers=len(by_gpu)) as executor:
            futures = [executor.submit(run_gpu, queue) for queue in by_gpu.values()]
            for future in futures:
                future.result()
        return results

    def _finish(self, result: SprintResult) -> SprintResult:
        self.store.finish(result)
        return result

    def _run_one(
        self,
        spec: CommandSpec,
        *,
        deadline: float,
        max_retries: int,
    ) -> SprintResult | None:
        started = self._monotonic()
        if not self.store.claim(spec.job, attempt=0):
            return None
        if started >= deadline:
            return self._finish(
                SprintResult(spec.job.job_id, "deadline", 0, 0.0, None, "launch deadline")
            )

        while True:
            try:
                snapshots = self._gpu_query(spec.job.gpu)
            except Exception as error:
                elapsed = max(0.0, self._monotonic() - started)
                return self._finish(
                    SprintResult(
                        spec.job.job_id,
                        "failed",
                        0,
                        elapsed,
                        None,
                        f"GPU query failed: {error}",
                    )
                )
            if len(snapshots) != 1:
                elapsed = max(0.0, self._monotonic() - started)
                return self._finish(
                    SprintResult(
                        spec.job.job_id,
                        "failed",
                        0,
                        elapsed,
                        None,
                        f"assigned GPU {spec.job.gpu} not uniquely visible",
                    )
                )
            if snapshots[0].memory_free_mib >= spec.min_free_memory_mib:
                break
            if self._monotonic() >= deadline:
                elapsed = max(0.0, self._monotonic() - started)
                return self._finish(
                    SprintResult(
                        spec.job.job_id,
                        "deadline",
                        0,
                        elapsed,
                        None,
                        "memory guard did not clear before deadline",
                    )
                )
            self._sleep(1.0)

        for attempt in range(max_retries + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                elapsed = max(0.0, self._monotonic() - started)
                return self._finish(
                    SprintResult(
                        spec.job.job_id, "deadline", attempt, elapsed, None, "launch deadline"
                    )
                )
            stdout_path = self.logs / f"{spec.job.job_id}.attempt-{attempt}.stdout"
            stderr_path = self.logs / f"{spec.job.job_id}.attempt-{attempt}.stderr"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(spec.job.gpu)
            environment["ACTIONSHIFT_JOB_ID"] = spec.job.job_id
            environment["ACTIONSHIFT_ARTIFACT_ROOT"] = str(self.store.root.resolve())
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    process = subprocess.run(
                        spec.argv,
                        check=False,
                        cwd=spec.cwd,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=remaining,
                    )
            except subprocess.TimeoutExpired:
                elapsed = max(0.0, self._monotonic() - started)
                return self._finish(
                    SprintResult(
                        spec.job.job_id,
                        "deadline",
                        attempt,
                        elapsed,
                        None,
                        "process exceeded sprint deadline",
                    )
                )
            elapsed = max(0.0, self._monotonic() - started)
            if process.returncode == 0:
                checkpoint = None
                if spec.checkpoint is not None:
                    if not spec.checkpoint.is_file():
                        return self._finish(
                            SprintResult(
                                spec.job.job_id,
                                "failed",
                                attempt,
                                elapsed,
                                None,
                                "declared checkpoint was not created",
                            )
                        )
                    checkpoint = str(spec.checkpoint.resolve())
                return self._finish(
                    SprintResult(
                        spec.job.job_id, "completed", attempt, elapsed, checkpoint, None
                    )
                )
            if process.returncode not in spec.transient_exit_codes or attempt == max_retries:
                return self._finish(
                    SprintResult(
                        spec.job.job_id,
                        "failed",
                        attempt,
                        elapsed,
                        None,
                        f"process exited {process.returncode} after {attempt + 1} attempts",
                    )
                )
        raise AssertionError("retry loop must return a terminal result")
