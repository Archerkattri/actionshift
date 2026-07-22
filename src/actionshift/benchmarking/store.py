"""Crash-safe, append-only storage for benchmark sprint artifacts."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from actionshift.benchmarking.schema import SprintJob, SprintResult


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


class ArtifactStore:
    """Own exclusive claims and immutable terminal results under one directory."""

    def __init__(self, root: Path, *, now: Callable[[], float] = time.time) -> None:
        self.root = root
        self.claims = root / "claims"
        self.results = root / "results"
        self.events = root / "events.jsonl"
        self._now = now
        self.claims.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def claim(self, job: SprintJob, *, attempt: int) -> bool:
        """Claim a nonterminal job without racing another local worker."""
        if attempt < 0:
            raise ValueError("attempt must be nonnegative")
        result_path = self.results / f"{job.job_id}.json"
        if result_path.exists():
            _read_object(result_path)
            return False
        claim_path = self.claims / f"{job.job_id}.json"
        payload = {"job": job.to_dict(), "attempt": attempt, "claimed_at": self._now()}
        try:
            with claim_path.open("x", encoding="utf-8") as destination:
                json.dump(payload, destination, sort_keys=True, allow_nan=False)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
        except FileExistsError:
            return False
        return True

    def finish(self, result: SprintResult) -> None:
        """Publish one terminal result and append its event exactly once."""
        destination = self.results / f"{result.job_id}.json"
        if destination.exists():
            raise FileExistsError(f"terminal result already exists: {result.job_id}")
        payload = _canonical_line(result.to_dict())
        temporary = self.results / f".{result.job_id}.{os.getpid()}.tmp"
        try:
            with temporary.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"terminal result already exists: {result.job_id}"
                ) from error
        finally:
            temporary.unlink(missing_ok=True)

        descriptor = os.open(self.events, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        (self.claims / f"{result.job_id}.json").unlink(missing_ok=True)

    def pending(
        self,
        jobs: Iterable[SprintJob],
        *,
        stale_after_seconds: float,
    ) -> list[SprintJob]:
        """Return unclaimed work, reclaiming claims older than the supplied limit."""
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be nonnegative")
        pending: list[SprintJob] = []
        for job in jobs:
            result_path = self.results / f"{job.job_id}.json"
            if result_path.exists():
                _read_object(result_path)
                continue
            claim_path = self.claims / f"{job.job_id}.json"
            if not claim_path.exists():
                pending.append(job)
                continue
            _read_object(claim_path)
            age = self._now() - claim_path.stat().st_mtime
            if age > stale_after_seconds:
                claim_path.unlink(missing_ok=True)
                pending.append(job)
        return pending
