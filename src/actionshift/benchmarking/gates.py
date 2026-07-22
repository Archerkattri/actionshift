"""Preregistered competence and wrapper-parity decision rules."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

_FLOORS = {
    "pick_cube": 0.50,
    "push_cube": 0.50,
    "pull_cube": 0.50,
    "stack_cube": 0.50,
    "peg_insertion_side": 0.20,
}


@dataclass(frozen=True, slots=True)
class BinomialInterval:
    successes: int
    total: int
    rate: float
    lower: float
    upper: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompetenceVerdict:
    task: str
    floor: float
    interval: BinomialInterval
    passed: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "interval": self.interval.to_dict()}


@dataclass(frozen=True, slots=True)
class ParityVerdict:
    reference: BinomialInterval
    wrapped: BinomialInterval
    tolerance: float
    absolute_difference: float
    intervals_overlap: bool
    passed: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reference": self.reference.to_dict(),
            "wrapped": self.wrapped.to_dict(),
        }


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> BinomialInterval:
    """Compute a two-sided Wilson score interval for an observed success rate."""
    if total <= 0:
        raise ValueError("at least one episode is required")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie between zero and total")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    rate = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (rate + z_squared / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / total + z_squared / (4 * total * total))
        / denominator
    )
    return BinomialInterval(
        successes, total, rate, center - half_width, center + half_width, confidence
    )


def competence_verdict(task: str, successes: int, total: int) -> CompetenceVerdict:
    if task not in _FLOORS:
        raise ValueError(f"unknown competence task: {task}")
    interval = wilson_interval(successes, total)
    floor = _FLOORS[task]
    passed = interval.rate >= floor
    reason = None if passed else f"success {interval.rate:.4f} below floor {floor:.4f}"
    return CompetenceVerdict(task, floor, interval, passed, reason)


def parity_verdict(
    reference_successes: int,
    reference_total: int,
    wrapped_successes: int,
    wrapped_total: int,
    *,
    tolerance: float = 0.02,
) -> ParityVerdict:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    reference = wilson_interval(reference_successes, reference_total)
    wrapped = wilson_interval(wrapped_successes, wrapped_total)
    difference = abs(reference.rate - wrapped.rate)
    overlap = max(reference.lower, wrapped.lower) <= min(reference.upper, wrapped.upper)
    passed = difference <= tolerance + 1e-12 or overlap
    reason = None
    if not passed:
        reason = (
            f"absolute success difference {difference:.4f} exceeds {tolerance:.4f} "
            "and CIs do not overlap"
        )
    return ParityVerdict(reference, wrapped, tolerance, difference, overlap, passed, reason)


def _success_count(path: Path) -> tuple[int, int]:
    successes = 0
    total = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("success"), bool):
            raise ValueError(f"invalid parity episode on {path}:{line_number}")
        total += 1
        successes += int(value["success"])
    if total == 0:
        raise ValueError(f"at least one episode is required in {path}")
    return successes, total


def analyze_parity_directory(path: Path) -> dict[str, Any]:
    """Apply competence and parity gates to conventionally named raw episode files."""
    task_reports: dict[str, Any] = {}
    all_passed = True
    available_tasks = 0
    for task in _FLOORS:
        files = {
            condition: path / f"{task}-{condition}.jsonl"
            for condition in ("unwrapped", "identity", "oracle_nonidentity")
        }
        if not all(candidate.is_file() for candidate in files.values()):
            task_reports[task] = {
                "passed": False,
                "reason": "missing parity episode files",
                "files": {key: str(value) for key, value in files.items()},
            }
            continue
        available_tasks += 1
        counts = {condition: _success_count(candidate) for condition, candidate in files.items()}
        reference_successes, reference_total = counts["unwrapped"]
        competence = competence_verdict(task, reference_successes, reference_total)
        identity = parity_verdict(
            reference_successes, reference_total, *counts["identity"]
        )
        oracle = parity_verdict(
            reference_successes, reference_total, *counts["oracle_nonidentity"]
        )
        passed = competence.passed and identity.passed and oracle.passed
        task_reports[task] = {
            "passed": passed,
            "reason": None if passed else "one or more preregistered task gates failed",
            "competence": competence.to_dict(),
            "identity_parity": identity.to_dict(),
            "oracle_parity": oracle.to_dict(),
            "files": {key: str(value) for key, value in files.items()},
        }
        all_passed = all_passed and passed
    return {
        "schema_version": "1.0",
        "tasks": task_reports,
        "all_available_tasks_passed": available_tasks > 0 and all_passed,
        "all_tasks_complete": available_tasks == len(_FLOORS),
    }


def overall_gate_status(parity_report: dict[str, Any]) -> tuple[str, str | None]:
    """Explain whether Gate 0 passed, failed scientifically, or lacks artifacts."""
    if not bool(parity_report.get("all_tasks_complete")):
        return "incomplete", "competence and parity require all episode-level artifacts"
    if not bool(parity_report.get("all_available_tasks_passed")):
        return "failed", "one or more preregistered task gates failed"
    return "passed", None
