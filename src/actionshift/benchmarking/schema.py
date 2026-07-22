"""Immutable schemas shared by sprint planners, workers, and analyzers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

Gate = Literal["gate0", "gate1", "actionabi"]
TerminalStatus = Literal["completed", "failed", "pruned", "deadline", "inapplicable"]
_TERMINAL_STATUSES = frozenset({"completed", "failed", "pruned", "deadline", "inapplicable"})


@dataclass(frozen=True, slots=True)
class SprintJob:
    """One scientifically distinct, explicitly assigned benchmark run."""

    gate: Gate
    task: str
    method: str
    seed: int
    gpu: int
    budget_steps: int
    condition: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.gate not in {"gate0", "gate1", "actionabi"}:
            raise ValueError("gate must be gate0, gate1, or actionabi")
        if not self.task:
            raise ValueError("task must be nonempty")
        if not self.method:
            raise ValueError("method must be nonempty")
        if self.gpu < 0:
            raise ValueError("gpu must be nonnegative")
        if self.budget_steps <= 0:
            raise ValueError("budget_steps must be positive")
        if not self.condition:
            raise ValueError("condition must be nonempty")

    @property
    def job_id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, **asdict(self)}


@dataclass(frozen=True, slots=True)
class SprintResult:
    """Terminal status written exactly once for a sprint job."""

    job_id: str
    status: TerminalStatus
    attempt: int
    elapsed_seconds: float
    checkpoint: str | None
    error: str | None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must be nonempty")
        if self.status not in _TERMINAL_STATUSES:
            raise ValueError("status must be terminal")
        if self.attempt < 0:
            raise ValueError("attempt must be nonnegative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
