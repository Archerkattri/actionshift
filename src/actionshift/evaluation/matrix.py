"""Frozen cross-task, cross-contract, cross-method evaluation matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

DEFAULT_SEEDS = (20260718, 20260719, 20260720, 20260721, 20260722)
TASKS = ("pick_cube", "push_cube", "peg_insertion_side")
SPLITS = ("seen", "unseen_value", "unseen_composition", "long_lag", "task_transfer")
METHODS = (
    "oracle",
    "no_adapt",
    "domain_randomized",
    "recurrent",
    "osi",
    "rma",
    "random_probes",
    "fixed_probes",
    "dualabi",
    "dualabi_entropy",
)
InformationMode = Literal["task_regret", "entropy", "none", "not_applicable"]


@dataclass(frozen=True, slots=True)
class MatrixJob:
    task: str
    split: str
    method: str
    seed: int
    information_mode: InformationMode
    backend: str = "maniskill"
    schema_version: str = "1.0"

    @property
    def job_id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, **asdict(self)}


def _information_mode(method: str) -> InformationMode:
    if method == "dualabi":
        return "task_regret"
    if method == "dualabi_entropy":
        return "entropy"
    return "not_applicable"


def build_matrix() -> tuple[MatrixJob, ...]:
    """Return the immutable 750-job headline matrix (five independent seeds)."""
    return tuple(
        MatrixJob(task, split, method, seed, _information_mode(method))
        for task in TASKS
        for split in SPLITS
        for method in METHODS
        for seed in DEFAULT_SEEDS
    )


def matrix_jsonl() -> str:
    return "\n".join(json.dumps(job.to_dict(), sort_keys=True) for job in build_matrix()) + "\n"
