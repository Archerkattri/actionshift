"""Atomic, hash-verified training checkpoints."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointState:
    step: int
    metadata: dict[str, Any]


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    torch.save(
        {
            "schema_version": "1.0",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)
    return _sha256(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_sha256: str,
) -> CheckpointState:
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return CheckpointState(step=int(payload["step"]), metadata=dict(payload["metadata"]))
