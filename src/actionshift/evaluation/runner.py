"""Atomic artifacts for the frozen evaluation matrix and episode summaries."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from actionshift.evaluation.matrix import build_matrix
from actionshift.evaluation.metrics import EpisodeMetrics, summarize


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def write_matrix(path: Path) -> int:
    jobs = build_matrix()
    contents = "\n".join(json.dumps(job.to_dict(), sort_keys=True) for job in jobs) + "\n"
    _atomic_text(path, contents)
    return len(jobs)


def load_episodes(path: Path) -> list[EpisodeMetrics]:
    allowed = {field.name for field in fields(EpisodeMetrics)}
    episodes: list[EpisodeMetrics] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != allowed:
            raise ValueError(f"episode schema mismatch on line {line_number}")
        episodes.append(EpisodeMetrics(**value))
    return episodes


def summarize_file(source: Path, destination: Path) -> dict[str, Any]:
    report = summarize(load_episodes(source))
    _atomic_text(destination, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
