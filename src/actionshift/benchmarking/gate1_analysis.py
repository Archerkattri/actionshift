"""Aggregate episode-level hidden-contract PPO slices with matched pairing."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from actionshift.benchmarking.gates import wilson_interval
from actionshift.evaluation.statistics import paired_success_difference, superiority_allowed


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sorted(path.glob("*.jsonl")):
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            required = {
                "task", "split", "method", "seed", "contract_sha256",
                "episode_index", "success", "task_return",
            }
            if not isinstance(value, dict) or not required <= set(value):
                raise ValueError(f"invalid Gate 1 record at {source}:{line_number}")
            if not isinstance(value["success"], bool):
                raise ValueError(f"success must be boolean at {source}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"no Gate 1 JSONL records under {path}")
    return records


def analyze_gate1_directory(
    path: Path, *, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    records = _load_records(path)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["task"], record["split"], record["method"])].append(record)
    group_reports: dict[str, Any] = {}
    for (task, split, method), rows in sorted(groups.items()):
        successes = sum(bool(row["success"]) for row in rows)
        group_reports[f"{task}/{split}/{method}"] = {
            "episodes": len(rows),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "success": asdict(wilson_interval(successes, len(rows))),
            "mean_return": fmean(float(row["task_return"]) for row in rows),
        }

    comparisons: dict[str, Any] = {}
    task_splits = sorted({(record["task"], record["split"]) for record in records})
    for task, split in task_splits:
        rows = [
            record
            for record in records
            if record["task"] == task and record["split"] == split
        ]
        by_method: dict[str, dict[tuple[int, str, int], dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            key = (int(row["seed"]), str(row["contract_sha256"]), int(row["episode_index"]))
            if key in by_method[str(row["method"])]:
                raise ValueError(f"duplicate pairing key for {task}/{split}/{row['method']}")
            by_method[str(row["method"])][key] = row
        if "oracle" not in by_method or "no_adapt" not in by_method:
            continue
        oracle_keys = set(by_method["oracle"])
        baseline_keys = set(by_method["no_adapt"])
        if oracle_keys != baseline_keys:
            raise ValueError(f"pairing keys differ for {task}/{split}")
        keys = sorted(oracle_keys)
        seeds = sorted({key[0] for key in keys})
        interval = paired_success_difference(
            [bool(by_method["oracle"][key]["success"]) for key in keys],
            [bool(by_method["no_adapt"][key]["success"]) for key in keys],
            samples=bootstrap_samples,
            seed=20260718,
        )
        comparisons[f"{task}/{split}"] = {
            "oracle_minus_no_adapt": asdict(interval),
            "matched_seeds": seeds,
            "three_seed_comparison": superiority_allowed(seeds),
            "interpretation": "privileged ceiling gap; not an adaptation-method superiority claim",
        }
    three_seed = bool(comparisons) and all(
        comparison["three_seed_comparison"] for comparison in comparisons.values()
    )
    return {
        "schema_version": "1.0",
        "groups": group_reports,
        "comparisons": comparisons,
        "claim_status": (
            "matched_three_seed_ceiling_comparison" if three_seed else "descriptive_only"
        ),
    }
