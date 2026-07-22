"""Aggregate adaptation slices into the preregistered tournament table.

Reads every hash-addressed slice under the output directory, pools the
three-seed cells per (task, method, split) with Wilson intervals, and computes
paired bootstrap differences between methods on episode pairs matched by
(seed, contract, episode index) — the same pairing and statistics discipline as
the Gate 1 analyzer. A paired 95% interval excluding zero is the promotion
criterion signal; nothing here relabels a one-seed cell as promotable.

Usage: .venv/bin/python experiments/analyze_adaptation.py \
  --slices artifacts/adaptation/slices --output artifacts/adaptation/tournament.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from actionshift.benchmarking.gates import wilson_interval
from actionshift.evaluation.statistics import paired_success_difference

_HEADLINE_COMPARISONS = (
    ("entropy_probes", "fixed_probes"),
    ("entropy_probes", "exact_belief"),
    ("fixed_probes", "exact_belief"),
    ("random_probes", "exact_belief"),
    ("dualabi", "entropy_probes"),
    ("dualabi", "fixed_probes"),
    ("dualabi", "exact_belief"),
)


def load_slices(directory: Path) -> list[dict[str, Any]]:
    slices = []
    for summary_path in sorted(directory.glob("*.summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        episodes_path = directory / f"{summary['job_id']}.jsonl"
        summary["episodes"] = [
            json.loads(line) for line in episodes_path.read_text().splitlines()
        ]
        slices.append(summary)
    return slices


def pooled_cells(slices: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in slices:
        grouped[(record["task"], record["method"], record["split"])].append(record)
    for key, group in sorted(grouped.items()):
        episodes = [episode for record in group for episode in record["episodes"]]
        successes = sum(1 for episode in episodes if episode["success"])
        interval = wilson_interval(successes, len(episodes))
        cells[key] = {
            "seeds": sorted({record["seed"] for record in group}),
            "per_seed_rates": {
                str(record["seed"]): record["overall_success_rate"] for record in group
            },
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes),
            "wilson_95": [interval.lower, interval.upper]
            if hasattr(interval, "lower")
            else list(interval),
            "mean_probe_steps": sum(e["probe_steps"] for e in episodes) / len(episodes),
            "mean_probe_displacement": sum(
                e["probe_displacement"] for e in episodes
            ) / len(episodes),
        }
    return {"|".join(key): value for key, value in cells.items()}


def paired_comparisons(slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str, int], dict[tuple[str, int, int], bool]] = {}
    for record in slices:
        key = (record["task"], record["method"], record["split"], record["seed"])
        indexed[key] = {
            (episode["contract_sha256"], episode["seed"], episode["episode_index"]):
                episode["success"]
            for episode in record["episodes"]
        }
    results = []
    tasks = sorted({record["task"] for record in slices})
    splits = ("seen", "unseen_composition")
    for task in tasks:
        for split in splits:
            for candidate, baseline in _HEADLINE_COMPARISONS:
                candidate_values: list[bool] = []
                baseline_values: list[bool] = []
                seeds_used = []
                for seed in (20260718, 20260719, 20260720):
                    left = indexed.get((task, candidate, split, seed))
                    right = indexed.get((task, baseline, split, seed))
                    if left is None or right is None:
                        continue
                    shared = sorted(set(left) & set(right))
                    candidate_values.extend(left[pair] for pair in shared)
                    baseline_values.extend(right[pair] for pair in shared)
                    seeds_used.append(seed)
                if not candidate_values:
                    continue
                interval = paired_success_difference(candidate_values, baseline_values)
                results.append(
                    {
                        "task": task,
                        "split": split,
                        "candidate": candidate,
                        "baseline": baseline,
                        "seeds": seeds_used,
                        "three_seed": len(seeds_used) >= 3,
                        **asdict(interval),
                        "significant_95": interval.lower > 0 or interval.upper < 0,
                    }
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    slices = load_slices(arguments.slices)
    report = {
        "schema_version": "1.0",
        "slice_count": len(slices),
        "cells": pooled_cells(slices),
        "paired_comparisons": paired_comparisons(slices),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, cell in report["cells"].items():
        print(
            f"{name}: {cell['success_rate']:.3f} "
            f"[{cell['wilson_95'][0]:.3f}, {cell['wilson_95'][1]:.3f}] "
            f"({len(cell['seeds'])} seeds, {cell['episodes']} eps)"
        )
    print()
    for comparison in report["paired_comparisons"]:
        marker = "SIG" if comparison["significant_95"] else "ns"
        print(
            f"{comparison['task']}/{comparison['split']}: "
            f"{comparison['candidate']} - {comparison['baseline']} = "
            f"{comparison['estimate']:+.3f} "
            f"[{comparison['lower']:+.3f}, {comparison['upper']:+.3f}] "
            f"{marker} ({'3-seed' if comparison['three_seed'] else 'partial'})"
        )


if __name__ == "__main__":
    main()
