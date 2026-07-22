"""Aggregate imitation-brittleness slices into the DP-vs-PPO comparison tables.

Reads the per-task imitation manifests (competence gate + per-cell success),
pools multi-seed cells with Wilson intervals from the hash-addressed summaries,
pairs each cell against the frozen PPO tournament/Gate-1 reference, and prints
the three report tables: clean competence, brittleness (no_adapt), and rescue
(belief family). PPO references are read from the committed tournament.json plus
the Gate-1 slice JSONL (no_adapt / oracle cells).
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from actionshift.benchmarking.gates import wilson_interval

_TASKS = ("pick_cube", "push_cube")
_SPLITS = ("seen", "unseen_composition")
_METHODS = ("no_adapt", "oracle", "exact_belief", "entropy_probes", "dualabi")


def ppo_reference() -> dict[tuple[str, str, str], float]:
    ref: dict[tuple[str, str, str], float] = {}
    tournament = json.loads(Path("artifacts/adaptation/tournament.json").read_text())
    for key, cell in tournament["cells"].items():
        task, method, split = key.split("|")
        ref[(task, method, split)] = cell["success_rate"]
    agg: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for path in glob.glob("artifacts/sprint/gate1/*.jsonl"):
        name = Path(path).name
        if not (("no_adapt" in name or "oracle" in name)
                and ("-seen-" in name or "unseen_composition" in name)):
            continue
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if "success" not in record:
                continue
            key3 = (record["task"], record["method"], record["split"])
            agg[key3][0] += int(bool(record["success"]))
            agg[key3][1] += 1
    for key3, (successes, total) in agg.items():
        if total:
            ref[key3] = successes / total
    return ref


def pool_cell(slices_dir: Path, task: str, method: str, split: str) -> tuple[float, int]:
    successes = 0
    total = 0
    for path in glob.glob(str(slices_dir / "*.summary.json")):
        summary = json.loads(Path(path).read_text())
        if (summary.get("backbone") == "diffusion_policy" and summary["task"] == task
                and summary["method"] == method and summary["split"] == split):
            for contract in summary["per_contract"]:
                successes += contract["successes"]
                total += contract["episodes"]
    if total == 0:
        return float("nan"), 0
    return successes / total, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", type=Path,
                        default=Path("artifacts/adaptation/imitation_slices"))
    arguments = parser.parse_args()
    ref = ppo_reference()

    print("=== CLEAN-INTERFACE COMPETENCE (identity contract, no_adapt) ===")
    for task in _TASKS:
        manifest_path = arguments.slices / f"manifest_{task}.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            gate = manifest.get("competence_clean", float("nan"))
            print(f"  DP {task:11s} clean identity success = {gate:.3f}")

    print("\n=== BRITTLENESS + RESCUE (DP vs PPO, pooled) ===")
    header = (f"{'task/split':28s} {'method':16s} {'DP':>7s} {'PPO':>7s} "
              f"{'DP 95% CI':>18s} {'n':>5s}")
    print(header)
    for task in _TASKS:
        for split in _SPLITS:
            for method in _METHODS:
                dp_rate, n = pool_cell(arguments.slices, task, method, split)
                ppo_rate = ref.get((task, method, split), float("nan"))
                if n:
                    interval = wilson_interval(round(dp_rate * n), n)
                    ci = f"[{interval.lower:.3f}, {interval.upper:.3f}]"
                else:
                    ci = "--"
                print(f"{task + '/' + split:28s} {method:16s} {dp_rate:7.3f} "
                      f"{ppo_rate:7.3f} {ci:>18s} {n:5d}")
        print()


if __name__ == "__main__":
    main()
