"""Aggregate delay-aware slice summaries into a table with Wilson 95% intervals.

Reads all ``*.summary.json`` under a delay-slices directory and prints, per
(task, method, split), the pooled success rate with a Wilson 95% interval and the
per-contract (per-lag) breakdown. Frozen-backbone baselines from the tournament are
printed alongside for context, with an explicit NEW-BACKBONE disclaimer.

Usage:
  .venv/bin/python experiments/analyze_delay_slices.py \
    --slices artifacts/adaptation/delay_slices
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# Frozen-backbone tournament references (reports/adaptation_tournament.md). These
# are a DIFFERENT (frozen) backbone; printed for context only, not a like-for-like
# method contest against the delay-aware backbone.
FROZEN_REFERENCE = {
    ("pick_cube", "long_lag"): {"oracle": 0.027, "exact_belief": 0.022, "fixed_probes": 0.037},
    ("push_cube", "long_lag"): {"oracle": 0.153, "exact_belief": 0.117, "fixed_probes": 0.215},
    ("pick_cube", "seen"): {"oracle": 1.000, "exact_belief": 0.928},
    ("push_cube", "seen"): {"oracle": 1.000, "exact_belief": 0.983},
}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return phat, max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", type=Path, default=Path("artifacts/adaptation/delay_slices"))
    args = parser.parse_args()

    def variant_of(checkpoint: str) -> str:
        # Resolve the backbone variant (randomized vs curriculum) from the run's
        # config.json so same-directory slices from different backbones never pool.
        config = Path(checkpoint).parent / "config.json"
        try:
            curriculum = json.loads(config.read_text())["lag_curriculum"]
        except (OSError, KeyError, json.JSONDecodeError):
            return "unknown"
        return "curriculum" if curriculum else "randomized"

    pooled: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    seeds: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    per_lag: dict[tuple[str, str, str, str], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    # Episode-weighted probe cost: [sum(mean_probe_steps * episodes), sum(mean_disp
    # * episodes), sum(episodes)]. Absent on oracle/belief slices -> stays zero.
    probe_cost: dict[tuple[str, str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    for path in sorted(args.slices.glob("*.summary.json")):
        summary = json.loads(path.read_text())
        backbone = variant_of(summary.get("checkpoint", ""))
        key = (summary["task"], backbone, summary["method"], summary["split"])
        seeds[key].add(summary["seed"])
        for cell in summary["per_contract"]:
            pooled[key][0] += cell["successes"]
            pooled[key][1] += cell["episodes"]
            lag = cell.get("contract_lag", -1)
            per_lag[key][lag][0] += cell["successes"]
            per_lag[key][lag][1] += cell["episodes"]
            episodes = cell["episodes"]
            probe_cost[key][0] += cell.get("mean_probe_steps", 0.0) * episodes
            probe_cost[key][1] += cell.get("mean_probe_displacement", 0.0) * episodes
            probe_cost[key][2] += episodes

    print(f"{'task':10} {'backbone':11} {'method':13} {'split':9} {'succ':>6} {'wilson95':>16} "
          f"{'n':>5} {'seeds':>5} {'probe_steps':>11} {'probe_disp':>10}  per-lag  |  frozen ref")
    print("-" * 140)
    for key in sorted(pooled):
        task, backbone, method, split = key
        succ, total = pooled[key]
        rate, lo, hi = wilson(succ, total)
        lags = "  ".join(
            f"lag{lag}={wilson(s, n)[0]:.3f}(n{n})"
            for lag, (s, n) in sorted(per_lag[key].items())
        )
        ref = FROZEN_REFERENCE.get((task, split), {}).get(method)
        ref_s = f"{method}={ref:.3f}" if ref is not None else "-"
        steps_sum, disp_sum, weight = probe_cost[key]
        mean_steps = steps_sum / weight if weight else 0.0
        mean_disp = disp_sum / weight if weight else 0.0
        print(f"{task:10} {backbone:11} {method:13} {split:9} {rate:.3f} [{lo:.3f},{hi:.3f}] "
              f"{total:5} {len(seeds[key]):5} {mean_steps:11.3f} {mean_disp:10.4f}  "
              f"{lags}  |  {ref_s}")


if __name__ == "__main__":
    main()
