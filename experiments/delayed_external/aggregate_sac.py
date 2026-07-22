"""Aggregate the delay-aware SAC battery into a comparison table + retention analysis.

Reads every ``summary.json`` under the battery output, groups by env / role, and
prints (and writes ``aggregate.json``) the 3-seed mean +/- 95% CI final return for
the delay-5 augmented runs, the undelayed ceiling, the naive collapse control, the
retention ratio, and the side-by-side against DCAC's Figure-6 approximate reads.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

# DCAC Figure-6 approximate visual reads (constant total delay 5), from
# reports/delayed_rl_external.md. Figure reads only -- NOT a published table.
DCAC_FIG6 = {
    "HalfCheetah-v4": {"dcac": 5200, "sac_aug": 2000, "rtac": 2100},
    "Walker2d-v4": {"dcac": 4000, "sac_aug": 2200, "rtac": 1600},
    "Ant-v4": {"dcac": 3200, "sac_aug": 600, "rtac": 700},
}
ENVS = ["HalfCheetah-v4", "Walker2d-v4", "Ant-v4"]


def mean_ci(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    # 95% CI half-width, normal approx (t would widen for n=3; noted in report)
    ci = 1.96 * sd / math.sqrt(n)
    return m, sd, ci


def load(output: Path) -> dict:
    rows: dict = {}
    for f in glob.glob(str(output / "**" / "summary.json"), recursive=True):
        s = json.loads(Path(f).read_text())
        rows[Path(f).parent.name] = s
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/delayed_external_sac"))
    args = p.parse_args()
    output = args.output.resolve()
    rows = list(load(output).values())

    def pick(env: str, od: int, ad: int, aug: int) -> list[dict]:
        return [
            r for r in rows
            if r["env_id"] == env and r["obs_delay"] == od
            and r["act_delay"] == ad and r["augment"] == aug
        ]

    out: dict = {"envs": {}}
    print(f"{'Env':<16} {'delay5-aug (mean±CI)':<24} {'undelayed':<11} "
          f"{'naive':<9} {'retention':<10} {'DCAC':>6} {'SAC-aug':>8} {'RTAC':>6}")
    for env in ENVS:
        aug = pick(env, 2, 3, 1)
        und = pick(env, 0, 0, 1)
        naive = pick(env, 2, 3, 0)
        aug_finals = sorted(r["final_mean_return_last100"] for r in aug)
        m, sd, ci = mean_ci(aug_finals) if aug_finals else (float("nan"),) * 3
        und_val = und[0]["final_mean_return_last100"] if und else float("nan")
        naive_val = naive[0]["final_mean_return_last100"] if naive else float("nan")
        valid_und = und_val and not math.isnan(und_val) and und_val != 0
        retention = (m / und_val) if valid_und else float("nan")
        d = DCAC_FIG6[env]
        out["envs"][env] = {
            "delay5_aug_seeds": aug_finals,
            "delay5_aug_mean": m, "delay5_aug_sd": sd, "delay5_aug_ci95": ci,
            "undelayed": und_val, "naive_delay5": naive_val,
            "retention_vs_undelayed": retention,
            "dcac_fig6": d,
            "n_aug_seeds": len(aug_finals),
        }
        print(f"{env:<16} {m:>8.1f} ± {ci:<11.1f} {und_val:>10.1f} "
              f"{naive_val:>8.1f} {retention*100:>8.1f}% {d['dcac']:>6} "
              f"{d['sac_aug']:>8} {d['rtac']:>6}")

    (output / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {output / 'aggregate.json'}")


if __name__ == "__main__":
    main()
