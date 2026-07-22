#!/usr/bin/env python3
"""Evaluate one real hidden-contract PPO Gate 1 slice."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from actionshift.benchmarking.gate1_eval import (
    Gate1RunnableMethod,
    evaluate_gate1_job,
    write_gate1_records,
)
from actionshift.contracts.splits import SplitRule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("pick_cube", "push_cube", "pull_cube", "stack_cube", "peg_insertion_side"),
        required=True,
    )
    parser.add_argument("--method", choices=("oracle", "no_adapt"), required=True)
    parser.add_argument(
        "--split", choices=("seen", "unseen_composition", "long_lag"), required=True
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--episodes-per-contract", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = evaluate_gate1_job(
        arguments.checkpoint,
        task=arguments.task,
        method=cast(Gate1RunnableMethod, arguments.method),
        split=cast(SplitRule, arguments.split),
        seed=arguments.seed,
        episodes_per_contract=arguments.episodes_per_contract,
        num_envs=arguments.num_envs,
    )
    write_gate1_records(arguments.output, records)
    print(arguments.output)


if __name__ == "__main__":
    main()
