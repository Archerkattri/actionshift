#!/usr/bin/env python3
"""Run episode-level unwrapped/identity/oracle parity for a frozen PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from actionshift.benchmarking.ppo_parity import (
    ParityCondition,
    evaluate_ppo_checkpoint,
    write_parity_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--condition",
        choices=("unwrapped", "identity", "oracle_nonidentity", "noadapt_nonidentity"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    condition = cast(ParityCondition, arguments.condition)
    episodes = evaluate_ppo_checkpoint(
        arguments.checkpoint,
        task=arguments.task,
        condition=condition,
        seed=arguments.seed,
        episodes=arguments.episodes,
        num_envs=arguments.num_envs,
    )
    write_parity_episodes(arguments.output, episodes)
    print(arguments.output)


if __name__ == "__main__":
    main()
