"""Real PPO hidden-contract evaluation slice for runnable Gate 1 ceilings."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from actionshift.benchmarking.ppo_parity import (
    ParityCondition,
    evaluate_ppo_checkpoint,
    oracle_contract,
)
from actionshift.contracts.splits import SplitRule, contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

Gate1RunnableMethod = Literal["oracle", "no_adapt"]


def plan_gate1_slice_jobs(
    checkpoints: dict[str, Path],
    *,
    seeds: tuple[int, ...],
    output_directory: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for task, checkpoint in sorted(checkpoints.items()):
        checkpoint_hash = sha256_file(checkpoint)
        for seed in seeds:
            for method in ("oracle", "no_adapt"):
                for split in ("seen", "unseen_composition", "long_lag"):
                    scientific = {
                        "schema_version": "1.0",
                        "task": task,
                        "method": method,
                        "split": split,
                        "seed": seed,
                        "checkpoint_sha256": checkpoint_hash,
                        "episodes_per_contract": 100,
                        "contracts": 2,
                        "num_envs": 16,
                    }
                    canonical = json.dumps(
                        scientific, sort_keys=True, separators=(",", ":")
                    )
                    job_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                    jobs.append(
                        {
                            "job_id": job_id,
                            **scientific,
                            "checkpoint": str(checkpoint),
                            "output": str(
                                output_directory
                                / f"{task}-{method}-{split}-seed{seed}.jsonl"
                            ),
                        }
                    )
    return jobs


def condition_for_method(method: str) -> ParityCondition:
    if method == "oracle":
        return "oracle_nonidentity"
    if method == "no_adapt":
        return "noadapt_nonidentity"
    raise ValueError(f"method is not end-to-end runnable with a frozen PPO checkpoint: {method}")


def representative_contracts(split: str) -> tuple[ActionContract, ...]:
    """Return frozen 6-DoF representatives; these are evaluation, not training, splits."""
    if split == "seen":
        return (
            oracle_contract(),
            ActionContract(
                permutation=(5, 1, 2, 3, 4, 0),
                sign=(1, -1, 1, -1, 1, 1),
                scale=(1.5, 0.75, 1.0, 1.25, 0.5, 2.0),
                target="delta",
                frame="base",
                lag=0,
                gripper_inverted=False,
            ),
        )
    if split == "unseen_composition":
        return (
            ActionContract(
                permutation=(1, 0, 2, 4, 5, 3),
                sign=(-1, 1, -1, 1, -1, 1),
                scale=(0.5, 2.0, 1.5, 0.75, 1.25, 0.6),
                target="absolute",
                frame="tool",
                lag=0,
                gripper_inverted=True,
            ),
            ActionContract(
                permutation=(5, 4, 3, 2, 1, 0),
                sign=(1, -1, -1, 1, 1, -1),
                scale=(1.5, 1.5, 0.5, 2.0, 0.75, 1.25),
                target="absolute",
                frame="base",
                lag=0,
                gripper_inverted=True,
            ),
        )
    if split == "long_lag":
        base = oracle_contract()
        return (
            ActionContract(**{**asdict(base), "lag": 2}),
            ActionContract(**{**asdict(base), "lag": 4}),
        )
    raise ValueError(f"unsupported Gate 1 split: {split}")


def evaluate_gate1_job(
    checkpoint: Path,
    *,
    task: str,
    method: Gate1RunnableMethod,
    split: SplitRule,
    seed: int,
    episodes_per_contract: int,
    num_envs: int,
) -> list[dict[str, Any]]:
    condition = condition_for_method(method)
    records: list[dict[str, Any]] = []
    for contract_index, contract in enumerate(representative_contracts(split)):
        episodes = evaluate_ppo_checkpoint(
            checkpoint,
            task=task,
            condition=condition,
            seed=seed + contract_index,
            episodes=episodes_per_contract,
            num_envs=num_envs,
            contract=contract,
        )
        records.extend(
            {
                **asdict(episode),
                "schema_version": "1.0",
                "environment_seed": episode.seed,
                "seed": seed,
                "method": method,
                "split": split,
                "contract_index": contract_index,
                "contract_sha256": contract_hash(contract),
                "contract": json.loads(contract.to_json()),
            }
            for episode in episodes
        )
    return records


def write_gate1_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)
