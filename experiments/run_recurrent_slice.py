"""Evaluate the trained recurrent episode-length adapter end-to-end (unprivileged).

Mirrors ``run_adaptation_slice.py`` episode accounting exactly, but the method is
the trained ``RecurrentOsiAdapter`` (no declared pool, no true contract at
evaluation). Writes hash-addressed per-contract episode JSONL plus a summary with
Wilson intervals for every split's representative evaluation contracts.

Usage:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/run_recurrent_slice.py \
    --task pick_cube --split seen --seed 20260718 --episodes 100 \
    --model artifacts/adaptation/recurrent/recurrent_regressor.pt \
    --output artifacts/adaptation/recurrent_slices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from actionshift.adaptation.maniskill import (
    AdaptationEpisode,
    evaluate_adapter,
    load_or_run_calibration,
)
from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiAdapter,
    RecurrentOsiRegressor,
)
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.gates import wilson_interval
from actionshift.contracts.splits import contract_hash
from actionshift.evaluation.provenance import sha256_file


def frozen_checkpoint(task: str) -> Path:
    """Resolve the frozen Gate 1 PPO checkpoint recorded for one task."""
    for line in Path("artifacts/sprint/gate1/jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job.get("task") == task:
            return Path(job["checkpoint"])
    raise LookupError(f"no frozen Gate 1 checkpoint recorded for task {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--split", required=True,
        choices=("seen", "unseen_composition", "long_lag"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()

    checkpoint = frozen_checkpoint(arguments.task)
    calibration = load_or_run_calibration(
        arguments.task,
        Path(f"artifacts/adaptation/calibration/{arguments.task}.json"),
        seed=20260720,
    )
    model = RecurrentOsiRegressor(hidden=64, gru_hidden=64)
    model.load_state_dict(torch.load(arguments.model, map_location="cpu", weights_only=True))
    model.eval()

    job = {
        "schema_version": "1.0",
        "task": arguments.task,
        "method": "recurrent",
        "split": arguments.split,
        "seed": arguments.seed,
        "episodes_per_contract": arguments.episodes,
        "num_envs": arguments.num_envs,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_sha256": sha256_file(arguments.model),
    }
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    arguments.output.mkdir(parents=True, exist_ok=True)

    contracts = representative_contracts(arguments.split)
    all_records: list[AdaptationEpisode] = []
    per_contract = []
    for contract_index, contract in enumerate(contracts):
        adapter = RecurrentOsiAdapter(
            model, batch_size=arguments.num_envs, device="cpu"
        )
        records = evaluate_adapter(
            checkpoint,
            task=arguments.task,
            method="recurrent",
            adapter=adapter,
            contract=contract,
            calibration=calibration,
            seed=arguments.seed + contract_index,
            episodes=arguments.episodes,
            num_envs=arguments.num_envs,
        )
        all_records.extend(records)
        successes = sum(1 for record in records if record.success)
        interval = wilson_interval(successes, len(records))
        per_contract.append(
            {
                "contract_index": contract_index,
                "contract_sha256": contract_hash(contract),
                "episodes": len(records),
                "successes": successes,
                "success_rate": successes / len(records),
                "wilson_low": interval.lower,
                "wilson_high": interval.upper,
            }
        )

    episodes_path = arguments.output / f"{job_id}.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps(asdict(record), sort_keys=True) + "\n" for record in all_records
        ),
        encoding="utf-8",
    )
    total_success = sum(1 for r in all_records if r.success)
    overall = wilson_interval(total_success, len(all_records))
    summary = {
        **job,
        "job_id": job_id,
        "per_contract": per_contract,
        "overall_successes": total_success,
        "overall_episodes": len(all_records),
        "overall_success_rate": total_success / len(all_records),
        "overall_wilson_low": overall.lower,
        "overall_wilson_high": overall.upper,
        "elapsed_seconds": time.time() - started,
    }
    (arguments.output / f"{job_id}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
