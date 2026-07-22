"""Evaluate the probe-augmented learned identifier against the frozen Gate 1 setup.

Mirrors ``experiments/run_factorized_slice.py`` exactly (frozen Gate 1 checkpoints,
shared contract-independent calibration, frozen per-split evaluation contracts,
identical episode accounting, hash-addressed outputs) but swaps the belief adapter
for the trained ``ProbeOsiAdapter``. There is NO pool and NO grammar knowledge: the
adapter regresses the continuous contract parameters from probe-excited evidence.
Its only privileges are the bounded probe budget (shared with the probe family) and
the shared response calibration; the trained model saw only hash-disjoint contracts.
The adapter never receives the true contract.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/run_probe_osi_slice.py \
    --task pick_cube --split seen --seed 20260718 --episodes 100 \
    --checkpoint artifacts/adaptation/probe_osi/probe_osi_regressor.pt \
    --output artifacts/adaptation/probe_osi_slices
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
from actionshift.adaptation.probe_osi import ProbeOsiAdapter
from actionshift.adaptation.recurrent_adapter import RecurrentOsiRegressor
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.contracts.splits import contract_hash
from actionshift.evaluation.provenance import sha256_file

_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
_WARMUP = 8


def frozen_checkpoint(task: str) -> Path:
    for line in Path("artifacts/sprint/gate1/jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job.get("task") == task:
            return Path(job["checkpoint"])
    raise LookupError(f"no frozen Gate 1 checkpoint recorded for task {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", required=True,
                        choices=("seen", "unseen_composition", "long_lag"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()

    backbone = frozen_checkpoint(arguments.task)
    calibration = load_or_run_calibration(
        arguments.task,
        Path(f"artifacts/adaptation/calibration/{arguments.task}.json"),
        seed=20260720,
    )
    model = RecurrentOsiRegressor(hidden=64, gru_hidden=64)
    model.load_state_dict(
        torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    job = {
        "schema_version": "1.0",
        "task": arguments.task,
        "method": "probe_osi",
        "split": arguments.split,
        "seed": arguments.seed,
        "episodes_per_contract": arguments.episodes,
        "num_envs": arguments.num_envs,
        "checkpoint_sha256": sha256_file(backbone),
        "model_sha256": sha256_file(arguments.checkpoint),
        "probe_budget": _PROBE_BUDGET,
        "probe_amplitude": _PROBE_AMPLITUDE,
        "warmup": _WARMUP,
    }
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    arguments.output.mkdir(parents=True, exist_ok=True)

    contracts = representative_contracts(arguments.split)
    all_records: list[AdaptationEpisode] = []
    per_contract = []
    for contract_index, contract in enumerate(contracts):
        adapter = ProbeOsiAdapter(
            model,
            batch_size=arguments.num_envs,
            budget=_PROBE_BUDGET,
            amplitude=_PROBE_AMPLITUDE,
            warmup=_WARMUP,
            device="cpu",
        )
        records = evaluate_adapter(
            backbone,
            task=arguments.task,
            method="probe_osi",
            adapter=adapter,
            contract=contract,
            calibration=calibration,
            seed=arguments.seed + contract_index,
            episodes=arguments.episodes,
            num_envs=arguments.num_envs,
        )
        all_records.extend(records)
        successes = sum(1 for record in records if record.success)
        per_contract.append(
            {
                "contract_index": contract_index,
                "contract_sha256": contract_hash(contract),
                "contract": json.loads(contract.to_json()),
                "episodes": len(records),
                "successes": successes,
                "success_rate": successes / len(records),
                "mean_probe_steps": sum(r.probe_steps for r in records) / len(records),
                "mean_probe_displacement": sum(
                    r.probe_displacement for r in records
                ) / len(records),
            }
        )

    episodes_path = arguments.output / f"{job_id}.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in all_records),
        encoding="utf-8",
    )
    summary = {
        **job,
        "job_id": job_id,
        "per_contract": per_contract,
        "overall_success_rate": sum(1 for r in all_records if r.success) / len(all_records),
        "elapsed_seconds": time.time() - started,
        "mean_seconds_per_step": (time.time() - started)
        / max(sum(r.episode_steps for r in all_records), 1),
    }
    (arguments.output / f"{job_id}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
