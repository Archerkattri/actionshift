"""Run one adaptation-method evaluation slice against the frozen Gate 1 setup.

Evaluates a runnable adapter (exact_belief, or belief + fixed/random/entropy
probes) on the frozen per-split evaluation contracts of one task, writing
hash-addressed per-contract episode JSONL plus a summary. The declared finite
pool (the exact-belief privilege) is identity + all six frozen representative
contracts + two fixed distractors, matching the Stage 1/2 probe runs.

Usage:
  .venv/bin/python experiments/run_adaptation_slice.py \
    --task pick_cube --method exact_belief --split seen --seed 20260718 \
    --episodes 100 --output artifacts/adaptation/slices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from actionshift.adaptation.adapters import ContractAdapter, ExactBeliefAdapter
from actionshift.adaptation.dualabi_adapter import DualABIProbeAdapter
from actionshift.adaptation.hypotheses import ExactBeliefDriver
from actionshift.adaptation.maniskill import (
    AdaptationEpisode,
    evaluate_adapter,
    load_or_run_calibration,
)
from actionshift.adaptation.probes import ProbingBeliefAdapter
from actionshift.adaptation.response import ResponseModel
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.ppo_parity import identity_contract
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

_METHODS = ("exact_belief", "fixed_probes", "random_probes", "entropy_probes", "dualabi")
_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
# Calibrated on a 48-episode seed-20260718 pilot (task-regret trajectory decays
# ~2.7 -> 0.3 over the six-step budget) then frozen for the three-seed run. The
# probe-efficiency win is robust across 0.4 <= threshold <= 1.5; 1.5 gives the
# fewest probe steps while holding entropy-level success on all four cells.
_DUALABI_REGRET_THRESHOLD = 1.5


def frozen_checkpoint(task: str) -> Path:
    for line in Path("artifacts/sprint/gate1/jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job.get("task") == task:
            return Path(job["checkpoint"])
    raise LookupError(f"no frozen Gate 1 checkpoint recorded for task {task}")


def declared_pool() -> tuple[ActionContract, ...]:
    distractors = (
        ActionContract(permutation=(3, 1, 4, 0, 5, 2), sign=(-1, -1, 1, 1, -1, 1),
                       scale=(0.6, 1.25, 0.75, 1.5, 2.0, 0.5), target="delta",
                       frame="base", lag=0, gripper_inverted=False),
        ActionContract(permutation=(4, 2, 0, 1, 5, 3), sign=(1, -1, 1, 1, -1, -1),
                       scale=(1.25, 0.6, 2.0, 0.5, 1.5, 0.75), target="delta",
                       frame="base", lag=0, gripper_inverted=True),
    )
    representatives = tuple(
        contract
        for split in ("seen", "unseen_composition", "long_lag")
        for contract in representative_contracts(split)
    )
    return (identity_contract(), *representatives, *distractors)


def build_adapter(
    method: str, pool: tuple[ActionContract, ...], response: ResponseModel,
    *, num_envs: int, seed: int, regret_threshold: float = _DUALABI_REGRET_THRESHOLD,
) -> ContractAdapter:
    if method == "exact_belief":
        return ExactBeliefAdapter(
            pool, batch_size=num_envs, response=response, device="cuda"
        )
    driver = ExactBeliefDriver(
        pool, batch_size=num_envs, response=response, device="cuda"
    )
    if method == "dualabi":
        return DualABIProbeAdapter(
            driver,
            budget=_PROBE_BUDGET,
            amplitude=_PROBE_AMPLITUDE,
            regret_threshold=regret_threshold,
        )
    strategy = method.removesuffix("_probes")
    if strategy not in ("fixed", "random", "entropy"):
        raise ValueError(f"unknown method: {method}")
    return ProbingBeliefAdapter(
        driver, strategy=strategy, budget=_PROBE_BUDGET,  # type: ignore[arg-type]
        amplitude=_PROBE_AMPLITUDE, seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True, choices=_METHODS)
    parser.add_argument("--split", required=True,
                        choices=("seen", "unseen_composition", "long_lag"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--regret-threshold", type=float, default=_DUALABI_REGRET_THRESHOLD,
        help="DualABI early-stop task-regret threshold (ignored by other methods)",
    )
    parser.add_argument(
        "--calibration-version", choices=("v1", "v2"), default="v1",
        help="v1: pose-only linear (reproduces prior results); v2: gripper channel "
        "+ magnitude-dependent gain evidence.",
    )
    parser.add_argument(
        "--rotation-mode", choices=("identity", "real"), default="identity",
        help="identity: decode/belief with identity ee-rotation (reproduces prior "
        "results); real: decode tool-frame contracts against the live tcp rotation "
        "and feed the belief replicas the observed rotation (the v2 variant).",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()

    version_two = arguments.calibration_version == "v2"
    checkpoint = frozen_checkpoint(arguments.task)
    calibration_name = f"{arguments.task}_v2" if version_two else arguments.task
    calibration = load_or_run_calibration(
        arguments.task,
        Path(f"artifacts/adaptation/calibration/{calibration_name}.json"),
        seed=20260720,
        calibrate_gripper=version_two,
        magnitude_gain=version_two,
    )
    pool = declared_pool()
    response = ResponseModel(
        alpha=calibration.alpha,
        sigma=calibration.sigma,
        alpha_c0=calibration.alpha_c0 if calibration.gain_model == "saturating" else None,
        gripper_alpha=calibration.gripper_alpha,
        gripper_sigma=calibration.gripper_sigma,
    )

    job = {
        "schema_version": "1.1",
        "task": arguments.task,
        "method": arguments.method,
        "split": arguments.split,
        "seed": arguments.seed,
        "episodes_per_contract": arguments.episodes,
        "num_envs": arguments.num_envs,
        "checkpoint_sha256": sha256_file(checkpoint),
        "pool_sha256": hashlib.sha256(
            "".join(contract_hash(c) for c in pool).encode()
        ).hexdigest(),
        "calibration_version": arguments.calibration_version,
        "calibration_sha256": hashlib.sha256(calibration.to_json().encode()).hexdigest(),
        "probe_budget": _PROBE_BUDGET if arguments.method != "exact_belief" else 0,
        "probe_amplitude": _PROBE_AMPLITUDE if arguments.method != "exact_belief" else 0.0,
    }
    if arguments.method == "dualabi":
        job["regret_threshold"] = arguments.regret_threshold
    # The rotation mode enters the job hash only for the v2 real-rotation variant,
    # so every pre-existing identity-mode artifact reproduces byte-for-byte.
    if arguments.rotation_mode != "identity":
        job["rotation_mode"] = arguments.rotation_mode
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    arguments.output.mkdir(parents=True, exist_ok=True)

    contracts = representative_contracts(arguments.split)
    all_records: list[AdaptationEpisode] = []
    per_contract = []
    for contract_index, contract in enumerate(contracts):
        adapter = build_adapter(
            arguments.method, pool, response,
            num_envs=arguments.num_envs, seed=arguments.seed,
            regret_threshold=arguments.regret_threshold,
        )
        records = evaluate_adapter(
            checkpoint,
            task=arguments.task,
            method=arguments.method,
            adapter=adapter,
            contract=contract,
            calibration=calibration,
            seed=arguments.seed + contract_index,
            episodes=arguments.episodes,
            num_envs=arguments.num_envs,
            rotation_mode=arguments.rotation_mode,
        )
        all_records.extend(records)
        successes = sum(1 for record in records if record.success)
        per_contract.append(
            {
                "contract_index": contract_index,
                "contract_sha256": contract_hash(contract),
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
    }
    (arguments.output / f"{job_id}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
