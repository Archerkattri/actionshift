"""Evaluate a delay-aware backbone on one split, via oracle or exact-belief.

Mirrors ``experiments/run_adaptation_slice.py`` conventions (declared finite pool,
frozen calibration, hash-addressed per-contract JSONL + summary) but loads a
delay-aware augmented-state checkpoint instead of a frozen Gate 0 backbone, and
feeds the augmented observation via ``evaluate_delay_aware_adapter``.

Two methods:
  * ``oracle``       -- oracle-encode + delay-aware backbone. The agent knows the
                        true contract (privileged) and encodes with it; the env
                        applies the same contract including its lag. This is the
                        oracle path (0.027 Pick / 0.153 Push under lag with the
                        frozen backbone) rebuilt on the delay-aware backbone.
  * ``exact_belief`` -- exact finite-pool belief + delay-aware backbone. The
                        unprivileged-identification version (pool-privileged only).

Splits: ``long_lag`` (lag 2 and 4 contracts) is the headline; ``seen`` (lag 0) is
the instantaneous-competence check the delay-aware backbone must still pass.

HONESTY: the delay-aware backbone is a NEW policy. These numbers are NOT comparable
to the frozen-backbone tournament as a like-for-like method contest; the claim is
that the lag SPLIT is solvable with delay-aware training. The report states this.

Usage:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/run_delay_slice.py \
    --task pick_cube --method oracle --split long_lag --seed 20260718 \
    --checkpoint artifacts/delay_aware/pick_cube/<run_id>/final_ckpt.pt \
    --episodes 100 --output artifacts/adaptation/delay_slices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from actionshift.adaptation.adapters import ContractAdapter, ExactBeliefAdapter, OracleAdapter
from actionshift.adaptation.delay_aware import (
    DEFAULT_HISTORY,
    AdaptationEpisode,
    evaluate_delay_aware_adapter,
)
from actionshift.adaptation.dualabi_adapter import DualABIProbeAdapter
from actionshift.adaptation.hypotheses import ExactBeliefDriver
from actionshift.adaptation.maniskill import load_or_run_calibration
from actionshift.adaptation.probes import ProbingBeliefAdapter
from actionshift.adaptation.response import ResponseModel
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.ppo_parity import identity_contract
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

# Probe-family methods reuse the exact-belief driver + declared pool on the
# delay-aware backbone; constants match ``experiments/run_adaptation_slice.py`` so
# the probe machinery is matched-privilege to the frozen-backbone tournament.
_PROBE_METHODS = ("fixed_probes", "entropy_probes", "dualabi")
_METHODS = ("oracle", "exact_belief", *_PROBE_METHODS)
_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
_DUALABI_REGRET_THRESHOLD = 1.5


def declared_pool() -> tuple[ActionContract, ...]:
    """The exact-belief privilege: the same declared finite pool as the tournament.

    Replicated verbatim from ``experiments/run_adaptation_slice.declared_pool`` so
    the exact-belief adapter shares an identical pool with the frozen-backbone runs
    (identity + all six frozen representatives across the three splits + two fixed
    distractors). Kept inline to avoid a cross-script import that depends on how the
    launcher sets ``sys.path``.
    """
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
    method: str,
    contract: ActionContract,
    pool: tuple[ActionContract, ...],
    response: ResponseModel,
    *,
    num_envs: int,
    seed: int,
    regret_threshold: float = _DUALABI_REGRET_THRESHOLD,
) -> ContractAdapter:
    if method == "oracle":
        return OracleAdapter(contract, batch_size=num_envs)
    if method == "exact_belief":
        return ExactBeliefAdapter(pool, batch_size=num_envs, response=response, device="cuda")
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
    if strategy not in ("fixed", "entropy"):
        raise ValueError(f"unknown method: {method}")
    return ProbingBeliefAdapter(
        driver, strategy=strategy, budget=_PROBE_BUDGET,  # type: ignore[arg-type]
        amplitude=_PROBE_AMPLITUDE, seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True, choices=_METHODS)
    parser.add_argument("--split", required=True, choices=("seen", "long_lag"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--regret-threshold", type=float, default=_DUALABI_REGRET_THRESHOLD,
        help="DualABI early-stop task-regret threshold (ignored by other methods)",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    is_probe = arguments.method in _PROBE_METHODS
    started = time.time()

    checkpoint = arguments.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    calibration = load_or_run_calibration(
        arguments.task,
        Path(f"artifacts/adaptation/calibration/{arguments.task}.json"),
        seed=20260720,
    )
    pool = declared_pool()
    response = ResponseModel(alpha=calibration.alpha, sigma=calibration.sigma)

    job = {
        "schema_version": "1.0",
        "backbone": "delay_aware",
        "task": arguments.task,
        "method": arguments.method,
        "split": arguments.split,
        "seed": arguments.seed,
        "history": arguments.history,
        "episodes_per_contract": arguments.episodes,
        "num_envs": arguments.num_envs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "pool_sha256": hashlib.sha256(
            "".join(contract_hash(c) for c in pool).encode()
        ).hexdigest(),
    }
    if is_probe:
        # Probe-specific fields are added only for the probe family so the existing
        # oracle / exact-belief job hashes (and their slices) are never disturbed.
        job["probe_budget"] = _PROBE_BUDGET
        job["probe_amplitude"] = _PROBE_AMPLITUDE
        if arguments.method == "dualabi":
            job["regret_threshold"] = arguments.regret_threshold
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    arguments.output.mkdir(parents=True, exist_ok=True)

    contracts = representative_contracts(arguments.split)
    all_records: list[AdaptationEpisode] = []
    per_contract = []
    for contract_index, contract in enumerate(contracts):
        adapter = build_adapter(
            arguments.method, contract, pool, response,
            num_envs=arguments.num_envs, seed=arguments.seed,
            regret_threshold=arguments.regret_threshold,
        )
        records = evaluate_delay_aware_adapter(
            checkpoint,
            task=arguments.task,
            method=f"delay_aware_{arguments.method}",
            adapter=adapter,
            contract=contract,
            calibration=calibration,
            seed=arguments.seed + contract_index,
            history=arguments.history,
            episodes=arguments.episodes,
            num_envs=arguments.num_envs,
        )
        all_records.extend(records)
        successes = sum(1 for record in records if record.success)
        per_contract.append(
            {
                "contract_index": contract_index,
                "contract_sha256": contract_hash(contract),
                "contract_lag": contract.lag,
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
