"""Evaluate the full-grammar factorized belief against the frozen Gate 1 setup.

Mirrors ``experiments/run_adaptation_slice.py`` exactly (same frozen checkpoints,
same contract-independent calibration, same per-split frozen evaluation contracts,
same episode accounting, hash-addressed outputs) but swaps the pool-privileged
belief for the unprivileged full-grammar factorized belief. There is NO declared
pool: the only privilege recorded is the finite grammar plus the shared response
calibration. The adapter never receives the true contract.

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/run_factorized_slice.py \
    --task pick_cube --method factorized_grammar --split seen --seed 20260718 \
    --episodes 100 --output artifacts/adaptation/factorized_slices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from actionshift.adaptation.adapters import ContractAdapter
from actionshift.adaptation.cpp_backend import CppCellScorer
from actionshift.adaptation.factorized_grammar import (
    GRAMMAR_LAGS,
    GRAMMAR_SCALES,
    GRAMMAR_SIGNS,
    GRAMMAR_TARGETS,
    CellScorer,
    FactorizedGrammarAdapter,
    FactorizedGrammarDriver,
    FactorizedGrammarProbingAdapter,
)
from actionshift.adaptation.hold_probe import (
    FactorizedGrammarHoldProbingAdapter,
    HoldProbeSchedule,
)
from actionshift.adaptation.maniskill import (
    AdaptationEpisode,
    evaluate_adapter,
    load_or_run_calibration,
)
from actionshift.adaptation.response import ResponseModel
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.contracts.splits import contract_hash
from actionshift.evaluation.provenance import sha256_file

_METHODS = (
    "factorized_grammar",
    "factorized_grammar_probes",
    "factorized_grammar_hold_probes",
)
_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
_HOLD_AMPLITUDE = 0.5


def frozen_checkpoint(task: str) -> Path:
    for line in Path("artifacts/sprint/gate1/jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job.get("task") == task:
            return Path(job["checkpoint"])
    raise LookupError(f"no frozen Gate 1 checkpoint recorded for task {task}")


def build_adapter(
    method: str,
    response: ResponseModel,
    *,
    num_envs: int,
    reanchor_period: int,
    scale_correction: bool = False,
    cell_scorer: CellScorer | None = None,
    hold_schedule: HoldProbeSchedule | None = None,
) -> ContractAdapter:
    if method == "factorized_grammar":
        return FactorizedGrammarAdapter(
            batch_size=num_envs, response=response, device="cuda",
            reanchor_period=reanchor_period, cell_scorer=cell_scorer,
            scale_correction=scale_correction,
        )
    if method == "factorized_grammar_probes":
        driver = FactorizedGrammarDriver(
            batch_size=num_envs, response=response, device="cuda",
            reanchor_period=reanchor_period, cell_scorer=cell_scorer,
            scale_correction=scale_correction,
        )
        return FactorizedGrammarProbingAdapter(
            driver, budget=_PROBE_BUDGET, amplitude=_PROBE_AMPLITUDE
        )
    if method == "factorized_grammar_hold_probes":
        driver = FactorizedGrammarDriver(
            batch_size=num_envs, response=response, device="cuda",
            reanchor_period=reanchor_period, cell_scorer=cell_scorer,
            scale_correction=scale_correction,
        )
        assert hold_schedule is not None
        return FactorizedGrammarHoldProbingAdapter(driver, schedule=hold_schedule)
    raise ValueError(f"unknown method: {method}")


def grammar_signature() -> str:
    payload = json.dumps(
        {
            "scales": list(GRAMMAR_SCALES),
            "signs": list(GRAMMAR_SIGNS),
            "targets": list(GRAMMAR_TARGETS),
            "lags": list(GRAMMAR_LAGS),
            "permutations": 720,
            "frame": "collapsed:base",
            "gripper": "unidentified:false",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


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
        "--calibration-version", choices=("v1", "v2"), default="v1",
        help="v1: pose-only linear (reproduces prior results); v2: gripper channel "
        "+ magnitude-dependent gain evidence.",
    )
    parser.add_argument(
        "--reanchor-period", type=int, default=0,
        help="periodic re-anchoring of the tracked absolute target from the observed "
        "tcp pose (0 = off); bounds absolute-target drift.",
    )
    parser.add_argument(
        "--scale-correction", action="store_true",
        help="wrap a drift-based closed-loop scale corrector around the MAP encode: "
        "refines the effective per-channel scale from the integrated ratio of "
        "observed tcp response to commanded intent, rescuing absolute-target control.",
    )
    parser.add_argument(
        "--backend", choices=("torch", "cpp"), default="torch",
        help="evidence-scoring backend: torch (default) or the ActionABI C++ "
        "identification core (cpp).",
    )
    parser.add_argument(
        "--hold-steps", type=int, default=2,
        help="hold-probe: consecutive steps each raw channel is held (>=2). Only "
        "used by the factorized_grammar_hold_probes method. Default 2 (one excursion "
        "+ one decay) is the identified best operating point: it identifies the "
        "absolute contract on the real coherent-hold response yet is short enough to "
        "leave control budget on the identifiable delta split (no regression).",
    )
    parser.add_argument(
        "--hold-rounds", type=int, default=1,
        help="hold-probe: how many times the 6-channel hold sweep repeats.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="recompute even if the hash-addressed summary already exists (the "
        "default skips completed cells, so the runner is resumable).",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()

    hold_schedule: HoldProbeSchedule | None = None
    if arguments.method == "factorized_grammar_hold_probes":
        hold_schedule = HoldProbeSchedule(
            amplitude=_HOLD_AMPLITUDE,
            hold_steps=arguments.hold_steps,
            rounds=arguments.hold_rounds,
        )

    version_two = arguments.calibration_version == "v2"
    checkpoint = frozen_checkpoint(arguments.task)
    calibration_name = (
        f"{arguments.task}_v2" if version_two else arguments.task
    )
    calibration = load_or_run_calibration(
        arguments.task,
        Path(f"artifacts/adaptation/calibration/{calibration_name}.json"),
        seed=20260720,
        calibrate_gripper=version_two,
        magnitude_gain=version_two,
    )
    response = ResponseModel(
        alpha=calibration.alpha,
        sigma=calibration.sigma,
        alpha_c0=calibration.alpha_c0 if calibration.gain_model == "saturating" else None,
        gripper_alpha=calibration.gripper_alpha,
        gripper_sigma=calibration.gripper_sigma,
    )

    hold_probed = arguments.method == "factorized_grammar_hold_probes"
    probed = arguments.method == "factorized_grammar_probes" or hold_probed
    if hold_probed:
        assert hold_schedule is not None
        probe_budget = hold_schedule.total_steps
    elif probed:
        probe_budget = _PROBE_BUDGET
    else:
        probe_budget = 0
    job = {
        "schema_version": "1.1",
        "task": arguments.task,
        "method": arguments.method,
        "split": arguments.split,
        "seed": arguments.seed,
        "episodes_per_contract": arguments.episodes,
        "num_envs": arguments.num_envs,
        "checkpoint_sha256": sha256_file(checkpoint),
        "grammar_sha256": grammar_signature(),
        "calibration_version": arguments.calibration_version,
        "calibration_sha256": hashlib.sha256(
            calibration.to_json().encode()
        ).hexdigest(),
        "reanchor_period": arguments.reanchor_period,
        "probe_budget": probe_budget,
        "probe_amplitude": _HOLD_AMPLITUDE if hold_probed else (
            _PROBE_AMPLITUDE if probed else 0.0
        ),
    }
    if hold_probed:
        # The hold schedule version enters the hash so a schedule change gets its own
        # hash-addressed slice; the new method already differs from prior job ids.
        assert hold_schedule is not None
        job["hold_schedule"] = hold_schedule.version
    if arguments.backend != "torch":
        # Keep torch-backend job ids identical to prior runs; only the C++ backend
        # perturbs the hash (it is a numerically-equivalent scoring backend).
        job["backend"] = arguments.backend
    if arguments.scale_correction:
        # Only perturb the job id when the corrector is on, so prior artifacts stay
        # reproducible; the corrector run gets its own hash-addressed slice.
        job["scale_correction"] = True
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.output / f"{job_id}.summary.json"
    if summary_path.is_file() and not arguments.force:
        # Resumable: a completed cell is hash-addressed, so re-invoking is a no-op.
        print(summary_path.read_text(encoding="utf-8"))
        return

    contracts = representative_contracts(arguments.split)
    all_records: list[AdaptationEpisode] = []
    per_contract = []
    for contract_index, contract in enumerate(contracts):
        cell_scorer = CppCellScorer() if arguments.backend == "cpp" else None
        adapter = build_adapter(
            arguments.method, response, num_envs=arguments.num_envs,
            reanchor_period=arguments.reanchor_period, cell_scorer=cell_scorer,
            scale_correction=arguments.scale_correction,
            hold_schedule=hold_schedule,
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
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
