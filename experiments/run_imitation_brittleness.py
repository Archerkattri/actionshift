"""Imitation-backbone brittleness + rescue evaluation on the ActionShift benchmark.

Swaps a frozen Diffusion Policy backbone (trained on the clean ``pd_ee_delta_pose``
interface) into the exact adapter machinery the PPO tournament uses. Because the
adapters consume canonical actions, this is a drop-in ``policy=DiffusionPolicyShim``
into ``evaluate_adapter`` — nothing about the belief family changes.

For each (task, method, split, seed) cell it evaluates the DP backbone under the
frozen hidden contracts and writes a hash-addressed episode JSONL + summary under
``artifacts/adaptation/imitation_slices/``. A top-level ``manifest.json`` records
the clean-interface competence gate and every completed cell; rerunning the
driver SKIPS finished cells (resumable) and only fills the gaps.

Methods:
  * ``no_adapt``   raw DP under the contract — the brittleness number;
  * ``oracle``     DP + true contract — the ceiling;
  * ``exact_belief`` / ``entropy_probes`` / ``dualabi`` — the belief-family
    rescue adapters, identical to the PPO tournament's runnable methods.

Usage (one task per GPU, run in parallel):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/run_imitation_brittleness.py \
    --task pick_cube --seeds 20260718
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from actionshift.adaptation.adapters import (
    ContractAdapter,
    ExactBeliefAdapter,
    NoAdaptAdapter,
    OracleAdapter,
)
from actionshift.adaptation.dp_policy import DiffusionPolicyConfig, DiffusionPolicyShim
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

_METHODS = ("no_adapt", "oracle", "exact_belief", "entropy_probes", "dualabi")
_SPLITS = ("seen", "unseen_composition")
_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
_DUALABI_REGRET_THRESHOLD = 1.5
# Imitation backbones imitate motion-planning demos at demo speed; the official DP
# baseline evaluates at max_episode_steps=100 (vs the task-default 50 the frozen PPO
# cells used). DP is scored at its competent native horizon; brittleness is the
# relative drop from clean at THIS horizon (documented in the report caveats).
_DP_MAX_EPISODE_STEPS = 100
_BACKBONE_DIRS = {
    "pick_cube": "artifacts/adaptation/imitation_backbones/pick_cube_dp",
    "push_cube": "artifacts/adaptation/imitation_backbones/push_cube_dp",
}


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


def load_backbone(task: str, device: str = "cuda") -> tuple[Path, DiffusionPolicyShim, str]:
    run_dir = Path(_BACKBONE_DIRS[task])
    checkpoint = run_dir / "final_ckpt.pt"
    config = DiffusionPolicyConfig.load(run_dir / "dp_config.json")
    shim = DiffusionPolicyShim(checkpoint, config, device=device)
    return checkpoint, shim, sha256_file(checkpoint)


def build_cell_adapter(
    method: str,
    contract: ActionContract,
    pool: tuple[ActionContract, ...],
    response: ResponseModel,
    *,
    num_envs: int,
    seed: int,
) -> ContractAdapter:
    if method == "no_adapt":
        return NoAdaptAdapter()
    if method == "oracle":
        return OracleAdapter(contract, batch_size=num_envs)
    if method == "exact_belief":
        return ExactBeliefAdapter(pool, batch_size=num_envs, response=response, device="cuda")
    driver = ExactBeliefDriver(pool, batch_size=num_envs, response=response, device="cuda")
    if method == "dualabi":
        return DualABIProbeAdapter(
            driver, budget=_PROBE_BUDGET, amplitude=_PROBE_AMPLITUDE,
            regret_threshold=_DUALABI_REGRET_THRESHOLD,
        )
    strategy = method.removesuffix("_probes")
    return ProbingBeliefAdapter(
        driver, strategy=strategy, budget=_PROBE_BUDGET,  # type: ignore[arg-type]
        amplitude=_PROBE_AMPLITUDE, seed=seed,
    )


def cell_job(task: str, method: str, split: str, seed: int, episodes: int,
             num_envs: int, backbone_sha: str, checkpoint: Path,
             calibration_sha: str, pool_sha: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "backbone": "diffusion_policy",
        "backbone_sha256": backbone_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
        "task": task,
        "method": method,
        "split": split,
        "seed": seed,
        "episodes_per_contract": episodes,
        "num_envs": num_envs,
        "max_episode_steps": _DP_MAX_EPISODE_STEPS,
        "pool_sha256": pool_sha,
        "calibration_sha256": calibration_sha,
        "probe_budget": _PROBE_BUDGET if method in ("entropy_probes", "dualabi") else 0,
        "probe_amplitude": _PROBE_AMPLITUDE if method in ("entropy_probes", "dualabi") else 0.0,
    }


def run_cell(task: str, method: str, split: str, seed: int, *, shim: DiffusionPolicyShim,
             checkpoint: Path, backbone_sha: str, calibration, response: ResponseModel,
             pool: tuple[ActionContract, ...], pool_sha: str, episodes: int, num_envs: int,
             output_dir: Path) -> dict[str, object]:
    calibration_sha = hashlib.sha256(calibration.to_json().encode()).hexdigest()
    job = cell_job(task, method, split, seed, episodes, num_envs, backbone_sha,
                   checkpoint, calibration_sha, pool_sha)
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    summary_path = output_dir / f"{job_id}.summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())

    started = time.time()
    contracts = representative_contracts(split)
    all_records: list[AdaptationEpisode] = []
    per_contract = []
    for contract_index, contract in enumerate(contracts):
        # Reset the receding-horizon backbone buffers between contracts.
        shim.reset_state()
        adapter = build_cell_adapter(method, contract, pool, response,
                                     num_envs=num_envs, seed=seed)
        records = evaluate_adapter(
            checkpoint, task=task, method=method, adapter=adapter, contract=contract,
            calibration=calibration, seed=seed + contract_index, episodes=episodes,
            num_envs=num_envs, policy=shim, max_episode_steps=_DP_MAX_EPISODE_STEPS,
        )
        all_records.extend(records)
        successes = sum(1 for record in records if record.success)
        per_contract.append({
            "contract_index": contract_index,
            "contract_sha256": contract_hash(contract),
            "episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
            "mean_probe_steps": sum(r.probe_steps for r in records) / len(records),
            "mean_probe_displacement": sum(r.probe_displacement for r in records) / len(records),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job_id}.jsonl").write_text(
        "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in all_records),
        encoding="utf-8",
    )
    summary = {
        **job, "job_id": job_id, "per_contract": per_contract,
        "overall_success_rate": sum(1 for r in all_records if r.success) / len(all_records),
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return summary


def competence_gate(task: str, *, shim: DiffusionPolicyShim, checkpoint: Path,
                    calibration, seed: int, episodes: int, num_envs: int) -> float:
    shim.reset_state()
    records = evaluate_adapter(
        checkpoint, task=task, method="no_adapt", adapter=NoAdaptAdapter(),
        contract=identity_contract(), calibration=calibration, seed=seed,
        episodes=episodes, num_envs=num_envs, policy=shim,
        max_episode_steps=_DP_MAX_EPISODE_STEPS,
    )
    return sum(1 for r in records if r.success) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=tuple(_BACKBONE_DIRS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260718])
    parser.add_argument("--methods", nargs="+", default=list(_METHODS))
    parser.add_argument("--splits", nargs="+", default=list(_SPLITS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--competence-episodes", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/adaptation/imitation_slices"))
    arguments = parser.parse_args()

    checkpoint, shim, backbone_sha = load_backbone(arguments.task)
    calibration = load_or_run_calibration(
        arguments.task, Path(f"artifacts/adaptation/calibration/{arguments.task}.json"),
    )
    response = ResponseModel(
        alpha=calibration.alpha, sigma=calibration.sigma,
        alpha_c0=calibration.alpha_c0 if calibration.gain_model == "saturating" else None,
        gripper_alpha=calibration.gripper_alpha, gripper_sigma=calibration.gripper_sigma,
    )
    pool = declared_pool()
    pool_sha = hashlib.sha256(
        "".join(contract_hash(c) for c in pool).encode()
    ).hexdigest()

    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output / f"manifest_{arguments.task}.json"
    manifest: dict[str, object] = {"task": arguments.task, "backbone_sha256": backbone_sha,
                                   "cells": {}}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("cells", {})

    if "competence_clean" not in manifest:
        gate = competence_gate(arguments.task, shim=shim, checkpoint=checkpoint,
                               calibration=calibration, seed=arguments.seeds[0],
                               episodes=arguments.competence_episodes,
                               num_envs=arguments.num_envs)
        manifest["competence_clean"] = gate
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                 encoding="utf-8")
        print(f"[gate] {arguments.task} clean identity success = {gate:.3f}", flush=True)

    cells = manifest["cells"]
    assert isinstance(cells, dict)
    for seed in arguments.seeds:
        for split in arguments.splits:
            for method in arguments.methods:
                key = f"{method}|{split}|{seed}"
                if key in cells:
                    print(f"[skip] {arguments.task} {key} -> "
                          f"{cells[key]['overall_success_rate']:.3f}", flush=True)
                    continue
                summary = run_cell(
                    arguments.task, method, split, seed, shim=shim, checkpoint=checkpoint,
                    backbone_sha=backbone_sha, calibration=calibration, response=response,
                    pool=pool, pool_sha=pool_sha, episodes=arguments.episodes,
                    num_envs=arguments.num_envs, output_dir=arguments.output,
                )
                cells[key] = {
                    "job_id": summary["job_id"],
                    "overall_success_rate": summary["overall_success_rate"],
                    "per_contract": [p["success_rate"] for p in summary["per_contract"]],
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                         encoding="utf-8")
                print(f"[cell] {arguments.task} {key} -> "
                      f"{summary['overall_success_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
