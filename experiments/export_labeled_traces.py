"""Export labeled ManiSkill trajectories for ActionABI supervised evaluation.

This is the "bridge run" exporter. It rolls out the frozen Gate 0 PPO backbone on
real PickCube GPU simulation under sampled *hidden* action-interface contracts whose
fields are known by construction, and writes each rollout as an ActionABI-compatible
canonical JSONL trajectory plus a ground-truth label file. ActionABI then infers the
contract from the trajectory alone; the label is never an inference input.

Response model (documented in the manifest, and central to why this works):

  ActionABI's ``score_cpu`` compares ``decode(raw_action)`` against the observed state
  transition. The physical observable here is the tcp-pose-delta response returned by
  ``actionshift.adaptation.calibration.response_from_observations`` (translation delta +
  relative rotation vector, 6 channels). The pd_ee_delta_pose controller gain is tiny
  (|alpha| ~ 0.02-0.04, from the frozen contract-independent calibration), so the raw
  response is ~30x smaller than the commanded action; feeding it directly would make
  ActionABI trivially minimize command norm. We therefore express the response in
  commanded units by dividing by the per-channel, contract-independent controller gain
  ``alpha`` (measured on the *unwrapped identity* environment - task knowledge, not
  contract knowledge, and already shared by every ActionShift adaptation method). We
  then integrate the normalized response into a pseudo-pose state (cumulative sum) so
  that ActionABI's delta observable (s[t+off]-s[t]) equals the per-step normalized
  response and its absolute observable (s[t+off]) telescopes to the commanded target -
  making the contract's delta/absolute target flag, lag, permutation, sign and scale
  identifiable from the same trace.

Scope / honest boundaries (also recorded in the manifest):
  * 6 pose channels only. The gripper channel is NOT observable from the tcp-pose
    response, so gripper_inverted is exported in the label but excluded from ActionABI
    identification (a declared structural limitation, scored as unsupported downstream).
  * The wrapper uses an identity end-effector rotation, so base and tool frames are
    observationally identical (frame is a degenerate equivalence class, not identifiable).
  * space is always cartesian; it is not varied.
  * scales are drawn from a finite grid so ActionABI's finite declared grammar can
    express them and supervised scale accuracy is well defined.

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/export_labeled_traces.py \
    --output artifacts/actionabi_bridge --contracts 32 --num-envs 16 --steps 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from actionshift.adaptation.calibration import (
    ResponseCalibration,
    response_from_observations,
)
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.ppo_parity import PpoAgent, _make_environment
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

_CHECKPOINT = Path(
    "third_party/maniskill/examples/baselines/ppo/runs/8131f330bd69aa6b/final_ckpt.pt"
)
_CALIBRATION = Path("artifacts/adaptation/calibration/pick_cube.json")

# Finite scale grid: ActionABI evaluates a finite declared grammar, so the exported
# contracts draw their per-channel scale from exactly this set. It is also the scale
# alphabet the ActionABI-side grammar declares, making scale accuracy well defined.
FINITE_SCALES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
# ActionABI's declared lag alphabet for this run.
LAG_CLASSES: tuple[int, ...] = (0, 1, 2)
_DT_NS = 20_000_000  # 0.02 s/step, strictly increasing; irrelevant to delta/absolute.


def evaluation_hashes() -> frozenset[str]:
    """Frozen Gate 1 evaluation contracts, excluded so this labeled set is disjoint."""
    contracts: list[ActionContract] = []
    for split in ("seen", "unseen_composition", "long_lag"):
        contracts.extend(representative_contracts(split))
    return frozenset(contract_hash(contract) for contract in contracts)


def sample_finite_contract(
    generator: torch.Generator, *, excluded_hashes: frozenset[str]
) -> ActionContract:
    """Sample a 6-DoF contract with scale on the finite grid, disjoint from eval.

    Mirrors ``actionshift.adaptation.training.sample_training_contract`` (same 6-DoF
    grammar and hash-rejection discipline) but restricts scale to ``FINITE_SCALES`` and
    lag to ``LAG_CLASSES`` so the sampled space matches ActionABI's declared finite
    grammar exactly.
    """
    scale_grid = torch.tensor(FINITE_SCALES)
    lag_grid = LAG_CLASSES
    for _ in range(256):
        permutation = tuple(torch.randperm(6, generator=generator).tolist())
        sign = tuple(
            int(s) for s in (torch.randint(0, 2, (6,), generator=generator) * 2 - 1)
        )
        scale_idx = torch.randint(0, len(FINITE_SCALES), (6,), generator=generator)
        scale = tuple(float(scale_grid[i]) for i in scale_idx)

        def _coin() -> bool:
            return bool(torch.randint(0, 2, (1,), generator=generator).item())

        lag = lag_grid[int(torch.randint(0, len(lag_grid), (1,), generator=generator))]
        contract = ActionContract(
            permutation=permutation,
            sign=sign,
            scale=scale,
            target="absolute" if _coin() else "delta",
            frame="tool" if _coin() else "base",
            lag=lag,
            gripper_inverted=_coin(),
        )
        if contract_hash(contract) not in excluded_hashes:
            return contract
    raise RuntimeError("could not sample a contract outside the excluded set")


def rollout_trace(
    contract: ActionContract,
    *,
    calibration: ResponseCalibration,
    alpha: torch.Tensor,
    excitation: str,
    num_envs: int,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the frozen PPO backbone under one hidden contract.

    Returns ``(raw_pose, normalized_response)`` each of shape ``(steps, num_envs, 6)``.
    ``excitation='policy'`` sends the frozen deterministic policy action pass-through;
    ``excitation='random'`` sends uniform random excitation in the action bounds. In
    both cases the wrapper decodes the sent action by the hidden contract, so both are
    honest, differently-exciting probes of the same latent contract.
    """
    environment = _make_environment("pick_cube", "noadapt_nonidentity", num_envs, contract)
    try:
        observation, _ = environment.reset(seed=seed)
        observation_dimension = int(np.prod(environment.single_observation_space.shape))
        action_dimension = int(np.prod(environment.single_action_space.shape))
        agent = PpoAgent(observation_dimension, action_dimension).to(environment.device)
        payload = torch.load(_CHECKPOINT, map_location=environment.device, weights_only=True)
        agent.load_state_dict(payload)
        agent.eval()
        low = torch.as_tensor(
            environment.single_action_space.low, device=environment.device, dtype=observation.dtype
        )
        high = torch.as_tensor(
            environment.single_action_space.high, device=environment.device, dtype=observation.dtype
        )
        alpha_device = alpha.to(device=environment.device, dtype=observation.dtype)
        excite = torch.Generator().manual_seed(seed)
        raws: list[torch.Tensor] = []
        responses: list[torch.Tensor] = []
        previous = observation
        for _ in range(steps):
            with torch.no_grad():
                if excitation == "policy":
                    sent = torch.clamp(agent.deterministic_action(previous), low, high)
                elif excitation == "random":
                    sample = torch.rand(
                        (num_envs, low.shape[-1]), generator=excite
                    ).to(device=low.device, dtype=low.dtype)
                    sent = low.unsqueeze(0) + sample * (high - low).unsqueeze(0)
                else:
                    raise ValueError(f"unknown excitation: {excitation}")
                observation, _, _, _, info = environment.step(sent)
            if "final_info" in info:
                raise RuntimeError(
                    "rollout crossed an episode boundary; reduce --steps below the horizon"
                )
            response = response_from_observations(calibration, previous, observation)
            responses.append((response / alpha_device).detach().cpu())
            raws.append(sent[:, :6].detach().cpu())
            previous = observation
        raw = torch.stack(raws).to(torch.float64).numpy()
        normalized = torch.stack(responses).to(torch.float64).numpy()
        return raw, normalized
    finally:
        environment.close()


def write_trace_jsonl(
    path: Path, raw: np.ndarray, normalized: np.ndarray
) -> tuple[str, int]:
    """Write one canonical ActionABI JSONL trajectory. Returns (sha256, episode_count).

    Each parallel environment becomes one contiguous episode. State is the cumulative
    sum of the per-step normalized response (a pseudo-pose track, ``s[0] = 0``); action
    is the 6 raw pose channels. Rows are written episode-by-episode (contiguous) with
    strictly increasing per-episode timestamps, as the C++ loader requires.
    """
    steps, num_envs, dimension = raw.shape
    state_columns = [
        "tcp_pos_x", "tcp_pos_y", "tcp_pos_z", "tcp_rot_x", "tcp_rot_y", "tcp_rot_z"
    ]
    state_units = ["normalized_command"] * dimension
    sample_lines: list[str] = []
    for env_index in range(num_envs):
        state = np.zeros((steps + 1, dimension), dtype=np.float64)
        state[1:] = np.cumsum(normalized[:, env_index, :], axis=0)
        # steps rows carrying (state[t], action[t]); a final row carries state[steps]
        # with a neutral zero action (never scored: row+offset exceeds the episode end).
        for t in range(steps):
            sample_lines.append(
                json.dumps(
                    {
                        "record_type": "sample",
                        "episode_id": int(env_index),
                        "t_ns": int(t) * _DT_NS,
                        "state": [float(v) for v in state[t]],
                        "action": [float(v) for v in raw[t, env_index]],
                    },
                    sort_keys=True,
                )
            )
        sample_lines.append(
            json.dumps(
                {
                    "record_type": "sample",
                    "episode_id": int(env_index),
                    "t_ns": int(steps) * _DT_NS,
                    "state": [float(v) for v in state[steps]],
                    "action": [0.0] * dimension,
                },
                sort_keys=True,
            )
        )
    body = "\n".join(sample_lines)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    metadata = {
        "record_type": "metadata",
        "schema_version": "1.0",
        "source_filename": path.name,
        "source_sha256": digest,
        "extraction_date": "2026-07-21",
        "state_columns": state_columns,
        "state_units": state_units,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, sort_keys=True) + "\n" + body + "\n", encoding="utf-8"
    )
    return digest, num_envs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/actionabi_bridge"))
    parser.add_argument("--contracts", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--excitations", nargs="+", default=["policy", "random"], choices=["policy", "random"]
    )
    arguments = parser.parse_args()
    started = time.time()
    traces_dir = arguments.output / "traces"
    labels_dir = arguments.output / "labels"
    traces_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    calibration = ResponseCalibration.load(_CALIBRATION)
    alpha = torch.tensor(calibration.alpha, dtype=torch.float32)
    backbone_sha = sha256_file(_CHECKPOINT)
    excluded = evaluation_hashes()
    generator = torch.Generator().manual_seed(arguments.seed)
    contracts = [
        sample_finite_contract(generator, excluded_hashes=excluded)
        for _ in range(arguments.contracts)
    ]

    trace_records: list[dict] = []
    for contract_index, contract in enumerate(contracts):
        c_hash = contract_hash(contract)
        for excitation in arguments.excitations:
            trace_id = f"c{contract_index:03d}_{excitation}"
            seed = arguments.seed + 1000 * contract_index + (0 if excitation == "policy" else 500)
            raw, normalized = rollout_trace(
                contract,
                calibration=calibration,
                alpha=alpha,
                excitation=excitation,
                num_envs=arguments.num_envs,
                steps=arguments.steps,
                seed=seed,
            )
            trace_path = traces_dir / f"{trace_id}.jsonl"
            digest, episodes = write_trace_jsonl(trace_path, raw, normalized)
            label = {
                "trace_id": trace_id,
                "task": "pick_cube",
                "excitation": excitation,
                "seed": seed,
                "episodes": episodes,
                "steps": arguments.steps,
                "trace_sha256": digest,
                "contract_hash": c_hash,
                # Ground truth: the full latent contract, known by construction.
                "contract": {
                    "permutation": list(contract.permutation),
                    "sign": list(contract.sign),
                    "scale": list(contract.scale),
                    "target": contract.target,
                    "frame": contract.frame,
                    "lag": contract.lag,
                    "gripper_inverted": contract.gripper_inverted,
                },
            }
            (labels_dir / f"{trace_id}.json").write_text(
                json.dumps(label, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            trace_records.append(
                {
                    "trace_id": trace_id,
                    "excitation": excitation,
                    "contract_index": contract_index,
                    "contract_hash": c_hash,
                    "trace_sha256": digest,
                    "episodes": episodes,
                    "lag": contract.lag,
                    "target": contract.target,
                }
            )
            print(
                f"exported {trace_id} (lag={contract.lag} "
                f"target={contract.target} sha={digest[:12]})"
            )

    manifest = {
        "schema_version": "1.0",
        "generated": "2026-07-21",
        "task": "pick_cube",
        "backbone_checkpoint": str(_CHECKPOINT),
        "backbone_sha256": backbone_sha,
        "calibration_path": str(_CALIBRATION),
        "calibration": json.loads(calibration.to_json()),
        "num_contracts": arguments.contracts,
        "num_envs": arguments.num_envs,
        "steps": arguments.steps,
        "excitations": arguments.excitations,
        "seed": arguments.seed,
        "finite_scale_grid": list(FINITE_SCALES),
        "lag_alphabet": list(LAG_CLASSES),
        "response_model": {
            "observable": (
            "tcp-pose-delta response "
            "(translation delta + relative rotation vector, 6 channels)"
        ),
            "source_function": "actionshift.adaptation.calibration.response_from_observations",
            "normalization": "divide by per-channel contract-independent controller gain alpha",
            "alpha": list(calibration.alpha),
            "alpha_fit_r2": list(calibration.fit_r2),
            "state_construction": (
            "cumulative sum of normalized response (pseudo-pose track, s[0]=0)"
        ),
            "state_semantics": (
                "ActionABI delta observable = per-step normalized response; absolute"
                " observable telescopes to commanded target"
            ),
        },
        "scope_and_limitations": {
            "pose_channels": 6,
            "gripper": (
                "gripper_inverted is labeled but NOT observable from the pose response;"
                " excluded from ActionABI identification (structural limitation)"
            ),
            "frame": (
                "identity end-effector rotation makes base and tool frames"
                " observationally identical (degenerate equivalence class)"
            ),
            "space": "always cartesian; not varied",
            "dynamics": (
                "real ManiSkill GPU simulation, PickCube-v1, pd_ee_delta_pose,"
                " identity-rotation hidden-contract wrapper; NOT hardware"
            ),
        },
        "evaluation_contracts_excluded": sorted(excluded),
        "traces": trace_records,
        "elapsed_seconds": time.time() - started,
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\nexported {len(trace_records)} traces from {arguments.contracts} contracts "
        f"in {manifest['elapsed_seconds']:.1f}s -> {arguments.output}"
    )


if __name__ == "__main__":
    main()
