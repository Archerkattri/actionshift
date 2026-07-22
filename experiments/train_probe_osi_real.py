"""Real-ManiSkill matched-budget training of the probe-augmented learned identifier.

This is the recurrent episode-length identifier (``RecurrentOsiRegressor`` +
``RunningLagFeatures``, unchanged) trained on PROBE-EXCITED transitions. The single
difference from ``train_recurrent_real.py`` is the excitation: every collected
rollout spends its first ``budget`` steps sending the SAME fixed basis-pulse
schedule the probe family uses (``adaptation.probes.fixed_probe_pulse``, amplitude
0.5), then the frozen policy passes through. So the identical machinery that failed
passively (perm ~0.35, sign ~chance) is retrained on the strong, structured
excitation a basis pulse provides -- isolating active probing as the single lever.

Collection spans both eval tasks (pick_cube + push_cube) with each task's own
contract-independent calibration, so one model transfers to both end-to-end
evaluations. Training contracts are hash-disjoint from every frozen evaluation
contract. The report is identification BY FIELD (perm/sign/scale/target/lag/gripper)
on probe-excited held-out windows, to be read directly against the passive
(perm 0.29, sign 0.55) and random-excitation (perm 0.52) baselines.

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/train_probe_osi_real.py \
    --output artifacts/adaptation/probe_osi
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from actionshift.adaptation.calibration import (
    ResponseCalibration,
    response_from_observations,
)
from actionshift.adaptation.probes import fixed_probe_pulse
from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiRegressor,
    RunningLagFeatures,
    recurrent_loss,
)
from actionshift.adaptation.training import decode_prediction, sample_training_contract
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.ppo_parity import PpoAgent, _make_environment
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

_STEPS_PER_CONTRACT = 45
_MIN_SAMPLES = 6
_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
_TASKS = ("pick_cube", "push_cube")
_CHECKPOINTS = {
    "pick_cube": Path(
        "third_party/maniskill/examples/baselines/ppo/runs/8131f330bd69aa6b/final_ckpt.pt"
    ),
    "push_cube": Path(
        "third_party/maniskill/examples/baselines/ppo/runs/a1df53155e11ee42/final_ckpt.pt"
    ),
}
_STEP_CHECKPOINTS = (5, 10, 20, 45)


def evaluation_hashes() -> frozenset[str]:
    contracts: list[ActionContract] = []
    for split in ("seen", "unseen_composition", "long_lag"):
        contracts.extend(representative_contracts(split))
    return frozenset(contract_hash(contract) for contract in contracts)


def collect_probe_sequences(
    task_contracts: list[tuple[str, ActionContract]],
    *,
    calibrations: dict[str, ResponseCalibration],
    num_envs: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[ActionContract], int]:
    """Probe-excited pass-through rollouts: first ``budget`` steps = basis pulses.

    Returns ``(raws, responses, labels, env_steps)`` with ``raws`` of shape
    ``(steps, sequences, 7)`` and one contract label per sequence (env column).
    """
    all_raws, all_responses, labels = [], [], []
    env_steps = 0
    for index, (task, contract) in enumerate(task_contracts):
        calibration = calibrations[task]
        checkpoint = _CHECKPOINTS[task]
        environment = _make_environment(task, "noadapt_nonidentity", num_envs, contract)
        try:
            observation, _ = environment.reset(seed=seed + index)
            observation_dimension = int(np.prod(environment.single_observation_space.shape))
            action_dimension = int(np.prod(environment.single_action_space.shape))
            agent = PpoAgent(observation_dimension, action_dimension).to(environment.device)
            payload = torch.load(
                checkpoint, map_location=environment.device, weights_only=True
            )
            agent.load_state_dict(payload)
            agent.eval()
            low = torch.as_tensor(
                environment.single_action_space.low,
                device=environment.device,
                dtype=observation.dtype,
            )
            high = torch.as_tensor(
                environment.single_action_space.high,
                device=environment.device,
                dtype=observation.dtype,
            )
            raws, responses = [], []
            previous = observation
            for step in range(_STEPS_PER_CONTRACT):
                with torch.no_grad():
                    if step < _PROBE_BUDGET:
                        step_index = torch.full((num_envs,), step, dtype=torch.long)
                        canonical = fixed_probe_pulse(
                            step_index, amplitude=_PROBE_AMPLITUDE
                        ).to(device=environment.device, dtype=observation.dtype)
                    else:
                        canonical = torch.clamp(
                            agent.deterministic_action(previous), low, high
                        )
                    observation, _, _, _, info = environment.step(canonical)
                if "final_info" in info:
                    raise RuntimeError(
                        "collection crossed an episode boundary; shorten the rollout"
                    )
                responses.append(
                    response_from_observations(calibration, previous, observation)
                    .detach()
                    .cpu()
                )
                raws.append(canonical.detach().cpu())
                previous = observation
                env_steps += num_envs
            all_raws.append(torch.stack(raws).to(torch.float32))
            all_responses.append(torch.stack(responses).to(torch.float32))
            labels.extend([contract] * num_envs)
        finally:
            environment.close()
    return (
        torch.cat(all_raws, dim=1),
        torch.cat(all_responses, dim=1),
        labels,
        env_steps,
    )


def feature_sequence(
    raws: torch.Tensor, responses: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Streaming running-feature sequence: (steps, sequences, 301)."""
    accumulator = RunningLagFeatures(
        raws.shape[1], device=device, min_samples=_MIN_SAMPLES
    )
    steps = raws.shape[0]
    return torch.stack(
        [
            accumulator.push(raws[t].to(device), responses[t].to(device))
            for t in range(steps)
        ]
    )


def train_recurrent(
    model: RecurrentOsiRegressor,
    features: torch.Tensor,
    labels: list[ActionContract],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    sequences = features.shape[1]
    losses: list[float] = []
    for _ in range(epochs):
        ordering = torch.randperm(sequences, generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, sequences, batch_size):
            columns = ordering[start : start + batch_size]
            predictions = model(features[:, columns])
            loss = recurrent_loss(predictions, [labels[int(c)] for c in columns])
            if not torch.isfinite(loss):
                raise FloatingPointError("recurrent loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / batches)
    return losses


def _field_metrics(
    estimates: list[ActionContract], labels: list[ActionContract]
) -> dict[str, float]:
    """Identification accuracy by field on one steps-observed slice."""
    sign = permutation = scale_hit = target = lag = gripper = 0
    log_errors: list[float] = []
    for estimate, contract in zip(estimates, labels, strict=True):
        sign += sum(a == b for a, b in zip(estimate.sign, contract.sign, strict=True))
        permutation += sum(
            a == b
            for a, b in zip(estimate.permutation, contract.permutation, strict=True)
        )
        for est_scale, true_scale in zip(estimate.scale, contract.scale, strict=True):
            log_error = abs(float(np.log(est_scale / true_scale)))
            log_errors.append(log_error)
            scale_hit += int(log_error < float(np.log(1.25)))
        target += int(estimate.target == contract.target)
        lag += int(estimate.lag == contract.lag)
        gripper += int(estimate.gripper_inverted == contract.gripper_inverted)
    count = len(labels)
    return {
        "permutation_accuracy": permutation / (6 * count),
        "sign_accuracy": sign / (6 * count),
        "scale_within_25pct": scale_hit / (6 * count),
        "scale_median_abs_log_error": float(np.median(log_errors)),
        "target_accuracy": target / count,
        "lag_accuracy": lag / count,
        "gripper_accuracy": gripper / count,
    }


def identification_by_field(
    model: RecurrentOsiRegressor,
    features: torch.Tensor,
    labels: list[ActionContract],
) -> dict[str, dict[str, float]]:
    """Held-out identification by field at each steps-observed checkpoint."""
    with torch.no_grad():
        predictions = model(features)
    curve: dict[str, dict[str, float]] = {}
    for observed in _STEP_CHECKPOINTS:
        step_index = min(observed, features.shape[0]) - 1
        estimates = [
            decode_prediction(
                {key: value[step_index, column] for key, value in predictions.items()}
            )
            for column in range(len(labels))
        ]
        curve[str(observed)] = _field_metrics(estimates, labels)
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-contracts", type=int, default=96)
    parser.add_argument("--held-out-contracts", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibrations = {
        task: ResponseCalibration.load(
            Path(f"artifacts/adaptation/calibration/{task}.json")
        )
        for task in _TASKS
    }
    excluded = evaluation_hashes()
    generator = torch.Generator().manual_seed(arguments.seed)
    train = [
        (_TASKS[i % len(_TASKS)],
         sample_training_contract(generator, excluded_hashes=excluded, max_lag=2))
        for i in range(arguments.train_contracts)
    ]
    held = [
        (_TASKS[i % len(_TASKS)],
         sample_training_contract(generator, excluded_hashes=excluded, max_lag=2))
        for i in range(arguments.held_out_contracts)
    ]

    train_raws, train_responses, train_labels, train_steps = collect_probe_sequences(
        train, calibrations=calibrations, num_envs=arguments.num_envs,
        seed=arguments.seed,
    )
    held_raws, held_responses, held_labels, held_steps = collect_probe_sequences(
        held, calibrations=calibrations, num_envs=arguments.num_envs,
        seed=arguments.seed + 10_000,
    )

    train_features = feature_sequence(train_raws, train_responses, device=device)
    held_features = feature_sequence(held_raws, held_responses, device=device)

    torch.manual_seed(arguments.seed)
    model = RecurrentOsiRegressor(hidden=64, gru_hidden=64).to(device)
    losses = train_recurrent(
        model, train_features, train_labels,
        epochs=arguments.epochs, batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate, seed=arguments.seed,
    )
    curve = identification_by_field(model, held_features, held_labels)
    checkpoint_path = arguments.output / "probe_osi_regressor.pt"
    torch.save(model.to("cpu").state_dict(), checkpoint_path)

    result = {
        "schema_version": "1.0",
        "steps_per_contract": _STEPS_PER_CONTRACT,
        "min_samples": _MIN_SAMPLES,
        "probe_budget": _PROBE_BUDGET,
        "probe_amplitude": _PROBE_AMPLITUDE,
        "tasks": list(_TASKS),
        "training": {
            "contracts": arguments.train_contracts,
            "env_steps": train_steps,
            "sequences": train_features.shape[1],
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "epochs": arguments.epochs,
        },
        "held_out": {
            "contracts": arguments.held_out_contracts,
            "env_steps": held_steps,
            "sequences": held_features.shape[1],
            "identification_by_field": curve,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "backbone_sha256": {
            task: sha256_file(path) for task, path in _CHECKPOINTS.items()
        },
        "calibration": {
            task: json.loads(calibration.to_json())
            for task, calibration in calibrations.items()
        },
        "seed": arguments.seed,
        "elapsed_seconds": time.time() - started,
    }
    (arguments.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: result[k] for k in ("training", "held_out")}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
