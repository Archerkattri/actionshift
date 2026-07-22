"""Real-ManiSkill matched-budget training of the recurrent episode-length adapter.

Collects FULL-EPISODE (45-step) ``[raw, response]`` sequences from the frozen
Gate 0 PPO backbone acting pass-through under sampled hidden training contracts on
real PickCube GPU simulation, with policy and random excitation mixed across the
parallel environments. Trains the recurrent identifier with deep supervision (the
OSI loss at every timestep so early-step estimates also train), then reports the
key curve: held-out-contract identification accuracy as a function of
steps-observed (5/15/30/45). Training contracts are hash-disjoint from every
frozen evaluation contract by construction, matching the OSI run's privilege and
env-step budget.

Usage: CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/train_recurrent_real.py \
    --output artifacts/adaptation/recurrent
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from actionshift.adaptation.calibration import ResponseCalibration, response_from_observations
from actionshift.adaptation.maniskill import evaluate_adapter, summarize
from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiAdapter,
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
_CHECKPOINT = Path(
    "third_party/maniskill/examples/baselines/ppo/runs/8131f330bd69aa6b/final_ckpt.pt"
)
_CALIBRATION = Path("artifacts/adaptation/calibration/pick_cube.json")
_STEP_CHECKPOINTS = (5, 15, 30, 45)


def evaluation_hashes() -> frozenset[str]:
    contracts: list[ActionContract] = []
    for split in ("seen", "unseen_composition", "long_lag"):
        contracts.extend(representative_contracts(split))
    return frozenset(contract_hash(contract) for contract in contracts)


def collect_real_sequences(
    contracts: list[ActionContract],
    *,
    calibration: ResponseCalibration,
    num_envs: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[ActionContract], int]:
    """Full-episode pass-through rollouts with mixed excitation across envs.

    Half of each contract's parallel environments follow the frozen policy's
    deterministic action; the other half take bounded random excitation, so both
    the weak-excitation and strong-excitation regimes appear under every contract.
    Returns ``(raws, responses, labels, env_steps)`` where ``raws`` is
    ``(steps, sequences, 7)`` and one label is recorded per sequence.
    """
    all_raws, all_responses, labels = [], [], []
    env_steps = 0
    half = num_envs // 2
    random_mask = torch.zeros(num_envs, dtype=torch.bool)
    random_mask[half:] = True
    for index, contract in enumerate(contracts):
        environment = _make_environment(
            "pick_cube", "noadapt_nonidentity", num_envs, contract
        )
        try:
            observation, _ = environment.reset(seed=seed + index)
            observation_dimension = int(np.prod(environment.single_observation_space.shape))
            action_dimension = int(np.prod(environment.single_action_space.shape))
            agent = PpoAgent(observation_dimension, action_dimension).to(environment.device)
            payload = torch.load(
                _CHECKPOINT, map_location=environment.device, weights_only=True
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
            span = (high - low).unsqueeze(0)
            excitation_generator = torch.Generator().manual_seed(seed + index)
            mask = random_mask.to(environment.device).unsqueeze(-1)
            raws, responses = [], []
            previous = observation
            for _ in range(_STEPS_PER_CONTRACT):
                with torch.no_grad():
                    policy_action = torch.clamp(
                        agent.deterministic_action(previous), low, high
                    )
                    sample = torch.rand(
                        (num_envs, low.shape[-1]), generator=excitation_generator
                    ).to(device=low.device, dtype=low.dtype)
                    random_action = low.unsqueeze(0) + sample * span
                    canonical = torch.where(mask, random_action, policy_action)
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


def identification_vs_steps(
    model: RecurrentOsiRegressor,
    features: torch.Tensor,
    labels: list[ActionContract],
) -> dict[str, dict[str, float]]:
    """Held-out identification accuracy at each steps-observed checkpoint."""
    with torch.no_grad():
        predictions = model(features)
    curve: dict[str, dict[str, float]] = {}
    for observed in _STEP_CHECKPOINTS:
        step_index = min(observed, features.shape[0]) - 1
        sign = permutation = target = lag = 0
        for column, contract in enumerate(labels):
            single = {
                key: value[step_index, column] for key, value in predictions.items()
            }
            estimate = decode_prediction(single)
            sign += sum(
                a == b for a, b in zip(estimate.sign, contract.sign, strict=True)
            )
            permutation += sum(
                a == b
                for a, b in zip(estimate.permutation, contract.permutation, strict=True)
            )
            target += int(estimate.target == contract.target)
            lag += int(estimate.lag == contract.lag)
        count = len(labels)
        curve[str(observed)] = {
            "sign_accuracy": sign / (6 * count),
            "permutation_accuracy": permutation / (6 * count),
            "target_accuracy": target / count,
            "lag_accuracy": lag / count,
        }
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
    parser.add_argument("--eval-episodes", type=int, default=16)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibration = ResponseCalibration.load(_CALIBRATION)
    excluded = evaluation_hashes()
    generator = torch.Generator().manual_seed(arguments.seed)
    train_contracts = [
        sample_training_contract(generator, excluded_hashes=excluded, max_lag=2)
        for _ in range(arguments.train_contracts)
    ]
    held_contracts = [
        sample_training_contract(generator, excluded_hashes=excluded, max_lag=2)
        for _ in range(arguments.held_out_contracts)
    ]

    train_raws, train_responses, train_labels, train_steps = collect_real_sequences(
        train_contracts,
        calibration=calibration,
        num_envs=arguments.num_envs,
        seed=arguments.seed,
    )
    held_raws, held_responses, held_labels, held_steps = collect_real_sequences(
        held_contracts,
        calibration=calibration,
        num_envs=arguments.num_envs,
        seed=arguments.seed + 10_000,
    )

    train_features = feature_sequence(train_raws, train_responses, device=device)
    held_features = feature_sequence(held_raws, held_responses, device=device)

    torch.manual_seed(arguments.seed)
    model = RecurrentOsiRegressor(hidden=64, gru_hidden=64).to(device)
    losses = train_recurrent(
        model,
        train_features,
        train_labels,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
    curve = identification_vs_steps(model, held_features, held_labels)
    checkpoint_path = arguments.output / "recurrent_regressor.pt"
    torch.save(model.state_dict(), checkpoint_path)

    evaluations = {}
    eval_contracts = {
        "seen_1": representative_contracts("seen")[1],
        "unseen_0": representative_contracts("unseen_composition")[0],
        "unseen_1": representative_contracts("unseen_composition")[1],
    }
    cpu_model = RecurrentOsiRegressor(hidden=64, gru_hidden=64)
    cpu_model.load_state_dict(model.to("cpu").state_dict())
    for name, contract in eval_contracts.items():
        if arguments.eval_episodes <= 0:
            break
        adapter = RecurrentOsiAdapter(
            cpu_model, batch_size=arguments.num_envs, device="cpu"
        )
        records = evaluate_adapter(
            _CHECKPOINT,
            task="pick_cube",
            method="recurrent",
            adapter=adapter,
            contract=contract,
            calibration=calibration,
            seed=arguments.seed,
            episodes=arguments.eval_episodes,
            num_envs=arguments.num_envs,
        )
        evaluations[name] = summarize(records)

    result = {
        "schema_version": "1.0",
        "steps_per_contract": _STEPS_PER_CONTRACT,
        "min_samples": _MIN_SAMPLES,
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
            "identification_vs_steps": curve,
        },
        "evaluation_probe": evaluations,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "backbone_sha256": sha256_file(_CHECKPOINT),
        "calibration": json.loads(calibration.to_json()),
        "seed": arguments.seed,
        "elapsed_seconds": time.time() - started,
    }
    (arguments.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: result[k] for k in ("training", "held_out", "evaluation_probe")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
