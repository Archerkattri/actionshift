"""Real-ManiSkill matched-budget UP-OSI-style training run.

Collects (raw, response) histories from the frozen Gate 0 PPO backbone acting
pass-through (no-adapt) under sampled hidden training contracts on real PickCube
GPU simulation, trains the OSI regressor on those real responses, reports
held-out-contract identification, and evaluates the trained adapter on the
frozen Gate 1 evaluation contracts. Training contracts are hash-disjoint from
every frozen evaluation contract by construction.

Usage: .venv/bin/python experiments/train_osi_real.py --output artifacts/adaptation/osi
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
from actionshift.adaptation.training import (
    HistoryWindow,
    OsiAdapter,
    OsiRegressor,
    decode_prediction,
    sample_training_contract,
    train_osi,
)
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.ppo_parity import PpoAgent, _make_environment
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

_WINDOW = 14
_STEPS_PER_CONTRACT = 45
_WINDOW_STRIDE = 4
_CHECKPOINT = Path(
    "third_party/maniskill/examples/baselines/ppo/runs/8131f330bd69aa6b/final_ckpt.pt"
)
_CALIBRATION = Path("artifacts/adaptation/calibration/pick_cube.json")


def evaluation_hashes() -> frozenset[str]:
    contracts: list[ActionContract] = []
    for split in ("seen", "unseen_composition", "long_lag"):
        contracts.extend(representative_contracts(split))
    return frozenset(contract_hash(contract) for contract in contracts)


def collect_real_windows(
    contracts: list[ActionContract],
    *,
    calibration: ResponseCalibration,
    num_envs: int,
    seed: int,
    excitation: str = "policy",
) -> tuple[list[HistoryWindow], int]:
    """Pass-through frozen-PPO rollouts under each hidden contract; cut windows."""
    windows: list[HistoryWindow] = []
    env_steps = 0
    for index, contract in enumerate(contracts):
        environment = _make_environment("pick_cube", "noadapt_nonidentity", num_envs, contract)
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
            raws, responses = [], []
            previous = observation
            excitation_generator = torch.Generator().manual_seed(seed + index)
            for _ in range(_STEPS_PER_CONTRACT):
                with torch.no_grad():
                    if excitation == "policy":
                        canonical = torch.clamp(
                            agent.deterministic_action(previous), low, high
                        )
                    else:
                        span = (high - low).unsqueeze(0)
                        sample = torch.rand(
                            (num_envs, low.shape[-1]), generator=excitation_generator
                        ).to(device=low.device, dtype=low.dtype)
                        canonical = low.unsqueeze(0) + sample * span
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
            stacked_raw = torch.stack(raws)
            stacked_response = torch.stack(responses)
            for start in range(0, _STEPS_PER_CONTRACT - _WINDOW + 1, _WINDOW_STRIDE):
                for row in range(num_envs):
                    history = torch.cat(
                        (
                            stacked_raw[start : start + _WINDOW, row],
                            stacked_response[start : start + _WINDOW, row],
                        ),
                        dim=-1,
                    ).to(torch.float32)
                    windows.append(HistoryWindow(history=history, contract=contract))
        finally:
            environment.close()
    return windows, env_steps


def identification_metrics(
    model: OsiRegressor, samples: list[HistoryWindow]
) -> dict[str, float]:
    histories = torch.stack([sample.history for sample in samples])
    with torch.no_grad():
        predictions = model(histories)
    sign = permutation = target = lag = 0
    for index, sample in enumerate(samples):
        estimate = decode_prediction(
            {key: value[index] for key, value in predictions.items()}
        )
        sign += sum(a == b for a, b in zip(estimate.sign, sample.contract.sign, strict=True))
        permutation += sum(
            a == b
            for a, b in zip(estimate.permutation, sample.contract.permutation, strict=True)
        )
        target += int(estimate.target == sample.contract.target)
        lag += int(estimate.lag == sample.contract.lag)
    count = len(samples)
    return {
        "sign_accuracy": sign / (6 * count),
        "permutation_accuracy": permutation / (6 * count),
        "target_accuracy": target / count,
        "lag_accuracy": lag / count,
        "windows": float(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-contracts", type=int, default=96)
    parser.add_argument("--held-out-contracts", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--excitation", choices=("policy", "random"), default="policy")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

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

    train_windows, train_steps = collect_real_windows(
        train_contracts,
        calibration=calibration,
        num_envs=arguments.num_envs,
        seed=arguments.seed,
        excitation=arguments.excitation,
    )
    held_windows, held_steps = collect_real_windows(
        held_contracts,
        calibration=calibration,
        num_envs=arguments.num_envs,
        seed=arguments.seed + 10_000,
        excitation=arguments.excitation,
    )

    torch.manual_seed(arguments.seed)
    model = OsiRegressor(window=_WINDOW, hidden=64)
    losses = train_osi(
        model,
        train_windows,
        epochs=arguments.epochs,
        batch_size=256,
        seed=arguments.seed,
    )
    held_metrics = identification_metrics(model, held_windows)
    checkpoint_path = arguments.output / "osi_regressor.pt"
    torch.save(model.state_dict(), checkpoint_path)

    evaluations = {}
    eval_contracts = {
        "seen_1": representative_contracts("seen")[1],
        "unseen_0": representative_contracts("unseen_composition")[0],
        "unseen_1": representative_contracts("unseen_composition")[1],
    }
    for name, contract in eval_contracts.items():
        if arguments.eval_episodes <= 0:
            break
        adapter = OsiAdapter(model, batch_size=arguments.num_envs)
        records = evaluate_adapter(
            _CHECKPOINT,
            task="pick_cube",
            method="up_osi",
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
        "window": _WINDOW,
        "training": {
            "contracts": arguments.train_contracts,
            "env_steps": train_steps,
            "windows": len(train_windows),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "epochs": arguments.epochs,
        },
        "held_out": {"contracts": arguments.held_out_contracts, "env_steps": held_steps,
                     **held_metrics},
        "evaluation_probe": evaluations,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "backbone_sha256": sha256_file(_CHECKPOINT),
        "calibration": json.loads(calibration.to_json()),
        "excitation": arguments.excitation,
        "seed": arguments.seed,
        "elapsed_seconds": time.time() - started,
    }
    (arguments.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in ("training", "held_out", "evaluation_probe")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
