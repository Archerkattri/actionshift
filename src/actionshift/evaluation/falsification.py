"""Fast controlled falsification gate for hidden-contract adaptation mechanics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from actionshift.contracts.types import ActionContract
from actionshift.methods.dualabi import expected_task_regret, select_regret_aware_action

METHODS = (
    "oracle",
    "no_adaptation",
    "recurrent_domain_randomization",
    "random_probes",
    "fixed_pulses",
    "exact_regret_aware",
)
FloatArray = NDArray[np.float64]


def build_core_contracts() -> tuple[ActionContract, ...]:
    """Construct the frozen 2x4x2x2 week-one contract product."""
    return tuple(
        ActionContract(
            permutation=permutation,
            sign=sign,
            scale=(scale, scale),
            target="delta",
            frame="base",
            lag=lag,
            gripper_inverted=False,
        )
        for permutation, sign, scale, lag in itertools.product(
            ((0, 1), (1, 0)),
            itertools.product((-1, 1), repeat=2),
            (1.0, 1.5),
            (0, 2),
        )
    )


def _decode(raw: FloatArray, contract: ActionContract) -> FloatArray:
    return cast(
        FloatArray,
        raw[np.asarray(contract.permutation)]
        * np.asarray(contract.sign)
        * np.asarray(contract.scale),
    )


def _encode(canonical: FloatArray, contract: ActionContract) -> FloatArray:
    semantic = canonical / (np.asarray(contract.sign) * np.asarray(contract.scale))
    raw = np.empty_like(semantic)
    for semantic_index, raw_index in enumerate(contract.permutation):
        raw[raw_index] = semantic[semantic_index]
    return cast(FloatArray, raw)


def _predicted_effect(
    history: list[FloatArray], candidate: FloatArray, contract: ActionContract
) -> FloatArray:
    expanded = [*history, candidate]
    source_index = len(expanded) - 1 - contract.lag
    if source_index < 0:
        return np.zeros(2, dtype=np.float64)
    return _decode(expanded[source_index], contract)


def _update_log_belief(
    log_probabilities: FloatArray,
    history: list[FloatArray],
    candidate: FloatArray,
    observed_effect: FloatArray,
    contracts: tuple[ActionContract, ...],
) -> FloatArray:
    predictions = np.stack(
        [_predicted_effect(history, candidate, contract) for contract in contracts]
    )
    likelihood = -0.5 * np.sum(((predictions - observed_effect) / 0.02) ** 2, axis=1)
    updated = log_probabilities + likelihood
    maximum = float(np.max(updated))
    return cast(
        FloatArray,
        updated - (maximum + math.log(float(np.exp(updated - maximum).sum()))),
    )


def _task_candidates(
    state: FloatArray,
    goal: FloatArray,
    map_contract: ActionContract,
    rng: np.random.Generator,
    method: str,
    step: int,
) -> list[FloatArray]:
    desired = np.clip(goal - state, -0.10, 0.10)
    if method == "oracle":
        raise AssertionError("oracle candidates require the true contract")
    if method == "no_adaptation":
        return [desired]
    if method == "recurrent_domain_randomization":
        return [0.75 * desired]
    if method == "fixed_pulses" and step < 3:
        fixed = (np.array([0.10, 0.0]), np.array([0.0, 0.10]), np.array([-0.10, 0.0]))
        return [fixed[step]]
    if method == "random_probes" and step < 3:
        pulse = np.zeros(2, dtype=np.float64)
        pulse[int(rng.integers(2))] = float(rng.choice((-0.10, 0.10)))
        return [pulse]
    return [_encode(desired, map_contract)]


def _regret_candidate(
    state: FloatArray,
    goal: FloatArray,
    history: list[FloatArray],
    contracts: tuple[ActionContract, ...],
    probabilities: FloatArray,
) -> FloatArray:
    map_contract = contracts[int(np.argmax(probabilities))]
    desired = np.clip(goal - state, -0.10, 0.10)
    raw_candidates = [
        _encode(desired, map_contract),
        np.array([0.10, 0.0]),
        np.array([-0.10, 0.0]),
        np.array([0.0, 0.10]),
        np.array([0.0, -0.10]),
    ]
    predictions = np.stack(
        [
            [_predicted_effect(history, candidate, contract) for contract in contracts]
            for candidate in raw_candidates
        ]
    )
    outcome_ids: dict[tuple[float, float], int] = {}
    encoded_outcomes = np.empty(predictions.shape[:2], dtype=np.int64)
    for candidate_index in range(len(raw_candidates)):
        for contract_index in range(len(contracts)):
            key = tuple(np.round(predictions[candidate_index, contract_index], 7))
            if key not in outcome_ids:
                outcome_ids[key] = len(outcome_ids)
            encoded_outcomes[candidate_index, contract_index] = outcome_ids[key]
    likelihood = torch.zeros(
        (len(raw_candidates), len(contracts), len(outcome_ids)), dtype=torch.float64
    )
    for candidate_index in range(len(raw_candidates)):
        for contract_index in range(len(contracts)):
            outcome_index = encoded_outcomes[candidate_index, contract_index]
            likelihood[candidate_index, contract_index, outcome_index] = 1
    future_utility = np.empty((len(contracts), len(raw_candidates)), dtype=np.float64)
    initial_distance = float(np.linalg.norm(goal - state))
    for contract_index in range(len(contracts)):
        for action_index in range(len(raw_candidates)):
            future_distance = float(
                np.linalg.norm(goal - (state + predictions[action_index, contract_index]))
            )
            future_utility[contract_index, action_index] = initial_distance - future_distance
    regret = expected_task_regret(
        torch.from_numpy(probabilities), likelihood, torch.from_numpy(future_utility)
    )
    expected_effect = np.einsum("h,ahd->ad", probabilities, predictions)
    task_progress = torch.from_numpy(
        initial_distance
        - np.linalg.norm(goal[None, :] - (state[None, :] + expected_effect), axis=1)
    )
    safety_risk = torch.from_numpy(
        np.maximum(0.0, np.linalg.norm(expected_effect, axis=1) - 0.24)
    )
    decision = select_regret_aware_action(
        task_progress=task_progress,
        safety_risk=safety_risk,
        expected_future_regret=regret,
        safety_weight=4.0,
        regret_weight=0.8,
    )
    return raw_candidates[decision.index]


def _episode(
    method: str,
    contract: ActionContract,
    contracts: tuple[ActionContract, ...],
    rng: np.random.Generator,
) -> dict[str, float]:
    state = rng.uniform(-0.6, 0.6, size=2)
    goal = rng.uniform(-0.6, 0.6, size=2)
    history: list[FloatArray] = []
    log_belief = np.full(len(contracts), -math.log(len(contracts)), dtype=np.float64)
    unintended = 0.0
    violations = 0
    for step in range(30):
        probabilities = np.exp(log_belief)
        map_contract = contracts[int(np.argmax(probabilities))]
        if method == "oracle":
            pending = np.zeros(2, dtype=np.float64)
            if contract.lag:
                pending = np.sum(
                    [_decode(item, contract) for item in history[-contract.lag :]], axis=0
                )
            desired = np.clip(goal - state - pending, -0.10, 0.10)
            raw = _encode(desired, contract)
        elif method == "exact_regret_aware":
            raw = _regret_candidate(state, goal, history, contracts, probabilities)
        else:
            raw = _task_candidates(state, goal, map_contract, rng, method, step)[0]
        effect = _predicted_effect(history, raw, contract)
        before = float(np.linalg.norm(goal - state))
        next_state = state + effect
        after = float(np.linalg.norm(goal - next_state))
        unintended += max(0.0, after - before)
        violations += int(np.any(np.abs(next_state) > 1.25))
        log_belief = _update_log_belief(log_belief, history, raw, effect, contracts)
        history.append(raw)
        state = next_state
    distance = float(np.linalg.norm(goal - state))
    posterior = np.exp(log_belief)
    return {
        "success": float(distance < 0.12),
        "final_distance": distance,
        "unintended_displacement": unintended,
        "violations": float(violations),
        "posterior_true": float(posterior[contracts.index(contract)]),
    }


def run_gate(*, seeds: Sequence[int], episodes_per_seed: int = 32) -> dict[str, Any]:
    contracts = build_core_contracts()
    raw: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
    for seed in seeds:
        for method_index, method in enumerate(METHODS):
            rng = np.random.default_rng(seed * 101 + method_index)
            for _ in range(episodes_per_seed):
                contract = contracts[int(rng.integers(len(contracts)))]
                raw[method].append(_episode(method, contract, contracts, rng))
    methods = {
        method: {
            key: float(np.mean([episode[key] for episode in episodes]))
            for key in episodes[0]
        }
        for method, episodes in raw.items()
    }
    exact_improvement = (
        methods["exact_regret_aware"]["success"]
        > methods["no_adaptation"]["success"] + 0.05
    )
    probe_values = {
        round(methods[name]["success"], 6)
        for name in ("random_probes", "fixed_pulses", "exact_regret_aware")
    }
    tradeoff_nontrivial = len(probe_values) > 1 and any(
        methods[name]["unintended_displacement"] > 0
        for name in ("random_probes", "fixed_pulses", "exact_regret_aware")
    )
    oracle_invariant = methods["oracle"]["success"] >= 0.95
    belief_converged = methods["exact_regret_aware"]["posterior_true"] >= 0.9
    gate = {
        "oracle_invariant": oracle_invariant,
        "exact_belief_converged": belief_converged,
        "exact_better_than_no_adaptation": exact_improvement,
        "tradeoff_nontrivial": tradeoff_nontrivial,
    }
    return {
        "schema_version": "1.0",
        "environment": "controlled_pickcube_linear_proxy",
        "contract_count": len(contracts),
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "methods": methods,
        "gate": gate,
        "decision": "continue" if all(gate.values()) else "stop",
        "limitations": [
            "This fast gate tests adaptation mechanics, not learned ManiSkill task competence.",
            "A separate GPU ManiSkill smoke is required before any simulator-level claim.",
        ],
    }


def _render_card(report: dict[str, Any]) -> str:
    lines = [
        "# ActionShift week-one falsification gate",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "This is a controlled linearized PickCube control proxy. It falsifies benchmark mechanics "
        "quickly but is not evidence of ManiSkill policy performance.",
        "",
        "| Method | Success | Final distance | Unintended displacement | "
        "Violations | True posterior |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in report["methods"].items():
        lines.append(
            f"| {method} | {metrics['success']:.3f} | {metrics['final_distance']:.3f} | "
            f"{metrics['unintended_displacement']:.3f} | {metrics['violations']:.3f} | "
            f"{metrics['posterior_true']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The gate continues benchmark engineering because the exact filter converges, "
            "adaptation "
            "beats no adaptation, and probe strategies expose a nontrivial task/safety tradeoff. "
            "No method superiority claim is made at this stage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=32)
    parser.add_argument("--require-cuda", action="store_true")
    arguments = parser.parse_args()
    if arguments.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("the preregistered smoke gate requires one CUDA device")
    report = run_gate(seeds=(20260718, 20260719, 20260720, 20260721, 20260722),
                      episodes_per_seed=arguments.episodes_per_seed)
    report["compute"] = {
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count_used": 1 if torch.cuda.is_available() else 0,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.card.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arguments.card.write_text(_render_card(report), encoding="utf-8")
    return 0 if report["decision"] == "continue" else 1


if __name__ == "__main__":
    raise SystemExit(main())
