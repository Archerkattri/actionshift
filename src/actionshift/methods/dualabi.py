"""Task-regret-aware action selection for DualABI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CandidateEstimates:
    task_value: Tensor
    terminal_risk: Tensor
    predicted_displacement: Tensor
    collision_risk: Tensor
    dynamics_disagreement: Tensor


class CandidateEvaluator(nn.Module):
    """Learned critic plus dynamics ensemble for auditable candidate scoring."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        posterior_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, posterior_dim, ensemble_size, hidden_dim) <= 0:
            raise ValueError("candidate evaluator dimensions must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.posterior_dim = posterior_dim
        input_dim = state_dim + action_dim + posterior_dim
        self.dynamics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 4),
                )
                for _ in range(ensemble_size)
            ]
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, state: Tensor, candidates: Tensor, posterior_summary: Tensor
    ) -> CandidateEstimates:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("state must be batch by state dimension")
        batch = state.shape[0]
        if candidates.ndim != 3 or candidates.shape[0] != batch:
            raise ValueError("candidates must align with the state batch")
        if candidates.shape[-1] != self.action_dim:
            raise ValueError("candidate action dimension mismatch")
        if posterior_summary.shape != (batch, self.posterior_dim):
            raise ValueError("posterior summary shape mismatch")
        count = candidates.shape[1]
        features = torch.cat(
            (
                state[:, None].expand(-1, count, -1),
                candidates,
                posterior_summary[:, None].expand(-1, count, -1),
            ),
            dim=-1,
        )
        ensemble = torch.stack(
            [cast(Tensor, model(features)) for model in self.dynamics], dim=0
        )
        displacement = ensemble[..., :3]
        predicted_displacement = displacement.mean(dim=0)
        dynamics_disagreement = displacement.var(dim=0, unbiased=False).mean(dim=-1)
        collision_risk = torch.sigmoid(ensemble[..., 3]).mean(dim=0)
        critic_output = cast(Tensor, self.critic(features))
        task_value = critic_output[..., 0]
        terminal_risk = torch.sigmoid(critic_output[..., 1])
        return CandidateEstimates(
            task_value=task_value,
            terminal_risk=terminal_risk,
            predicted_displacement=predicted_displacement,
            collision_risk=collision_risk,
            dynamics_disagreement=dynamics_disagreement,
        )


def expected_task_regret(
    prior: Tensor,
    outcome_likelihood: Tensor,
    future_utility: Tensor,
) -> Tensor:
    """Expected post-observation Bayes regret for each candidate action.

    ``outcome_likelihood`` is candidate by contract by discrete observation.
    ``future_utility`` is contract by future action.
    """
    if prior.ndim != 1:
        raise ValueError("prior must have one probability per contract")
    if outcome_likelihood.ndim != 3:
        raise ValueError("outcome likelihood must be candidate by contract by observation")
    if future_utility.ndim != 2:
        raise ValueError("future utility must be contract by future action")
    if outcome_likelihood.shape[1] != len(prior) or future_utility.shape[0] != len(prior):
        raise ValueError("contract dimensions must match")
    if torch.any(prior < 0) or not torch.isclose(prior.sum(), prior.new_tensor(1.0)):
        raise ValueError("prior must be a normalized probability vector")
    likelihood_sums = outcome_likelihood.sum(dim=-1)
    if torch.any(outcome_likelihood < 0) or not torch.allclose(
        likelihood_sums, torch.ones_like(likelihood_sums)
    ):
        raise ValueError("each contract's outcome likelihood must be normalized")

    joint = outcome_likelihood * prior[None, :, None]
    oracle_by_contract = future_utility.max(dim=-1).values
    oracle_weighted = torch.einsum("aho,h->ao", joint, oracle_by_contract)
    bayes_action_values = torch.einsum("aho,hk->aok", joint, future_utility)
    bayes_weighted = bayes_action_values.max(dim=-1).values
    return (oracle_weighted - bayes_weighted).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class RegretAwareDecision:
    index: int
    score: Tensor
    task_progress: Tensor
    safety_penalty: Tensor
    future_regret_penalty: Tensor


InformationMode = Literal["task_regret", "entropy", "none"]


@dataclass(frozen=True, slots=True)
class DualABIDecision:
    """Selected action and auditable decomposition of its score."""

    index: int
    score: Tensor
    regret_reduction: Tensor
    logged_terms: dict[str, Tensor]


def select_dualabi_candidates(
    *,
    task_value: Tensor,
    safety_risk: Tensor,
    current_task_regret: Tensor,
    expected_future_regret: Tensor,
    valid_mask: Tensor,
    safety_weight: float,
    information_weight: float,
    information_mode: InformationMode,
    entropy_reduction: Tensor | None = None,
    terminal_risk: Tensor | None = None,
    terminal_weight: float = 0.0,
) -> DualABIDecision:
    """Score safe candidates using task utility and task-relevant information."""
    vectors = (task_value, safety_risk, expected_future_regret, valid_mask)
    if any(value.ndim != 1 or value.shape != task_value.shape for value in vectors):
        raise ValueError("candidate inputs must be equal-length vectors")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean")
    if not valid_mask.any():
        raise ValueError("at least one safe candidate is required")
    if current_task_regret.numel() != 1:
        raise ValueError("current_task_regret must be scalar")
    if any(
        not math.isfinite(weight) or weight < 0
        for weight in (safety_weight, information_weight, terminal_weight)
    ):
        raise ValueError("selector weights must be finite and nonnegative")
    if not all(torch.isfinite(value).all() for value in vectors[:3]):
        raise ValueError("candidate scores must be finite")

    regret_reduction = torch.clamp(
        current_task_regret.reshape(()) - expected_future_regret, min=0
    )
    if information_mode == "task_regret":
        raw_information = regret_reduction
    elif information_mode == "entropy":
        if entropy_reduction is None or entropy_reduction.shape != task_value.shape:
            raise ValueError("entropy mode requires one entropy reduction per candidate")
        if not torch.isfinite(entropy_reduction).all():
            raise ValueError("entropy reductions must be finite")
        raw_information = entropy_reduction
    elif information_mode == "none":
        raw_information = torch.zeros_like(task_value)
    else:
        raise ValueError(f"unknown information mode: {information_mode}")

    safety_penalty = safety_weight * safety_risk
    if terminal_risk is None:
        terminal_risk = torch.zeros_like(task_value)
    if terminal_risk.shape != task_value.shape or not torch.isfinite(terminal_risk).all():
        raise ValueError("terminal risk must be a finite candidate vector")
    terminal_penalty = terminal_weight * terminal_risk
    information_bonus = information_weight * raw_information
    score = task_value - safety_penalty - terminal_penalty + information_bonus
    score = score.masked_fill(~valid_mask, -torch.inf)
    return DualABIDecision(
        index=int(torch.argmax(score).item()),
        score=score,
        regret_reduction=regret_reduction,
        logged_terms={
            "task_value": task_value,
            "safety_penalty": safety_penalty,
            "terminal_penalty": terminal_penalty,
            "information_bonus": information_bonus,
            "valid": valid_mask,
        },
    )


def select_regret_aware_action(
    *,
    task_progress: Tensor,
    safety_risk: Tensor,
    expected_future_regret: Tensor,
    safety_weight: float,
    regret_weight: float,
) -> RegretAwareDecision:
    if not (
        task_progress.ndim == safety_risk.ndim == expected_future_regret.ndim == 1
        and task_progress.shape == safety_risk.shape == expected_future_regret.shape
    ):
        raise ValueError("candidate score components must be equal-length vectors")
    if any(not math.isfinite(weight) or weight < 0 for weight in (safety_weight, regret_weight)):
        raise ValueError("selector weights must be finite and nonnegative")
    safety_penalty = safety_weight * safety_risk
    regret_penalty = regret_weight * expected_future_regret
    score = task_progress - safety_penalty - regret_penalty
    return RegretAwareDecision(
        index=int(torch.argmax(score).item()),
        score=score,
        task_progress=task_progress,
        safety_penalty=safety_penalty,
        future_regret_penalty=regret_penalty,
    )
