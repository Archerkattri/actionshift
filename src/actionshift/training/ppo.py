"""Compact clipped PPO update shared across ActionShift baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from actionshift.policies.actor_critic import ActorCritic


@dataclass(frozen=True, slots=True)
class PPOBatch:
    observations: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    returns: Tensor
    advantages: Tensor
    context: Tensor | None


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    epochs: int,
    minibatch_size: int,
    clip_ratio: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.0,
    max_gradient_norm: float = 0.5,
) -> dict[str, float]:
    sample_count = len(batch.observations)
    if epochs <= 0 or minibatch_size <= 0 or sample_count == 0:
        raise ValueError("epochs, minibatch size, and sample count must be positive")
    advantages = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std(unbiased=False) + 1e-8
    )
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    for _ in range(epochs):
        ordering = torch.randperm(sample_count, device=batch.observations.device)
        for start in range(0, sample_count, minibatch_size):
            indices = ordering[start : start + minibatch_size]
            context = None if batch.context is None else batch.context[indices]
            log_probability, entropy, value = model.evaluate_actions(
                batch.observations[indices], batch.actions[indices], context
            )
            ratio = torch.exp(log_probability - batch.old_log_probabilities[indices])
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[
                indices
            ]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.square(value - batch.returns[indices]).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy_mean
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy_mean.detach())
            updates += 1
    return {key: value / updates for key, value in totals.items()}
