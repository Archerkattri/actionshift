"""Vectorized rollout return and advantage calculations."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    done: Tensor,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    if rewards.shape != done.shape or values.shape != (len(rewards) + 1, *rewards.shape[1:]):
        raise ValueError("values require one bootstrap row and rewards/done must match")
    if done.dtype != torch.bool:
        raise ValueError("done masks must be boolean")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in (discount, gae_lambda)):
        raise ValueError("discount and GAE lambda must be finite values in [0, 1]")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in range(len(rewards) - 1, -1, -1):
        continuation = (~done[step]).to(dtype=rewards.dtype)
        delta = rewards[step] + discount * values[step + 1] * continuation - values[step]
        running = delta + discount * gae_lambda * continuation * running
        advantages[step] = running
    return advantages, advantages + values[:-1]
