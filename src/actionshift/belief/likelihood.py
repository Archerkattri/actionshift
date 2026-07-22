"""Observation likelihoods for exact and learned contract beliefs."""

from __future__ import annotations

import math

from torch import Tensor


def gaussian_transition_log_likelihood(
    predicted_observations: Tensor,
    observed_transition: Tensor,
    *,
    sigma: float,
) -> Tensor:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if predicted_observations.ndim != 2:
        raise ValueError("predictions must have contract and observation dimensions")
    if observed_transition.shape != predicted_observations.shape[1:]:
        raise ValueError("observed transition dimension must match predictions")
    standardized = (predicted_observations - observed_transition) / sigma
    return -0.5 * standardized.square().sum(dim=-1)
