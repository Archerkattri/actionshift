"""Task, adaptation, safety, and belief metrics with confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from statistics import NormalDist
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    success: bool
    episode_steps: int
    recovery_steps: int | None
    unintended_displacement: float
    safety_violations: int
    cumulative_action_cost: float
    task_return: float
    posterior_true_probability: float
    posterior_entropy: float

    def __post_init__(self) -> None:
        if self.episode_steps <= 0 or self.safety_violations < 0:
            raise ValueError("step count must be positive and violation count nonnegative")
        if self.recovery_steps is not None and self.recovery_steps < 0:
            raise ValueError("recovery_steps must be nonnegative")
        numeric = (
            self.unintended_displacement,
            self.cumulative_action_cost,
            self.task_return,
            self.posterior_true_probability,
            self.posterior_entropy,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("episode metrics must be finite")
        if not 0 <= self.posterior_true_probability <= 1:
            raise ValueError("posterior probability must be in [0, 1]")


def _interval(values: list[float], confidence: float) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean())
    if len(values) == 1:
        half_width = 0.0
    else:
        standard_error = float(tensor.std(unbiased=True)) / math.sqrt(len(values))
        quantile = NormalDist().inv_cdf(0.5 + confidence / 2)
        half_width = quantile * standard_error
    return {"mean": mean, "lower": mean - half_width, "upper": mean + half_width}


def _wilson(successes: int, total: int, confidence: float) -> dict[str, float]:
    rate = successes / total
    quantile = NormalDist().inv_cdf(0.5 + confidence / 2)
    squared = quantile * quantile
    denominator = 1 + squared / total
    center = (rate + squared / (2 * total)) / denominator
    half_width = (
        quantile
        * math.sqrt(rate * (1 - rate) / total + squared / (4 * total * total))
        / denominator
    )
    return {"mean": rate, "lower": center - half_width, "upper": center + half_width}


def summarize(
    episodes: list[EpisodeMetrics], *, confidence: float = 0.95
) -> dict[str, Any]:
    """Aggregate primary and diagnostic metrics without silently dropping failures."""
    if not episodes:
        raise ValueError("at least one episode is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    # Revalidate reconstructed inputs in case callers bypassed frozen construction.
    for episode in episodes:
        for field in fields(episode):
            value = getattr(episode, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("episode metrics must be finite")

    recovery = [float(item.recovery_steps) for item in episodes if item.recovery_steps is not None]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "episode_count": len(episodes),
        "confidence": confidence,
        "success_rate": _wilson(sum(item.success for item in episodes), len(episodes), confidence),
        "episode_steps": _interval([float(item.episode_steps) for item in episodes], confidence),
        "unintended_displacement": _interval(
            [item.unintended_displacement for item in episodes], confidence
        ),
        "safety_violation_rate": _wilson(
            sum(item.safety_violations > 0 for item in episodes), len(episodes), confidence
        ),
        "safety_violations": _interval(
            [float(item.safety_violations) for item in episodes], confidence
        ),
        "cumulative_action_cost": _interval(
            [item.cumulative_action_cost for item in episodes], confidence
        ),
        "task_return": _interval([item.task_return for item in episodes], confidence),
        "posterior_true_probability": _interval(
            [item.posterior_true_probability for item in episodes], confidence
        ),
        "posterior_brier_score": _interval(
            [(1.0 - item.posterior_true_probability) ** 2 for item in episodes], confidence
        ),
        "posterior_entropy": _interval(
            [item.posterior_entropy for item in episodes], confidence
        ),
    }
    recovery_interval = (
        _interval(recovery, confidence)
        if recovery
        else {"mean": None, "lower": None, "upper": None}
    )
    report["recovery_steps"] = {
        **recovery_interval,
        "observed_count": len(recovery),
        "censored_count": len(episodes) - len(recovery),
    }
    return report
