"""Paired uncertainty, multiplicity, and censoring utilities for sprint reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

import torch


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


@dataclass(frozen=True, slots=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    pairs: int


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    observed_count: int
    censored_count: int
    observed_mean: float | None
    restricted_mean: float
    horizon: int


def bootstrap_interval(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 20260718,
) -> BootstrapInterval:
    """Return a deterministic percentile bootstrap interval for the mean."""
    if not values:
        raise ValueError("at least one value is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("confidence and samples are invalid")
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(values), (samples, len(values)), generator=generator, device="cpu"
    )
    means = tensor[indices].mean(dim=1)
    tail = (1 - confidence) / 2
    lower, upper = torch.quantile(means, torch.tensor([tail, 1 - tail], dtype=torch.float64))
    return BootstrapInterval(
        estimate=float(tensor.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def paired_success_difference(
    candidate: list[bool],
    baseline: list[bool],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 20260718,
) -> PairedInterval:
    """Bootstrap candidate-minus-baseline success using intact episode pairs."""
    if len(candidate) != len(baseline):
        raise ValueError("paired inputs must contain the same number of episodes")
    differences = [
        float(left) - float(right)
        for left, right in zip(candidate, baseline, strict=True)
    ]
    interval = bootstrap_interval(
        differences, confidence=confidence, samples=samples, seed=seed
    )
    return PairedInterval(
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        confidence=interval.confidence,
        samples=interval.samples,
        pairs=len(differences),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's step-down family-wise error correction."""
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    count = len(p_values)
    for rank, (original_index, value) in enumerate(ordered):
        current = min(1.0, (count - rank) * value)
        previous = max(previous, current)
        adjusted[original_index] = previous
    return adjusted


def recovery_summary(values: list[int | None], *, horizon: int) -> RecoverySummary:
    """Retain non-recovery as right-censoring at a declared episode horizon."""
    if not values:
        raise ValueError("at least one recovery observation is required")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    observed = [value for value in values if value is not None]
    if any(value < 0 or value > horizon for value in observed):
        raise ValueError("recovery steps must lie within the episode horizon")
    restricted = [float(value) if value is not None else float(horizon) for value in values]
    return RecoverySummary(
        observed_count=len(observed),
        censored_count=len(values) - len(observed),
        observed_mean=fmean(observed) if observed else None,
        restricted_mean=fmean(restricted),
        horizon=horizon,
    )


def superiority_allowed(seeds: list[int]) -> bool:
    """Enforce the preregistration that one or two seeds cannot support superiority."""
    return len(set(seeds)) >= 3
