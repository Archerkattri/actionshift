"""Exact finite Bayesian contract belief used as a correctness oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from actionshift.contracts.types import ActionContract


@dataclass(frozen=True, slots=True)
class ExactContractBelief:
    contracts: tuple[ActionContract, ...]
    log_probabilities: Tensor

    def __post_init__(self) -> None:
        if not self.contracts:
            raise ValueError("belief requires at least one contract")
        if self.log_probabilities.shape != (len(self.contracts),):
            raise ValueError("one log probability is required per contract")
        if not self.log_probabilities.is_floating_point():
            raise ValueError("log probabilities must use a floating-point dtype")
        if not torch.isfinite(torch.logsumexp(self.log_probabilities, dim=0)):
            raise ValueError("belief must assign finite mass to at least one contract")

    @classmethod
    def uniform(
        cls,
        contracts: tuple[ActionContract, ...],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> ExactContractBelief:
        if not contracts:
            raise ValueError("belief requires at least one contract")
        count = torch.tensor(float(len(contracts)), dtype=dtype, device=device)
        log_probability = -torch.log(count)
        return cls(contracts, log_probability.repeat(len(contracts)))

    @property
    def probabilities(self) -> Tensor:
        return torch.softmax(self.log_probabilities, dim=0)

    def update(self, observation_log_likelihood: Tensor) -> ExactContractBelief:
        if observation_log_likelihood.shape != self.log_probabilities.shape:
            raise ValueError("likelihood shape must match the finite contract set")
        unnormalized = self.log_probabilities + observation_log_likelihood.to(
            device=self.log_probabilities.device,
            dtype=self.log_probabilities.dtype,
        )
        normalized = unnormalized - torch.logsumexp(unnormalized, dim=0)
        return ExactContractBelief(self.contracts, normalized)
