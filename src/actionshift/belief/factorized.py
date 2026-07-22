"""Structured neural posterior over action-contract fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from actionshift.contracts.types import ActionContract


class FactorizedContractBelief(nn.Module):
    def __init__(
        self,
        *,
        transition_dim: int,
        hidden_dim: int,
        cardinalities: Mapping[str, int],
    ) -> None:
        super().__init__()
        if min(transition_dim, hidden_dim) <= 0 or not cardinalities:
            raise ValueError("belief dimensions and field set must be positive")
        if any(cardinality <= 1 for cardinality in cardinalities.values()):
            raise ValueError("every modeled field requires at least two categories")
        self.transition_dim = transition_dim
        self.hidden_dim = hidden_dim
        self.cardinalities = dict(cardinalities)
        self.encoder = nn.GRU(transition_dim, hidden_dim, batch_first=True)
        self.cell = nn.GRUCell(transition_dim, hidden_dim)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, size) for field, size in self.cardinalities.items()}
        )
        self.hidden_state: Tensor
        self.register_buffer("hidden_state", torch.empty(0, hidden_dim), persistent=False)

    def _marginals(self, features: Tensor) -> dict[str, Tensor]:
        return {
            field: torch.softmax(cast(Tensor, head(features)), dim=-1)
            for field, head in self.heads.items()
        }

    def forward(self, history: Tensor) -> dict[str, Tensor]:
        if history.ndim != 3 or history.shape[-1] != self.transition_dim:
            raise ValueError("history must be batch by time by transition dimension")
        _sequence, hidden = self.encoder(history)
        return self._marginals(hidden[-1])

    def step(self, transition: Tensor, *, reset_mask: Tensor) -> dict[str, Tensor]:
        if transition.ndim != 2 or transition.shape[-1] != self.transition_dim:
            raise ValueError("transition must be batch by transition dimension")
        batch_size = transition.shape[0]
        if reset_mask.shape != (batch_size,) or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be boolean with one value per batch row")
        if self.hidden_state.shape[0] != batch_size:
            self.hidden_state = transition.new_zeros((batch_size, self.hidden_dim))
        self.hidden_state = torch.where(
            reset_mask.unsqueeze(-1), torch.zeros_like(self.hidden_state), self.hidden_state
        )
        self.hidden_state = self.cell(transition, self.hidden_state)
        return self._marginals(self.hidden_state)

    @staticmethod
    def sample(marginals: Mapping[str, Tensor], *, samples: int) -> dict[str, Tensor]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        return {
            field: torch.multinomial(probabilities, samples, replacement=True)
            for field, probabilities in marginals.items()
        }

    @staticmethod
    def map_composition(marginals: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {field: probabilities.argmax(dim=-1) for field, probabilities in marginals.items()}

    @staticmethod
    def auxiliary_loss(marginals: Mapping[str, Tensor], labels: Mapping[str, Tensor]) -> Tensor:
        if set(marginals) != set(labels):
            raise ValueError("privileged labels must match modeled fields")
        losses = [
            functional.nll_loss(torch.log(marginals[field].clamp_min(1e-12)), labels[field])
            for field in marginals
        ]
        return torch.stack(losses).mean()


def _field_value(contract: ActionContract, field: str) -> Any:
    values = {
        "permutation": contract.permutation,
        "sign": contract.sign,
        "scale": contract.scale,
        "target": contract.target,
        "frame": contract.frame,
        "lag": contract.lag,
        "gripper": contract.gripper_inverted,
    }
    return values[field]


def exact_factor_marginals(
    contracts: tuple[ActionContract, ...], probabilities: Tensor
) -> dict[str, Tensor]:
    if not contracts or probabilities.shape != (len(contracts),):
        raise ValueError("one exact probability is required per contract")
    if torch.any(probabilities < 0) or not torch.isclose(
        probabilities.sum(), probabilities.new_tensor(1.0)
    ):
        raise ValueError("exact probabilities must be normalized")
    result: dict[str, Tensor] = {}
    for field in ("permutation", "sign", "scale", "target", "frame", "lag", "gripper"):
        vocabulary = sorted({_field_value(contract, field) for contract in contracts}, key=str)
        marginal = probabilities.new_zeros(len(vocabulary))
        for contract, probability in zip(contracts, probabilities, strict=True):
            marginal[vocabulary.index(_field_value(contract, field))] += probability
        result[field] = marginal
    return result


def calibration_error(confidence: Tensor, correct: Tensor, *, bins: int = 10) -> dict[str, float]:
    if confidence.ndim != 1 or correct.shape != confidence.shape or correct.dtype != torch.bool:
        raise ValueError("confidence and correctness must be aligned one-dimensional tensors")
    if bins <= 0 or torch.any(confidence < 0) or torch.any(confidence > 1):
        raise ValueError("bins must be positive and confidence must lie in [0, 1]")
    correctness = correct.to(dtype=confidence.dtype)
    brier = torch.square(confidence - correctness).mean()
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    ece = confidence.new_tensor(0.0)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index + 1 == bins else confidence < upper
        )
        if torch.any(selected):
            weight = selected.to(dtype=confidence.dtype).mean()
            ece += weight * torch.abs(confidence[selected].mean() - correctness[selected].mean())
    return {"brier": float(brier), "ece": float(ece)}
