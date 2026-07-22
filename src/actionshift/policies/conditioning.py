"""Leakage-safe transition history and posterior conditioning utilities."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class TransitionHistory:
    def __init__(self, *, batch_size: int, history_length: int, transition_dim: int) -> None:
        if min(batch_size, history_length, transition_dim) <= 0:
            raise ValueError("history dimensions must be positive")
        self.batch_size = batch_size
        self.history_length = history_length
        self.transition_dim = transition_dim
        self._tensor = torch.zeros(batch_size, history_length, transition_dim)

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def append(self, transition: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        if transition.shape != (self.batch_size, self.transition_dim):
            raise ValueError("transition shape mismatch")
        if self._tensor.device != transition.device or self._tensor.dtype != transition.dtype:
            self._tensor = self._tensor.to(device=transition.device, dtype=transition.dtype)
        if reset_mask is not None:
            if reset_mask.shape != (self.batch_size,):
                raise ValueError("reset_mask requires one value per environment")
            mask = reset_mask.to(device=transition.device, dtype=torch.bool)[:, None, None]
            self._tensor = torch.where(mask, torch.zeros_like(self._tensor), self._tensor)
        self._tensor = torch.cat((self._tensor[:, 1:], transition[:, None]), dim=1)
        return self._tensor


class ContractConditionedActor(nn.Module):
    """FiLM-conditioned task actor driven by posterior summaries, never labels."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        posterior_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, posterior_dim, hidden_dim) <= 0:
            raise ValueError("actor dimensions must be positive")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.posterior_dim = posterior_dim
        self.hidden_dim = hidden_dim
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh()
        )
        self.film = nn.Linear(posterior_dim, 2 * hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observation: Tensor, posterior_summary: Tensor) -> Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("observation dimension mismatch")
        if posterior_summary.shape != (*observation.shape[:-1], self.posterior_dim):
            raise ValueError("posterior summary shape mismatch")
        features = cast(Tensor, self.observation_encoder(observation))
        gamma, beta = cast(Tensor, self.film(posterior_summary)).chunk(2, dim=-1)
        conditioned = features * (1.0 + torch.tanh(gamma)) + beta
        return torch.tanh(cast(Tensor, self.action_head(conditioned)))

    def oracle_action(self, observation: Tensor, contract_index: Tensor) -> Tensor:
        if contract_index.shape != observation.shape[:-1]:
            raise ValueError("contract index must align with observation batch dimensions")
        one_hot = torch.nn.functional.one_hot(
            contract_index, num_classes=self.posterior_dim
        ).to(device=observation.device, dtype=observation.dtype)
        return cast(Tensor, self(observation, one_hot))

    def propose_candidates(
        self,
        observation: Tensor,
        posterior_summary: Tensor,
        *,
        candidate_count: int,
        radius: float,
        lower: float,
        upper: float,
        pose_dimensions: int,
    ) -> Tensor:
        if candidate_count <= 0 or radius < 0 or lower >= upper:
            raise ValueError("candidate count, radius, or bounds are invalid")
        if pose_dimensions <= 0 or pose_dimensions > self.action_dim:
            raise ValueError("pose_dimensions must fit the action")
        task_action = cast(Tensor, self(observation, posterior_summary))
        candidates = task_action[:, None].expand(-1, candidate_count, -1).clone()
        for index in range(1, candidate_count):
            axis = (index - 1) % pose_dimensions
            direction = 1.0 if ((index - 1) // pose_dimensions) % 2 == 0 else -1.0
            candidates[:, index, axis] += direction * radius
        return torch.clamp(candidates, min=lower, max=upper)
