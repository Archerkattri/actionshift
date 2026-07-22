"""Online system identification and RMA-style privileged teacher baselines."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class OSIAdapter(nn.Module):
    def __init__(
        self, *, transition_dim: int, history_length: int, latent_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.transition_dim = transition_dim
        self.history_length = history_length
        self.network = nn.Sequential(
            nn.Linear(transition_dim * history_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, history: Tensor) -> Tensor:
        if history.shape[-2:] != (self.history_length, self.transition_dim):
            raise ValueError("transition history shape mismatch")
        return cast(Tensor, self.network(history.flatten(start_dim=-2)))


class RMATeacherStudent(nn.Module):
    def __init__(
        self,
        *,
        privileged_dim: int,
        transition_dim: int,
        history_length: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.teacher = nn.Sequential(
            nn.Linear(privileged_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )
        self.student = OSIAdapter(
            transition_dim=transition_dim,
            history_length=history_length,
            latent_dim=latent_dim,
        )

    def adaptation_loss(self, history: Tensor, privileged_context: Tensor) -> Tensor:
        target = self.teacher(privileged_context).detach()
        return functional.mse_loss(self.student(history), target)
