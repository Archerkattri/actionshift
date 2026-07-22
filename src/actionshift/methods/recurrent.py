"""History-adaptive recurrent domain-randomization baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RecurrentMethod(nn.Module):
    def __init__(self, *, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent = nn.GRUCell(observation_dim + action_dim + 2, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.hidden_state: Tensor
        self.register_buffer("hidden_state", torch.empty(0, hidden_dim), persistent=False)

    def act(
        self,
        observation: Tensor,
        previous_action: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = observation.shape[0]
        if observation.shape != (batch_size, self.observation_dim):
            raise ValueError("observation shape mismatch")
        if previous_action.shape != (batch_size, self.action_dim):
            raise ValueError("previous_action shape mismatch")
        if reward.shape != (batch_size,) or done.shape != (batch_size,):
            raise ValueError("reward and done require one value per environment")
        if self.hidden_state.shape[0] != batch_size:
            self.hidden_state = observation.new_zeros((batch_size, self.hidden_dim))
        self.hidden_state = torch.where(
            done.to(dtype=torch.bool).unsqueeze(-1),
            torch.zeros_like(self.hidden_state),
            self.hidden_state,
        )
        recurrent_input = torch.cat(
            (observation, previous_action, reward.unsqueeze(-1), done.unsqueeze(-1)), dim=-1
        )
        self.hidden_state = self.recurrent(recurrent_input, self.hidden_state)
        return torch.tanh(self.actor(self.hidden_state)), self.critic(self.hidden_state).squeeze(-1)
