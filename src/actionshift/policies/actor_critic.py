"""Parameter-efficient state-based actor and critic shared by baselines."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, hidden_dim) <= 0 or context_dim < 0:
            raise ValueError("network dimensions must be positive and context nonnegative")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_standard_deviation = nn.Parameter(torch.full((action_dim,), -0.5))

    def _features(self, observation: Tensor, context: Tensor | None) -> Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("observation dimension does not match actor")
        if self.context_dim:
            if context is None or context.shape != (*observation.shape[:-1], self.context_dim):
                raise ValueError("context is required with the configured dimension")
            policy_input = torch.cat((observation, context), dim=-1)
        else:
            if context is not None:
                raise ValueError("this policy must not receive privileged context")
            policy_input = observation
        return cast(Tensor, self.backbone(policy_input))

    def forward(self, observation: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features = self._features(observation, context)
        mean = cast(Tensor, self.actor_mean(features))
        value = cast(Tensor, self.critic(features))
        return torch.tanh(mean), value.squeeze(-1)

    def distribution(
        self, observation: Tensor, context: Tensor | None = None
    ) -> tuple[Normal, Tensor]:
        mean, value = self(observation, context)
        standard_deviation = self.log_standard_deviation.exp().expand_as(mean)
        return Normal(mean, standard_deviation), value

    def sample(
        self, observation: Tensor, context: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        distribution, value = self.distribution(observation, context)
        action = distribution.rsample()
        log_probability = distribution.log_prob(action).sum(dim=-1)  # type: ignore[no-untyped-call]
        entropy = distribution.entropy().sum(dim=-1)  # type: ignore[no-untyped-call]
        return action, log_probability, entropy, value

    def evaluate_actions(
        self, observation: Tensor, action: Tensor, context: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, value = self.distribution(observation, context)
        return (
            distribution.log_prob(action).sum(dim=-1),  # type: ignore[no-untyped-call]
            distribution.entropy().sum(dim=-1),  # type: ignore[no-untyped-call]
            value,
        )

    @property
    def parameter_count(self) -> int:
        return sum(math.prod(parameter.shape) for parameter in self.parameters())
