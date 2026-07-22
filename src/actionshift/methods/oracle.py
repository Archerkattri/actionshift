"""Privileged contract-conditioned upper-bound baseline."""

from __future__ import annotations

from typing import cast

from torch import Tensor

from actionshift.policies.actor_critic import ActorCritic


class OracleMethod:
    def __init__(self, actor_critic: ActorCritic) -> None:
        if actor_critic.context_dim <= 0:
            raise ValueError("oracle actor requires privileged contract context")
        self.actor_critic = actor_critic

    def act(
        self, observation: Tensor, contract_context: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        if deterministic:
            return cast(tuple[Tensor, Tensor], self.actor_critic(observation, contract_context))
        action, _log_probability, _entropy, value = self.actor_critic.sample(
            observation, contract_context
        )
        return action, value
