"""No-history and domain-randomized baselines without privileged execution inputs."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from actionshift.policies.actor_critic import ActorCritic


class NoAdaptMethod:
    def __init__(self, actor_critic: ActorCritic) -> None:
        if actor_critic.context_dim != 0:
            raise ValueError("no-adaptation actor cannot accept contract context")
        self.actor_critic = actor_critic

    def act(self, observation: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        if deterministic:
            return cast(tuple[Tensor, Tensor], self.actor_critic(observation))
        action, _log_probability, _entropy, value = self.actor_critic.sample(observation)
        return action, value


class DomainRandomizedContractSampler:
    def __init__(self, *, train_contract_ids: tuple[int, ...], seed: int) -> None:
        if not train_contract_ids:
            raise ValueError("at least one training contract is required")
        self.contract_ids = torch.tensor(train_contract_ids, dtype=torch.long)
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, reset_mask: Tensor, *, current: Tensor | None = None) -> Tensor:
        if reset_mask.ndim != 1 or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be a one-dimensional boolean tensor")
        indices = torch.randint(
            len(self.contract_ids), reset_mask.shape, generator=self.generator
        )
        sampled = self.contract_ids[indices].to(reset_mask.device)
        if current is None:
            if not torch.all(reset_mask):
                raise ValueError("current contracts are required for non-reset environments")
            return sampled
        if current.shape != reset_mask.shape:
            raise ValueError("current contract IDs must match reset_mask")
        return torch.where(reset_mask, sampled, current)
