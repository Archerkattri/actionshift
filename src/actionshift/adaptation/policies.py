"""Canonical-action policies that drive the adapter tournament backbone-agnostically.

Every adaptation method consumes *canonical* commands and turns them into raw
wrapper actions; the backbone that produces those canonical commands is a
separate concern. ``CanonicalPolicy`` is the minimal per-step interface the
evaluation loop needs, so a frozen PPO actor and a frozen Diffusion Policy are
interchangeable behind it. The PPO wrapper is stateless; stateful backbones
(receding-horizon action chunking, observation history) implement ``act`` with
their own buffers and honour ``reset_mask`` to clear per-environment state at
episode boundaries.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from actionshift.benchmarking.ppo_parity import PpoAgent


class CanonicalPolicy(Protocol):
    """Produce one canonical command per environment per step."""

    def act(self, observation: Tensor, *, reset_mask: Tensor | None = None) -> Tensor:
        """Return the ``(num_envs, action_dim)`` canonical command for this step.

        ``reset_mask`` marks environments whose previous step crossed an episode
        boundary (so the observation is the first frame of a fresh episode);
        stateful policies clear those environments' internal buffers.
        """
        ...


class PpoCanonicalPolicy:
    """Stateless wrapper exposing a frozen PPO actor as a ``CanonicalPolicy``."""

    def __init__(self, agent: PpoAgent) -> None:
        self.agent = agent

    def act(self, observation: Tensor, *, reset_mask: Tensor | None = None) -> Tensor:
        return self.agent.deterministic_action(observation)
