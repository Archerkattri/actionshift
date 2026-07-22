from __future__ import annotations

import torch

from actionshift.training.rollout import generalized_advantage_estimate


def test_generalized_advantage_stops_at_done_masks() -> None:
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5], [0.0]])
    done = torch.tensor([[False], [True], [False]])

    advantages, returns = generalized_advantage_estimate(
        rewards, values, done, discount=1.0, gae_lambda=1.0
    )

    torch.testing.assert_close(advantages[:, 0], torch.tensor([1.5, 0.5, 0.5]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([2.0, 1.0, 1.0]))
