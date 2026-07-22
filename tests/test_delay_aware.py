"""Tests for the delay-aware augmented-state backbone helpers."""

from __future__ import annotations

import torch

from actionshift.adaptation.delay_aware import (
    DEFAULT_HISTORY,
    ActionHistoryBuffer,
    DelayAwarePpoAgent,
)
from actionshift.contracts.transforms import ActionLag


def test_augment_appends_flattened_history() -> None:
    buffer = ActionHistoryBuffer(2, 7, history=4)
    observation = torch.zeros((2, 11))
    augmented = buffer.augment(observation)
    assert augmented.shape == (2, 11 + 4 * 7)
    # Zero history at start -> augmented tail is all zeros.
    assert torch.all(augmented[:, 11:] == 0)


def test_push_orders_most_recent_first() -> None:
    buffer = ActionHistoryBuffer(1, 2, history=3)
    first = torch.tensor([[1.0, 1.0]])
    second = torch.tensor([[2.0, 2.0]])
    buffer.push(first)
    buffer.push(second)
    augmented = buffer.augment(torch.zeros((1, 0)))
    # [a_{t-1}=second, a_{t-2}=first, a_{t-3}=zeros]
    assert torch.equal(augmented[0, :2], second[0])
    assert torch.equal(augmented[0, 2:4], first[0])
    assert torch.all(augmented[0, 4:] == 0)


def test_reset_zeros_selected_environments() -> None:
    buffer = ActionHistoryBuffer(2, 2, history=2)
    buffer.push(torch.tensor([[3.0, 3.0], [4.0, 4.0]]))
    buffer.reset(torch.tensor([True, False]))
    augmented = buffer.augment(torch.zeros((2, 0)))
    assert torch.all(augmented[0] == 0)
    assert torch.equal(augmented[1, :2], torch.tensor([4.0, 4.0]))


def test_lag_execute_matches_action_lag_for_uniform_lag() -> None:
    """Per-env lag executor must be bit-identical to ActionLag at a uniform lag."""
    for steps in (0, 1, 2, 4):
        num_envs, action_dimension = 3, 7
        buffer = ActionHistoryBuffer(num_envs, action_dimension, history=DEFAULT_HISTORY)
        reference = ActionLag(steps=steps)
        lag = torch.full((num_envs,), steps, dtype=torch.long)
        generator = torch.Generator().manual_seed(steps + 1)
        for _ in range(10):
            action = torch.randn(
                (num_envs, action_dimension), generator=generator
            )
            executed = buffer.lag_execute(action, lag)
            buffer.push(action)
            expected = reference.step(action)
            assert torch.allclose(executed, expected), f"mismatch at lag={steps}"


def test_lag_execute_respects_per_env_reset() -> None:
    """After a per-env reset the lagged output falls back to neutral zeros."""
    buffer = ActionHistoryBuffer(2, 2, history=4)
    lag = torch.tensor([2, 2])
    a1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    buffer.lag_execute(a1, lag)
    buffer.push(a1)
    a2 = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    buffer.lag_execute(a2, lag)
    buffer.push(a2)
    buffer.reset(torch.tensor([True, False]))
    a3 = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    executed = buffer.lag_execute(a3, lag)
    # env 0 was reset -> lag-2 output is zero; env 1 keeps a1.
    assert torch.all(executed[0] == 0)
    assert torch.equal(executed[1], a1[1])


def test_agent_input_width_includes_history() -> None:
    agent = DelayAwarePpoAgent(11, 7, history=4)
    first_layer = agent.actor_mean[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 11 + 4 * 7
    output = agent.deterministic_action(torch.zeros((5, 11 + 4 * 7)))
    assert output.shape == (5, 7)
