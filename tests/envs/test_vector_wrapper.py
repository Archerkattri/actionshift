from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from actionshift.contracts.types import ActionContract
from actionshift.envs.wrapper import HiddenContractWrapper


class BatchedCompleteEnv(gym.Env[dict[str, Any], torch.Tensor]):
    def __init__(self) -> None:
        self.state = torch.zeros(2, 7)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.state.zero_()
        self.steps = 0
        return {"state": self.state.clone()}, {"nested": {"contract_id": "remove", "ok": 1}}

    def step(self, action: torch.Tensor):
        self.state += action
        self.steps += 1
        terminated = torch.tensor([self.steps == 1, False])
        truncated = torch.zeros(2, dtype=torch.bool)
        info = {"nested": {"oracle/contract": "remove", "ok": 2}}
        return {"state": self.state.clone()}, torch.zeros(2), terminated, truncated, info


def _contract(*, lag: int = 0) -> ActionContract:
    return ActionContract(
        permutation=tuple(range(6)),
        sign=(1,) * 6,
        scale=(1.0,) * 6,
        target="delta",
        frame="base",
        lag=lag,
        gripper_inverted=False,
    )


def test_unbatched_rotation_is_broadcast_and_partial_reset_is_isolated() -> None:
    wrapped = HiddenContractWrapper(
        BatchedCompleteEnv(),
        _contract(lag=1),
        ee_rotation_provider=lambda: torch.eye(3),
    )
    _, reset_info = wrapped.reset()
    assert reset_info == {"nested": {"ok": 1}}

    wrapped.step(torch.tensor([[1.0] * 7, [10.0] * 7]))
    observation, _, _, _, info = wrapped.step(
        torch.tensor([[2.0] * 7, [20.0] * 7])
    )

    torch.testing.assert_close(observation["state"][0], torch.zeros(7))
    torch.testing.assert_close(observation["state"][1], torch.full((7,), 10.0))
    assert info == {"nested": {"ok": 2}}


def test_complete_decoder_rejects_silent_batch_size_change() -> None:
    wrapped = HiddenContractWrapper(BatchedCompleteEnv(), _contract())
    wrapped.reset()
    wrapped.step(torch.zeros(2, 7))

    with pytest.raises(ValueError, match="batch size changed"):
        wrapped.step(torch.zeros(1, 7))


def test_batched_numpy_identity_round_trip_preserves_type_and_values() -> None:
    class NumpyEnv(gym.Env[np.ndarray, np.ndarray]):
        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return np.zeros((2, 7), dtype=np.float32), {}

        def step(self, action: np.ndarray):
            assert isinstance(action, np.ndarray)
            return action, np.zeros(2), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), {}

    wrapped = HiddenContractWrapper(NumpyEnv(), _contract())
    wrapped.reset()
    action = np.arange(14, dtype=np.float32).reshape(2, 7)

    observation, *_ = wrapped.step(action)

    assert isinstance(observation, np.ndarray)
    np.testing.assert_array_equal(observation, action)
