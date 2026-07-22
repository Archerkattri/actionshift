from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from actionshift.contracts.types import ActionContract
from actionshift.envs.factory import make_hidden_contract_env
from actionshift.envs.wrapper import HiddenContractWrapper, OracleRecorder


def contract(*, lag: int = 0) -> ActionContract:
    return ActionContract(
        permutation=(1, 0),
        sign=(1, -1),
        scale=(0.5, 2.0),
        target="delta",
        frame="base",
        lag=lag,
        gripper_inverted=False,
    )


class TensorEnv(gym.Env[dict[str, Any], torch.Tensor]):
    def __init__(self, num_envs: int = 1, action_dim: int = 2) -> None:
        self.num_envs = num_envs
        self.state = torch.zeros(num_envs, action_dim)
        self.step_count = 0
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_envs, action_dim), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_envs, action_dim),
                    dtype=np.float32,
                ),
                "contract_id": gym.spaces.Discrete(100),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state = torch.zeros_like(self.state)
        self.step_count = 0
        observation = {"state": self.state.clone(), "contract_id": 99}
        return observation, {"contract_id": 99, "safe": "kept"}

    def step(self, action: torch.Tensor):
        self.state = self.state + action
        self.step_count += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.num_envs == 2 and self.step_count == 1:
            terminated[0] = True
        truncated = torch.zeros_like(terminated)
        observation = {"state": self.state.clone(), "decoded_action": action.clone()}
        info = {"decoded_action": action.clone(), "safe": "kept"}
        return observation, self.state.sum(dim=-1), terminated, truncated, info


def test_wrapper_removes_contract_side_channels_from_observation_and_info() -> None:
    wrapped = HiddenContractWrapper(TensorEnv(), contract())

    observation, reset_info = wrapped.reset()
    next_observation, _, _, _, step_info = wrapped.step(torch.tensor([[2.0, 3.0]]))

    assert "contract_id" not in observation
    assert "contract_id" not in reset_info
    assert "decoded_action" not in next_observation
    assert "decoded_action" not in step_info
    assert reset_info["safe"] == "kept"
    assert step_info["safe"] == "kept"


def test_oracle_recorder_is_separate_from_agent_info() -> None:
    recorder = OracleRecorder()
    wrapped = HiddenContractWrapper(TensorEnv(), contract(), oracle_recorder=recorder)
    wrapped.reset()

    _, _, _, _, info = wrapped.step(torch.tensor([[2.0, 3.0]]))

    assert not any(key.startswith("oracle/") for key in info)
    assert len(recorder.records) == 1
    assert recorder.records[0]["oracle/contract"] == contract().to_json()
    torch.testing.assert_close(
        recorder.records[0]["oracle/decoded_action"], torch.tensor([[1.5, -4.0]])
    )


def test_vectorized_done_mask_clears_lag_only_for_finished_environment() -> None:
    one_dimensional_contract = ActionContract(
        permutation=(0,),
        sign=(1,),
        scale=(1.0,),
        target="delta",
        frame="base",
        lag=1,
        gripper_inverted=False,
    )
    wrapped = HiddenContractWrapper(
        TensorEnv(num_envs=2, action_dim=1), one_dimensional_contract
    )
    wrapped.reset()

    wrapped.step(torch.tensor([[1.0], [10.0]]))
    observation, _, _, _, _ = wrapped.step(torch.tensor([[2.0], [20.0]]))

    torch.testing.assert_close(observation["state"], torch.tensor([[0.0], [10.0]]))


def test_numpy_action_returns_numpy_action_to_numpy_environment() -> None:
    class NumpyEnv(gym.Env[np.ndarray, np.ndarray]):
        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: np.ndarray):
            assert isinstance(action, np.ndarray)
            return action, 0.0, False, False, {}

    wrapped = HiddenContractWrapper(NumpyEnv(), contract())
    wrapped.reset()

    observation, *_ = wrapped.step(np.array([2.0, 3.0], dtype=np.float32))

    np.testing.assert_allclose(observation, np.array([1.5, -4.0], dtype=np.float32))


def test_factory_wraps_a_registered_gymnasium_environment() -> None:
    environment_id = "ActionShiftTensorTest-v0"
    if environment_id not in gym.registry:
        gym.register(environment_id, entry_point=TensorEnv)

    wrapped = make_hidden_contract_env(environment_id, contract(), num_envs=1)

    assert isinstance(wrapped, HiddenContractWrapper)


def test_wrapper_uses_complete_six_dof_and_named_gripper_semantics() -> None:
    complete_contract = ActionContract(
        permutation=tuple(range(6)),
        sign=(1,) * 6,
        scale=(1.0,) * 6,
        target="delta",
        frame="tool",
        lag=0,
        gripper_inverted=True,
    )
    rotation = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    wrapped = HiddenContractWrapper(
        TensorEnv(action_dim=7),
        complete_contract,
        ee_rotation_provider=lambda: rotation,
    )
    wrapped.reset()

    observation, *_ = wrapped.step(
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
    )

    torch.testing.assert_close(
        observation["state"], torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5]])
    )
