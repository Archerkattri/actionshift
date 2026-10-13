from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from actionshift.contracts.types import ActionContract
from actionshift.envs.factory import make_hidden_contract_env
from actionshift.envs.wrapper import HiddenContractWrapper


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


def test_factory_wraps_a_registered_gymnasium_environment() -> None:
    environment_id = "ActionShiftTensorTest-v0"
    if environment_id not in gym.registry:
        gym.register(environment_id, entry_point=TensorEnv)

    wrapped = make_hidden_contract_env(environment_id, contract(), num_envs=1)

    assert isinstance(wrapped, HiddenContractWrapper)
