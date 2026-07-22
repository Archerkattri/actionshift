from __future__ import annotations

import gymnasium as gym
import numpy as np

from actionshift.contracts.transforms import encode_pose
from actionshift.contracts.types import ActionContract
from actionshift.envs.wrapper import HiddenContractWrapper


class IntegratorEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.state = np.zeros(2, dtype=np.float64)

    def reset(self, *, seed: int | None = None, options=None):
        self.state = np.zeros(2, dtype=np.float64)
        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        self.state += action
        reward = -float(np.square(self.state).sum())
        return self.state.copy(), reward, False, False, {}


def test_oracle_encoding_matches_unwrapped_environment_for_1000_steps() -> None:
    contract = ActionContract(
        permutation=(1, 0),
        sign=(-1, 1),
        scale=(0.25, 2.0),
        target="delta",
        frame="base",
        lag=0,
        gripper_inverted=False,
    )
    canonical_env = IntegratorEnv()
    wrapped_env = HiddenContractWrapper(IntegratorEnv(), contract)
    canonical_observation, _ = canonical_env.reset(seed=7)
    wrapped_observation, _ = wrapped_env.reset(seed=7)
    rng = np.random.default_rng(20260718)

    for _ in range(1000):
        canonical_action = rng.uniform(-0.01, 0.01, size=2)
        raw_action = encode_pose_array(canonical_action, contract)
        canonical_observation, canonical_reward, *_ = canonical_env.step(canonical_action)
        wrapped_observation, wrapped_reward, *_ = wrapped_env.step(raw_action)
        np.testing.assert_allclose(wrapped_observation, canonical_observation, atol=1e-12)
        assert wrapped_reward == canonical_reward


def encode_pose_array(action: np.ndarray, contract: ActionContract) -> np.ndarray:
    import torch

    return encode_pose(torch.from_numpy(action), contract).numpy()
