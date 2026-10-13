from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from actionshift.contracts.types import ActionContract
from actionshift.envs.unknown_ack import UnknownAckRecorder, UnknownAckSchedule
from actionshift.envs.wrapper import HiddenContractWrapper


class RecordingArmEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.delivered: list[np.ndarray] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        return np.zeros(1, dtype=np.float32), {"action_delivery_truth": "must be removed"}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.delivered.append(np.array(action, copy=True))
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            False,
            False,
            {"action_delivery_outcome": "must be removed", "success": False},
        )


def test_paired_unknown_ack_delivery_through_hidden_contract_wrapper() -> None:
    applied_schedule, held_schedule = UnknownAckSchedule.paired([0], pair_id="pair-1")
    applied_recorder = UnknownAckRecorder()
    held_recorder = UnknownAckRecorder()
    applied_env = RecordingArmEnv()
    held_env = RecordingArmEnv()
    contract = ActionContract(
        permutation=tuple(range(6)),
        sign=(1, 1, 1, 1, 1, 1),
        scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        target="absolute",
        frame="base",
        lag=0,
        gripper_inverted=False,
    )
    applied = HiddenContractWrapper(
        applied_env,
        contract,
        unknown_ack_schedule=applied_schedule,
        unknown_ack_recorder=applied_recorder,
    )
    held = HiddenContractWrapper(
        held_env,
        contract,
        unknown_ack_schedule=held_schedule,
        unknown_ack_recorder=held_recorder,
    )
    applied.reset(seed=7)
    held.reset(seed=7)
    action = np.array([0.2, -0.3, 0.1, 0.05, -0.08, 0.12, 0.7], dtype=np.float32)

    applied_result = applied.step(action)
    held_result = held.step(action)

    assert applied_result[4] == held_result[4] == {
        "success": False,
        "action_acknowledgement": "unknown",
        "action_ack_event_id": "event-000",
    }
    assert np.allclose(applied_env.delivered[0], action)
    assert np.allclose(held_env.delivered[0][:6], 0.0)
    assert held_env.delivered[0][6] == action[6]
    assert torch.allclose(
        applied._complete_decoder.tracked_target,
        torch.as_tensor(action[:6], dtype=torch.float32).reshape(1, 6),
    )
    assert torch.count_nonzero(held._complete_decoder.tracked_target) == 0
    assert applied_recorder.records[0]["outcome"] == "applied"
    assert held_recorder.records[0]["outcome"] == "held"
    assert applied_schedule.public_manifest() == held_schedule.public_manifest()
