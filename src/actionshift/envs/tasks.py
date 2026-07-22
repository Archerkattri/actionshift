"""Frozen ManiSkill task adapters for cross-task contract evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from actionshift.contracts.types import ActionContract
from actionshift.envs.factory import make_hidden_contract_env
from actionshift.envs.wrapper import HiddenContractWrapper, OracleRecorder


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Stable task metadata independent of ManiSkill observation internals."""

    environment_id: str
    family: str
    max_episode_steps: int
    control_mode: str = "pd_ee_delta_pose"
    success_key: str = "success"

    def __post_init__(self) -> None:
        if not self.environment_id or not self.family or self.max_episode_steps <= 0:
            raise ValueError("task identifiers must be nonempty and horizon must be positive")

    def success(self, info: dict[str, Any]) -> Tensor:
        """Normalize ManiSkill's task-specific success value to a batch vector."""
        if self.success_key not in info:
            raise KeyError(f"missing task success key: {self.success_key}")
        result = torch.as_tensor(info[self.success_key], dtype=torch.int64)
        return result.reshape(-1)


TASKS: dict[str, TaskSpec] = {
    "pick_cube": TaskSpec("PickCube-v1", "pick", max_episode_steps=50),
    "push_cube": TaskSpec("PushCube-v1", "push", max_episode_steps=50),
    "pull_cube": TaskSpec("PullCube-v1", "pull", max_episode_steps=50),
    "stack_cube": TaskSpec("StackCube-v1", "stack", max_episode_steps=50),
    "peg_insertion_side": TaskSpec(
        "PegInsertionSide-v1", "insertion", max_episode_steps=100
    ),
}


def make_task_env(
    task: TaskSpec,
    contract: ActionContract,
    *,
    oracle_recorder: OracleRecorder | None = None,
    **environment_kwargs: Any,
) -> HiddenContractWrapper:
    """Create a registered task with the benchmark's shared action interface."""
    import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401

    kwargs: dict[str, Any] = {
        "obs_mode": "state",
        "control_mode": task.control_mode,
        "max_episode_steps": task.max_episode_steps,
    }
    kwargs.update(environment_kwargs)
    return make_hidden_contract_env(
        task.environment_id,
        contract,
        oracle_recorder=oracle_recorder,
        **kwargs,
    )
