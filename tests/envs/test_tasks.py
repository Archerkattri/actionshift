from __future__ import annotations

import numpy as np
import pytest
import torch

from actionshift.contracts.types import ActionContract
from actionshift.envs.tasks import TASKS, TaskSpec, make_task_env


def _identity_contract() -> ActionContract:
    return ActionContract(
        permutation=(0, 1, 2, 3, 4, 5),
        sign=(1, 1, 1, 1, 1, 1),
        scale=(1, 1, 1, 1, 1, 1),
        target="delta",
        frame="base",
        lag=0,
        gripper_inverted=False,
    )


def test_frozen_task_registry_covers_pick_push_pull_stack_and_insertion() -> None:
    assert set(TASKS) == {
        "pick_cube",
        "push_cube",
        "pull_cube",
        "stack_cube",
        "peg_insertion_side",
    }
    assert len({spec.family for spec in TASKS.values()}) == 5
    assert all(spec.control_mode == "pd_ee_delta_pose" for spec in TASKS.values())


def test_task_success_normalizes_scalar_array_and_tensor() -> None:
    spec = TaskSpec("Test-v0", "test", max_episode_steps=10)
    assert spec.success({"success": True}).shape == (1,)
    torch.testing.assert_close(spec.success({"success": np.array([0, 1])}), torch.tensor([0, 1]))
    torch.testing.assert_close(spec.success({"success": torch.tensor([True])}), torch.tensor([1]))
    with pytest.raises(KeyError, match="success"):
        spec.success({})


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_real_maniskill_task_oracle_path_smoke(task_name: str) -> None:
    pytest.importorskip("mani_skill")
    environment = make_task_env(
        TASKS[task_name],
        _identity_contract(),
        num_envs=1,
        sim_backend="cpu",
    )
    try:
        observation, info = environment.reset(seed=7)
        assert torch.as_tensor(observation).shape[0] == 1
        assert TASKS[task_name].success(info).shape == (1,)
        next_observation, _, _, _, next_info = environment.step(torch.zeros(1, 7))
        assert torch.isfinite(torch.as_tensor(next_observation)).all()
        assert TASKS[task_name].success(next_info).shape == (1,)
    finally:
        environment.close()
