from __future__ import annotations

import pytest
import torch

from actionshift.contracts.types import ActionContract
from actionshift.envs.tasks import TASKS, make_task_env


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
