"""Real ManiSkill probe environment for ``--real`` self-test runs.

Kept in its own module so the ManiSkill / sim dependency is imported lazily and
never touched by the synthetic demo and unit-test paths. The probe phase is short
(``budget`` steps from a fresh reset), so no auto-reset boundary can fall inside
it; identity end-effector rotation is used, matching the tool's declared scope.
"""

from __future__ import annotations

import torch
from torch import Tensor

from actionshift.adaptation.calibration import (
    ResponseCalibration,
    response_from_observations,
)
from actionshift.contracts.types import ActionContract


class RealProbeEnvironment:
    """Drive probe pulses through a real ManiSkill hidden-contract environment."""

    def __init__(
        self,
        task: str,
        contract: ActionContract,
        calibration: ResponseCalibration,
        *,
        num_envs: int,
        seed: int,
    ) -> None:
        from actionshift.benchmarking.ppo_parity import _make_environment

        self._environment = _make_environment(
            task, "noadapt_nonidentity", num_envs, contract
        )
        observation, _ = self._environment.reset(seed=seed)
        self._previous = observation
        self.batch_size = num_envs
        self.channels = 7 if calibration.has_gripper else 6
        self._calibration = calibration

    def step(self, raw_action: Tensor) -> Tensor:
        raw = raw_action.to(self._environment.device)
        observation, _, _, _, _ = self._environment.step(raw)
        response = response_from_observations(
            self._calibration, self._previous, observation
        )
        self._previous = observation
        return response.detach().to(device="cpu", dtype=torch.float32)

    def close(self) -> None:
        self._environment.close()
