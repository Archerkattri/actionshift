"""Factories for registered Gymnasium and ManiSkill environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from actionshift.contracts.types import ActionContract
from actionshift.envs.wrapper import HiddenContractWrapper, OracleRecorder


def make_hidden_contract_env(
    environment_id: str,
    contract: ActionContract,
    *,
    oracle_recorder: OracleRecorder | None = None,
    **environment_kwargs: Any,
) -> HiddenContractWrapper:
    environment = gym.make(environment_id, **environment_kwargs)
    return HiddenContractWrapper(
        environment,
        contract,
        oracle_recorder=oracle_recorder,
    )
