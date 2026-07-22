"""Leakage-safe environment wrapper for hidden action contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from actionshift.contracts.transforms import ActionLag, CompleteActionDecoder, decode_pose
from actionshift.contracts.types import ActionContract

_RESERVED_KEYS = {
    "canonical_action",
    "contract",
    "contract_id",
    "decoded_action",
    "hidden_contract",
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if key not in _RESERVED_KEYS and not str(key).startswith("oracle/")
        }
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


@dataclass
class OracleRecorder:
    """Evaluation-only ground truth kept outside agent observations and info."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, contract: ActionContract, decoded_action: Tensor) -> None:
        self.records.append(
            {
                "oracle/contract": contract.to_json(),
                "oracle/decoded_action": decoded_action.detach().clone(),
            }
        )


class HiddenContractWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Decode policy actions while withholding the active contract from the agent."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        contract: ActionContract,
        *,
        oracle_recorder: OracleRecorder | None = None,
        ee_rotation_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.contract = contract
        self.oracle_recorder = oracle_recorder
        self.ee_rotation_provider = ee_rotation_provider
        self._lag = ActionLag(steps=contract.lag)
        self._complete_decoder: CompleteActionDecoder | None = None
        self._pending_reset: Tensor | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._lag = ActionLag(steps=self.contract.lag)
        self._complete_decoder = None
        self._pending_reset = None
        observation, info = self.env.reset(seed=seed, options=options)
        return _sanitize(observation), _sanitize(info)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        input_is_numpy = isinstance(action, np.ndarray)
        tensor_action = torch.from_numpy(action) if input_is_numpy else action
        if not isinstance(tensor_action, Tensor):
            raise TypeError("action must be a torch.Tensor or numpy.ndarray")
        complete_semantics = (
            len(self.contract.permutation) == 6 and tensor_action.shape[-1:] == (7,)
        )
        if complete_semantics:
            input_batched = tensor_action.ndim > 1
            batched_action = tensor_action if input_batched else tensor_action.unsqueeze(0)
            if self._complete_decoder is None:
                self._complete_decoder = CompleteActionDecoder(
                    self.contract, batch_size=batched_action.shape[0]
                )
            elif self._complete_decoder.batch_size != batched_action.shape[0]:
                raise ValueError("action batch size changed while decoder state is live")
            rotation_value = (
                self.ee_rotation_provider()
                if self.ee_rotation_provider is not None
                else torch.eye(
                    3, device=batched_action.device, dtype=batched_action.dtype
                ).expand(batched_action.shape[0], 3, 3)
            )
            rotation = torch.as_tensor(
                rotation_value, device=batched_action.device, dtype=batched_action.dtype
            )
            if rotation.shape == (3, 3):
                rotation = rotation.expand(batched_action.shape[0], 3, 3)
            reset_mask = self._pending_reset
            if reset_mask is not None and reset_mask.ndim == 0:
                reset_mask = reset_mask.reshape(1)
            decoded_batch = self._complete_decoder.step(
                batched_action, ee_rotation=rotation, reset_mask=reset_mask
            )
            canonical = decoded_batch if input_batched else decoded_batch.squeeze(0)
        else:
            decoded = decode_pose(tensor_action, self.contract)
            canonical = self._lag.step(decoded, reset_mask=self._pending_reset)
        if self.oracle_recorder is not None:
            self.oracle_recorder.record(self.contract, canonical)
        environment_action: Tensor | np.ndarray = (
            canonical.detach().cpu().numpy() if input_is_numpy else canonical
        )
        observation, reward, terminated, truncated, info = self.env.step(environment_action)
        terminated_tensor = torch.as_tensor(terminated, device=canonical.device, dtype=torch.bool)
        truncated_tensor = torch.as_tensor(truncated, device=canonical.device, dtype=torch.bool)
        self._pending_reset = terminated_tensor | truncated_tensor
        return _sanitize(observation), reward, terminated, truncated, _sanitize(info)
