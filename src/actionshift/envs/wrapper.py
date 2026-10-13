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
from actionshift.envs.unknown_ack import (
    UnknownAckRecorder,
    UnknownAckSchedule,
    hold_arm_action,
)

_RESERVED_KEYS = {
    "canonical_action",
    "contract",
    "contract_id",
    "decoded_action",
    "hidden_contract",
    "action_acknowledgement",
    "action_ack_event_id",
    "action_delivery_outcome",
    "action_delivery_truth",
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
        unknown_ack_schedule: UnknownAckSchedule | None = None,
        unknown_ack_recorder: UnknownAckRecorder | None = None,
    ) -> None:
        super().__init__(env)
        self.contract = contract
        self.oracle_recorder = oracle_recorder
        self.ee_rotation_provider = ee_rotation_provider
        self.unknown_ack_schedule = unknown_ack_schedule
        self.unknown_ack_recorder = unknown_ack_recorder
        if unknown_ack_schedule is not None and (
            len(contract.permutation) != 6 or contract.lag != 0
        ):
            raise ValueError("unknown-ACK schedules require six pose channels and zero lag")
        self._lag = ActionLag(steps=contract.lag)
        self._complete_decoder: CompleteActionDecoder | None = None
        self._pending_reset: Tensor | None = None
        self._episode_step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._lag = ActionLag(steps=self.contract.lag)
        self._complete_decoder = None
        self._pending_reset = None
        self._episode_step = 0
        observation, info = self.env.reset(seed=seed, options=options)
        return _sanitize(observation), _sanitize(info)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        input_is_numpy = isinstance(action, np.ndarray)
        tensor_action = torch.from_numpy(action) if input_is_numpy else action
        if not isinstance(tensor_action, Tensor):
            raise TypeError("action must be a torch.Tensor or numpy.ndarray")
        fault_event = (
            self.unknown_ack_schedule.event_at(self._episode_step)
            if self.unknown_ack_schedule is not None
            else None
        )
        held_fault = fault_event is not None and fault_event.outcome == "held"
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
            target_update_mask = (
                torch.zeros(batched_action.shape[0], device=batched_action.device, dtype=torch.bool)
                if held_fault
                else None
            )
            decoded_batch = self._complete_decoder.step(
                batched_action,
                ee_rotation=rotation,
                reset_mask=reset_mask,
                target_update_mask=target_update_mask,
            )
            canonical = decoded_batch if input_batched else decoded_batch.squeeze(0)
        else:
            decoded = decode_pose(tensor_action, self.contract)
            canonical = self._lag.step(decoded, reset_mask=self._pending_reset)
        requested = canonical
        if fault_event is not None:
            if canonical.shape[-1] != 7:
                raise ValueError("unknown-ACK events require a seven-channel arm/gripper action")
            if held_fault:
                canonical = hold_arm_action(canonical)
        if self.oracle_recorder is not None:
            self.oracle_recorder.record(self.contract, canonical)
        environment_action: Tensor | np.ndarray = (
            canonical.detach().cpu().numpy() if input_is_numpy else canonical
        )
        observation, reward, terminated, truncated, info = self.env.step(environment_action)
        if fault_event is not None and self.unknown_ack_recorder is not None:
            self.unknown_ack_recorder.record(
                fault_event,
                requested_action=requested,
                delivered_action=canonical,
            )
        terminated_tensor = torch.as_tensor(terminated, device=canonical.device, dtype=torch.bool)
        truncated_tensor = torch.as_tensor(truncated, device=canonical.device, dtype=torch.bool)
        self._pending_reset = terminated_tensor | truncated_tensor
        safe_info = _sanitize(info)
        if fault_event is not None:
            safe_info["action_acknowledgement"] = "unknown"
            safe_info["action_ack_event_id"] = fault_event.event_id
        self._episode_step += 1
        return _sanitize(observation), reward, terminated, truncated, safe_info
