"""Paired evaluator-only schedules for ambiguous action delivery acknowledgements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

DeliveryOutcome = Literal["applied", "held"]


@dataclass(frozen=True, slots=True)
class UnknownAckEvent:
    """One hidden delivery outcome behind an adapter-visible unknown ACK."""

    step: int
    event_id: str
    pair_id: str
    outcome: DeliveryOutcome

    def __post_init__(self) -> None:
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
            or not self.event_id
            or not self.pair_id
        ):
            raise ValueError("event step, event_id, and pair_id must be valid")
        if self.outcome not in {"applied", "held"}:
            raise ValueError("outcome must be applied or held")


@dataclass(frozen=True, slots=True)
class UnknownAckSchedule:
    """A fixed, paired fault schedule whose applied/held truth stays evaluator-only."""

    events: tuple[UnknownAckEvent, ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("schedule must contain at least one event")
        steps = tuple(event.step for event in self.events)
        if steps != tuple(sorted(set(steps))):
            raise ValueError("event steps must be unique and increasing")
        if len({event.pair_id for event in self.events}) != 1:
            raise ValueError("all events in one schedule must share a pair_id")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event_id values must be unique within a schedule")

    @classmethod
    def paired(
        cls, steps: int | Sequence[int], *, pair_id: str
    ) -> tuple[UnknownAckSchedule, UnknownAckSchedule]:
        """Build applied/held schedules with identical public event timing."""
        normalized = (steps,) if isinstance(steps, int) else tuple(steps)
        if not pair_id:
            raise ValueError("pair_id must be nonempty")
        if (
            not normalized
            or any(isinstance(step, bool) or not isinstance(step, int) for step in normalized)
            or normalized != tuple(sorted(set(normalized)))
        ):
            raise ValueError("steps must be nonempty, unique, and increasing")
        event_ids = tuple(f"event-{index:03d}" for index in range(len(normalized)))
        applied = tuple(
            UnknownAckEvent(step, event_id, pair_id, "applied")
            for step, event_id in zip(normalized, event_ids, strict=True)
        )
        held = tuple(
            UnknownAckEvent(step, event_id, pair_id, "held")
            for step, event_id in zip(normalized, event_ids, strict=True)
        )
        return cls(applied), cls(held)

    def event_at(self, step: int) -> UnknownAckEvent | None:
        return next((event for event in self.events if event.step == step), None)

    def public_manifest(self) -> list[dict[str, object]]:
        """Return schedule timing and the shared ACK, without delivery truth."""
        return [
            {"step": event.step, "event_id": event.event_id, "acknowledgement": "unknown"}
            for event in self.events
        ]

    def private_manifest(self) -> list[dict[str, object]]:
        """Return evaluator-only outcomes for paired scoring and audit."""
        return [
            {
                **entry,
                "pair_id": event.pair_id,
                "outcome": event.outcome,
            }
            for event, entry in zip(self.events, self.public_manifest(), strict=True)
        ]


@dataclass
class UnknownAckRecorder:
    """Private delivery ledger; never attach this object or its rows to adapter info."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        event: UnknownAckEvent,
        *,
        requested_action: Tensor,
        delivered_action: Tensor,
    ) -> None:
        self.records.append(
            {
                "step": event.step,
                "event_id": event.event_id,
                "pair_id": event.pair_id,
                "outcome": event.outcome,
                "requested_action": requested_action.detach().cpu().tolist(),
                "delivered_action": delivered_action.detach().cpu().tolist(),
            }
        )


def hold_arm_action(action: Tensor) -> Tensor:
    """Replace six arm channels with zero while preserving the gripper command."""
    if action.ndim == 0 or action.shape[-1] != 7:
        raise ValueError("unknown-ACK hold faults require six arm channels and one gripper")
    return torch.cat((torch.zeros_like(action[..., :6]), action[..., 6:]), dim=-1)


__all__ = [
    "DeliveryOutcome",
    "UnknownAckEvent",
    "UnknownAckRecorder",
    "UnknownAckSchedule",
    "hold_arm_action",
]
