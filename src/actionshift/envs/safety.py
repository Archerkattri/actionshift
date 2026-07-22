"""Fail-closed candidate safety checks used by adaptive controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SafetyBounds:
    """Hard action, workspace, motion, gripper, and collision limits."""

    action_lower: Tensor
    action_upper: Tensor
    workspace_lower: Tensor
    workspace_upper: Tensor
    max_translation_norm: float
    max_gripper_magnitude: float
    max_collision_risk: float

    def __post_init__(self) -> None:
        if self.action_lower.ndim != 1 or self.action_lower.shape != self.action_upper.shape:
            raise ValueError("action bounds must be equal-length vectors")
        if self.workspace_lower.shape != (3,) or self.workspace_upper.shape != (3,):
            raise ValueError("workspace bounds must be three-vectors")
        tensors = (
            self.action_lower,
            self.action_upper,
            self.workspace_lower,
            self.workspace_upper,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("safety bounds must be finite")
        if torch.any(self.action_lower > self.action_upper):
            raise ValueError("action lower bound exceeds upper bound")
        if torch.any(self.workspace_lower > self.workspace_upper):
            raise ValueError("workspace lower bound exceeds upper bound")
        limits = (
            self.max_translation_norm,
            self.max_gripper_magnitude,
            self.max_collision_risk,
        )
        if any(not math.isfinite(limit) or limit < 0 for limit in limits):
            raise ValueError("scalar safety limits must be finite and nonnegative")

    @classmethod
    def symmetric(cls, action_dim: int) -> SafetyBounds:
        """Construct conservative normalized-coordinate limits."""
        if action_dim < 4:
            raise ValueError("action_dim must include translation and gripper channels")
        return cls(
            action_lower=torch.full((action_dim,), -1.0),
            action_upper=torch.full((action_dim,), 1.0),
            workspace_lower=torch.full((3,), -1.0),
            workspace_upper=torch.full((3,), 1.0),
            max_translation_norm=1.0,
            max_gripper_magnitude=1.0,
            max_collision_risk=1.0,
        )

    def mask(
        self,
        candidates: Tensor,
        *,
        current_position: Tensor,
        predicted_displacement: Tensor,
        collision_risk: Tensor,
    ) -> Tensor:
        """Return a batch-by-candidate validity mask, failing closed on NaNs."""
        if candidates.ndim != 3 or candidates.shape[-1] != len(self.action_lower):
            raise ValueError("candidates must be batch by candidate by action")
        batch, count, _ = candidates.shape
        if current_position.shape != (batch, 3):
            raise ValueError("current_position must be batch by three")
        if predicted_displacement.shape != (batch, count, 3):
            raise ValueError("predicted_displacement shape does not match candidates")
        if collision_risk.shape != (batch, count):
            raise ValueError("collision_risk shape does not match candidates")

        lower = self.action_lower.to(device=candidates.device, dtype=candidates.dtype)
        upper = self.action_upper.to(device=candidates.device, dtype=candidates.dtype)
        workspace_lower = self.workspace_lower.to(
            device=candidates.device, dtype=candidates.dtype
        )
        workspace_upper = self.workspace_upper.to(
            device=candidates.device, dtype=candidates.dtype
        )
        next_position = current_position[:, None, :] + predicted_displacement
        finite = (
            torch.isfinite(candidates).all(dim=-1)
            & torch.isfinite(current_position).all(dim=-1, keepdim=True)
            & torch.isfinite(predicted_displacement).all(dim=-1)
            & torch.isfinite(collision_risk)
        )
        action_valid = ((candidates >= lower) & (candidates <= upper)).all(dim=-1)
        translation_valid = (
            torch.linalg.vector_norm(predicted_displacement, dim=-1)
            <= self.max_translation_norm
        )
        workspace_valid = (
            (next_position >= workspace_lower) & (next_position <= workspace_upper)
        ).all(dim=-1)
        gripper_valid = candidates[..., -1].abs() <= self.max_gripper_magnitude
        collision_valid = (collision_risk >= 0) & (
            collision_risk <= self.max_collision_risk
        )
        valid: Tensor = (
            finite
            & action_valid
            & translation_valid
            & workspace_valid
            & gripper_valid
            & collision_valid
        )
        return valid
