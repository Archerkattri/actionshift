"""Minimal differentiable contract transforms used by the week-one benchmark."""

from __future__ import annotations

import torch
from torch import Tensor

from actionshift.contracts.types import ActionContract


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """Convert unit quaternions ``(w, x, y, z)`` to proper rotation matrices.

    Accepts any leading batch shape and returns ``(*leading, 3, 3)``. The
    quaternion is normalized first, so a raw tcp-pose quaternion (which is unit up
    to floating-point noise) yields an orthonormal, right-handed matrix that
    passes :func:`_validate_rotation`. ManiSkill's tcp pose uses this ``wxyz``
    convention, matching ``adaptation.calibration``'s quaternion helpers.
    """
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four components (w, x, y, z)")
    normalized = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = normalized.unbind(-1)
    two = normalized.new_tensor(2.0)
    row0 = torch.stack(
        (1 - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y)), dim=-1
    )
    row1 = torch.stack(
        (two * (x * y + w * z), 1 - two * (x * x + z * z), two * (y * z - w * x)), dim=-1
    )
    row2 = torch.stack(
        (two * (x * z - w * y), two * (y * z + w * x), 1 - two * (x * x + y * y)), dim=-1
    )
    return torch.stack((row0, row1, row2), dim=-2)


def _validate_action(action: Tensor, contract: ActionContract) -> None:
    if action.ndim == 0 or action.shape[-1] != len(contract.permutation):
        raise ValueError("action's final dimension must match the contract")
    if not action.is_floating_point():
        raise ValueError("action must use a floating-point dtype")


def decode_pose(
    raw_action: Tensor,
    contract: ActionContract,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> Tensor:
    """Decode raw pose channels in permutation → sign → scale order."""
    _validate_action(raw_action, contract)
    permutation = torch.tensor(contract.permutation, device=raw_action.device)
    sign = raw_action.new_tensor(contract.sign)
    scale = raw_action.new_tensor(contract.scale)
    decoded = raw_action.index_select(-1, permutation) * sign * scale
    if lower is not None or upper is not None:
        decoded = torch.clamp(decoded, min=lower, max=upper)
    return decoded


def encode_pose(canonical_action: Tensor, contract: ActionContract) -> Tensor:
    """Invert the instantaneous permutation/sign/scale transform."""
    _validate_action(canonical_action, contract)
    sign = canonical_action.new_tensor(contract.sign)
    scale = canonical_action.new_tensor(contract.scale)
    semantic = canonical_action / (sign * scale)
    inverse = [0] * len(contract.permutation)
    for semantic_index, raw_index in enumerate(contract.permutation):
        inverse[raw_index] = semantic_index
    inverse_permutation = torch.tensor(inverse, device=canonical_action.device)
    return semantic.index_select(-1, inverse_permutation)


def _validate_rotation(rotation: Tensor, leading_shape: torch.Size) -> None:
    if rotation.shape != (*leading_shape, 3, 3):
        raise ValueError("ee_rotation must match action batch dimensions and end in 3x3")
    if not rotation.is_floating_point() or not torch.all(torch.isfinite(rotation)):
        raise ValueError("ee_rotation must be a finite floating-point rotation matrix")
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    orthogonality = rotation.transpose(-1, -2) @ rotation
    if not torch.allclose(orthogonality, identity.expand_as(orthogonality), rtol=1e-5, atol=1e-6):
        raise ValueError("ee_rotation must be orthonormal")
    if torch.any(torch.linalg.det(rotation) <= 0):
        raise ValueError("ee_rotation must be a proper rotation")


def decode_complete_action(
    raw_action: Tensor,
    contract: ActionContract,
    *,
    ee_rotation: Tensor,
    tracked_target: Tensor,
    workspace_lower: Tensor | None = None,
    workspace_upper: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Decode a 6D pose twist plus named gripper channel.

    The complete instantaneous order is permutation, sign, scale, frame, and
    target. Lag is stateful and therefore applied by :class:`CompleteActionDecoder`.
    """
    if len(contract.permutation) != 6 or raw_action.shape[-1:] != (7,):
        raise ValueError("complete semantics require six pose channels plus gripper")
    if tracked_target.shape != (*raw_action.shape[:-1], 6):
        raise ValueError("tracked_target must match action batch dimensions")
    if tracked_target.device != raw_action.device or tracked_target.dtype != raw_action.dtype:
        raise ValueError("tracked_target must preserve action device and dtype")
    _validate_rotation(ee_rotation, raw_action.shape[:-1])
    rotation = ee_rotation.to(device=raw_action.device, dtype=raw_action.dtype)
    pose = decode_pose(raw_action[..., :6], contract)
    if contract.frame == "tool":
        translation = torch.einsum("...ij,...j->...i", rotation, pose[..., :3])
        rotation_vector = torch.einsum("...ij,...j->...i", rotation, pose[..., 3:])
        pose = torch.cat((translation, rotation_vector), dim=-1)
    next_target = pose if contract.target == "absolute" else tracked_target + pose
    if (workspace_lower is None) != (workspace_upper is None):
        raise ValueError("workspace lower and upper bounds must be provided together")
    if workspace_lower is not None and workspace_upper is not None:
        lower = workspace_lower.to(device=raw_action.device, dtype=raw_action.dtype)
        upper = workspace_upper.to(device=raw_action.device, dtype=raw_action.dtype)
        if lower.shape != (3,) or upper.shape != (3,) or torch.any(lower >= upper):
            raise ValueError("workspace bounds must be ordered three-vectors")
        clipped_position = torch.minimum(torch.maximum(next_target[..., :3], lower), upper)
        next_target = torch.cat((clipped_position, next_target[..., 3:]), dim=-1)
    canonical_pose = next_target - tracked_target
    gripper = raw_action[..., 6:]
    if contract.gripper_inverted:
        gripper = -gripper
    return torch.cat((canonical_pose, gripper), dim=-1), next_target


def encode_complete_action(
    canonical_action: Tensor,
    contract: ActionContract,
    *,
    ee_rotation: Tensor,
    tracked_target: Tensor,
) -> Tensor:
    """Encode a desired canonical delta for an oracle-conditioned policy path."""
    if len(contract.permutation) != 6 or canonical_action.shape[-1:] != (7,):
        raise ValueError("complete semantics require six pose channels plus gripper")
    if tracked_target.shape != (*canonical_action.shape[:-1], 6):
        raise ValueError("tracked_target must match action batch dimensions")
    _validate_rotation(ee_rotation, canonical_action.shape[:-1])
    pose = canonical_action[..., :6]
    semantic_pose = tracked_target + pose if contract.target == "absolute" else pose
    if contract.frame == "tool":
        inverse_rotation = ee_rotation.to(
            device=canonical_action.device, dtype=canonical_action.dtype
        ).transpose(-1, -2)
        translation = torch.einsum(
            "...ij,...j->...i", inverse_rotation, semantic_pose[..., :3]
        )
        rotation_vector = torch.einsum(
            "...ij,...j->...i", inverse_rotation, semantic_pose[..., 3:]
        )
        semantic_pose = torch.cat((translation, rotation_vector), dim=-1)
    raw_pose = encode_pose(semantic_pose, contract)
    gripper = canonical_action[..., 6:]
    if contract.gripper_inverted:
        gripper = -gripper
    return torch.cat((raw_pose, gripper), dim=-1)


class CompleteActionDecoder:
    """Stateful target tracker and per-environment lag for complete contracts."""

    def __init__(
        self,
        contract: ActionContract,
        *,
        batch_size: int,
        workspace_lower: Tensor | None = None,
        workspace_upper: Tensor | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.contract = contract
        self.batch_size = batch_size
        self.workspace_lower = workspace_lower
        self.workspace_upper = workspace_upper
        self._target: Tensor | None = None
        self._lag = ActionLag(steps=contract.lag)

    @property
    def tracked_target(self) -> Tensor | None:
        return self._target

    def step(
        self,
        raw_action: Tensor,
        *,
        ee_rotation: Tensor,
        reset_mask: Tensor | None = None,
    ) -> Tensor:
        if raw_action.shape != (self.batch_size, 7):
            raise ValueError("raw_action must be batch_size by seven")
        if self._target is None:
            self._target = torch.zeros_like(raw_action[..., :6])
        if reset_mask is not None:
            if reset_mask.shape != (self.batch_size,):
                raise ValueError("reset_mask must contain one value per environment")
            mask = reset_mask.to(device=raw_action.device, dtype=torch.bool).unsqueeze(-1)
            self._target = torch.where(mask, torch.zeros_like(self._target), self._target)
        canonical, self._target = decode_complete_action(
            raw_action,
            self.contract,
            ee_rotation=ee_rotation,
            tracked_target=self._target,
            workspace_lower=self.workspace_lower,
            workspace_upper=self.workspace_upper,
        )
        return self._lag.step(canonical, reset_mask=reset_mask)


class ActionLag:
    """Per-vectorized-environment FIFO with neutral zero initialization."""

    def __init__(self, *, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        self.steps = steps
        self._history: tuple[Tensor, ...] | None = None
        self._action_shape: torch.Size | None = None

    def step(self, action: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        if self.steps == 0:
            return action
        if action.ndim == 0:
            raise ValueError("action must have at least one dimension")
        if self._history is None:
            self._action_shape = action.shape
            self._history = tuple(torch.zeros_like(action) for _ in range(self.steps))
        elif action.shape != self._action_shape:
            raise ValueError("action shape changed; create a new lag buffer")

        history = self._history
        if reset_mask is not None:
            expected_shape = action.shape[:-1]
            if reset_mask.shape != expected_shape:
                raise ValueError("reset_mask must match action batch dimensions")
            mask = reset_mask.to(device=action.device, dtype=torch.bool).unsqueeze(-1)
            history = tuple(torch.where(mask, torch.zeros_like(item), item) for item in history)

        output = history[0]
        self._history = (*history[1:], action)
        return output
