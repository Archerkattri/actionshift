"""Contract-independent response calibration on the unwrapped environment.

Locating the tcp-pose slice inside the flat state observation and fitting the
linear tracking model are task knowledge, not contract knowledge: both are
measured on the plain, unwrapped environment with identity semantics and are
shared by every tournament method. Nothing here may touch a hidden contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

# Candidate saturation constants for the magnitude-dependent gain model. The
# saturating form ``observed = alpha * c / (1 + |c| / c0)`` reduces to the linear
# model as ``c0 -> inf``; a large ceiling value keeps that limit reachable.
_SATURATION_GRID: tuple[float, ...] = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6)
_GRIPPER_WIDTH = 2


class SupportsPoseProbe(Protocol):
    """Environment probe returning flat observations and true tcp poses."""

    def step_random(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (observation, tcp_position, tcp_quaternion, canonical_action)."""
        ...


@runtime_checkable
class SupportsGripperProbe(Protocol):
    """Probe that also exposes the current gripper finger joint positions."""

    def gripper_positions(self) -> Tensor:
        """Return the current finger joint positions, shape (batch, width)."""
        ...


@dataclass(frozen=True, slots=True)
class ResponseCalibration:
    """Frozen, JSON-serializable response calibration for one task.

    Version 1 is the original pose-only linear model (``alpha``/``sigma`` per
    pose channel). Version 2 optionally adds two contract-independent extensions,
    both measured on the same unwrapped rollouts:

    - a magnitude-dependent pose gain (``gain_model == "saturating"`` with a
      per-channel ``alpha_c0``): ``observed = alpha * c / (1 + |c| / c0)``,
      capturing mild PD under-tracking of large commands; and
    - a gripper evidence channel (``gripper_start``/``gripper_alpha``/
      ``gripper_sigma``): the command-to-finger-qpos-delta model that lets a
      belief identify ``gripper_inverted``.

    Every version-2 field defaults to the version-1 behaviour, so older JSON
    payloads (and callers that never opt in) stay bit-reproducible.
    """

    task: str
    position_start: int
    quaternion_start: int
    alpha: tuple[float, ...]
    sigma: tuple[float, ...]
    fit_r2: tuple[float, ...]
    steps: int
    version: int = 1
    gain_model: str = "linear"
    alpha_c0: tuple[float, ...] | None = None
    saturating_fit_r2: tuple[float, ...] | None = None
    gripper_start: int | None = None
    gripper_alpha: float | None = None
    gripper_sigma: float | None = None
    gripper_fit_r2: float | None = None

    def __post_init__(self) -> None:
        if self.position_start < 0 or self.quaternion_start < 0:
            raise ValueError("observation slice offsets must be nonnegative")
        for name in ("alpha", "sigma", "fit_r2"):
            if len(getattr(self, name)) != 6:
                raise ValueError(f"{name} must carry six channels")
        if any(s <= 0 or not math.isfinite(s) for s in self.sigma):
            raise ValueError("sigma entries must be finite and positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.version < 1:
            raise ValueError("version must be at least one")
        if self.gain_model not in ("linear", "saturating"):
            raise ValueError("gain_model must be 'linear' or 'saturating'")
        if self.gain_model == "saturating":
            if self.alpha_c0 is None or len(self.alpha_c0) != 6:
                raise ValueError("saturating gain requires a six-channel alpha_c0")
            if any(c <= 0 or not math.isfinite(c) for c in self.alpha_c0):
                raise ValueError("alpha_c0 entries must be finite and positive")
        self._validate_gripper()

    def _validate_gripper(self) -> None:
        present = [
            self.gripper_start is not None,
            self.gripper_alpha is not None,
            self.gripper_sigma is not None,
        ]
        if any(present) and not all(present):
            raise ValueError("gripper calibration fields must be provided together")
        if self.gripper_start is not None and self.gripper_start < 0:
            raise ValueError("gripper_start must be nonnegative")
        if self.gripper_sigma is not None and (
            self.gripper_sigma <= 0 or not math.isfinite(self.gripper_sigma)
        ):
            raise ValueError("gripper_sigma must be finite and positive")
        if self.gripper_alpha is not None and (
            self.gripper_alpha == 0 or not math.isfinite(self.gripper_alpha)
        ):
            raise ValueError("gripper_alpha must be finite and nonzero")

    @property
    def has_gripper(self) -> bool:
        return self.gripper_start is not None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> ResponseCalibration:
        value: dict[str, Any] = json.loads(payload)
        for name in ("alpha", "sigma", "fit_r2"):
            value[name] = tuple(value[name])
        for name in ("alpha_c0", "saturating_fit_r2"):
            if value.get(name) is not None:
                value[name] = tuple(value[name])
        return cls(**value)

    @classmethod
    def load(cls, path: Path) -> ResponseCalibration:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def quaternion_to_rotation_vector(quaternion: Tensor) -> Tensor:
    """Convert unit quaternions (w, x, y, z) to rotation vectors."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four components")
    normalized = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w: Tensor = normalized[..., 0].clamp(-1.0, 1.0)
    vector = normalized[..., 1:]
    angle = 2.0 * torch.acos(w.abs())
    axis_norm = vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w)).unsqueeze(-1)
    rotation_vector: Tensor = sign * vector / axis_norm * angle.unsqueeze(-1)
    return rotation_vector


def quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Hamilton product of (w, x, y, z) quaternions."""
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_conjugate(quaternion: Tensor) -> Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def pose_delta(
    previous_position: Tensor,
    previous_quaternion: Tensor,
    position: Tensor,
    quaternion: Tensor,
) -> Tensor:
    """Observed six-channel response: translation delta plus rotation vector."""
    translation = position - previous_position
    relative = quaternion_multiply(quaternion, quaternion_conjugate(previous_quaternion))
    return torch.cat((translation, quaternion_to_rotation_vector(relative)), dim=-1)


def locate_contiguous_slice(observations: Tensor, targets: Tensor) -> int:
    """Find the flat-observation offset holding the target block exactly.

    ``observations`` is (steps, batch, obs_dim); ``targets`` is
    (steps, batch, width). The offset must reproduce the block across every
    step and environment; anything less is a layout mismatch, not a match.
    """
    if observations.ndim != 3 or targets.ndim != 3:
        raise ValueError("observations and targets must be (steps, batch, width)")
    width = targets.shape[-1]
    for start in range(observations.shape[-1] - width + 1):
        if torch.allclose(
            observations[..., start : start + width], targets, atol=1e-5, rtol=1e-4
        ):
            return start
    raise LookupError("target block does not appear contiguously in the observation")


def fit_linear_response(
    commanded: Tensor, observed: Tensor, *, minimum_sigma: float = 1e-4
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Per-channel least-squares gain, residual scale, and fit quality."""
    if commanded.shape != observed.shape or commanded.ndim != 2 or commanded.shape[-1] != 6:
        raise ValueError("commanded and observed must both be (samples, 6)")
    alphas, sigmas, r2s = [], [], []
    for channel in range(6):
        x = commanded[:, channel]
        y = observed[:, channel]
        denominator = float(x.square().sum())
        alpha = float((x * y).sum()) / denominator if denominator > 0 else 0.0
        residual = y - alpha * x
        sigma = max(float(residual.std(unbiased=False)), minimum_sigma)
        total = float((y - y.mean()).square().sum())
        r2 = 1.0 - float(residual.square().sum()) / total if total > 0 else 0.0
        alphas.append(alpha)
        sigmas.append(sigma)
        r2s.append(r2)
    return tuple(alphas), tuple(sigmas), tuple(r2s)


def fit_saturating_response(
    commanded: Tensor,
    observed: Tensor,
    *,
    grid: tuple[float, ...] = _SATURATION_GRID,
    minimum_sigma: float = 1e-4,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Per-channel magnitude-dependent gain ``alpha * c / (1 + |c| / c0)``.

    For each channel the saturation constant ``c0`` is chosen from ``grid`` by
    best R2, and ``alpha`` is the closed-form least-squares gain on the resulting
    saturating feature. Returns per-channel ``(alpha, c0, fit_r2)``. A large ``c0``
    recovers the linear model, so this never fits worse than linear on the grid.
    """
    if commanded.shape != observed.shape or commanded.ndim != 2 or commanded.shape[-1] != 6:
        raise ValueError("commanded and observed must both be (samples, 6)")
    alphas, c0s, r2s = [], [], []
    for channel in range(6):
        x = commanded[:, channel]
        y = observed[:, channel]
        total = float((y - y.mean()).square().sum())
        best_alpha, best_c0, best_r2 = 0.0, grid[-1], -math.inf
        for c0 in grid:
            feature = x / (1.0 + x.abs() / c0)
            denominator = float(feature.square().sum())
            alpha = float((feature * y).sum()) / denominator if denominator > 0 else 0.0
            residual = y - alpha * feature
            r2 = 1.0 - float(residual.square().sum()) / total if total > 0 else 0.0
            if r2 > best_r2:
                best_alpha, best_c0, best_r2 = alpha, c0, r2
        _ = minimum_sigma  # sigma is shared with the linear fit; kept for symmetry
        alphas.append(best_alpha)
        c0s.append(best_c0)
        r2s.append(best_r2)
    return tuple(alphas), tuple(c0s), tuple(r2s)


def fit_gripper_response(
    commanded: Tensor, observed: Tensor, *, minimum_sigma: float = 1e-4
) -> tuple[float, float, float]:
    """Least-squares gain, residual scale, and R2 for the gripper channel.

    ``commanded`` and ``observed`` are one-dimensional: the gripper command in
    ``[-1, 1]`` and the observed finger-qpos delta it produced.
    """
    if commanded.shape != observed.shape or commanded.ndim != 1:
        raise ValueError("commanded and observed must both be one-dimensional")
    denominator = float(commanded.square().sum())
    alpha = float((commanded * observed).sum()) / denominator if denominator > 0 else 0.0
    residual = observed - alpha * commanded
    sigma = max(float(residual.std(unbiased=False)), minimum_sigma)
    total = float((observed - observed.mean()).square().sum())
    r2 = 1.0 - float(residual.square().sum()) / total if total > 0 else 0.0
    return alpha, sigma, r2


def calibrate_response(
    probe: SupportsPoseProbe,
    *,
    task: str,
    steps: int,
    calibrate_gripper: bool = False,
    magnitude_gain: bool = False,
) -> ResponseCalibration:
    """Measure slice offsets and the tracking model from unwrapped rollouts.

    ``calibrate_gripper`` additionally locates the gripper finger block and fits
    the command-to-finger-delta model (requires a :class:`SupportsGripperProbe`).
    ``magnitude_gain`` additionally fits the saturating pose gain and records it as
    ``gain_model == "saturating"``. Either flag lifts the calibration to version 2;
    both extensions are contract-independent task knowledge measured here.
    """
    if steps < 8:
        raise ValueError("calibration requires at least eight probe steps")
    if calibrate_gripper and not isinstance(probe, SupportsGripperProbe):
        raise TypeError("gripper calibration requires a SupportsGripperProbe")
    observations, positions, quaternions, actions = [], [], [], []
    gripper_positions: list[Tensor] = []
    for _ in range(steps):
        observation, position, quaternion, action = probe.step_random()
        observations.append(observation)
        positions.append(position)
        quaternions.append(quaternion)
        actions.append(action)
        if calibrate_gripper:
            assert isinstance(probe, SupportsGripperProbe)
            gripper_positions.append(probe.gripper_positions())
    observation_stack = torch.stack(observations)
    position_stack = torch.stack(positions)
    quaternion_stack = torch.stack(quaternions)
    position_start = locate_contiguous_slice(observation_stack, position_stack)
    quaternion_start = locate_contiguous_slice(observation_stack, quaternion_stack)
    responses = pose_delta(
        position_stack[:-1].flatten(0, 1),
        quaternion_stack[:-1].flatten(0, 1),
        position_stack[1:].flatten(0, 1),
        quaternion_stack[1:].flatten(0, 1),
    )
    commanded = torch.stack(actions)[1:, :, :6].flatten(0, 1)
    alpha, sigma, fit_r2 = fit_linear_response(commanded, responses)

    version = 1
    gain_model = "linear"
    alpha_c0: tuple[float, ...] | None = None
    saturating_fit_r2: tuple[float, ...] | None = None
    if magnitude_gain:
        version = 2
        gain_model = "saturating"
        alpha, alpha_c0, saturating_fit_r2 = fit_saturating_response(commanded, responses)

    gripper_start: int | None = None
    gripper_alpha: float | None = None
    gripper_sigma: float | None = None
    gripper_fit_r2: float | None = None
    if calibrate_gripper:
        version = 2
        gripper_stack = torch.stack(gripper_positions)
        gripper_start = locate_contiguous_slice(observation_stack, gripper_stack)
        finger_delta = (gripper_stack[1:] - gripper_stack[:-1]).mean(dim=-1).flatten(0, 1)
        gripper_command = torch.stack(actions)[1:, :, 6].flatten(0, 1)
        gripper_alpha, gripper_sigma, gripper_fit_r2 = fit_gripper_response(
            gripper_command, finger_delta
        )

    return ResponseCalibration(
        task=task,
        position_start=position_start,
        quaternion_start=quaternion_start,
        alpha=alpha,
        sigma=sigma,
        fit_r2=fit_r2,
        steps=steps,
        version=version,
        gain_model=gain_model,
        alpha_c0=alpha_c0,
        saturating_fit_r2=saturating_fit_r2,
        gripper_start=gripper_start,
        gripper_alpha=gripper_alpha,
        gripper_sigma=gripper_sigma,
        gripper_fit_r2=gripper_fit_r2,
    )


def response_from_observations(
    calibration: ResponseCalibration, previous_observation: Tensor, observation: Tensor
) -> Tensor:
    """Extract the observed response from consecutive flat observations.

    Returns six pose channels for a version-1 calibration; a version-2 calibration
    that located the gripper block appends a seventh channel: the mean finger-qpos
    delta (the evidence that identifies ``gripper_inverted``).
    """
    p = calibration.position_start
    q = calibration.quaternion_start
    pose = pose_delta(
        previous_observation[..., p : p + 3],
        previous_observation[..., q : q + 4],
        observation[..., p : p + 3],
        observation[..., q : q + 4],
    )
    if calibration.gripper_start is None:
        return pose
    g = calibration.gripper_start
    previous_finger = previous_observation[..., g : g + _GRIPPER_WIDTH].mean(dim=-1)
    finger = observation[..., g : g + _GRIPPER_WIDTH].mean(dim=-1)
    gripper_delta = (finger - previous_finger).unsqueeze(-1)
    return torch.cat((pose, gripper_delta), dim=-1)
