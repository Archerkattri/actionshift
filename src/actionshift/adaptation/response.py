"""Observed-response likelihoods for hidden-contract belief updates.

The wrapper decodes a raw policy action into a canonical command and hands it to
the underlying controller. The only evidence an unprivileged agent may condition
on is the observable response (for ManiSkill state tasks, the tcp pose delta).
The likelihood compares that response against each hypothesis contract's
predicted canonical command through a calibrated linear tracking model
``observed ~ alpha * predicted + noise(sigma)``. ``alpha`` and ``sigma`` are
contract-independent task knowledge calibrated on the unwrapped environment and
may be per-channel because translation and rotation track differently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor


def _channel_vector(
    value: tuple[float, ...] | float,
    channels: int,
    name: str,
    *,
    allow_negative: bool = False,
) -> tuple[float, ...]:
    vector = (value,) * channels if isinstance(value, float | int) else tuple(value)
    if len(vector) != channels:
        raise ValueError(f"{name} must be scalar or provide one value per channel")
    if any(not math.isfinite(item) or item == 0 for item in vector):
        raise ValueError(f"{name} entries must be finite and nonzero")
    if not allow_negative and any(item < 0 for item in vector):
        raise ValueError(f"{name} entries must be positive")
    return vector


@dataclass(frozen=True, slots=True)
class ResponseModel:
    """Calibrated Gaussian response model over the leading canonical channels.

    The expected response for a predicted canonical command ``c`` is ``alpha * c``
    by default. When ``alpha_c0`` is supplied the model uses the magnitude-dependent
    (saturating) gain ``alpha * c / (1 + |c| / c0)``, capturing mild PD
    under-tracking of large commands; this is opt-in so the linear model stays the
    reproducible default. When ``gripper_alpha``/``gripper_sigma`` are supplied the
    model additionally scores a gripper evidence channel (finger-qpos delta versus
    the predicted canonical gripper command), which lets a belief identify
    ``gripper_inverted`` that the pose channels cannot reveal.
    """

    alpha: tuple[float, ...] | float
    sigma: tuple[float, ...] | float
    channels: int = 6
    alpha_c0: tuple[float, ...] | float | None = None
    gripper_alpha: float | None = None
    gripper_sigma: float | None = None

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        _channel_vector(self.alpha, self.channels, "alpha", allow_negative=True)
        _channel_vector(self.sigma, self.channels, "sigma")
        if self.alpha_c0 is not None:
            _channel_vector(self.alpha_c0, self.channels, "alpha_c0")
        if (self.gripper_alpha is None) != (self.gripper_sigma is None):
            raise ValueError("gripper_alpha and gripper_sigma must be provided together")
        if self.gripper_alpha is not None and (
            not math.isfinite(self.gripper_alpha) or self.gripper_alpha == 0
        ):
            raise ValueError("gripper_alpha must be finite and nonzero")
        if self.gripper_sigma is not None and (
            not math.isfinite(self.gripper_sigma) or self.gripper_sigma <= 0
        ):
            raise ValueError("gripper_sigma must be finite and positive")

    @property
    def has_gripper(self) -> bool:
        return self.gripper_alpha is not None

    def expected(self, predicted: Tensor) -> Tensor:
        """Map predicted canonical pose commands to their expected response."""
        alpha = predicted.new_tensor(
            _channel_vector(self.alpha, self.channels, "alpha", allow_negative=True)
        )
        scaled = alpha * predicted[..., : self.channels]
        if self.alpha_c0 is None:
            return scaled
        c0 = predicted.new_tensor(_channel_vector(self.alpha_c0, self.channels, "alpha_c0"))
        return scaled / (1.0 + predicted[..., : self.channels].abs() / c0)

    def log_likelihood(self, predicted: Tensor, observed: Tensor) -> Tensor:
        """Score hypotheses against one observed pose response.

        ``predicted`` stacks per-hypothesis canonical commands as (H, B, C');
        ``observed`` is (B, C) with ``C == channels <= C'``. Returns (B, H).
        """
        if predicted.ndim != 3:
            raise ValueError("predicted must be (hypotheses, batch, channels)")
        if observed.ndim != 2 or observed.shape[0] != predicted.shape[1]:
            raise ValueError("observed must be (batch, channels) matching predicted")
        if self.channels > predicted.shape[-1] or observed.shape[-1] != self.channels:
            raise ValueError("observed channel count must equal the model channels")
        sigma = observed.new_tensor(_channel_vector(self.sigma, self.channels, "sigma"))
        residual = observed.unsqueeze(0) - self.expected(predicted)
        standardized = residual / sigma
        return (-0.5 * standardized.square().sum(dim=-1)).transpose(0, 1)

    def gripper_log_likelihood(
        self, predicted_gripper: Tensor, observed_gripper: Tensor
    ) -> Tensor:
        """Score the gripper channel; ``predicted_gripper`` is (H, B), obs is (B,).

        The predicted canonical gripper command already carries each hypothesis's
        ``gripper_inverted`` sign, so the residual against the calibrated
        finger-delta gain distinguishes inverting from non-inverting contracts.
        Returns (B, H).
        """
        if self.gripper_alpha is None or self.gripper_sigma is None:
            raise ValueError("gripper likelihood requires a gripper-calibrated model")
        if predicted_gripper.ndim != 2:
            raise ValueError("predicted_gripper must be (hypotheses, batch)")
        if observed_gripper.ndim != 1 or observed_gripper.shape[0] != predicted_gripper.shape[1]:
            raise ValueError("observed_gripper must be (batch,) matching predicted")
        residual = observed_gripper.unsqueeze(0) - self.gripper_alpha * predicted_gripper
        standardized = residual / self.gripper_sigma
        return (-0.5 * standardized.square()).transpose(0, 1)
