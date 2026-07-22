"""Exact finite-hypothesis contract belief driven by wrapper-faithful replicas.

Each hypothesis replays the exact ``HiddenContractWrapper`` pipeline (decode with
identity end-effector rotation, then stateful lag) through its own
``CompleteActionDecoder`` clone, so predicted canonical commands stay bit-faithful
to what the wrapper would have executed had that hypothesis been true.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import CompleteActionDecoder, encode_complete_action
from actionshift.contracts.types import ActionContract


def identity_rotation(batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3)


def resolve_rotation(
    ee_rotation: Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a ``(batch_size, 3, 3)`` rotation for the belief replicas.

    ``None`` reproduces the version-1 identity-rotation approximation exactly; a
    supplied rotation (the live tcp rotation of the v2 real-rotation variant) is
    cast, and a single ``(3, 3)`` matrix is broadcast across the batch.
    """
    if ee_rotation is None:
        return identity_rotation(batch_size, device, dtype)
    rotation = ee_rotation.to(device=device, dtype=dtype)
    if rotation.shape == (3, 3):
        rotation = rotation.expand(batch_size, 3, 3)
    if rotation.shape != (batch_size, 3, 3):
        raise ValueError("ee_rotation must be (3, 3) or (batch_size, 3, 3)")
    return rotation


class HypothesisSimulator:
    """Per-hypothesis stateful replicas of the hidden wrapper's action pipeline."""

    def __init__(self, contracts: tuple[ActionContract, ...], *, batch_size: int) -> None:
        if not contracts:
            raise ValueError("at least one hypothesis contract is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.contracts = contracts
        self.batch_size = batch_size
        self._decoders = tuple(
            CompleteActionDecoder(contract, batch_size=batch_size) for contract in contracts
        )

    def tracked_target(self, hypothesis_index: int, reference: Tensor) -> Tensor:
        target = self._decoders[hypothesis_index].tracked_target
        if target is None:
            return torch.zeros(
                (self.batch_size, 6), device=reference.device, dtype=reference.dtype
            )
        return target

    def step(
        self,
        raw_action: Tensor,
        *,
        reset_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> Tensor:
        """Advance every hypothesis with the raw action the wrapper received.

        ``ee_rotation`` (the live tcp rotation for the v2 real-rotation variant)
        makes each replica's decode bit-faithful to the wrapper under a non-identity
        end-effector frame; ``None`` keeps the version-1 identity approximation.
        """
        if raw_action.shape != (self.batch_size, 7):
            raise ValueError("raw_action must be (batch_size, 7)")
        rotation = resolve_rotation(
            ee_rotation,
            batch_size=self.batch_size,
            device=raw_action.device,
            dtype=raw_action.dtype,
        )
        return torch.stack(
            [
                decoder.step(raw_action, ee_rotation=rotation, reset_mask=reset_mask)
                for decoder in self._decoders
            ]
        )


class ExactBeliefDriver:
    """Per-environment exact Bayesian belief over a declared finite contract pool."""

    def __init__(
        self,
        contracts: tuple[ActionContract, ...],
        *,
        batch_size: int,
        response: ResponseModel,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        persist_across_episodes: bool = False,
    ) -> None:
        self.simulator = HypothesisSimulator(contracts, batch_size=batch_size)
        self.response = response
        self.persist_across_episodes = persist_across_episodes
        self._device = torch.device(device)
        self._dtype = dtype
        count = torch.tensor(float(len(contracts)), device=self._device, dtype=dtype)
        self.log_probabilities = (-torch.log(count)).repeat(batch_size, len(contracts))

    @property
    def contracts(self) -> tuple[ActionContract, ...]:
        return self.simulator.contracts

    @property
    def batch_size(self) -> int:
        return self.simulator.batch_size

    def update(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        """Fold one observed transition into the belief.

        ``reset_mask`` mirrors the wrapper's *pending* per-environment reset (it
        applies one step after an episode boundary) and resets the hypothesis
        replicas exactly as the wrapper resets its decoder. ``invalid_mask``
        marks environments whose transition this step crosses an auto-reset
        boundary: their evidence is discarded and (unless persistence is
        requested) their belief returns to uniform for the new episode.
        """
        for name, mask in (("reset_mask", reset_mask), ("invalid_mask", invalid_mask)):
            if mask is not None and mask.shape != (self.batch_size,):
                raise ValueError(f"{name} must contain one value per environment")
        predicted = self.simulator.step(
            raw_action, reset_mask=reset_mask, ee_rotation=ee_rotation
        )
        predicted = predicted.to(device=self._device, dtype=self._dtype)
        observed = observed_response.to(device=self._device, dtype=self._dtype)
        pose_observed = observed[..., :6]
        log_likelihood = self.response.log_likelihood(predicted, pose_observed)
        if self.response.has_gripper:
            if observed.shape[-1] < 7:
                raise ValueError("gripper-calibrated belief needs a seven-channel response")
            log_likelihood = log_likelihood + self.response.gripper_log_likelihood(
                predicted[..., 6], observed[..., 6]
            )
        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._device, dtype=torch.bool).unsqueeze(-1)
            log_likelihood = torch.where(
                boundary, torch.zeros_like(log_likelihood), log_likelihood
            )
            if not self.persist_across_episodes:
                uniform = torch.full_like(
                    self.log_probabilities, -math.log(len(self.contracts))
                )
                self.log_probabilities = torch.where(
                    boundary, uniform, self.log_probabilities
                )
        unnormalized = self.log_probabilities + log_likelihood
        self.log_probabilities = unnormalized - torch.logsumexp(
            unnormalized, dim=1, keepdim=True
        )

    def map_indices(self) -> Tensor:
        return self.log_probabilities.argmax(dim=1)

    def map_encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        """Encode a canonical command under each environment's MAP hypothesis.

        ``ee_rotation`` must be the current tcp rotation the wrapper will decode
        against (v2); ``None`` keeps the version-1 identity behaviour.
        """
        if canonical_action.shape != (self.batch_size, 7):
            raise ValueError("canonical_action must be (batch_size, 7)")
        rotation = resolve_rotation(
            ee_rotation,
            batch_size=self.batch_size,
            device=canonical_action.device,
            dtype=canonical_action.dtype,
        )
        encoded = torch.empty_like(canonical_action)
        map_indices = self.map_indices()
        for hypothesis_index in torch.unique(map_indices).tolist():
            rows = (map_indices == hypothesis_index).nonzero(as_tuple=True)[0]
            rows = rows.to(canonical_action.device)
            target = self.simulator.tracked_target(int(hypothesis_index), canonical_action)
            encoded[rows] = encode_complete_action(
                canonical_action[rows],
                self.contracts[int(hypothesis_index)],
                ee_rotation=rotation[rows],
                tracked_target=target[rows].to(
                    device=canonical_action.device, dtype=canonical_action.dtype
                ),
            )
        return encoded
