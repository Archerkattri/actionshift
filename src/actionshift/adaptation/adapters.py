"""Interface adapters between a frozen canonical policy and the hidden wrapper.

Every tournament method shares the same frozen task backbone; a method is an
adapter that turns the backbone's canonical command into the raw action sent to
the hidden-contract environment. Adapters never receive the true contract; the
oracle ceiling takes it explicitly and is labeled privileged, and the exact
belief adapter's only privilege is knowledge of the declared finite pool.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from actionshift.adaptation.hypotheses import ExactBeliefDriver, resolve_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import encode_complete_action
from actionshift.contracts.types import ActionContract


class ContractAdapter(Protocol):
    """Encode canonical commands and learn from observed responses."""

    name: str

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        """Map a (batch, 7) canonical command to the raw wrapper action.

        ``ee_rotation`` is the current tcp rotation the hidden wrapper will decode
        against. ``None`` (the default) reproduces the identity-rotation variant
        (v1); a supplied rotation drives the v2 real-rotation variant.
        """
        ...

    def observe(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        """Fold the transition produced by ``raw_action`` into internal state.

        ``ee_rotation`` is the tcp rotation the wrapper decoded ``raw_action``
        against (the pre-step tcp frame); ``None`` keeps the v1 identity variant.
        """
        ...


class NoAdaptAdapter:
    """Send the canonical command unchanged; the no-adaptation baseline."""

    name = "no_adapt"

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        return canonical_action

    def observe(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        return None


class OracleAdapter:
    """Privileged ceiling: encode with the true contract, mirroring Gate 1."""

    name = "oracle"

    def __init__(self, true_contract: ActionContract, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.contract = true_contract
        self.batch_size = batch_size
        self._tracked_target: Tensor | None = None

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        if canonical_action.shape != (self.batch_size, 7):
            raise ValueError("canonical_action must be (batch_size, 7)")
        if self._tracked_target is None:
            self._tracked_target = torch.zeros(
                (self.batch_size, 6),
                device=canonical_action.device,
                dtype=canonical_action.dtype,
            )
        rotation = resolve_rotation(
            ee_rotation,
            batch_size=self.batch_size,
            device=canonical_action.device,
            dtype=canonical_action.dtype,
        )
        encoded = encode_complete_action(
            canonical_action,
            self.contract,
            ee_rotation=rotation,
            tracked_target=self._tracked_target,
        )
        if self.contract.target == "absolute":
            self._tracked_target = self._tracked_target + canonical_action[..., :6]
        return encoded

    def observe(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        if invalid_mask is not None and self._tracked_target is not None:
            mask = invalid_mask.to(
                device=self._tracked_target.device, dtype=torch.bool
            ).unsqueeze(-1)
            self._tracked_target = torch.where(
                mask, torch.zeros_like(self._tracked_target), self._tracked_target
            )


class ExactBeliefAdapter:
    """Exact Bayes over a declared finite pool; encodes with the per-env MAP."""

    name = "exact_belief"

    def __init__(
        self,
        pool: tuple[ActionContract, ...],
        *,
        batch_size: int,
        response: ResponseModel,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        persist_across_episodes: bool = False,
    ) -> None:
        self.driver = ExactBeliefDriver(
            pool,
            batch_size=batch_size,
            response=response,
            device=device,
            dtype=dtype,
            persist_across_episodes=persist_across_episodes,
        )

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        return self.driver.map_encode(canonical_action, ee_rotation=ee_rotation)

    def observe(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        self.driver.update(
            raw_action,
            observed_response,
            reset_mask=reset_mask,
            invalid_mask=invalid_mask,
            ee_rotation=ee_rotation,
        )
