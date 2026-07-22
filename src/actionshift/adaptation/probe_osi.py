"""Probe-augmented learned identification: the first unprivileged method that works.

The recurrent negative (``reports/adaptation_recurrent.md``) pinned the wall
precisely: on the real ActionShift response model, passive learned identification
is INFORMATION-limited, not horizon-limited. Smooth policy actions are weak
excitation (per-step SNR ~1), so the continuous permutation/sign map never
resolves no matter how long the episode accumulates. The belief family clears the
same task only because bounded active probing -- six fixed basis pulses of
amplitude 0.5 -- injects strong, structured excitation: a basis pulse isolates one
raw channel per step, so the per-channel response is nearly a direct read of the
contract column.

This module gives the learned identifier that same excitation. It reuses the
proven episode-length machinery unchanged -- ``RunningLagFeatures`` (so the strong
probe evidence at the episode start is retained for the whole episode, never
rolled out of a window) and the equivariant ``RecurrentOsiRegressor`` (equivariant
heads for the continuous map, a GRU for the discrete flags). The ONLY change from
the recurrent negative is the excitation: the adapter spends the first ``budget``
steps sending the SAME fixed probe schedule the probe family uses, and the model
is trained on probe-excited transitions. That isolates the probe excitation as the
single lever, so the identification-by-field deltas measure exactly what bounded
active probing buys a learned method.

Privilege. What this method assumes: (1) a bounded probe budget -- a privilege
SHARED with the probe family (bounded active probing), (2) the contract-independent
response calibration shared by every tournament method, and (3) a model trained on
hash-disjoint contracts. It does NOT use the pool and does NOT use grammar
knowledge: the model regresses the continuous contract parameters, it never scores
a declared candidate set. The probe schedule is the agent's own known excitation,
so reading it back out of the accumulated evidence is legitimate self-knowledge,
not privilege. A test asserts the constructor takes no contract/pool argument.
"""

from __future__ import annotations

import torch
from torch import Tensor

from actionshift.adaptation.probes import fixed_probe_pulse
from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiRegressor,
    RunningLagFeatures,
)
from actionshift.adaptation.training import decode_prediction
from actionshift.contracts.types import ActionContract

_DEFAULT_BUDGET = 6
_DEFAULT_AMPLITUDE = 0.5
_DEFAULT_WARMUP = 8
_DEFAULT_MIN_SAMPLES = 6


class ProbeOsiAdapter:
    """Eval-time probe-augmented learned adapter (``ContractAdapter`` protocol).

    A bounded per-episode probe phase (``budget`` fixed basis pulses, identical to
    ``adaptation.probes.fixed_probe_pulse``) drives strong excitation into the
    running least-squares accumulator; the trained recurrent identifier then reads
    the accumulated maps and the estimate refines every step. After the warmup the
    canonical command is encoded under the current per-environment MAP estimate;
    during the probe phase and warmup it passes through (the belief probe family
    likewise controls only after its probe budget).

    Unprivileged: it observes only its own raw actions and calibrated responses,
    never the true contract or a pool.
    """

    name = "probe_osi"

    def __init__(
        self,
        model: RecurrentOsiRegressor,
        *,
        batch_size: int,
        budget: int = _DEFAULT_BUDGET,
        amplitude: float = _DEFAULT_AMPLITUDE,
        warmup: int = _DEFAULT_WARMUP,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        self.model = model.eval()
        self.batch_size = batch_size
        self.budget = budget
        self.amplitude = amplitude
        self.warmup = warmup
        self._device = torch.device(device)
        self._accumulator = RunningLagFeatures(
            batch_size, device="cpu", min_samples=min_samples
        )
        self._hidden = model.initial_hidden(
            batch_size, device=self._device, dtype=torch.float32
        )
        self._steps = torch.zeros(batch_size, dtype=torch.long)
        self._filled = torch.zeros(batch_size, dtype=torch.long)
        self._estimates: list[ActionContract | None] = [None] * batch_size
        self._tracked_target: Tensor | None = None
        self.last_probe_mask: Tensor | None = None

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        from actionshift.adaptation.hypotheses import identity_rotation
        from actionshift.contracts.transforms import encode_complete_action

        del ee_rotation  # learned identifier does not model the end-effector frame
        if canonical_action.shape != (self.batch_size, 7):
            raise ValueError("canonical_action must be (batch_size, 7)")
        if self._tracked_target is None:
            self._tracked_target = torch.zeros(
                (self.batch_size, 6),
                device=canonical_action.device,
                dtype=canonical_action.dtype,
            )
        probing = self._steps < self.budget
        probe = fixed_probe_pulse(self._steps, amplitude=self.amplitude).to(
            device=canonical_action.device, dtype=canonical_action.dtype
        )
        encoded = canonical_action.clone()
        rotation = identity_rotation(1, canonical_action.device, canonical_action.dtype)
        for row in range(self.batch_size):
            if bool(probing[row]):
                continue
            estimate = self._estimates[row]
            if estimate is None or int(self._filled[row]) < self.warmup:
                continue
            encoded[row : row + 1] = encode_complete_action(
                canonical_action[row : row + 1],
                estimate,
                ee_rotation=rotation,
                tracked_target=self._tracked_target[row : row + 1],
            )
            if estimate.target == "absolute":
                self._tracked_target[row] = (
                    self._tracked_target[row] + canonical_action[row, :6]
                )
        raw = torch.where(
            probing.to(canonical_action.device).unsqueeze(-1), probe, encoded
        )
        self.last_probe_mask = probing.clone()
        self._steps = self._steps + 1
        return raw

    def observe(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        del ee_rotation  # learned identifier does not model the end-effector frame
        if invalid_mask is not None:
            boundary = invalid_mask.detach().cpu().to(torch.bool)
            if boundary.any():
                self._accumulator.reset(boundary)
                self._hidden[:, boundary.to(self._hidden.device)] = 0.0
                self._filled = torch.where(
                    boundary, torch.zeros_like(self._filled), self._filled
                )
                self._steps = torch.where(
                    boundary, torch.zeros_like(self._steps), self._steps
                )
                if self._tracked_target is not None:
                    self._tracked_target[
                        boundary.to(self._tracked_target.device)
                    ] = 0.0
                for row in boundary.nonzero(as_tuple=True)[0].tolist():
                    self._estimates[row] = None
            valid = ~boundary
        else:
            valid = torch.ones(self.batch_size, dtype=torch.bool)
        raw = raw_action.detach().cpu().to(torch.float32)
        response = observed_response.detach().cpu().to(torch.float32)
        features = self._accumulator.push(raw, response, active=valid)
        self._filled = torch.where(valid, self._filled + 1, self._filled)
        previous_hidden = self._hidden
        with torch.no_grad():
            predictions, stepped_hidden = self.model.step(
                features.to(self._device), self._hidden
            )
        valid_device = valid.to(self._hidden.device)
        self._hidden = torch.where(
            valid_device[None, :, None], stepped_hidden, previous_hidden
        )
        for row in valid.nonzero(as_tuple=True)[0].tolist():
            single = {key: value[row] for key, value in predictions.items()}
            self._estimates[row] = decode_prediction(single)


__all__ = ["ProbeOsiAdapter"]
