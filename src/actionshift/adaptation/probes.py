"""Bounded active probing on top of the exact belief driver.

A probing adapter spends the first ``budget`` steps of every episode sending
bounded raw-space probe actions instead of task actions, folds the observed
responses into the belief exactly like passive evidence, and then controls with
the per-environment MAP encoding. Probe selection strategies:

- ``fixed``: alternating-sign raw basis pulses cycling through the pose channels;
- ``random``: uniform bounded raw pulses;
- ``entropy``: greedy expected-posterior-entropy minimization over a candidate
  pulse set, using a stateless instantaneous preview of each hypothesis's
  response (lag and target statefulness are ignored for *selection only*; the
  belief update itself stays exact).

Probes never see the true contract; amplitude stays inside the declared bound.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from actionshift.adaptation.hypotheses import ExactBeliefDriver, resolve_rotation
from actionshift.contracts.transforms import decode_complete_action

ProbeStrategy = Literal["fixed", "random", "entropy"]


def fixed_probe_pulse(
    step_index: Tensor, *, amplitude: float, channels: int = 6
) -> Tensor:
    """Alternating-sign basis pulse for each environment's probe step index."""
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    batch = step_index.shape[0]
    action = step_index.new_zeros((batch, 7), dtype=torch.float32)
    channel = (step_index % channels).long()
    sign = torch.where((step_index // channels) % 2 == 0, 1.0, -1.0)
    action[torch.arange(batch), channel] = amplitude * sign
    return action


class ProbingBeliefAdapter:
    """Exact-belief adapter with a bounded per-episode active probe phase."""

    def __init__(
        self,
        driver: ExactBeliefDriver,
        *,
        strategy: ProbeStrategy,
        budget: int,
        amplitude: float = 0.5,
        seed: int = 0,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if strategy not in ("fixed", "random", "entropy"):
            raise ValueError(f"unknown probe strategy: {strategy}")
        self.name = f"{strategy}_probes"
        self.driver = driver
        self.strategy: ProbeStrategy = strategy
        self.budget = budget
        self.amplitude = amplitude
        self._generator = torch.Generator().manual_seed(seed)
        self._steps = torch.zeros(driver.batch_size, dtype=torch.long)
        self.last_probe_mask: Tensor | None = None

    def _candidate_pulses(self, reference: Tensor) -> Tensor:
        eye = torch.eye(6, device=reference.device, dtype=reference.dtype)
        pulses = torch.cat((eye, -eye), dim=0) * self.amplitude
        return torch.cat(
            (pulses, torch.zeros((12, 1), device=reference.device, dtype=reference.dtype)),
            dim=-1,
        )

    def _entropy_probe(
        self, reference: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        """Greedy expected-entropy minimization with a stateless response preview."""
        candidates = self._candidate_pulses(reference)
        belief = self.driver.log_probabilities.exp()
        best_entropy: Tensor | None = None
        best_action = candidates[0].expand(self.driver.batch_size, 7).clone()
        for candidate in candidates:
            raw = candidate.expand(self.driver.batch_size, 7)
            predicted = self._preview(raw, ee_rotation=ee_rotation)
            expected = self._expected_posterior_entropy(predicted, belief)
            if best_entropy is None:
                best_entropy = expected
                best_action = raw.clone()
            else:
                better = expected < best_entropy
                best_entropy = torch.where(better, expected, best_entropy)
                best_action = torch.where(better.unsqueeze(-1), raw, best_action)
        return best_action

    def _preview(
        self, raw_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        """Instantaneous per-hypothesis response preview without state mutation."""
        rotation = resolve_rotation(
            ee_rotation,
            batch_size=self.driver.batch_size,
            device=raw_action.device,
            dtype=raw_action.dtype,
        )
        outputs = []
        for index, contract in enumerate(self.driver.contracts):
            target = self.driver.simulator.tracked_target(index, raw_action)
            canonical, _ = decode_complete_action(
                raw_action,
                contract,
                ee_rotation=rotation,
                tracked_target=target.to(
                    device=raw_action.device, dtype=raw_action.dtype
                ),
            )
            outputs.append(canonical)
        return torch.stack(outputs)

    def _expected_posterior_entropy(self, predicted: Tensor, belief: Tensor) -> Tensor:
        """E_h[H(posterior | mean response of h)] per environment."""
        response = self.driver.response
        entropies = []
        for hypothesis in range(predicted.shape[0]):
            observed = response.log_likelihood(
                predicted, predicted[hypothesis, :, : response.channels]
            )
            posterior = torch.log_softmax(
                self.driver.log_probabilities + observed, dim=1
            )
            entropies.append(-(posterior.exp() * posterior).sum(dim=1))
        stacked = torch.stack(entropies, dim=1)
        return (belief * stacked).sum(dim=1)

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        probing = (self._steps < self.budget).to(canonical_action.device)
        task_action = self.driver.map_encode(canonical_action, ee_rotation=ee_rotation)
        if self.budget == 0 or not bool(probing.any()):
            self.last_probe_mask = probing
            self._steps = self._steps + 1
            return task_action
        if self.strategy == "fixed":
            probe = fixed_probe_pulse(self._steps, amplitude=self.amplitude).to(
                device=canonical_action.device, dtype=canonical_action.dtype
            )
        elif self.strategy == "random":
            sample = (
                torch.rand((self.driver.batch_size, 7), generator=self._generator) * 2.0
                - 1.0
            ) * self.amplitude
            sample[:, 6] = 0.0
            probe = sample.to(device=canonical_action.device, dtype=canonical_action.dtype)
        else:
            probe = self._entropy_probe(canonical_action, ee_rotation=ee_rotation)
        raw = torch.where(probing.unsqueeze(-1), probe, task_action)
        self.last_probe_mask = probing
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
        self.driver.update(
            raw_action,
            observed_response,
            reset_mask=reset_mask,
            invalid_mask=invalid_mask,
            ee_rotation=ee_rotation,
        )
        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._steps.device, dtype=torch.bool)
            self._steps = torch.where(
                boundary, torch.zeros_like(self._steps), self._steps
            )
