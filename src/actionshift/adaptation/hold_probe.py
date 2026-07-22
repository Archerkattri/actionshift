"""Absolute-specific hold-probe excitation for the factorized grammar belief.

The per-step pulse probe (``factorized_grammar.FactorizedGrammarProbingAdapter``)
excites one channel for a single step -- ideal for a *delta* decode, whose response
is the instantaneous drive. It is weak for an *absolute* decode: the belief scores
absolute modes against the **differenced** drive ``raw(t) - raw(t-1)`` (bit-faithful
to ``CompleteActionDecoder``'s zero-initialized cumulative target), which halves an
already attenuated signal, so on the all-absolute ``unseen_composition`` split the
discrete identification (permutation *and* absolute-vs-delta target) collapses
upstream of any scale refinement (see ``reports/adaptation_scale_corrector.md``).

A **held** raw value separates the two targets sharply and restores the absolute
signal:

- **Target discriminator.** Hold one raw channel at ``+amplitude`` for several
  steps. A delta decode *integrates* the hold -- the tcp keeps moving every step --
  while an absolute decode *holds position*: it jumps once (the first step) then the
  response decays to zero. Scored against each mode's integrated drive, the delta
  mode predicts ``len * command`` while the absolute mode predicts a single net
  excursion, two predictions that diverge with the hold length.
- **Permutation SNR.** Summing the observed response across a held window telescopes
  the differenced absolute drive back to the full undifferenced excursion
  (``sum_t (x_t - x_{t-1}) = x_T - x_0``), restoring the signal the per-step
  differenced score halves, so the raw channel feeding each semantic channel becomes
  identifiable from a strong integrated observable rather than a weak per-step one.

This module contributes only the *schedule* and the *probing adapter*; the telescoped
evidence itself is folded by ``FactorizedGrammarDriver.update`` under a ``hold_mask``
(additive, off by default). It is contract-independent task knowledge: the schedule is
a fixed sequence of bounded raw excitations and never sees the true contract.
"""

from __future__ import annotations

import torch
from torch import Tensor

from actionshift.adaptation.factorized_grammar import FactorizedGrammarDriver

_ACTION_WIDTH = 7
_POSE_CHANNELS = 6


class HoldProbeSchedule:
    """A fixed sweep of sustained per-channel holds.

    Each pose channel is held at ``+amplitude`` for ``hold_steps`` consecutive steps;
    the sweep over all ``channels`` repeats ``rounds`` times. Every hold's first step
    is the excursion (``0 -> amplitude``) and its last step closes the integration
    window, so the accumulated observable telescopes to the net excursion for an
    absolute decode and to ``hold_steps`` times the command for a delta decode.
    """

    def __init__(
        self,
        *,
        amplitude: float = 0.5,
        hold_steps: int = 4,
        rounds: int = 1,
        channels: int = _POSE_CHANNELS,
    ) -> None:
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if hold_steps < 2:
            raise ValueError("hold_steps must be at least two (excursion + decay)")
        if rounds < 1:
            raise ValueError("rounds must be at least one")
        if channels <= 0 or channels > _POSE_CHANNELS:
            raise ValueError("channels must be in 1..6")
        self.amplitude = amplitude
        self.hold_steps = hold_steps
        self.rounds = rounds
        self.channels = channels

    @property
    def total_steps(self) -> int:
        return self.rounds * self.channels * self.hold_steps

    @property
    def version(self) -> str:
        return f"holdv1-a{self.amplitude:g}-h{self.hold_steps}-r{self.rounds}-c{self.channels}"

    def action(self, step_index: Tensor) -> Tensor:
        """Raw (batch, 7) action holding the active channel at ``+amplitude``."""
        batch = step_index.shape[0]
        action = step_index.new_zeros((batch, _ACTION_WIDTH), dtype=torch.float32)
        channel = ((step_index // self.hold_steps) % self.channels).long()
        action[torch.arange(batch), channel] = self.amplitude
        return action

    def is_window_close(self, step_index: Tensor) -> Tensor:
        """Per-environment mask: does this step end a hold window?"""
        return (step_index % self.hold_steps) == (self.hold_steps - 1)


class FactorizedGrammarHoldProbingAdapter:
    """Factorized-grammar belief with an absolute-specific hold-probe phase.

    For the first ``schedule.total_steps`` steps of every episode the adapter sends
    the hold schedule instead of task actions, routing each held step's evidence into
    the driver's telescoped-window accumulator (``hold_mask``) and folding one
    integrated Gaussian term per window (``window_close``). It withholds cumulative
    intent from the tracked absolute target while probing (the sent action is a probe,
    not the intent) and, when the probe phase ends, re-anchors the tracked target to
    the achieved tcp displacement, so control resumes from the true achieved pose with
    the discrete contract identified. The scale corrector (if enabled) refines the
    continuous scale downstream on the control steps, which alone feed it.
    """

    def __init__(
        self,
        driver: FactorizedGrammarDriver,
        *,
        schedule: HoldProbeSchedule,
    ) -> None:
        self.name = "factorized_grammar_hold_probes"
        self.driver = driver
        self.schedule = schedule
        self._steps = torch.zeros(driver.batch_size, dtype=torch.long)
        self._exec_steps = torch.zeros(driver.batch_size, dtype=torch.long)
        self.last_probe_mask: Tensor | None = None

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        del ee_rotation  # frame collapsed to base; tcp rotation not consumed
        device = canonical_action.device
        steps = self._steps.to(device)
        total = self.schedule.total_steps
        probing = steps < total
        transition = steps == total
        if bool(transition.any()):
            # The probe phase just ended for these environments: re-anchor the tracked
            # absolute target to the achieved displacement before the first control encode.
            self.driver.anchor_tracked_target(transition)
        control_mask = ~probing
        task_action = self.driver.map_encode(canonical_action, target_mask=control_mask)
        probe_action = self.schedule.action(steps).to(device=device, dtype=canonical_action.dtype)
        raw = torch.where(probing.unsqueeze(-1), probe_action, task_action)
        self.last_probe_mask = probing
        self._exec_steps = self._steps.clone()
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
        del ee_rotation  # frame collapsed to base; tcp rotation not consumed
        device = raw_action.device
        if self.last_probe_mask is None:
            probing = torch.zeros(self.driver.batch_size, dtype=torch.bool, device=device)
        else:
            probing = self.last_probe_mask.to(device=device, dtype=torch.bool)
        window_close = self.schedule.is_window_close(self._exec_steps.to(device)) & probing
        self.driver.update(
            raw_action,
            observed_response,
            reset_mask=reset_mask,
            invalid_mask=invalid_mask,
            corrector_mask=~probing,
            hold_mask=probing,
            window_close=window_close,
        )
        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._steps.device, dtype=torch.bool)
            self._steps = torch.where(boundary, torch.zeros_like(self._steps), self._steps)


__all__ = ["FactorizedGrammarHoldProbingAdapter", "HoldProbeSchedule"]
