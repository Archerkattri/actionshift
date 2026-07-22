"""DualABI probe adapter: task-regret-aware active probing with early stopping.

DualABI is the ActionShift project's named experimental method. This adapter wires
it onto the *same* exact-belief driver and probe machinery as
``adaptation.probes.ProbingBeliefAdapter`` (identical declared contract pool,
per-episode budget cap, amplitude bound, gripper-never-probed safety, and belief
update), so every comparison against ``entropy_probes`` / ``fixed_probes`` is
matched-privilege. The *only* difference is how a probe pulse is chosen and when
probing stops.

Premise. Probe where it matters *for the task*, not for pure information. Two
hypotheses that decode the policy's likely next canonical actions to the same
executed action need not be separated: the belief may stay ambiguous about them
because acting on either is task-equivalent. DualABI therefore scores a candidate
pulse by its expected reduction in **task regret** -- the belief-weighted error of
the executed task action when we act on the MAP contract but some other pooled
contract is true -- rather than by expected posterior entropy.

Reuse vs reimplementation (audited, see the module-level constants and methods):

- REUSED verbatim from ``adaptation.probes``: the amplitude-bounded 12-pulse
  candidate set (``_candidate_pulses``), the stateless per-hypothesis response
  preview (``_preview``), and the exact belief update through the shared
  ``ExactBeliefDriver``. This guarantees matched-privilege fairness.
- REUSED from ``methods.dualabi``: ``select_dualabi_candidates`` performs the final
  per-environment scoring (``task_value - safety - terminal + information`` with
  ``information_mode='task_regret'``), so the selection is the project's named
  DualABI selector, not an ad-hoc rule.
- REIMPLEMENTED here: the task-regret *functional* itself. ``methods.dualabi``'s
  ``expected_task_regret`` is a discrete-observation Bayes regret over a supplied
  ``future_utility`` table; the ActionShift task action is continuous, so we
  realize the same "oracle-minus-Bayes action value" idea directly in raw-action
  space as ``decode_j(encode_i(a)) - a`` averaged over recent canonical actions
  ``a``. This is exactly the continuous analogue: the round trip is zero when the
  acted contract equals the true one, and small when they produce the same
  executed action for the actions the policy actually issues.

The task-regret preview shares the ``_preview`` selection approximation (lag and
target statefulness ignored for *selection only*; the belief update stays exact).
"""

from __future__ import annotations

import torch
from torch import Tensor

from actionshift.adaptation.hypotheses import ExactBeliefDriver, resolve_rotation
from actionshift.contracts.transforms import (
    decode_complete_action,
    encode_complete_action,
)
from actionshift.methods.dualabi import select_dualabi_candidates

_POSE_CHANNELS = 6
_ACTION_WIDTH = 7


class DualABIProbeAdapter:
    """Exact-belief adapter with a task-regret-aware, early-stopping probe phase."""

    def __init__(
        self,
        driver: ExactBeliefDriver,
        *,
        budget: int,
        amplitude: float = 0.5,
        regret_threshold: float = 0.05,
        recent_window: int = 8,
        information_weight: float = 1.0,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        if regret_threshold < 0:
            raise ValueError("regret_threshold must be nonnegative")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if information_weight < 0:
            raise ValueError("information_weight must be nonnegative")
        self.name = "dualabi"
        self.driver = driver
        self.budget = budget
        self.amplitude = amplitude
        self.regret_threshold = regret_threshold
        self.recent_window = recent_window
        self.information_weight = information_weight
        batch = driver.batch_size
        self._steps = torch.zeros(batch, dtype=torch.long)
        self._stopped = torch.zeros(batch, dtype=torch.bool)
        self._recent: Tensor | None = None
        self._fresh = torch.ones(batch, dtype=torch.bool)
        self.last_probe_mask: Tensor | None = None
        self.last_task_regret: Tensor | None = None

    # ------------------------------------------------------------------ #
    # Candidate pulses and response preview (mirrors adaptation.probes).  #
    # ------------------------------------------------------------------ #
    def _candidate_pulses(self, reference: Tensor) -> Tensor:
        eye = torch.eye(_POSE_CHANNELS, device=reference.device, dtype=reference.dtype)
        pulses = torch.cat((eye, -eye), dim=0) * self.amplitude
        return torch.cat(
            (
                pulses,
                torch.zeros(
                    (2 * _POSE_CHANNELS, 1),
                    device=reference.device,
                    dtype=reference.dtype,
                ),
            ),
            dim=-1,
        )

    def _preview(
        self, raw_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        """Instantaneous per-hypothesis canonical response, no state mutation."""
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

    def _posterior(self, predicted: Tensor, hypothesis: int) -> Tensor:
        """Posterior belief (batch, H) after observing hypothesis ``h``'s response."""
        response = self.driver.response
        observed = response.log_likelihood(
            predicted, predicted[hypothesis, :, : response.channels]
        )
        return torch.log_softmax(self.driver.log_probabilities + observed, dim=1).exp()

    # ------------------------------------------------------------------ #
    # Task-regret functional (continuous analogue of expected_task_regret) #
    # ------------------------------------------------------------------ #
    def _pairwise_task_error(
        self, recent: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        """Per-env task-action error E[i, j, b] of acting on ``i`` when ``j`` is true.

        E[i, j, b] = mean_k || decode_j(encode_i(a_k)) - a_k ||_pose. The round trip
        is exactly zero on the diagonal, so hypotheses that produce the same executed
        action for the sampled canonical intents contribute no regret.
        """
        contracts = self.driver.contracts
        hypotheses = len(contracts)
        window, batch, _ = recent.shape
        device, dtype = recent.device, recent.dtype
        flat = recent.reshape(window * batch, _ACTION_WIDTH)
        per_env = resolve_rotation(
            ee_rotation, batch_size=batch, device=device, dtype=dtype
        )
        rotation = per_env.unsqueeze(0).expand(window, batch, 3, 3).reshape(
            window * batch, 3, 3
        )
        targets = [
            self.driver.simulator.tracked_target(index, recent)
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(window, batch, _POSE_CHANNELS)
            .reshape(window * batch, _POSE_CHANNELS)
            for index in range(hypotheses)
        ]
        encoded = [
            encode_complete_action(
                flat, contracts[i], ee_rotation=rotation, tracked_target=targets[i]
            )
            for i in range(hypotheses)
        ]
        error = flat.new_zeros((hypotheses, hypotheses, batch))
        pose_reference = flat[..., :_POSE_CHANNELS]
        for i in range(hypotheses):
            for j in range(hypotheses):
                if i == j:
                    continue
                executed, _ = decode_complete_action(
                    encoded[i],
                    contracts[j],
                    ee_rotation=rotation,
                    tracked_target=targets[j],
                )
                deviation = (executed[..., :_POSE_CHANNELS] - pose_reference).norm(dim=-1)
                error[i, j] = deviation.reshape(window, batch).mean(dim=0)
        return error

    def _task_regret(self, belief: Tensor, error: Tensor) -> Tensor:
        """Belief-weighted task regret of acting on the per-env MAP contract.

        R[b] = sum_j belief[b, j] * E[argmax_j' belief[b, j'], j, b].
        """
        map_index = belief.argmax(dim=1)
        batch = belief.shape[0]
        rows = torch.arange(batch, device=belief.device)
        acted = error[map_index, :, rows]
        return (belief * acted).sum(dim=1)

    def _select_probe(
        self,
        canonical_action: Tensor,
        belief: Tensor,
        error: Tensor,
        current_regret: Tensor,
        *,
        ee_rotation: Tensor | None = None,
    ) -> Tensor:
        """Choose the task-regret-reducing pulse per environment via DualABI scoring."""
        candidates = self._candidate_pulses(canonical_action)
        batch = self.driver.batch_size
        expected_future = candidates.new_zeros((candidates.shape[0], batch))
        for candidate_index, candidate in enumerate(candidates):
            raw = candidate.expand(batch, _ACTION_WIDTH)
            predicted = self._preview(raw, ee_rotation=ee_rotation)
            regret_by_hypothesis = candidates.new_zeros((len(self.driver.contracts), batch))
            for hypothesis in range(predicted.shape[0]):
                posterior = self._posterior(predicted, hypothesis)
                regret_by_hypothesis[hypothesis] = self._task_regret(posterior, error)
            expected_future[candidate_index] = (belief * regret_by_hypothesis.transpose(0, 1)).sum(
                dim=1
            )
        zeros = candidates.new_zeros(candidates.shape[0])
        valid = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
        chosen = torch.empty(
            (batch, _ACTION_WIDTH), device=candidates.device, dtype=candidates.dtype
        )
        for environment in range(batch):
            decision = select_dualabi_candidates(
                task_value=zeros,
                safety_risk=zeros,
                current_task_regret=current_regret[environment].reshape(()),
                expected_future_regret=expected_future[:, environment],
                valid_mask=valid,
                safety_weight=0.0,
                information_weight=self.information_weight,
                information_mode="task_regret",
            )
            chosen[environment] = candidates[decision.index]
        return chosen

    # ------------------------------------------------------------------ #
    # ContractAdapter protocol.                                          #
    # ------------------------------------------------------------------ #
    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        if canonical_action.shape != (self.driver.batch_size, _ACTION_WIDTH):
            raise ValueError("canonical_action must be (batch_size, 7)")
        self._push_recent(canonical_action)
        task_action = self.driver.map_encode(canonical_action, ee_rotation=ee_rotation)
        steps = self._steps.to(canonical_action.device)
        stopped = self._stopped.to(canonical_action.device)
        eligible = (steps < self.budget) & ~stopped

        # Skip the task-regret evaluation entirely once no environment can probe
        # this step (budget spent or already early-stopped): pure task control.
        if self.budget == 0 or not bool(eligible.any()):
            self.last_probe_mask = eligible.to("cpu")
            self.last_task_regret = None
            self._steps = self._steps + 1
            return task_action

        belief = self.driver.log_probabilities.exp()
        assert self._recent is not None  # _push_recent populated the buffer above
        recent = self._recent.to(device=canonical_action.device, dtype=canonical_action.dtype)
        error = self._pairwise_task_error(recent, ee_rotation=ee_rotation)
        current_regret = self._task_regret(belief, error)
        self.last_task_regret = current_regret.detach().to("cpu")

        # Sticky early stop: once regret of acting on the MAP drops below the
        # threshold, commit to task control for the rest of the episode.
        newly_stopped = current_regret < self.regret_threshold
        stopped = stopped | newly_stopped
        self._stopped = stopped.to(self._stopped.device)
        probing = eligible & ~stopped

        if not bool(probing.any()):
            self.last_probe_mask = probing.to("cpu")
            self._steps = self._steps + 1
            return task_action

        probe = self._select_probe(
            canonical_action, belief, error, current_regret, ee_rotation=ee_rotation
        ).to(device=canonical_action.device, dtype=canonical_action.dtype)
        raw = torch.where(probing.unsqueeze(-1), probe, task_action)
        self.last_probe_mask = probing.to("cpu")
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
            self._steps = torch.where(boundary, torch.zeros_like(self._steps), self._steps)
            self._stopped = self._stopped & ~boundary
            # Mark the recent-action buffer stale for reset environments so the
            # next episode's regret sample never mixes in the prior episode.
            self._fresh = self._fresh | boundary.to(self._fresh.device)

    def _push_recent(self, canonical_action: Tensor) -> None:
        """Append the current canonical intent to the per-env recent-action buffer.

        On a fresh episode every slot is filled with the current intent so the
        mean over the window is contamination-free and always well defined; on
        subsequent steps the newest intent shifts in and the oldest drops out.
        """
        detached = canonical_action.detach().to("cpu")
        if self._recent is None:
            self._recent = detached.unsqueeze(0).repeat(self.recent_window, 1, 1)
            self._fresh = torch.zeros(self.driver.batch_size, dtype=torch.bool)
            return
        shifted = torch.cat((detached.unsqueeze(0), self._recent[:-1]), dim=0)
        refilled = detached.unsqueeze(0).repeat(self.recent_window, 1, 1)
        fresh = self._fresh.reshape(1, self.driver.batch_size, 1)
        self._recent = torch.where(fresh, refilled, shifted)
        self._fresh = torch.zeros(self.driver.batch_size, dtype=torch.bool)
