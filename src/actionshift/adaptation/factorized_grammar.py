"""Full-grammar factorized belief: discrete contract scoring without the pool.

The pool-privileged exact belief (``adaptation.hypotheses``) scores a declared
finite pool of nine full contracts. That is a strong privilege: the true contract
is one of nine known candidates. This module removes it. It scores the *entire*
declared finite contract grammar -- every permutation in ``S6``, every per-channel
sign and finite scale, every target and lag -- by exploiting the fact that, under
the benchmark's identity end-effector rotation, the decode is **separable per
channel**. Semantic channel ``i`` is produced by exactly one raw channel with one
sign and one scale, so the joint grammar factorizes into a small grid of per-cell
evidence scores that are accumulated as tensor operations and resolved into a MAP
contract by a linear-assignment over channels.

Privilege statement. The only knowledge used here is (a) the declared finite
grammar (the benchmark's own contract space) and (b) the contract-independent
response calibration (``alpha``/``sigma``), measured on the unwrapped identity
environment and shared by every tournament method. It never receives the true
contract and never enumerates the pool. This converts "pool privilege" into the
strictly weaker "grammar knowledge".

Identifiability limits, documented and enforced by construction:

- **frame** is degenerate under the identity rotation (``base`` and ``tool`` are
  observationally identical), so it is collapsed to ``base``;
- **gripper** inversion is unobservable from the tcp-pose response, so it is fixed
  to ``False`` (marked unidentified). Contracts that invert the gripper therefore
  cap achievable success on gripper-dependent tasks -- a real ceiling of the
  unprivileged setting, reported plainly.

The per-cell prediction is bit-faithful to ``contracts.transforms``: for a global
mode ``m = (target, lag)`` the executed response of semantic channel ``i`` fed by
raw channel ``j`` with sign ``s`` and scale ``k`` is

    delta:    obs_i(t) ~ alpha_i * s * k * raw_j(t - lag)
    absolute: obs_i(t) ~ alpha_i * s * k * (raw_j(t - lag) - raw_j(t - lag - 1))

(the lag ring and the zero initialization reproduce ``ActionLag`` and the
zero-initialized target of ``CompleteActionDecoder`` exactly). Summed over the six
assigned cells this equals the pool belief's per-step Gaussian log-likelihood for
the corresponding full contract, so the two families are directly comparable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from torch import Tensor

from actionshift.adaptation.response import ResponseModel
from actionshift.adaptation.scale_corrector import ScaleCorrector
from actionshift.contracts.transforms import encode_complete_action
from actionshift.contracts.types import ActionContract, Target

# The declared finite grammar. The scale grid covers every frozen evaluation
# contract's scale (verified against benchmarking/gate1_eval.representative_contracts
# and run_adaptation_slice.declared_pool: {0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0}).
GRAMMAR_SCALES: tuple[float, ...] = (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0)
GRAMMAR_SIGNS: tuple[float, ...] = (-1.0, 1.0)
GRAMMAR_TARGETS: tuple[Target, ...] = ("delta", "absolute")
GRAMMAR_LAGS: tuple[int, ...] = (0, 1, 2, 4)
_POSE_CHANNELS = 6
_ACTION_WIDTH = 7


def _identity_rotation(batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.eye(3, device=device, dtype=dtype).expand(batch_size, 3, 3)


class CellScorer(Protocol):
    """Pluggable evidence-scoring backend for the per-cell grammar contribution.

    An implementation computes the ``(modes, batch, 6[i], 6[j], |signs|, |scales|)``
    Gaussian log-evidence tensor that the driver would otherwise build in torch:

        contribution[m, b, i, j, s, k] =
            -0.5 * ((observed[b, i]
                     - alpha[i] * base_m[b, j] * signs[s] * scales[k]) / sigma[i]) ** 2

    where ``base_m[b, j]`` is ``history[lag_m][b, j]`` for a delta mode and
    ``history[lag_m][b, j] - history[lag_m + 1][b, j]`` for an absolute mode (the
    fixed single-step-delayed lag alignment). The default backend is the torch
    path inlined in :meth:`FactorizedGrammarDriver._compute_contribution`; the
    ActionABI C++ backend lives in :mod:`actionshift.adaptation.cpp_backend`.
    """

    def score(
        self,
        *,
        history: Tensor,
        observed: Tensor,
        alpha: Tensor,
        sigma: Tensor,
        signs: tuple[float, ...],
        scales: tuple[float, ...],
        mode_targets: tuple[int, ...],
        mode_lags: tuple[int, ...],
    ) -> Tensor:
        """Return the per-cell contribution tensor for one observed transition."""
        ...


class FactorizedGrammarDriver:
    """Per-channel factorized belief over the entire declared contract grammar.

    State per environment: an accumulated per-cell log-evidence tensor of shape
    ``(modes, 6 semantic, 6 raw, |signs|, |scales|)`` and a raw-action history ring
    that reproduces the wrapper's lag alignment. ``update`` folds one observed
    transition into the evidence; ``map_encode`` resolves the current MAP contract
    per environment (best sign/scale per cell, linear-assignment permutation, best
    global mode) and encodes the canonical command under it.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        response: ResponseModel,
        scales: tuple[float, ...] = GRAMMAR_SCALES,
        signs: tuple[float, ...] = GRAMMAR_SIGNS,
        targets: tuple[Target, ...] = GRAMMAR_TARGETS,
        lags: tuple[int, ...] = GRAMMAR_LAGS,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        persist_across_episodes: bool = False,
        cell_scorer: CellScorer | None = None,
        reanchor_period: int = 0,
        scale_correction: bool = False,
        scale_window: int = 16,
        scale_bounds: tuple[float, float] = (0.4, 2.5),
        scale_command_floor: float = 0.05,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if response.channels != _POSE_CHANNELS:
            raise ValueError("the factorized grammar belief models six pose channels")
        if not scales or not signs or not targets or not lags:
            raise ValueError("grammar grids must be non-empty")
        if any(lag < 0 for lag in lags):
            raise ValueError("lags must be nonnegative")
        if reanchor_period < 0:
            raise ValueError("reanchor_period must be nonnegative")
        if cell_scorer is not None and response.alpha_c0 is not None:
            raise ValueError("the saturating gain requires the default torch backend")
        self.batch_size = batch_size
        self.response = response
        self.persist_across_episodes = persist_across_episodes
        self._device = torch.device(device)
        self._dtype = dtype
        self._scales = tuple(scales)
        self._signs = tuple(signs)
        self._targets = tuple(targets)
        self._lags = tuple(lags)
        self._modes = tuple((target, lag) for target in self._targets for lag in self._lags)
        # Injected evidence-scoring backend (default: the inlined torch path). The
        # mode grids are flattened to integer codes the backend can consume.
        self._cell_scorer = cell_scorer
        self._mode_target_codes = tuple(
            1 if target == "absolute" else 0 for target, _ in self._modes
        )
        self._mode_lag_values = tuple(lag for _, lag in self._modes)
        # coefficient grid c[s, k] = sign_s * scale_k
        sign_tensor = torch.tensor(self._signs, device=self._device, dtype=dtype)
        scale_tensor = torch.tensor(self._scales, device=self._device, dtype=dtype)
        self._coeff = sign_tensor[:, None] * scale_tensor[None, :]  # (S, K)
        self._alpha = torch.tensor(
            _as_vector(response.alpha), device=self._device, dtype=dtype
        )
        self._sigma = torch.tensor(
            _as_vector(response.sigma), device=self._device, dtype=dtype
        )
        # Magnitude-dependent (saturating) pose gain, opt-in via the response model.
        self._alpha_c0: Tensor | None = None
        if response.alpha_c0 is not None:
            self._alpha_c0 = torch.tensor(
                _as_vector(response.alpha_c0), device=self._device, dtype=dtype
            )
        # Gripper sign belief: row 0 == not inverted, row 1 == inverted. Opt-in via
        # the gripper-calibrated response model; otherwise the flag stays False and
        # the seventh evidence channel is ignored (version-1 reproducible behaviour).
        self._has_gripper = response.has_gripper
        self._gripper_alpha = float(response.gripper_alpha or 0.0)
        self._gripper_sigma = float(response.gripper_sigma or 1.0)
        self._gripper_scores = torch.zeros(
            (2, batch_size), device=self._device, dtype=dtype
        )
        # Periodic re-anchoring of the tracked absolute target from the observed tcp
        # displacement (contract-independent task knowledge) to bound absolute drift.
        self._reanchor_period = reanchor_period
        self._observed_cumulative = torch.zeros(
            (batch_size, _POSE_CHANNELS), device=self._device, dtype=dtype
        )
        self._steps_since_anchor = torch.zeros(
            batch_size, device=self._device, dtype=torch.long
        )
        num_modes = len(self._modes)
        self._scores = torch.zeros(
            (num_modes, batch_size, _POSE_CHANNELS, _POSE_CHANNELS, len(self._signs),
             len(self._scales)),
            device=self._device,
            dtype=dtype,
        )
        # history[k] = raw pose channels k steps ago (0 = current). Needs max_lag + 1
        # back for the absolute previous-step term, plus the current slot.
        self._history_depth = max(self._lags) + 2
        self._history = torch.zeros(
            (self._history_depth, batch_size, _POSE_CHANNELS),
            device=self._device,
            dtype=dtype,
        )
        self._tracked_target = torch.zeros(
            (batch_size, _POSE_CHANNELS), device=self._device, dtype=dtype
        )
        # Hold-probe telescoped-window accumulators. During a sustained (held) raw
        # excitation the absolute mode's differenced drive telescopes over the
        # window back to the full undifferenced excursion, so integrating the
        # observed response and the per-mode drive across the window and scoring the
        # single integrated statistic restores the SNR the per-step differenced
        # score halves. These stay at zero unless ``update`` is called with a
        # ``hold_mask`` (the hold-probing adapter), so the per-step path is
        # bit-unchanged when they are unused.
        self._hold_base_sum = torch.zeros(
            (num_modes, batch_size, _POSE_CHANNELS), device=self._device, dtype=dtype
        )
        self._hold_sum_obs = torch.zeros(
            (batch_size, _POSE_CHANNELS), device=self._device, dtype=dtype
        )
        self._hold_len = torch.zeros(batch_size, device=self._device, dtype=torch.long)
        # Closed-loop drift-based scale correction wrapped around the MAP encode.
        # It refines the effective per-channel scale from the integrated ratio of
        # observed tcp response to commanded intent (task knowledge only), rescuing
        # absolute-target control that grid-MAP scale non-identifiability collapses.
        self._scale_corrector: ScaleCorrector | None = None
        self._last_commanded_pose: Tensor | None = None
        self._last_effective_scale: Tensor | None = None
        if scale_correction:
            self._scale_corrector = ScaleCorrector(
                batch_size=batch_size,
                alpha=self._alpha,
                channels=_POSE_CHANNELS,
                window=scale_window,
                min_correction=scale_bounds[0],
                max_correction=scale_bounds[1],
                command_floor=scale_command_floor,
                device=self._device,
                dtype=dtype,
            )

    @property
    def scale_correction(self) -> ScaleCorrector | None:
        """The drift-based scale corrector, or ``None`` when disabled."""
        return self._scale_corrector

    @property
    def modes(self) -> tuple[tuple[Target, int], ...]:
        return self._modes

    def _predicted_base(self, target: Target, lag: int) -> Tensor:
        """Per-raw-channel drive value feeding the decode for one mode: (B, 6)."""
        lagged = self._history[lag]
        if target == "absolute":
            return lagged - self._history[lag + 1]
        return lagged

    def _advance_history(self, raw_action: Tensor, reset_mask: Tensor | None) -> None:
        """Push one raw pose action into the lag ring (with the pending reset)."""
        if reset_mask is not None:
            reset = reset_mask.to(device=self._device, dtype=torch.bool)
            keep = (~reset).to(self._dtype).view(1, self.batch_size, 1)
            self._history = self._history * keep
        raw6 = raw_action[..., :_POSE_CHANNELS].to(device=self._device, dtype=self._dtype)
        self._history = torch.cat((raw6.unsqueeze(0), self._history[:-1]), dim=0)

    def update(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        corrector_mask: Tensor | None = None,
        hold_mask: Tensor | None = None,
        window_close: Tensor | None = None,
    ) -> None:
        """Fold one observed transition into the per-cell grammar evidence.

        Mask semantics mirror ``ExactBeliefDriver.update`` exactly: ``reset_mask``
        is the wrapper's *pending* decoder reset (it lands one step after an episode
        boundary) and zeros the history ring for those environments before the
        current raw action is pushed; ``invalid_mask`` marks environments whose
        transition crosses an auto-reset boundary, whose evidence is discarded and
        (unless persistence is requested) whose accumulated scores and tracked
        target return to the new-episode zero state. ``corrector_mask`` (per
        environment) marks steps whose observed response answers the encoded task
        command and so may feed the scale corrector -- probe-pulse steps are
        excluded by the probing adapter.

        ``hold_mask`` (per environment) marks steps that are part of a sustained
        hold-probe window: their per-step contribution is routed into the
        telescoped-window accumulator instead of the per-step scores, and folded in
        as one integrated Gaussian evidence term when ``window_close`` fires. Both
        default to ``None``, leaving the per-step path bit-unchanged.
        """
        if raw_action.shape != (self.batch_size, _ACTION_WIDTH):
            raise ValueError("raw_action must be (batch_size, 7)")
        expected_width = _POSE_CHANNELS + (1 if self._has_gripper else 0)
        if observed_response.shape != (self.batch_size, expected_width):
            raise ValueError(f"observed_response must be (batch_size, {expected_width})")
        masks = (
            ("reset_mask", reset_mask),
            ("invalid_mask", invalid_mask),
            ("corrector_mask", corrector_mask),
            ("hold_mask", hold_mask),
            ("window_close", window_close),
        )
        for name, mask in masks:
            if mask is not None and mask.shape != (self.batch_size,):
                raise ValueError(f"{name} must contain one value per environment")

        self._advance_history(raw_action, reset_mask)

        observed = observed_response.to(device=self._device, dtype=self._dtype)
        pose_observed = observed[..., :_POSE_CHANNELS]
        self._update_scale_corrector(
            pose_observed, invalid_mask=invalid_mask, corrector_mask=corrector_mask
        )
        contribution = self._compute_contribution(pose_observed)  # (M, B, i, j, s, k)

        if self._has_gripper:
            self._accumulate_gripper_evidence(
                raw_action[..., 6].to(device=self._device, dtype=self._dtype),
                observed[..., _POSE_CHANNELS],
                invalid_mask=invalid_mask,
            )
        self._reanchor_target(pose_observed, invalid_mask=invalid_mask)

        hold = None
        if hold_mask is not None:
            hold = hold_mask.to(device=self._device, dtype=torch.bool)
            self._accumulate_hold(
                pose_observed, hold, window_close=window_close, invalid_mask=invalid_mask
            )

        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._device, dtype=torch.bool)
            valid = (~boundary).to(self._dtype).view(1, self.batch_size, 1, 1, 1, 1)
            contribution = contribution * valid
        if hold is not None:
            # Per-step evidence is suppressed for hold steps; those steps contribute
            # only the integrated window term (avoiding a correlated double count).
            per_step = (~hold).to(self._dtype).view(1, self.batch_size, 1, 1, 1, 1)
            contribution = contribution * per_step
        self._scores = self._scores + contribution

        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._device, dtype=torch.bool)
            if not self.persist_across_episodes:
                keep_scores = (~boundary).to(self._dtype).view(1, self.batch_size, 1, 1, 1, 1)
                self._scores = self._scores * keep_scores
            keep_target = (~boundary).to(self._dtype).view(self.batch_size, 1)
            self._tracked_target = self._tracked_target * keep_target

    def _accumulate_gripper_evidence(
        self, raw_gripper: Tensor, observed_gripper: Tensor, *, invalid_mask: Tensor | None
    ) -> None:
        """Fold one gripper transition into the binary sign belief (0=not, 1=inv)."""
        predicted = torch.stack((raw_gripper, -raw_gripper))  # (2, B): not-inv, inv
        residual = observed_gripper.unsqueeze(0) - self._gripper_alpha * predicted
        contribution = -0.5 * (residual / self._gripper_sigma).square()  # (2, B)
        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._device, dtype=torch.bool)
            contribution = contribution * (~boundary).to(self._dtype).unsqueeze(0)
        self._gripper_scores = self._gripper_scores + contribution
        if invalid_mask is not None and not self.persist_across_episodes:
            boundary = invalid_mask.to(device=self._device, dtype=torch.bool)
            self._gripper_scores = self._gripper_scores * (~boundary).to(self._dtype).unsqueeze(0)

    def _reanchor_target(
        self, pose_observed: Tensor, *, invalid_mask: Tensor | None
    ) -> None:
        """Track achieved tcp displacement and re-anchor the tracked target to it.

        The observed pose delta is the true tcp motion in the base frame, so its
        running sum is the achieved canonical displacement. Re-anchoring the tracked
        target's position channels to it every ``reanchor_period`` steps re-references
        absolute-target encoding to reality, capping accumulated drift. Rotation
        channels keep the cumulative-intent semantics (rotation vectors do not sum
        linearly). A no-op for delta-target encoding, which ignores the target.
        """
        self._observed_cumulative = self._observed_cumulative + pose_observed
        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._device, dtype=torch.bool)
            keep = (~boundary).to(self._dtype).unsqueeze(-1)
            self._observed_cumulative = self._observed_cumulative * keep
            self._steps_since_anchor = torch.where(
                boundary, torch.zeros_like(self._steps_since_anchor), self._steps_since_anchor
            )
        if self._reanchor_period <= 0:
            return
        self._steps_since_anchor = self._steps_since_anchor + 1
        due = self._steps_since_anchor >= self._reanchor_period
        if bool(due.any()):
            mask = due.unsqueeze(-1)
            anchored = torch.where(
                mask, self._observed_cumulative[:, :3], self._tracked_target[:, :3]
            )
            self._tracked_target = torch.cat(
                (anchored, self._tracked_target[:, 3:]), dim=-1
            )
            self._steps_since_anchor = torch.where(
                due, torch.zeros_like(self._steps_since_anchor), self._steps_since_anchor
            )

    def _update_scale_corrector(
        self,
        pose_observed: Tensor,
        *,
        invalid_mask: Tensor | None,
        corrector_mask: Tensor | None,
    ) -> None:
        """Feed one commanded/observed transition to the drift-based scale corrector.

        The commanded intent is the canonical pose from the immediately preceding
        ``map_encode``; the observed pose is this step's tcp response. Boundary
        transitions and probe steps (``corrector_mask`` false) are excluded, and
        the corrector resets on an episode boundary (unless persistence is kept).
        """
        corrector = self._scale_corrector
        if corrector is None:
            return
        active = torch.ones(self.batch_size, device=self._device, dtype=torch.bool)
        if corrector_mask is not None:
            active = active & corrector_mask.to(device=self._device, dtype=torch.bool)
        if invalid_mask is not None:
            active = active & ~invalid_mask.to(device=self._device, dtype=torch.bool)
        if self._last_commanded_pose is not None and self._last_effective_scale is not None:
            corrector.accumulate(
                self._last_commanded_pose,
                pose_observed,
                self._last_effective_scale,
                active_mask=active,
            )
        if invalid_mask is not None and not self.persist_across_episodes:
            corrector.reset(invalid_mask.to(device=self._device, dtype=torch.bool))

    def _mode_bases(self) -> Tensor:
        """Per-mode per-raw-channel drive from the current history: (M, B, 6[j])."""
        return torch.stack(
            [self._predicted_base(target, lag) for target, lag in self._modes], dim=0
        )

    def _accumulate_hold(
        self,
        pose_observed: Tensor,
        hold: Tensor,
        *,
        window_close: Tensor | None,
        invalid_mask: Tensor | None,
    ) -> None:
        """Integrate a hold-probe window and fold its telescoped evidence at close.

        For each held step we add this step's per-mode drive ``base_m(t)`` and the
        observed pose response into per-environment accumulators. Over the window
        the absolute mode's drive ``raw(t) - raw(t-1)`` telescopes to the net raw
        excursion, and the sustained (delta) drive accumulates to ``len`` times the
        held command -- two distinct integrated predictions -- so scoring the single
        integrated observable ``sum_t obs`` against ``alpha * sum_t base_m * coeff``
        at a per-window noise scale ``sigma * sqrt(len)`` sharply separates target
        mode and, with the full undifferenced signal restored, the permutation.
        """
        active = hold
        if invalid_mask is not None:
            active = active & ~invalid_mask.to(device=self._device, dtype=torch.bool)
        add_pose = active.to(self._dtype).view(self.batch_size, 1)
        add_mode = active.to(self._dtype).view(1, self.batch_size, 1)
        self._hold_sum_obs = self._hold_sum_obs + pose_observed * add_pose
        self._hold_base_sum = self._hold_base_sum + self._mode_bases() * add_mode
        self._hold_len = self._hold_len + active.long()

        close = active
        if window_close is not None:
            close = close & window_close.to(device=self._device, dtype=torch.bool)
        else:
            close = torch.zeros_like(active)
        if bool(close.any()):
            length = self._hold_len.clamp_min(1).to(self._dtype)
            sigma_eff = self._sigma[None, :] * length.sqrt()[:, None]  # (B, i)
            # predicted integrated obs for cell (i, j) under mode m, sign s, scale k
            predicted = (
                self._alpha[None, None, :, None, None, None]
                * self._hold_base_sum[:, :, None, :, None, None]
                * self._coeff[None, None, None, None, :, :]
            )  # (M, B, i, j, s, k)
            residual = self._hold_sum_obs[None, :, :, None, None, None] - predicted
            standardized = residual / sigma_eff[None, :, :, None, None, None]
            window_contribution = -0.5 * standardized.square()
            keep = close.to(self._dtype).view(1, self.batch_size, 1, 1, 1, 1)
            self._scores = self._scores + window_contribution * keep
        # Reset the window accumulators for environments that just closed (or whose
        # transition crossed an episode boundary), so the next window is clean.
        cleared = close
        if invalid_mask is not None:
            cleared = cleared | invalid_mask.to(device=self._device, dtype=torch.bool)
        reset_row = cleared.view(self.batch_size, 1)
        self._hold_sum_obs = torch.where(
            reset_row, torch.zeros_like(self._hold_sum_obs), self._hold_sum_obs
        )
        self._hold_base_sum = torch.where(
            cleared.view(1, self.batch_size, 1),
            torch.zeros_like(self._hold_base_sum),
            self._hold_base_sum,
        )
        self._hold_len = torch.where(
            cleared, torch.zeros_like(self._hold_len), self._hold_len
        )

    def anchor_tracked_target(self, mask: Tensor) -> None:
        """Snap the tracked absolute target to the achieved tcp displacement.

        The running sum of observed pose deltas (``_observed_cumulative``, task
        knowledge) is the achieved canonical displacement, so re-referencing the
        tracked target to it for masked environments re-anchors absolute-target
        encoding to reality after a hold-probe phase that did not execute task
        intent -- the discrete contract is now identified, so control resumes from
        the true achieved pose rather than an intent sum that never ran.
        """
        if mask.shape != (self.batch_size,):
            raise ValueError("mask must contain one value per environment")
        row = mask.to(device=self._device, dtype=torch.bool).unsqueeze(-1)
        self._tracked_target = torch.where(row, self._observed_cumulative, self._tracked_target)

    def _compute_contribution(self, observed: Tensor) -> Tensor:
        """Per-cell Gaussian log-evidence for one observed transition.

        Shape ``(modes, batch, 6[i], 6[j], |signs|, |scales|)``. When a
        :class:`CellScorer` backend is injected the computation is delegated to it
        (the ActionABI C++ core); otherwise the original torch path runs. The two
        paths are numerically equivalent (parity-tested to <=1e-6 relative).
        """
        if self._cell_scorer is not None:
            return self._cell_scorer.score(
                history=self._history,
                observed=observed,
                alpha=self._alpha,
                sigma=self._sigma,
                signs=self._signs,
                scales=self._scales,
                mode_targets=self._mode_target_codes,
                mode_lags=self._mode_lag_values,
            )
        # residual/likelihood per mode, cell (i, j), sign s, scale k.
        contributions = []
        for target, lag in self._modes:
            base = self._predicted_base(target, lag)  # (B, 6[j])
            # predicted semantic value driven by raw channel j under (s, k): (B, j, s, k)
            predicted = base[:, :, None, None] * self._coeff[None, None, :, :]
            # expected observed for semantic channel i: alpha_i * predicted, with the
            # magnitude-dependent (saturating) gain folded in when calibrated.
            expected = self._alpha[None, :, None, None, None] * predicted[:, None, :, :, :]
            if self._alpha_c0 is not None:
                expected = expected / (
                    1.0
                    + predicted[:, None, :, :, :].abs()
                    / self._alpha_c0[None, :, None, None, None]
                )
            residual = observed[:, :, None, None, None] - expected  # (B, i, j, s, k)
            standardized = residual / self._sigma[None, :, None, None, None]
            contributions.append(-0.5 * standardized.square())
        return torch.stack(contributions, dim=0)  # (M, B, i, j, s, k)

    def _map_fields(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Resolve per-environment MAP (permutation, sign, scale, target index)."""
        num_signs = len(self._signs)
        num_scales = len(self._scales)
        flat = self._scores.reshape(
            len(self._modes), self.batch_size, _POSE_CHANNELS, _POSE_CHANNELS,
            num_signs * num_scales,
        )
        best_cell, best_sk = flat.max(dim=-1)  # (M, B, i, j)
        sign_index = best_sk // num_scales
        scale_index = best_sk % num_scales
        cost = (-best_cell).detach().cpu().numpy()  # (M, B, i, j); minimize
        num_modes = len(self._modes)

        permutation = torch.zeros(
            (self.batch_size, _POSE_CHANNELS), dtype=torch.long, device=self._device
        )
        assigned_perm = torch.zeros(
            (num_modes, self.batch_size, _POSE_CHANNELS), dtype=torch.long
        )
        mode_totals = torch.full(
            (num_modes, self.batch_size), float("-inf"), device=self._device
        )
        rows = torch.arange(_POSE_CHANNELS)
        for mode in range(num_modes):
            for env in range(self.batch_size):
                _, col = linear_sum_assignment(cost[mode, env])
                col_tensor = torch.as_tensor(col, dtype=torch.long)
                assigned_perm[mode, env] = col_tensor
                mode_totals[mode, env] = best_cell[mode, env, rows, col_tensor].sum()
        mode_choice = mode_totals.argmax(dim=0)  # (B,)
        for env in range(self.batch_size):
            permutation[env] = assigned_perm[mode_choice[env], env].to(self._device)

        env_index = torch.arange(self.batch_size, device=self._device)
        chosen_sign_index = sign_index[mode_choice, env_index]  # (B, i, j)
        chosen_scale_index = scale_index[mode_choice, env_index]
        # sign/scale index at the assigned cell (i, perm[i])
        cols = permutation.unsqueeze(-1)  # (B, i, 1)
        sign_sel = torch.gather(chosen_sign_index, 2, cols).squeeze(-1)  # (B, i)
        scale_sel = torch.gather(chosen_scale_index, 2, cols).squeeze(-1)
        return permutation, sign_sel, scale_sel, mode_choice

    def map_contracts(self) -> list[ActionContract]:
        """Build the per-environment MAP ``ActionContract`` estimates.

        Sign and scale are looked up in the original Python grammar grids (not a
        float tensor) so the recovered contract carries the exact declared values.
        """
        permutation, sign_index, scale_index, mode_choice = self._map_fields()
        # MAP gripper sign per environment: inverted iff its accumulated evidence
        # exceeds the non-inverted hypothesis. Fixed False without gripper evidence.
        if self._has_gripper:
            gripper_inverted = (self._gripper_scores[1] > self._gripper_scores[0]).tolist()
        else:
            gripper_inverted = [False] * self.batch_size
        contracts: list[ActionContract] = []
        for env in range(self.batch_size):
            target, lag = self._modes[int(mode_choice[env].item())]
            contracts.append(
                ActionContract(
                    permutation=tuple(int(p) for p in permutation[env].tolist()),
                    sign=tuple(int(self._signs[int(s)]) for s in sign_index[env].tolist()),
                    scale=tuple(self._scales[int(k)] for k in scale_index[env].tolist()),
                    target=target,
                    frame="base",
                    lag=lag,
                    gripper_inverted=bool(gripper_inverted[env]),
                )
            )
        return contracts

    def map_encode(
        self, canonical_action: Tensor, *, target_mask: Tensor | None = None
    ) -> Tensor:
        """Encode a canonical command under each environment's MAP contract.

        ``target_mask`` (per environment) gates the cumulative-intent accumulation:
        only masked environments add this step's command to the tracked absolute
        target. It lets a hold-probing adapter withhold accumulation for
        environments still in the probe phase (whose sent action is a probe, not
        this intent), while controlling environments accumulate as usual. ``None``
        accumulates for every environment (the original behaviour).
        """
        if canonical_action.shape != (self.batch_size, _ACTION_WIDTH):
            raise ValueError("canonical_action must be (batch_size, 7)")
        if target_mask is not None and target_mask.shape != (self.batch_size,):
            raise ValueError("target_mask must contain one value per environment")
        contracts = self.map_contracts()
        if self._scale_corrector is not None:
            # Refine the MAP grid scale by the drift-based correction, then remember
            # this step's commanded pose so the next observed response can be paired
            # with it in the corrector update.
            base_scale = torch.tensor(
                [contract.scale for contract in contracts],
                device=self._device,
                dtype=self._dtype,
            )
            effective = self._scale_corrector.effective_scale(base_scale)
            contracts = [
                replace(contract, scale=tuple(float(s) for s in effective[env].tolist()))
                for env, contract in enumerate(contracts)
            ]
            self._last_commanded_pose = (
                canonical_action[..., :_POSE_CHANNELS]
                .detach()
                .to(device=self._device, dtype=self._dtype)
            )
            self._last_effective_scale = effective.detach()
        encoded = torch.empty_like(canonical_action)
        rotation = _identity_rotation(
            1, canonical_action.device, canonical_action.dtype
        )
        target = self._tracked_target.to(
            device=canonical_action.device, dtype=canonical_action.dtype
        )
        # Group environments sharing an identical encode contract (frame/lag do not
        # affect the instantaneous encode, so only perm/sign/scale/target matter).
        groups: dict[str, list[int]] = {}
        keys = [
            _encode_key(contracts[env]) for env in range(self.batch_size)
        ]
        for env, key in enumerate(keys):
            groups.setdefault(key, []).append(env)
        for members in groups.values():
            contract = contracts[members[0]]
            index = torch.tensor(members, device=canonical_action.device)
            rows = canonical_action.index_select(0, index)
            encoded[index] = encode_complete_action(
                rows,
                contract,
                ee_rotation=rotation.expand(len(members), 3, 3),
                tracked_target=target.index_select(0, index),
            )
        # Accumulate the canonical intent so absolute-target encoding stays exact
        # across the episode (contract-independent cumulative target).
        increment = canonical_action[..., :_POSE_CHANNELS].to(
            device=self._device, dtype=self._dtype
        )
        if target_mask is not None:
            increment = increment * target_mask.to(
                device=self._device, dtype=self._dtype
            ).unsqueeze(-1)
        self._tracked_target = self._tracked_target + increment
        return encoded


def _encode_key(contract: ActionContract) -> str:
    return "|".join(
        (
            ",".join(str(p) for p in contract.permutation),
            ",".join(str(s) for s in contract.sign),
            ",".join(f"{s:.6g}" for s in contract.scale),
            contract.target,
            "gi" if contract.gripper_inverted else "gn",
        )
    )


def _as_vector(value: tuple[float, ...] | float, channels: int = _POSE_CHANNELS) -> tuple[
    float, ...
]:
    if isinstance(value, float | int):
        return (float(value),) * channels
    vector = tuple(float(item) for item in value)
    if len(vector) != channels:
        raise ValueError("response parameters must provide one value per pose channel")
    return vector


class FactorizedGrammarAdapter:
    """Passive full-grammar factorized belief adapter (``ContractAdapter``).

    Encodes with the per-environment MAP contract and folds every observed
    transition into the grammar evidence. It never receives the true contract; its
    only privilege is grammar knowledge plus the shared response calibration.
    """

    name = "factorized_grammar"

    def __init__(
        self,
        *,
        batch_size: int,
        response: ResponseModel,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        persist_across_episodes: bool = False,
        cell_scorer: CellScorer | None = None,
        reanchor_period: int = 0,
        scale_correction: bool = False,
        scale_window: int = 16,
        scale_bounds: tuple[float, float] = (0.4, 2.5),
        scale_command_floor: float = 0.05,
        **grammar: object,
    ) -> None:
        self.driver = FactorizedGrammarDriver(
            batch_size=batch_size,
            response=response,
            device=device,
            dtype=dtype,
            persist_across_episodes=persist_across_episodes,
            cell_scorer=cell_scorer,
            reanchor_period=reanchor_period,
            scale_correction=scale_correction,
            scale_window=scale_window,
            scale_bounds=scale_bounds,
            scale_command_floor=scale_command_floor,
        )

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        # The factorized grammar collapses frame to base (documented identifiability
        # limit), so the live tcp rotation is not consumed here.
        del ee_rotation
        return self.driver.map_encode(canonical_action)

    def observe(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        reset_mask: Tensor | None = None,
        invalid_mask: Tensor | None = None,
        ee_rotation: Tensor | None = None,
    ) -> None:
        del ee_rotation
        self.driver.update(
            raw_action, observed_response, reset_mask=reset_mask, invalid_mask=invalid_mask
        )


def fixed_grammar_probe_pulse(
    step_index: Tensor, *, amplitude: float, channels: int = _POSE_CHANNELS
) -> Tensor:
    """Alternating-sign basis pulse cycling through the pose channels.

    Identical construction to ``adaptation.probes.fixed_probe_pulse`` (kept local
    to avoid a cross-module coupling): the probe never drives the gripper channel.
    """
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    batch = step_index.shape[0]
    action = step_index.new_zeros((batch, _ACTION_WIDTH), dtype=torch.float32)
    channel = (step_index % channels).long()
    sign = torch.where((step_index // channels) % 2 == 0, 1.0, -1.0)
    action[torch.arange(batch), channel] = amplitude * sign
    return action


class FactorizedGrammarProbingAdapter:
    """Full-grammar factorized belief with a bounded per-episode fixed-probe phase.

    Probing matters more here than for the pool belief: the hypothesis space is the
    whole grammar, so the first ``budget`` steps send bounded raw basis pulses that
    excite every channel before task control begins. The probe pulses fold into the
    grammar evidence exactly like passive observations; the amplitude stays inside
    the declared bound and the gripper is never probed.
    """

    def __init__(
        self,
        driver: FactorizedGrammarDriver,
        *,
        budget: int,
        amplitude: float = 0.5,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        if amplitude <= 0:
            raise ValueError("amplitude must be positive")
        self.name = "factorized_grammar_probes"
        self.driver = driver
        self.budget = budget
        self.amplitude = amplitude
        self._steps = torch.zeros(driver.batch_size, dtype=torch.long)
        self.last_probe_mask: Tensor | None = None

    def encode(
        self, canonical_action: Tensor, *, ee_rotation: Tensor | None = None
    ) -> Tensor:
        del ee_rotation  # frame collapsed to base; tcp rotation not consumed
        probing = (self._steps < self.budget).to(canonical_action.device)
        task_action = self.driver.map_encode(canonical_action)
        if self.budget == 0 or not bool(probing.any()):
            self.last_probe_mask = probing
            self._steps = self._steps + 1
            return task_action
        probe = fixed_grammar_probe_pulse(self._steps, amplitude=self.amplitude).to(
            device=canonical_action.device, dtype=canonical_action.dtype
        )
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
        del ee_rotation  # frame collapsed to base; tcp rotation not consumed
        # Only steps that sent the encoded task command (not a probe pulse) may feed
        # the scale corrector, since the probe response does not answer the intent.
        corrector_mask: Tensor | None = None
        if self.last_probe_mask is not None:
            corrector_mask = ~self.last_probe_mask.to(
                device=raw_action.device, dtype=torch.bool
            )
        self.driver.update(
            raw_action,
            observed_response,
            reset_mask=reset_mask,
            invalid_mask=invalid_mask,
            corrector_mask=corrector_mask,
        )
        if invalid_mask is not None:
            boundary = invalid_mask.to(device=self._steps.device, dtype=torch.bool)
            self._steps = torch.where(boundary, torch.zeros_like(self._steps), self._steps)


__all__ = [
    "GRAMMAR_LAGS",
    "GRAMMAR_SCALES",
    "GRAMMAR_SIGNS",
    "GRAMMAR_TARGETS",
    "CellScorer",
    "FactorizedGrammarAdapter",
    "FactorizedGrammarDriver",
    "FactorizedGrammarProbingAdapter",
    "fixed_grammar_probe_pulse",
]
