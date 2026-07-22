"""Drift-based closed-loop scale correction for absolute-target control.

The full-grammar factorized belief (:mod:`actionshift.adaptation.factorized_grammar`)
recovers a contract's permutation and per-channel sign from the tcp-pose response,
but *scale* is not identifiable at per-step SNR ~1: the grid MAP estimate is
biased and noisy. Under a **delta** target a residual scale error only rescales
each step, so the frozen policy's closed loop absorbs it. Under an **absolute**
target the same residual integrates through the cumulative tracked target, so the
tcp is driven to a systematically mis-scaled absolute position -- the failure mode
that collapses the all-absolute ``unseen_composition`` split to ~0.

The lever this module exploits is that the mis-scaling is itself a strong,
accumulating signal. Derive it from :mod:`actionshift.contracts.transforms`. With
the MAP permutation and sign matched, encoding a canonical intent ``c_t`` under a
*fixed* effective scale ``eff_i`` and executing it under the true scale ``s`` gives,
per semantic channel ``i`` and **while ``eff_i`` is held constant** (both delta and
absolute target),

    executed_i(t) = (s_i / eff_i) * c_i(t),   obs_i(t) = alpha_i * executed_i(t)

(plus per-step noise). The constancy matters: under an absolute target the executed
delta is ``s * (S_t/eff_t - S_{t-1}/eff_{t-1})`` in the cumulative intent ``S_t``, so
a *change* in the effective scale within a step injects a large cumulative jump term
``s * S_{t-1} * (1/eff_t - 1/eff_{t-1})`` that would bias a naive ratio. Holding the
effective scale fixed across an integration window kills that term, and the simple
integrated ratio then recovers the true scale, with integration averaging the
per-step noise that defeats the instantaneous grid MAP:

    s_hat_i = eff_i * sum_t |obs_i(t)| / ( |alpha_i| * sum_t |c_i(t)| )  ->  s_i.

The window auto-restarts whenever the effective scale it is measuring against
changes (the belief's grid MAP is still settling, or a prior estimate was just
committed), so every committed window was measured at a single fixed effective
scale. The recovered true scale replaces the effective scale in the MAP encode,
closing the loop -- and under an absolute target the commit re-references the
cumulative target to the truth (``W_t = S_t * s/eff_t = S_t`` once ``eff_t = s``),
snapping the tcp back onto the intended trajectory. It uses only task knowledge --
the calibrated ``alpha`` and the adapter's own commanded intent, observed response,
and the scale it itself chose -- never a contract, pool, or true scale.

Fail-safes: the estimate is clamped to a bounded scale band ``[min, max]``; a
channel whose integrated command magnitude is below ``command_floor`` over a window
is frozen (thin evidence) and keeps deferring to the grid MAP; window integration
makes the estimate robust to per-step noise and to a small command/response lag;
and every accumulator resets on an episode boundary so a fresh estimate is formed
per episode.
"""

from __future__ import annotations

import torch
from torch import Tensor

_FLOOR = 1e-6


class ScaleCorrector:
    """Per-environment, per-channel effective-scale estimation from drift.

    Until a channel has produced an estimate, :meth:`effective_scale` defers to the
    grid-MAP scale it is handed. Every ``window`` accumulated steps (integrated at a
    single fixed effective scale) it forms the integrated-ratio estimate of the true
    scale for sufficiently excited channels and thereafter encodes with that
    (clamped) estimate.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        alpha: Tensor,
        channels: int = 6,
        window: int = 16,
        min_correction: float = 0.4,
        max_correction: float = 2.5,
        command_floor: float = 0.05,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 < min_correction <= 1.0 <= max_correction:
            raise ValueError("require 0 < min_correction <= 1 <= max_correction")
        if command_floor < 0.0:
            raise ValueError("command_floor must be nonnegative")
        if alpha.shape != (channels,):
            raise ValueError("alpha must provide one gain per channel")
        self.batch_size = batch_size
        self.channels = channels
        self.window = window
        self.min_correction = min_correction
        self.max_correction = max_correction
        self.command_floor = command_floor
        self._device = torch.device(device)
        self._dtype = dtype
        # Only the magnitude of the calibrated gain enters the ratio; its sign is
        # absorbed by the absolute values (|obs| = |alpha| * |executed|).
        self._abs_alpha = alpha.detach().to(device=self._device, dtype=dtype).abs().clamp_min(
            _FLOOR
        )
        self._estimate = torch.ones((batch_size, channels), device=self._device, dtype=dtype)
        self._valid = torch.zeros((batch_size, channels), device=self._device, dtype=torch.bool)
        self._sum_obs = torch.zeros((batch_size, channels), device=self._device, dtype=dtype)
        self._sum_cmd = torch.zeros((batch_size, channels), device=self._device, dtype=dtype)
        # The fixed effective scale the open window is measured against (set when a
        # window starts and used at commit), the previous step's effective scale
        # (to detect the absolute-target jump step that must be dropped), and a
        # per-environment flag for whether a previous step exists.
        self._eff_window = torch.zeros((batch_size, channels), device=self._device, dtype=dtype)
        self._last_eff = torch.zeros((batch_size, channels), device=self._device, dtype=dtype)
        self._last_valid = torch.zeros(batch_size, device=self._device, dtype=torch.bool)
        self._count = torch.zeros(batch_size, device=self._device, dtype=torch.long)

    @property
    def estimate(self) -> Tensor:
        """The current per-channel effective-scale estimate (identity until formed)."""
        return self._estimate

    @property
    def valid(self) -> Tensor:
        """Per-channel mask of which estimates have been formed."""
        return self._valid

    def effective_scale(self, base_scale: Tensor) -> Tensor:
        """Return the estimated scale where formed, else the grid-MAP ``base_scale``."""
        if base_scale.shape != (self.batch_size, self.channels):
            raise ValueError("base_scale must be (batch_size, channels)")
        estimate = self._estimate.to(device=base_scale.device, dtype=base_scale.dtype)
        valid = self._valid.to(device=base_scale.device)
        return torch.where(valid, estimate, base_scale).clamp_min(_FLOOR)

    def accumulate(
        self,
        commanded_pose: Tensor,
        observed_pose: Tensor,
        effective_scale: Tensor,
        *,
        active_mask: Tensor | None = None,
    ) -> None:
        """Fold one commanded/observed transition into the windowed estimate.

        ``effective_scale`` is the per-channel scale actually used to encode this
        step (so a still-settling grid MAP does not bias the estimate).
        ``active_mask`` (per environment) suppresses steps whose observed response
        does not answer the commanded intent -- e.g. probe-pulse steps.
        """
        for name, tensor in (
            ("commanded_pose", commanded_pose),
            ("observed_pose", observed_pose),
            ("effective_scale", effective_scale),
        ):
            if tensor.shape != (self.batch_size, self.channels):
                raise ValueError(f"{name} must be (batch_size, channels)")
        commanded = commanded_pose.detach().to(device=self._device, dtype=self._dtype)
        observed = observed_pose.detach().to(device=self._device, dtype=self._dtype)
        eff = effective_scale.detach().to(device=self._device, dtype=self._dtype).clamp_min(_FLOOR)
        if active_mask is None:
            active = torch.ones(self.batch_size, device=self._device, dtype=torch.bool)
        else:
            if active_mask.shape != (self.batch_size,):
                raise ValueError("active_mask must contain one value per environment")
            active = active_mask.to(device=self._device, dtype=torch.bool)
        # Window bookkeeping, per environment. A step whose effective scale differs
        # from the *previous* step's carries the absolute-target jump term, so it is
        # dropped and the open window is emptied (the next same-scale step starts a
        # clean window). Otherwise an empty window starts here and a matching one
        # extends -- every committed window is thus measured at a single fixed scale.
        changed = self._last_valid & ~torch.isclose(
            eff, self._last_eff, rtol=1e-4, atol=1e-6
        ).all(dim=-1)
        skip = active & changed
        step = active & ~changed
        is_empty = self._count == 0
        start = step & is_empty
        cont = step & ~is_empty
        clear = (skip | start).unsqueeze(-1)
        self._sum_obs = torch.where(clear, torch.zeros_like(self._sum_obs), self._sum_obs)
        self._sum_cmd = torch.where(clear, torch.zeros_like(self._sum_cmd), self._sum_cmd)
        self._eff_window = torch.where(start.unsqueeze(-1), eff, self._eff_window)
        add = (start | cont).to(self._dtype).unsqueeze(-1)
        self._sum_obs = self._sum_obs + observed.abs() * add
        self._sum_cmd = self._sum_cmd + commanded.abs() * add
        count = self._count
        count = torch.where(skip, torch.zeros_like(count), count)
        count = torch.where(cont, self._count + 1, count)
        count = torch.where(start, torch.ones_like(count), count)
        self._count = count
        self._last_eff = torch.where(active.unsqueeze(-1), eff, self._last_eff)
        self._last_valid = self._last_valid | active
        self._maybe_update()

    def _maybe_update(self) -> None:
        due = self._count >= self.window
        if not bool(due.any()):
            return
        # Simple integrated ratio at the window's fixed effective scale; the eff
        # cancels the absolute-target jump term and integration averages the noise.
        ratio = self._sum_obs / (self._abs_alpha.unsqueeze(0) * self._sum_cmd.clamp_min(_FLOOR))
        estimate = (self._eff_window * ratio).clamp(self.min_correction, self.max_correction)
        excited = self._sum_cmd > self.command_floor
        due_channels = due.unsqueeze(-1)
        commit = due_channels & excited
        self._estimate = torch.where(commit, estimate, self._estimate)
        self._valid = self._valid | commit
        # Reset the window for environments that just updated.
        zeros_c = torch.zeros_like(self._sum_obs)
        self._sum_obs = torch.where(due_channels, zeros_c, self._sum_obs)
        self._sum_cmd = torch.where(due_channels, zeros_c, self._sum_cmd)
        self._count = torch.where(due, torch.zeros_like(self._count), self._count)

    def reset(self, mask: Tensor | None = None) -> None:
        """Drop estimates and accumulators for masked environments (episode boundary).

        With ``mask`` ``None`` every environment resets; otherwise only environments
        whose ``mask`` entry is true. A fresh scale estimate is formed per episode.
        """
        if mask is None:
            reset = torch.ones(self.batch_size, device=self._device, dtype=torch.bool)
        else:
            if mask.shape != (self.batch_size,):
                raise ValueError("mask must contain one value per environment")
            reset = mask.to(device=self._device, dtype=torch.bool)
        if not bool(reset.any()):
            return
        row = reset.unsqueeze(-1)
        self._estimate = torch.where(row, torch.ones_like(self._estimate), self._estimate)
        self._valid = self._valid & ~row
        self._sum_obs = torch.where(row, torch.zeros_like(self._sum_obs), self._sum_obs)
        self._sum_cmd = torch.where(row, torch.zeros_like(self._sum_cmd), self._sum_cmd)
        self._eff_window = torch.where(row, torch.zeros_like(self._eff_window), self._eff_window)
        self._last_eff = torch.where(row, torch.zeros_like(self._last_eff), self._last_eff)
        self._last_valid = self._last_valid & ~reset
        self._count = torch.where(reset, torch.zeros_like(self._count), self._count)


__all__ = ["ScaleCorrector"]
