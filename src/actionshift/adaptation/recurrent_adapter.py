"""Recurrent episode-length identification adapter (the motivated unprivileged method).

The tournament established the discriminative fact this file acts on: passive
short-window learned identification (UP-OSI-style, window 14) is
information-limited (calibration R2 0.09-0.34, per-step SNR ~1), while exact
belief succeeds by accumulating evidence across the WHOLE episode. The motivated
unprivileged move is therefore EPISODE-LENGTH accumulation: a recurrent adapter
that integrates every ``[raw, response]`` transition since the episode began and
continuously refines a contract estimate.

Design (both inductive biases kept, because both were proven necessary):

* a RUNNING ridge least-squares estimate of the lagged joint-regression maps,
  computed incrementally over the whole episode so far. These are exactly
  ``training.history_features``'s sufficient statistics, but accumulated
  step-by-step instead of over a fixed window, so they sharpen monotonically as
  the episode proceeds (the "accumulate over 50 steps" mechanism);
* the permutation-equivariant head from ``training.OsiRegressor`` reading those
  running maps (flat MLPs memorize permutations; the equivariant head
  generalizes by construction);
* a GRU carried across the episode that refines the global (target/frame/gripper
  flags and lag) heads from the running summary, the learned temporal-refinement
  component.

Nothing here ever sees the true contract at evaluation. Supervision labels exist
only at training time, exactly like the UP-OSI privilege declared in the
registry.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from actionshift.adaptation.training import (
    _LAG_CLASSES,
    contract_targets,
    decode_prediction,
    osi_loss,
)
from actionshift.contracts.types import ActionContract

_LAG_COUNT = len(_LAG_CLASSES)
_MAX_LAG = max(_LAG_CLASSES)
_CELL_CHANNELS = 2 * _LAG_COUNT
_POOLED_DIM = 2 * _CELL_CHANNELS + 7 + 6
_DEFAULT_MIN_SAMPLES = 6
_DEFAULT_WARMUP = 8


def _cells_from_features(features: Tensor) -> Tensor:
    """Reshape the running lagged maps into per-cell channels ``(..., 6, 6, 2L)``.

    Bit-for-bit the layout of ``training._cell_tensor``, but read from a running
    feature vector rather than recomputed from a fixed window: per lag, channel 0
    is the current-action coefficient block and channel 1 the previous-action
    block (~ -block0 for absolute targets, ~0 for delta).
    """
    blocks = features[..., : 72 * _LAG_COUNT]
    reshaped = blocks.reshape(*features.shape[:-1], _LAG_COUNT, 6, 2, 6)
    return reshaped.movedim(-4, -1).movedim(-3, -1).flatten(-2)


class RunningLagFeatures:
    """Incremental ridge least-squares of the lagged joint-regression maps.

    Maintains, per environment and per candidate lag, the Gram accumulators
    ``X^T X`` and ``X^T Y`` for the regression ``response_t ~ [raw_{t-l},
    raw_{t-l-1}]`` plus running raw-excitation power and mean ``|response|``. At
    any step it solves the ridge-regularized maps and emits the same
    301-dimensional feature vector as ``training.history_features`` (72 per lag +
    7 power + 6 magnitude), so the two share one equivariant head. The only
    difference is that evidence accumulates over the whole episode, not a window.
    """

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        ridge: float = 1e-2,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._device = torch.device(device)
        self._dtype = dtype
        self._ridge = ridge
        self._min_samples = min_samples
        self._eye = torch.eye(12, device=self._device, dtype=dtype)
        self._pose_history = torch.zeros(
            (batch_size, _MAX_LAG + 2, 6), device=self._device, dtype=dtype
        )
        self._xtx = torch.zeros(
            (batch_size, _LAG_COUNT, 12, 12), device=self._device, dtype=dtype
        )
        self._xty = torch.zeros(
            (batch_size, _LAG_COUNT, 12, 6), device=self._device, dtype=dtype
        )
        self._counts = torch.zeros(
            (batch_size, _LAG_COUNT), device=self._device, dtype=torch.long
        )
        self._power_sum = torch.zeros((batch_size, 7), device=self._device, dtype=dtype)
        self._magnitude_sum = torch.zeros(
            (batch_size, 6), device=self._device, dtype=dtype
        )
        self._steps = torch.zeros(batch_size, device=self._device, dtype=torch.long)

    def reset(self, mask: Tensor | None = None) -> None:
        """Reset accumulators for the flagged environments (all when ``None``)."""
        if mask is None:
            selector: Tensor | slice = slice(None)
        else:
            selector = mask.to(device=self._device, dtype=torch.bool)
        self._pose_history[selector] = 0.0
        self._xtx[selector] = 0.0
        self._xty[selector] = 0.0
        self._counts[selector] = 0
        self._power_sum[selector] = 0.0
        self._magnitude_sum[selector] = 0.0
        self._steps[selector] = 0

    def push(
        self,
        raw_action: Tensor,
        observed_response: Tensor,
        *,
        active: Tensor | None = None,
    ) -> Tensor:
        """Fold one transition into the accumulators and emit running features.

        ``active`` gates the update per environment: an environment whose
        transition crosses an auto-reset boundary is inactive this step, so its
        (freshly reset) accumulator is left untouched and the new episode's first
        evidence lands on the following step.
        """
        raw = raw_action.to(device=self._device, dtype=self._dtype)
        response = observed_response.to(device=self._device, dtype=self._dtype)
        if raw.shape != (self.batch_size, 7):
            raise ValueError("raw_action must be (batch_size, 7)")
        if response.shape != (self.batch_size, 6):
            raise ValueError("observed_response must be (batch_size, 6)")
        if active is None:
            gate = torch.ones(self.batch_size, device=self._device, dtype=self._dtype)
        else:
            gate = active.to(device=self._device, dtype=torch.bool).to(self._dtype)
        pose = raw[:, :6]
        rolled = torch.roll(self._pose_history, shifts=-1, dims=1)
        rolled[:, -1] = pose
        self._pose_history = torch.where(
            gate[:, None, None].to(torch.bool), rolled, self._pose_history
        )
        self._steps = self._steps + gate.long()
        for index, lag in enumerate(_LAG_CLASSES):
            contributing = (gate > 0) & (self._steps >= (lag + 2))
            current = self._pose_history[:, -1 - lag]
            previous = self._pose_history[:, -1 - lag - 1]
            regressor = torch.cat((current, previous), dim=-1)
            gain = contributing.to(self._dtype)[:, None, None]
            self._xtx[:, index] += gain * (
                regressor[:, :, None] * regressor[:, None, :]
            )
            self._xty[:, index] += gain * (
                regressor[:, :, None] * response[:, None, :]
            )
            self._counts[:, index] += contributing.long()
        self._power_sum += gate[:, None] * raw.square()
        self._magnitude_sum += gate[:, None] * response.abs()
        return self._features()

    def _features(self) -> Tensor:
        blocks = []
        for index in range(_LAG_COUNT):
            gram = self._xtx[:, index]
            scale = (
                gram.diagonal(dim1=-2, dim2=-1)
                .mean(dim=-1, keepdim=True)
                .clamp_min(1e-8)
                .unsqueeze(-1)
            )
            ridge = self._ridge * scale * self._eye
            solved = torch.linalg.solve(gram + ridge, self._xty[:, index])
            block = solved.transpose(-1, -2).flatten(-2)
            enough = (self._counts[:, index] >= self._min_samples).to(self._dtype)
            blocks.append(block * enough[:, None])
        denominator = self._steps.clamp_min(1).to(self._dtype)[:, None]
        power = self._power_sum / denominator
        magnitude = self._magnitude_sum / denominator
        return torch.cat((*blocks, power, magnitude), dim=-1)


def _equivariant_heads(
    cell_scorer: nn.Module, row_head: nn.Module, features: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Permutation-equivariant permutation/sign/scale plus the global pool.

    Identical computation to ``training.OsiRegressor.forward`` up to the global
    head, factored so the recurrent module can drive the same shared cell and row
    heads from running features. Returns ``(permutation, sign, log_scale,
    pooled)``.
    """
    cells = _cells_from_features(features)
    signed_and_magnitude = torch.cat((cells, cells.abs()), dim=-1)
    row_context = signed_and_magnitude.amax(dim=-2, keepdim=True).expand_as(
        signed_and_magnitude
    )
    column_context = signed_and_magnitude.amax(dim=-3, keepdim=True).expand_as(
        signed_and_magnitude
    )
    cell_inputs = torch.cat(
        (signed_and_magnitude, row_context, column_context), dim=-1
    )
    permutation = cell_scorer(cell_inputs).squeeze(-1)
    strongest = signed_and_magnitude.gather(
        -2,
        cells.abs()
        .sum(dim=-1, keepdim=True)
        .argmax(dim=-2, keepdim=True)
        .expand(*cells.shape[:-2], 1, signed_and_magnitude.shape[-1]),
    ).squeeze(-2)
    row_inputs = torch.cat((strongest, signed_and_magnitude.amax(dim=-2)), dim=-1)
    row_outputs = row_head(row_inputs)
    pooled = torch.cat(
        (
            cells.abs().amax(dim=(-3, -2)),
            cells.abs().mean(dim=(-3, -2)),
            features[..., 72 * _LAG_COUNT :],
        ),
        dim=-1,
    )
    return permutation, row_outputs[..., 0], row_outputs[..., 1], pooled


class RecurrentOsiRegressor(nn.Module):
    """Running-map equivariant heads + a GRU refining the global heads over time.

    ``forward`` consumes a whole ``(steps, batch, 301)`` running-feature sequence
    and returns per-timestep predictions for deep supervision; ``step`` advances
    one timestep with an external hidden state for the eval-time adapter. The
    permutation/sign/scale heads read the running maps directly (they sharpen as
    the maps accumulate), while the flags/lag heads read the GRU hidden state so
    they integrate the full sequence.
    """

    def __init__(self, *, hidden: int = 64, gru_hidden: int = 64) -> None:
        super().__init__()
        self.gru_hidden = gru_hidden
        cell_inputs = 3 * 2 * _CELL_CHANNELS
        self.cell_scorer = nn.Sequential(
            nn.Linear(cell_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        row_inputs = 2 * 2 * _CELL_CHANNELS
        self.row_head = nn.Sequential(
            nn.Linear(row_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )
        self.gru = nn.GRU(input_size=_POOLED_DIM, hidden_size=gru_hidden)
        self.global_head = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3 + _LAG_COUNT),
        )

    def forward(self, feature_sequence: Tensor) -> dict[str, Tensor]:
        if feature_sequence.ndim != 3:
            raise ValueError("feature_sequence must be (steps, batch, features)")
        steps, batch, _ = feature_sequence.shape
        flat = feature_sequence.reshape(steps * batch, -1)
        permutation, sign, log_scale, pooled = _equivariant_heads(
            self.cell_scorer, self.row_head, flat
        )
        pooled_sequence = pooled.reshape(steps, batch, _POOLED_DIM)
        gru_out, _ = self.gru(pooled_sequence)
        globals_ = self.global_head(gru_out)
        return {
            "permutation": permutation.reshape(steps, batch, 6, 6),
            "sign": sign.reshape(steps, batch, 6),
            "log_scale": log_scale.reshape(steps, batch, 6),
            "flags": globals_[..., :3],
            "lag": globals_[..., 3:],
        }

    def initial_hidden(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return torch.zeros((1, batch_size, self.gru_hidden), device=device, dtype=dtype)

    def step(
        self, features: Tensor, hidden: Tensor
    ) -> tuple[dict[str, Tensor], Tensor]:
        """One-timestep prediction with an external GRU hidden state."""
        if features.ndim != 2:
            raise ValueError("features must be (batch, features)")
        permutation, sign, log_scale, pooled = _equivariant_heads(
            self.cell_scorer, self.row_head, features
        )
        gru_out, new_hidden = self.gru(pooled.unsqueeze(0), hidden)
        globals_ = self.global_head(gru_out.squeeze(0))
        predictions = {
            "permutation": permutation,
            "sign": sign,
            "log_scale": log_scale,
            "flags": globals_[..., :3],
            "lag": globals_[..., 3:],
        }
        return predictions, new_hidden


def recurrent_loss(
    predictions: dict[str, Tensor], contracts: list[ActionContract]
) -> Tensor:
    """Deep-supervised loss: the OSI loss at every timestep of the sequence.

    ``predictions`` carry a leading ``(steps, batch)`` pair; each sequence's
    contract label is broadcast across all timesteps so early-step estimates also
    receive gradient.
    """
    steps = predictions["permutation"].shape[0]
    batch = predictions["permutation"].shape[1]
    if len(contracts) != batch:
        raise ValueError("one contract label per sequence is required")
    target_list = [contract_targets(contract) for contract in contracts]
    targets = {
        key: torch.stack([target[key] for target in target_list]).to(
            predictions["permutation"].device
        )
        for key in target_list[0]
    }
    flat_predictions = {
        "permutation": predictions["permutation"].reshape(steps * batch, 6, 6),
        "sign": predictions["sign"].reshape(steps * batch, 6),
        "log_scale": predictions["log_scale"].reshape(steps * batch, 6),
        "flags": predictions["flags"].reshape(steps * batch, 3),
        "lag": predictions["lag"].reshape(steps * batch, _LAG_COUNT),
    }
    flat_targets = {
        key: value.unsqueeze(0)
        .expand(steps, *value.shape)
        .reshape(steps * batch, *value.shape[1:])
        for key, value in targets.items()
    }
    return osi_loss(flat_predictions, flat_targets)


class RecurrentOsiAdapter:
    """Eval-time recurrent adapter: episode-length accumulation, MAP encode.

    Unprivileged: it observes only its own raw actions and calibrated responses.
    A running least-squares accumulator and a GRU hidden state are carried per
    environment across the whole episode and reset on auto-reset boundaries with
    the same timing discipline as ``OsiAdapter``. The estimate updates every
    step; ``encode`` passes the canonical action through unchanged until a warmup
    (default 8 steps), then encodes under the current per-environment estimate.
    """

    name = "recurrent"

    def __init__(
        self,
        model: RecurrentOsiRegressor,
        *,
        batch_size: int,
        warmup: int = _DEFAULT_WARMUP,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model.eval()
        self.batch_size = batch_size
        self.warmup = warmup
        self._device = torch.device(device)
        self._accumulator = RunningLagFeatures(
            batch_size, device="cpu", min_samples=min_samples
        )
        self._hidden = model.initial_hidden(
            batch_size, device=self._device, dtype=torch.float32
        )
        self._filled = torch.zeros(batch_size, dtype=torch.long)
        self._estimates: list[ActionContract | None] = [None] * batch_size
        self._tracked_target: Tensor | None = None

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
        encoded = canonical_action.clone()
        rotation = identity_rotation(1, canonical_action.device, canonical_action.dtype)
        for row in range(self.batch_size):
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
        del ee_rotation  # learned identifier does not model the end-effector frame
        if invalid_mask is not None:
            boundary = invalid_mask.detach().cpu().to(torch.bool)
            if boundary.any():
                self._accumulator.reset(boundary)
                self._hidden[:, boundary.to(self._hidden.device)] = 0.0
                self._filled = torch.where(
                    boundary, torch.zeros_like(self._filled), self._filled
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
        rows = valid.nonzero(as_tuple=True)[0].tolist()
        for row in rows:
            single = {key: value[row] for key, value in predictions.items()}
            self._estimates[row] = decode_prediction(single)
