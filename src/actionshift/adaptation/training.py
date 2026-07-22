"""Trained-adapter foundation: training distribution, collection, OSI regressor.

The training contract distribution is sampled over the full 6-DoF grammar and is
made disjoint from the frozen evaluation contracts by hash rejection, so no
trained method ever sees an evaluation contract during training. Supervision
labels (the sampled contract) exist only at training time, matching the UP-OSI
privilege declared in the method registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract

_LAG_CLASSES = (0, 1, 2, 4)


def sample_training_contract(
    generator: torch.Generator, *, excluded_hashes: frozenset[str], max_lag: int = 2
) -> ActionContract:
    """Sample a 6-DoF contract, rejecting any frozen evaluation contract."""
    lags = tuple(lag for lag in _LAG_CLASSES if lag <= max_lag)
    for _ in range(256):
        permutation = tuple(torch.randperm(6, generator=generator).tolist())
        sign = tuple(
            int(s) for s in (torch.randint(0, 2, (6,), generator=generator) * 2 - 1)
        )
        scale = tuple(
            float(s)
            for s in torch.exp(
                torch.empty(6).uniform_(
                    float(torch.log(torch.tensor(0.5))),
                    float(torch.log(torch.tensor(2.0))),
                    generator=generator,
                )
            )
        )
        def _coin() -> bool:
            return bool(torch.randint(0, 2, (1,), generator=generator).item())

        contract = ActionContract(
            permutation=permutation,
            sign=sign,
            scale=scale,
            target="absolute" if _coin() else "delta",
            frame="tool" if _coin() else "base",
            lag=lags[int(torch.randint(0, len(lags), (1,), generator=generator).item())],
            gripper_inverted=_coin(),
        )
        if contract_hash(contract) not in excluded_hashes:
            return contract
    raise RuntimeError("could not sample a training contract outside the excluded set")


def contract_targets(contract: ActionContract) -> dict[str, Tensor]:
    """Supervision targets for one contract."""
    return {
        "permutation": torch.tensor(contract.permutation, dtype=torch.long),
        "sign": torch.tensor([1.0 if s > 0 else 0.0 for s in contract.sign]),
        "log_scale": torch.log(torch.tensor(contract.scale, dtype=torch.float32)),
        "flags": torch.tensor(
            [
                1.0 if contract.target == "absolute" else 0.0,
                1.0 if contract.frame == "tool" else 0.0,
                1.0 if contract.gripper_inverted else 0.0,
            ]
        ),
        "lag": torch.tensor(_LAG_CLASSES.index(contract.lag), dtype=torch.long),
    }


@dataclass(frozen=True, slots=True)
class HistoryWindow:
    """One supervised training sample: an interaction history plus its contract."""

    history: Tensor
    contract: ActionContract


def collect_windows(
    step_environment: Callable[[Tensor], Tensor],
    *,
    contract: ActionContract,
    batch_size: int,
    window: int,
    windows_per_contract: int,
    action_source: Callable[[], Tensor],
) -> Iterator[HistoryWindow]:
    """Roll out an environment under one hidden contract and cut history windows.

    ``step_environment`` maps a raw (batch, 7) action to the observed (batch, 6)
    response; ``action_source`` supplies the raw actions actually sent (probe
    pulses, policy actions, or random excitation). Histories are (window, 13)
    stacks of [raw, response].
    """
    steps = window + windows_per_contract - 1
    raws, responses = [], []
    for _ in range(steps):
        raw = action_source()
        response = step_environment(raw)
        raws.append(raw)
        responses.append(response)
    for start in range(windows_per_contract):
        for row in range(batch_size):
            history = torch.cat(
                (
                    torch.stack([raws[start + k][row] for k in range(window)]),
                    torch.stack([responses[start + k][row] for k in range(window)]),
                ),
                dim=-1,
            )
            yield HistoryWindow(history=history, contract=contract)


def history_features(history: Tensor) -> Tensor:
    """Lagged response/raw cross-correlation features (the OSI inductive bias).

    For the linear decode family, the lagged least-squares map ``M_l`` from
    raw pose channels to responses satisfies ``M_l[i, j] = sign_i * scale_i``
    exactly when ``j == permutation[i]`` and ``l`` is the contract lag — the
    sufficient statistic a flat MLP fails to discover from raw windows (and raw
    correlations estimate too noisily at short windows: cross-talk ~1/sqrt(W)
    rivals the smallest scale signal). Features: one pinv-solved 6x6 map per
    candidate lag, per-channel raw excitation power, and mean |response|.
    """
    if history.ndim < 2 or history.shape[-1] != 13:
        raise ValueError("history must end in (window, 13) [raw(7) | response(6)]")
    raw = history[..., :7]
    pose_raw = raw[..., :6]
    response = history[..., 7:]
    window = history.shape[-2]
    blocks = []
    for lag in _LAG_CLASSES:
        samples = window - lag - 1
        if samples < 12:
            blocks.append(torch.zeros((*history.shape[:-2], 72), dtype=history.dtype))
            continue
        current = pose_raw[..., 1 : window - lag, :]
        previous = pose_raw[..., : window - lag - 1, :]
        regressors = torch.cat((current, previous), dim=-1)
        aligned_response = response[..., lag + 1 :, :]
        gram = regressors.transpose(-1, -2) @ regressors
        ridge = 1e-2 * gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(
            1e-8
        ).unsqueeze(-1) * torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        solved = torch.linalg.solve(
            gram + ridge, regressors.transpose(-1, -2) @ aligned_response
        )
        blocks.append(solved.transpose(-1, -2).flatten(-2))
    power = raw.square().mean(dim=-2)
    magnitude = response.abs().mean(dim=-2)
    return torch.cat((*blocks, power, magnitude), dim=-1)


def _cell_tensor(history: Tensor) -> Tensor:
    """Stack the joint lagged maps as per-cell channels: (..., 6, 6, 2L).

    Per lag, channel 0 is the current-action coefficient block (the PS estimate
    for both target families) and channel 1 the previous-action block (~ -PS for
    absolute targets, ~0 for delta — which is what identifies the target flag).
    """
    features = history_features(history)
    lag_count = len(_LAG_CLASSES)
    blocks = features[..., : 72 * lag_count]
    reshaped = blocks.reshape(*features.shape[:-1], lag_count, 6, 2, 6)
    return reshaped.movedim(-4, -1).movedim(-3, -1).flatten(-2)


class OsiRegressor(nn.Module):
    """History -> contract-parameter heads (UP-OSI-style local baseline).

    Permutation-equivariant by construction: one shared scorer is applied to
    every (semantic, raw) cell of the lagged least-squares maps, and one shared
    row head produces sign/scale per semantic channel — so generalization to
    permutations never seen in training holds structurally instead of being
    memorized.
    """

    def __init__(self, *, window: int, hidden: int = 64) -> None:
        super().__init__()
        self.window = window
        lag_count = len(_LAG_CLASSES)
        cell_channels = 2 * lag_count
        cell_inputs = 3 * 2 * cell_channels
        self.cell_scorer = nn.Sequential(
            nn.Linear(cell_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        row_inputs = 2 * 2 * cell_channels
        self.row_head = nn.Sequential(
            nn.Linear(row_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )
        global_inputs = 2 * cell_channels + 7 + 6
        self.global_head = nn.Sequential(
            nn.Linear(global_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3 + len(_LAG_CLASSES)),
        )

    def forward(self, history: Tensor) -> dict[str, Tensor]:
        features = history_features(history)
        cells = _cell_tensor(history)
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
        permutation = self.cell_scorer(cell_inputs).squeeze(-1)
        strongest = signed_and_magnitude.gather(
            -2,
            cells.abs()
            .sum(dim=-1, keepdim=True)
            .argmax(dim=-2, keepdim=True)
            .expand(*cells.shape[:-2], 1, signed_and_magnitude.shape[-1]),
        ).squeeze(-2)
        row_inputs = torch.cat(
            (strongest, signed_and_magnitude.amax(dim=-2)), dim=-1
        )
        row_outputs = self.row_head(row_inputs)
        pooled = torch.cat(
            (
                cells.abs().amax(dim=(-3, -2)),
                cells.abs().mean(dim=(-3, -2)),
                features[..., 72 * len(_LAG_CLASSES) :],
            ),
            dim=-1,
        )
        global_outputs = self.global_head(pooled)
        return {
            "permutation": permutation,
            "sign": row_outputs[..., 0],
            "log_scale": row_outputs[..., 1],
            "flags": global_outputs[..., :3],
            "lag": global_outputs[..., 3:],
        }


def osi_loss(predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    cross_entropy = nn.functional.cross_entropy
    binary = nn.functional.binary_cross_entropy_with_logits
    loss = cross_entropy(
        predictions["permutation"].reshape(-1, 6), targets["permutation"].reshape(-1)
    )
    loss = loss + binary(predictions["sign"], targets["sign"])
    loss = loss + nn.functional.mse_loss(predictions["log_scale"], targets["log_scale"])
    loss = loss + binary(predictions["flags"], targets["flags"])
    loss = loss + cross_entropy(predictions["lag"], targets["lag"])
    return loss


def decode_prediction(predictions: dict[str, Tensor]) -> ActionContract:
    """Greedy conflict-free assignment from one un-batched prediction."""
    scores = predictions["permutation"].detach().clone()
    if scores.shape != (6, 6):
        raise ValueError("decode_prediction expects an un-batched prediction")
    permutation = [-1] * 6
    available = set(range(6))
    order = torch.argsort(scores.max(dim=1).values, descending=True).tolist()
    for semantic in order:
        ranked = torch.argsort(scores[semantic], descending=True).tolist()
        choice = next(raw for raw in ranked if raw in available)
        permutation[semantic] = choice
        available.discard(choice)
    sign = tuple(1 if float(s) > 0 else -1 for s in predictions["sign"])
    scale = tuple(
        float(s) for s in predictions["log_scale"].exp().clamp(0.4, 2.5)
    )
    flags = predictions["flags"]
    return ActionContract(
        permutation=tuple(permutation),
        sign=sign,
        scale=scale,
        target="absolute" if float(flags[0]) > 0 else "delta",
        frame="tool" if float(flags[1]) > 0 else "base",
        lag=_LAG_CLASSES[int(predictions["lag"].argmax())],
        gripper_inverted=float(flags[2]) > 0,
    )


def train_osi(
    model: OsiRegressor,
    samples: list[HistoryWindow],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Supervised training over collected windows; returns per-epoch losses."""
    if not samples or epochs <= 0 or batch_size <= 0:
        raise ValueError("samples, epochs, and batch_size must be nonempty/positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    histories = torch.stack([sample.history for sample in samples])
    target_list = [contract_targets(sample.contract) for sample in samples]
    targets = {
        key: torch.stack([target[key] for target in target_list])
        for key in target_list[0]
    }
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _ in range(epochs):
        ordering = torch.randperm(len(samples), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(samples), batch_size):
            indices = ordering[start : start + batch_size]
            predictions = model(histories[indices])
            loss = osi_loss(
                predictions, {key: value[indices] for key, value in targets.items()}
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("OSI loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / batches)
    return losses


class OsiAdapter:
    """Eval-time UP-OSI-style adapter: rolling history -> point-estimate encode.

    Unprivileged at evaluation: it sees only its own raw actions and calibrated
    responses. Until the history fills, it passes the canonical action through
    unchanged (no-adapt behavior), then re-estimates the contract every step.
    """

    name = "up_osi"

    def __init__(self, model: OsiRegressor, *, batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size
        self._history: Tensor | None = None
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
            if estimate is None:
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
        window = self.model.window
        step = torch.cat(
            (raw_action.detach().cpu(), observed_response.detach().cpu()), dim=-1
        ).to(torch.float32)
        if self._history is None:
            self._history = torch.zeros((self.batch_size, window, 13))
        if invalid_mask is not None:
            boundary = invalid_mask.detach().cpu().to(torch.bool)
            self._filled = torch.where(
                boundary, torch.zeros_like(self._filled), self._filled
            )
            self._history[boundary] = 0.0
            if self._tracked_target is not None:
                self._tracked_target[boundary.to(self._tracked_target.device)] = 0.0
            for row in boundary.nonzero(as_tuple=True)[0].tolist():
                self._estimates[row] = None
            valid = ~boundary
        else:
            valid = torch.ones(self.batch_size, dtype=torch.bool)
        rows = valid.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return
        self._history[rows] = torch.roll(self._history[rows], shifts=-1, dims=1)
        self._history[rows, -1] = step[rows]
        self._filled[rows] = (self._filled[rows] + 1).clamp(max=window)
        ready = rows[(self._filled[rows] >= window)]
        if ready.numel() == 0:
            return
        with torch.no_grad():
            predictions = self.model(self._history[ready])
        for position, row in enumerate(ready.tolist()):
            single = {key: value[position] for key, value in predictions.items()}
            self._estimates[row] = decode_prediction(single)
