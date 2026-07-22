"""Delay-aware augmented-state PPO backbone: agent, history buffer, and eval loop.

The delay-aware backbone is a NEW task policy (not a frozen Gate 0 checkpoint).
Its observation is the base state augmented with the last ``history`` canonical
actions it emitted; trained under randomized action lag it can compensate for
delay because the augmented state restores the Markov property (Katsikopoulos &
Engelbrecht 2003 augmented-state formulation of the delayed MDP).

This module holds the pieces shared by training (``experiments/train_delay_aware.py``)
and evaluation (``experiments/run_delay_slice.py``):

* ``ActionHistoryBuffer`` -- the rolling per-environment canonical-action buffer
  used for observation augmentation, plus a per-environment lag executor that
  reduces to ``contracts.transforms.ActionLag`` when every environment shares one
  lag (verified in ``tests/test_delay_aware.py``);
* ``DelayAwarePpoAgent`` -- the checkpoint-compatible network (identical body to
  ``ppo_parity.PpoAgent`` but with the augmented input width);
* ``evaluate_delay_aware_adapter`` -- an episode-accounting evaluation loop that
  mirrors ``adaptation.maniskill.evaluate_adapter`` exactly, threading the same
  auto-reset masks, but feeds the augmented observation to the agent.

The eval loop is adapter-generic: pass ``OracleAdapter`` for the oracle-encode +
delay-aware backbone method, or ``ExactBeliefAdapter`` for the belief + delay-aware
method. The env applies the true hidden contract (including its lag) exactly as in
the Gate 1 / tournament path; the delay-aware agent only changes what the policy
observes.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from actionshift.adaptation.adapters import ContractAdapter
from actionshift.adaptation.calibration import (
    ResponseCalibration,
    response_from_observations,
)
from actionshift.adaptation.maniskill import AdaptationEpisode
from actionshift.benchmarking.ppo_parity import _layer_init, _make_environment
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

DEFAULT_HISTORY = 4
"""Number of past canonical actions in the augmented state (K=4 covers lag 4)."""

DEFAULT_LAG_SET: tuple[int, ...] = (0, 1, 2, 4)
"""Per-episode randomized lags used in training; matches the long-lag split {2, 4}."""


class ActionHistoryBuffer:
    """Rolling per-environment buffer of the last ``history`` canonical actions.

    Index convention: ``buffer[:, 0]`` is the most recent past action (a_{t-1}),
    ``buffer[:, k]`` is a_{t-1-k}. The buffer is zero at reset, so lagged actions
    early in an episode are neutral zeros exactly like ``ActionLag``.
    """

    def __init__(
        self,
        num_envs: int,
        action_dimension: int,
        *,
        history: int = DEFAULT_HISTORY,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0 or action_dimension <= 0 or history <= 0:
            raise ValueError("num_envs, action_dimension, and history must be positive")
        self.num_envs = num_envs
        self.action_dimension = action_dimension
        self.history = history
        self._buffer = torch.zeros(
            (num_envs, history, action_dimension), device=device, dtype=dtype
        )

    def augment(self, observation: Tensor) -> Tensor:
        """Concatenate the flattened action history onto the base observation."""
        flat = self._buffer.reshape(self.num_envs, self.history * self.action_dimension)
        return torch.cat((observation, flat.to(observation.dtype)), dim=-1)

    def push(self, canonical_action: Tensor) -> None:
        """Insert the newest canonical action, dropping the oldest."""
        if canonical_action.shape != (self.num_envs, self.action_dimension):
            raise ValueError("canonical_action must be (num_envs, action_dimension)")
        newest = canonical_action.to(self._buffer.dtype).unsqueeze(1)
        self._buffer = torch.cat((newest, self._buffer[:, :-1]), dim=1)

    def reset(self, mask: Tensor | None) -> None:
        """Zero the history for environments flagged in ``mask`` (per-env boolean)."""
        if mask is None:
            return
        flat = mask.to(device=self._buffer.device, dtype=torch.bool).reshape(self.num_envs)
        selector = flat.reshape(self.num_envs, 1, 1)
        self._buffer = torch.where(selector, torch.zeros_like(self._buffer), self._buffer)

    def lag_execute(self, canonical_action: Tensor, lag: Tensor) -> Tensor:
        """Return the action to execute now under a per-environment lag.

        ``executed[i] = canonical_action[i]`` when ``lag[i] == 0`` else the action
        emitted ``lag[i]`` steps ago (``buffer[i, lag[i] - 1]``). This must be called
        BEFORE :meth:`push` so the buffer still holds a_{t-1} .. a_{t-history}. For a
        uniform lag it is bit-identical to ``ActionLag`` (tested).
        """
        if canonical_action.shape != (self.num_envs, self.action_dimension):
            raise ValueError("canonical_action must be (num_envs, action_dimension)")
        if lag.shape != (self.num_envs,):
            raise ValueError("lag must contain one value per environment")
        lag_long = lag.to(device=self._buffer.device, dtype=torch.long)
        if int(lag_long.max().item()) > self.history:
            raise ValueError("lag exceeds available history")
        index = (lag_long - 1).clamp(min=0)
        gathered = self._buffer[torch.arange(self.num_envs, device=self._buffer.device), index]
        zero_lag = (lag_long == 0).unsqueeze(-1)
        return torch.where(
            zero_lag, canonical_action, gathered.to(canonical_action.dtype)
        )


class DelayAwarePpoAgent(nn.Module):
    """PPO network whose input is the base observation plus the action history.

    Body is identical to ``ppo_parity.PpoAgent`` (three 256-wide Tanh layers); only
    the input width differs (``base_observation_dimension + history * action_dimension``).
    """

    def __init__(
        self,
        base_observation_dimension: int,
        action_dimension: int,
        *,
        history: int = DEFAULT_HISTORY,
    ) -> None:
        super().__init__()
        self.base_observation_dimension = base_observation_dimension
        self.action_dimension = action_dimension
        self.history = history
        augmented = base_observation_dimension + history * action_dimension
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(augmented, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(augmented, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, action_dimension), 0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dimension) * -0.5)

    def deterministic_action(self, augmented_observation: Tensor) -> Tensor:
        return cast(Tensor, self.actor_mean(augmented_observation))


def evaluate_delay_aware_adapter(
    checkpoint: Path,
    *,
    task: str,
    method: str,
    adapter: ContractAdapter,
    contract: ActionContract,
    calibration: ResponseCalibration,
    seed: int,
    history: int = DEFAULT_HISTORY,
    episodes: int = 100,
    num_envs: int = 16,
) -> list[AdaptationEpisode]:
    """Evaluate a delay-aware backbone through an adapter on one hidden contract.

    Episode accounting mirrors ``adaptation.maniskill.evaluate_adapter`` exactly:
    the env is wrapped ``noadapt_nonidentity`` (the wrapper applies the true
    contract, including its lag), the adapter maps canonical -> raw, and reset /
    invalid masks are threaded identically. The single difference is that the agent
    observes ``[base_obs, last-K canonical actions]``.
    """
    if episodes <= 0 or num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    environment = _make_environment(task, "noadapt_nonidentity", num_envs, contract)
    try:
        observation, _ = environment.reset(seed=seed)
        base_observation_dimension = int(np.prod(environment.single_observation_space.shape))
        action_dimension = int(np.prod(environment.single_action_space.shape))
        agent = DelayAwarePpoAgent(
            base_observation_dimension, action_dimension, history=history
        ).to(environment.device)
        payload = torch.load(checkpoint, map_location=environment.device, weights_only=True)
        agent.load_state_dict(payload)
        agent.eval()
        low = torch.as_tensor(
            environment.single_action_space.low,
            device=environment.device,
            dtype=observation.dtype,
        )
        high = torch.as_tensor(
            environment.single_action_space.high,
            device=environment.device,
            dtype=observation.dtype,
        )
        buffer = ActionHistoryBuffer(
            num_envs,
            action_dimension,
            history=history,
            device=environment.device,
            dtype=observation.dtype,
        )
        checkpoint_digest = sha256_file(checkpoint)
        contract_digest = contract_hash(contract)
        records: list[AdaptationEpisode] = []
        pending_reset: Tensor | None = None
        previous_observation = observation
        # Probe-cost accounting mirrors ``adaptation.maniskill.evaluate_adapter``:
        # for non-probing adapters (oracle / exact-belief) ``last_probe_mask`` is
        # absent, so both accumulators stay zero and the recorded probe fields keep
        # their defaults -- the delay-aware oracle/belief slices are unchanged.
        probe_steps_used = torch.zeros(num_envs, dtype=torch.long)
        probe_displacement = torch.zeros(num_envs, dtype=torch.float64)
        while len(records) < episodes:
            with torch.no_grad():
                augmented = buffer.augment(observation)
                canonical = torch.clamp(agent.deterministic_action(augmented), low, high)
                raw = adapter.encode(canonical)
                probe_mask = getattr(adapter, "last_probe_mask", None)
                buffer.push(canonical)
                observation, _, _, _, info = environment.step(raw)
                boundary: Tensor | None = None
                if "final_info" in info:
                    boundary = (
                        torch.as_tensor(info["_final_info"], dtype=torch.bool)
                        .reshape(-1)
                        .to(environment.device)
                    )
                response = response_from_observations(
                    calibration, previous_observation, observation
                )
                if probe_mask is not None:
                    probe_cpu = probe_mask.detach().reshape(-1).cpu()
                    probe_steps_used += probe_cpu.long()
                    measurable = probe_cpu
                    if boundary is not None:
                        measurable = probe_cpu & ~boundary.detach().reshape(-1).cpu()
                    probe_displacement += torch.where(
                        measurable,
                        response[..., :3].norm(dim=-1).detach().reshape(-1).cpu(),
                        torch.zeros(num_envs),
                    ).to(torch.float64)
                adapter.observe(
                    raw,
                    response,
                    reset_mask=pending_reset,
                    invalid_mask=boundary,
                )
                buffer.reset(boundary)
                previous_observation = observation
                pending_reset = boundary
            if boundary is None:
                continue
            metrics = info["final_info"]["episode"]
            mask = boundary.cpu()
            indices = mask.nonzero(as_tuple=True)[0]
            success = torch.as_tensor(metrics["success_once"]).reshape(-1).cpu()[mask]
            returns = torch.as_tensor(metrics["return"]).reshape(-1).cpu()[mask]
            lengths = torch.as_tensor(metrics["episode_len"]).reshape(-1).cpu()[mask]
            for environment_index, succeeded, task_return, length in zip(
                indices, success, returns, lengths, strict=True
            ):
                if len(records) == episodes:
                    break
                records.append(
                    AdaptationEpisode(
                        task=task,
                        method=method,
                        episode_index=len(records),
                        seed=seed,
                        success=bool(succeeded.item()),
                        task_return=float(task_return.item()),
                        episode_steps=int(length.item()),
                        checkpoint_sha256=checkpoint_digest,
                        contract_sha256=contract_digest,
                        probe_steps=int(probe_steps_used[environment_index].item()),
                        probe_displacement=float(
                            probe_displacement[environment_index].item()
                        ),
                    )
                )
            probe_steps_used[indices] = 0
            probe_displacement[indices] = 0.0
        return records
    finally:
        environment.close()
