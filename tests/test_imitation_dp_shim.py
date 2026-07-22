"""Unit tests for the receding-horizon + reset logic of the DP backbone shim.

These exercise the stateful control path (frame stacking, action-chunk
dispensing, episode-boundary resets) without constructing the diffusion network,
by building the shim via ``__new__`` and stubbing ``_plan``. The end-to-end wiring
through ``evaluate_adapter`` is covered by the imitation-brittleness experiment.
"""

from __future__ import annotations

import torch

from actionshift.adaptation.dp_policy import DiffusionPolicyConfig, DiffusionPolicyShim


def _stub_shim(obs_dim: int, act_dim: int, obs_horizon: int, act_horizon: int):
    shim = object.__new__(DiffusionPolicyShim)
    shim.config = DiffusionPolicyConfig(
        observation_dim=obs_dim,
        action_dim=act_dim,
        obs_horizon=obs_horizon,
        act_horizon=act_horizon,
        pred_horizon=obs_horizon + act_horizon - 1,
    )
    shim.device = torch.device("cpu")
    shim._frames = None
    shim._buffer = None
    shim._buffer_pos = 0
    plans: list[int] = []

    def fake_plan(observation_sequence: torch.Tensor) -> torch.Tensor:
        # Encode the plan index and the conditioning observation's first value so
        # the test can see when a re-plan happened and on which observation.
        batch = observation_sequence.shape[0]
        index = len(plans)
        plans.append(index)
        newest = observation_sequence[:, -1, 0]  # last frame, first feature
        chunk = torch.zeros(batch, shim.config.act_horizon, act_dim)
        chunk[:, :, 0] = float(index)
        chunk[:, :, 1] = newest.unsqueeze(-1)
        return chunk

    shim._plan = fake_plan  # type: ignore[assignment]
    return shim, plans


def test_replans_every_act_horizon_steps() -> None:
    shim, plans = _stub_shim(obs_dim=4, act_dim=3, obs_horizon=2, act_horizon=4)
    for step in range(9):
        observation = torch.full((2, 4), float(step))
        action = shim.act(observation)
        assert action.shape == (2, 3)
        # plan index advances only every act_horizon steps
        assert action[0, 0].item() == step // 4
    assert plans == [0, 1, 2]  # steps 0-3, 4-7, 8 -> three plans


class _StepReturn:
    def __init__(self, prev_sample: torch.Tensor) -> None:
        self.prev_sample = prev_sample


class _RecordingScheduler:
    """Fake DDPM scheduler: identity step that records the generator it was handed."""

    def __init__(self) -> None:
        self.timesteps = [2, 1, 0]
        self.generators: list[object] = []

    def step(
        self,
        *,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: object,
    ) -> _StepReturn:
        self.generators.append(generator)
        return _StepReturn(sample)


def _plan_only_shim(seed: int | None) -> DiffusionPolicyShim:
    shim = object.__new__(DiffusionPolicyShim)
    shim.config = DiffusionPolicyConfig(
        observation_dim=4,
        action_dim=3,
        obs_horizon=2,
        act_horizon=4,
        pred_horizon=5,
    )
    shim.device = torch.device("cpu")
    shim._generator = (
        None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    )
    shim.scheduler = _RecordingScheduler()  # type: ignore[assignment]
    shim.net = lambda *, sample, timestep, global_cond: torch.zeros_like(sample)  # type: ignore[assignment]
    return shim


def test_plan_threads_seeded_generator_into_every_scheduler_step() -> None:
    shim = _plan_only_shim(seed=0)
    shim._plan(torch.zeros(2, 2, 4))
    scheduler = shim.scheduler
    assert isinstance(scheduler, _RecordingScheduler)
    # The seeded generator is passed to every reverse-diffusion step, not just the
    # initial noise draw — otherwise sampling stays non-reproducible.
    assert len(scheduler.generators) == len(scheduler.timesteps)
    assert all(generator is shim._generator for generator in scheduler.generators)


def test_plan_is_reproducible_across_shims_with_the_same_seed() -> None:
    first = _plan_only_shim(seed=7)._plan(torch.zeros(2, 2, 4))
    same = _plan_only_shim(seed=7)._plan(torch.zeros(2, 2, 4))
    other = _plan_only_shim(seed=8)._plan(torch.zeros(2, 2, 4))
    assert torch.equal(first, same)
    assert not torch.equal(first, other)


def test_reset_mask_forces_replan_and_refills_frames() -> None:
    shim, plans = _stub_shim(obs_dim=4, act_dim=3, obs_horizon=2, act_horizon=4)
    shim.act(torch.zeros(2, 4))  # step 0 -> plan 0
    shim.act(torch.ones(2, 4))  # step 1, still plan 0's chunk
    assert len(plans) == 1
    # Boundary on env 0 only: it must re-plan from the fresh observation.
    reset = torch.tensor([True, False])
    fresh = torch.full((2, 4), 5.0)
    action = shim.act(fresh, reset_mask=reset)
    assert len(plans) == 2  # a re-plan happened
    # newest frame conditioning equals the fresh observation value (5.0)
    assert action[0, 1].item() == 5.0
    # env 0's frame history was refilled with the fresh frame on reset
    assert shim._frames is not None
    for frame in shim._frames:
        assert frame[0, 0].item() == 5.0
