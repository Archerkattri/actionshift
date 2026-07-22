"""Frozen Diffusion Policy backbone exposed as a canonical-action policy.

Wraps the official ManiSkill state-based Diffusion Policy baseline
(``examples/baselines/diffusion_policy``) so a checkpoint trained on the clean
``pd_ee_delta_pose`` interface drives the adapter tournament exactly like the
frozen PPO actor. The receding-horizon action-chunk semantics and the
observation frame-stack are reproduced faithfully from the baseline's
``evaluate`` loop and ``FrameStack`` wrapper:

- the policy re-plans a ``pred_horizon`` chunk and executes ``act_horizon`` of
  those actions before re-planning (open-loop within a chunk), matching
  ``diffusion_policy/evaluate.py``;
- the observation conditioning is the last ``obs_horizon`` frames, filled with
  the reset frame at episode boundaries, matching ``FrameStack``.

The network architecture and DDPM sampler are rebuilt to match the baseline's
``Agent`` (default hyper-parameters), and the EMA weights are loaded — the
weights the baseline itself evaluates and checkpoints on.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

_DP_BASELINE_DIR = (
    Path(__file__).resolve().parents[3]
    / "third_party/maniskill/examples/baselines/diffusion_policy"
)


def _import_unet() -> type[nn.Module]:
    if str(_DP_BASELINE_DIR) not in sys.path:
        sys.path.insert(0, str(_DP_BASELINE_DIR))
    from diffusion_policy.conditional_unet1d import (  # type: ignore[import-not-found]
        ConditionalUnet1D,
    )

    return ConditionalUnet1D  # type: ignore[no-any-return]


@dataclass(frozen=True, slots=True)
class DiffusionPolicyConfig:
    """Architecture + sampler configuration of a trained DP checkpoint.

    Defaults mirror the official baseline ``Args`` so a checkpoint trained with
    the unmodified ``train.py`` reloads exactly. ``observation_dim`` and
    ``action_dim`` are the per-frame state dimension and control dimension.
    """

    observation_dim: int
    action_dim: int
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    num_diffusion_iters: int = 100

    def to_json(self) -> str:
        payload = {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "obs_horizon": self.obs_horizon,
            "act_horizon": self.act_horizon,
            "pred_horizon": self.pred_horizon,
            "diffusion_step_embed_dim": self.diffusion_step_embed_dim,
            "unet_dims": list(self.unet_dims),
            "n_groups": self.n_groups,
            "num_diffusion_iters": self.num_diffusion_iters,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> DiffusionPolicyConfig:
        payload = json.loads(Path(path).read_text())
        payload["unet_dims"] = tuple(payload["unet_dims"])
        return cls(**payload)


class DiffusionPolicyShim:
    """Stateful ``CanonicalPolicy`` around a frozen state-based Diffusion Policy."""

    def __init__(
        self,
        checkpoint: Path,
        config: DiffusionPolicyConfig,
        *,
        device: torch.device | str = "cuda",
        weights: str = "ema_agent",
    ) -> None:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.config = config
        self.device = torch.device(device)
        unet_cls = _import_unet()
        self.net = unet_cls(
            input_dim=config.action_dim,
            global_cond_dim=config.obs_horizon * config.observation_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=list(config.unet_dims),
            n_groups=config.n_groups,
        ).to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = payload.get(weights, payload)
        prefix = "noise_pred_net."
        net_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not net_state:  # a bare noise_pred_net state dict
            net_state = state
        self.net.load_state_dict(net_state)
        self.net.eval()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self._frames: deque[Tensor] | None = None
        self._buffer: Tensor | None = None
        self._buffer_pos = 0

    def reset_state(self) -> None:
        """Drop the observation history and action chunk (fresh episode context)."""
        self._frames = None
        self._buffer = None
        self._buffer_pos = 0

    @torch.no_grad()
    def _plan(self, observation_sequence: Tensor) -> Tensor:
        """Denoise a fresh action chunk from the stacked observation history."""
        batch = observation_sequence.shape[0]
        obs_cond = observation_sequence.flatten(start_dim=1)
        noisy = torch.randn(
            (batch, self.config.pred_horizon, self.config.action_dim),
            device=self.device,
        )
        for step in self.scheduler.timesteps:
            noise_pred = self.net(sample=noisy, timestep=step, global_cond=obs_cond)
            noisy = self.scheduler.step(
                model_output=noise_pred, timestep=step, sample=noisy
            ).prev_sample
        start = self.config.obs_horizon - 1
        end = start + self.config.act_horizon
        return noisy[:, start:end]

    def act(self, observation: Tensor, *, reset_mask: Tensor | None = None) -> Tensor:
        observation = observation.to(self.device)
        horizon = self.config.obs_horizon
        if self._frames is None:
            self._frames = deque(
                (observation.clone() for _ in range(horizon)), maxlen=horizon
            )
            self._buffer = None
        else:
            if reset_mask is not None:
                mask = reset_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
                if bool(mask.any()):
                    for frame in self._frames:
                        frame[mask] = observation[mask]
                    self._buffer = None  # boundary: re-plan from the fresh frame
            self._frames.append(observation.clone())
        need_plan = self._buffer is None or self._buffer_pos >= self._buffer.shape[1]
        if need_plan:
            observation_sequence = torch.stack(list(self._frames), dim=1)
            self._buffer = self._plan(observation_sequence)
            self._buffer_pos = 0
        assert self._buffer is not None
        action = self._buffer[:, self._buffer_pos]
        self._buffer_pos += 1
        return action
