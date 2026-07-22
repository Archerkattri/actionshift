"""Episode-level parity evaluation for frozen official ManiSkill PPO checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from actionshift.contracts.transforms import (
    encode_complete_action,
    quaternion_to_rotation_matrix,
)
from actionshift.contracts.types import ActionContract
from actionshift.envs.wrapper import HiddenContractWrapper
from actionshift.evaluation.provenance import sha256_file

ParityCondition = Literal[
    "unwrapped", "identity", "oracle_nonidentity", "noadapt_nonidentity"
]
RotationMode = Literal["identity", "real"]


def make_tcp_rotation_provider(environment: Any) -> Callable[[], Tensor]:
    """Build a zero-arg provider returning the live tcp rotation matrix.

    Reads the end-effector orientation quaternion (``agent.tcp.pose.q``, ManiSkill
    ``wxyz``) from the underlying environment on demand and converts it to a
    batched ``(num_envs, 3, 3)`` rotation, on the simulator's device. This is the
    real-rotation hook the v2 benchmark variant threads into the hidden wrapper so
    that ``frame="tool"`` contracts are decoded against a genuinely non-identity
    end-effector axis rather than the identity placeholder.
    """

    def provider() -> Tensor:
        quaternion = torch.as_tensor(environment.unwrapped.agent.tcp.pose.q).reshape(-1, 4)
        return quaternion_to_rotation_matrix(quaternion)

    return provider


_ENVIRONMENT_IDS = {
    "pick_cube": "PickCube-v1",
    "push_cube": "PushCube-v1",
    "pull_cube": "PullCube-v1",
    "stack_cube": "StackCube-v1",
    "peg_insertion_side": "PegInsertionSide-v1",
}


def identity_contract() -> ActionContract:
    return ActionContract(
        permutation=tuple(range(6)),
        sign=(1,) * 6,
        scale=(1.0,) * 6,
        target="delta",
        frame="base",
        lag=0,
        gripper_inverted=False,
    )


def oracle_contract() -> ActionContract:
    return ActionContract(
        permutation=(1, 0, 2, 4, 5, 3),
        sign=(-1, 1, -1, 1, -1, 1),
        scale=(0.5, 2.0, 1.5, 0.75, 1.25, 0.6),
        target="delta",
        frame="base",
        lag=0,
        gripper_inverted=True,
    )


def policy_action_for_condition(
    action: Tensor,
    condition: ParityCondition,
    *,
    contract: ActionContract | None = None,
    tracked_target: Tensor | None = None,
    ee_rotation: Tensor | None = None,
) -> Tensor:
    """Encode canonical policy output only when an oracle contract is active.

    ``ee_rotation`` is the current tcp rotation the hidden wrapper will decode
    against this step. ``None`` (default) uses identity, reproducing the v1
    identity-rotation oracle path exactly; the v2 real-rotation variant passes the
    live rotation so the oracle inverts the wrapper's ``frame="tool"`` decode
    against the same axis the wrapper used, keeping parity exact.
    """
    if condition in {"unwrapped", "identity", "noadapt_nonidentity"}:
        return action
    if condition != "oracle_nonidentity":
        raise ValueError(f"unknown parity condition: {condition}")
    batch_shape = action.shape[:-1]
    if ee_rotation is None:
        rotation = torch.eye(3, device=action.device, dtype=action.dtype).expand(
            *batch_shape, 3, 3
        )
    else:
        rotation = ee_rotation.to(device=action.device, dtype=action.dtype)
        if rotation.shape == (3, 3):
            rotation = rotation.expand(*batch_shape, 3, 3)
    active_contract = contract or oracle_contract()
    target = tracked_target
    if target is None:
        target = torch.zeros((*batch_shape, 6), device=action.device, dtype=action.dtype)
    return encode_complete_action(
        action,
        active_contract,
        ee_rotation=rotation,
        tracked_target=target,
    )


def _layer_init(layer: nn.Linear, standard_deviation: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, standard_deviation)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class PpoAgent(nn.Module):
    """Checkpoint-compatible copy of pinned ManiSkill v3.0.1 PPO's network."""

    def __init__(self, observation_dimension: int, action_dimension: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(observation_dimension, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(observation_dimension, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, action_dimension), 0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dimension) * -0.5)

    def deterministic_action(self, observation: Tensor) -> Tensor:
        return cast(Tensor, self.actor_mean(observation))


@dataclass(frozen=True, slots=True)
class ParityEpisode:
    task: str
    condition: ParityCondition
    episode_index: int
    seed: int
    success: bool
    task_return: float
    episode_steps: int
    checkpoint_sha256: str


def _make_environment(
    task: str,
    condition: ParityCondition,
    num_envs: int,
    contract: ActionContract | None = None,
    *,
    rotation_mode: RotationMode = "identity",
    max_episode_steps: int | None = None,
) -> Any:
    import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import (  # type: ignore[import-untyped]
        ManiSkillVectorEnv,
    )

    environment_id = _ENVIRONMENT_IDS.get(task)
    if environment_id is None:
        raise ValueError(f"unknown task: {task}")
    # ``max_episode_steps`` defaults to the task's registered horizon (50 for
    # Pick/Push — the horizon every frozen PPO cell used). Imitation backbones
    # trained on motion-planning demos run at demo speed and need the official DP
    # baseline's longer horizon (~100) to be scored at their competent horizon.
    extra: dict[str, Any] = {}
    if max_episode_steps is not None:
        extra["max_episode_steps"] = max_episode_steps
    base = gym.make(
        environment_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reconfiguration_freq=1,
        disable_env_checker=True,
        **extra,
    )
    # Real-rotation variant (v2): the wrapper decodes tool-frame twists against the
    # live tcp orientation instead of the identity placeholder. Reads the raw env's
    # tcp pose on each step. Default "identity" keeps every v1 result reproducible.
    provider = make_tcp_rotation_provider(base) if rotation_mode == "real" else None
    if condition == "identity":
        base = HiddenContractWrapper(
            base, identity_contract(), ee_rotation_provider=provider
        )
    elif condition in {"oracle_nonidentity", "noadapt_nonidentity"}:
        base = HiddenContractWrapper(
            base, contract or oracle_contract(), ee_rotation_provider=provider
        )
    elif condition != "unwrapped":
        raise ValueError(f"unknown parity condition: {condition}")
    return ManiSkillVectorEnv(
        base,
        num_envs=num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )


def evaluate_ppo_checkpoint(
    checkpoint: Path,
    *,
    task: str,
    condition: ParityCondition,
    seed: int,
    episodes: int = 100,
    num_envs: int = 16,
    contract: ActionContract | None = None,
    rotation_mode: RotationMode = "identity",
) -> list[ParityEpisode]:
    """Evaluate one frozen PPO checkpoint and retain each completed episode.

    ``rotation_mode="real"`` (the v2 variant) decodes tool-frame contracts against
    the live tcp rotation and inverts the oracle path against the SAME rotation,
    read from the same env at the same pre-step state, so oracle parity stays
    exact. ``"identity"`` (default) reproduces every v1 parity/Gate-1 result.
    """
    if episodes <= 0 or num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    environment = _make_environment(
        task, condition, num_envs, contract, rotation_mode=rotation_mode
    )
    try:
        observation, _ = environment.reset(seed=seed)
        observation_dimension = int(np.prod(environment.single_observation_space.shape))
        action_dimension = int(np.prod(environment.single_action_space.shape))
        agent = PpoAgent(observation_dimension, action_dimension).to(environment.device)
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
        checkpoint_hash = sha256_file(checkpoint)
        records: list[ParityEpisode] = []
        active_contract = contract or oracle_contract()
        oracle_rotation_provider = (
            make_tcp_rotation_provider(environment) if rotation_mode == "real" else None
        )
        tracked_target = torch.zeros(
            (num_envs, 6), device=environment.device, dtype=observation.dtype
        )
        while len(records) < episodes:
            with torch.no_grad():
                canonical = torch.clamp(agent.deterministic_action(observation), low, high)
                # Read the pre-step tcp rotation the wrapper will decode against so
                # the oracle inverts the exact same axis (v2). None => identity (v1).
                rotation = (
                    oracle_rotation_provider()
                    if oracle_rotation_provider is not None
                    else None
                )
                action = policy_action_for_condition(
                    canonical,
                    condition,
                    contract=active_contract,
                    tracked_target=tracked_target,
                    ee_rotation=rotation,
                )
                observation, _, _, _, info = environment.step(action)
                if (
                    condition == "oracle_nonidentity"
                    and active_contract.target == "absolute"
                ):
                    tracked_target = tracked_target + canonical[..., :6]
            if "final_info" not in info:
                continue
            mask = torch.as_tensor(info["_final_info"], dtype=torch.bool).reshape(-1)
            tracked_target[mask.to(device=tracked_target.device)] = 0
            metrics = info["final_info"]["episode"]
            success = torch.as_tensor(metrics["success_once"]).reshape(-1)[mask]
            returns = torch.as_tensor(metrics["return"]).reshape(-1)[mask]
            lengths = torch.as_tensor(metrics["episode_len"]).reshape(-1)[mask]
            for succeeded, task_return, length in zip(success, returns, lengths, strict=True):
                if len(records) == episodes:
                    break
                records.append(
                    ParityEpisode(
                        task=task,
                        condition=condition,
                        episode_index=len(records),
                        seed=seed,
                        success=bool(succeeded.item()),
                        task_return=float(task_return.item()),
                        episode_steps=int(length.item()),
                        checkpoint_sha256=checkpoint_hash,
                    )
                )
        return records
    finally:
        environment.close()


def write_parity_episodes(path: Path, episodes: list[ParityEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        "".join(json.dumps(asdict(episode), sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    os.replace(temporary, path)
