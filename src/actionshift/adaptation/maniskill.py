"""Real ManiSkill wiring: calibration probe and adapter-generic evaluation.

The evaluation loop mirrors ``ppo_parity.evaluate_ppo_checkpoint`` episode
accounting exactly; the only change is that the action path runs through a
``ContractAdapter`` and the adapter observes calibrated responses. Auto-reset
timing is preserved faithfully: the transition that crosses an episode boundary
is invalid evidence this step, while the wrapper's decoder reset lands on the
following step, so the two masks are threaded separately.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from actionshift.adaptation.adapters import ContractAdapter
from actionshift.adaptation.calibration import (
    ResponseCalibration,
    response_from_observations,
)
from actionshift.adaptation.policies import CanonicalPolicy, PpoCanonicalPolicy
from actionshift.benchmarking.ppo_parity import (
    PpoAgent,
    RotationMode,
    _make_environment,
)
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.transforms import quaternion_to_rotation_matrix
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file


def rotation_from_observation(
    calibration: ResponseCalibration, observation: Tensor
) -> Tensor:
    """Recover the tcp rotation matrix from the calibrated quaternion slice.

    The calibration located the tcp quaternion block inside the flat state
    observation on the unwrapped environment (contract-independent task knowledge),
    so the same slice of a live observation is exactly the quaternion the wrapper's
    rotation provider read. Converting it yields the batched ``(num_envs, 3, 3)``
    rotation the belief replicas need to stay bit-faithful to the wrapper under the
    v2 real-rotation variant.
    """
    q = calibration.quaternion_start
    quaternion = observation[..., q : q + 4]
    return quaternion_to_rotation_matrix(quaternion)


class ManiSkillPoseProbe:
    """Unwrapped-environment probe used only for contract-independent calibration."""

    def __init__(self, task: str, *, num_envs: int, seed: int) -> None:
        self._environment = _make_environment(task, "unwrapped", num_envs)
        observation, _ = self._environment.reset(seed=seed)
        self._observation = observation
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        low = torch.as_tensor(self._environment.single_action_space.low)
        high = torch.as_tensor(self._environment.single_action_space.high)
        self._low = low
        self._high = high
        self._num_envs = num_envs

    def close(self) -> None:
        self._environment.close()

    def _tcp_pose(self) -> tuple[Tensor, Tensor]:
        pose = self._environment.unwrapped.agent.tcp.pose
        position = torch.as_tensor(pose.p).reshape(self._num_envs, 3)
        quaternion = torch.as_tensor(pose.q).reshape(self._num_envs, 4)
        return position.cpu(), quaternion.cpu()

    def gripper_positions(self) -> Tensor:
        """Current finger joint positions (the two trailing robot qpos entries)."""
        qpos = torch.as_tensor(self._environment.unwrapped.agent.robot.get_qpos())
        return qpos.reshape(self._num_envs, -1)[:, -2:].cpu()

    def step_random(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        span = (self._high - self._low).unsqueeze(0)
        sample = torch.rand(
            (self._num_envs, self._low.shape[-1]), generator=self._generator
        )
        action = (self._low.unsqueeze(0) + sample * span).to(
            device=self._observation.device, dtype=self._observation.dtype
        )
        observation, _, _, _, _ = self._environment.step(action)
        self._observation = observation
        position, quaternion = self._tcp_pose()
        return (
            torch.as_tensor(observation).reshape(self._num_envs, -1).cpu(),
            position,
            quaternion,
            action.reshape(self._num_envs, -1).cpu(),
        )


@dataclass(frozen=True, slots=True)
class AdaptationEpisode:
    """One completed hidden-contract episode under an adaptation method."""

    task: str
    method: str
    episode_index: int
    seed: int
    success: bool
    task_return: float
    episode_steps: int
    checkpoint_sha256: str
    contract_sha256: str
    probe_steps: int = 0
    probe_displacement: float = 0.0


def evaluate_adapter(
    checkpoint: Path,
    *,
    task: str,
    method: str,
    adapter: ContractAdapter,
    contract: ActionContract,
    calibration: ResponseCalibration,
    seed: int,
    episodes: int = 100,
    num_envs: int = 16,
    rotation_mode: RotationMode = "identity",
    policy: CanonicalPolicy | None = None,
    max_episode_steps: int | None = None,
) -> list[AdaptationEpisode]:
    """Evaluate a frozen backbone through an adapter on one hidden contract.

    The backbone is any ``CanonicalPolicy``: when ``policy is None`` a frozen PPO
    actor is loaded from ``checkpoint`` (the default, byte-reproducing every prior
    result); when supplied (e.g. a ``DiffusionPolicyShim``), it drives the same
    adapter machinery unchanged, since adapters consume canonical actions. The
    ``checkpoint`` path is still hashed for provenance regardless of the backbone.

    ``rotation_mode="real"`` (v2) wires the wrapper to decode tool-frame contracts
    against the live tcp rotation and feeds the adapter the observed rotation
    sequence (from the calibrated quaternion slice) so its belief replicas invert
    the wrapper exactly. ``"identity"`` (default) is the reproducible v1 variant:
    the adapter receives ``ee_rotation=None`` and behaves bit-identically to before.
    """
    if episodes <= 0 or num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    environment = _make_environment(
        task, "noadapt_nonidentity", num_envs, contract,
        rotation_mode=rotation_mode, max_episode_steps=max_episode_steps,
    )
    real_rotation = rotation_mode == "real"
    try:
        observation, _ = environment.reset(seed=seed)
        observation_dimension = int(np.prod(environment.single_observation_space.shape))
        action_dimension = int(np.prod(environment.single_action_space.shape))
        if policy is None:
            agent = PpoAgent(observation_dimension, action_dimension).to(
                environment.device
            )
            payload = torch.load(
                checkpoint, map_location=environment.device, weights_only=True
            )
            agent.load_state_dict(payload)
            agent.eval()
            policy = PpoCanonicalPolicy(agent)
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
        checkpoint_digest = sha256_file(checkpoint)
        contract_digest = contract_hash(contract)
        records: list[AdaptationEpisode] = []
        pending_reset: Tensor | None = None
        previous_observation = observation
        probe_steps_used = torch.zeros(num_envs, dtype=torch.long)
        probe_displacement = torch.zeros(num_envs, dtype=torch.float64)
        while len(records) < episodes:
            with torch.no_grad():
                canonical = torch.clamp(
                    policy.act(observation, reset_mask=pending_reset), low, high
                )
                # v2: the wrapper decodes this step against the pre-step tcp
                # rotation, recoverable from the current observation's quaternion
                # slice; the observe-time rotation of the just-applied raw action is
                # the same pre-step frame, held in ``previous_observation``.
                encode_rotation = (
                    rotation_from_observation(calibration, observation)
                    if real_rotation
                    else None
                )
                raw = adapter.encode(canonical, ee_rotation=encode_rotation)
                probe_mask = getattr(adapter, "last_probe_mask", None)
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
                observe_rotation = (
                    rotation_from_observation(calibration, previous_observation)
                    if real_rotation
                    else None
                )
                adapter.observe(
                    raw,
                    response,
                    reset_mask=pending_reset,
                    invalid_mask=boundary,
                    ee_rotation=observe_rotation,
                )
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


def write_adaptation_episodes(path: Path, episodes: list[AdaptationEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        "".join(json.dumps(asdict(episode), sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_calibration(
    task: str,
    *,
    num_envs: int = 8,
    seed: int = 20260720,
    steps: int = 64,
    calibrate_gripper: bool = False,
    magnitude_gain: bool = False,
) -> ResponseCalibration:
    """Calibrate the response model for one task on the unwrapped environment."""
    probe = ManiSkillPoseProbe(task, num_envs=num_envs, seed=seed)
    try:
        return calibrate_response_from_probe(
            probe,
            task=task,
            steps=steps,
            calibrate_gripper=calibrate_gripper,
            magnitude_gain=magnitude_gain,
        )
    finally:
        probe.close()


def calibrate_response_from_probe(
    probe: ManiSkillPoseProbe,
    *,
    task: str,
    steps: int,
    calibrate_gripper: bool = False,
    magnitude_gain: bool = False,
) -> ResponseCalibration:
    from actionshift.adaptation.calibration import calibrate_response

    return calibrate_response(
        probe,
        task=task,
        steps=steps,
        calibrate_gripper=calibrate_gripper,
        magnitude_gain=magnitude_gain,
    )


def load_or_run_calibration(
    task: str,
    path: Path,
    *,
    num_envs: int = 8,
    seed: int = 20260720,
    steps: int = 64,
    calibrate_gripper: bool = False,
    magnitude_gain: bool = False,
) -> ResponseCalibration:
    if path.is_file():
        return ResponseCalibration.load(path)
    calibration = run_calibration(
        task,
        num_envs=num_envs,
        seed=seed,
        steps=steps,
        calibrate_gripper=calibrate_gripper,
        magnitude_gain=magnitude_gain,
    )
    calibration.save(path)
    return calibration


def summarize(records: list[AdaptationEpisode]) -> dict[str, Any]:
    successes = sum(1 for record in records if record.success)
    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
    }
