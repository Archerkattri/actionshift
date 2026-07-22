"""Train a delay-aware augmented-state PPO backbone under randomized action lag.

This is a faithful adaptation of the pinned official ManiSkill v3.0.1 PPO baseline
(``third_party/maniskill/examples/baselines/ppo/ppo.py``) with exactly TWO changes,
both documented inline and in ``reports/adaptation_delay_aware.md``:

  (a) OBSERVATION AUGMENTATION -- the policy input is the base state concatenated
      with the last ``--history`` (K=4) canonical actions it emitted (zeros at
      reset). This restores the Markov property of the delayed MDP
      (Katsikopoulos & Engelbrecht 2003 augmented-state formulation), letting the
      policy plan for delay. Shared buffer/agent code lives in
      ``src/actionshift/adaptation/delay_aware.py`` so training and evaluation use
      byte-identical augmentation.

  (b) RANDOMIZED PER-EPISODE ACTION LAG -- before the environment executes it, the
      canonical action is delayed by a per-environment lag resampled each episode
      from ``--lag-set`` (default {0,1,2,4}). The lag is NOT observable to the
      policy. The per-env lag executor is bit-identical to
      ``contracts.transforms.ActionLag`` at a uniform lag (tested), so the training
      dynamics match the wrapper's long-lag split exactly. The contract is identity
      otherwise (no permutation/sign/scale/frame/target/gripper change), so applying
      the lag directly to the canonical action and stepping the plain env is
      equivalent to the identity-contract wrapper and avoids its per-step cost.

Everything else -- network body, hyperparameters, GAE, clipping, KL early-stop,
episode accounting, checkpoint format -- is unchanged from the official loop. Every
deviation from the exact Gate 0 launch is recorded in the emitted ``config.json`` /
``provenance.json``.

DETERMINISM / REPRODUCIBILITY: seeds numpy/torch/python; dumps full config,
package versions, and hardware; writes an eval curve JSONL; hash-addresses the run
directory and records the final checkpoint SHA-256.

GPU: launch with ``CUDA_VISIBLE_DEVICES=2``. Example (Pick, matching Gate 0 budget):

  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/train_delay_aware.py \
    --env-id PickCube-v1 --task pick_cube --total-timesteps 10000000 \
    --num-envs 1024 --seed 20260718 --output artifacts/delay_aware
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.normal import Normal

from actionshift.adaptation.delay_aware import (
    DEFAULT_HISTORY,
    ActionHistoryBuffer,
    DelayAwarePpoAgent,
)

_TASK_ENV_ID = {"pick_cube": "PickCube-v1", "push_cube": "PushCube-v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(_TASK_ENV_ID))
    parser.add_argument("--env-id", default=None, help="defaults from --task")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-eval-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-eval-steps", type=int, default=50)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--lag-set", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument(
        "--lag-curriculum",
        action="store_true",
        help="documented variant: train lag=0 for the first --curriculum-frac of "
        "iterations, then switch to the full randomized lag set",
    )
    parser.add_argument("--curriculum-frac", type=float, default=0.3)
    # Official PickCube/PushCube PPO hyperparameters (pd_ee_delta_pose Gate 0 setup).
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--eval-freq", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("artifacts/delay_aware"))
    return parser.parse_args()


class TrainAgent(DelayAwarePpoAgent):
    """Delay-aware agent with the official PPO get_action_and_value / get_value API."""

    def get_value(self, augmented: torch.Tensor) -> torch.Tensor:
        return self.critic(augmented)

    def get_action_and_value(
        self, augmented: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(augmented)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(augmented),
        )


def sample_lag(
    lag_set: torch.Tensor, count: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    idx = torch.randint(0, len(lag_set), (count,), generator=generator)
    return lag_set[idx].to(device)


def provenance(args: argparse.Namespace) -> dict[str, object]:
    packages = {}
    for name in ("torch", "mani_skill", "gymnasium", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "packages": packages,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "device_name": device_name,
        "deviation_from_gate0": (
            "Gate 0 used the unmodified official ppo.py; this run augments the "
            "observation with K past canonical actions and applies randomized "
            "per-episode action lag from lag_set. All other hyperparameters match "
            "the Gate 0 pd_ee_delta_pose PPO launch (num_envs=1024, num_steps=50, "
            "gamma=0.8, gae_lambda=0.9, update_epochs=4, num_minibatches=32, "
            "target_kl=0.1, lr=3e-4)."
        ),
    }


def main() -> None:
    args = parse_args()
    args.env_id = args.env_id or _TASK_ENV_ID[args.task]
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    config = {
        "task": args.task,
        "env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "history": args.history,
        "lag_set": sorted(args.lag_set),
        "lag_curriculum": args.lag_curriculum,
        "curriculum_frac": args.curriculum_frac,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "num_minibatches": args.num_minibatches,
        "update_epochs": args.update_epochs,
        "clip_coef": args.clip_coef,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "control_mode": args.control_mode,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "num_iterations": num_iterations,
    }
    run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_dir = args.output / args.task / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance(args), indent=2, sort_keys=True) + "\n"
    )
    curve_path = run_dir / "curve.jsonl"
    curve_path.write_text("")
    print(f"[delay-aware] run_dir={run_dir} num_iterations={num_iterations}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    import gymnasium as gym
    import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    from mani_skill.utils import gym_utils  # type: ignore[import-untyped]
    from mani_skill.vector.wrappers.gymnasium import (  # type: ignore[import-untyped]
        ManiSkillVectorEnv,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
    )
    envs = gym.make(args.env_id, num_envs=args.num_envs, reconfiguration_freq=None, **env_kwargs)
    eval_envs = gym.make(
        args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=1, **env_kwargs
    )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True
    )
    gym_utils.find_max_episode_steps_value(envs._env)

    base_obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = TrainAgent(base_obs_dim, action_dim, history=args.history).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    lag_set = torch.tensor(sorted(args.lag_set), device=device, dtype=torch.long)
    lag_generator = torch.Generator().manual_seed(args.seed + 777)

    obs_store = torch.zeros(
        (args.num_steps, args.num_envs, base_obs_dim + args.history * action_dim), device=device
    )
    actions_store = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action.detach(), action_low, action_high)

    buffer = ActionHistoryBuffer(
        args.num_envs, action_dim, history=args.history, device=device, dtype=next_obs.dtype
    )
    lag = sample_lag(lag_set, args.num_envs, lag_generator, device)

    for iteration in range(1, num_iterations + 1):
        # Documented curriculum variant: lag=0 warmup, then the full randomized set.
        curriculum_phase = "randomized"
        if args.lag_curriculum and iteration <= int(args.curriculum_frac * num_iterations):
            curriculum_phase = "warmup_lag0"
            lag = torch.zeros(args.num_envs, device=device, dtype=torch.long)
        active_lag_set = (
            torch.zeros(1, device=device, dtype=torch.long)
            if curriculum_phase == "warmup_lag0"
            else lag_set
        )

        agent.eval()
        if iteration % args.eval_freq == 1:
            eval_metrics = evaluate(
                agent, eval_envs, args, device, action_low, action_high, action_dim
            )
            record = {"global_step": global_step, "iteration": iteration, **eval_metrics}
            with curve_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[eval] it={iteration} step={global_step} {eval_metrics}")
            torch.save(agent.state_dict(), run_dir / f"ckpt_{iteration}.pt")

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        for step in range(args.num_steps):
            global_step += args.num_envs
            augmented = buffer.augment(next_obs)
            obs_store[step] = augmented
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(augmented)
                values[step] = value.flatten()
            actions_store[step] = action
            logprobs[step] = logprob

            executed = buffer.lag_execute(action, lag)
            buffer.push(action)
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(executed))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                final_base = infos["final_observation"]
                augmented_final = buffer.augment(final_base)
                with torch.no_grad():
                    idx = torch.arange(args.num_envs, device=device)[done_mask]
                    final_values[step, idx] = agent.get_value(
                        augmented_final[done_mask]
                    ).view(-1)
                buffer.reset(done_mask)
                resampled = sample_lag(
                    active_lag_set, int(done_mask.sum().item()), lag_generator, device
                )
                lag = lag.clone()
                lag[done_mask] = resampled

        with torch.no_grad():
            next_value = agent.get_value(buffer.augment(next_obs)).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages + values

        b_obs = obs_store.reshape(-1, obs_store.shape[-1])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions_store.reshape(-1, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        agent.train()
        b_inds = np.arange(batch_size)
        approx_kl = torch.zeros((), device=device)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > args.target_kl:
                    break
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if approx_kl > args.target_kl:
                break
        sps = int(global_step / (time.time() - start_time))
        if iteration % 5 == 0 or iteration == num_iterations:
            print(
                f"it={iteration}/{num_iterations} step={global_step} "
                f"sps={sps} phase={curriculum_phase}"
            )

    final_metrics = evaluate(agent, eval_envs, args, device, action_low, action_high, action_dim)
    final_record = {
        "global_step": global_step,
        "iteration": num_iterations,
        "final": True,
        **final_metrics,
    }
    with curve_path.open("a") as handle:
        handle.write(json.dumps(final_record, sort_keys=True) + "\n")
    final_ckpt = run_dir / "final_ckpt.pt"
    torch.save(agent.state_dict(), final_ckpt)
    digest = hashlib.sha256(final_ckpt.read_bytes()).hexdigest()
    (run_dir / "final_ckpt.sha256").write_text(digest + "\n")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "final_ckpt": str(final_ckpt),
        "final_ckpt_sha256": digest,
        "final_eval": final_metrics,
        "elapsed_seconds": time.time() - start_time,
        "config": config,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[delay-aware] DONE {json.dumps(summary['final_eval'])} ckpt={final_ckpt} sha={digest}")
    envs.close()
    eval_envs.close()


def evaluate(
    agent: TrainAgent,
    eval_envs: object,
    args: argparse.Namespace,
    device: torch.device,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    action_dim: int,
) -> dict[str, float]:
    """Deterministic eval per lag in the lag set; augmented obs + env-side lag.

    The env applies each fixed lag through the delay-aware backbone's own action
    history, giving an honest instantaneous-competence (lag 0) and worst-delay
    (lag 4) success signal during training.
    """
    metrics: dict[str, float] = {}
    for fixed_lag in sorted(args.lag_set):
        obs, _ = eval_envs.reset(seed=args.seed + 10_000 + fixed_lag)  # type: ignore[attr-defined]
        buffer = ActionHistoryBuffer(
            args.num_eval_envs, action_dim, history=args.history, device=device, dtype=obs.dtype
        )
        lag = torch.full((args.num_eval_envs,), fixed_lag, device=device, dtype=torch.long)
        successes = 0
        episodes = 0
        for _ in range(args.num_eval_steps * (fixed_lag + 2)):
            with torch.no_grad():
                augmented = buffer.augment(obs)
                action = torch.clamp(agent.deterministic_action(augmented), action_low, action_high)
            executed = buffer.lag_execute(action, lag)
            buffer.push(action)
            obs, _, _, _, infos = eval_envs.step(torch.clamp(executed, action_low, action_high))  # type: ignore[attr-defined]
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                successes += int(
                    infos["final_info"]["episode"]["success_once"][done_mask].sum().item()
                )
                episodes += int(done_mask.sum().item())
                buffer.reset(done_mask)
            if episodes >= args.num_eval_envs * 2:
                break
        metrics[f"success_lag{fixed_lag}"] = successes / max(episodes, 1)
        metrics[f"episodes_lag{fixed_lag}"] = float(episodes)
    return metrics


if __name__ == "__main__":
    main()
