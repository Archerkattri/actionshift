"""Delay-aware augmented-state PPO (local) on the DCAC MuJoCo delay benchmark.

PPO is a CLEANRL-STYLE continuous-control implementation (the canonical
``cleanrl/ppo_continuous_action.py`` recipe: shared MuJoCo wrapper stack
[RecordEpisodeStatistics, ClipAction, NormalizeObservation/Reward + clip],
Gaussian actor with a state-independent log-std, GAE, clipped surrogate,
value-clip-free critic, Adam, per-epoch minibatch updates). It is explicitly
labelled CleanRL-style and is NOT DCAC's actor-critic. The only benchmark-side
addition is ``ConstantDelayWrapper`` (see ``delay_wrapper.py``), which injects
their omega/alpha constant delay and augments the observation with the last
K = omega + alpha emitted actions -- the same augmented-state reduction
ActionShift uses, ported onto THEIR envs and delay spec.

Metric matches DCAC: mean episodic RETURN (true undelayed per-step rewards summed
over an episode), logged over training. We report the final-window mean return.

GPU: launch with CUDA_VISIBLE_DEVICES=0.

  CUDA_VISIBLE_DEVICES=0 .venv-delayext/bin/python \
    experiments/delayed_external/ppo_delay_mujoco.py \
    --env-id HalfCheetah-v4 --obs-delay 2 --act-delay 3 \
    --total-timesteps 1000000 --seed 1 --output artifacts/delayed_external
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from delay_wrapper import ConstantDelayWrapper
from torch import nn, optim
from torch.distributions.normal import Normal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="HalfCheetah-v4")
    p.add_argument("--obs-delay", type=int, default=2)
    p.add_argument("--act-delay", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--output", type=Path, default=Path("artifacts/delayed_external"))
    return p.parse_args()


def make_env(env_id: str, obs_delay: int, act_delay: int, gamma: float):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # true undelayed returns
        env = gym.wrappers.ClipAction(env)
        env = ConstantDelayWrapper(env, obs_delay=obs_delay, act_delay=act_delay)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda o: np.clip(o, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        return env

    return thunk


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def provenance() -> dict:
    pkgs = {}
    for name in ("torch", "gymnasium", "mujoco", "numpy"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "not-installed"
    return {
        "packages": pkgs,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ppo_impl": "CleanRL-style ppo_continuous_action (labelled; NOT DCAC)",
    }


def main() -> None:
    args = parse_args()
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    config = vars(args).copy()
    config["output"] = str(args.output)
    config["batch_size"] = batch_size
    config["k_history"] = args.obs_delay + args.act_delay
    run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    run_dir = (
        args.output / args.env_id / f"od{args.obs_delay}_ad{args.act_delay}"
        / f"seed{args.seed}_{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps(provenance(), indent=2) + "\n")
    curve_path = run_dir / "curve.jsonl"
    curve_path.write_text("")
    print(f"[delay-ext] env={args.env_id} od={args.obs_delay} ad={args.act_delay} "
          f"seed={args.seed} iters={num_iterations} run_dir={run_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.obs_delay, args.act_delay, args.gamma)
            for _ in range(args.num_envs)
        ]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(np.asarray(next_obs), dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    ep_returns: list[float] = []

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(np.asarray(next_obs_np), dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)

            if "episode" in infos:
                fin = infos["_episode"] if "_episode" in infos else infos["episode"]["_r"]
                mask = fin if fin.dtype == bool else fin.astype(bool)
                for r in np.asarray(infos["episode"]["r"])[mask]:
                    ep_returns.append(float(r))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape(-1, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            approx_kl = torch.zeros((), device=device)
            for start_i in range(0, batch_size, minibatch_size):
                mb = b_inds[start_i : start_i + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        recent = ep_returns[-50:]
        mean_ret = float(np.mean(recent)) if recent else float("nan")
        sps = int(global_step / (time.time() - start))
        record = {
            "global_step": global_step,
            "iteration": iteration,
            "mean_ep_return_last50": mean_ret,
            "num_episodes": len(ep_returns),
            "sps": sps,
        }
        with curve_path.open("a") as h:
            h.write(json.dumps(record) + "\n")
        if iteration % 5 == 0 or iteration == num_iterations:
            print(f"it={iteration}/{num_iterations} step={global_step} "
                  f"ret50={mean_ret:.1f} eps={len(ep_returns)} sps={sps}")

    final_window = ep_returns[-100:] if len(ep_returns) >= 100 else ep_returns
    summary = {
        "run_id": run_id,
        "env_id": args.env_id,
        "obs_delay": args.obs_delay,
        "act_delay": args.act_delay,
        "total_delay": args.obs_delay + args.act_delay,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "num_episodes": len(ep_returns),
        "final_mean_return_last100": float(np.mean(final_window)) if final_window else float("nan"),
        "final_std_return_last100": float(np.std(final_window)) if final_window else float("nan"),
        "best_mean_return_last50": float(
            max(json.loads(line)["mean_ep_return_last50"]
                for line in curve_path.read_text().splitlines()
                if not np.isnan(json.loads(line)["mean_ep_return_last50"]))
        ) if ep_returns else float("nan"),
        "elapsed_seconds": time.time() - start,
        "config": config,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[delay-ext] DONE {args.env_id} seed={args.seed} "
          f"final_ret={summary['final_mean_return_last100']:.1f} "
          f"best={summary['best_mean_return_last50']:.1f} "
          f"eps={len(ep_returns)} elapsed={summary['elapsed_seconds']:.0f}s")
    envs.close()


if __name__ == "__main__":
    main()
