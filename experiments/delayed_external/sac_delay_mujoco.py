"""Delay-aware augmented-state SAC (local) on the DCAC MuJoCo delay benchmark.

This is a CLEANRL-STYLE continuous-control Soft Actor-Critic
(the canonical ``cleanrl/sac_continuous_action.py`` recipe: twin soft
Q-networks with target nets + polyak averaging, a squashed-Gaussian (tanh)
actor, automatic entropy-temperature tuning, a uniform replay buffer, and
per-step gradient updates after ``learning_starts``). It is explicitly labelled
CleanRL-style and is NOT DCAC's delay-correcting actor-critic. The only
benchmark-side addition is ``ConstantDelayWrapper`` (see ``delay_wrapper.py``),
which injects their omega/alpha constant delay and (by default) augments the
observation with the last K = omega + alpha emitted actions -- the same
augmented-state reduction ActionShift uses, ported onto THEIR envs / delay spec.

This is the apples-to-apples OFF-POLICY comparison the prior PPO run
(``reports/delayed_rl_external.md``) could not make: DCAC and its SAC/RTAC
baselines are all off-policy SAC-derived, so a same-family augmented-state SAC
is the fair reference against their published Figure-6 curves.

Metric matches DCAC: mean episodic RETURN (true undelayed per-step rewards summed
over an episode), logged over training; we report the final-window mean return.

Resumability: full checkpoint (all network + optimizer + entropy-temp + RNG state
+ episode log + global_step) is written atomically every ``--ckpt-interval`` env
steps. A relaunch resumes from the latest checkpoint with a FRESH replay buffer
(the buffer itself is NOT persisted -- documented, and identical across every run
so the protocol is consistent). A completed run (``summary.json`` present) exits
immediately, so a driver can safely re-invoke it.

GPU: launch with CUDA_VISIBLE_DEVICES=2.

  CUDA_VISIBLE_DEVICES=2 .venv-delayext/bin/python \
    experiments/delayed_external/sac_delay_mujoco.py \
    --env-id HalfCheetah-v4 --obs-delay 2 --act-delay 3 \
    --total-timesteps 1000000 --seed 1 --output artifacts/delayed_external_sac
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import signal
import sys
import time
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from delay_wrapper import ConstantDelayWrapper
from torch import nn, optim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="HalfCheetah-v4")
    p.add_argument("--obs-delay", type=int, default=2)
    p.add_argument("--act-delay", type=int, default=3)
    p.add_argument("--augment", type=int, default=1, help="1=augmented-state, 0=naive control")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=1e-3)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--target-network-frequency", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--autotune", type=int, default=1)
    p.add_argument("--ckpt-interval", type=int, default=100_000)
    p.add_argument("--log-interval", type=int, default=2_000)
    p.add_argument("--output", type=Path, default=Path("artifacts/delayed_external_sac"))
    return p.parse_args()


def make_env(env_id: str, obs_delay: int, act_delay: int, augment: bool):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # true undelayed returns
        # No ClipAction: it rebinds the action space to +/-inf, and SAC's tanh
        # squashing already keeps actions inside the env's real [-1, 1] bounds
        # (CleanRL sac_continuous_action does not use ClipAction either).
        env = ConstantDelayWrapper(
            env, obs_delay=obs_delay, act_delay=act_delay, augment=augment
        )
        return env

    return thunk


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], 1))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ReplayBuffer:
    """Uniform replay buffer (fresh each run -- not persisted; see module docstring)."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.full = True

    def sample(self, batch_size: int):
        upper = self.capacity if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return to(self.obs), to(self.actions), to(self.next_obs), to(self.rewards), to(self.dones)


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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "sac_impl": "CleanRL-style sac_continuous_action (labelled; NOT DCAC)",
    }


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# Set by SIGTERM/SIGINT so the training loop can flush a final checkpoint and
# exit cleanly (resumable) instead of being torn down mid-step. A graceful kill
# thus never loses more than one env step of progress, regardless of the
# --ckpt-interval spacing.
_STOP = {"flag": False}


def _request_stop(signum, _frame) -> None:
    _STOP["flag"] = True


def main() -> None:
    args = parse_args()
    augment = bool(args.augment)

    config = vars(args).copy()
    config["output"] = str(args.output)
    config["augment"] = int(augment)
    config["k_history"] = (args.obs_delay + args.act_delay) if augment else 0
    run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    tag = "aug" if augment else "naive"
    run_dir = (
        args.output / args.env_id / f"od{args.obs_delay}_ad{args.act_delay}_{tag}"
        / f"seed{args.seed}_{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        print(f"[sac-delay] SKIP (already complete) {run_dir}")
        return
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps(provenance(), indent=2) + "\n")
    curve_path = run_dir / "curve.jsonl"
    ckpt_path = run_dir / "checkpoint.pt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.obs_delay, args.act_delay, augment)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    action_scale = torch.tensor(
        (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
        dtype=torch.float32, device=device,
    )
    action_bias = torch.tensor(
        (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
        dtype=torch.float32, device=device,
    )

    actor = Actor(obs_dim, act_dim, action_scale, action_bias).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        target_entropy = None
        log_alpha = None
        alpha = args.alpha
        a_optimizer = None

    rb = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device)
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    global_step = 0
    resumed_from = None

    # ---- resume from checkpoint (weights + opt + entropy-temp + RNG + ep log) ----
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(ck["actor"])
        qf1.load_state_dict(ck["qf1"])
        qf2.load_state_dict(ck["qf2"])
        qf1_target.load_state_dict(ck["qf1_target"])
        qf2_target.load_state_dict(ck["qf2_target"])
        q_optimizer.load_state_dict(ck["q_optimizer"])
        actor_optimizer.load_state_dict(ck["actor_optimizer"])
        if args.autotune and ck.get("log_alpha") is not None:
            with torch.no_grad():
                log_alpha.copy_(ck["log_alpha"].to(device))
            a_optimizer.load_state_dict(ck["a_optimizer"])
            alpha = log_alpha.exp().item()
        global_step = int(ck["global_step"])
        ep_returns = list(ck["ep_returns"])
        ep_steps = list(ck["ep_steps"])
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        resumed_from = global_step
        print(f"[sac-delay] RESUME {run_dir.name} at step {global_step} (fresh replay buffer)")
    else:
        curve_path.write_text("")

    print(f"[sac-delay] env={args.env_id} od={args.obs_delay} ad={args.act_delay} "
          f"aug={int(augment)} seed={args.seed} obs_dim={obs_dim} start_step={global_step} "
          f"target={args.total_timesteps} run_dir={run_dir}")

    def save_checkpoint(step: int) -> None:
        ck = {
            "global_step": step,
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(), "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(), "qf2_target": qf2_target.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "log_alpha": (log_alpha.detach().cpu() if args.autotune else None),
            "a_optimizer": (a_optimizer.state_dict() if args.autotune else None),
            "ep_returns": ep_returns, "ep_steps": ep_steps,
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        }
        bio = io.BytesIO()
        torch.save(ck, bio)
        atomic_write(ckpt_path, bio.getvalue())

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    start = time.time()
    last_log = global_step
    last_ckpt = global_step
    next_obs, _ = envs.reset(seed=args.seed + global_step)
    next_obs = np.asarray(next_obs, dtype=np.float32)

    while global_step < args.total_timesteps:
        if _STOP["flag"]:
            save_checkpoint(global_step)
            print(f"[sac-delay] SIGTERM/SIGINT: checkpointed at step {global_step}, exiting "
                  f"(resumable). {run_dir.name}")
            envs.close()
            sys.exit(0)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(1)])
        else:
            with torch.no_grad():
                a, _, _ = actor.get_action(torch.tensor(next_obs, device=device))
            actions = a.cpu().numpy()

        next_obs_np, rewards, terminations, _truncations, infos = envs.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs_np = np.asarray(next_obs_np, dtype=np.float32)

        if "episode" in infos:
            fin = infos["_episode"] if "_episode" in infos else np.ones(1, dtype=bool)
            mask = fin if fin.dtype == bool else fin.astype(bool)
            for r in np.asarray(infos["episode"]["r"])[mask]:
                ep_returns.append(float(r))
                ep_steps.append(global_step)

        # store real terminal next-obs (SyncVectorEnv autoresets)
        real_next = next_obs_np.copy()
        if "final_obs" in infos:
            for i, fo in enumerate(infos["final_obs"]):
                if fo is not None:
                    real_next[i] = np.asarray(fo, dtype=np.float32)
        rb.add(next_obs[0], real_next[0], actions[0], rewards[0], float(terminations[0]))
        next_obs = next_obs_np
        global_step += 1

        # ---- learning ----
        if global_step > args.learning_starts:
            o, a_b, no, r_b, d_b = rb.sample(args.batch_size)
            with torch.no_grad():
                na, nlp, _ = actor.get_action(no)
                q1_next = qf1_target(no, na)
                q2_next = qf2_target(no, na)
                min_q_next = torch.min(q1_next, q2_next) - alpha * nlp
                next_q = r_b + (1 - d_b) * args.gamma * min_q_next
            q1 = qf1(o, a_b)
            q2 = qf2(o, a_b)
            q1_loss = F.mse_loss(q1, next_q)
            q2_loss = F.mse_loss(q2, next_q)
            q_loss = q1_loss + q2_loss
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, lp, _ = actor.get_action(o)
                    q1_pi = qf1(o, pi)
                    q2_pi = qf2(o, pi)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = ((alpha * lp) - min_q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad():
                            _, lp2, _ = actor.get_action(o)
                        alpha_loss = (-log_alpha.exp() * (lp2 + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for pt, p in zip(qf1_target.parameters(), qf1.parameters(), strict=True):
                    pt.data.mul_(1 - args.tau)
                    pt.data.add_(args.tau * p.data)
                for pt, p in zip(qf2_target.parameters(), qf2.parameters(), strict=True):
                    pt.data.mul_(1 - args.tau)
                    pt.data.add_(args.tau * p.data)

        # ---- logging ----
        if global_step - last_log >= args.log_interval:
            last_log = global_step
            recent = ep_returns[-50:]
            mean_ret = float(np.mean(recent)) if recent else float("nan")
            sps = int((global_step - (resumed_from or 0)) / max(time.time() - start, 1e-9))
            with curve_path.open("a") as h:
                h.write(json.dumps({
                    "global_step": global_step,
                    "mean_ep_return_last50": mean_ret,
                    "num_episodes": len(ep_returns),
                    "alpha": float(alpha),
                    "sps": sps,
                }) + "\n")
            if global_step % (args.log_interval * 10) < args.log_interval:
                print(f"step={global_step}/{args.total_timesteps} ret50={mean_ret:.1f} "
                      f"eps={len(ep_returns)} alpha={alpha:.3f} sps={sps}")

        # ---- checkpoint ----
        if global_step - last_ckpt >= args.ckpt_interval:
            last_ckpt = global_step
            save_checkpoint(global_step)

    save_checkpoint(global_step)
    final_window = ep_returns[-100:] if len(ep_returns) >= 100 else ep_returns
    curve_vals = []
    if curve_path.exists():
        for line in curve_path.read_text().splitlines():
            v = json.loads(line)["mean_ep_return_last50"]
            if not (isinstance(v, float) and np.isnan(v)):
                curve_vals.append(v)
    summary = {
        "run_id": run_id,
        "env_id": args.env_id,
        "obs_delay": args.obs_delay,
        "act_delay": args.act_delay,
        "total_delay": args.obs_delay + args.act_delay,
        "augment": int(augment),
        "algo": "SAC (CleanRL-style, augmented-state)" if augment
                else "SAC (CleanRL-style, NAIVE non-augmented control)",
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "num_episodes": len(ep_returns),
        "final_mean_return_last100": float(np.mean(final_window)) if final_window else float("nan"),
        "final_std_return_last100": float(np.std(final_window)) if final_window else float("nan"),
        "best_mean_return_last50": float(max(curve_vals)) if curve_vals else float("nan"),
        "elapsed_seconds": time.time() - start,
        "resumed_from_step": resumed_from,
        "config": config,
    }
    atomic_write(summary_path, (json.dumps(summary, indent=2, default=str) + "\n").encode())
    print(f"[sac-delay] DONE {args.env_id} aug={int(augment)} seed={args.seed} "
          f"final_ret={summary['final_mean_return_last100']:.1f} "
          f"best={summary['best_mean_return_last50']:.1f} eps={len(ep_returns)}")
    envs.close()


if __name__ == "__main__":
    main()
