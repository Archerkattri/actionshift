"""Resumable Diffusion Policy trainer on the clean ``pd_ee_delta_pose`` interface.

This reuses the *official* ManiSkill Diffusion Policy baseline verbatim — its
``SmallDemoDataset_DiffusionPolicy`` dataset, its ``Agent`` (``ConditionalUnet1D``
+ DDPM sampler), and the identical AdamW / cosine-warmup / EMA(power=0.75)
training math from ``examples/baselines/diffusion_policy/train.py`` — loaded as a
module so nothing in third_party is edited. The only additions are the
resumability the evaluation harness requires:

* checkpoints every ``--checkpoint-every`` iterations holding the full training
  state (model, EMA, optimizer, LR schedule, iteration counter, RNG);
* a ``manifest.json`` recording progress;
* resume logic: rerunning the same command continues from the last checkpoint
  (``IterationBasedBatchSampler(start_iter=...)`` skips consumed batches);
* a clean ``final_ckpt.pt`` in the official ``{agent, ema_agent}`` layout (EMA
  weights copied into ``ema_agent``) plus a ``dp_config.json`` describing the
  architecture, both consumed by ``actionshift.adaptation.dp_policy``.

Documented deviations from the official config: (1) control mode is
``pd_ee_delta_pose`` (7-DoF), converted from the ``pd_joint_pos`` motion-planning
demos via ``mani_skill.trajectory.replay_trajectory``, so the backbone speaks the
exact action interface the ActionShift contracts transform — the official state
baseline uses ``pd_ee_delta_pos`` (4-DoF); (2) all successfully-replayed demos
are used rather than 100. All other hyper-parameters are the baseline defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_DP_DIR = _ROOT / "third_party/maniskill/examples/baselines/diffusion_policy"


def _load_official_train_module() -> object:
    """Import the official train.py as a module (its __main__ block never runs)."""
    if str(_DP_DIR) not in sys.path:
        sys.path.insert(0, str(_DP_DIR))
    spec = importlib.util.spec_from_file_location("dp_official_train", _DP_DIR / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dp_official_train"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class TrainArgs:
    task: str
    demo_path: str
    run_dir: str
    total_iters: int = 30000
    batch_size: int = 1024
    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: tuple[int, ...] = (64, 128, 256)
    n_groups: int = 8
    num_demos: int | None = None
    control_mode: str = "pd_ee_delta_pose"
    seed: int = 1
    checkpoint_every: int = 2000
    log_every: int = 500


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict[str, object]) -> None:
    # ``torch.load(map_location=cuda)`` can move the RNG byte tensors onto the GPU;
    # the setters require CPU ByteTensors, so coerce them back.
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))  # type: ignore[union-attr]
    torch.cuda.set_rng_state_all(
        [tensor.cpu().to(torch.uint8) for tensor in state["torch_cuda"]]  # type: ignore[attr-defined]
    )


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def train(args: TrainArgs) -> None:
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.sampler import BatchSampler, RandomSampler

    dp = _load_official_train_module()
    device = torch.device("cuda")

    # The official dataset/agent read a module-global ``args`` and ``device``.
    official_args = SimpleNamespace(
        control_mode=args.control_mode,
        obs_horizon=args.obs_horizon,
        act_horizon=args.act_horizon,
        pred_horizon=args.pred_horizon,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        unet_dims=list(args.unet_dims),
        n_groups=args.n_groups,
    )
    dp.args = official_args  # type: ignore[attr-defined]
    dp.device = device  # type: ignore[attr-defined]
    dp.obs_horizon = args.obs_horizon  # type: ignore[attr-defined]
    dp.pred_horizon = args.pred_horizon  # type: ignore[attr-defined]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    dataset = dp.SmallDemoDataset_DiffusionPolicy(  # type: ignore[attr-defined]
        args.demo_path, device, num_traj=args.num_demos
    )
    observation_dim = int(dataset.trajectories["observations"][0].shape[1])
    action_dim = int(dataset.trajectories["actions"][0].shape[1])

    fake_env = SimpleNamespace(
        single_observation_space=SimpleNamespace(
            shape=(args.obs_horizon, observation_dim)
        ),
        single_action_space=SimpleNamespace(
            shape=(action_dim,),
            high=np.ones(action_dim, dtype=np.float32),
            low=-np.ones(action_dim, dtype=np.float32),
        ),
    )
    agent = dp.Agent(fake_env, official_args).to(device)  # type: ignore[attr-defined]
    ema_agent = dp.Agent(fake_env, official_args).to(device)  # type: ignore[attr-defined]
    optimizer = torch.optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    ema = EMAModel(parameters=agent.parameters(), power=0.75)

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "train_state.pt"
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "dp_config.json"
    final_path = run_dir / "final_ckpt.pt"

    config = {
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "obs_horizon": args.obs_horizon,
        "act_horizon": args.act_horizon,
        "pred_horizon": args.pred_horizon,
        "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
        "unet_dims": list(args.unet_dims),
        "n_groups": args.n_groups,
        "num_diffusion_iters": 100,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    start_iter = 0
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(state["agent"])
        ema_agent.load_state_dict(state["ema_agent"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        start_iter = int(state["iteration"])
        _restore_rng(state["rng"])
        print(f"[resume] {args.task}: continuing from iteration {start_iter}", flush=True)
    else:
        print(f"[start] {args.task}: fresh training", flush=True)

    if start_iter >= args.total_iters:
        print(f"[done] {args.task}: already at {start_iter}/{args.total_iters}", flush=True)
        _finalize(agent, ema_agent, ema, final_path)
        return

    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = dp.IterationBasedBatchSampler(  # type: ignore[attr-defined]
        batch_sampler, args.total_iters, start_iter=start_iter
    )
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)

    agent.train()
    iteration = start_iter
    started = time.time()
    last_loss = float("nan")
    for offset, batch in enumerate(dataloader):
        iteration = start_iter + offset
        loss = agent.compute_loss(
            obs_seq=batch["observations"], action_seq=batch["actions"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())
        last_loss = float(loss.item())

        completed = iteration + 1
        if completed % args.log_every == 0:
            rate = (completed - start_iter) / max(time.time() - started, 1e-6)
            print(
                f"[{args.task}] iter {completed}/{args.total_iters} "
                f"loss {last_loss:.5f} ({rate:.1f} it/s)",
                flush=True,
            )
        if completed % args.checkpoint_every == 0 or completed == args.total_iters:
            _atomic_save(
                {
                    "agent": agent.state_dict(),
                    "ema_agent": ema_agent.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "iteration": completed,
                    "rng": _rng_state(),
                },
                checkpoint_path,
            )
            _write_manifest(
                manifest_path,
                {
                    "task": args.task,
                    "control_mode": args.control_mode,
                    "demo_path": args.demo_path,
                    "num_demos": len(dataset.trajectories["actions"]),
                    "observation_dim": observation_dim,
                    "action_dim": action_dim,
                    "total_iters": args.total_iters,
                    "iteration": completed,
                    "last_loss": last_loss,
                    "done": completed >= args.total_iters,
                    "seed": args.seed,
                },
            )

    _finalize(agent, ema_agent, ema, final_path)
    print(f"[done] {args.task}: trained to {args.total_iters}", flush=True)


def _finalize(agent: object, ema_agent: object, ema: object, final_path: Path) -> None:
    """Write the official-layout {agent, ema_agent} checkpoint with EMA weights."""
    ema.copy_to(ema_agent.parameters())  # type: ignore[attr-defined]
    _atomic_save(
        {
            "agent": agent.state_dict(),  # type: ignore[attr-defined]
            "ema_agent": ema_agent.state_dict(),  # type: ignore[attr-defined]
        },
        final_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--total-iters", type=int, default=30000)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    arguments = parser.parse_args()
    train(
        TrainArgs(
            task=arguments.task,
            demo_path=arguments.demo_path,
            run_dir=arguments.run_dir,
            total_iters=arguments.total_iters,
            num_demos=arguments.num_demos,
            seed=arguments.seed,
            checkpoint_every=arguments.checkpoint_every,
        )
    )


if __name__ == "__main__":
    main()
