"""Driver for the delay-aware augmented-state SAC battery on the DCAC benchmark.

Enumerates every run, keeps a JSON manifest of completed/in-progress/pending runs,
and launches ``sac_delay_mujoco.py`` subprocesses on GPU 2 with a fixed concurrency
cap. It is fully idempotent: a completed run (its ``summary.json`` exists) is skipped
by the training script itself, and an interrupted run resumes from its latest
checkpoint -- so this driver can be killed and re-launched at any time.

Battery (15 runs), all constant delay omega=2/alpha=3 (total 5) unless noted:
  * delay-5 AUGMENTED  : {HalfCheetah, Walker2d, Ant}-v4 x seeds {1,2,3}  = 9 (headline)
  * UNDELAYED augmented: {HalfCheetah, Walker2d, Ant}-v4 x seed 1          = 3 (retention denom)
  * delay-5 NAIVE       : {HalfCheetah, Walker2d, Ant}-v4 x seed 1          = 3 (collapse control)

Launch (background) on a single GPU:

  CUDA_VISIBLE_DEVICES=2 .venv-delayext/bin/python \
    experiments/delayed_external/run_sac_battery.py \
    --output artifacts/delayed_external_sac --max-concurrent 5

Fan-out across several GPUs (round-robin per run). ``--respect-manifest`` leaves any
run already marked ``running``/``complete`` in the existing manifest untouched (so a
separate driver's in-flight runs are never duplicated), launching only the pending
ones. ``--gpus`` overrides CUDA_VISIBLE_DEVICES for the children (one GPU per run):

  .venv-delayext/bin/python experiments/delayed_external/run_sac_battery.py \
    --output artifacts/delayed_external_sac --gpus 0,1,3 \
    --respect-manifest --max-concurrent 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENVS = ["HalfCheetah-v4", "Walker2d-v4", "Ant-v4"]
HERE = Path(__file__).resolve().parent


def build_runs() -> list[dict]:
    runs: list[dict] = []
    for env in ENVS:
        for seed in (1, 2, 3):  # delay-5 augmented (headline)
            runs.append(
                {"env": env, "seed": seed, "od": 2, "ad": 3, "aug": 1, "role": "delay5_aug"}
            )
    for env in ENVS:  # undelayed retention denominator
        runs.append({"env": env, "seed": 1, "od": 0, "ad": 0, "aug": 1, "role": "undelayed"})
    for env in ENVS:  # naive non-augmented delayed collapse control
        runs.append({"env": env, "seed": 1, "od": 2, "ad": 3, "aug": 0, "role": "naive_delay5"})
    return runs


def run_key(r: dict) -> str:
    tag = "aug" if r["aug"] else "naive"
    return f"{r['env']}_od{r['od']}_ad{r['ad']}_{tag}_s{r['seed']}"


def find_summary(output: Path, r: dict) -> Path | None:
    tag = "aug" if r["aug"] else "naive"
    pat = str(
        output / r["env"] / f"od{r['od']}_ad{r['ad']}_{tag}"
        / f"seed{r['seed']}_*" / "summary.json"
    )
    hits = glob.glob(pat)
    return Path(hits[0]) if hits else None


def is_complete(output: Path, r: dict) -> bool:
    return find_summary(output, r) is not None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/delayed_external_sac"))
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--omp-threads", type=int, default=4)
    p.add_argument(
        "--gpus", type=str, default=None,
        help="comma list of GPU ids to spread runs over (one GPU/run, balanced). "
             "Default: inherit CUDA_VISIBLE_DEVICES for every run.",
    )
    p.add_argument(
        "--respect-manifest", action="store_true",
        help="leave runs already marked running/complete in the existing manifest "
             "untouched (never relaunch another driver's in-flight runs).",
    )
    args = p.parse_args()
    gpu_list = [g.strip() for g in args.gpus.split(",")] if args.gpus else None

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    runs = build_runs()

    def write_manifest(states: dict) -> None:
        payload = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "total_timesteps": args.total_timesteps,
            "max_concurrent": args.max_concurrent,
            "n_runs": len(runs),
            "runs": states,
        }
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, manifest_path)

    prior: dict = {}
    if args.respect_manifest and manifest_path.exists():
        prior = json.loads(manifest_path.read_text()).get("runs", {})

    states: dict[str, dict] = {}
    for r in runs:
        k = run_key(r)
        if is_complete(output, r):
            status = "complete"
        elif args.respect_manifest and prior.get(k, {}).get("status") == "running":
            status = "running"  # owned by another driver -> do not relaunch
        else:
            status = "pending"
        states[k] = {**r, "status": status}
        if status == "complete":
            s = json.loads(find_summary(output, r).read_text())
            states[k]["final_return"] = s.get("final_mean_return_last100")
    write_manifest(states)

    pending = [r for r in runs if states[run_key(r)]["status"] == "pending"]
    print(f"[battery] {len(runs)} runs total, {len(pending)} pending, "
          f"{len(runs) - len(pending)} already complete. maxconc={args.max_concurrent}")

    active: dict[str, subprocess.Popen] = {}
    active_gpu: dict[str, str] = {}
    logs_dir = output / "logs"
    logs_dir.mkdir(exist_ok=True)
    queue = list(pending)

    def pick_gpu() -> str | None:
        if not gpu_list:
            return None  # inherit CUDA_VISIBLE_DEVICES
        # balanced: assign to the GPU currently running the fewest of our runs
        load = dict.fromkeys(gpu_list, 0)
        for g in active_gpu.values():
            if g in load:
                load[g] += 1
        return min(gpu_list, key=lambda g: load[g])

    def launch(r: dict) -> None:
        k = run_key(r)
        env = os.environ.copy()
        gpu = pick_gpu()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
            active_gpu[k] = gpu
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        cmd = [
            sys.executable, str(HERE / "sac_delay_mujoco.py"),
            "--env-id", r["env"], "--seed", str(r["seed"]),
            "--obs-delay", str(r["od"]), "--act-delay", str(r["ad"]),
            "--augment", str(r["aug"]),
            "--total-timesteps", str(args.total_timesteps),
            "--output", str(output),
        ]
        logf = (logs_dir / f"{k}.log").open("w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        active[k] = proc
        states[k]["status"] = "running"
        states[k]["pid"] = proc.pid
        states[k]["gpu"] = gpu if gpu is not None \
            else os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        states[k]["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_manifest(states)
        print(f"[battery] launch {k} gpu={states[k]['gpu']} pid={proc.pid} "
              f"({len(active)} active, {len(queue)} queued)")

    while queue or active:
        while queue and len(active) < args.max_concurrent:
            launch(queue.pop(0))
        time.sleep(5)
        for k, proc in list(active.items()):
            if proc.poll() is None:
                continue
            del active[k]
            active_gpu.pop(k, None)
            r = next(x for x in runs if run_key(x) == k)
            summ = find_summary(output, r)
            if summ is not None:
                s = json.loads(summ.read_text())
                states[k]["status"] = "complete"
                states[k]["final_return"] = s.get("final_mean_return_last100")
                states[k]["elapsed_seconds"] = s.get("elapsed_seconds")
            else:
                states[k]["status"] = f"failed_rc{proc.returncode}"
            states[k]["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_manifest(states)
            print(f"[battery] finished {k} -> {states[k]['status']} "
                  f"ret={states[k].get('final_return')}")

    done = sum(1 for v in states.values() if v["status"] == "complete")
    print(f"[battery] ALL DONE. {done}/{len(runs)} complete. manifest={manifest_path}")


if __name__ == "__main__":
    main()
