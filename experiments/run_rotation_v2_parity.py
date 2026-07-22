"""v2 real-rotation oracle-parity and Gate-1-style frame=tool ceiling cells.

Isolates the frame axis with a PURE-FRAME contract (identity permutation/sign/
scale, ``frame="tool"``): the only thing that differs from the unwrapped policy is
that the pose twist is expressed in the tool frame. Under the identity-rotation
wrapper this contract is observationally identical to base (the documented scope
weakness); under the live tcp rotation it is a genuine axis.

Cells per task (100 episodes, num_envs 16), all on the frozen Gate-1 backbone:

- ``ceiling``               identity condition           -> the clean backbone ceiling
- ``oracle_tool_real``      oracle,  tool, rotation real  -> ~ceiling proves exact inversion
- ``noadapt_tool_real``     no-adapt, tool, rotation real -> << ceiling proves the axis is live
- ``oracle_tool_identity``  oracle,  tool, rotation ident -> v1 control (~ceiling, degenerate)
- ``noadapt_tool_identity`` no-adapt, tool, rotation ident-> v1 control (~ceiling, DEGENERATE)

The pair (noadapt_tool_identity ~ ceiling) vs (noadapt_tool_real << ceiling) is the
de-degeneration evidence: v1 cannot separate base from tool, v2 can. Runs are
hash-addressed with rotation_mode in the job hash.

Usage (GPU 1 only):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/run_rotation_v2_parity.py \
    --task pick_cube --output artifacts/adaptation/v2_rotation_slices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from actionshift.benchmarking.ppo_parity import (
    ParityCondition,
    RotationMode,
    evaluate_ppo_checkpoint,
    identity_contract,
)
from actionshift.contracts.types import ActionContract
from actionshift.evaluation.provenance import sha256_file

_SEED = 20260718
_EPISODES = 100
_NUM_ENVS = 16


def frozen_checkpoint(task: str) -> Path:
    """Locate the frozen Gate-1 PPO checkpoint recorded for one task."""
    for line in Path("artifacts/sprint/gate1/jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job.get("task") == task:
            return Path(job["checkpoint"])
    raise LookupError(f"no frozen Gate 1 checkpoint recorded for task {task}")


def _tool_frame_contract() -> ActionContract:
    """Pure-frame contract: identity everywhere except ``frame="tool"``."""
    return replace(identity_contract(), frame="tool")


def _cell(
    checkpoint: Path,
    *,
    task: str,
    condition: ParityCondition,
    contract: ActionContract | None,
    rotation_mode: RotationMode,
) -> dict[str, object]:
    episodes = evaluate_ppo_checkpoint(
        checkpoint,
        task=task,
        condition=condition,
        seed=_SEED,
        episodes=_EPISODES,
        num_envs=_NUM_ENVS,
        contract=contract,
        rotation_mode=rotation_mode,
    )
    successes = sum(1 for episode in episodes if episode.success)
    return {
        "condition": condition,
        "rotation_mode": rotation_mode,
        "frame": contract.frame if contract is not None else None,
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()

    checkpoint = frozen_checkpoint(arguments.task)
    tool = _tool_frame_contract()
    cells = {
        "ceiling": _cell(
            checkpoint, task=arguments.task, condition="identity",
            contract=None, rotation_mode="real",
        ),
        "oracle_tool_real": _cell(
            checkpoint, task=arguments.task, condition="oracle_nonidentity",
            contract=tool, rotation_mode="real",
        ),
        "noadapt_tool_real": _cell(
            checkpoint, task=arguments.task, condition="noadapt_nonidentity",
            contract=tool, rotation_mode="real",
        ),
        "oracle_tool_identity": _cell(
            checkpoint, task=arguments.task, condition="oracle_nonidentity",
            contract=tool, rotation_mode="identity",
        ),
        "noadapt_tool_identity": _cell(
            checkpoint, task=arguments.task, condition="noadapt_nonidentity",
            contract=tool, rotation_mode="identity",
        ),
    }

    job = {
        "schema_version": "1.0",
        "kind": "rotation_v2_parity",
        "task": arguments.task,
        "seed": _SEED,
        "episodes": _EPISODES,
        "num_envs": _NUM_ENVS,
        "checkpoint_sha256": sha256_file(checkpoint),
        "tool_contract": json.loads(tool.to_json()),
    }
    job_id = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    summary = {**job, "job_id": job_id, "cells": cells, "elapsed_seconds": time.time() - started}
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / f"parity-{arguments.task}-{job_id}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
