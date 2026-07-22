"""Real ManiSkill oracle-encoding smoke for every frozen task adapter."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from actionshift.contracts.transforms import encode_complete_action
from actionshift.contracts.types import ActionContract
from actionshift.envs.tasks import TASKS, make_task_env
from actionshift.envs.wrapper import OracleRecorder


def _smoke_contract() -> ActionContract:
    return ActionContract(
        permutation=(1, 0, 2, 4, 3, 5),
        sign=(-1, 1, 1, -1, 1, 1),
        scale=(0.5, 1.5, 1.0, 1.0, 0.75, 1.25),
        target="delta",
        frame="base",
        lag=0,
        gripper_inverted=True,
    )


def run_maniskill_smoke(*, sim_backend: str = "cpu", steps: int = 3) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    contract = _smoke_contract()
    task_reports: dict[str, Any] = {}
    for task_name, task in TASKS.items():
        recorder = OracleRecorder()
        environment = make_task_env(
            task,
            contract,
            oracle_recorder=recorder,
            num_envs=1,
            sim_backend=sim_backend,
        )
        started = time.perf_counter()
        try:
            observation, info = environment.reset(seed=20260718)
            observation_tensor = torch.as_tensor(observation)
            device = observation_tensor.device
            dtype = observation_tensor.dtype
            canonical = torch.tensor(
                [[0.005, -0.003, 0.002, 0.0, 0.0, 0.002, 0.0]],
                device=device,
                dtype=dtype,
            )
            rotation = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            tracked_target = torch.zeros(1, 6, device=device, dtype=dtype)
            raw = encode_complete_action(
                canonical,
                contract,
                ee_rotation=rotation,
                tracked_target=tracked_target,
            )
            finite = bool(torch.isfinite(observation_tensor).all())
            success_seen = bool(task.success(info).any())
            for _ in range(steps):
                observation, _, _, _, info = environment.step(raw)
                finite = finite and bool(torch.isfinite(torch.as_tensor(observation)).all())
                success_seen = success_seen or bool(task.success(info).any())
            decoded = torch.stack(
                [record["oracle/decoded_action"].to(device=device) for record in recorder.records]
            )
            maximum_decode_error = float((decoded - canonical).abs().max())
            task_reports[task_name] = {
                "environment_id": task.environment_id,
                "observation_shape": list(torch.as_tensor(observation).shape),
                "action_shape": list(raw.shape),
                "steps": steps,
                "finite_observations": finite,
                "oracle_records": len(recorder.records),
                "maximum_decode_error": maximum_decode_error,
                "success_seen": success_seen,
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            environment.close()
    passed = all(
        report["finite_observations"]
        and report["oracle_records"] == steps
        and report["maximum_decode_error"] <= 1e-6
        for report in task_reports.values()
    )
    return {
        "schema_version": "1.0",
        "kind": "maniskill_oracle_encoding_smoke",
        "sim_backend": sim_backend,
        "steps_per_task": steps,
        "contract": json.loads(contract.to_json()),
        "tasks": task_reports,
        "passed": passed,
        "limitations": [
            "This verifies real simulator integration and oracle encoding, "
            "not learned-policy success.",
            "The short trajectory is not a performance comparison between adaptation methods.",
        ],
    }


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_maniskill_smoke(sim_backend=arguments.sim_backend, steps=arguments.steps)
    _write_atomic(arguments.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
