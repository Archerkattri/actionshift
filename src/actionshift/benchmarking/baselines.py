"""Adapters for pinned official ManiSkill task-policy implementations."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

_TASKS = {
    "pick_cube": "PickCube-v1",
    "push_cube": "PushCube-v1",
    "pull_cube": "PullCube-v1",
    "stack_cube": "StackCube-v1",
    "peg_insertion_side": "PegInsertionSide-v1",
}
_METHODS = frozenset({"ppo", "sac", "tdmpc2"})
_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


@dataclass(frozen=True, slots=True)
class BaselineCommand:
    method: str
    argv: tuple[str, ...]
    cwd: Path | None
    applicable: bool
    reason: str | None
    protocol_label: str


@dataclass(frozen=True, slots=True)
class BaselineObservation:
    method: str
    step: int | None
    success: float | None
    task_return: float | None
    throughput: float | None
    elapsed_seconds: float
    checkpoint: str | None
    failure: str | None

    def __post_init__(self) -> None:
        values = (self.success, self.task_return, self.throughput, self.elapsed_seconds)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("baseline observations must be finite")
        if self.success is not None and not 0 <= self.success <= 1:
            raise ValueError("success must be in [0, 1]")


def _last_number(text: str, patterns: tuple[str, ...]) -> float | None:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern + _NUMBER, text, flags=re.IGNORECASE))
    return float(matches[-1].replace(",", "")) if matches else None


def _last_checkpoint(text: str) -> str | None:
    matches = re.findall(
        r"(?:model saved to|saved model to)\s+([^\s]+)", text, flags=re.IGNORECASE
    )
    return matches[-1] if matches else None


def _csv_last(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return rows[-1] if rows else {}


def parse_baseline_output(
    method: str,
    text: str,
    *,
    elapsed_seconds: float,
    evaluation_csv: Path | None = None,
) -> BaselineObservation:
    """Normalize official console/CSV output without imputing absent metrics."""
    if method not in _METHODS:
        raise ValueError(f"unsupported official baseline: {method}")
    step_value = _last_number(
        text,
        (
            r"global[_ ]step\s*[=:]\s*",
            r"train\s+step\s*[=:]\s*",
        ),
    )
    success = _last_number(
        text,
        (
            r"eval_success(?:_once|_at_end)?_mean\s*=\s*",
            r"success_once\s*:\s*",
        ),
    )
    task_return = _last_number(
        text,
        (
            r"eval_return_mean\s*=\s*",
            r"return\s*:\s*",
        ),
    )
    throughput = _last_number(text, (r"SPS\s*:\s*", r"fps\s*:\s*"))
    if method == "tdmpc2":
        evaluations = re.findall(
            rf"^\s*eval\s+E:\s*[\d,]+\s+I:\s*([\d,]+)\s+R:\s*{_NUMBER}\s+S:\s*{_NUMBER}",
            text,
            flags=re.MULTILINE,
        )
        if evaluations:
            raw_step, raw_return, raw_success = evaluations[-1]
            step_value = float(raw_step.replace(",", ""))
            task_return = float(raw_return)
            success = float(raw_success)
    if evaluation_csv is not None and evaluation_csv.is_file():
        row = _csv_last(evaluation_csv)
        if row.get("step"):
            step_value = float(row["step"])
        if row.get("success_once"):
            success = float(row["success_once"])
        if row.get("return"):
            task_return = float(row["return"])
    return BaselineObservation(
        method=method,
        step=int(step_value) if step_value is not None else None,
        success=success,
        task_return=task_return,
        throughput=throughput,
        elapsed_seconds=elapsed_seconds,
        checkpoint=_last_checkpoint(text),
        failure=None if success is not None else "no evaluation success metric found",
    )


def build_baseline_command(
    method: str,
    *,
    task: str,
    checkout: Path,
    python: Path,
    budget_steps: int,
    seed: int,
    run_name: str,
) -> BaselineCommand:
    """Build a frozen offline command or an explicit inapplicability record."""
    if method == "fasttd3":
        return BaselineCommand(
            method,
            (),
            None,
            False,
            "official FastTD3 repository has no ManiSkill adapter",
            "inapplicable",
        )
    if method not in _METHODS:
        return BaselineCommand(
            method, (), None, False, f"unknown baseline method: {method}", "inapplicable"
        )
    environment_id = _TASKS.get(task)
    if environment_id is None:
        return BaselineCommand(
            method, (), None, False, f"unknown benchmark task: {task}", "inapplicable"
        )
    if budget_steps <= 0:
        raise ValueError("budget_steps must be positive")
    root = checkout / "examples" / "baselines" / method
    executable = str(python)
    argv: tuple[str, ...]
    if method == "ppo":
        argv = (
            executable,
            "ppo.py",
            f"--env_id={environment_id}",
            f"--seed={seed}",
            "--control_mode=pd_ee_delta_pose",
            f"--total_timesteps={budget_steps}",
            "--num_envs=1024",
            "--num_eval_envs=16",
            "--num-eval-steps=100",
            "--eval-freq=10",
            "--no-capture-video",
            f"--exp-name={run_name}",
        )
    elif method == "sac":
        argv = (
            executable,
            "sac.py",
            f"--env_id={environment_id}",
            f"--seed={seed}",
            "--control-mode=pd_ee_delta_pose",
            f"--total-timesteps={budget_steps}",
            "--num-envs=32",
            "--num-eval-envs=16",
            "--num-eval-steps=100",
            "--eval-freq=50000",
            "--no-capture-video",
            f"--exp-name={run_name}",
        )
    else:
        evaluation_frequency = max(32, budget_steps // 2)
        argv = (
            executable,
            "train.py",
            "model_size=5",
            f"steps={budget_steps}",
            f"seed={seed}",
            f"env_id={environment_id}",
            "env_type=gpu",
            "num_envs=32",
            "num_eval_envs=16",
            "eval_episodes_per_env=7",
            "control_mode=pd_ee_delta_pose",
            "obs=state",
            "wandb=false",
            "save_video_local=false",
            f"eval_freq={evaluation_frequency}",
            f"exp_name={run_name}",
        )
    return BaselineCommand(
        method=method,
        argv=argv,
        cwd=root,
        applicable=True,
        reason=None,
        protocol_label="official_code_common_controller_short_budget",
    )
