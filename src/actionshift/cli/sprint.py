"""Command-line entry point for planning, resuming, and auditing sprint jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from actionshift.benchmarking.baselines import build_baseline_command, parse_baseline_output
from actionshift.benchmarking.gates import analyze_parity_directory, overall_gate_status
from actionshift.benchmarking.scheduler import CommandSpec, SprintScheduler
from actionshift.benchmarking.schema import Gate, SprintJob, SprintResult
from actionshift.benchmarking.store import ArtifactStore


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def plan_jobs(config_path: str | Path) -> tuple[SprintJob, ...]:
    """Plan one anchor training per method/task; parity reuses frozen checkpoints."""
    config = _mapping(Path(config_path))
    tasks = config.get("tasks")
    methods = config.get("methods")
    seed = config.get("seed")
    gate = config.get("gate")
    condition = config.get("condition", "unwrapped")
    if not isinstance(tasks, dict) or not isinstance(methods, list):
        raise ValueError("tasks must be a mapping and methods must be a list")
    if not isinstance(seed, int) or gate not in {"gate0", "gate1", "actionabi"}:
        raise ValueError("seed and gate are invalid")
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a nonempty string")
    jobs: list[SprintJob] = []
    for gpu, (task, task_config) in enumerate(tasks.items()):
        if not isinstance(task, str) or not isinstance(task_config, dict):
            raise ValueError("task entries must be named mappings")
        budgets = task_config.get("budgets")
        assigned_gpu = task_config.get("gpu", gpu)
        if not isinstance(budgets, dict):
            raise ValueError(f"task {task} must define budgets")
        if isinstance(assigned_gpu, bool) or not isinstance(assigned_gpu, int):
            raise ValueError(f"task {task} GPU must be an integer")
        for method in methods:
            if not isinstance(method, str):
                raise ValueError("methods must contain strings")
            budget = budgets.get(method, 1 if method == "fasttd3" else None)
            if not isinstance(budget, int):
                raise ValueError(f"missing integer budget for {task}/{method}")
            jobs.append(
                SprintJob(
                    cast(Gate, gate), task, method, seed, assigned_gpu, budget, condition
                )
            )
    return tuple(jobs)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jobs(path: Path, jobs: tuple[SprintJob, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(
        "".join(json.dumps(job.to_dict(), sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_jobs(path: Path) -> tuple[SprintJob, ...]:
    jobs: list[SprintJob] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"job line {line_number} must be an object")
        job = SprintJob(
            gate=cast(Gate, value["gate"]),
            task=str(value["task"]),
            method=str(value["method"]),
            seed=int(value["seed"]),
            gpu=int(value["gpu"]),
            budget_steps=int(value["budget_steps"]),
            condition=str(value["condition"]),
            schema_version=str(value["schema_version"]),
        )
        if value.get("job_id") != job.job_id:
            raise ValueError(f"job ID mismatch on line {line_number}")
        jobs.append(job)
    return tuple(jobs)


def _plan(arguments: argparse.Namespace) -> int:
    jobs = plan_jobs(arguments.config)
    _write_jobs(arguments.output, jobs)
    print(json.dumps({"jobs": len(jobs), "output": str(arguments.output)}, sort_keys=True))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    jobs = _load_jobs(arguments.jobs)
    allowed_gpus = {int(value) for value in arguments.gpus.split(",")}
    if any(job.gpu not in allowed_gpus for job in jobs):
        raise ValueError("job matrix assigns a GPU outside --gpus")
    store = ArtifactStore(arguments.artifacts)
    specs: list[CommandSpec] = []
    for job in store.pending(jobs, stale_after_seconds=arguments.stale_after_minutes * 60):
        command = build_baseline_command(
            job.method,
            task=job.task,
            checkout=arguments.checkout,
            python=Path(sys.executable),
            budget_steps=job.budget_steps,
            seed=job.seed,
            run_name=job.job_id,
        )
        if not command.applicable:
            if store.claim(job, attempt=0):
                store.finish(
                    SprintResult(job.job_id, "inapplicable", 0, 0.0, None, command.reason)
                )
            continue
        specs.append(CommandSpec(job, command.argv, cwd=command.cwd))
    results = SprintScheduler(store).run(
        specs,
        deadline_seconds=arguments.deadline_minutes * 60,
        max_retries=arguments.max_retries,
    )
    print(json.dumps({"terminal_this_run": len(results)}, sort_keys=True))
    return 0 if all(result.status == "completed" for result in results.values()) else 1


def _status(arguments: argparse.Namespace) -> int:
    jobs = _load_jobs(arguments.jobs)
    store = ArtifactStore(arguments.artifacts)
    terminal = sum((store.results / f"{job.job_id}.json").is_file() for job in jobs)
    running = sum((store.claims / f"{job.job_id}.json").is_file() for job in jobs)
    print(json.dumps({"total": len(jobs), "terminal": terminal, "running": running}))
    return 0


def _analyze(arguments: argparse.Namespace) -> int:
    jobs = _load_jobs(arguments.jobs)
    store = ArtifactStore(arguments.artifacts)
    records: list[dict[str, Any]] = []
    for job in jobs:
        result_path = store.results / f"{job.job_id}.json"
        if not result_path.is_file():
            records.append({"job": job.to_dict(), "status": "missing"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            records.append({"job": job.to_dict(), "result": result})
            continue
        logs = store.root / "logs"
        stdout = logs / f"{job.job_id}.attempt-{result['attempt']}.stdout"
        stderr = logs / f"{job.job_id}.attempt-{result['attempt']}.stderr"
        text = ""
        for path in (stdout, stderr):
            if path.is_file():
                text += path.read_text(encoding="utf-8", errors="replace") + "\n"
        observation = parse_baseline_output(
            job.method, text, elapsed_seconds=float(result["elapsed_seconds"])
        )
        records.append({"job": job.to_dict(), "result": result, "observation": asdict(observation)})
    parity = analyze_parity_directory(store.root / "parity")
    gate_status, gate_reason = overall_gate_status(parity)
    report = {
        "schema_version": "1.0",
        "records": records,
        "parity_gates": parity,
        "gate_status": gate_status,
        "reason": gate_reason,
    }
    _atomic_json(arguments.output, report)
    print(arguments.output)
    return 0 if gate_status == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="actionshift-sprint")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(handler=_plan)
    run = subparsers.add_parser("run")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--gpus", default="0,1,2,3")
    run.add_argument("--deadline-minutes", type=float, required=True)
    run.add_argument("--stale-after-minutes", type=float, default=15.0)
    run.add_argument("--max-retries", type=int, default=1)
    run.add_argument("--checkout", type=Path, default=Path("third_party/maniskill"))
    run.set_defaults(handler=_run)
    status = subparsers.add_parser("status")
    status.add_argument("--jobs", type=Path, required=True)
    status.add_argument("--artifacts", type=Path, required=True)
    status.set_defaults(handler=_status)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--jobs", type=Path, required=True)
    analyze.add_argument("--artifacts", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(handler=_analyze)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        raise SystemExit(arguments.handler(arguments))
    except (KeyError, TypeError, ValueError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
