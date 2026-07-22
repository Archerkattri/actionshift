from __future__ import annotations

import json
from pathlib import Path

import pytest

from actionshift.benchmarking.gates import (
    analyze_parity_directory,
    competence_verdict,
    overall_gate_status,
    parity_verdict,
    wilson_interval,
)
from actionshift.cli.sprint import build_parser, plan_jobs


def test_wilson_interval_matches_known_binomial_case() -> None:
    interval = wilson_interval(50, 100)

    assert interval.rate == 0.5
    assert interval.lower == pytest.approx(0.4038315304)
    assert interval.upper == pytest.approx(0.5961684696)
    assert interval.successes == 50
    assert interval.total == 100


@pytest.mark.parametrize(
    ("task", "successes", "total", "passed"),
    [
        ("pick_cube", 50, 100, True),
        ("pick_cube", 49, 100, False),
        ("push_cube", 50, 100, True),
        ("peg_insertion_side", 20, 100, True),
        ("peg_insertion_side", 19, 100, False),
    ],
)
def test_competence_floors_are_preregistered(
    task: str, successes: int, total: int, passed: bool
) -> None:
    verdict = competence_verdict(task, successes, total)

    assert verdict.passed is passed
    assert verdict.reason == (
        None if passed else f"success {successes / total:.4f} below floor {verdict.floor:.4f}"
    )


def test_parity_accepts_two_points_or_overlapping_wilson_intervals() -> None:
    within_two_points = parity_verdict(75, 100, 73, 100)
    overlaps = parity_verdict(6, 10, 8, 10)

    assert within_two_points.passed
    assert within_two_points.reason is None
    assert overlaps.absolute_difference == pytest.approx(0.2)
    assert overlaps.intervals_overlap
    assert overlaps.passed


def test_parity_rejects_material_nonoverlapping_regression() -> None:
    verdict = parity_verdict(90, 100, 50, 100)

    assert not verdict.passed
    assert verdict.reason == (
        "absolute success difference 0.4000 exceeds 0.0200 and CIs do not overlap"
    )


@pytest.mark.parametrize("function", [competence_verdict, parity_verdict])
def test_gates_reject_missing_episode_counts(function: object) -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        if function is competence_verdict:
            competence_verdict("pick_cube", 0, 0)
        else:
            parity_verdict(0, 0, 0, 10)


def test_sprint_cli_exposes_four_commands_and_plans_one_training_per_anchor() -> None:
    parser = build_parser()
    assert {"plan", "run", "status", "analyze"} <= set(parser._subparsers._group_actions[0].choices)

    jobs = plan_jobs("configs/sprint/gate0.yaml")
    assert len(jobs) == 12
    assert {job.condition for job in jobs} == {"unwrapped"}
    assert {job.gpu for job in jobs if job.task == "pick_cube"} == {0}
    assert {job.gpu for job in jobs if job.task == "push_cube"} == {1}
    assert {job.gpu for job in jobs if job.task == "peg_insertion_side"} == {2}


def test_parity_directory_produces_machine_readable_gate_verdict(tmp_path: Path) -> None:
    for condition, successes in (
        ("unwrapped", 90),
        ("identity", 89),
        ("oracle_nonidentity", 90),
    ):
        path = tmp_path / f"push_cube-{condition}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"success": index < successes, "episode_index": index}) + "\n"
                for index in range(100)
            ),
            encoding="utf-8",
        )

    report = analyze_parity_directory(tmp_path)

    push = report["tasks"]["push_cube"]
    assert push["competence"]["passed"]
    assert push["identity_parity"]["passed"]
    assert push["oracle_parity"]["passed"]
    assert report["all_available_tasks_passed"]
    assert not report["all_tasks_complete"]


def test_parity_directory_preserves_missing_task_reason(tmp_path: Path) -> None:
    report = analyze_parity_directory(tmp_path)

    assert not report["all_available_tasks_passed"]
    assert not report["all_tasks_complete"]
    assert report["tasks"]["pick_cube"]["reason"] == "missing parity episode files"


def test_planner_includes_declared_protocol_correction_in_job_identity(tmp_path: Path) -> None:
    config = tmp_path / "correction.yaml"
    config.write_text(
        """
gate: gate0
seed: 20260718
condition: unwrapped_eval100
tasks:
  peg_insertion_side:
    gpu: 3
    budgets: {sac: 250000}
methods: [sac]
""",
        encoding="utf-8",
    )

    job = plan_jobs(config)[0]

    assert job.condition == "unwrapped_eval100"
    assert job.gpu == 3


def test_overall_gate_reason_distinguishes_failure_from_missing_artifacts() -> None:
    assert overall_gate_status(
        {"all_tasks_complete": True, "all_available_tasks_passed": False}
    ) == ("failed", "one or more preregistered task gates failed")
    assert overall_gate_status(
        {"all_tasks_complete": False, "all_available_tasks_passed": True}
    ) == ("incomplete", "competence and parity require all episode-level artifacts")
    assert overall_gate_status(
        {"all_tasks_complete": True, "all_available_tasks_passed": True}
    ) == ("passed", None)
