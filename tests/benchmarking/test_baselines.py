from __future__ import annotations

from pathlib import Path

import pytest

from actionshift.benchmarking.baselines import build_baseline_command, parse_baseline_output


def test_ppo_output_is_normalized() -> None:
    output = """
Epoch: 20, global_step=1945600
eval_return_mean=7.25
eval_success_once_mean=0.75
SPS: 42000
model saved to runs/gate0/ckpt_20.pt
"""

    record = parse_baseline_output("ppo", output, elapsed_seconds=47.0)

    assert record.step == 1_945_600
    assert record.success == 0.75
    assert record.task_return == 7.25
    assert record.throughput == 42_000.0
    assert record.elapsed_seconds == 47.0
    assert record.checkpoint == "runs/gate0/ckpt_20.pt"
    assert record.failure is None


def test_sac_and_tdmpc2_outputs_are_normalized(tmp_path: Path) -> None:
    sac = parse_baseline_output(
        "sac",
        "global_step=500000 success_once: 0.62, return: 4.50\n"
        "SPS: 12000\nmodel saved to runs/sac/ckpt_500000.pt\n",
        elapsed_seconds=80.0,
    )
    assert (sac.step, sac.success, sac.task_return) == (500_000, 0.62, 4.5)

    evaluation = tmp_path / "eval.csv"
    evaluation.write_text("step,return,success_once\n250000,1.0,0.2\n500000,3.0,0.6\n")
    tdmpc2 = parse_baseline_output(
        "tdmpc2",
        "train step: 500,000 fps: 3500\nSaved model to models/final.pt\n",
        elapsed_seconds=150.0,
        evaluation_csv=evaluation,
    )
    assert (tdmpc2.step, tdmpc2.success, tdmpc2.task_return) == (500_000, 0.6, 3.0)
    assert tdmpc2.throughput == 3500.0


def test_parser_records_missing_evaluation_instead_of_inventing_success() -> None:
    record = parse_baseline_output("ppo", "SPS: 1000\n", elapsed_seconds=1.0)

    assert record.success is None
    assert record.failure == "no evaluation success metric found"


def test_tdmpc2_parser_uses_eval_lines_not_training_rollouts() -> None:
    output = """
 eval             E: 0 I: 0 R: 1.25 S: 0.10 T: 0:00:01
 train            E: 32 I: 25000 R: 20.0 S: 0.95 T: 0:01:00
 eval             E: 32 I: 25600 R: 4.50 S: 0.40 T: 0:01:05
 train            E: 64 I: 50000 R: 30.0 S: 1.00 T: 0:02:00
"""

    record = parse_baseline_output("tdmpc2", output, elapsed_seconds=120.0)

    assert (record.step, record.success, record.task_return) == (25_600, 0.4, 4.5)


@pytest.mark.parametrize("method", ["ppo", "sac", "tdmpc2"])
def test_commands_freeze_task_controller_seed_budget_and_offline_logging(
    tmp_path: Path, method: str
) -> None:
    command = build_baseline_command(
        method,
        task="pick_cube",
        checkout=tmp_path / "maniskill",
        python=Path("/venv/bin/python"),
        budget_steps=123_456,
        seed=20260718,
        run_name="gate0-pick-identity",
    )
    joined = " ".join(command.argv)

    assert command.applicable
    assert command.reason is None
    assert "PickCube-v1" in joined
    assert "pd_ee_delta_pose" in joined
    assert "123456" in joined or "123_456" in joined
    assert "20260718" in joined
    assert "wandb=false" in joined or "--track" not in joined


def test_unknown_method_or_task_is_structurally_inapplicable(tmp_path: Path) -> None:
    command = build_baseline_command(
        "fasttd3",
        task="pick_cube",
        checkout=tmp_path,
        python=Path("python"),
        budget_steps=100,
        seed=1,
        run_name="ignored",
    )

    assert not command.applicable
    assert command.argv == ()
    assert command.reason == "official FastTD3 repository has no ManiSkill adapter"


def test_sac_evaluation_horizon_covers_peg_insertion_episode(tmp_path: Path) -> None:
    command = build_baseline_command(
        "sac",
        task="peg_insertion_side",
        checkout=tmp_path,
        python=Path("python"),
        budget_steps=250_000,
        seed=20260718,
        run_name="peg-corrected",
    )

    assert "--num-eval-steps=100" in command.argv


def test_tdmpc2_short_budget_schedules_a_midrun_evaluation(tmp_path: Path) -> None:
    command = build_baseline_command(
        "tdmpc2",
        task="pick_cube",
        checkout=tmp_path,
        python=Path("python"),
        budget_steps=50_000,
        seed=20260718,
        run_name="td-evaluated",
    )

    assert "eval_freq=25000" in command.argv
    assert "eval_episodes_per_env=7" in command.argv
