from __future__ import annotations

import math

import pytest

from actionshift.evaluation.metrics import EpisodeMetrics, summarize


def test_summary_reports_task_adaptation_safety_and_calibration() -> None:
    episodes = [
        EpisodeMetrics(True, 10, 2, 0.1, 0, 1.2, 0.8, 0.7, 0.2),
        EpisodeMetrics(False, 20, None, 0.3, 1, 2.0, 0.4, 0.5, 0.8),
    ]
    report = summarize(episodes, confidence=0.95)
    assert report["episode_count"] == 2
    assert report["success_rate"]["mean"] == 0.5
    assert report["safety_violation_rate"]["mean"] == 0.5
    assert report["recovery_steps"]["observed_count"] == 1
    assert report["posterior_brier_score"]["mean"] > 0
    assert math.isfinite(report["success_rate"]["lower"])
    assert report["success_rate"]["lower"] == pytest.approx(0.0945312057)
    assert report["success_rate"]["upper"] == pytest.approx(0.9054687943)


def test_summary_rejects_empty_or_nonfinite_records() -> None:
    try:
        summarize([])
    except ValueError as error:
        assert "episode" in str(error)
    else:
        raise AssertionError("empty evaluation should fail")

    try:
        summarize([EpisodeMetrics(True, 1, 0, float("nan"), 0, 0, 0, 1, 0)])
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite evaluation should fail")
