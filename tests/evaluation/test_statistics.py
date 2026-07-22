from __future__ import annotations

import pytest

from actionshift.evaluation.statistics import (
    bootstrap_interval,
    holm_adjust,
    paired_success_difference,
    recovery_summary,
    superiority_allowed,
)


def test_seeded_bootstrap_is_reproducible_and_contains_mean() -> None:
    first = bootstrap_interval([1.0, 2.0, 3.0, 4.0], samples=2000, seed=7)
    second = bootstrap_interval([1.0, 2.0, 3.0, 4.0], samples=2000, seed=7)

    assert first == second
    assert first.estimate == 2.5
    assert first.lower <= first.estimate <= first.upper


def test_paired_success_difference_preserves_pairing() -> None:
    candidate = [True, True, False, True, True]
    baseline = [False, True, False, False, True]

    interval = paired_success_difference(candidate, baseline, samples=2000, seed=3)

    assert interval.estimate == pytest.approx(0.4)
    assert interval.pairs == 5
    with pytest.raises(ValueError, match="same number"):
        paired_success_difference([True], [False, True])


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(0 <= value <= 1 for value in adjusted)


def test_recovery_summary_retains_censoring_at_episode_horizon() -> None:
    summary = recovery_summary([4, None, 8, None], horizon=10)

    assert summary.observed_count == 2
    assert summary.censored_count == 2
    assert summary.observed_mean == 6.0
    assert summary.restricted_mean == 8.0


def test_superiority_requires_three_matched_seeds() -> None:
    assert not superiority_allowed([20260718])
    assert not superiority_allowed([20260718, 20260719])
    assert superiority_allowed([20260718, 20260719, 20260720])
