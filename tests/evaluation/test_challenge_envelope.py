"""Adversarial public adapter inputs; no simulator or learned policy required."""

import math

import pytest

from actionshift.evaluation.challenge import (
    ChallengeDecision,
    IdentityChallengeAdapter,
    MethodBudget,
    run_challenge,
)


def test_fractional_probe_flag_cannot_bypass_the_probe_budget() -> None:
    class FractionalProbe(IdentityChallengeAdapter):
        def act(self, observation):
            return ChallengeDecision(observation.canonical_action, 0.5)

    with pytest.raises(TypeError, match=r"probe.*bool"):
        run_challenge(
            method="external", factory=lambda **_: FractionalProbe(), seeds=(3,), steps=16
        )


@pytest.mark.parametrize("bound", [math.nan, math.inf])
def test_action_envelope_requires_finite_bounds(bound: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MethodBudget(16, 6, max_action_linf=bound)


def test_duplicate_episode_seeds_cannot_inflate_the_denominator() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_challenge(method="no_adapt", seeds=(3, 3), steps=16)


def test_failed_adapters_remain_in_the_planned_episode_denominator() -> None:
    def broken_factory(**kwargs):
        raise RuntimeError("private exception detail should not be serialized")

    report = run_challenge(
        method="external",
        factory=broken_factory,
        seeds=(3,),
        steps=16,
        continue_on_error=True,
    )
    assert report["episode_count"] == report["failed_episodes"] == 5
    assert report["successes"] == report["scored_episode_count"] == 0
    assert report["mean_post_lag_mse"] is None
    assert all(record["status"] == "FAILED" for record in report["episodes"])
    assert "private exception detail" not in str(report)
