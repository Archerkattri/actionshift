from __future__ import annotations

import pytest

from actionshift.benchmarking.adaptation import (
    FrozenProtocol,
    method_registry,
    plan_gate1_pilots,
    promote_seeds,
    superiority_eligible,
)


def protocol() -> FrozenProtocol:
    return FrozenProtocol(
        backbone_sha256="a" * 64,
        actor_parameters=1234,
        critic_parameters=2345,
        history_steps=16,
        observation="state",
        environment_steps=1_000_000,
        updates=10_000,
        train_split_sha256="b" * 64,
        evaluation_split_sha256="c" * 64,
        probe_budget=4,
    )


def test_all_runnable_methods_share_protocol_except_declared_privileges() -> None:
    plan = plan_gate1_pilots(
        {"pick_cube": protocol()},
        seed=20260718,
        splits=("seen", "unseen_composition", "long_lag"),
    )

    assert plan.jobs
    assert {job.protocol for job in plan.jobs} == {protocol()}
    registry = method_registry()
    for job in plan.jobs:
        assert job.privileges == registry[job.method].privileges
    assert registry["oracle"].privileges == frozenset({"true_contract"})
    assert registry["rma"].privileges == frozenset({"teacher_contract_during_training"})
    assert registry["dualabi"].privileges == frozenset({"bounded_active_probe"})


def test_pilots_only_include_passing_tasks_and_record_external_exclusions() -> None:
    plan = plan_gate1_pilots({"push_cube": protocol()}, seed=20260718, splits=("seen",))

    assert {job.task for job in plan.jobs} == {"push_cube"}
    assert "transformer_icl" in plan.exclusions
    assert "SPACE_GLAM" in plan.exclusions["transformer_icl"]
    assert "dualabi" in plan.exclusions
    assert "trained" in plan.exclusions["dualabi"]
    assert {job.method for job in plan.jobs} == {"oracle", "no_adapt"}


def test_promoted_jobs_retain_first_seed_and_add_two_matched_seeds() -> None:
    pilot = plan_gate1_pilots({"pick_cube": protocol()}, seed=20260718, splits=("long_lag",))
    promoted = promote_seeds(pilot.jobs, (20260719, 20260720))

    assert {job.seed for job in promoted} == {20260718, 20260719, 20260720}
    assert len(promoted) == len(pilot.jobs) * 3


def test_one_seed_can_prune_but_cannot_claim_superiority() -> None:
    assert not superiority_eligible({20260718})
    assert superiority_eligible({20260718, 20260719, 20260720})
    with pytest.raises(ValueError, match="three matched seeds"):
        superiority_eligible({20260718}, require=True)
