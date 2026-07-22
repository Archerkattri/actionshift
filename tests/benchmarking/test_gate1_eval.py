from __future__ import annotations

import pytest

import actionshift.benchmarking.gate1_eval as gate1_eval
from actionshift.benchmarking.gate1_eval import (
    condition_for_method,
    evaluate_gate1_job,
    plan_gate1_slice_jobs,
    representative_contracts,
)
from actionshift.benchmarking.ppo_parity import ParityEpisode
from actionshift.contracts.splits import composition_signature


def test_gate1_methods_map_to_privileged_and_hidden_conditions() -> None:
    assert condition_for_method("oracle") == "oracle_nonidentity"
    assert condition_for_method("no_adapt") == "noadapt_nonidentity"
    with pytest.raises(ValueError, match="not end-to-end runnable"):
        condition_for_method("dualabi")


def test_representative_contracts_exercise_declared_split_stressors() -> None:
    seen = representative_contracts("seen")
    composition = representative_contracts("unseen_composition")
    lag = representative_contracts("long_lag")

    assert all(contract.lag == 0 and contract.target == "delta" for contract in seen)
    assert all(len(composition_signature(contract)) >= 3 for contract in composition)
    assert {contract.lag for contract in lag} == {2, 4}
    assert all(
        len(contract.permutation) == 6 for group in (seen, composition, lag) for contract in group
    )


def test_experimental_seed_is_not_confused_with_per_contract_environment_seed(
    monkeypatch, tmp_path
) -> None:
    def fake_evaluate(checkpoint, *, task, condition, seed, episodes, num_envs, contract):
        return [ParityEpisode(task, condition, 0, seed, True, 1.0, 10, "a" * 64)]

    monkeypatch.setattr(gate1_eval, "evaluate_ppo_checkpoint", fake_evaluate)

    records = evaluate_gate1_job(
        tmp_path / "checkpoint.pt",
        task="pick_cube",
        method="oracle",
        split="seen",
        seed=20260718,
        episodes_per_contract=1,
        num_envs=1,
    )

    assert {record["seed"] for record in records} == {20260718}
    assert {record["environment_seed"] for record in records} == {20260718, 20260719}


def test_slice_plan_is_complete_and_hash_addressed(tmp_path) -> None:
    pick = tmp_path / "pick.pt"
    push = tmp_path / "push.pt"
    pick.write_bytes(b"pick")
    push.write_bytes(b"push")

    jobs = plan_gate1_slice_jobs(
        {"pick_cube": pick, "push_cube": push},
        seeds=(20260718, 20260719, 20260720),
        output_directory=tmp_path / "out",
    )

    assert len(jobs) == 36
    assert len({job["job_id"] for job in jobs}) == 36
    assert {job["checkpoint_sha256"] for job in jobs} == {
        gate1_eval.sha256_file(pick),
        gate1_eval.sha256_file(push),
    }
