from __future__ import annotations

from actionshift.evaluation.matrix import DEFAULT_SEEDS, build_matrix


def test_frozen_matrix_is_complete_unique_and_uses_five_seeds() -> None:
    matrix = build_matrix()
    expected = 3 * 5 * 10 * 5
    assert len(matrix) == expected
    assert len({job.job_id for job in matrix}) == expected
    assert {job.seed for job in matrix} == set(DEFAULT_SEEDS)
    assert {job.task for job in matrix} == {
        "pick_cube",
        "push_cube",
        "peg_insertion_side",
    }
    assert {job.information_mode for job in matrix if job.method == "dualabi"} == {
        "task_regret"
    }


def test_matrix_serialization_preserves_provenance_fields() -> None:
    job = build_matrix()[0]
    record = job.to_dict()
    assert record["schema_version"] == "1.0"
    assert record["backend"] == "maniskill"
    assert record["job_id"] == job.job_id
