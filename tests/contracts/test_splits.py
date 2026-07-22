from __future__ import annotations

from dataclasses import replace

import pytest

from actionshift.contracts.splits import generate_split, validate_manifest


@pytest.mark.parametrize(
    "rule", ["seen", "unseen_value", "unseen_composition", "long_lag", "task_transfer"]
)
def test_split_generation_is_reproducible_disjoint_and_hashed(rule: str) -> None:
    first = generate_split(rule, seed=20260718)
    second = generate_split(rule, seed=20260718)

    assert first.to_json() == second.to_json()
    validation = validate_manifest(first)
    assert validation["overlap_count"] == 0
    assert validation["manifest_hash_valid"]
    assert validation["ordering_valid"]
    assert first.train
    assert first.test


def test_unseen_composition_has_disjoint_field_composition_signatures() -> None:
    manifest = generate_split("unseen_composition", seed=20260718)
    validation = validate_manifest(manifest)

    assert validation["composition_overlap_count"] == 0
    assert validation["training_field_support_valid"]


def test_validator_rejects_contract_overlap_and_tampered_hash() -> None:
    manifest = generate_split("seen", seed=20260718)
    overlapping = replace(manifest, test=(manifest.train[0], *manifest.test))

    validation = validate_manifest(overlapping)

    assert validation["overlap_count"] == 1
    assert not validation["manifest_hash_valid"]
