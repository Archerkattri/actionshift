from __future__ import annotations

import json

import pytest

from actionshift.contracts.compound import generate_compound_episode
from actionshift.contracts.sampler import core_composition_product


@pytest.mark.parametrize("kind", ["static", "abrupt", "gradual", "recurring"])
def test_compound_episode_is_deterministic_and_covers_all_steps(kind: str) -> None:
    first = generate_compound_episode(kind, seed=7, steps=32)
    second = generate_compound_episode(kind, seed=7, steps=32)
    assert first.private_manifest() == second.private_manifest()
    assert first.segments[0].start == 0
    assert first.segments[-1].stop == 32
    assert sum(segment.stop - segment.start for segment in first.segments) == 32
    assert json.dumps(first.public_manifest(), sort_keys=True) == json.dumps(
        second.public_manifest(), sort_keys=True
    )


def test_public_manifest_never_exposes_hidden_contract_fields() -> None:
    episode = generate_compound_episode("abrupt", seed=11, steps=24)
    public = json.dumps(episode.public_manifest(), sort_keys=True)
    assert "permutation" not in public
    assert "scale" not in public
    assert "contract_sha256" not in public
    assert episode.contract_at(0) != episode.contract_at(episode.steps // 2)


def test_outside_grammar_case_is_explicit_and_not_in_core_pool() -> None:
    episode = generate_compound_episode("abrupt", seed=3, steps=24, outside_grammar=True)
    assert episode.outside_grammar
    assert episode.contract_at(episode.steps - 1) not in core_composition_product()
    assert len(episode.private_manifest()["contract_sha256"]) == 2


def test_invalid_schedule_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        generate_compound_episode("unknown", seed=1)
    with pytest.raises(ValueError):
        generate_compound_episode("static", seed=1, steps=7)

