from __future__ import annotations

import json
from pathlib import Path

import pytest

from actionshift.evaluation.challenge import (
    CHALLENGE_CASES,
    ChallengeDecision,
    ChallengeObservation,
    ChallengeTransition,
    MethodBudget,
    build_challenge_manifest,
    challenge_method_specs,
    load_adapter_factory,
    run_challenge,
    run_reference_smoke,
    verify_challenge_manifest,
    write_challenge_manifest,
)
from actionshift.evaluation.matrix import METHODS


class _SpyAdapter:
    def __init__(self, *, probe: bool = False) -> None:
        self.probe = probe
        self.manifests: list[dict[str, object]] = []
        self.observations: list[ChallengeObservation] = []
        self.transitions: list[ChallengeTransition] = []

    def reset(self, *, manifest: dict[str, object], budget: MethodBudget) -> None:
        del budget
        self.manifests.append(dict(manifest))

    def act(self, observation: ChallengeObservation) -> ChallengeDecision:
        self.observations.append(observation)
        return ChallengeDecision(observation.canonical_action, self.probe)

    def observe(self, transition: ChallengeTransition) -> None:
        self.transitions.append(transition)


def test_frozen_manifest_is_complete_deterministic_and_self_verifying() -> None:
    root = Path(__file__).resolve().parents[2]
    first = build_challenge_manifest(root, seeds=(7,), steps=32)
    second = build_challenge_manifest(root, seeds=(7,), steps=32)
    assert first == second
    verify_challenge_manifest(first)
    assert len(first["episodes"]) == len(CHALLENGE_CASES)
    assert all(len(source["sha256"]) == 64 for source in first["sources"])
    serialized = json.dumps(first["episodes"], sort_keys=True)
    assert "contract_sha256" not in serialized
    assert "permutation" not in serialized
    damaged = dict(first)
    damaged["status"] = "CHANGED"
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_challenge_manifest(damaged)


def test_manifest_writer_is_idempotent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "challenge.json"
    first = write_challenge_manifest(destination, root)
    second = write_challenge_manifest(destination, root)
    assert first == second


def test_method_envelopes_match_headline_matrix_and_share_episode_budget() -> None:
    specs = challenge_method_specs()
    assert tuple(spec.name for spec in specs) == METHODS
    probing_budgets = {
        spec.max_probe_steps
        for spec in specs
        if "probes" in spec.name or "dualabi" in spec.name
    }
    assert probing_budgets == {6}
    assert next(spec for spec in specs if spec.name == "oracle").privileges == ("true_contract",)


def test_runner_covers_every_shift_and_keeps_contracts_out_of_adapter_abi() -> None:
    adapters: list[_SpyAdapter] = []

    def factory(*, seed: int, budget: MethodBudget) -> _SpyAdapter:
        del seed, budget
        adapter = _SpyAdapter()
        adapters.append(adapter)
        return adapter

    report = run_challenge(method="external", factory=factory, seeds=(5,), steps=32)
    assert report["episode_count"] == len(CHALLENGE_CASES)
    observed_cases = {
        (record["shift_kind"], record["outside_grammar"])
        for record in report["episodes"]
    }
    assert observed_cases == set(CHALLENGE_CASES)
    assert len(adapters) == len(CHALLENGE_CASES)
    for adapter in adapters:
        exposed = json.dumps(adapter.manifests, sort_keys=True)
        assert "contract" not in exposed
        assert len(adapter.observations) == 32
        assert len(adapter.transitions) == 32
        assert not hasattr(adapter.observations[0], "contract")
        assert not hasattr(adapter.transitions[0], "contract")


def test_oracle_is_a_labeled_ceiling_and_reference_smoke_is_cpu_only(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    report = run_reference_smoke(output, seeds=(11,), steps=32)
    oracle, no_adapt = report["methods"]
    assert oracle["privileges"] == ["true_contract"]
    assert oracle["successes"] == len(CHALLENGE_CASES)
    assert oracle["mean_post_lag_mse"] < no_adapt["mean_post_lag_mse"]
    assert report["runtime"]["device"] == "cpu"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_probe_and_amplitude_budgets_are_enforced() -> None:
    def factory(*, seed: int, budget: MethodBudget) -> _SpyAdapter:
        del seed, budget
        return _SpyAdapter(probe=True)

    with pytest.raises(ValueError, match="probe-step budget"):
        run_challenge(method="external", factory=factory, seeds=(3,), steps=16)

    class Oversized(_SpyAdapter):
        def act(self, observation: ChallengeObservation) -> ChallengeDecision:
            return ChallengeDecision((2.0,) * len(observation.canonical_action))

    with pytest.raises(ValueError, match="declared bound"):
        run_challenge(
            method="external",
            factory=lambda **_: Oversized(),
            seeds=(3,),
            steps=16,
        )


def test_external_factory_loads_by_module_reference() -> None:
    factory = load_adapter_factory(
        "actionshift.evaluation.challenge:identity_adapter_factory"
    )
    report = run_challenge(method="external", factory=factory, seeds=(2,), steps=16)
    assert report["episode_count"] == len(CHALLENGE_CASES)
