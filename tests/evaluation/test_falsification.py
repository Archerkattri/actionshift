from __future__ import annotations

from actionshift.evaluation.falsification import build_core_contracts, run_gate


def test_core_gate_uses_32_distinct_contracts_and_all_baselines() -> None:
    contracts = build_core_contracts()
    assert len(contracts) == 32
    assert len({contract.to_json() for contract in contracts}) == 32

    report = run_gate(seeds=(0, 1, 2, 3, 4), episodes_per_seed=8)

    assert set(report["methods"]) == {
        "oracle",
        "no_adaptation",
        "recurrent_domain_randomization",
        "random_probes",
        "fixed_pulses",
        "exact_regret_aware",
    }
    assert report["contract_count"] == 32
    assert report["gate"]["exact_better_than_no_adaptation"]
    assert report["gate"]["tradeoff_nontrivial"]
    assert report["decision"] == "continue"
