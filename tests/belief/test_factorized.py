from __future__ import annotations

import pytest
import torch

from actionshift.belief.factorized import (
    FactorizedContractBelief,
    calibration_error,
    exact_factor_marginals,
)
from actionshift.evaluation.falsification import build_core_contracts

CARDINALITIES = {
    "permutation": 2,
    "sign": 4,
    "scale": 2,
    "target": 2,
    "frame": 2,
    "lag": 2,
    "gripper": 2,
}


def test_factorized_marginals_are_normalized_and_support_sampling_and_map() -> None:
    model = FactorizedContractBelief(transition_dim=5, hidden_dim=12, cardinalities=CARDINALITIES)
    history = torch.randn(4, 6, 5)

    marginals = model(history)
    sampled = model.sample(marginals, samples=3)
    mapped = model.map_composition(marginals)

    assert set(marginals) == set(CARDINALITIES)
    for field, probabilities in marginals.items():
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))
        assert sampled[field].shape == (4, 3)
        assert mapped[field].shape == (4,)


def test_step_reset_matches_fresh_hidden_state_for_selected_environment() -> None:
    torch.manual_seed(3)
    model = FactorizedContractBelief(transition_dim=5, hidden_dim=12, cardinalities=CARDINALITIES)
    fresh = FactorizedContractBelief(transition_dim=5, hidden_dim=12, cardinalities=CARDINALITIES)
    fresh.load_state_dict(model.state_dict())
    model.step(torch.randn(2, 5), reset_mask=torch.tensor([False, False]))
    transition = torch.randn(2, 5)

    reset_output = model.step(transition, reset_mask=torch.tensor([True, False]))
    fresh_output = fresh.step(transition[:1], reset_mask=torch.tensor([True]))

    for field in CARDINALITIES:
        torch.testing.assert_close(reset_output[field][0], fresh_output[field][0])


def test_exact_finite_belief_projects_to_factor_marginals() -> None:
    contracts = build_core_contracts()
    probabilities = torch.zeros(len(contracts), dtype=torch.float64)
    probabilities[7] = 1.0

    marginals = exact_factor_marginals(contracts, probabilities)

    assert all(torch.count_nonzero(value).item() == 1 for value in marginals.values())
    assert all(value.sum().item() == 1.0 for value in marginals.values())


def test_calibration_error_matches_hand_computable_binary_case() -> None:
    probabilities = torch.tensor([0.9, 0.8, 0.2, 0.1])
    correct = torch.tensor([True, True, True, False])

    result = calibration_error(probabilities, correct, bins=2)

    assert result["brier"] == pytest.approx(0.175)
    assert 0.0 <= result["ece"] <= 1.0
