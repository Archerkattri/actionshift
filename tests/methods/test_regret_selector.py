from __future__ import annotations

import torch

from actionshift.methods.dualabi import (
    expected_task_regret,
    select_regret_aware_action,
)


def informative_and_uninformative_outcomes() -> torch.Tensor:
    # candidate by contract by observation
    return torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=torch.float64,
    )


def test_expected_regret_matches_manual_two_contract_example() -> None:
    prior = torch.tensor([0.5, 0.5], dtype=torch.float64)
    future_utility = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    regret = expected_task_regret(
        prior, informative_and_uninformative_outcomes(), future_utility
    )

    torch.testing.assert_close(regret, torch.tensor([0.0, 0.5], dtype=torch.float64))


def test_selector_trades_immediate_progress_for_task_relevant_information() -> None:
    prior = torch.tensor([0.5, 0.5], dtype=torch.float64)
    regret = expected_task_regret(
        prior,
        informative_and_uninformative_outcomes(),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )

    decision = select_regret_aware_action(
        task_progress=torch.tensor([0.0, 0.2], dtype=torch.float64),
        safety_risk=torch.zeros(2, dtype=torch.float64),
        expected_future_regret=regret,
        safety_weight=1.0,
        regret_weight=1.0,
    )

    assert decision.index == 0
    torch.testing.assert_close(decision.score, torch.tensor([0.0, -0.3], dtype=torch.float64))


def test_selector_ignores_information_when_uncertainty_is_task_irrelevant() -> None:
    prior = torch.tensor([0.5, 0.5], dtype=torch.float64)
    regret = expected_task_regret(
        prior,
        informative_and_uninformative_outcomes(),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
    )

    decision = select_regret_aware_action(
        task_progress=torch.tensor([0.0, 0.2], dtype=torch.float64),
        safety_risk=torch.zeros(2, dtype=torch.float64),
        expected_future_regret=regret,
        safety_weight=1.0,
        regret_weight=1.0,
    )

    torch.testing.assert_close(regret, torch.zeros(2, dtype=torch.float64))
    assert decision.index == 1


def test_selector_can_reject_an_informative_but_unsafe_probe() -> None:
    decision = select_regret_aware_action(
        task_progress=torch.tensor([0.0, 0.0]),
        safety_risk=torch.tensor([1.0, 0.0]),
        expected_future_regret=torch.tensor([0.0, 0.5]),
        safety_weight=2.0,
        regret_weight=1.0,
    )

    assert decision.index == 1
