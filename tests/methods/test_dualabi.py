from __future__ import annotations

import torch

from actionshift.methods.dualabi import CandidateEvaluator, select_dualabi_candidates
from actionshift.policies.conditioning import ContractConditionedActor


def test_delta_posterior_matches_oracle_conditioned_actor_path() -> None:
    torch.manual_seed(11)
    actor = ContractConditionedActor(
        observation_dim=4, action_dim=3, posterior_dim=8, hidden_dim=16
    )
    observation = torch.randn(5, 4)
    true_contract = torch.tensor([2, 4, 1, 7, 0])
    delta_posterior = torch.nn.functional.one_hot(true_contract, num_classes=8).float()

    posterior_action = actor(observation, delta_posterior)
    oracle_action = actor.oracle_action(observation, true_contract)

    torch.testing.assert_close(posterior_action, oracle_action)


def test_conditioned_actor_proposes_bounded_safe_perturbations() -> None:
    actor = ContractConditionedActor(
        observation_dim=4, action_dim=7, posterior_dim=8, hidden_dim=16
    )
    proposal = actor.propose_candidates(
        torch.zeros(2, 4),
        torch.full((2, 8), 1 / 8),
        candidate_count=5,
        radius=0.1,
        lower=-0.25,
        upper=0.25,
        pose_dimensions=6,
    )

    assert proposal.shape == (2, 5, 7)
    assert torch.all(proposal <= 0.25)
    assert torch.all(proposal >= -0.25)
    torch.testing.assert_close(
        proposal[:, :, -1], proposal[:, :1, -1].expand_as(proposal[:, :, -1])
    )


def test_dualabi_logs_formula_terms_and_masks_unsafe_candidate() -> None:
    decision = select_dualabi_candidates(
        task_value=torch.tensor([0.3, 0.1, 0.5]),
        safety_risk=torch.tensor([0.0, 0.0, 0.0]),
        current_task_regret=torch.tensor(0.6),
        expected_future_regret=torch.tensor([0.5, 0.1, 0.0]),
        valid_mask=torch.tensor([True, True, False]),
        safety_weight=2.0,
        information_weight=1.0,
        information_mode="task_regret",
    )

    assert decision.index == 1
    torch.testing.assert_close(decision.regret_reduction, torch.tensor([0.1, 0.5, 0.6]))
    assert torch.isneginf(decision.score[2])
    assert set(decision.logged_terms) == {
        "task_value",
        "safety_penalty",
        "terminal_penalty",
        "information_bonus",
        "valid",
    }


def test_entropy_ablation_can_probe_task_irrelevant_uncertainty() -> None:
    task_regret = select_dualabi_candidates(
        task_value=torch.tensor([0.2, 0.0]),
        safety_risk=torch.zeros(2),
        current_task_regret=torch.tensor(0.0),
        expected_future_regret=torch.zeros(2),
        entropy_reduction=torch.tensor([0.0, 1.0]),
        valid_mask=torch.ones(2, dtype=torch.bool),
        safety_weight=1.0,
        information_weight=1.0,
        information_mode="task_regret",
    )
    entropy = select_dualabi_candidates(
        task_value=torch.tensor([0.2, 0.0]),
        safety_risk=torch.zeros(2),
        current_task_regret=torch.tensor(0.0),
        expected_future_regret=torch.zeros(2),
        entropy_reduction=torch.tensor([0.0, 1.0]),
        valid_mask=torch.ones(2, dtype=torch.bool),
        safety_weight=1.0,
        information_weight=1.0,
        information_mode="entropy",
    )

    assert task_regret.index == 0
    assert entropy.index == 1


def test_candidate_evaluator_ensembles_dynamics_and_critic() -> None:
    torch.manual_seed(3)
    evaluator = CandidateEvaluator(
        state_dim=4,
        action_dim=7,
        posterior_dim=8,
        ensemble_size=3,
        hidden_dim=16,
    )
    estimates = evaluator(
        torch.zeros(2, 4),
        torch.zeros(2, 5, 7),
        torch.full((2, 8), 1 / 8),
    )
    assert estimates.task_value.shape == (2, 5)
    assert estimates.terminal_risk.shape == (2, 5)
    assert estimates.predicted_displacement.shape == (2, 5, 3)
    assert estimates.collision_risk.shape == (2, 5)
    assert estimates.dynamics_disagreement.shape == (2, 5)
    assert torch.all((estimates.collision_risk >= 0) & (estimates.collision_risk <= 1))
    assert torch.all((estimates.terminal_risk >= 0) & (estimates.terminal_risk <= 1))
    assert torch.all(estimates.dynamics_disagreement >= 0)


def test_terminal_controllability_penalty_rejects_informative_dead_end() -> None:
    decision = select_dualabi_candidates(
        task_value=torch.tensor([0.2, 0.1]),
        safety_risk=torch.zeros(2),
        current_task_regret=torch.tensor(1.0),
        expected_future_regret=torch.tensor([0.0, 0.8]),
        valid_mask=torch.ones(2, dtype=torch.bool),
        safety_weight=1.0,
        information_weight=1.0,
        information_mode="task_regret",
        terminal_risk=torch.tensor([1.0, 0.0]),
        terminal_weight=2.0,
    )
    assert decision.index == 1
    torch.testing.assert_close(
        decision.logged_terms["terminal_penalty"], torch.tensor([2.0, 0.0])
    )
