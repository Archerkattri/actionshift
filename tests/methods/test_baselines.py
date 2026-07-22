from __future__ import annotations

import torch

from actionshift.methods.no_adapt import DomainRandomizedContractSampler, NoAdaptMethod
from actionshift.methods.oracle import OracleMethod
from actionshift.methods.osi import OSIAdapter, RMATeacherStudent
from actionshift.methods.recurrent import RecurrentMethod
from actionshift.policies.actor_critic import ActorCritic
from actionshift.policies.conditioning import TransitionHistory


def test_oracle_uses_contract_features_but_no_adapt_rejects_them() -> None:
    actor = ActorCritic(observation_dim=4, action_dim=2, context_dim=3, hidden_dim=16)
    oracle = OracleMethod(actor)
    observation = torch.zeros(5, 4)
    context = torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])]

    action, value = oracle.act(observation, context, deterministic=True)

    assert action.shape == (5, 2)
    assert value.shape == (5,)
    no_adapt = NoAdaptMethod(ActorCritic(4, 2, context_dim=0, hidden_dim=16))
    no_action, _ = no_adapt.act(observation, deterministic=True)
    assert no_action.shape == (5, 2)


def test_domain_randomization_samples_only_training_contracts_per_reset() -> None:
    sampler = DomainRandomizedContractSampler(train_contract_ids=(2, 4, 8), seed=7)
    first = sampler.sample(torch.tensor([True, True, True, True]))
    second = sampler.sample(torch.tensor([False, True, False, True]), current=first)

    assert set(second.tolist()) <= {2, 4, 8}
    assert second[0] == first[0]
    assert second[2] == first[2]


def test_recurrent_done_mask_erases_only_finished_history() -> None:
    method = RecurrentMethod(observation_dim=4, action_dim=2, hidden_dim=8)
    observation = torch.randn(2, 4)
    previous_action = torch.randn(2, 2)
    reward = torch.randn(2)
    method.act(observation, previous_action, reward, torch.zeros(2, dtype=torch.bool))
    prior = method.hidden_state.clone()

    method.act(observation, previous_action, reward, torch.tensor([True, False]))

    assert not torch.equal(method.hidden_state[1], torch.zeros_like(method.hidden_state[1]))
    assert not torch.equal(method.hidden_state[1], prior[1])


def test_osi_and_rma_use_transition_history_without_labels_at_execution() -> None:
    adapter = OSIAdapter(transition_dim=9, history_length=4, latent_dim=5, hidden_dim=16)
    history = torch.randn(3, 4, 9)
    latent = adapter(history)
    assert latent.shape == (3, 5)

    rma = RMATeacherStudent(privileged_dim=7, transition_dim=9, history_length=4, latent_dim=5)
    privileged = torch.randn(3, 7)
    teacher_latent = rma.teacher(privileged)
    student_latent = rma.student(history)
    loss = rma.adaptation_loss(history, privileged)

    assert teacher_latent.shape == student_latent.shape == (3, 5)
    assert torch.isfinite(loss)


def test_transition_history_done_mask_removes_prior_episode_data() -> None:
    history = TransitionHistory(batch_size=2, history_length=3, transition_dim=2)
    history.append(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    history.append(
        torch.tensor([[3.0, 3.0], [4.0, 4.0]]), reset_mask=torch.tensor([True, False])
    )

    torch.testing.assert_close(history.tensor[0, -2], torch.zeros(2))
    torch.testing.assert_close(history.tensor[0, -1], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(history.tensor[1, -2], torch.tensor([2.0, 2.0]))
