from __future__ import annotations

import torch

from actionshift.belief.exact import ExactContractBelief
from actionshift.belief.likelihood import gaussian_transition_log_likelihood
from actionshift.contracts.types import ActionContract


def contract(sign: int) -> ActionContract:
    return ActionContract(
        permutation=(0,),
        sign=(sign,),
        scale=(1.0,),
        target="delta",
        frame="base",
        lag=0,
        gripper_inverted=False,
    )


def test_uniform_belief_updates_to_manual_bayes_posterior() -> None:
    belief = ExactContractBelief.uniform((contract(1), contract(-1)), dtype=torch.float64)

    updated = belief.update(torch.log(torch.tensor([0.9, 0.1], dtype=torch.float64)))

    torch.testing.assert_close(updated.probabilities, torch.tensor([0.9, 0.1], dtype=torch.float64))
    torch.testing.assert_close(belief.probabilities, torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_update_remains_normalized_after_many_observations() -> None:
    belief = ExactContractBelief.uniform((contract(1), contract(-1)), dtype=torch.float64)

    for _ in range(100):
        belief = belief.update(torch.tensor([-0.01, -4.0], dtype=torch.float64))

    torch.testing.assert_close(belief.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isfinite(belief.log_probabilities).all()


def test_gaussian_transition_likelihood_prefers_matching_prediction() -> None:
    predictions = torch.tensor([[1.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
    observation = torch.tensor([0.9, -1.1], dtype=torch.float64)

    likelihood = gaussian_transition_log_likelihood(predictions, observation, sigma=0.1)

    assert likelihood[0] > likelihood[1]
    torch.testing.assert_close(likelihood[0], torch.tensor(-1.0, dtype=torch.float64))


def test_likelihood_rejects_nonpositive_noise() -> None:
    predictions = torch.zeros(2, 1)
    observation = torch.zeros(1)

    try:
        gaussian_transition_log_likelihood(predictions, observation, sigma=0.0)
    except ValueError as error:
        assert str(error) == "sigma must be finite and positive"
    else:
        raise AssertionError("nonpositive sigma was accepted")
