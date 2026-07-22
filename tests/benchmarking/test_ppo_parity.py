from __future__ import annotations

import torch

from actionshift.benchmarking.ppo_parity import oracle_contract, policy_action_for_condition
from actionshift.contracts.transforms import CompleteActionDecoder


def test_oracle_condition_encodes_to_exact_canonical_action() -> None:
    canonical = torch.tensor(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
            [-0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1],
        ]
    )
    raw = policy_action_for_condition(canonical, "oracle_nonidentity")
    decoder = CompleteActionDecoder(oracle_contract(), batch_size=2)
    decoded = decoder.step(raw, ee_rotation=torch.eye(3).expand(2, 3, 3))

    torch.testing.assert_close(decoded, canonical)
    assert not torch.equal(raw, canonical)


def test_unwrapped_and_identity_conditions_do_not_modify_policy_action() -> None:
    canonical = torch.randn(4, 7)

    assert policy_action_for_condition(canonical, "unwrapped") is canonical
    assert policy_action_for_condition(canonical, "identity") is canonical


def test_noadapt_nonidentity_does_not_receive_oracle_encoding() -> None:
    canonical = torch.randn(4, 7)

    assert policy_action_for_condition(canonical, "noadapt_nonidentity") is canonical
