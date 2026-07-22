from __future__ import annotations

import torch

from actionshift.contracts.transforms import ActionLag, decode_pose
from actionshift.contracts.types import ActionContract


def test_minimal_decode_order_ends_with_lag() -> None:
    contract = ActionContract(
        permutation=(1, 0),
        sign=(1, -1),
        scale=(0.5, 2.0),
        target="delta",
        frame="base",
        lag=1,
        gripper_inverted=False,
    )
    lag = ActionLag(steps=contract.lag)

    first = lag.step(decode_pose(torch.tensor([[2.0, 3.0]]), contract))
    second = lag.step(decode_pose(torch.tensor([[4.0, 5.0]]), contract))

    torch.testing.assert_close(first, torch.tensor([[0.0, 0.0]]))
    torch.testing.assert_close(second, torch.tensor([[1.5, -4.0]]))
