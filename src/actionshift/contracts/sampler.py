"""Deterministic finite contract products used by split generation."""

from __future__ import annotations

import itertools
from typing import cast

from actionshift.contracts.types import ActionContract, Frame, Target


def contract_product() -> tuple[ActionContract, ...]:
    contracts = (
        ActionContract(
            permutation=permutation,
            sign=sign,
            scale=(scale, scale),
            target=cast(Target, target),
            frame=cast(Frame, frame),
            lag=lag,
            gripper_inverted=gripper,
        )
        for permutation, sign, scale, target, frame, lag, gripper in itertools.product(
            ((0, 1), (1, 0)),
            itertools.product((-1, 1), repeat=2),
            (0.5, 1.0, 1.5),
            ("delta", "absolute"),
            ("base", "tool"),
            (0, 1, 2, 4),
            (False, True),
        )
    )
    return tuple(sorted(contracts, key=ActionContract.to_json))


def core_composition_product() -> tuple[ActionContract, ...]:
    contracts = (
        ActionContract(
            permutation=permutation,
            sign=sign,
            scale=(scale, scale),
            target="delta",
            frame="base",
            lag=lag,
            gripper_inverted=False,
        )
        for permutation, sign, scale, lag in itertools.product(
            ((0, 1), (1, 0)),
            itertools.product((-1, 1), repeat=2),
            (1.0, 1.5),
            (0, 2),
        )
    )
    return tuple(sorted(contracts, key=ActionContract.to_json))
