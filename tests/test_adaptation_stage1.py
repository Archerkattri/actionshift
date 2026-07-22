"""Stage 1 adaptation tests: exact belief must earn oracle parity honestly."""

from __future__ import annotations

import inspect
import unittest

import torch

from actionshift.adaptation.adapters import (
    ExactBeliefAdapter,
    NoAdaptAdapter,
    OracleAdapter,
)
from actionshift.adaptation.hypotheses import HypothesisSimulator, identity_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract

_BATCH = 3


def _contract(
    permutation: tuple[int, ...],
    sign: tuple[int, ...],
    scale: tuple[float, ...],
    target: str = "delta",
    frame: str = "base",
    lag: int = 0,
    gripper_inverted: bool = False,
) -> ActionContract:
    return ActionContract(
        permutation=permutation,
        sign=sign,
        scale=scale,
        target=target,  # type: ignore[arg-type]
        frame=frame,  # type: ignore[arg-type]
        lag=lag,
        gripper_inverted=gripper_inverted,
    )


def _true_contract() -> ActionContract:
    return _contract(
        (1, 0, 2, 4, 5, 3),
        (-1, 1, -1, 1, -1, 1),
        (0.5, 2.0, 1.5, 0.75, 1.25, 0.6),
        gripper_inverted=True,
    )


def _pool(include_true: bool) -> tuple[ActionContract, ...]:
    distractors = (
        _contract((0, 1, 2, 3, 4, 5), (1,) * 6, (1.0,) * 6),
        _contract((5, 1, 2, 3, 4, 0), (1, -1, 1, -1, 1, 1), (1.5, 0.75, 1.0, 1.25, 0.5, 2.0)),
        _contract(
            (5, 4, 3, 2, 1, 0),
            (1, -1, -1, 1, 1, -1),
            (1.5, 1.5, 0.5, 2.0, 0.75, 1.25),
            target="absolute",
            gripper_inverted=True,
        ),
        _contract(
            (2, 0, 1, 5, 3, 4),
            (1, 1, -1, -1, 1, 1),
            (2.0, 0.5, 0.75, 1.5, 1.0, 1.25),
            frame="tool",
        ),
        _contract((0, 1, 2, 3, 4, 5), (1,) * 6, (1.0,) * 6, lag=2),
        _contract((3, 1, 4, 0, 5, 2), (-1, -1, 1, 1, -1, 1), (0.6, 1.25, 0.75, 1.5, 2.0, 0.5)),
        _contract((4, 2, 0, 1, 5, 3), (1, -1, 1, 1, -1, -1), (1.25, 0.6, 2.0, 0.5, 1.5, 0.75)),
    )
    return (*distractors, _true_contract()) if include_true else distractors


class SyntheticHiddenEnvironment:
    """Bit-faithful stand-in for the hidden wrapper: decode-then-lag, identity rotation."""

    def __init__(self, contract: ActionContract, batch_size: int) -> None:
        self._decoder = CompleteActionDecoder(contract, batch_size=batch_size)
        self._batch_size = batch_size

    def step(
        self, raw_action: torch.Tensor, reset_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        rotation = identity_rotation(self._batch_size, raw_action.device, raw_action.dtype)
        return self._decoder.step(raw_action, ee_rotation=rotation, reset_mask=reset_mask)


def _run_adaptation(
    adapter: ExactBeliefAdapter, steps: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive the synthetic loop; return last-step intent and executed canonical."""
    environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
    intent = executed = torch.zeros((_BATCH, 7))
    for _ in range(steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6])
    return intent, executed


class ExactBeliefConvergenceTest(unittest.TestCase):
    def test_identifies_true_contract_and_reaches_oracle_parity(self) -> None:
        generator = torch.Generator().manual_seed(20260720)
        adapter = ExactBeliefAdapter(
            _pool(include_true=True),
            batch_size=_BATCH,
            response=ResponseModel(alpha=1.0, sigma=0.05),
        )
        intent, executed = _run_adaptation(adapter, steps=40, generator=generator)
        true_index = len(_pool(include_true=True)) - 1
        self.assertTrue(
            bool((adapter.driver.map_indices() == true_index).all()),
            "belief must concentrate on the true contract",
        )
        torch.testing.assert_close(executed, intent, atol=1e-5, rtol=1e-4)

    def test_pool_without_true_contract_fails_parity(self) -> None:
        generator = torch.Generator().manual_seed(20260720)
        adapter = ExactBeliefAdapter(
            _pool(include_true=False),
            batch_size=_BATCH,
            response=ResponseModel(alpha=1.0, sigma=0.05),
        )
        intent, executed = _run_adaptation(adapter, steps=40, generator=generator)
        error = (executed - intent).abs().max()
        self.assertGreater(
            float(error), 0.1, "a pool without the truth must not silently reach parity"
        )


class WrapperReplicaParityTest(unittest.TestCase):
    def test_replica_matches_wrapper_for_stateful_contract(self) -> None:
        contract = _contract(
            (5, 4, 3, 2, 1, 0),
            (1, -1, -1, 1, 1, -1),
            (1.5, 1.5, 0.5, 2.0, 0.75, 1.25),
            target="absolute",
            frame="tool",
            lag=2,
            gripper_inverted=True,
        )
        generator = torch.Generator().manual_seed(7)
        environment = SyntheticHiddenEnvironment(contract, _BATCH)
        simulator = HypothesisSimulator((contract,), batch_size=_BATCH)
        for step in range(12):
            raw = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            reset = None
            if step == 6:
                reset = torch.tensor([True, False, False])
            executed = environment.step(raw, reset_mask=reset)
            predicted = simulator.step(raw, reset_mask=reset)[0]
            torch.testing.assert_close(predicted, executed)


class OracleAdapterTest(unittest.TestCase):
    def test_absolute_target_roundtrip(self) -> None:
        contract = _contract(
            (2, 0, 1, 5, 3, 4),
            (1, 1, -1, -1, 1, 1),
            (2.0, 0.5, 0.75, 1.5, 1.0, 1.25),
            target="absolute",
            gripper_inverted=True,
        )
        generator = torch.Generator().manual_seed(11)
        adapter = OracleAdapter(contract, batch_size=_BATCH)
        environment = SyntheticHiddenEnvironment(contract, _BATCH)
        for _ in range(8):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            executed = environment.step(adapter.encode(intent))
            torch.testing.assert_close(executed, intent, atol=1e-5, rtol=1e-4)


class LeakageGuardTest(unittest.TestCase):
    def test_belief_adapter_cannot_receive_the_true_contract(self) -> None:
        parameters = inspect.signature(ExactBeliefAdapter.__init__).parameters
        self.assertNotIn("true_contract", parameters)
        self.assertNotIn("contract", parameters)

    def test_no_adapt_is_identity(self) -> None:
        action = torch.rand((_BATCH, 7))
        self.assertIs(NoAdaptAdapter().encode(action), action)


if __name__ == "__main__":
    unittest.main()
