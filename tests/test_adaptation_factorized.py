"""Full-grammar factorized belief tests: honest convergence without pool privilege.

These mirror ``tests/test_adaptation_stage1`` for the pool replica: a bit-faithful
synthetic hidden environment (decode-then-lag under identity rotation) drives the
adapter, and the grammar belief must converge to a randomly sampled *full-grammar*
contract and reach oracle parity -- with no pool and no true contract handed in.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from actionshift.adaptation.factorized_grammar import (
    GRAMMAR_SCALES,
    FactorizedGrammarAdapter,
    FactorizedGrammarDriver,
    FactorizedGrammarProbingAdapter,
)
from actionshift.adaptation.hypotheses import identity_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract

_BATCH = 3


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


def _drive(
    adapter: FactorizedGrammarAdapter,
    contract: ActionContract,
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the synthetic loop; return the final intent and executed canonical pose."""
    environment = SyntheticHiddenEnvironment(contract, _BATCH)
    intent = executed = torch.zeros((_BATCH, 7))
    for _ in range(steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6])
    return intent, executed


def _clean_response() -> ResponseModel:
    """Noiseless unit-gain response: attenuation-free, so scale is identifiable."""
    return ResponseModel(alpha=1.0, sigma=0.02)


class GrammarConvergenceTest(unittest.TestCase):
    def test_recovers_random_full_grammar_contract_and_reaches_parity(self) -> None:
        contract = ActionContract(
            permutation=(4, 0, 5, 1, 3, 2),
            sign=(1, -1, -1, 1, 1, -1),
            scale=(0.5, 2.0, 1.25, 0.75, 1.5, 0.6),
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response()
        )
        generator = torch.Generator().manual_seed(20260718)
        intent, executed = _drive(adapter, contract, steps=48, generator=generator)
        recovered = adapter.driver.map_contracts()
        for env in range(_BATCH):
            self.assertEqual(recovered[env].permutation, contract.permutation)
            self.assertEqual(recovered[env].sign, contract.sign)
            self.assertEqual(recovered[env].scale, contract.scale)
            self.assertEqual(recovered[env].target, contract.target)
        torch.testing.assert_close(executed[:, :6], intent[:, :6], atol=1e-4, rtol=1e-3)

    def test_recovers_absolute_target_contract(self) -> None:
        contract = ActionContract(
            permutation=(2, 5, 0, 3, 1, 4),
            sign=(-1, 1, 1, -1, 1, -1),
            scale=(1.25, 0.75, 2.0, 0.5, 1.5, 1.0),
            target="absolute",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response()
        )
        generator = torch.Generator().manual_seed(11)
        intent, executed = _drive(adapter, contract, steps=48, generator=generator)
        recovered = adapter.driver.map_contracts()
        for env in range(_BATCH):
            self.assertEqual(recovered[env].permutation, contract.permutation)
            self.assertEqual(recovered[env].target, "absolute")
            self.assertEqual(recovered[env].scale, contract.scale)
        torch.testing.assert_close(executed[:, :6], intent[:, :6], atol=1e-4, rtol=1e-3)


class AssignmentCorrectnessTest(unittest.TestCase):
    def test_assignment_is_a_valid_bijection_matching_the_truth(self) -> None:
        contract = ActionContract(
            permutation=(3, 1, 4, 0, 5, 2),
            sign=(1, 1, -1, -1, 1, 1),
            scale=(0.75, 1.0, 1.5, 2.0, 0.6, 1.25),
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response()
        )
        generator = torch.Generator().manual_seed(3)
        _drive(adapter, contract, steps=48, generator=generator)
        for recovered in adapter.driver.map_contracts():
            self.assertEqual(sorted(recovered.permutation), list(range(6)))
            self.assertEqual(recovered.permutation, contract.permutation)


class LagIdentificationTest(unittest.TestCase):
    def test_identifies_a_lagged_mode(self) -> None:
        contract = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1, 1, 1, 1, 1, 1),
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            target="delta",
            frame="base",
            lag=2,
            gripper_inverted=False,
        )
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response()
        )
        generator = torch.Generator().manual_seed(5)
        _drive(adapter, contract, steps=48, generator=generator)
        for recovered in adapter.driver.map_contracts():
            self.assertEqual(recovered.lag, 2)


class IdentifiabilityLimitsTest(unittest.TestCase):
    def test_frame_is_collapsed_but_parity_holds_under_identity_rotation(self) -> None:
        # A tool-frame contract is observationally identical to base under identity
        # rotation, so the belief reports base yet still reaches parity.
        tool = ActionContract(
            permutation=(1, 0, 2, 3, 4, 5),
            sign=(1, -1, 1, 1, -1, 1),
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            target="delta",
            frame="tool",
            lag=0,
            gripper_inverted=False,
        )
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response()
        )
        generator = torch.Generator().manual_seed(7)
        intent, executed = _drive(adapter, tool, steps=48, generator=generator)
        for recovered in adapter.driver.map_contracts():
            self.assertEqual(recovered.frame, "base")
        torch.testing.assert_close(executed[:, :6], intent[:, :6], atol=1e-4, rtol=1e-3)

    def test_gripper_inversion_is_unidentified(self) -> None:
        contract = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1, 1, 1, 1, 1, 1),
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=True,
        )
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response()
        )
        generator = torch.Generator().manual_seed(9)
        intent, executed = _drive(adapter, contract, steps=32, generator=generator)
        for recovered in adapter.driver.map_contracts():
            self.assertFalse(recovered.gripper_inverted)
        # Pose is recovered; the gripper channel is inverted because the belief
        # cannot observe gripper inversion from the pose response.
        torch.testing.assert_close(executed[:, :6], intent[:, :6], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(executed[:, 6], -intent[:, 6], atol=1e-4, rtol=1e-3)


class BitFaithfulPredictionTest(unittest.TestCase):
    def test_history_ring_prediction_matches_wrapper_decode(self) -> None:
        """The driver's per-cell prediction for the true contract equals the wrapper."""
        contract = ActionContract(
            permutation=(2, 4, 0, 5, 1, 3),
            sign=(1, -1, 1, -1, 1, -1),
            scale=(1.5, 0.5, 1.25, 2.0, 0.6, 0.75),
            target="absolute",
            frame="base",
            lag=2,
            gripper_inverted=False,
        )
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=_clean_response()
        )
        environment = SyntheticHiddenEnvironment(contract, _BATCH)
        generator = torch.Generator().manual_seed(13)
        sign = torch.tensor(contract.sign, dtype=torch.float32)
        scale = torch.tensor(contract.scale, dtype=torch.float32)
        perm = torch.tensor(contract.permutation)
        for _ in range(20):
            raw = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            executed = environment.step(raw)
            driver.update(raw, executed[:, :6])
            base = driver._predicted_base(contract.target, contract.lag)  # (B, 6[j])
            predicted = sign * scale * base.index_select(1, perm)
            torch.testing.assert_close(predicted, executed[:, :6], atol=1e-5, rtol=1e-4)


class MaskTimingTest(unittest.TestCase):
    def test_invalid_mask_discards_evidence_and_resets_scores(self) -> None:
        contract = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1,) * 6,
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=_clean_response()
        )
        environment = SyntheticHiddenEnvironment(contract, _BATCH)
        generator = torch.Generator().manual_seed(17)
        for _ in range(6):
            raw = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            executed = environment.step(raw)
            driver.update(raw, executed[:, :6])
        self.assertGreater(float(driver._scores.abs().max()), 0.0)
        raw = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        executed = environment.step(raw)
        boundary = torch.tensor([True, False, False])
        driver.update(raw, executed[:, :6], invalid_mask=boundary)
        # Environment 0 crossed a boundary: its accumulated evidence is wiped.
        self.assertEqual(float(driver._scores[:, 0].abs().max()), 0.0)
        self.assertGreater(float(driver._scores[:, 1].abs().max()), 0.0)

    def test_reset_mask_zeros_history_ring(self) -> None:
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=_clean_response()
        )
        raw = torch.ones((_BATCH, 7))
        driver.update(raw, torch.zeros((_BATCH, 6)))
        reset = torch.tensor([True, False, False])
        driver.update(raw, torch.zeros((_BATCH, 6)), reset_mask=reset)
        # Env 0's history was zeroed before this step's push, so only the current
        # slot is nonzero; env 1 retains two nonzero slots.
        self.assertEqual(float(driver._history[1, 0].abs().max()), 0.0)
        self.assertGreater(float(driver._history[1, 1].abs().max()), 0.0)


class GrammarCoverageTest(unittest.TestCase):
    def test_scale_grid_covers_every_frozen_evaluation_contract(self) -> None:
        grid = set(GRAMMAR_SCALES)
        for split in ("seen", "unseen_composition", "long_lag"):
            for contract in representative_contracts(split):
                for value in contract.scale:
                    self.assertIn(value, grid)
                self.assertIn(contract.lag, (0, 1, 2, 4))
                self.assertIn(contract.target, ("delta", "absolute"))


class LeakageGuardTest(unittest.TestCase):
    def test_adapters_cannot_receive_the_true_contract(self) -> None:
        for cls in (FactorizedGrammarAdapter, FactorizedGrammarDriver):
            parameters = inspect.signature(cls.__init__).parameters
            self.assertNotIn("true_contract", parameters)
            self.assertNotIn("contract", parameters)
            self.assertNotIn("pool", parameters)

    def test_probing_adapter_only_receives_a_driver(self) -> None:
        parameters = inspect.signature(FactorizedGrammarProbingAdapter.__init__).parameters
        self.assertNotIn("true_contract", parameters)
        self.assertNotIn("contract", parameters)


class ProbingAdapterTest(unittest.TestCase):
    def test_probe_phase_emits_bounded_pulses_then_controls(self) -> None:
        contract = ActionContract(
            permutation=(1, 0, 3, 2, 5, 4),
            sign=(1, -1, 1, -1, 1, -1),
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=_clean_response()
        )
        adapter = FactorizedGrammarProbingAdapter(driver, budget=6, amplitude=0.5)
        environment = SyntheticHiddenEnvironment(contract, _BATCH)
        generator = torch.Generator().manual_seed(19)
        for _step in range(6):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            raw = adapter.encode(intent)
            self.assertLessEqual(float(raw.abs().max()), 0.5 + 1e-6)
            self.assertIsNotNone(adapter.last_probe_mask)
            assert adapter.last_probe_mask is not None
            self.assertTrue(bool(adapter.last_probe_mask.all()))
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        assert adapter.last_probe_mask is not None
        self.assertFalse(bool(adapter.last_probe_mask.any()))
        for recovered in driver.map_contracts():
            self.assertEqual(recovered.permutation, contract.permutation)


if __name__ == "__main__":
    unittest.main()
