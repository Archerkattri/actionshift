"""Gripper evidence channel, magnitude-dependent gain, and drift re-anchoring.

These cover the version-2 calibration upgrades: locating and fitting the gripper
finger block, the saturating pose-gain fit, the belief families identifying
``gripper_inverted`` from the seventh evidence channel (which the pose channels
cannot reveal), periodic target re-anchoring, and the leakage guard for the new
constructor arguments.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from actionshift.adaptation.calibration import (
    ResponseCalibration,
    calibrate_response,
    fit_gripper_response,
    fit_saturating_response,
    quaternion_multiply,
    response_from_observations,
)
from actionshift.adaptation.factorized_grammar import (
    FactorizedGrammarAdapter,
    FactorizedGrammarDriver,
)
from actionshift.adaptation.hypotheses import ExactBeliefDriver, identity_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract

_BATCH = 3


class _SyntheticGripperProbe:
    """Probe with a known finger layout and a known linear gripper gain.

    Observation layout: [noise(2), position(3), quaternion(4), fingers(2)].
    Finger positions integrate ``gripper_gain * command`` each step, so the
    per-step finger delta is ``gripper_gain * command`` (both fingers move
    together, as with the Panda mimic joint).
    """

    def __init__(self, *, gripper_gain: float = 0.05) -> None:
        self._generator = torch.Generator().manual_seed(9)
        self._position = torch.zeros(2, 3)
        self._quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        self._fingers = torch.full((2, 2), 0.02)
        self._action = torch.zeros(2, 7)
        self._gain = gripper_gain

    def gripper_positions(self) -> torch.Tensor:
        return self._fingers.clone()

    def step_random(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._action = torch.rand((2, 7), generator=self._generator) * 2.0 - 1.0
        self._position = self._position + 0.1 * self._action[:, :3]
        rotation_vector = 0.3 * self._action[:, 3:6]
        angle = rotation_vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        axis = rotation_vector / angle
        half = angle / 2.0
        step = torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)
        self._quaternion = quaternion_multiply(step, self._quaternion)
        self._fingers = self._fingers + self._gain * self._action[:, 6:7]
        observation = torch.cat(
            (
                torch.rand((2, 2), generator=self._generator),
                self._position,
                self._quaternion,
                self._fingers,
            ),
            dim=-1,
        )
        return observation, self._position.clone(), self._quaternion.clone(), self._action


class GripperCalibrationTest(unittest.TestCase):
    def test_locates_finger_block_and_fits_gain(self) -> None:
        calibration = calibrate_response(
            _SyntheticGripperProbe(gripper_gain=0.05),
            task="synthetic",
            steps=64,
            calibrate_gripper=True,
        )
        self.assertEqual(calibration.version, 2)
        self.assertTrue(calibration.has_gripper)
        # fingers sit at observation offset 2 + 3 + 4 = 9.
        self.assertEqual(calibration.gripper_start, 9)
        assert calibration.gripper_alpha is not None
        self.assertAlmostEqual(calibration.gripper_alpha, 0.05, places=3)
        assert calibration.gripper_fit_r2 is not None
        self.assertGreater(calibration.gripper_fit_r2, 0.99)
        restored = ResponseCalibration.from_json(calibration.to_json())
        self.assertEqual(restored, calibration)

    def test_gripper_fit_recovers_sign(self) -> None:
        generator = torch.Generator().manual_seed(5)
        command = torch.rand(400, generator=generator) * 2.0 - 1.0
        observed = -0.03 * command + 1e-4 * torch.randn(400, generator=generator)
        alpha, sigma, r2 = fit_gripper_response(command, observed)
        self.assertAlmostEqual(alpha, -0.03, places=3)
        self.assertGreater(r2, 0.99)
        self.assertLess(sigma, 0.01)

    def test_response_extraction_appends_finger_delta(self) -> None:
        probe = _SyntheticGripperProbe(gripper_gain=0.05)
        calibration = calibrate_response(
            _SyntheticGripperProbe(gripper_gain=0.05),
            task="synthetic",
            steps=64,
            calibrate_gripper=True,
        )
        previous, _, _, _ = probe.step_random()
        current, _, _, action = probe.step_random()
        response = response_from_observations(calibration, previous, current)
        self.assertEqual(response.shape[-1], 7)
        torch.testing.assert_close(
            response[:, 6], 0.05 * action[:, 6], atol=1e-5, rtol=1e-4
        )


class MagnitudeGainTest(unittest.TestCase):
    def test_saturating_fit_recovers_c0(self) -> None:
        generator = torch.Generator().manual_seed(7)
        commanded = torch.rand((2000, 6), generator=generator) * 2.0 - 1.0
        c0_true = 1.0
        gains = torch.tensor([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
        feature = commanded / (1.0 + commanded.abs() / c0_true)
        observed = feature * gains + 1e-4 * torch.randn((2000, 6), generator=generator)
        alpha, c0, r2 = fit_saturating_response(commanded, observed)
        for channel in range(6):
            self.assertAlmostEqual(c0[channel], c0_true, places=6)
            self.assertAlmostEqual(alpha[channel], float(gains[channel]), places=2)
            self.assertGreater(r2[channel], 0.99)

    def test_calibrate_response_records_saturating_model(self) -> None:
        calibration = calibrate_response(
            _SyntheticGripperProbe(),
            task="synthetic",
            steps=64,
            magnitude_gain=True,
        )
        self.assertEqual(calibration.version, 2)
        self.assertEqual(calibration.gain_model, "saturating")
        self.assertIsNotNone(calibration.alpha_c0)
        self.assertIsNotNone(calibration.saturating_fit_r2)


class _SyntheticGraspEnvironment:
    """Bit-faithful hidden wrapper that also emits a gripper finger response.

    Pose channels are the decoder output; the seventh response channel is
    ``gripper_gain * canonical_gripper`` (the canonical gripper already carries the
    contract's inversion), mirroring the real command-to-finger-delta relation.
    """

    def __init__(self, contract: ActionContract, batch_size: int, *, gripper_gain: float) -> None:
        self._decoder = CompleteActionDecoder(contract, batch_size=batch_size)
        self._batch_size = batch_size
        self._gain = gripper_gain

    def step(self, raw_action: torch.Tensor) -> torch.Tensor:
        rotation = identity_rotation(self._batch_size, raw_action.device, raw_action.dtype)
        canonical = self._decoder.step(raw_action, ee_rotation=rotation)
        finger = self._gain * canonical[:, 6:7]
        return torch.cat((canonical[:, :6], finger), dim=-1)


def _gripper_response() -> ResponseModel:
    return ResponseModel(alpha=1.0, sigma=0.02, gripper_alpha=0.05, gripper_sigma=0.01)


def _drive_grasp(
    adapter: FactorizedGrammarAdapter,
    contract: ActionContract,
    steps: int,
    generator: torch.Generator,
) -> None:
    environment = _SyntheticGraspEnvironment(contract, _BATCH, gripper_gain=0.05)
    for _ in range(steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed)


class FactorizedGripperIdentificationTest(unittest.TestCase):
    def test_identifies_gripper_inversion_with_the_evidence_channel(self) -> None:
        contract = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1, 1, 1, 1, 1, 1),
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=True,
        )
        adapter = FactorizedGrammarAdapter(batch_size=_BATCH, response=_gripper_response())
        generator = torch.Generator().manual_seed(21)
        _drive_grasp(adapter, contract, steps=32, generator=generator)
        for recovered in adapter.driver.map_contracts():
            self.assertTrue(recovered.gripper_inverted)

    def test_reports_non_inverted_gripper_correctly(self) -> None:
        contract = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1, 1, 1, 1, 1, 1),
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        adapter = FactorizedGrammarAdapter(batch_size=_BATCH, response=_gripper_response())
        generator = torch.Generator().manual_seed(22)
        _drive_grasp(adapter, contract, steps=32, generator=generator)
        for recovered in adapter.driver.map_contracts():
            self.assertFalse(recovered.gripper_inverted)

    def test_gripper_belief_resets_on_boundary(self) -> None:
        driver = FactorizedGrammarDriver(batch_size=_BATCH, response=_gripper_response())
        raw = torch.zeros((_BATCH, 7))
        raw[:, 6] = 1.0
        observed = torch.zeros((_BATCH, 7))
        observed[:, 6] = -0.05  # matches the inverted hypothesis
        for _ in range(4):
            driver.update(raw, observed)
        self.assertGreater(float(driver._gripper_scores[1, 0]), float(driver._gripper_scores[0, 0]))
        boundary = torch.tensor([True, False, False])
        driver.update(raw, observed, invalid_mask=boundary)
        self.assertEqual(float(driver._gripper_scores[:, 0].abs().max()), 0.0)
        self.assertGreater(float(driver._gripper_scores[:, 1].abs().max()), 0.0)


class PoolGripperDisambiguationTest(unittest.TestCase):
    def test_pool_belief_prefers_the_correct_gripper_sign(self) -> None:
        base = dict(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1, 1, 1, 1, 1, 1),
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=0,
        )
        not_inverted = ActionContract(**base, gripper_inverted=False)  # type: ignore[arg-type]
        inverted = ActionContract(**base, gripper_inverted=True)  # type: ignore[arg-type]
        driver = ExactBeliefDriver(
            (not_inverted, inverted), batch_size=_BATCH, response=_gripper_response()
        )
        environment = _SyntheticGraspEnvironment(inverted, _BATCH, gripper_gain=0.05)
        generator = torch.Generator().manual_seed(23)
        for _ in range(16):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            executed = environment.step(intent)
            driver.update(intent, executed)
        for env in range(_BATCH):
            self.assertEqual(int(driver.map_indices()[env].item()), 1)


class ReanchorTest(unittest.TestCase):
    def test_reanchor_sets_target_to_observed_cumulative(self) -> None:
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=ResponseModel(alpha=1.0, sigma=0.02), reanchor_period=1
        )
        raw = torch.zeros((_BATCH, 7))
        observed = torch.zeros((_BATCH, 6))
        observed[:, 0] = 0.1  # 0.1 m of achieved x-displacement per step
        for _ in range(3):
            driver.update(raw, observed)
        # After three steps the tracked target's x channel equals 0.3 m of reality.
        torch.testing.assert_close(
            driver._tracked_target[:, 0], torch.full((_BATCH,), 0.3), atol=1e-5, rtol=1e-4
        )

    def test_reanchor_off_by_default(self) -> None:
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=ResponseModel(alpha=1.0, sigma=0.02)
        )
        raw = torch.zeros((_BATCH, 7))
        observed = torch.zeros((_BATCH, 6))
        observed[:, 0] = 0.1
        driver.update(raw, observed)
        self.assertEqual(float(driver._tracked_target.abs().max()), 0.0)


class GripperLeakageGuardTest(unittest.TestCase):
    def test_new_arguments_do_not_leak_the_contract(self) -> None:
        for cls in (FactorizedGrammarAdapter, FactorizedGrammarDriver):
            parameters = inspect.signature(cls.__init__).parameters
            for forbidden in ("true_contract", "contract", "pool"):
                self.assertNotIn(forbidden, parameters)
            # the new evidence knobs are present and contract-independent
            self.assertIn("reanchor_period", parameters)


if __name__ == "__main__":
    unittest.main()
