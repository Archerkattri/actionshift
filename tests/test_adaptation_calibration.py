"""Calibration math tests: slice location, quaternion deltas, linear fits."""

from __future__ import annotations

import math
import unittest

import torch

from actionshift.adaptation.calibration import (
    ResponseCalibration,
    calibrate_response,
    fit_linear_response,
    locate_contiguous_slice,
    pose_delta,
    quaternion_multiply,
    quaternion_to_rotation_vector,
    response_from_observations,
)


def _axis_angle_quaternion(axis: torch.Tensor, angle: float) -> torch.Tensor:
    unit = axis / axis.norm()
    half = angle / 2.0
    return torch.cat((torch.tensor([math.cos(half)]), math.sin(half) * unit))


class QuaternionMathTest(unittest.TestCase):
    def test_rotation_vector_roundtrip(self) -> None:
        axis = torch.tensor([0.3, -0.5, 0.8])
        angle = 0.7
        quaternion = _axis_angle_quaternion(axis, angle).unsqueeze(0)
        vector = quaternion_to_rotation_vector(quaternion)[0]
        torch.testing.assert_close(vector.norm(), torch.tensor(angle), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            vector / vector.norm(), axis / axis.norm(), atol=1e-5, rtol=1e-5
        )

    def test_pose_delta_recovers_relative_rotation(self) -> None:
        base = _axis_angle_quaternion(torch.tensor([0.0, 0.0, 1.0]), 0.4).unsqueeze(0)
        step = _axis_angle_quaternion(torch.tensor([0.0, 1.0, 0.0]), 0.25).unsqueeze(0)
        composed = quaternion_multiply(step, base)
        delta = pose_delta(
            torch.zeros(1, 3), base, torch.tensor([[0.1, -0.2, 0.3]]), composed
        )
        torch.testing.assert_close(delta[0, :3], torch.tensor([0.1, -0.2, 0.3]))
        torch.testing.assert_close(
            delta[0, 3:].norm(), torch.tensor(0.25), atol=1e-5, rtol=1e-5
        )


class SliceLocationTest(unittest.TestCase):
    def test_locates_embedded_block_and_rejects_absent_block(self) -> None:
        generator = torch.Generator().manual_seed(3)
        target = torch.rand((5, 2, 3), generator=generator)
        noise = torch.rand((5, 2, 9), generator=generator)
        observations = torch.cat((noise[..., :4], target, noise[..., 4:]), dim=-1)
        self.assertEqual(locate_contiguous_slice(observations, target), 4)
        with self.assertRaises(LookupError):
            locate_contiguous_slice(noise, target)


class LinearFitTest(unittest.TestCase):
    def test_recovers_known_gains(self) -> None:
        generator = torch.Generator().manual_seed(5)
        commanded = torch.rand((400, 6), generator=generator) * 2.0 - 1.0
        gains = torch.tensor([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
        observed = commanded * gains + 0.001 * torch.randn((400, 6), generator=generator)
        alpha, sigma, r2 = fit_linear_response(commanded, observed)
        for channel in range(6):
            self.assertAlmostEqual(alpha[channel], float(gains[channel]), places=2)
            self.assertGreater(r2[channel], 0.99)
            self.assertLess(sigma[channel], 0.01)


class _SyntheticProbe:
    """Probe with a known layout and a known linear response."""

    def __init__(self) -> None:
        self._generator = torch.Generator().manual_seed(9)
        self._position = torch.zeros(2, 3)
        self._quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        self._action = torch.zeros(2, 7)

    def step_random(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._action = torch.rand((2, 7), generator=self._generator) * 2.0 - 1.0
        self._position = self._position + 0.1 * self._action[:, :3]
        rotation_vector = 0.3 * self._action[:, 3:6]
        angle = rotation_vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        axis = rotation_vector / angle
        half = angle / 2.0
        step = torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)
        self._quaternion = quaternion_multiply(step, self._quaternion)
        observation = torch.cat(
            (torch.rand((2, 5), generator=self._generator), self._position, self._quaternion),
            dim=-1,
        )
        return observation, self._position.clone(), self._quaternion.clone(), self._action

    def run(self) -> ResponseCalibration:
        return calibrate_response(self, task="synthetic", steps=64)


class CalibrationEndToEndTest(unittest.TestCase):
    def test_calibrates_layout_and_gains_from_probe(self) -> None:
        calibration = _SyntheticProbe().run()
        self.assertEqual(calibration.position_start, 5)
        self.assertEqual(calibration.quaternion_start, 8)
        for channel in range(3):
            self.assertAlmostEqual(calibration.alpha[channel], 0.1, places=2)
        self.assertGreater(min(calibration.fit_r2[:3]), 0.99)
        restored = ResponseCalibration.from_json(calibration.to_json())
        self.assertEqual(restored, calibration)

    def test_response_extraction_matches_probe_dynamics(self) -> None:
        probe = _SyntheticProbe()
        calibration = _SyntheticProbe().run()
        previous, _, _, _ = probe.step_random()
        current, _, _, action = probe.step_random()
        response = response_from_observations(calibration, previous, current)
        torch.testing.assert_close(
            response[:, :3], 0.1 * action[:, :3], atol=1e-5, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
