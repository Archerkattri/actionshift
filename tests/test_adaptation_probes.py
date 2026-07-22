"""Stage 2 probe-adapter tests on the synthetic hidden environment."""

from __future__ import annotations

import unittest

import torch

from actionshift.adaptation.hypotheses import ExactBeliefDriver
from actionshift.adaptation.probes import ProbingBeliefAdapter, fixed_probe_pulse
from actionshift.adaptation.response import ResponseModel
from tests.test_adaptation_stage1 import (
    _BATCH,
    SyntheticHiddenEnvironment,
    _pool,
    _true_contract,
)


def _driver() -> ExactBeliefDriver:
    return ExactBeliefDriver(
        _pool(include_true=True),
        batch_size=_BATCH,
        response=ResponseModel(alpha=1.0, sigma=0.05),
    )


def _run(adapter: ProbingBeliefAdapter, steps: int, seed: int = 20260720) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
    for _ in range(steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6])
    return adapter.driver.map_indices()


class FixedProbePulseTest(unittest.TestCase):
    def test_cycles_channels_with_alternating_sign(self) -> None:
        steps = torch.arange(14)
        pulses = fixed_probe_pulse(steps, amplitude=0.5)
        self.assertEqual(pulses.shape, (14, 7))
        self.assertAlmostEqual(float(pulses[0, 0]), 0.5)
        self.assertAlmostEqual(float(pulses[6, 0]), -0.5)
        self.assertAlmostEqual(float(pulses[12, 0]), 0.5)
        self.assertTrue(bool((pulses[:, 6] == 0).all()))
        self.assertTrue(bool((pulses.abs().sum(dim=-1) == 0.5).all()))


class ProbeConvergenceTest(unittest.TestCase):
    def test_each_strategy_identifies_the_true_contract_within_budget(self) -> None:
        true_index = len(_pool(include_true=True)) - 1
        for strategy in ("fixed", "random", "entropy"):
            adapter = ProbingBeliefAdapter(
                _driver(), strategy=strategy, budget=12, amplitude=0.5, seed=1
            )
            map_indices = _run(adapter, steps=13)
            self.assertTrue(
                bool((map_indices == true_index).all()),
                f"{strategy} probes must identify the true contract",
            )

    def test_probe_mask_reflects_budget_then_clears(self) -> None:
        adapter = ProbingBeliefAdapter(_driver(), strategy="fixed", budget=3)
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        masks = []
        for _ in range(5):
            raw = adapter.encode(torch.zeros((_BATCH, 7)))
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
            assert adapter.last_probe_mask is not None
            masks.append(bool(adapter.last_probe_mask.any()))
        self.assertEqual(masks, [True, True, True, False, False])

    def test_probe_counter_resets_on_episode_boundary(self) -> None:
        adapter = ProbingBeliefAdapter(_driver(), strategy="fixed", budget=2)
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        for _ in range(4):
            raw = adapter.encode(torch.zeros((_BATCH, 7)))
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
        assert adapter.last_probe_mask is not None
        self.assertFalse(bool(adapter.last_probe_mask.any()))
        boundary = torch.tensor([True, False, False])
        adapter.observe(raw, executed[:, :6], invalid_mask=boundary)
        raw = adapter.encode(torch.zeros((_BATCH, 7)))
        assert adapter.last_probe_mask is not None
        self.assertTrue(bool(adapter.last_probe_mask[0]))
        self.assertFalse(bool(adapter.last_probe_mask[1:].any()))


class EntropySelectionTest(unittest.TestCase):
    def test_entropy_probe_is_a_bounded_pose_pulse(self) -> None:
        adapter = ProbingBeliefAdapter(_driver(), strategy="entropy", budget=1)
        probe = adapter.encode(torch.zeros((_BATCH, 7)))
        self.assertTrue(bool((probe.abs() <= 0.5 + 1e-6).all()))
        self.assertTrue(bool((probe[:, 6] == 0).all()))
        self.assertGreater(float(probe.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
