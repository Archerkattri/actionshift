"""DualABI probe-adapter tests: task-regret selection, early stop, leakage guard."""

from __future__ import annotations

import unittest

import torch

from actionshift.adaptation.dualabi_adapter import DualABIProbeAdapter
from actionshift.adaptation.hypotheses import ExactBeliefDriver
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.types import ActionContract
from tests.test_adaptation_stage1 import (
    _BATCH,
    SyntheticHiddenEnvironment,
    _contract,
    _pool,
    _true_contract,
)


def _driver(pool: tuple[ActionContract, ...] | None = None) -> ExactBeliefDriver:
    return ExactBeliefDriver(
        pool if pool is not None else _pool(include_true=True),
        batch_size=_BATCH,
        response=ResponseModel(alpha=1.0, sigma=0.05),
    )


def _run(
    adapter: DualABIProbeAdapter,
    steps: int,
    *,
    true_contract: ActionContract | None = None,
    seed: int = 20260720,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    environment = SyntheticHiddenEnvironment(
        true_contract if true_contract is not None else _true_contract(), _BATCH
    )
    for _ in range(steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6])
    return adapter.driver.map_indices()


class DualABIIdentificationTest(unittest.TestCase):
    def test_identifies_true_contract_within_budget(self) -> None:
        true_index = len(_pool(include_true=True)) - 1
        adapter = DualABIProbeAdapter(
            _driver(), budget=12, amplitude=0.5, regret_threshold=0.0
        )
        map_indices = _run(adapter, steps=13)
        self.assertTrue(
            bool((map_indices == true_index).all()),
            "DualABI probes must identify the true contract",
        )

    def test_selected_probe_is_a_bounded_pose_pulse(self) -> None:
        # A non-zero intent from a uniform prior carries real task regret, so the
        # adapter probes; the pulse must stay inside the amplitude bound and never
        # touch the gripper channel (the fail-closed safety by construction).
        adapter = DualABIProbeAdapter(
            _driver(), budget=1, amplitude=0.5, regret_threshold=0.0
        )
        probe = adapter.encode(torch.full((_BATCH, 7), 0.3))
        self.assertTrue(bool((probe.abs() <= 0.5 + 1e-6).all()))
        self.assertTrue(bool((probe[:, 6] == 0).all()), "gripper channel is never probed")
        self.assertGreater(float(probe.abs().sum()), 0.0)


class TaskRegretSelectionTest(unittest.TestCase):
    def test_probe_reduces_task_regret_from_a_uniform_prior(self) -> None:
        # From a uniform belief, the chosen pulse's expected posterior regret must
        # not exceed the current regret: the selector buys task-relevant separation.
        adapter = DualABIProbeAdapter(_driver(), budget=6, regret_threshold=0.0)
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        intent = torch.full((_BATCH, 7), 0.3)
        adapter.encode(intent)
        assert adapter.last_task_regret is not None
        first_regret = adapter.last_task_regret.clone()
        raw = adapter.driver.map_encode(intent)  # not used; keep belief untouched
        del raw
        # drive several probe steps and confirm regret is monotonically pushed down
        regrets = [float(first_regret.mean())]
        for _ in range(5):
            raw = adapter.encode(intent)
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
            if adapter.last_task_regret is not None:
                regrets.append(float(adapter.last_task_regret.mean()))
        self.assertLess(regrets[-1], regrets[0], "probing must reduce mean task regret")

    def test_regret_ignores_task_equivalent_hypotheses(self) -> None:
        # Two identical contracts in the pool are task-equivalent: a belief split
        # between them must carry zero task regret even though entropy is maximal.
        duplicate = _contract((0, 1, 2, 3, 4, 5), (1,) * 6, (1.0,) * 6)
        pool = (duplicate, duplicate)
        adapter = DualABIProbeAdapter(_driver(pool), budget=6, regret_threshold=0.01)
        adapter.encode(torch.full((_BATCH, 7), 0.4))
        assert adapter.last_task_regret is not None
        self.assertTrue(
            bool((adapter.last_task_regret.abs() < 1e-5).all()),
            "task-equivalent hypotheses must contribute no regret",
        )


class EarlyStopTest(unittest.TestCase):
    def test_high_threshold_stops_before_spending_budget(self) -> None:
        # A large threshold makes the MAP look good enough immediately: no probe.
        adapter = DualABIProbeAdapter(
            _driver(), budget=6, regret_threshold=1e9
        )
        probe = adapter.encode(torch.full((_BATCH, 7), 0.3))
        assert adapter.last_probe_mask is not None
        self.assertFalse(
            bool(adapter.last_probe_mask.any()), "early stop must skip all probing"
        )
        expected = adapter.driver.map_encode(torch.full((_BATCH, 7), 0.3))
        torch.testing.assert_close(probe, expected)

    def test_early_stop_is_sticky_and_uses_fewer_steps_than_budget(self) -> None:
        # With a permissive threshold, DualABI stops once identified and therefore
        # spends strictly fewer than ``budget`` probe steps.
        adapter = DualABIProbeAdapter(
            _driver(), budget=6, amplitude=0.5, regret_threshold=0.05
        )
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        generator = torch.Generator().manual_seed(7)
        probe_steps = torch.zeros(_BATCH, dtype=torch.long)
        for _ in range(12):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            raw = adapter.encode(intent)
            assert adapter.last_probe_mask is not None
            probe_steps += adapter.last_probe_mask.long()
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
        self.assertTrue(
            bool((probe_steps < 6).all()),
            f"early stop must save probe steps, used {probe_steps.tolist()}",
        )
        self.assertTrue(bool((probe_steps > 0).all()), "some probing must occur first")

    def test_stop_state_resets_on_episode_boundary(self) -> None:
        adapter = DualABIProbeAdapter(_driver(), budget=4, regret_threshold=1e9)
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        raw = adapter.encode(torch.zeros((_BATCH, 7)))
        executed = environment.step(raw)
        assert adapter.last_probe_mask is not None
        self.assertFalse(bool(adapter.last_probe_mask.any()))  # stopped by threshold
        boundary = torch.tensor([True, False, False])
        adapter.observe(raw, executed[:, :6], invalid_mask=boundary)
        # After the boundary, env 0 restarts with a fresh (probing) episode when the
        # threshold is lowered; verify the sticky-stop flag cleared for env 0 only.
        self.assertFalse(bool(adapter._stopped[0]))
        self.assertTrue(bool(adapter._stopped[1:].all()))


class LeakageGuardTest(unittest.TestCase):
    def test_encode_signature_never_receives_the_true_contract(self) -> None:
        import inspect

        signature = inspect.signature(DualABIProbeAdapter.encode)
        # ``ee_rotation`` is the observed live tcp rotation (contract-independent
        # task knowledge for the v2 real-rotation variant), never the true
        # contract; it defaults to None (the identity variant).
        self.assertEqual(
            list(signature.parameters), ["self", "canonical_action", "ee_rotation"]
        )
        self.assertIsNone(signature.parameters["ee_rotation"].default)

    def test_identification_uses_only_the_declared_pool(self) -> None:
        # The true contract is deliberately absent from the pool: DualABI cannot
        # reach it and must land on a pooled hypothesis, never the hidden truth.
        pool = _pool(include_true=False)
        adapter = DualABIProbeAdapter(_driver(pool), budget=12, regret_threshold=0.0)
        map_indices = _run(adapter, steps=13)
        self.assertTrue(
            bool((map_indices < len(pool)).all()),
            "belief must stay inside the declared pool",
        )

    def test_recent_buffer_does_not_cross_episode_boundaries(self) -> None:
        adapter = DualABIProbeAdapter(_driver(), budget=6, regret_threshold=0.0)
        first = torch.full((_BATCH, 7), 0.5)
        adapter.encode(first)
        boundary = torch.tensor([True, False, False])
        adapter.observe(
            adapter.driver.map_encode(first),
            torch.zeros((_BATCH, 6)),
            invalid_mask=boundary,
        )
        second = torch.full((_BATCH, 7), -0.2)
        adapter.encode(second)
        assert adapter._recent is not None
        # env 0 was reset: its whole window must be the new intent, not the old one.
        torch.testing.assert_close(
            adapter._recent[:, 0], second[0].unsqueeze(0).expand(adapter.recent_window, 7)
        )


if __name__ == "__main__":
    unittest.main()
