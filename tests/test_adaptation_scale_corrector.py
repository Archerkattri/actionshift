"""Drift-based scale-corrector tests: closed-loop convergence without privilege.

Two layers are covered. Unit tests drive the :class:`ScaleCorrector` in a closed
loop -- the synthetic observed response reflects the *current* effective scale, so
a correct corrector must reach a fixed point at the true scale -- and check the
fail-safes (thin-evidence freeze, multiplicative bound, boundary reset). Integration
tests wrap the corrector around the full-grammar factorized adapter against the
bit-faithful synthetic hidden wrapper and show that an off-grid true scale, which
the grid MAP cannot represent, is recovered so the executed canonical matches the
intent under an absolute target -- and that an already-identifiable delta cell is
left unperturbed.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from actionshift.adaptation.factorized_grammar import (
    FactorizedGrammarAdapter,
    FactorizedGrammarDriver,
    FactorizedGrammarProbingAdapter,
)
from actionshift.adaptation.hypotheses import identity_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.adaptation.scale_corrector import ScaleCorrector
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract

_BATCH = 3
_CHANNELS = 6


def _clean_response() -> ResponseModel:
    """Noiseless unit-gain response: attenuation-free, so scale is identifiable."""
    return ResponseModel(alpha=1.0, sigma=0.02)


class SyntheticHiddenEnvironment:
    """Bit-faithful stand-in for the hidden wrapper: decode-then-lag, identity rotation."""

    def __init__(self, contract: ActionContract, batch_size: int) -> None:
        self._decoder = CompleteActionDecoder(contract, batch_size=batch_size)
        self._batch_size = batch_size

    def step(self, raw_action: torch.Tensor) -> torch.Tensor:
        rotation = identity_rotation(self._batch_size, raw_action.device, raw_action.dtype)
        return self._decoder.step(raw_action, ee_rotation=rotation)


def _drive(
    adapter: FactorizedGrammarAdapter,
    contract: ActionContract,
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    environment = SyntheticHiddenEnvironment(contract, _BATCH)
    intent = executed = torch.zeros((_BATCH, 7))
    for _ in range(steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6])
    return intent, executed


class ScaleCorrectorConvergenceTest(unittest.TestCase):
    def _run_closed_loop(
        self,
        *,
        alpha: torch.Tensor,
        true_scale: torch.Tensor,
        base_scale: torch.Tensor,
        window: int,
        steps: int,
        noise: float,
        seed: int,
    ) -> ScaleCorrector:
        corrector = ScaleCorrector(
            batch_size=_BATCH, alpha=alpha, window=window, command_floor=0.01
        )
        generator = torch.Generator().manual_seed(seed)
        base = base_scale.expand(_BATCH, _CHANNELS)
        for _ in range(steps):
            commanded = torch.rand((_BATCH, _CHANNELS), generator=generator) * 2.0 - 1.0
            effective = corrector.effective_scale(base)
            ratio = true_scale.expand(_BATCH, _CHANNELS) / effective
            observed = alpha.unsqueeze(0) * ratio * commanded
            if noise > 0.0:
                observed = observed + noise * torch.randn(
                    (_BATCH, _CHANNELS), generator=generator
                )
            corrector.accumulate(commanded, observed, effective)
        return corrector

    def test_noiseless_recovers_true_scale_in_one_window(self) -> None:
        alpha = torch.tensor([0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
        true_scale = torch.tensor([1.3, 0.7, 1.5, 0.6, 2.0, 0.9])
        base_scale = torch.ones(_CHANNELS)
        corrector = self._run_closed_loop(
            alpha=alpha, true_scale=true_scale, base_scale=base_scale,
            window=8, steps=24, noise=0.0, seed=1,
        )
        effective = corrector.effective_scale(base_scale.expand(_BATCH, _CHANNELS))
        for env in range(_BATCH):
            torch.testing.assert_close(effective[env], true_scale, atol=1e-4, rtol=1e-3)

    def test_off_unit_base_recovers_true_scale_ratio(self) -> None:
        # The window integrates at a fixed effective scale (restarting when it
        # changes at a commit), so it recovers the true scale off an off-true base.
        alpha = torch.tensor([0.4, 0.6, 0.5, -0.3, 0.7, -0.5])
        true_scale = torch.tensor([1.1, 0.85, 1.4, 0.55, 1.7, 1.05])
        base_scale = torch.tensor([1.0, 0.75, 1.25, 0.5, 2.0, 1.0])
        corrector = self._run_closed_loop(
            alpha=alpha, true_scale=true_scale, base_scale=base_scale,
            window=8, steps=24, noise=0.0, seed=2,
        )
        effective = corrector.effective_scale(base_scale.expand(_BATCH, _CHANNELS))
        for env in range(_BATCH):
            torch.testing.assert_close(effective[env], true_scale, atol=1e-3, rtol=1e-3)

    def test_integration_averages_per_step_noise(self) -> None:
        alpha = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        true_scale = torch.tensor([1.4, 0.6, 1.5, 0.7, 1.8, 0.9])
        base_scale = torch.ones(_CHANNELS)
        corrector = self._run_closed_loop(
            alpha=alpha, true_scale=true_scale, base_scale=base_scale,
            window=48, steps=48 * 4, noise=0.05, seed=3,
        )
        effective = corrector.effective_scale(base_scale.expand(_BATCH, _CHANNELS))
        for env in range(_BATCH):
            torch.testing.assert_close(effective[env], true_scale, atol=0.1, rtol=0.1)


class ScaleCorrectorFailsafeTest(unittest.TestCase):
    def test_thin_evidence_channel_is_frozen(self) -> None:
        alpha = torch.ones(_CHANNELS)
        base = torch.ones((_BATCH, _CHANNELS))
        corrector = ScaleCorrector(
            batch_size=_BATCH, alpha=alpha, window=8, command_floor=0.5
        )
        generator = torch.Generator().manual_seed(4)
        for _ in range(24):
            commanded = torch.rand((_BATCH, _CHANNELS), generator=generator) * 2.0 - 1.0
            commanded[:, 2] = 0.0  # channel 2 receives no excitation
            observed = 3.0 * commanded  # a large apparent ratio on excited channels
            corrector.accumulate(commanded, observed, base)
        # The un-excited channel never forms an estimate (defers to the grid MAP);
        # excited channels do form one and move off the base.
        self.assertFalse(bool(corrector.valid[:, 2].any()))
        self.assertTrue(bool(corrector.valid[:, 0].all()))
        effective = corrector.effective_scale(base)
        self.assertTrue(torch.allclose(effective[:, 2], torch.ones(_BATCH)))
        self.assertFalse(torch.allclose(effective[:, 0], torch.ones(_BATCH)))

    def test_estimate_is_bounded(self) -> None:
        alpha = torch.ones(_CHANNELS)
        base = torch.ones((_BATCH, _CHANNELS))
        corrector = ScaleCorrector(
            batch_size=_BATCH, alpha=alpha, window=4, command_floor=0.01,
            min_correction=0.4, max_correction=2.5,
        )
        generator = torch.Generator().manual_seed(5)
        for _ in range(16):
            commanded = torch.rand((_BATCH, _CHANNELS), generator=generator) + 0.5
            observed = 10.0 * commanded  # ratio 10 -> must clamp to max
            corrector.accumulate(commanded, observed, base)
        effective = corrector.effective_scale(base)
        self.assertTrue(torch.all(effective <= 2.5 + 1e-6))
        self.assertTrue(torch.allclose(effective, torch.full((_BATCH, _CHANNELS), 2.5)))

    def test_boundary_resets_selected_environments(self) -> None:
        alpha = torch.ones(_CHANNELS)
        base = torch.ones((_BATCH, _CHANNELS))
        corrector = ScaleCorrector(
            batch_size=_BATCH, alpha=alpha, window=4, command_floor=0.01
        )
        generator = torch.Generator().manual_seed(6)
        for _ in range(8):
            commanded = torch.rand((_BATCH, _CHANNELS), generator=generator) + 0.5
            corrector.accumulate(commanded, 1.6 * commanded, base)
        self.assertTrue(bool(corrector.valid.any()))
        mask = torch.tensor([True, False, False])
        corrector.reset(mask)
        self.assertFalse(bool(corrector.valid[0].any()))
        self.assertTrue(bool(corrector.valid[1].any()))
        self.assertTrue(torch.allclose(corrector.effective_scale(base)[0], torch.ones(_CHANNELS)))


class ScaleCorrectorIntegrationTest(unittest.TestCase):
    """The corrector wrapped around the factorized adapter on the synthetic wrapper."""

    def _off_grid_contract(self, target: str) -> ActionContract:
        # Scales deliberately between grammar grid points, so the grid MAP cannot
        # represent them and only the corrector can close the residual.
        return ActionContract(
            permutation=(2, 0, 1, 4, 5, 3),
            sign=(1, -1, 1, 1, -1, 1),
            scale=(1.1, 0.85, 1.4, 0.55, 1.7, 1.05),
            target=target,  # type: ignore[arg-type]
            frame="base",
            lag=0,
            gripper_inverted=False,
        )

    def test_absolute_off_grid_scale_recovered_only_with_corrector(self) -> None:
        contract = self._off_grid_contract("absolute")

        baseline = FactorizedGrammarAdapter(batch_size=_BATCH, response=_clean_response())
        intent, executed = _drive(baseline, contract, 40, torch.Generator().manual_seed(7))
        baseline_error = (executed[:, :6] - intent[:, :6]).abs().max().item()
        self.assertGreater(baseline_error, 0.05)  # off-grid scale drifts uncorrected

        corrected = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response(),
            scale_correction=True, scale_window=8, scale_command_floor=0.02,
        )
        intent2, executed2 = _drive(corrected, contract, 40, torch.Generator().manual_seed(7))
        corrected_error = (executed2[:, :6] - intent2[:, :6]).abs().max().item()
        self.assertLess(corrected_error, 1e-2)  # corrector closes the residual
        self.assertLess(corrected_error, baseline_error)

    def test_delta_identifiable_scale_left_unperturbed(self) -> None:
        # On-grid, identifiable scale under a delta target: the grid MAP is already
        # exact, so the corrector must stay near identity and not degrade parity.
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
            batch_size=_BATCH, response=_clean_response(),
            scale_correction=True, scale_window=8, scale_command_floor=0.02,
        )
        intent, executed = _drive(adapter, contract, 40, torch.Generator().manual_seed(8))
        torch.testing.assert_close(executed[:, :6], intent[:, :6], atol=1e-3, rtol=1e-3)
        corrector = adapter.driver.scale_correction
        assert corrector is not None
        # The estimate recovers the identifiable on-grid scale, so the effective
        # scale is unchanged and parity is preserved (no perturbation).
        true_scale = torch.tensor(contract.scale)
        effective = corrector.effective_scale(true_scale.expand(_BATCH, _CHANNELS))
        self.assertTrue(torch.all((effective - true_scale).abs() < 0.15))

    def test_delta_off_grid_scale_also_converges(self) -> None:
        # The same integrated ratio recovers scale under a delta target too, so the
        # corrector is safe (and helpful) there as well as under absolute.
        contract = self._off_grid_contract("delta")
        adapter = FactorizedGrammarAdapter(
            batch_size=_BATCH, response=_clean_response(),
            scale_correction=True, scale_window=8, scale_command_floor=0.02,
        )
        intent, executed = _drive(adapter, contract, 40, torch.Generator().manual_seed(9))
        self.assertLess((executed[:, :6] - intent[:, :6]).abs().max().item(), 1e-2)


class ScaleCorrectorLeakageGuardTest(unittest.TestCase):
    def test_corrector_takes_no_contract(self) -> None:
        parameters = inspect.signature(ScaleCorrector.__init__).parameters
        for forbidden in ("contract", "true_contract", "pool", "scale", "true_scale"):
            self.assertNotIn(forbidden, parameters)

    def test_scale_correction_adds_no_contract_argument(self) -> None:
        for cls in (FactorizedGrammarAdapter, FactorizedGrammarDriver):
            parameters = inspect.signature(cls.__init__).parameters
            self.assertIn("scale_correction", parameters)
            for forbidden in ("contract", "true_contract", "pool"):
                self.assertNotIn(forbidden, parameters)

    def test_probing_adapter_still_receives_only_a_driver(self) -> None:
        parameters = inspect.signature(FactorizedGrammarProbingAdapter.__init__).parameters
        for forbidden in ("contract", "true_contract", "pool"):
            self.assertNotIn(forbidden, parameters)


if __name__ == "__main__":
    unittest.main()
