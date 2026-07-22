"""Hold-probe excitation tests: absolute identification without pool privilege.

The bit-faithful synthetic hidden wrapper (decode-then-lag, identity rotation)
drives the hold-probing factorized adapter. The claims under test, all on
*absolute*-target contracts the per-step differenced score cannot identify:

1. the telescoped hold-window evidence identifies the absolute-vs-delta *target*
   and the *permutation* (and on-grid sign/scale) from the probe phase alone;
2. with the scale corrector downstream, an off-grid absolute scale is recovered so
   the executed canonical matches the intent after the probe;
3. a delta contract is still identified correctly (no target regression); and
4. the per-step path is bit-unchanged when no ``hold_mask`` is supplied.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from actionshift.adaptation.factorized_grammar import (
    FactorizedGrammarDriver,
    FactorizedGrammarProbingAdapter,
)
from actionshift.adaptation.hold_probe import (
    FactorizedGrammarHoldProbingAdapter,
    HoldProbeSchedule,
)
from actionshift.adaptation.hypotheses import identity_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract

_BATCH = 3
_CHANNELS = 6


def _clean_response() -> ResponseModel:
    return ResponseModel(alpha=1.0, sigma=0.02)


def _attenuated_response() -> ResponseModel:
    # A per-channel, partly sign-flipped low gain in the spirit of the real
    # pd_ee_delta_pose calibration -- still identifiable via integration.
    return ResponseModel(
        alpha=(0.6, -0.5, 0.55, -0.45, 0.65, -0.5),
        sigma=(0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
    )


class SyntheticHiddenEnvironment:
    """Bit-faithful stand-in for the hidden wrapper: decode-then-lag, identity rotation."""

    def __init__(self, contract: ActionContract, batch_size: int, response: ResponseModel) -> None:
        self._decoder = CompleteActionDecoder(contract, batch_size=batch_size)
        self._batch_size = batch_size
        self._alpha = torch.tensor(
            response.alpha if isinstance(response.alpha, tuple) else (response.alpha,) * 6
        )
        self._sigma = torch.tensor(
            response.sigma if isinstance(response.sigma, tuple) else (response.sigma,) * 6
        )

    def step(self, raw_action: torch.Tensor, *, noise: bool = False) -> torch.Tensor:
        rotation = identity_rotation(self._batch_size, raw_action.device, raw_action.dtype)
        executed = self._decoder.step(raw_action, ee_rotation=rotation)
        observed = self._alpha.unsqueeze(0) * executed[:, :6]
        if noise:
            observed = observed + self._sigma.unsqueeze(0) * torch.randn_like(observed)
        return torch.cat((observed, executed[:, 6:]), dim=-1)


def _drive_hold(
    adapter: FactorizedGrammarHoldProbingAdapter,
    contract: ActionContract,
    response: ResponseModel,
    control_steps: int,
    generator: torch.Generator,
    *,
    noise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the probe phase plus ``control_steps`` control steps; return last transition."""
    environment = SyntheticHiddenEnvironment(contract, _BATCH, response)
    total = adapter.schedule.total_steps
    intent = torch.zeros((_BATCH, 7))
    executed = torch.zeros((_BATCH, 7))
    for _ in range(total + control_steps):
        intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
        raw = adapter.encode(intent)
        response_obs = environment.step(raw, noise=noise)
        adapter.observe(raw, response_obs[:, :6])
        # The synthetic response carries alpha == 1 on the clean model, so the
        # observed pose is the executed canonical pose for the parity checks.
        executed = response_obs
    return intent, executed


def _random_absolute_contract(
    generator: torch.Generator, target: str = "absolute"
) -> ActionContract:
    scales = (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0)
    perm = torch.randperm(_CHANNELS, generator=generator).tolist()
    signs = tuple(1 if torch.rand(1, generator=generator).item() > 0.5 else -1 for _ in range(6))
    scale = tuple(
        scales[int(torch.randint(0, len(scales), (1,), generator=generator).item())]
        for _ in range(6)
    )
    return ActionContract(
        permutation=tuple(int(p) for p in perm),
        sign=signs,
        scale=scale,
        target=target,  # type: ignore[arg-type]
        frame="base",
        lag=0,
        gripper_inverted=False,
    )


def _make_hold_adapter(
    response: ResponseModel,
    *,
    hold_steps: int = 4,
    rounds: int = 1,
    scale_correction: bool = False,
    scale_window: int = 8,
) -> FactorizedGrammarHoldProbingAdapter:
    driver = FactorizedGrammarDriver(
        batch_size=_BATCH,
        response=response,
        scale_correction=scale_correction,
        scale_window=scale_window,
        scale_command_floor=0.02,
    )
    schedule = HoldProbeSchedule(amplitude=0.5, hold_steps=hold_steps, rounds=rounds)
    return FactorizedGrammarHoldProbingAdapter(driver, schedule=schedule)


class HoldProbeScheduleTest(unittest.TestCase):
    def test_total_steps_and_window_closes(self) -> None:
        schedule = HoldProbeSchedule(amplitude=0.5, hold_steps=4, rounds=2)
        self.assertEqual(schedule.total_steps, 2 * 6 * 4)
        steps = torch.arange(8)
        # channel 0 held for steps 0..3, channel 1 for 4..7
        action = schedule.action(steps)
        self.assertTrue(torch.all(action[0:4, 0] == 0.5))
        self.assertTrue(torch.all(action[4:8, 1] == 0.5))
        self.assertTrue(torch.all(action[0:4, 1] == 0.0))
        closes = schedule.is_window_close(steps)
        self.assertEqual(closes.tolist(), [False, False, False, True] * 2)

    def test_rejects_degenerate_parameters(self) -> None:
        for kwargs in ({"hold_steps": 1}, {"rounds": 0}, {"amplitude": 0.0}):
            with self.assertRaises(ValueError):
                HoldProbeSchedule(**kwargs)  # type: ignore[arg-type]


class HoldProbeIdentificationTest(unittest.TestCase):
    def test_absolute_target_and_permutation_identified_clean(self) -> None:
        response = _clean_response()
        generator = torch.Generator().manual_seed(11)
        for _ in range(5):
            contract = _random_absolute_contract(generator)
            adapter = _make_hold_adapter(response, hold_steps=4, rounds=1)
            _drive_hold(adapter, contract, response, control_steps=2, generator=generator)
            recovered = adapter.driver.map_contracts()
            for env in range(_BATCH):
                self.assertEqual(recovered[env].target, "absolute")
                self.assertEqual(recovered[env].permutation, contract.permutation)
                self.assertEqual(recovered[env].sign, contract.sign)
                self.assertEqual(recovered[env].scale, contract.scale)

    def test_absolute_identified_under_noise(self) -> None:
        response = _attenuated_response()
        generator = torch.Generator().manual_seed(23)
        hits = 0
        trials = 6
        for _ in range(trials):
            contract = _random_absolute_contract(generator)
            adapter = _make_hold_adapter(response, hold_steps=4, rounds=2)
            _drive_hold(
                adapter, contract, response, control_steps=2, generator=generator, noise=True
            )
            recovered = adapter.driver.map_contracts()
            if all(
                recovered[env].target == "absolute"
                and recovered[env].permutation == contract.permutation
                for env in range(_BATCH)
            ):
                hits += 1
        # Under attenuation + noise the integrated evidence still identifies the
        # absolute target and permutation on essentially every trial.
        self.assertGreaterEqual(hits, trials - 1)

    def test_delta_contract_still_identified(self) -> None:
        response = _clean_response()
        generator = torch.Generator().manual_seed(31)
        contract = _random_absolute_contract(generator, target="delta")
        adapter = _make_hold_adapter(response, hold_steps=4, rounds=1)
        _drive_hold(adapter, contract, response, control_steps=2, generator=generator)
        recovered = adapter.driver.map_contracts()
        for env in range(_BATCH):
            self.assertEqual(recovered[env].target, "delta")
            self.assertEqual(recovered[env].permutation, contract.permutation)


class HoldProbeParityTest(unittest.TestCase):
    def test_absolute_off_grid_scale_recovered_with_corrector(self) -> None:
        response = _clean_response()
        contract = ActionContract(
            permutation=(2, 0, 1, 4, 5, 3),
            sign=(1, -1, 1, 1, -1, 1),
            scale=(1.1, 0.85, 1.4, 0.55, 1.7, 1.05),  # off the grammar grid
            target="absolute",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        adapter = _make_hold_adapter(
            response, hold_steps=4, rounds=1, scale_correction=True, scale_window=8
        )
        intent, executed = _drive_hold(
            adapter, contract, response, control_steps=40,
            generator=torch.Generator().manual_seed(7),
        )
        recovered = adapter.driver.map_contracts()
        for env in range(_BATCH):
            self.assertEqual(recovered[env].target, "absolute")
            self.assertEqual(recovered[env].permutation, contract.permutation)
        # After the probe locks the discrete contract and the corrector refines the
        # off-grid scale, the executed canonical tracks the intent.
        error = (executed[:, :6] - intent[:, :6]).abs().max().item()
        self.assertLess(error, 5e-2)

    def test_absolute_off_grid_without_corrector_drifts(self) -> None:
        response = _clean_response()
        contract = ActionContract(
            permutation=(2, 0, 1, 4, 5, 3),
            sign=(1, -1, 1, 1, -1, 1),
            scale=(1.1, 0.85, 1.4, 0.55, 1.7, 1.05),
            target="absolute",
            frame="base",
            lag=0,
            gripper_inverted=False,
        )
        adapter = _make_hold_adapter(response, hold_steps=4, rounds=1, scale_correction=False)
        intent, executed = _drive_hold(
            adapter, contract, response, control_steps=40,
            generator=torch.Generator().manual_seed(7),
        )
        # The discrete contract is identified, but the off-grid scale is not on the
        # grid, so absolute control drifts without the corrector.
        recovered = adapter.driver.map_contracts()
        for env in range(_BATCH):
            self.assertEqual(recovered[env].permutation, contract.permutation)
        error = (executed[:, :6] - intent[:, :6]).abs().max().item()
        self.assertGreater(error, 5e-2)


class HoldProbeReproducibilityTest(unittest.TestCase):
    def test_update_without_hold_mask_is_unchanged(self) -> None:
        # A per-step probing adapter (no hold_mask) must produce identical scores to
        # the pre-hold-probe path: the hold accumulators stay untouched.
        response = _clean_response()
        contract = _random_absolute_contract(torch.Generator().manual_seed(3))
        driver = FactorizedGrammarDriver(batch_size=_BATCH, response=response)
        adapter = FactorizedGrammarProbingAdapter(driver, budget=6, amplitude=0.5)
        environment = SyntheticHiddenEnvironment(contract, _BATCH, response)
        generator = torch.Generator().manual_seed(4)
        for _ in range(20):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2.0 - 1.0
            raw = adapter.encode(intent)
            obs = environment.step(raw)
            adapter.observe(raw, obs[:, :6])
        self.assertTrue(torch.all(driver._hold_sum_obs == 0.0))
        self.assertTrue(torch.all(driver._hold_len == 0))


class HoldProbeLeakageGuardTest(unittest.TestCase):
    def test_schedule_and_adapter_take_no_contract(self) -> None:
        for target in (
            HoldProbeSchedule.__init__,
            FactorizedGrammarHoldProbingAdapter.__init__,
        ):
            parameters = inspect.signature(target).parameters
            for forbidden in ("contract", "true_contract", "pool", "scale", "true_scale"):
                self.assertNotIn(forbidden, parameters)


if __name__ == "__main__":
    unittest.main()
