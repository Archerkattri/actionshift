"""Probe-augmented learned identification tests.

Two things are checked. (1) The eval adapter is protocol-correct and leakage-safe:
it spends exactly ``budget`` steps sending the shared fixed basis pulses, then
adapts, and resets its whole state on an auto-reset boundary. (2) The scientific
lever: on the same synthetic hidden contract, the fixed probe schedule drives the
running least-squares map far closer to the ground-truth sign*scale structure than
weak (policy-like) excitation does, and a model trained on probe-excited sequences
identifies the held-out permutation. This is the recurrent negative's crux in
miniature -- the wall was excitation, not architecture.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from actionshift.adaptation.probe_osi import ProbeOsiAdapter
from actionshift.adaptation.probes import fixed_probe_pulse
from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiRegressor,
    RunningLagFeatures,
    _cells_from_features,
    recurrent_loss,
)
from actionshift.adaptation.training import (
    decode_prediction,
    sample_training_contract,
)
from actionshift.contracts.types import ActionContract
from tests.test_adaptation_stage1 import SyntheticHiddenEnvironment, _true_contract

_BATCH = 4
_EPISODE = 45
_BUDGET = 6
_AMPLITUDE = 0.5


# The real ActionShift response model is weak (per-step SNR ~1); the synthetic
# decoder is noiseless, so a matched observation-noise term is injected to make the
# excitation-quality difference (the whole point) visible: at fixed noise, probe
# pulses (amplitude 0.5) carry ~6x the signal of policy-like weak actions (~0.08).
_RESPONSE_NOISE = 0.05


def _probe_then_weak_rollout(
    contract: ActionContract,
    batch: int,
    steps: int,
    generator: torch.Generator,
    *,
    weak_amplitude: float,
    noise: float = _RESPONSE_NOISE,
    budget: int = _BUDGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed basis pulses for the probe budget, then weak (policy-like) excitation."""
    environment = SyntheticHiddenEnvironment(contract, batch)
    raws, responses = [], []
    for step in range(steps):
        if step < budget:
            index = torch.full((batch,), step, dtype=torch.long)
            raw = fixed_probe_pulse(index, amplitude=_AMPLITUDE)
        else:
            raw = (torch.rand((batch, 7), generator=generator) * 2 - 1) * weak_amplitude
            raw[:, 6] = 0.0
        executed = environment.step(raw)[:, :6]
        executed = executed + noise * torch.randn(
            (batch, 6), generator=generator
        )
        raws.append(raw)
        responses.append(executed)
    return torch.stack(raws), torch.stack(responses)


def _weak_rollout(
    contract: ActionContract, batch: int, steps: int, generator: torch.Generator,
    *, weak_amplitude: float, noise: float = _RESPONSE_NOISE,
) -> tuple[torch.Tensor, torch.Tensor]:
    environment = SyntheticHiddenEnvironment(contract, batch)
    raws, responses = [], []
    for _ in range(steps):
        raw = (torch.rand((batch, 7), generator=generator) * 2 - 1) * weak_amplitude
        raw[:, 6] = 0.0
        executed = environment.step(raw)[:, :6]
        executed = executed + noise * torch.randn((batch, 6), generator=generator)
        raws.append(raw)
        responses.append(executed)
    return torch.stack(raws), torch.stack(responses)


def _feature_sequence(
    raws: torch.Tensor, responses: torch.Tensor, *, min_samples: int
) -> torch.Tensor:
    accumulator = RunningLagFeatures(raws.shape[1], min_samples=min_samples)
    return torch.stack(
        [accumulator.push(raws[t], responses[t]) for t in range(raws.shape[0])]
    )


class ProbeExcitationLeverTest(unittest.TestCase):
    def test_probe_recovers_the_map_far_better_than_weak_excitation(self) -> None:
        # Downstream, only the permutation ARGMAX of the recovered lag-0 map
        # matters. Under matched observation noise, strong basis pulses recover it
        # exactly; weak (policy-like) excitation cannot -- averaged over several
        # contracts, the argmax accuracy gap is the excitation lever.
        generator = torch.Generator().manual_seed(11)
        noise = 0.12
        probe_correct = weak_correct = total = 0
        for seed in range(8):
            base = sample_training_contract(
                torch.Generator().manual_seed(seed), excluded_hashes=frozenset(),
                max_lag=0,
            )
            contract = ActionContract(
                permutation=base.permutation, sign=base.sign, scale=base.scale,
                target="delta", frame="base", lag=0, gripper_inverted=False,
            )
            probe_raw, probe_resp = _probe_then_weak_rollout(
                contract, 1, _EPISODE, generator, weak_amplitude=0.05,
                budget=_EPISODE, noise=noise,
            )
            weak_raw, weak_resp = _weak_rollout(
                contract, 1, _EPISODE, generator, weak_amplitude=0.05, noise=noise
            )
            probe_map = _cells_from_features(
                _feature_sequence(probe_raw, probe_resp, min_samples=_BUDGET)[-1]
            )[0, :, :, 0]
            weak_map = _cells_from_features(
                _feature_sequence(weak_raw, weak_resp, min_samples=_BUDGET)[-1]
            )[0, :, :, 0]
            for i in range(6):
                probe_correct += int(int(probe_map[i].abs().argmax()) == contract.permutation[i])
                weak_correct += int(int(weak_map[i].abs().argmax()) == contract.permutation[i])
                total += 1
        self.assertGreater(probe_correct / total, 0.95, "probes recover permutation")
        self.assertGreater(
            probe_correct / total,
            weak_correct / total + 0.25,
            "probes must beat weak excitation clearly",
        )


class ProbeAdapterProtocolTest(unittest.TestCase):
    def test_leakage_guard(self) -> None:
        parameters = inspect.signature(ProbeOsiAdapter.__init__).parameters
        self.assertNotIn("true_contract", parameters)
        self.assertNotIn("contract", parameters)
        self.assertNotIn("pool", parameters)

    def test_probe_phase_sends_basis_pulses_then_adapts(self) -> None:
        torch.manual_seed(0)
        model = RecurrentOsiRegressor(hidden=16, gru_hidden=16)
        adapter = ProbeOsiAdapter(
            model, batch_size=_BATCH, budget=3, warmup=3, min_samples=2
        )
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        intent = torch.rand((_BATCH, 7)) * 2 - 1
        masks = []
        for step in range(6):
            raw = adapter.encode(intent)
            if step < 3:
                # Probe steps send the shared fixed basis pulse, not the intent.
                expected = fixed_probe_pulse(
                    torch.full((_BATCH,), step, dtype=torch.long), amplitude=0.5
                )
                torch.testing.assert_close(raw, expected)
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
            assert adapter.last_probe_mask is not None
            masks.append(bool(adapter.last_probe_mask.any()))
        self.assertEqual(masks, [True, True, True, False, False, False])

    def test_boundary_reset_clears_state(self) -> None:
        torch.manual_seed(1)
        model = RecurrentOsiRegressor(hidden=16, gru_hidden=16)
        adapter = ProbeOsiAdapter(
            model, batch_size=_BATCH, budget=3, warmup=4, min_samples=2
        )
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        for _ in range(9):
            intent = torch.rand((_BATCH, 7)) * 2 - 1
            raw = adapter.encode(intent)
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
        boundary = torch.tensor([True, False, False, False])
        intent = torch.rand((_BATCH, 7)) * 2 - 1
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6], invalid_mask=boundary)
        self.assertEqual(int(adapter._steps[0]), 0)
        self.assertEqual(int(adapter._filled[0]), 0)
        self.assertIsNone(adapter._estimates[0])
        self.assertGreater(int(adapter._filled[1]), 0)
        # After reset env 0 must re-enter the probe phase.
        raw = adapter.encode(torch.rand((_BATCH, 7)) * 2 - 1)
        assert adapter.last_probe_mask is not None
        self.assertTrue(bool(adapter.last_probe_mask[0]))


class ProbeLearningTest(unittest.TestCase):
    def test_trained_model_identifies_permutation_from_probes(self) -> None:
        generator = torch.Generator().manual_seed(20260721)
        train = [
            sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
            for _ in range(48)
        ]
        raws_list, responses_list, labels = [], [], []
        for contract in train:
            roll_raw, roll_resp = _probe_then_weak_rollout(
                contract, _BATCH, _EPISODE, generator, weak_amplitude=0.1
            )
            raws_list.append(roll_raw)
            responses_list.append(roll_resp)
            labels.extend([contract] * _BATCH)
        features = _feature_sequence(
            torch.cat(raws_list, dim=1), torch.cat(responses_list, dim=1),
            min_samples=_BUDGET,
        )
        model = RecurrentOsiRegressor(hidden=64, gru_hidden=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(40):
            loss = recurrent_loss(model(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        held = [
            sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
            for _ in range(16)
        ]
        held_raw, held_resp, held_labels = [], [], []
        for contract in held:
            roll_raw, roll_resp = _probe_then_weak_rollout(
                contract, _BATCH, _EPISODE, generator, weak_amplitude=0.1
            )
            held_raw.append(roll_raw)
            held_resp.append(roll_resp)
            held_labels.extend([contract] * _BATCH)
        held_features = _feature_sequence(
            torch.cat(held_raw, dim=1), torch.cat(held_resp, dim=1), min_samples=_BUDGET
        )
        with torch.no_grad():
            predictions = model(held_features)
        correct = total = 0
        for column, contract in enumerate(held_labels):
            single = {key: value[-1, column] for key, value in predictions.items()}
            estimate = decode_prediction(single)
            correct += sum(
                a == b
                for a, b in zip(estimate.permutation, contract.permutation, strict=True)
            )
            total += 6
        self.assertGreater(
            correct / total, 0.75, "probe-excited identification must succeed"
        )


if __name__ == "__main__":
    unittest.main()
