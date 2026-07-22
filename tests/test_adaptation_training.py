"""Stage 3 foundation tests: sampler disjointness, OSI learning, adapter use."""

from __future__ import annotations

import unittest

import torch

from actionshift.adaptation.training import (
    HistoryWindow,
    OsiAdapter,
    OsiRegressor,
    collect_windows,
    decode_prediction,
    sample_training_contract,
    train_osi,
)
from actionshift.contracts.splits import contract_hash
from actionshift.contracts.types import ActionContract
from tests.test_adaptation_stage1 import SyntheticHiddenEnvironment, _true_contract

_WINDOW = 14
_BATCH = 4


def _collect(
    contracts: list[ActionContract], generator: torch.Generator
) -> list[HistoryWindow]:
    samples: list[HistoryWindow] = []
    for contract in contracts:
        environment = SyntheticHiddenEnvironment(contract, _BATCH)
        samples.extend(
            collect_windows(
                lambda raw, env=environment: env.step(raw)[:, :6],
                contract=contract,
                batch_size=_BATCH,
                window=_WINDOW,
                windows_per_contract=2,
                action_source=lambda: torch.rand((_BATCH, 7), generator=generator) * 2 - 1,
            )
        )
    return samples


class SamplerTest(unittest.TestCase):
    def test_respects_exclusions_and_ranges(self) -> None:
        excluded = frozenset({contract_hash(_true_contract())})
        generator = torch.Generator().manual_seed(1)
        for _ in range(50):
            contract = sample_training_contract(
                generator, excluded_hashes=excluded, max_lag=2
            )
            self.assertNotIn(contract_hash(contract), excluded)
            self.assertEqual(sorted(contract.permutation), list(range(6)))
            self.assertTrue(all(0.5 <= s <= 2.0 for s in contract.scale))
            self.assertLessEqual(contract.lag, 2)


class OsiLearningTest(unittest.TestCase):
    def test_learns_identification_and_adapter_beats_no_adapt(self) -> None:
        generator = torch.Generator().manual_seed(20260720)
        train_contracts = [
            sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
            for _ in range(80)
        ]
        samples = _collect(train_contracts, generator)
        model = OsiRegressor(window=_WINDOW, hidden=64)
        losses = train_osi(model, samples, epochs=25, batch_size=64, seed=3)
        self.assertLess(losses[-1], losses[0] * 0.5, "training must reduce the loss")

        held_out = [
            sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
            for _ in range(10)
        ]
        held_samples = _collect(held_out, generator)
        histories = torch.stack([sample.history for sample in held_samples])
        with torch.no_grad():
            predictions = model(histories)
        sign_correct = permutation_correct = total = 0
        for index, sample in enumerate(held_samples):
            single = {key: value[index] for key, value in predictions.items()}
            estimate = decode_prediction(single)
            for a, b in zip(estimate.sign, sample.contract.sign, strict=True):
                sign_correct += int(a == b)
            for a, b in zip(
                estimate.permutation, sample.contract.permutation, strict=True
            ):
                permutation_correct += int(a == b)
            total += 6
        self.assertGreater(sign_correct / total, 0.8)
        self.assertGreater(permutation_correct / total, 0.5)

        target = sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
        environment = SyntheticHiddenEnvironment(target, _BATCH)
        adapter = OsiAdapter(model, batch_size=_BATCH)
        adapted_errors: list[float] = []
        for step in range(24):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2 - 1
            raw = adapter.encode(intent)
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
            if step >= _WINDOW + 2:
                adapted_errors.append(float((executed[:, :6] - intent[:, :6]).abs().mean()))
        no_adapt_env = SyntheticHiddenEnvironment(target, _BATCH)
        no_adapt_errors: list[float] = []
        for step in range(24):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2 - 1
            executed = no_adapt_env.step(intent)
            if step >= _WINDOW + 2:
                no_adapt_errors.append(float((executed[:, :6] - intent[:, :6]).abs().mean()))
        adapted = sum(adapted_errors) / len(adapted_errors)
        unadapted = sum(no_adapt_errors) / len(no_adapt_errors)
        self.assertLess(
            adapted, unadapted * 0.6, f"adapted {adapted:.3f} vs no-adapt {unadapted:.3f}"
        )


if __name__ == "__main__":
    unittest.main()
