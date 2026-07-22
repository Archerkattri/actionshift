"""Recurrent episode-length adapter tests: running features, learning, timing.

Mirrors ``test_adaptation_training.py`` style: a synthetic hidden environment
supplies full-episode sequences, and the running least-squares accumulator is
checked bit-for-bit against ``training.history_features`` so the recurrent method
shares one sufficient statistic and one equivariant head with the OSI baseline.
"""

from __future__ import annotations

import unittest

import torch

from actionshift.adaptation.recurrent_adapter import (
    RecurrentOsiAdapter,
    RecurrentOsiRegressor,
    RunningLagFeatures,
    _cells_from_features,
)
from actionshift.adaptation.training import (
    _cell_tensor,
    contract_targets,
    decode_prediction,
    history_features,
    sample_training_contract,
)
from actionshift.contracts.types import ActionContract
from tests.test_adaptation_stage1 import SyntheticHiddenEnvironment, _true_contract

_BATCH = 4
_EPISODE = 45


def _rollout(
    contract: ActionContract, batch: int, steps: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-episode random-excitation rollout: (steps, batch, 7)/(steps, batch, 6)."""
    environment = SyntheticHiddenEnvironment(contract, batch)
    raws, responses = [], []
    for _ in range(steps):
        raw = torch.rand((batch, 7), generator=generator) * 2 - 1
        executed = environment.step(raw)
        raws.append(raw)
        responses.append(executed[:, :6])
    return torch.stack(raws), torch.stack(responses)


def _feature_sequence(
    raws: torch.Tensor, responses: torch.Tensor, *, min_samples: int
) -> torch.Tensor:
    accumulator = RunningLagFeatures(raws.shape[1], min_samples=min_samples)
    return torch.stack(
        [accumulator.push(raws[t], responses[t]) for t in range(raws.shape[0])]
    )


class RunningFeatureTest(unittest.TestCase):
    def test_running_features_match_windowed_history_features(self) -> None:
        # Fed the same steps as a fixed window, the running accumulator must
        # reproduce history_features exactly (same lagged least-squares maps).
        generator = torch.Generator().manual_seed(5)
        window = 20
        raws, responses = _rollout(_true_contract(), _BATCH, window, generator)
        features = _feature_sequence(raws, responses, min_samples=12)[-1]
        histories = torch.cat(
            (raws.transpose(0, 1), responses.transpose(0, 1)), dim=-1
        )
        reference = history_features(histories)
        torch.testing.assert_close(features, reference, atol=1e-4, rtol=1e-3)

    def test_cells_from_features_match_cell_tensor(self) -> None:
        generator = torch.Generator().manual_seed(6)
        raws, responses = _rollout(_true_contract(), _BATCH, 20, generator)
        histories = torch.cat(
            (raws.transpose(0, 1), responses.transpose(0, 1)), dim=-1
        )
        torch.testing.assert_close(
            _cells_from_features(history_features(histories)),
            _cell_tensor(histories),
        )

    def test_evidence_accumulates_and_resets(self) -> None:
        generator = torch.Generator().manual_seed(7)
        accumulator = RunningLagFeatures(_BATCH, min_samples=6)
        raws, responses = _rollout(_true_contract(), _BATCH, 30, generator)
        early = late = None
        for t in range(30):
            features = accumulator.push(raws[t], responses[t])
            if t == 8:
                early = features.norm().item()
            if t == 29:
                late = features.norm().item()
        assert early is not None and late is not None
        self.assertGreater(late, early, "the running maps must sharpen over the episode")
        accumulator.reset()
        zeroed = accumulator.push(raws[0], responses[0])
        self.assertLess(zeroed.norm().item(), late, "reset must clear accumulated maps")


class RecurrentLearningTest(unittest.TestCase):
    def test_identification_improves_with_steps_observed(self) -> None:
        generator = torch.Generator().manual_seed(20260721)
        train_contracts = [
            sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
            for _ in range(64)
        ]
        raws_list, responses_list, labels = [], [], []
        for contract in train_contracts:
            roll_raw, roll_response = _rollout(contract, _BATCH, _EPISODE, generator)
            raws_list.append(roll_raw)
            responses_list.append(roll_response)
            labels.extend([contract] * _BATCH)
        raws = torch.cat(raws_list, dim=1)
        responses = torch.cat(responses_list, dim=1)
        feature_sequence = _feature_sequence(raws, responses, min_samples=6)

        model = RecurrentOsiRegressor(hidden=64, gru_hidden=64)
        from actionshift.adaptation.recurrent_adapter import recurrent_loss

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = last = 0.0
        for epoch in range(40):
            predictions = model(feature_sequence)
            loss = recurrent_loss(predictions, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            if epoch == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        self.assertLess(last, first * 0.6, "deep-supervised training must reduce the loss")

        held = [
            sample_training_contract(generator, excluded_hashes=frozenset(), max_lag=0)
            for _ in range(16)
        ]
        held_raws, held_responses, held_labels = [], [], []
        for contract in held:
            roll_raw, roll_response = _rollout(contract, _BATCH, _EPISODE, generator)
            held_raws.append(roll_raw)
            held_responses.append(roll_response)
            held_labels.extend([contract] * _BATCH)
        held_sequence = _feature_sequence(
            torch.cat(held_raws, dim=1), torch.cat(held_responses, dim=1), min_samples=6
        )
        with torch.no_grad():
            held_predictions = model(held_sequence)

        def permutation_accuracy(step_index: int) -> float:
            correct = total = 0
            for column, contract in enumerate(held_labels):
                single = {
                    key: value[step_index, column]
                    for key, value in held_predictions.items()
                }
                estimate = decode_prediction(single)
                correct += sum(
                    a == b
                    for a, b in zip(
                        estimate.permutation, contract.permutation, strict=True
                    )
                )
                total += 6
            return correct / total

        early = permutation_accuracy(4)
        late = permutation_accuracy(_EPISODE - 1)
        self.assertGreater(late, 0.6, "episode-length identification must succeed")
        self.assertGreaterEqual(
            late, early, "more observed steps must not hurt identification"
        )


class RecurrentAdapterTimingTest(unittest.TestCase):
    def test_warmup_pass_through_then_adapts(self) -> None:
        torch.manual_seed(0)
        model = RecurrentOsiRegressor(hidden=16, gru_hidden=16)
        adapter = RecurrentOsiAdapter(model, batch_size=_BATCH, warmup=8)
        action = torch.rand((_BATCH, 7))
        # No observations yet -> pure pass-through.
        torch.testing.assert_close(adapter.encode(action), action)

    def test_leakage_guard_and_boundary_reset(self) -> None:
        import inspect

        parameters = inspect.signature(RecurrentOsiAdapter.__init__).parameters
        self.assertNotIn("true_contract", parameters)
        self.assertNotIn("contract", parameters)

        torch.manual_seed(1)
        model = RecurrentOsiRegressor(hidden=16, gru_hidden=16)
        adapter = RecurrentOsiAdapter(model, batch_size=_BATCH, warmup=4)
        generator = torch.Generator().manual_seed(2)
        environment = SyntheticHiddenEnvironment(_true_contract(), _BATCH)
        for _ in range(10):
            intent = torch.rand((_BATCH, 7), generator=generator) * 2 - 1
            raw = adapter.encode(intent)
            executed = environment.step(raw)
            adapter.observe(raw, executed[:, :6])
        # A boundary on env 0 must clear its state back to warmup pass-through.
        boundary = torch.tensor([True, False, False, False])
        intent = torch.rand((_BATCH, 7), generator=generator) * 2 - 1
        raw = adapter.encode(intent)
        executed = environment.step(raw)
        adapter.observe(raw, executed[:, :6], invalid_mask=boundary)
        self.assertEqual(int(adapter._filled[0]), 0)
        self.assertIsNone(adapter._estimates[0])
        self.assertGreater(int(adapter._filled[1]), 0)


class TargetBroadcastTest(unittest.TestCase):
    def test_contract_targets_are_reused(self) -> None:
        contract = _true_contract()
        targets = contract_targets(contract)
        self.assertEqual(targets["permutation"].tolist(), list(contract.permutation))


if __name__ == "__main__":
    unittest.main()
