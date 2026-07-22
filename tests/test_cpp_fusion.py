"""Parity tests for the ActionABI C++ evidence-scoring backend.

The C++ cell/pool scorers must reproduce the torch belief exactly: same per-cell
log-evidence (parity target <=1e-6 relative, met at ~1e-13 in float64), the same
belief updates, and the same MAP contract decisions on synthetic sequences. Tests
skip cleanly when the compiled extension has not been built.
"""

from __future__ import annotations

import unittest

import torch

from actionshift.adaptation.cpp_backend import (
    CppCellScorer,
    CppPoolScorer,
    cpp_backend_available,
)
from actionshift.adaptation.factorized_grammar import FactorizedGrammarDriver
from actionshift.adaptation.hypotheses import identity_rotation
from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.transforms import CompleteActionDecoder
from actionshift.contracts.types import ActionContract

_BATCH = 4
_RESPONSE = ResponseModel(
    alpha=(1.0, 0.9, 1.1, 0.8, 1.2, 1.0), sigma=(0.3, 0.4, 0.5, 0.2, 0.6, 0.3)
)


@unittest.skipUnless(cpp_backend_available(), "actionabi_cells extension not built")
class CellScorerParityTest(unittest.TestCase):
    def _drive_and_compare(self, dtype: torch.dtype, tolerance: float) -> float:
        torch_driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=_RESPONSE, dtype=dtype
        )
        cpp_driver = FactorizedGrammarDriver(
            batch_size=_BATCH, response=_RESPONSE, dtype=dtype, cell_scorer=CppCellScorer()
        )
        generator = torch.Generator().manual_seed(20260721)
        max_rel = 0.0
        for _ in range(40):
            raw = torch.rand((_BATCH, 7), generator=generator, dtype=dtype) * 2 - 1
            observed = torch.rand((_BATCH, 6), generator=generator, dtype=dtype) * 2 - 1
            reset = None
            invalid = None
            torch_driver.update(
                raw.clone(), observed.clone(), reset_mask=reset, invalid_mask=invalid
            )
            cpp_driver.update(
                raw.clone(), observed.clone(), reset_mask=reset, invalid_mask=invalid
            )
            a = torch_driver._scores
            b = cpp_driver._scores
            rel = ((a - b).abs() / (a.abs() + 1e-9)).max().item()
            max_rel = max(max_rel, rel)
        torch_map = torch_driver.map_contracts()
        cpp_map = cpp_driver.map_contracts()
        for env in range(_BATCH):
            self.assertEqual(torch_map[env].permutation, cpp_map[env].permutation)
            self.assertEqual(torch_map[env].sign, cpp_map[env].sign)
            self.assertEqual(torch_map[env].scale, cpp_map[env].scale)
            self.assertEqual(torch_map[env].target, cpp_map[env].target)
            self.assertEqual(torch_map[env].lag, cpp_map[env].lag)
        self.assertLess(max_rel, tolerance)
        return max_rel

    def test_float64_scores_match_within_1e6_relative(self) -> None:
        # The headline numeric-parity claim: <=1e-6 relative on the accumulated
        # per-cell evidence across a 40-step sequence spanning every grammar mode.
        max_rel = self._drive_and_compare(torch.float64, tolerance=1e-6)
        self.assertLess(max_rel, 1e-6)

    def test_float32_map_decisions_are_identical(self) -> None:
        # Production dtype: float32 rounding differs per accumulation order but the
        # MAP decisions (the only thing that drives control) are bit-identical.
        self._drive_and_compare(torch.float32, tolerance=1e-3)


@unittest.skipUnless(cpp_backend_available(), "actionabi_cells extension not built")
class SyntheticControlParityTest(unittest.TestCase):
    """Full synthetic hidden-env drive: identical recovery through the C++ backend."""

    def _run(self, contract: ActionContract, cell_scorer: object | None) -> ActionContract:
        driver = FactorizedGrammarDriver(
            batch_size=_BATCH,
            response=ResponseModel(alpha=1.0, sigma=0.02),
            dtype=torch.float64,
            cell_scorer=cell_scorer,  # type: ignore[arg-type]
        )
        decoder = CompleteActionDecoder(contract, batch_size=_BATCH)
        generator = torch.Generator().manual_seed(31)
        for _ in range(48):
            intent = torch.rand((_BATCH, 7), generator=generator, dtype=torch.float64) * 2 - 1
            raw = driver.map_encode(intent)
            rotation = identity_rotation(_BATCH, raw.device, raw.dtype)
            executed = decoder.step(raw, ee_rotation=rotation)
            driver.update(raw, executed[:, :6])
        return driver.map_contracts()[0]

    def test_cpp_backend_recovers_same_contract_as_torch(self) -> None:
        contract = ActionContract(
            permutation=(4, 0, 5, 1, 3, 2),
            sign=(1, -1, -1, 1, 1, -1),
            scale=(0.5, 2.0, 1.25, 0.75, 1.5, 0.6),
            target="delta",
            frame="base",
            lag=1,
            gripper_inverted=False,
        )
        torch_recovered = self._run(contract, None)
        cpp_recovered = self._run(contract, CppCellScorer())
        self.assertEqual(torch_recovered.permutation, cpp_recovered.permutation)
        self.assertEqual(torch_recovered.sign, cpp_recovered.sign)
        self.assertEqual(torch_recovered.scale, cpp_recovered.scale)
        self.assertEqual(torch_recovered.target, cpp_recovered.target)
        self.assertEqual(torch_recovered.lag, cpp_recovered.lag)
        # And it recovers the true contract (sanity, not just self-consistency).
        self.assertEqual(cpp_recovered.permutation, contract.permutation)


@unittest.skipUnless(cpp_backend_available(), "actionabi_cells extension not built")
class PoolScorerParityTest(unittest.TestCase):
    def test_pool_log_likelihood_matches_response_model(self) -> None:
        hypotheses, batch, channels = 9, 6, 6
        generator = torch.Generator().manual_seed(7)
        predicted = torch.rand(
            (hypotheses, batch, channels), generator=generator, dtype=torch.float64
        )
        observed = torch.rand((batch, channels), generator=generator, dtype=torch.float64)
        response = ResponseModel(
            alpha=(1.0, 0.9, 1.1, 0.8, 1.2, 1.0), sigma=(0.3, 0.4, 0.5, 0.2, 0.6, 0.3)
        )
        reference = response.log_likelihood(predicted, observed)
        alpha = observed.new_tensor(response.alpha)
        sigma = observed.new_tensor(response.sigma)
        scorer = CppPoolScorer()
        cpp = scorer.log_likelihood(predicted, observed, alpha, sigma)
        max_rel = (
            (reference - cpp).abs() / (reference.abs() + 1e-9)
        ).max().item()
        self.assertLess(max_rel, 1e-6)


if __name__ == "__main__":
    unittest.main()
