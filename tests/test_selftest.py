"""Unit + CPU-synthetic tests for the plug-and-verify self-test tool.

The verdict-logic tests are pure (hand-built identification results, no probing).
The identification tests drive the real probe stack against the bit-faithful
synthetic hidden environment on CPU -- no ManiSkill, no GPU.
"""

from __future__ import annotations

import unittest

from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.types import ActionContract
from actionshift.selftest.cli import main
from actionshift.selftest.diffs import (
    POSE_FIELDS,
    describe_field_diff,
    differing_fields,
)
from actionshift.selftest.identify import (
    SyntheticProbeEnvironment,
    identify_contract,
)
from actionshift.selftest.library import (
    default_pool,
    demo_contract,
    identity_contract,
    named_contract,
    resolve_pool,
)
from actionshift.selftest.verdict import (
    EXIT_CODES,
    IdentificationResult,
    decide_verdict,
)


def _identity() -> ActionContract:
    return identity_contract()


def _result(
    identified: ActionContract,
    *,
    confidence: dict[str, float] | None = None,
    fit_ratio: float = 1.0,
) -> IdentificationResult:
    """Hand-build an identification result for pure verdict tests."""
    field_confidence = {field: 1.0 for field in POSE_FIELDS}
    if confidence is not None:
        field_confidence.update(confidence)
    return IdentificationResult(
        map_contract=identified,
        pool_posterior=(1.0,),
        field_confidence=field_confidence,
        fit_ratio=fit_ratio,
        probe_steps=6.0,
        probe_displacement=0.1,
        strategy="entropy",
    )


class VerdictLogicTest(unittest.TestCase):
    def test_pass_when_identified_matches_expected_with_confidence(self) -> None:
        verdict = decide_verdict(_result(_identity()), _identity())
        self.assertEqual(verdict.status, "PASS")
        self.assertEqual(verdict.exit_code, 0)
        self.assertEqual(verdict.diffs, {})

    def test_mismatch_reports_swapped_axes(self) -> None:
        swapped = named_contract("swapped-axes")
        verdict = decide_verdict(_result(swapped), _identity())
        self.assertEqual(verdict.status, "MISMATCH")
        self.assertEqual(verdict.exit_code, 1)
        self.assertIn("permutation", verdict.diffs)
        self.assertEqual(verdict.diffs["permutation"], "channels 0 and 1 SWAPPED")

    def test_mismatch_reports_combined_swap_and_sign_flip(self) -> None:
        verdict = decide_verdict(_result(named_contract("miswired")), _identity())
        self.assertEqual(verdict.status, "MISMATCH")
        self.assertIn("SWAPPED", verdict.reason)
        self.assertIn("sign FLIPPED", verdict.reason)

    def test_inconclusive_when_differing_field_is_low_confidence(self) -> None:
        # A wiring differs on permutation, but the belief did not concentrate on
        # that field: fail-closed -> abstain, never flag a mismatch we do not trust.
        swapped = named_contract("swapped-axes")
        verdict = decide_verdict(
            _result(swapped, confidence={"permutation": 0.4}), _identity()
        )
        self.assertEqual(verdict.status, "INCONCLUSIVE")
        self.assertEqual(verdict.exit_code, 2)
        self.assertIn("permutation", verdict.unresolved)

    def test_pass_requires_every_field_resolved(self) -> None:
        # Identified equals expected, but one field is unresolved: cannot certify.
        verdict = decide_verdict(
            _result(_identity(), confidence={"lag": 0.5}), _identity()
        )
        self.assertEqual(verdict.status, "INCONCLUSIVE")
        self.assertIn("lag", verdict.unresolved)

    def test_misspecification_forces_inconclusive(self) -> None:
        # Even a confidently-identified, matching wiring must abstain when the best
        # hypothesis fits the responses badly (true wiring likely outside the pool).
        verdict = decide_verdict(_result(_identity(), fit_ratio=40.0), _identity())
        self.assertEqual(verdict.status, "INCONCLUSIVE")
        self.assertIn("misspecification", verdict.reason)

    def test_confident_mismatch_survives_an_unresolved_sibling_field(self) -> None:
        swapped = named_contract("swapped-axes")
        verdict = decide_verdict(
            _result(swapped, confidence={"lag": 0.3}), _identity()
        )
        self.assertEqual(verdict.status, "MISMATCH")
        self.assertIn("lag", verdict.unresolved)
        self.assertIn("not resolved", verdict.reason)

    def test_invalid_thresholds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decide_verdict(_result(_identity()), _identity(), confidence_floor=0.0)
        with self.assertRaises(ValueError):
            decide_verdict(_result(_identity()), _identity(), misspec_ratio=0.0)

    def test_exit_codes_are_distinct_and_meaningful(self) -> None:
        self.assertEqual(EXIT_CODES, {"PASS": 0, "MISMATCH": 1, "INCONCLUSIVE": 2})


class DiffDescriptionTest(unittest.TestCase):
    def test_each_field_has_a_readable_sentence(self) -> None:
        identity = _identity()
        self.assertEqual(
            describe_field_diff("permutation", named_contract("swapped-axes"), identity),
            "channels 0 and 1 SWAPPED",
        )
        self.assertEqual(
            describe_field_diff("sign", named_contract("sign-flip"), identity),
            "channel 3 sign FLIPPED",
        )
        self.assertIn(
            "scale",
            describe_field_diff("scale", named_contract("scaled"), identity),
        )
        self.assertIn(
            "lag", describe_field_diff("lag", named_contract("lagged"), identity)
        )

    def test_gripper_is_never_in_the_pose_field_diffs(self) -> None:
        inverted = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1,) * 6,
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=0,
            gripper_inverted=True,
        )
        self.assertNotIn("gripper_inverted", POSE_FIELDS)
        self.assertEqual(differing_fields(inverted, _identity()), {})


class LibraryTest(unittest.TestCase):
    def test_default_pool_members_are_unique(self) -> None:
        pool = default_pool()
        serialized = [contract.to_json() for contract in pool]
        self.assertEqual(len(serialized), len(set(serialized)))

    def test_resolve_pool_adds_expected_without_duplicating(self) -> None:
        expected = _identity()  # already in the pool
        self.assertEqual(len(resolve_pool(expected)), len(default_pool()))
        novel = ActionContract(
            permutation=(0, 1, 2, 3, 4, 5),
            sign=(1,) * 6,
            scale=(1.0,) * 6,
            target="delta",
            frame="base",
            lag=7,
            gripper_inverted=False,
        )
        self.assertEqual(len(resolve_pool(novel)), len(default_pool()) + 1)

    def test_unknown_demo_rejected(self) -> None:
        with self.assertRaises(ValueError):
            demo_contract("does-not-exist")


class SyntheticIdentificationTest(unittest.TestCase):
    """CPU-synthetic smoke: the real probe stack against the bit-faithful env."""

    def _response(self) -> ResponseModel:
        return ResponseModel(alpha=1.0, sigma=0.05)

    def _identify(self, contract: ActionContract) -> IdentificationResult:
        response = self._response()
        environment = SyntheticProbeEnvironment(
            contract, batch_size=8, response=response, seed=20260720
        )
        try:
            return identify_contract(
                environment,
                resolve_pool(_identity()),
                response,
                strategy="entropy",
                budget=6,
                seed=20260720,
            )
        finally:
            environment.close()

    def test_probe_identifies_swapped_axes_as_mismatch(self) -> None:
        result = self._identify(named_contract("swapped-axes"))
        self.assertEqual(result.map_contract, named_contract("swapped-axes"))
        self.assertGreaterEqual(result.field_confidence["permutation"], 0.9)
        self.assertLess(result.fit_ratio, 4.0)
        verdict = decide_verdict(result, _identity())
        self.assertEqual(verdict.status, "MISMATCH")

    def test_probe_passes_a_correctly_wired_robot(self) -> None:
        result = self._identify(_identity())
        verdict = decide_verdict(result, _identity())
        self.assertEqual(verdict.status, "PASS")

    def test_out_of_pool_wiring_is_flagged_by_misspecification(self) -> None:
        result = self._identify(demo_contract("unmodeled"))
        self.assertGreater(result.fit_ratio, 4.0)
        verdict = decide_verdict(result, _identity())
        self.assertEqual(verdict.status, "INCONCLUSIVE")

    def test_fixed_strategy_also_identifies(self) -> None:
        response = self._response()
        environment = SyntheticProbeEnvironment(
            named_contract("sign-flip"), batch_size=8, response=response, seed=1
        )
        try:
            result = identify_contract(
                environment,
                resolve_pool(_identity()),
                response,
                strategy="fixed",
                budget=6,
                seed=1,
            )
        finally:
            environment.close()
        self.assertEqual(result.map_contract, named_contract("sign-flip"))


class CliEndToEndTest(unittest.TestCase):
    def test_demo_exit_codes(self) -> None:
        self.assertEqual(main(["--demo", "identity"]), 0)
        self.assertEqual(main(["--demo", "miswired"]), 1)
        self.assertEqual(main(["--demo", "unmodeled"]), 2)

    def test_json_mode_runs(self) -> None:
        self.assertEqual(main(["--demo", "swapped-axes", "--json"]), 1)


if __name__ == "__main__":
    unittest.main()
