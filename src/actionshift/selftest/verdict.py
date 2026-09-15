"""Verdict logic for the plug-and-verify self-test (pure, fail-closed).

Given an :class:`IdentificationResult` (posterior over the declared pool, per-field
confidence, and a misspecification statistic) and the wiring the policy *expects*,
:func:`decide_verdict` returns exactly one of:

- ``PASS`` -- every observable field is resolved with high confidence *and* equals
  the expected wiring;
- ``MISMATCH`` -- at least one observable field is resolved with high confidence
  *and* differs from the expected wiring (the exact diffs are reported);
- ``INCONCLUSIVE`` -- the belief did not concentrate (low posterior on some field)
  or the best hypothesis fits the responses poorly (misspecification: the true
  wiring is likely outside the declared pool).

The design is fail-closed: it certifies PASS only when it can, flags MISMATCH only
on evidence it trusts, and otherwise abstains rather than guessing. This module has
no torch dependency so the decision is trivially unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from actionshift.contracts.types import ActionContract
from actionshift.selftest.diffs import POSE_FIELDS, differing_fields

Status = Literal["PASS", "MISMATCH", "INCONCLUSIVE"]

EXIT_CODES: dict[Status, int] = {"PASS": 0, "MISMATCH": 1, "INCONCLUSIVE": 2}

# Defaults. The confidence floor is the marginal posterior mass an observable
# field must carry before the tool trusts it. The misspecification ratio is the
# multiple of the calibrated noise scale (chi-square per degree of freedom) above
# which the best hypothesis is judged too poor a fit to trust -- a signal the true
# wiring is outside the declared pool.
DEFAULT_CONFIDENCE_FLOOR = 0.9
DEFAULT_MISSPEC_RATIO = 4.0


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """The output of the probe phase, before a verdict is rendered."""

    map_contract: ActionContract
    pool_posterior: tuple[float, ...]
    field_confidence: dict[str, float]
    fit_ratio: float
    probe_steps: float
    probe_displacement: float
    strategy: str
    # The current self-test uses identity-frame evidence. A future probe backend
    # may opt in only after supplying rotation-bearing observations.
    frame_capable: bool = False

    @property
    def map_posterior(self) -> float:
        return max(self.pool_posterior) if self.pool_posterior else 0.0


@dataclass(frozen=True, slots=True)
class SelfTestVerdict:
    """A rendered verdict with everything needed to explain it."""

    status: Status
    reason: str
    identified: ActionContract
    expected: ActionContract
    field_confidence: dict[str, float]
    diffs: dict[str, str]
    unresolved: tuple[str, ...]
    fit_ratio: float
    exit_code: int


def decide_verdict(
    result: IdentificationResult,
    expected: ActionContract,
    *,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    misspec_ratio: float = DEFAULT_MISSPEC_RATIO,
) -> SelfTestVerdict:
    """Render a fail-closed verdict from an identification result."""
    if not math.isfinite(confidence_floor) or not 0.0 < confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in (0, 1]")
    if not math.isfinite(misspec_ratio) or misspec_ratio <= 0.0:
        raise ValueError("misspec_ratio must be positive")

    identified = result.map_contract
    confidence = result.field_confidence

    def build(
        status: Status, reason: str, diffs: dict[str, str], unresolved: tuple[str, ...]
    ) -> SelfTestVerdict:
        return SelfTestVerdict(
            status=status,
            reason=reason,
            identified=identified,
            expected=expected,
            field_confidence=confidence,
            diffs=diffs,
            unresolved=unresolved,
            fit_ratio=result.fit_ratio,
            exit_code=EXIT_CODES[status],
        )

    # Validate all evidence before using comparisons to render a verdict. In
    # particular, NaN makes both ``>`` and ``<`` false and could otherwise turn
    # an invalid payload into a false PASS. Invalid evidence is an abstention:
    # the self-test has no basis for certifying either wiring.
    invalid_evidence: list[str] = []
    fit_ratio = result.fit_ratio
    try:
        fit_value = float(fit_ratio)
    except (TypeError, ValueError):
        fit_value = math.nan
    if not math.isfinite(fit_value) or fit_value < 0.0:
        invalid_evidence.append(
            f"fit_ratio={fit_ratio!r} (must be finite and nonnegative)"
        )

    if not result.pool_posterior:
        invalid_evidence.append("pool_posterior is empty")
    else:
        posterior_values: list[float] = []
        for index, posterior in enumerate(result.pool_posterior):
            try:
                posterior_value = float(posterior)
            except (TypeError, ValueError):
                posterior_value = math.nan
            if not math.isfinite(posterior_value) or not 0.0 <= posterior_value <= 1.0:
                invalid_evidence.append(
                    f"pool_posterior[{index}]={posterior!r} "
                    "(must be finite in [0, 1])"
                )
            posterior_values.append(posterior_value)
        posterior_total = sum(posterior_values)
        if (
            not invalid_evidence
            and not math.isclose(posterior_total, 1.0, rel_tol=1e-5, abs_tol=1e-6)
        ):
            invalid_evidence.append(
                f"pool_posterior sums to {posterior_total:g} (must sum to 1)"
            )

    invalid_confidence_fields: list[str] = []
    for field, confidence_value in result.field_confidence.items():
        try:
            confidence_number = float(confidence_value)
        except (TypeError, ValueError):
            confidence_number = math.nan
        if not math.isfinite(confidence_number) or not 0.0 <= confidence_number <= 1.0:
            invalid_confidence_fields.append(field)
            invalid_evidence.append(
                f"field_confidence[{field!r}]={confidence_value!r} "
                "(must be finite in [0, 1])"
            )

    if invalid_evidence:
        unresolved = tuple(
            field for field in POSE_FIELDS if field in invalid_confidence_fields
        ) or tuple(POSE_FIELDS)
        return build(
            "INCONCLUSIVE",
            "invalid or nonfinite evidence: "
            + "; ".join(invalid_evidence)
            + ". Abstaining.",
            {},
            unresolved,
        )

    # Tool-frame decoding is only verifiable when the probe carries the live
    # end-effector rotation. The existing synthetic and real self-test paths do
    # not provide that capability, so never certify a tool-frame request here.
    if (
        (expected.frame == "tool" or identified.frame == "tool")
        and not result.frame_capable
    ):
        return build(
            "INCONCLUSIVE",
            "unsupported frame verification: tool-frame evidence requires a "
            "rotation-bearing probe capability; this self-test has none. Abstaining.",
            {},
            ("frame",),
        )

    # Fail-closed guard 1: misspecification. Even the best pool hypothesis fits the
    # observed responses poorly, so the true wiring is likely unmodeled. Never
    # convert a bad fit into a confident PASS or MISMATCH.
    if fit_value > misspec_ratio:
        reason = (
            f"response misspecification: the best hypothesis fits the probe responses "
            f"at {result.fit_ratio:.1f}x the calibrated noise scale "
            f"(threshold {misspec_ratio:.1f}x) -- the true wiring is likely outside "
            f"the declared pool. Abstaining."
        )
        return build("INCONCLUSIVE", reason, {}, tuple(POSE_FIELDS))

    unresolved = tuple(
        field
        for field in POSE_FIELDS
        if confidence.get(field, 0.0) < confidence_floor
    )
    all_diffs = differing_fields(identified, expected, fields=POSE_FIELDS)
    confident_diffs = {
        field: message
        for field, message in all_diffs.items()
        if confidence.get(field, 0.0) >= confidence_floor
    }

    # A confidently-resolved field disagrees with the expected wiring: MISMATCH,
    # regardless of whether other (unobservable) fields stayed ambiguous.
    if confident_diffs:
        reason = "; ".join(confident_diffs.values())
        if unresolved:
            reason += (
                f" (note: field(s) {list(unresolved)} were not resolved "
                f"and are not part of this verdict)"
            )
        return build("MISMATCH", reason, confident_diffs, unresolved)

    # No confident disagreement. If any observable field failed to resolve we
    # cannot certify PASS -- abstain rather than guess.
    if unresolved:
        weakest = min(
            (confidence.get(field, 0.0) for field in unresolved), default=0.0
        )
        reason = (
            f"insufficient posterior mass to resolve field(s) {list(unresolved)} "
            f"(weakest {weakest:.2f} < floor {confidence_floor:.2f}). Abstaining "
            f"rather than guessing."
        )
        return build("INCONCLUSIVE", reason, {}, unresolved)

    # Every observable field resolved with confidence and equals the expected
    # wiring. The gripper direction is out of scope and reported separately.
    reason = (
        "every observable field resolved with high confidence and matches the "
        "expected wiring"
    )
    return build("PASS", reason, {}, ())
