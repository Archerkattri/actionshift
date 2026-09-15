"""Deterministic compound action-contract shift episodes.

This generator separates the public episode schedule a policy may see from
the private contracts used by an evaluator.  It supports static, abrupt,
gradual and recurring changes plus an explicit outside-grammar case.  The
episode object is intentionally simulator-agnostic so benchmark protocol and
split tests can run without ManiSkill.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .sampler import core_composition_product
from .types import ActionContract

ShiftKind = Literal["static", "abrupt", "gradual", "recurring"]
GENERATOR_VERSION = "compound-shift-v1"


def _contract_hash(contract: ActionContract) -> str:
    return hashlib.sha256(contract.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ShiftSegment:
    """Private evaluator segment; never pass ``contract`` to a policy."""

    start: int
    stop: int
    contract: ActionContract

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("segment must satisfy 0 <= start < stop")

    @property
    def contract_sha256(self) -> str:
        return _contract_hash(self.contract)


@dataclass(frozen=True, slots=True)
class CompoundShiftEpisode:
    """A private ground-truth episode with a contract-free public view."""

    episode_id: str
    seed: int
    steps: int
    shift_kind: ShiftKind
    segments: tuple[ShiftSegment, ...]
    outside_grammar: bool = False

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not self.segments:
            raise ValueError("episode must contain at least one segment")
        if self.segments[0].start != 0 or self.segments[-1].stop != self.steps:
            raise ValueError("segments must cover the complete episode")
        if any(
            a.stop != b.start
            for a, b in zip(self.segments, self.segments[1:], strict=False)
        ):
            raise ValueError("segments must be contiguous")

    def contract_at(self, step: int) -> ActionContract:
        """Evaluator-only lookup of the hidden contract at a step."""
        if not 0 <= step < self.steps:
            raise IndexError(step)
        for segment in self.segments:
            if segment.start <= step < segment.stop:
                return segment.contract
        raise RuntimeError("invalid segment coverage")

    def public_manifest(self) -> dict[str, object]:
        """Return only information safe to expose to a policy or runner."""
        return {
            "schema_version": "1.0",
            "generator_version": GENERATOR_VERSION,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "steps": self.steps,
            "shift_kind": self.shift_kind,
            "segment_boundaries": [
                {"start": segment.start, "stop": segment.stop}
                for segment in self.segments
            ],
            "outside_grammar": self.outside_grammar,
        }

    def private_manifest(self) -> dict[str, object]:
        """Return evaluator-only identity without serializing raw semantics by default."""
        return {
            **self.public_manifest(),
            "contract_sha256": [segment.contract_sha256 for segment in self.segments],
        }


def _outside_contract(base: ActionContract) -> ActionContract:
    """Create a contract outside the declared grammar while preserving shape."""
    return ActionContract(
        permutation=base.permutation,
        sign=base.sign,
        scale=tuple(1.25 + 0.05 * i for i in range(len(base.scale))),
        target=base.target,
        frame=base.frame,
        lag=base.lag,
        gripper_inverted=base.gripper_inverted,
    )


def _different(
    pool: list[ActionContract], first: ActionContract, rng: np.random.Generator
) -> ActionContract:
    candidates = [contract for contract in pool if contract != first]
    if not candidates:
        raise ValueError("contract pool must contain at least two distinct contracts")
    return candidates[int(rng.integers(0, len(candidates)))]


def generate_compound_episode(
    kind: str,
    *,
    seed: int,
    steps: int = 64,
    outside_grammar: bool = False,
) -> CompoundShiftEpisode:
    """Generate one deterministic hidden-contract episode.

    ``kind`` controls the change schedule.  The schedule and hidden contract
    identities are derived from ``seed`` but are not included in the public
    observation beyond timing boundaries; an evaluation harness should retain
    the private manifest separately.
    """
    if kind not in {"static", "abrupt", "gradual", "recurring"}:
        raise ValueError("kind must be static, abrupt, gradual, or recurring")
    if steps < 8:
        raise ValueError("steps must be >= 8")
    rng = np.random.default_rng(seed)
    pool = list(core_composition_product())
    if not pool:
        raise ValueError("core contract pool is empty")
    first = pool[int(rng.integers(0, len(pool)))]
    second = _different(pool, first, rng)
    if outside_grammar:
        second = _outside_contract(second)

    if kind == "static":
        spans = [steps]
    elif kind == "abrupt":
        spans = [steps // 2, steps - steps // 2]
    elif kind == "recurring":
        q = steps // 4
        spans = [q, q, q, steps - 3 * q]
    else:
        count = min(8, max(2, steps // 8))
        base, remainder = divmod(steps, count)
        spans = [base + (1 if i < remainder else 0) for i in range(count)]

    contracts = [first if i % 2 == 0 else second for i in range(len(spans))]
    if kind == "static":
        contracts = [first]
    cursor = 0
    segments = []
    for span, contract in zip(spans, contracts, strict=True):
        segments.append(ShiftSegment(cursor, cursor + span, contract))
        cursor += span
    encoded = json.dumps(
        {"version": GENERATOR_VERSION, "kind": kind, "seed": int(seed), "steps": int(steps)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    episode_id = hashlib.sha256(encoded).hexdigest()[:16]
    return CompoundShiftEpisode(
        episode_id=episode_id,
        seed=int(seed),
        steps=int(steps),
        shift_kind=kind,  # type: ignore[arg-type]
        segments=tuple(segments),
        outside_grammar=bool(outside_grammar),
    )


__all__ = [
    "CompoundShiftEpisode",
    "ShiftKind",
    "ShiftSegment",
    "generate_compound_episode",
]
