"""Validated, immutable representation of a hidden action-interface contract."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

Target = Literal["delta", "absolute"]
Frame = Literal["base", "tool"]


@dataclass(frozen=True, slots=True)
class ActionContract:
    """One locally invertible action-interface contract.

    Component fields describe pose channels only. The gripper is a named channel
    controlled separately by ``gripper_inverted``.
    """

    permutation: tuple[int, ...]
    sign: tuple[int, ...]
    scale: tuple[float, ...]
    target: Target
    frame: Frame
    lag: int
    gripper_inverted: bool

    def __post_init__(self) -> None:
        dimension = len(self.permutation)
        if dimension == 0:
            raise ValueError("contract must have at least one pose component")
        if len(self.sign) != dimension or len(self.scale) != dimension:
            raise ValueError("contract component dimensions must match")
        if sorted(self.permutation) != list(range(dimension)):
            raise ValueError("permutation must be a bijection over component indices")
        if any(value not in (-1, 1) for value in self.sign):
            raise ValueError("sign values must be -1 or 1")
        if any(not math.isfinite(value) or value <= 0 for value in self.scale):
            raise ValueError("scale values must be finite and positive")
        if isinstance(self.lag, bool) or not isinstance(self.lag, int):
            raise ValueError("lag must be an integer")
        if self.lag < 0:
            raise ValueError("lag must be nonnegative")
        if self.target not in ("delta", "absolute"):
            raise ValueError("target must be 'delta' or 'absolute'")
        if self.frame not in ("base", "tool"):
            raise ValueError("frame must be 'base' or 'tool'")
        if not isinstance(self.gripper_inverted, bool):
            raise ValueError("gripper_inverted must be boolean")

    def to_json(self) -> str:
        """Serialize with deterministic field ordering and no whitespace."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> ActionContract:
        """Deserialize JSON while rejecting undeclared side channels."""
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("contract JSON must contain an object")
        expected = {
            "permutation",
            "sign",
            "scale",
            "target",
            "frame",
            "lag",
            "gripper_inverted",
        }
        unknown = sorted(set(decoded) - expected)
        if unknown:
            raise ValueError(f"unknown contract fields: {', '.join(unknown)}")
        missing = sorted(expected - set(decoded))
        if missing:
            raise ValueError(f"missing contract fields: {', '.join(missing)}")
        lag = decoded["lag"]
        if isinstance(lag, bool) or not isinstance(lag, int):
            raise ValueError("lag must be an integer")
        return cls(
            permutation=tuple(int(value) for value in decoded["permutation"]),
            sign=tuple(int(value) for value in decoded["sign"]),
            scale=tuple(float(value) for value in decoded["scale"]),
            target=cast(Target, decoded["target"]),
            frame=cast(Frame, decoded["frame"]),
            lag=lag,
            gripper_inverted=cast(bool, decoded["gripper_inverted"]),
        )
