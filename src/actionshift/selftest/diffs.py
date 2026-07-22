"""Human-readable field differences between two action contracts.

A miswiring is only useful to a user if it is stated plainly: "channels 0 and 1
SWAPPED; channel 3 sign FLIPPED", not two opaque permutation tuples. Each helper
turns one contract field's difference into such a sentence.
"""

from __future__ import annotations

from actionshift.contracts.types import ActionContract

# Pose fields a bounded probe can observe. ``gripper_inverted`` is intentionally
# absent: the probe never actuates the gripper, so its direction is out of scope
# (reported as "not checked", never PASS/MISMATCH). See the
# ``actionshift-selftest`` section of ``README.md``.
POSE_FIELDS: tuple[str, ...] = (
    "permutation",
    "sign",
    "scale",
    "target",
    "frame",
    "lag",
)
UNCHECKED_FIELDS: tuple[str, ...] = ("gripper_inverted",)


def _describe_permutation(identified: tuple[int, ...], expected: tuple[int, ...]) -> str:
    changed = [c for c in range(len(identified)) if identified[c] != expected[c]]
    # A clean two-channel transposition: identified swaps exactly one pair.
    if len(changed) == 2:
        a, b = changed
        if identified[a] == expected[b] and identified[b] == expected[a]:
            return f"channels {a} and {b} SWAPPED"
    return (
        "channel routing changed at channels "
        f"{changed}: expected permutation {expected}, wired {identified}"
    )


def _describe_sign(identified: tuple[int, ...], expected: tuple[int, ...]) -> str:
    flipped = [c for c in range(len(identified)) if identified[c] != expected[c]]
    if len(flipped) == 1:
        return f"channel {flipped[0]} sign FLIPPED"
    return f"channels {flipped} sign FLIPPED"


def _describe_scale(identified: tuple[float, ...], expected: tuple[float, ...]) -> str:
    parts = [
        f"channel {c} scale {expected[c]:g}->{identified[c]:g}"
        for c in range(len(identified))
        if identified[c] != expected[c]
    ]
    return "; ".join(parts)


def describe_field_diff(
    field: str, identified: ActionContract, expected: ActionContract
) -> str:
    """One-sentence description of how ``field`` differs (assumes it differs)."""
    if field == "permutation":
        return _describe_permutation(identified.permutation, expected.permutation)
    if field == "sign":
        return _describe_sign(identified.sign, expected.sign)
    if field == "scale":
        return _describe_scale(identified.scale, expected.scale)
    if field == "target":
        return (
            f"target encoding differs: expected {expected.target}, "
            f"wired {identified.target}"
        )
    if field == "frame":
        return (
            f"reference frame differs: expected {expected.frame}, "
            f"wired {identified.frame}"
        )
    if field == "lag":
        return (
            f"actuation lag differs: expected {expected.lag} step(s), "
            f"wired {identified.lag} step(s)"
        )
    if field == "gripper_inverted":
        return "gripper direction INVERTED"
    raise ValueError(f"unknown contract field: {field}")


def differing_fields(
    identified: ActionContract,
    expected: ActionContract,
    *,
    fields: tuple[str, ...] = POSE_FIELDS,
) -> dict[str, str]:
    """Map each differing field to its human-readable diff sentence."""
    diffs: dict[str, str] = {}
    for field in fields:
        if getattr(identified, field) != getattr(expected, field):
            diffs[field] = describe_field_diff(field, identified, expected)
    return diffs
