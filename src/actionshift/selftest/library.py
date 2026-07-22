"""Named contract library for the plug-and-verify self-test.

The self-test identifies a hidden action contract within a *declared finite pool*
of plausible wirings (the exact-belief privilege documented in
``reports/adaptation_tournament.md``). This module curates that pool from a small
library of named, human-readable wirings plus a demo registry the tool injects to
prove it catches miswirings.

Every pool member here is ``target="delta"``, ``frame="base"``, and
``gripper_inverted=False`` on purpose: those are the wirings a bounded pose probe
can *observe*. Absolute-target scale, tool frame under identity rotation, and the
gripper direction are structurally unobservable from the probe (see the
``actionshift-selftest`` section of ``README.md``), so they are not part of the
default identifiable pool.
"""

from __future__ import annotations

from actionshift.contracts.types import ActionContract

_POSE = 6


def _pose_contract(
    permutation: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    sign: tuple[int, ...] = (1,) * _POSE,
    scale: tuple[float, ...] = (1.0,) * _POSE,
    *,
    lag: int = 0,
) -> ActionContract:
    """Build a delta/base pose contract (the identifiable family)."""
    return ActionContract(
        permutation=permutation,
        sign=sign,
        scale=scale,
        target="delta",
        frame="base",
        lag=lag,
        gripper_inverted=False,
    )


def identity_contract() -> ActionContract:
    """The wiring a policy assumes by default: pass-through, no remapping."""
    return _pose_contract()


# The default declared pool: identity plus a spread of single- and multi-fault
# wirings, all mutually distinguishable by a bounded pose probe. The named demo
# wirings are members so the tool can identify (and therefore catch) them.
_LIBRARY: dict[str, ActionContract] = {
    "identity": identity_contract(),
    "swapped-axes": _pose_contract(permutation=(1, 0, 2, 3, 4, 5)),
    "sign-flip": _pose_contract(sign=(1, 1, 1, -1, 1, 1)),
    "miswired": _pose_contract(permutation=(1, 0, 2, 3, 4, 5), sign=(1, 1, 1, -1, 1, 1)),
    "scaled": _pose_contract(scale=(0.5, 1.0, 1.0, 1.0, 1.0, 1.0)),
    "relabeled": _pose_contract(
        permutation=(2, 0, 1, 5, 3, 4),
        sign=(1, 1, -1, -1, 1, 1),
        scale=(2.0, 0.5, 0.75, 1.5, 1.0, 1.25),
    ),
    "reversed": _pose_contract(
        permutation=(5, 4, 3, 2, 1, 0),
        sign=(1, -1, -1, 1, 1, -1),
        scale=(1.5, 1.5, 0.5, 2.0, 0.75, 1.25),
    ),
    "lagged": _pose_contract(lag=2),
}


def default_pool() -> tuple[ActionContract, ...]:
    """Return the declared finite pool of identifiable wirings."""
    return tuple(_LIBRARY.values())


# Demo hidden contracts the tool can inject to prove each verdict path. Every
# demo except ``unmodeled`` is a member of the default pool (so it is
# identifiable); ``unmodeled`` deliberately sits *outside* the pool to exercise
# the misspecification / fail-closed path.
_UNMODELED = _pose_contract(
    permutation=(3, 1, 4, 0, 5, 2),
    sign=(-1, -1, 1, 1, -1, 1),
    scale=(0.6, 1.25, 0.75, 1.5, 2.0, 0.5),
)

_DEMOS: dict[str, ActionContract] = {
    "identity": _LIBRARY["identity"],
    "swapped-axes": _LIBRARY["swapped-axes"],
    "sign-flip": _LIBRARY["sign-flip"],
    "miswired": _LIBRARY["miswired"],
    "scaled": _LIBRARY["scaled"],
    "reversed": _LIBRARY["reversed"],
    "lagged": _LIBRARY["lagged"],
    "unmodeled": _UNMODELED,
}


def demo_names() -> tuple[str, ...]:
    return tuple(_DEMOS)


def demo_contract(name: str) -> ActionContract:
    """Return the hidden contract a named demo injects."""
    try:
        return _DEMOS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown demo {name!r}; choose from {', '.join(demo_names())}"
        ) from error


def named_contract(name: str) -> ActionContract:
    """Return a library wiring by name (for the ``--expected`` shortcut)."""
    try:
        return _LIBRARY[name]
    except KeyError as error:
        raise ValueError(
            f"unknown wiring {name!r}; choose from {', '.join(_LIBRARY)}"
        ) from error


def resolve_pool(
    *extra: ActionContract, base: tuple[ActionContract, ...] | None = None
) -> tuple[ActionContract, ...]:
    """Return the pool augmented with ``extra`` contracts, de-duplicated.

    The expected contract (and, in demo mode, any in-pool demo) must be
    representable, so callers fold them in here. Order is preserved and existing
    members win, so the default pool stays stable.
    """
    pool = list(default_pool() if base is None else base)
    seen = {contract.to_json() for contract in pool}
    for contract in extra:
        key = contract.to_json()
        if key not in seen:
            pool.append(contract)
            seen.add(key)
    return tuple(pool)
