"""Plug-and-verify self-test: is this robot wired the way the policy thinks?

Packages the action-interface pair's probe-phase finding as a practical startup
check. The public surface is the verdict logic, the identification runner, the
named-wiring library, and the ``actionshift-selftest`` console entry point.
"""

from __future__ import annotations

from actionshift.selftest.identify import (
    ProbeEnvironment,
    SyntheticProbeEnvironment,
    identify_contract,
)
from actionshift.selftest.library import (
    default_pool,
    demo_contract,
    demo_names,
    identity_contract,
    named_contract,
    resolve_pool,
)
from actionshift.selftest.verdict import (
    EXIT_CODES,
    IdentificationResult,
    SelfTestVerdict,
    Status,
    decide_verdict,
)

__all__ = [
    "EXIT_CODES",
    "IdentificationResult",
    "ProbeEnvironment",
    "SelfTestVerdict",
    "Status",
    "SyntheticProbeEnvironment",
    "decide_verdict",
    "default_pool",
    "demo_contract",
    "demo_names",
    "identify_contract",
    "identity_contract",
    "named_contract",
    "resolve_pool",
]
