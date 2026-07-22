"""``actionshift-selftest`` -- plug-and-verify: is this robot wired the way the
policy thinks?

Runs a bounded 6-step probe phase against an environment (a synthetic bit-faithful
stand-in by default, or a real ManiSkill task with ``--real``), identifies the
hidden action contract within a declared pool of plausible wirings, and renders a
fail-closed PASS / MISMATCH / INCONCLUSIVE verdict with per-field confidence and a
meaningful exit code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionshift.adaptation.response import ResponseModel
from actionshift.contracts.types import ActionContract
from actionshift.selftest.diffs import POSE_FIELDS, UNCHECKED_FIELDS
from actionshift.selftest.identify import (
    ProbeEnvironment,
    SyntheticProbeEnvironment,
    identify_contract,
)
from actionshift.selftest.library import (
    demo_contract,
    demo_names,
    identity_contract,
    named_contract,
    resolve_pool,
)
from actionshift.selftest.verdict import (
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_MISSPEC_RATIO,
    IdentificationResult,
    SelfTestVerdict,
    decide_verdict,
)

_SYNTHETIC_SIGMA = 0.05


def _parse_contract(value: str) -> ActionContract:
    """Resolve a contract from a library name, a JSON file path, or JSON text."""
    try:
        return named_contract(value)
    except ValueError:
        pass
    candidate = Path(value)
    if candidate.is_file():
        return ActionContract.from_json(candidate.read_text(encoding="utf-8"))
    return ActionContract.from_json(value)


def _field_value(contract: ActionContract, field: str) -> object:
    return getattr(contract, field)


def _render_text(
    verdict: SelfTestVerdict,
    result: IdentificationResult,
    *,
    task: str,
    backend: str,
    budget: int,
    amplitude: float,
    pool_size: int,
    misspec_ratio: float,
) -> str:
    lines: list[str] = []
    lines.append("ActionShift plug-and-verify self-test")
    lines.append("=====================================")
    lines.append(f"Task:           {task} ({backend})")
    lines.append(
        f"Probe strategy: {result.strategy}   budget {budget} steps   "
        f"amplitude {amplitude:g}"
    )
    lines.append(f"Declared pool:  {pool_size} candidate wirings")
    lines.append("")
    displacement_unit = "m" if backend.startswith("real") else "synthetic units"
    lines.append(
        "Probe safety: bounded raw pulses only, |action| <= "
        f"{amplitude:g} per pose channel, gripper never actuated. "
        f"Estimated end-effector displacement during probing: "
        f"{result.probe_displacement:.4f} {displacement_unit} "
        f"(mean over {result.probe_steps:.1f} probe steps)."
    )
    lines.append("")
    lines.append("Identified wiring (MAP over the declared pool):")
    for field in POSE_FIELDS:
        confidence = result.field_confidence.get(field, 0.0)
        value = _field_value(verdict.identified, field)
        lines.append(
            f"  {field:<12} {value!s:<32} confidence {confidence:.2f}"
        )
    for field in UNCHECKED_FIELDS:
        lines.append(f"  {field:<12} not checked (out of scope for a pose probe)")
    fit_flag = "ok" if result.fit_ratio <= misspec_ratio else "TOO LARGE"
    lines.append(
        f"  fit residual: {result.fit_ratio:.2f}x calibrated noise scale ({fit_flag})"
    )
    lines.append("")
    lines.append(f"VERDICT: {verdict.status}  (exit {verdict.exit_code})")
    lines.append(f"  {verdict.reason}")
    return "\n".join(lines)


def _render_json(
    verdict: SelfTestVerdict, result: IdentificationResult
) -> str:
    payload = {
        "status": verdict.status,
        "exit_code": verdict.exit_code,
        "reason": verdict.reason,
        "strategy": result.strategy,
        "identified_contract": json.loads(verdict.identified.to_json()),
        "expected_contract": json.loads(verdict.expected.to_json()),
        "field_confidence": result.field_confidence,
        "unchecked_fields": list(UNCHECKED_FIELDS),
        "unresolved_fields": list(verdict.unresolved),
        "field_diffs": verdict.diffs,
        "fit_ratio": result.fit_ratio,
        "probe_steps": result.probe_steps,
        "probe_displacement": result.probe_displacement,
        "map_posterior": result.map_posterior,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actionshift-selftest",
        description=(
            "Verify a robot's action wiring matches what the policy expects, using a "
            "bounded 6-step probe phase and a fail-closed verdict."
        ),
    )
    parser.add_argument("--task", default="pick_cube", help="ManiSkill task key")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo",
        choices=demo_names(),
        help="inject a named demo wiring as the hidden contract (synthetic)",
    )
    source.add_argument(
        "--hidden-contract",
        help="inject an arbitrary hidden contract (library name, JSON, or path)",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help="the wiring the policy expects (library name, JSON, or path); "
        "defaults to identity",
    )
    parser.add_argument(
        "--strategy", choices=("fixed", "entropy"), default="entropy"
    )
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--confidence-floor", type=float, default=DEFAULT_CONFIDENCE_FLOOR
    )
    parser.add_argument("--misspec-ratio", type=float, default=DEFAULT_MISSPEC_RATIO)
    parser.add_argument(
        "--real",
        action="store_true",
        help="probe a real ManiSkill environment (auto-runs calibration if missing) "
        "instead of the synthetic stand-in",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration JSON path for --real (default under artifacts/adaptation)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    expected = (
        identity_contract()
        if arguments.expected is None
        else _parse_contract(arguments.expected)
    )
    if arguments.demo is not None:
        hidden = demo_contract(arguments.demo)
    elif arguments.hidden_contract is not None:
        hidden = _parse_contract(arguments.hidden_contract)
    else:
        hidden = expected  # a plain self-test assumes the wiring under test is correct

    # The pool must be able to represent the wiring the policy expects; the hidden
    # truth is deliberately NOT added, so a wiring outside the pool stays honestly
    # inconclusive (misspecification) rather than being silently vindicated.
    pool = resolve_pool(expected)

    environment: ProbeEnvironment
    if arguments.real:
        from actionshift.adaptation.maniskill import load_or_run_calibration
        from actionshift.selftest.real_env import RealProbeEnvironment

        calibration_path = arguments.calibration or Path(
            f"artifacts/adaptation/calibration/{arguments.task}.json"
        )
        calibration = load_or_run_calibration(arguments.task, calibration_path)
        response = ResponseModel(
            alpha=calibration.alpha,
            sigma=calibration.sigma,
            alpha_c0=(
                calibration.alpha_c0
                if calibration.gain_model == "saturating"
                else None
            ),
            gripper_alpha=calibration.gripper_alpha,
            gripper_sigma=calibration.gripper_sigma,
        )
        backend = "real ManiSkill"
        environment = RealProbeEnvironment(
            arguments.task,
            hidden,
            calibration,
            num_envs=arguments.num_envs,
            seed=arguments.seed,
        )
    else:
        response = ResponseModel(alpha=1.0, sigma=_SYNTHETIC_SIGMA)
        backend = "synthetic bit-faithful stand-in"
        environment = SyntheticProbeEnvironment(
            hidden,
            batch_size=arguments.num_envs,
            response=response,
            seed=arguments.seed,
        )

    try:
        result = identify_contract(
            environment,
            pool,
            response,
            strategy=arguments.strategy,
            budget=arguments.budget,
            amplitude=arguments.amplitude,
            seed=arguments.seed,
        )
    finally:
        environment.close()

    verdict = decide_verdict(
        result,
        expected,
        confidence_floor=arguments.confidence_floor,
        misspec_ratio=arguments.misspec_ratio,
    )

    if arguments.json:
        print(_render_json(verdict, result))
    else:
        print(
            _render_text(
                verdict,
                result,
                task=arguments.task,
                backend=backend,
                budget=arguments.budget,
                amplitude=arguments.amplitude,
                pool_size=len(pool),
                misspec_ratio=arguments.misspec_ratio,
            )
        )
    return verdict.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
