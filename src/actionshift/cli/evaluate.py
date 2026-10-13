"""Generate the frozen matrix or summarize JSONL episode records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from actionshift.evaluation.challenge import run_reference_smoke, write_challenge_manifest
from actionshift.evaluation.runner import summarize_file, write_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("output", type=Path)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("episodes", type=Path)
    summary.add_argument("output", type=Path)
    challenge_manifest = subparsers.add_parser("challenge-manifest")
    challenge_manifest.add_argument("output", type=Path)
    challenge_manifest.add_argument("--repository-root", type=Path)
    challenge_smoke = subparsers.add_parser("challenge-smoke")
    challenge_smoke.add_argument("output", type=Path)
    challenge_smoke.add_argument("--steps", type=int, default=64)
    challenge_smoke.add_argument(
        "--seeds", type=int, nargs="+", default=[20260718, 20260719, 20260720]
    )
    challenge_smoke.add_argument("--adapter", help="optional external module:factory")
    challenge_smoke.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record failed external-adapter episodes and preserve the planned denominator",
    )
    challenge_smoke.add_argument(
        "--adapter-seed",
        type=int,
        default=0,
        help="adapter RNG seed, independent of the hidden episode seed",
    )
    challenge_smoke.add_argument(
        "--action-seed",
        type=int,
        default=0,
        help="seed for the canonical action trajectory, independent of the hidden episode seed",
    )
    arguments = parser.parse_args()
    try:
        if arguments.command == "matrix":
            print(json.dumps({"job_count": write_matrix(arguments.output)}))
        elif arguments.command == "summarize":
            print(json.dumps(summarize_file(arguments.episodes, arguments.output), sort_keys=True))
        elif arguments.command == "challenge-manifest":
            report = write_challenge_manifest(arguments.output, arguments.repository_root)
            print(json.dumps({"protocol_sha256": report["protocol_sha256"]}))
        else:
            report = run_reference_smoke(
                arguments.output,
                seeds=arguments.seeds,
                steps=arguments.steps,
                external_reference=arguments.adapter,
                adapter_seed=arguments.adapter_seed,
                action_seed=arguments.action_seed,
                continue_on_error=arguments.continue_on_error,
            )
            methods = report.get("methods")
            if not isinstance(methods, list):
                raise TypeError("challenge report methods must be a list")
            print(json.dumps({"method_count": len(methods)}))
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
