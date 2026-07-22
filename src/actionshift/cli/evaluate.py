"""Generate the frozen matrix or summarize JSONL episode records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from actionshift.evaluation.runner import summarize_file, write_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("output", type=Path)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("episodes", type=Path)
    summary.add_argument("output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "matrix":
            print(json.dumps({"job_count": write_matrix(arguments.output)}))
        else:
            print(
                json.dumps(summarize_file(arguments.episodes, arguments.output), sort_keys=True)
            )
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
