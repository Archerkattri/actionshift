#!/usr/bin/env python3
"""Analyze completed real hidden-contract PPO Gate 1 slices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from actionshift.benchmarking.gate1_analysis import analyze_gate1_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = analyze_gate1_directory(arguments.input)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.parent / f".{arguments.output.name}.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
