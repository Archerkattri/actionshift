#!/usr/bin/env python3
"""Write the frozen real PPO Gate 1 slice manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from actionshift.benchmarking.gate1_eval import plan_gate1_slice_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-checkpoint", type=Path, required=True)
    parser.add_argument("--push-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    jobs = plan_gate1_slice_jobs(
        {
            "pick_cube": arguments.pick_checkpoint,
            "push_cube": arguments.push_checkpoint,
        },
        seeds=(20260718, 20260719, 20260720),
        output_directory=arguments.output_directory,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.parent / f".{arguments.output.name}.tmp"
    temporary.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    os.replace(temporary, arguments.output)
    print(json.dumps({"jobs": len(jobs), "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
