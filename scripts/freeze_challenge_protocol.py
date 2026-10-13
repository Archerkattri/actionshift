"""Freeze current CPU protocol sources without running any policy or simulator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from actionshift.evaluation import challenge
from actionshift.evaluation.challenge import CHALLENGE_VERSION, build_challenge_manifest


def write_frozen_outputs(
    outputs: dict[Path, str], root: Path, *, replace_existing: bool = False
) -> None:
    for path, content in outputs.items():
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"frozen protocol destination must stay inside the checkout: {path}")
        if path.exists() and path.read_text(encoding="utf-8") != content and not replace_existing:
            raise ValueError(f"refusing to overwrite frozen protocol: {path}")

    for path, content in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
            continue
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary_name = stream.name
                stream.write(content)
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="explicitly refresh draft protocol artifacts after source changes",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if Path(challenge.__file__).resolve() != root / "src/actionshift/evaluation/challenge.py":
        raise ValueError("freeze must import this checkout's source; set PYTHONPATH=src")
    manifest = build_challenge_manifest(root)
    template = {
        key: value for key, value in manifest.items() if key not in {"episodes", "protocol_sha256"}
    }
    outputs = {
        root / "experiments/manifests" / f"{CHALLENGE_VERSION}.json": manifest,
        root / "src/actionshift/evaluation/data" / f"{CHALLENGE_VERSION}.json": {
            "protocol": template,
            "protocol_sha256": manifest["protocol_sha256"],
        },
    }
    rendered = {
        path: json.dumps(value, indent=2, sort_keys=True) + "\n" for path, value in outputs.items()
    }
    write_frozen_outputs(rendered, root, replace_existing=args.replace_existing)
    print(manifest["protocol_sha256"])


if __name__ == "__main__":
    main()
