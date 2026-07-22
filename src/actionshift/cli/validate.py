"""Validate immutable ActionShift split configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from actionshift.contracts.splits import generate_split, validate_manifest


def validate_split_config(config_path: Path, *, write_manifest: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("split config must contain a mapping")
    rule = str(config["split_rule"])
    seed = int(config["seed"])
    manifest = generate_split(rule, seed=seed)
    manifest_path = config_path.parent / str(config["manifest"])
    if write_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    if not manifest_path.is_file():
        raise ValueError(f"materialized manifest is missing: {manifest_path}")
    materialized_matches = manifest_path.read_text(encoding="utf-8") == manifest.to_json()
    validation = validate_manifest(manifest)
    validation["materialized_manifest_matches"] = materialized_matches
    validation["config"] = str(config_path)
    if bool(config.get("require_disjoint_composition_signatures", False)):
        validation["composition_requirement_passed"] = (
            validation["composition_overlap_count"] == 0
        )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split", help="validate a split configuration")
    split.add_argument("config", type=Path)
    split.add_argument("--write-manifest", action="store_true")
    arguments = parser.parse_args()
    validation = validate_split_config(
        arguments.config, write_manifest=bool(arguments.write_manifest)
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    required = (
        validation["overlap_count"] == 0
        and validation["ordering_valid"]
        and validation["record_hashes_valid"]
        and validation["manifest_hash_valid"]
        and validation["materialized_manifest_matches"]
        and validation.get("composition_requirement_passed", True)
    )
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
