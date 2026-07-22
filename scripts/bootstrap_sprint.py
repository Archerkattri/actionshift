#!/usr/bin/env python3
"""Validate, dry-run, or materialize the sprint's immutable external inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from actionshift.benchmarking.manifest import GitSource, HuggingFaceAsset, load_manifest
from actionshift.evaluation.provenance import sha256_file


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = _run(command)
    if result.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {"command": command, "returncode": result.returncode, "stderr": result.stderr},
                sort_keys=True,
            )
        )
    return result


def _hf_command(asset: HuggingFaceAsset, *, destination: Path | None, dry_run: bool) -> list[str]:
    command = [
        "hf",
        "download",
        asset.repository,
        "--type",
        "dataset",
        "--revision",
        asset.revision,
    ]
    for pattern in asset.include:
        command.extend(["--include", pattern])
    if destination is not None:
        command.extend(["--local-dir", str(destination)])
    if dry_run:
        command.extend(["--dry-run", "--format", "json"])
    return command


def _apply_patch(destination: Path, patch: Path) -> dict[str, str]:
    patch = patch.resolve()
    if not patch.is_file():
        raise RuntimeError(f"declared patch does not exist: {patch}")
    reverse = _run(["git", "-C", str(destination), "apply", "--reverse", "--check", str(patch)])
    if reverse.returncode != 0:
        _checked(["git", "-C", str(destination), "apply", "--check", str(patch)])
        _checked(["git", "-C", str(destination), "apply", str(patch)])
    return {"path": str(patch), "sha256": sha256_file(patch)}


def _checkout(source: GitSource, destination: Path, project_root: Path) -> dict[str, object]:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _checked(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source.repository,
                str(destination),
            ]
        )
    _checked(["git", "-C", str(destination), "fetch", "--depth=1", "origin", source.commit])
    _checked(["git", "-C", str(destination), "checkout", "--detach", source.commit])
    observed = _checked(["git", "-C", str(destination), "rev-parse", "HEAD"]).stdout.strip()
    if observed != source.commit:
        raise RuntimeError(f"source checkout mismatch for {source.name}: {observed}")
    missing = [path for path in source.paths if not (destination / path).is_file()]
    if missing:
        raise RuntimeError(f"source {source.name} missing declared paths: {missing}")
    patches = [_apply_patch(destination, project_root / patch) for patch in source.patches]
    diff = _checked(["git", "-C", str(destination), "diff", "--binary"]).stdout
    return {
        **asdict(source),
        "checkout": str(destination.resolve()),
        "observed_commit": observed,
        "applied_patches": patches,
        "working_diff": diff,
    }


def _download(asset: HuggingFaceAsset, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    _checked(_hf_command(asset, destination=destination, dry_run=False))
    _checked(
        [
            "hf",
            "cache",
            "verify",
            asset.repository,
            "--type",
            "dataset",
            "--revision",
            asset.revision,
            "--local-dir",
            str(destination),
        ]
    )
    hashes = {
        str(path.relative_to(destination)): sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != ".cache"
    }
    return {**asdict(asset), "local_dir": str(destination.resolve()), "files": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/sprint/sources.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    manifest = load_manifest(arguments.manifest)
    summary = {
        "estimated_download_bytes": manifest.estimated_download_bytes,
        "max_download_bytes": manifest.max_download_bytes,
        "git_sources": len(manifest.git_sources),
        "hf_assets": len(manifest.hf_assets),
    }
    if arguments.validate_only:
        print(json.dumps(summary, sort_keys=True))
        return
    if arguments.dry_run:
        for asset in manifest.hf_assets:
            result = _checked(_hf_command(asset, destination=None, dry_run=True))
            print(result.stdout.strip())
        print(json.dumps(summary, sort_keys=True))
        return

    source_records = [
        _checkout(source, arguments.root / "third_party" / source.name, arguments.root)
        for source in manifest.git_sources
    ]
    asset_records = [
        _download(asset, arguments.root / "data" / "sprint" / asset.name)
        for asset in manifest.hf_assets
    ]
    lock = {"summary": summary, "git_sources": source_records, "hf_assets": asset_records}
    destination = arguments.root / "artifacts" / "source-lock.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
