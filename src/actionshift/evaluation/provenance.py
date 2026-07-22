"""Immutable experiment provenance captured beside every result artifact."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Provenance:
    git_commit: str
    git_branch: str
    git_dirty: bool | None
    config_sha256: str
    config_files: tuple[str, ...]
    python_version: str
    package_versions: dict[str, str]
    cuda_visible_devices: str | None
    hardware: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_output(repo_root: Path, arguments: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def _git_details(repo_root: Path) -> tuple[str, str, bool | None]:
    commit_code, commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    branch_code, branch = _git_output(repo_root, ["branch", "--show-current"])
    status_code, status = _git_output(repo_root, ["status", "--porcelain"])
    return (
        commit if commit_code == 0 else "unknown",
        branch if branch_code == 0 and branch else "unknown",
        bool(status) if status_code == 0 else None,
    )


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _total_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 0


def _gpu_inventory(
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = command_runner(command)
    except OSError as error:
        return [], str(error)
    if result.returncode != 0:
        return [], result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            return [], f"malformed nvidia-smi row: {line}"
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "driver_version": fields[3],
                }
            )
        except ValueError:
            return [], f"malformed nvidia-smi row: {line}"
    return gpus, None


def capture_provenance(
    config_paths: list[Path],
    *,
    repo_root: Path,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
    environment: Mapping[str, str] = os.environ,
) -> Provenance:
    """Hash exact configs and record source, dependency, and hardware identities."""
    digest = hashlib.sha256()
    resolved = sorted((path.resolve() for path in config_paths), key=str)
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded_name = str(path).encode()
        contents = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    git_commit, git_branch, git_dirty = _git_details(repo_root)
    gpus, nvidia_smi_error = _gpu_inventory(command_runner)
    return Provenance(
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        config_sha256=digest.hexdigest(),
        config_files=tuple(str(path) for path in resolved),
        python_version=platform.python_version(),
        package_versions={
            "actionshift": _version("actionshift"),
            "torch": _version("torch"),
            "gymnasium": _version("gymnasium"),
            "mani-skill": _version("mani-skill"),
        },
        cuda_visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
        hardware={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
            "torch_cuda_runtime": torch.version.cuda,
            "gpus": gpus,
            "nvidia_smi_error": nvidia_smi_error,
        },
    )
