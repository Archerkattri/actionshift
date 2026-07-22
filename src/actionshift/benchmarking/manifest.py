"""Validated source and data pins for the authorized benchmark sprint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class GitSource:
    name: str
    repository: str
    commit: str
    license: str
    paths: tuple[str, ...]
    patches: tuple[str, ...]
    applicable: bool
    reason: str | None

    def __post_init__(self) -> None:
        if not self.repository.startswith("https://"):
            raise ValueError(f"Git source {self.name} repository must use HTTPS")
        if _COMMIT.fullmatch(self.commit) is None:
            raise ValueError(f"Git source {self.name} commit must be a 40-character SHA")
        if not self.license:
            raise ValueError(f"Git source {self.name} license must be nonempty")
        if not self.applicable and not self.reason:
            raise ValueError(f"inapplicable Git source {self.name} requires a reason")


@dataclass(frozen=True, slots=True)
class HuggingFaceAsset:
    name: str
    repository: str
    revision: str
    license: str
    estimated_bytes: int
    include: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.repository.count("/") != 1:
            raise ValueError(f"Hugging Face repository {self.name} must be owner/name")
        if _COMMIT.fullmatch(self.revision) is None:
            raise ValueError(f"Hugging Face asset {self.name} revision must be a 40-character SHA")
        if not self.license:
            raise ValueError(f"Hugging Face asset {self.name} license must be nonempty")
        if self.estimated_bytes < 0:
            raise ValueError(f"Hugging Face asset {self.name} estimate must be nonnegative")
        if not self.include:
            raise ValueError(f"Hugging Face asset {self.name} include list must be nonempty")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    max_download_bytes: int
    git_sources: tuple[GitSource, ...]
    hf_assets: tuple[HuggingFaceAsset, ...]

    @property
    def estimated_download_bytes(self) -> int:
        return sum(asset.estimated_bytes for asset in self.hf_assets)

    def __post_init__(self) -> None:
        if self.max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive")
        names = [source.name for source in self.git_sources]
        names.extend(asset.name for asset in self.hf_assets)
        if len(names) != len(set(names)):
            raise ValueError("duplicate source name in manifest")
        if self.estimated_download_bytes > self.max_download_bytes:
            raise ValueError(
                f"estimated downloads exceed download cap: "
                f"{self.estimated_download_bytes} > {self.max_download_bytes}"
            )


def _git_source(value: Any) -> GitSource:
    if not isinstance(value, dict):
        raise ValueError("git_sources entries must be mappings")
    applicable = value.get("applicable", True)
    if not isinstance(applicable, bool):
        raise ValueError("applicable must be boolean")
    reason_value = value.get("reason")
    if reason_value is not None and not isinstance(reason_value, str):
        raise ValueError("reason must be a string")
    return GitSource(
        name=_required_string(value.get("name"), "name"),
        repository=_required_string(value.get("repository"), "repository"),
        commit=_required_string(value.get("commit"), "commit"),
        license=_required_string(value.get("license"), "license"),
        paths=_string_tuple(value.get("paths"), "paths"),
        patches=_string_tuple(value.get("patches", []), "patches"),
        applicable=applicable,
        reason=reason_value,
    )


def _hf_asset(value: Any) -> HuggingFaceAsset:
    if not isinstance(value, dict):
        raise ValueError("hf_assets entries must be mappings")
    estimated_bytes = value.get("estimated_bytes")
    if not isinstance(estimated_bytes, int):
        raise ValueError("estimated_bytes must be an integer")
    return HuggingFaceAsset(
        name=_required_string(value.get("name"), "name"),
        repository=_required_string(value.get("repository"), "repository"),
        revision=_required_string(value.get("revision"), "revision"),
        license=_required_string(value.get("license"), "license"),
        estimated_bytes=estimated_bytes,
        include=_string_tuple(value.get("include"), "include"),
    )


def load_manifest(path: Path) -> SourceManifest:
    """Load and validate all immutable source, attribution, and size fields."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source manifest must be a mapping")
    maximum = value.get("max_download_bytes")
    if not isinstance(maximum, int):
        raise ValueError("max_download_bytes must be an integer")
    git_values = value.get("git_sources")
    hf_values = value.get("hf_assets")
    if not isinstance(git_values, list) or not isinstance(hf_values, list):
        raise ValueError("git_sources and hf_assets must be lists")
    return SourceManifest(
        max_download_bytes=maximum,
        git_sources=tuple(_git_source(item) for item in git_values),
        hf_assets=tuple(_hf_asset(item) for item in hf_values),
    )
