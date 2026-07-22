"""Tests for the actionshift-validate CLI entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from actionshift.cli import validate

_CONFIG = """schema_version: "1.0"
name: seen
seed: 20260718
generator_version: "split-v1"
split_rule: seen
manifest: manifests/seen.json
require_disjoint_contracts: true
"""


def _write_config(directory: Path, body: str) -> Path:
    config_path = directory / "split.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_validate_split_config_passes_on_materialized_manifest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _CONFIG)
    result = validate.validate_split_config(config_path, write_manifest=True)
    assert result["overlap_count"] == 0
    assert result["ordering_valid"] is True
    assert result["record_hashes_valid"] is True
    assert result["manifest_hash_valid"] is True
    assert result["materialized_manifest_matches"] is True


def test_validate_split_config_detects_tampered_manifest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _CONFIG)
    validate.validate_split_config(config_path, write_manifest=True)
    manifest_path = tmp_path / "manifests" / "seen.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = validate.validate_split_config(config_path, write_manifest=False)
    assert result["materialized_manifest_matches"] is False


@pytest.mark.parametrize("missing_key", ["split_rule", "seed", "manifest"])
def test_validate_split_config_rejects_missing_required_key(
    tmp_path: Path, missing_key: str
) -> None:
    body = "\n".join(
        line for line in _CONFIG.splitlines() if not line.startswith(f"{missing_key}:")
    )
    config_path = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=missing_key):
        validate.validate_split_config(config_path, write_manifest=True)


def test_validate_split_config_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        validate.validate_split_config(config_path, write_manifest=False)


def test_main_returns_zero_on_valid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, _CONFIG)
    monkeypatch.setattr(
        "sys.argv", ["actionshift-validate", "split", str(config_path), "--write-manifest"]
    )
    assert validate.main() == 0


def test_main_returns_one_on_unmaterialized_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without --write-manifest the manifest is absent, so validate_split_config raises;
    # main() must translate that into a clean exit code 2, not a raw traceback.
    config_path = _write_config(tmp_path, _CONFIG)
    monkeypatch.setattr("sys.argv", ["actionshift-validate", "split", str(config_path)])
    assert validate.main() == 2


def test_main_returns_two_on_missing_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr("sys.argv", ["actionshift-validate", "split", str(missing)])
    assert validate.main() == 2


def test_main_returns_two_on_malformed_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, "split_rule: seen\n seed: [unbalanced\n")
    monkeypatch.setattr("sys.argv", ["actionshift-validate", "split", str(config_path)])
    assert validate.main() == 2
