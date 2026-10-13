"""CLI-level checks for validating a materialized split manifest."""

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


def test_validate_cli_writes_a_valid_materialized_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, _CONFIG)
    monkeypatch.setattr(
        "sys.argv", ["actionshift-validate", "split", str(config_path), "--write-manifest"]
    )
    assert validate.main() == 0
    assert (tmp_path / "manifests" / "seen.json").is_file()


def test_validate_cli_reports_an_unmaterialized_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, _CONFIG)
    monkeypatch.setattr("sys.argv", ["actionshift-validate", "split", str(config_path)])
    assert validate.main() == 2


def test_validate_cli_reports_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr("sys.argv", ["actionshift-validate", "split", str(missing)])
    assert validate.main() == 2


def test_validate_cli_reports_malformed_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, "split_rule: seen\n seed: [unbalanced\n")
    monkeypatch.setattr("sys.argv", ["actionshift-validate", "split", str(config_path)])
    assert validate.main() == 2
