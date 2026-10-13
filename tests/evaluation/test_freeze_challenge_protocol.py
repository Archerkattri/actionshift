"""Frozen protocol output handling stays explicit and checkout-bounded."""

from pathlib import Path

import pytest

from scripts.freeze_challenge_protocol import write_frozen_outputs


def test_freeze_refuses_changed_artifacts_without_explicit_refresh(tmp_path: Path):
    root = tmp_path.resolve()
    output = root / "experiments" / "manifest.json"
    output.parent.mkdir()
    output.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_frozen_outputs({output: "new"}, root)

    assert output.read_text(encoding="utf-8") == "old"


def test_freeze_can_explicitly_refresh_a_changed_draft(tmp_path: Path):
    root = tmp_path.resolve()
    output = root / "experiments" / "manifest.json"
    output.parent.mkdir()
    output.write_text("old", encoding="utf-8")

    write_frozen_outputs({output: "new"}, root, replace_existing=True)

    assert output.read_text(encoding="utf-8") == "new"
    assert sorted(path.name for path in output.parent.iterdir()) == ["manifest.json"]


def test_freeze_rejects_outputs_outside_checkout(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="inside the checkout"):
        write_frozen_outputs({outside: "data"}, root, replace_existing=True)

    assert not outside.exists()
