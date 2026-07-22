"""Simulator-free tests for the actionshift-smoke CLI plumbing.

The real ManiSkill oracle smoke needs the simulator; here we cover main()'s
argument parsing, exit-code mapping, and the atomic report write by faking
run_maniskill_smoke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from actionshift.evaluation import maniskill_smoke


def test_write_atomic_writes_valid_json(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "report.json"
    payload = {"passed": True, "value": 3}
    maniskill_smoke._write_atomic(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    # No temporary file is left behind.
    assert list(destination.parent.glob(".*.tmp")) == []


@pytest.mark.parametrize(("passed", "expected_code"), [(True, 0), (False, 1)])
def test_main_maps_passed_to_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    expected_code: int,
) -> None:
    output = tmp_path / "smoke.json"
    captured: dict[str, Any] = {}

    def fake_run(*, sim_backend: str, steps: int) -> dict[str, Any]:
        captured["sim_backend"] = sim_backend
        captured["steps"] = steps
        return {"kind": "maniskill_oracle_encoding_smoke", "passed": passed}

    monkeypatch.setattr(maniskill_smoke, "run_maniskill_smoke", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["actionshift-smoke", "--sim-backend", "cpu", "--steps", "5", "--output", str(output)],
    )

    assert maniskill_smoke.main() == expected_code
    assert captured == {"sim_backend": "cpu", "steps": 5}
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is passed


def test_run_maniskill_smoke_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        maniskill_smoke.run_maniskill_smoke(steps=0)
