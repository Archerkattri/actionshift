from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_real_simulator_smokes_pass_and_disclaim_policy_performance() -> None:
    report_paths = ["cpu"]
    # GPU evidence is intentionally optional in CPU-only environments. If a
    # committed GPU report is present, validate its contract exactly as well;
    # its absence must not turn into an invented GPU claim.
    if (ROOT / "reports" / "maniskill_gpu_smoke.json").is_file():
        report_paths.append("gpu")
    for backend in report_paths:
        report = json.loads(
            (ROOT / "reports" / f"maniskill_{backend}_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["passed"] is True
        assert report["sim_backend"] == backend
        assert len(report["tasks"]) >= 3
        assert any("not learned-policy success" in item for item in report["limitations"])
