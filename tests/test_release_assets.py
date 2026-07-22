from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_documents_and_ci_exist() -> None:
    for relative in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        ".github/workflows/ci.yml",
    ):
        assert (ROOT / relative).is_file(), relative


def test_readme_carries_the_consolidated_claim_and_reproduction_discipline() -> None:
    # REPRODUCING.md, CLAIM_AUDIT.md, and the SOTA claim ledger were consolidated into
    # README.md. Assert their load-bearing sections and claim-discipline text survived
    # the merge, so the claim ledger cannot silently vanish from the shipped README.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for section in (
        "## Claims (graded SOTA ledger)",
        "## Claim audit (adversarial red-team)",
        "## Reproducing",
        "### DO NOT CLAIM (ActionShift)",
        "## Benchmark card",
    ):
        assert section in readme, section
    # Normalize whitespace so line-wrapped phrases still match as contiguous text.
    normalized = " ".join(readme.split())
    for discipline in (
        # claim-boundary / honesty statements that must remain present verbatim
        '"Implemented" means code and tests exist; it does not mean the'
        " scientific hypothesis succeeded",
        "no broad-novelty or method-superiority-over-SOTA claim is made",
        "safety mask is a software constraint check",
        'No "first" claims anywhere',
    ):
        assert discipline in normalized, discipline


def test_committed_real_simulator_smokes_pass_and_disclaim_policy_performance() -> None:
    for backend in ("cpu", "gpu"):
        report = json.loads(
            (ROOT / "reports" / f"maniskill_{backend}_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["passed"] is True
        assert report["sim_backend"] == backend
        assert len(report["tasks"]) == 3
        assert any("not learned-policy success" in item for item in report["limitations"])
