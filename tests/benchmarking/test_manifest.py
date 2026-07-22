from __future__ import annotations

from pathlib import Path

import pytest

from actionshift.benchmarking.manifest import load_manifest


def test_checked_in_manifest_pins_sources_and_fits_download_cap() -> None:
    manifest = load_manifest(Path("configs/sprint/sources.yaml"))

    assert {source.name for source in manifest.git_sources} >= {"maniskill", "fasttd3"}
    assert all(len(source.commit) == 40 for source in manifest.git_sources)
    assert all(source.repository.startswith("https://") for source in manifest.git_sources)
    assert all(source.license for source in manifest.git_sources)
    maniskill = next(source for source in manifest.git_sources if source.name == "maniskill")
    assert maniskill.patches == ("patches/maniskill-tdmpc2-gymnasium-1.3.patch",)
    assert len(manifest.hf_assets) == 6
    assert all(len(asset.revision) == 40 for asset in manifest.hf_assets)
    assert all(asset.license for asset in manifest.hf_assets)
    assert manifest.estimated_download_bytes < manifest.max_download_bytes == 250_000_000_000


def test_manifest_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
max_download_bytes: 100
git_sources:
  - name: duplicate
    repository: https://example.invalid/a
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    license: MIT
    paths: []
hf_assets:
  - name: duplicate
    repository: owner/data
    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    license: MIT
    estimated_bytes: 1
    include: ['data/**']
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate source name"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("commit", "main", "40-character"),
        ("license", "", "license"),
        ("repository", "git@example.invalid:repo", "HTTPS"),
    ],
)
def test_manifest_rejects_unpinned_or_unattributed_git_source(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    source = {
        "name": "source",
        "repository": "https://example.invalid/repo",
        "commit": "a" * 40,
        "license": "MIT",
        "paths": [],
    }
    source[field] = value
    path = tmp_path / "sources.yaml"
    path.write_text(
        "max_download_bytes: 100\ngit_sources:\n"
        + "  - "
        + repr(source).replace("'", '"')
        + "\nhf_assets: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_manifest_rejects_download_estimate_over_cap(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
max_download_bytes: 10
git_sources: []
hf_assets:
  - name: data
    repository: owner/data
    revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    license: MIT
    estimated_bytes: 11
    include: ['data/**']
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="download cap"):
        load_manifest(path)
