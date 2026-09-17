# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-17

### Added

- Frozen compound-shift external evaluation protocol.
- PyPI release with PyPI/downloads badges; Hugging Face dataset and
  checkpoints links.
- Zenodo DOI (10.5281/zenodo.21500713) in README/CITATION.
- This changelog.

### Changed

- Refreshed README visuals; dropped the CI badge per badge policy; updated
  CI to Node 24 runtimes.

### Fixed

- Aligned the mypy type-check target with numpy 2.5 stub syntax (Python 3.12
  target); runtime support for Python 3.11 is unchanged.

- Falsification return annotated so `mypy --strict` passes on both
  numpy 2.5 and newer stub inference (CI green). Suite: 320 passed,
  10 skipped.

## [1.0.0] - 2026-07-22

Hidden action-interface adaptation benchmark: composable action contracts,
simulated envs, adaptation methods including the DualABI controller, compound
evaluation, CLI entry points, self-test harnesses, hardened CLI error
boundaries, reproducible DP-shim rollouts, and paper media.
