from __future__ import annotations

import json

from actionshift.benchmarking.gate1_analysis import analyze_gate1_directory


def test_gate1_analysis_pairs_methods_and_requires_three_seeds(tmp_path) -> None:
    records = []
    for seed in (20260718, 20260719, 20260720):
        for episode, (oracle, baseline) in enumerate(((True, False), (True, True))):
            common = {
                "task": "pick_cube",
                "split": "seen",
                "seed": seed,
                "contract_sha256": "a" * 64,
                "episode_index": episode,
                "task_return": 1.0,
            }
            records.extend(
                ({**common, "method": "oracle", "success": oracle},
                 {**common, "method": "no_adapt", "success": baseline})
            )
    (tmp_path / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = analyze_gate1_directory(tmp_path, bootstrap_samples=500)

    comparison = report["comparisons"]["pick_cube/seen"]
    assert comparison["oracle_minus_no_adapt"]["estimate"] == 0.5
    assert comparison["matched_seeds"] == [20260718, 20260719, 20260720]
    assert comparison["three_seed_comparison"] is True
    assert report["groups"]["pick_cube/seen/oracle"]["success"]["rate"] == 1.0


def test_one_seed_report_is_descriptive_not_superiority(tmp_path) -> None:
    rows = [
        {
            "task": "push_cube", "split": "long_lag", "seed": 20260718,
            "contract_sha256": "b" * 64, "episode_index": 0,
            "task_return": 0.0, "method": method, "success": success,
        }
        for method, success in (("oracle", True), ("no_adapt", False))
    ]
    (tmp_path / "one.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = analyze_gate1_directory(tmp_path, bootstrap_samples=100)

    assert report["comparisons"]["push_cube/long_lag"]["three_seed_comparison"] is False
    assert report["claim_status"] == "descriptive_only"
