from __future__ import annotations

import json
from pathlib import Path

from actionshift.evaluation.provenance import capture_provenance
from actionshift.evaluation.runner import write_matrix


def test_matrix_writer_is_atomic_and_machine_readable(tmp_path: Path) -> None:
    destination = tmp_path / "matrix.jsonl"
    count = write_matrix(destination)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert count == len(lines) == 750
    assert all(json.loads(line)["job_id"] for line in lines)
    assert not (tmp_path / ".matrix.jsonl.tmp").exists()


def test_provenance_records_dirty_branch_and_cuda_inventory(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Benchmark"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "benchmark@example.invalid"],
        check=True,
    )
    config = tmp_path / "gate0.yaml"
    config.write_text("gate: 0\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "gate0.yaml"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
    config.write_text("gate: 0\nseed: 1\n", encoding="utf-8")

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="0, NVIDIA GeForce RTX 5090, 32607, 590.48.01\n",
            stderr="",
        )

    provenance = capture_provenance(
        [config],
        repo_root=tmp_path,
        command_runner=run,
        environment={"CUDA_VISIBLE_DEVICES": "2"},
    )

    assert provenance.git_dirty is True
    assert provenance.git_branch in {"main", "master"}
    assert provenance.cuda_visible_devices == "2"
    assert provenance.hardware["cpu_count"]
    assert provenance.hardware["total_memory_bytes"] > 0
    assert provenance.hardware["gpus"] == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5090",
            "memory_total_mib": 32607,
            "driver_version": "590.48.01",
        }
    ]
