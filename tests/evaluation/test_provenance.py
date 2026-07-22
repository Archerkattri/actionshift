from __future__ import annotations

import json
import subprocess
from pathlib import Path

from actionshift.evaluation.provenance import capture_provenance, sha256_file
from actionshift.evaluation.runner import write_matrix


def test_provenance_hashes_exact_config_bytes(tmp_path: Path) -> None:
    config = tmp_path / "method.yaml"
    config.write_text("name: dualabi\n", encoding="utf-8")
    first = capture_provenance([config], repo_root=tmp_path)
    config.write_text("name: oracle\n", encoding="utf-8")
    second = capture_provenance([config], repo_root=tmp_path)
    assert first.config_sha256 != second.config_sha256
    assert first.git_commit == "unknown"
    assert first.python_version


def test_matrix_writer_is_atomic_and_machine_readable(tmp_path: Path) -> None:
    destination = tmp_path / "matrix.jsonl"
    count = write_matrix(destination)
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert count == len(lines) == 750
    assert all(json.loads(line)["job_id"] for line in lines)
    assert not (tmp_path / ".matrix.jsonl.tmp").exists()


def test_provenance_records_dirty_branch_and_cuda_inventory(tmp_path: Path) -> None:
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


def test_provenance_preserves_nvidia_smi_failure_and_hashes_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "episodes.jsonl"
    artifact.write_bytes(b'{"success":true}\n')

    def fail(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="driver unavailable")

    provenance = capture_provenance(
        [artifact], repo_root=tmp_path, command_runner=fail, environment={}
    )

    assert provenance.hardware["gpus"] == []
    assert provenance.hardware["nvidia_smi_error"] == "driver unavailable"
    assert sha256_file(artifact) == (
        "b493cdb3b30ea63f6a924f814dfccfcfe305dac02106f9994ce2bcb2e8ed28c4"
    )
