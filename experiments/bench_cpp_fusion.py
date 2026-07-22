"""Honest torch-vs-C++ benchmark for the evidence-scoring step of the belief loop.

Sweeps ``num_envs x hypothesis-space x device x backend`` and reports per-step
scoring latency (median, p10, p90) for:

* the full-grammar factorized per-cell evidence (``(modes,B,6,6,|signs|,|scales|)``,
  the heavy identification kernel), and
* the nine-hypothesis pooled log-likelihood (``(B,9)``, the light kernel);

against the torch reference. C++ CPU is measured single-thread and multi-thread;
C++ CUDA is measured transfer-inclusive when the extension was built with CUDA.
Only the evidence-scoring step is timed -- the MAP assignment, history ring, and
encode are identical across backends and excluded.

Usage:
  .venv/bin/python experiments/bench_cpp_fusion.py --device cpu \
      --out artifacts/cpp_fusion/bench_cpu.json
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/bench_cpp_fusion.py \
      --device cuda --out artifacts/cpp_fusion/bench_gpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from actionshift.adaptation.cpp_backend import (
    CppCellScorer,
    CppPoolScorer,
    cpp_backend_available,
    load_cells_module,
)
from actionshift.adaptation.factorized_grammar import FactorizedGrammarDriver
from actionshift.adaptation.response import ResponseModel

_ENV_COUNTS = (8, 64, 256, 1024)
_WARMUP = 5
_REPS = 50
_RESPONSE = ResponseModel(
    alpha=(1.0, 0.9, 1.1, 0.8, 1.2, 1.0), sigma=(0.3, 0.4, 0.5, 0.2, 0.6, 0.3)
)


def _time(fn: Callable[[], object], *, device: str, reps: int = _REPS) -> dict[str, float]:
    for _ in range(_WARMUP):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, int(0.1 * len(samples)) - 1)],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
    }


def _factorized_rows(device: str, dtype: torch.dtype, threads: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    module_has_cuda = cpp_backend_available() and bool(
        getattr(load_cells_module(), "has_cuda", False)
    )
    for num_envs in _ENV_COUNTS:
        torch_driver = FactorizedGrammarDriver(
            batch_size=num_envs, response=_RESPONSE, device=device, dtype=dtype
        )
        generator = torch.Generator().manual_seed(20260721)
        for _ in range(3):
            raw = torch.rand((num_envs, 7), generator=generator, dtype=dtype) * 2 - 1
            obs = torch.rand((num_envs, 6), generator=generator, dtype=dtype) * 2 - 1
            torch_driver.update(raw.to(device), obs.to(device))
        observed = (torch.rand((num_envs, 6), generator=generator, dtype=dtype) * 2 - 1).to(
            device
        )

        backends: list[tuple[str, int, FactorizedGrammarDriver | None]] = []
        # torch reference driver
        backends.append(("torch", torch.get_num_threads(), torch_driver))
        # C++ CPU single-thread and multi-thread
        for th, label in ((1, "cpp_cpu_1t"), (threads, f"cpp_cpu_{threads}t")):
            cpp_driver = FactorizedGrammarDriver(
                batch_size=num_envs,
                response=_RESPONSE,
                device=device,
                dtype=dtype,
                cell_scorer=CppCellScorer(num_threads=th),
            )
            cpp_driver._history = torch_driver._history.clone()
            backends.append((label, th, cpp_driver))
        # C++ CUDA transfer-inclusive
        if module_has_cuda:
            cuda_driver = FactorizedGrammarDriver(
                batch_size=num_envs,
                response=_RESPONSE,
                device=device,
                dtype=torch.float32,
                cell_scorer=CppCellScorer(use_cuda=True),
            )
            cuda_driver._history = torch_driver._history.to(torch.float32)
            backends.append(("cpp_cuda", 0, cuda_driver))

        for label, th, driver in backends:
            assert driver is not None
            timing = _time(
                lambda d=driver, o=observed: d._compute_contribution(
                    o.to(d._history.dtype)
                ),
                device=device,
            )
            rows.append(
                {
                    "hypothesis_space": "factorized_grammar",
                    "num_envs": num_envs,
                    "device": device,
                    "backend": label,
                    "threads": th,
                    **timing,
                }
            )
    return rows


def _pool_rows(device: str, dtype: torch.dtype, threads: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hypotheses = 9
    alpha = torch.tensor(_RESPONSE.alpha, device=device, dtype=dtype)
    sigma = torch.tensor(_RESPONSE.sigma, device=device, dtype=dtype)
    for num_envs in _ENV_COUNTS:
        generator = torch.Generator().manual_seed(11)
        predicted = (
            torch.rand((hypotheses, num_envs, 6), generator=generator, dtype=dtype) * 2 - 1
        ).to(device)
        observed = (torch.rand((num_envs, 6), generator=generator, dtype=dtype) * 2 - 1).to(
            device
        )

        torch_timing = _time(
            lambda p=predicted, o=observed: _RESPONSE.log_likelihood(p, o),
            device=device,
        )
        rows.append(
            {
                "hypothesis_space": "pool9",
                "num_envs": num_envs,
                "device": device,
                "backend": "torch",
                "threads": torch.get_num_threads(),
                **torch_timing,
            }
        )
        for th, label in ((1, "cpp_cpu_1t"), (threads, f"cpp_cpu_{threads}t")):
            scorer = CppPoolScorer(num_threads=th)
            cpp_timing = _time(
                lambda s=scorer, p=predicted, o=observed, a=alpha, sg=sigma: s.log_likelihood(
                    p, o, a, sg
                ),
                device=device,
            )
            rows.append(
                {
                    "hypothesis_space": "pool9",
                    "num_envs": num_envs,
                    "device": device,
                    "backend": label,
                    "threads": th,
                    **cpp_timing,
                }
            )
    return rows


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| space | envs | device | backend | threads | median ms | p10 | p90 |",
        "|---|--:|---|---|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['hypothesis_space']} | {r['num_envs']} | {r['device']} | "
            f"{r['backend']} | {r['threads']} | {float(r['median_ms']):.4f} | "
            f"{float(r['p10_ms']):.4f} | {float(r['p90_ms']):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=min(8, torch.get_num_threads()))
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if not cpp_backend_available():
        raise SystemExit("actionabi_cells extension not built; build ACTIONABI_BUILD_PYBIND")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda is unavailable")

    dtype = torch.float32
    rows = _factorized_rows(arguments.device, dtype, arguments.threads)
    rows += _pool_rows(arguments.device, dtype, arguments.threads)
    report = {
        "schema_version": "1.0",
        "host": platform.node(),
        "device": arguments.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "cpp_threads": arguments.threads,
        "dtype": "float32",
        "warmup": _WARMUP,
        "reps": _REPS,
        "cpp_has_cuda": bool(getattr(load_cells_module(), "has_cuda", False)),
        "rows": rows,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(_markdown(rows))
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
