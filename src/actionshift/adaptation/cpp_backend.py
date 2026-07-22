"""ActionABI C++ scoring core as an ActionShift identification backend.

This wires the (post-``scorer_fixes``) ActionABI C++ evidence-scoring kernels into
ActionShift's belief loop as an *alternative* evidence-scoring backend, without
rewriting the torch drivers. Two seams are provided:

* :class:`CppCellScorer` implements the :class:`~actionshift.adaptation.
  factorized_grammar.CellScorer` protocol -- it computes the full-grammar
  per-cell Gaussian log-evidence tensor in fused C++ instead of torch, and is
  injected into an unmodified :class:`FactorizedGrammarDriver` via its
  ``cell_scorer`` hook.
* :class:`CppPoolScorer` computes the nine-hypothesis pooled log-likelihood
  (``ResponseModel.log_likelihood``) in C++; :func:`cpp_response_model` wraps it
  as a drop-in ``ResponseModel`` so the pool :class:`ExactBeliefDriver` can score
  through C++ with no driver change.

The compiled extension (``actionabi_cells``) is produced by the ActionABI CMake
target ``ACTIONABI_BUILD_PYBIND`` (see ``code/actionabi/bindings/``). It is located
via, in order: the ``ACTIONABI_CELLS_DIR`` environment variable, an explicit path,
then the default ``code/actionabi/build-pybind`` build directory.

Claim boundary: only the *evidence-scoring* step is delegated to C++. The history
ring, masks, MAP linear-assignment, and encode remain in torch -- this backend is
the identification scorer, not the whole adapter.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_DEFAULT_BUILD_DIR = (
    Path(__file__).resolve().parents[4] / "actionabi" / "build-pybind"
)


class CppBackendUnavailable(RuntimeError):
    """Raised when the compiled ``actionabi_cells`` extension cannot be imported."""


def _candidate_dirs(explicit: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("ACTIONABI_CELLS_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(_DEFAULT_BUILD_DIR)
    return candidates


@lru_cache(maxsize=8)
def load_cells_module(build_dir: str | Path | None = None) -> Any:
    """Import and return the compiled ``actionabi_cells`` extension.

    Raises :class:`CppBackendUnavailable` with the searched locations if the
    extension is not importable.
    """
    for directory in _candidate_dirs(build_dir):
        if not directory.exists():
            continue
        matches = list(directory.glob("actionabi_cells*.so"))
        if not matches:
            continue
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        import actionabi_cells  # type: ignore[import-not-found]

        return actionabi_cells
    searched = ", ".join(str(d) for d in _candidate_dirs(build_dir))
    raise CppBackendUnavailable(
        "actionabi_cells extension not found. Build it with the ActionABI CMake "
        "target (ACTIONABI_BUILD_PYBIND=ON) or set ACTIONABI_CELLS_DIR. Searched: "
        f"{searched}"
    )


def cpp_backend_available(build_dir: str | Path | None = None) -> bool:
    """True when the compiled extension can be imported."""
    try:
        load_cells_module(build_dir)
    except CppBackendUnavailable:
        return False
    return True


def _numpy_dtype(dtype: torch.dtype) -> Any:
    import numpy as np

    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float64:
        return np.float64
    raise TypeError(f"cpp backend supports float32/float64, not {dtype}")


class CppCellScorer:
    """C++-backed per-cell grammar evidence scorer (``CellScorer`` protocol).

    Computes the ``(modes, batch, 6, 6, |signs|, |scales|)`` Gaussian log-evidence
    tensor with the fused ActionABI kernel. Inputs are moved to host, scored in
    C++ (CPU, optionally multi-threaded; or transfer-inclusive CUDA when built and
    requested), and the result is returned on the input tensors' device/dtype.
    """

    def __init__(
        self,
        *,
        build_dir: str | Path | None = None,
        num_threads: int = 1,
        use_cuda: bool = False,
    ) -> None:
        import numpy as np

        self._np = np
        self._module = load_cells_module(build_dir)
        self._num_threads = int(num_threads)
        self._use_cuda = bool(use_cuda)
        if self._use_cuda and not bool(getattr(self._module, "has_cuda", False)):
            raise CppBackendUnavailable(
                "actionabi_cells was built without CUDA; rebuild with "
                "ACTIONABI_CELLS_ENABLE_CUDA=ON to use use_cuda=True"
            )

    @property
    def uses_cuda(self) -> bool:
        return self._use_cuda

    def score(
        self,
        *,
        history: Tensor,
        observed: Tensor,
        alpha: Tensor,
        sigma: Tensor,
        signs: tuple[float, ...],
        scales: tuple[float, ...],
        mode_targets: tuple[int, ...],
        mode_lags: tuple[int, ...],
    ) -> Tensor:
        np = self._np
        device = history.device
        dtype = history.dtype
        np_dtype = _numpy_dtype(dtype)
        history_np = np.ascontiguousarray(
            history.detach().cpu().numpy(), dtype=np_dtype
        )
        observed_np = np.ascontiguousarray(
            observed.detach().cpu().numpy(), dtype=np_dtype
        )
        alpha_np = np.ascontiguousarray(alpha.detach().cpu().numpy(), dtype=np_dtype)
        sigma_np = np.ascontiguousarray(sigma.detach().cpu().numpy(), dtype=np_dtype)
        signs_np = np.asarray(signs, dtype=np_dtype)
        scales_np = np.asarray(scales, dtype=np_dtype)
        target_np = np.asarray(mode_targets, dtype=np.int32)
        lag_np = np.asarray(mode_lags, dtype=np.int32)

        if self._use_cuda:
            out = self._module.score_cells_cuda(
                history_np.astype(np.float32),
                observed_np.astype(np.float32),
                alpha_np.astype(np.float32),
                sigma_np.astype(np.float32),
                signs_np.astype(np.float32),
                scales_np.astype(np.float32),
                target_np,
                lag_np,
            )
        elif dtype == torch.float64:
            out = self._module.score_cells_f64(
                history_np, observed_np, alpha_np, sigma_np, signs_np, scales_np,
                target_np, lag_np, self._num_threads,
            )
        else:
            out = self._module.score_cells_f32(
                history_np, observed_np, alpha_np, sigma_np, signs_np, scales_np,
                target_np, lag_np, self._num_threads,
            )
        return torch.from_numpy(out).to(device=device, dtype=dtype)


class CppPoolScorer:
    """C++-backed pooled per-hypothesis Gaussian log-likelihood."""

    def __init__(
        self, *, build_dir: str | Path | None = None, num_threads: int = 1
    ) -> None:
        import numpy as np

        self._np = np
        self._module = load_cells_module(build_dir)
        self._num_threads = int(num_threads)

    def log_likelihood(
        self, predicted: Tensor, observed: Tensor, alpha: Tensor, sigma: Tensor
    ) -> Tensor:
        """Return ``(batch, hypotheses)`` log-likelihood; ``predicted`` is (H,B,C)."""
        np = self._np
        device = observed.device
        dtype = observed.dtype
        np_dtype = _numpy_dtype(dtype)
        channels = int(alpha.shape[0])
        predicted_np = np.ascontiguousarray(
            predicted[..., :channels].detach().cpu().numpy(), dtype=np_dtype
        )
        observed_np = np.ascontiguousarray(
            observed[..., :channels].detach().cpu().numpy(), dtype=np_dtype
        )
        alpha_np = np.ascontiguousarray(alpha.detach().cpu().numpy(), dtype=np_dtype)
        sigma_np = np.ascontiguousarray(sigma.detach().cpu().numpy(), dtype=np_dtype)
        if dtype == torch.float64:
            out = self._module.score_hypotheses_f64(
                predicted_np, observed_np, alpha_np, sigma_np, self._num_threads
            )
        else:
            out = self._module.score_hypotheses_f32(
                predicted_np, observed_np, alpha_np, sigma_np, self._num_threads
            )
        return torch.from_numpy(out).to(device=device, dtype=dtype)


__all__ = [
    "CppBackendUnavailable",
    "CppCellScorer",
    "CppPoolScorer",
    "cpp_backend_available",
    "load_cells_module",
]
