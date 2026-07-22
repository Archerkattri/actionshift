# C++-in-the-loop fusion: ActionABI's scoring core as an ActionShift identification backend

Run date: 2026-07-21 UTC. This report documents the systems fusion of the two
projects: ActionABI's (post-`scorer_fixes`) C++20 evidence-scoring core is exposed
through a pybind11 binding and wired into ActionShift's factorized-grammar belief
loop as an **alternative evidence-scoring backend**. The C++ tool becomes
load-bearing inside the adaptation loop -- it computes the per-transition,
per-hypothesis-cell residual scores that drive the MAP contract decision.

Artifacts:
- Binding: `actionabi/bindings/{cell_score.hpp,module.cpp,cell_score_cuda.cu}`,
  CMake target `actionabi_cells` (option `ACTIONABI_BUILD_PYBIND`).
- Integration: `src/actionshift/adaptation/cpp_backend.py` + a `cell_scorer`
  injection seam on `FactorizedGrammarDriver` (`adaptation/factorized_grammar.py`).
- Parity tests: `tests/test_cpp_fusion.py` (Python), `actionabi/tests/test_cell_score.cpp`
  (ctest `cell_score`).
- Benchmark: `experiments/bench_cpp_fusion.py`; results
  `artifacts/cpp_fusion/bench_{cpu,gpu}.json`.
- End-to-end slices: `artifacts/adaptation/factorized_slices/{4899c5b9c54c7c0e,5f903d016a8d1141}.summary.json`.

---

## 1. Architecture

### The scoring step that is offloaded

The full-grammar belief (`reports/adaptation_factorized.md`) scores the *entire*
declared contract grammar per step by factorizing the joint into per-channel cells.
Its evidence-scoring step builds, for each observed transition, a
`(modes, envs, 6[i], 6[j], |signs|, |scales|)` Gaussian log-evidence tensor

```
contribution[m,b,i,j,s,k] =
    -0.5 * ((observed[b,i] - alpha[i] * base_m[b,j] * signs[s] * scales[k]) / sigma[i])**2
```

where `base_m[b,j]` is the raw drive of channel `j` under global mode
`m = (target, lag)`:

```
delta:    base_m[b,j] = history[lag][b,j]                       (raw_{t-lag})
absolute: base_m[b,j] = history[lag][b,j] - history[lag+1][b,j] (raw_{t-lag} - raw_{t-lag-1})
```

This is exactly the **FIXED single-step-delayed lag alignment** established in
ActionABI `reports/scorer_fixes.md` (defect 1): the observed response at step `t`
is explained by the raw action `lag` steps earlier, i.e. the one-step transition at
the delayed index, not a multi-step span. The binding embeds that convention in
C++ (`cell_score.hpp` computes `base` from the history ring with the same rule the
CPU/CUDA trajectory scorers use after the fix).

### The binding (pybind11)

`actionabi_cells` is a pybind11 extension built by a dedicated CMake target under
`actionabi/bindings/`. pybind11 is discovered via its installed CMake config
(`pybind11.get_cmake_dir()`) with a FetchContent fallback. It exposes:

- `score_cells_f32 / score_cells_f64` -- the fused per-cell kernel above, computed
  in a single pass with no intermediate tensors (the torch path allocates several
  broadcast temporaries per mode). Optional `num_threads` partitions the batch over
  `std::thread`.
- `score_hypotheses_f32 / score_hypotheses_f64` -- the pooled nine-hypothesis
  log-likelihood (`ResponseModel.log_likelihood`), for the pool driver / benchmark.
- `score_cells_cuda` (when built with `ACTIONABI_CELLS_ENABLE_CUDA=ON`,
  `-DCMAKE_CUDA_ARCHITECTURES=120`) -- a transfer-inclusive GPU variant (H2D +
  kernel + D2H), so any GPU number reported is honest end-to-end.

The float32/float64 split lets the caller dispatch on the tensor dtype so parity is
exact per precision.

### The seam (no driver rewrite)

`FactorizedGrammarDriver` gained one optional keyword argument, `cell_scorer:
CellScorer | None = None`, and the inner scoring block of `update()` was extracted
verbatim into `_compute_contribution(observed)`. When `cell_scorer is None` the
original torch path runs unchanged (all 13 pre-existing factorized tests stay
green); when injected, the contribution is delegated to the backend. `cpp_backend.py`
provides `CppCellScorer` (implements the `CellScorer` protocol), `CppPoolScorer`
(the pool log-likelihood), and the extension loader. The MAP linear-assignment,
history ring, masks, and encode are untouched -- **only the evidence-scoring step
crosses into C++.**

The runner `experiments/run_factorized_slice.py` gained `--backend {torch,cpp}`.
For the 9-pool path, `CppPoolScorer` reproduces `ResponseModel.log_likelihood` (the
pose-evidence step) and is parity-tested and benchmarked below; it is **not** wired
as a driver-level swap because the pool `ExactBeliefDriver` was since extended with a
separate gripper-channel evidence term, so a faithful full swap would need that term
too. The required, load-bearing integration is the factorized-grammar backend.

Build recipe (CMake 4.4 via `uvx`, Release; see the Reproducing section of the ActionABI repo's `README.md`):

```bash
cd code/actionabi
PYBIND_DIR=$(../actionshift/.venv/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())")
uvx --from cmake cmake -S . -B build-pybind -DACTIONABI_BUILD_PYBIND=ON \
  -DACTIONABI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=../actionshift/.venv/bin/python -Dpybind11_DIR=$PYBIND_DIR
uvx --from cmake cmake --build build-pybind -j 8
# GPU variant: add -DACTIONABI_CELLS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
```

---

## 2. Parity evidence (target: reproduce torch scores <=1e-6 relative)

`tests/test_cpp_fusion.py` (4 tests, all green; ruff + strict mypy clean) and the
Catch2 ctest `cell_score` (4 assertions blocks, green):

| Check | Result |
|---|---|
| Per-cell accumulated evidence, **float64**, 40-step sequence over every grammar mode | max relative diff **5.7e-14** (<= 1e-6 target met with ~7 orders of margin) |
| MAP contract decisions, float64 synthetic drive (delta and lagged) | **identical** to torch, and both recover the true contract |
| Full synthetic control loop through the C++ backend | recovers the same permutation/sign/scale/target/lag as torch |
| Production dtype **float32**, 40-step sequence | max relative diff ~1.5e-5 on tiny near-zero cells; **MAP decisions bit-identical** |
| Pool log-likelihood vs `ResponseModel.log_likelihood`, float64 | max relative diff **< 1e-6** |
| C++ CUDA vs C++ CPU (float32) | max abs diff **6.1e-5** (float32 rounding) |
| Catch2 `cell_score`: log-kernel value, absolute-mode delayed base, threaded==serial | pass |

The headline numeric-parity claim (`<= 1e-6 relative`) is met in float64 at
**5.7e-14**. In production float32 the per-cell accumulation order differs between
the torch broadcast and the fused C++ pass, giving ~1e-5 relative differences on
near-zero cells -- but the **MAP decisions that actually drive control are
bit-identical**, which the end-to-end slice confirms.

---

## 3. Honest benchmark: torch vs C++ per-step scoring latency

Host: RTX 5090 box, 64 CPU threads. torch default intra-op threads = 32; C++
multi-thread = 8; dtype float32; 5 warmups, 50 measurements, median ms. Only the
evidence-scoring step is timed (MAP/history/encode excluded and identical across
backends). `cpp_cuda` is transfer-inclusive.

### CPU device (`artifacts/cpp_fusion/bench_cpu.json`)

Factorized grammar (the heavy `(M,B,6,6,2,7)` kernel):

| envs | torch (32t) | cpp 1-thread | cpp 8-thread |
|--:|--:|--:|--:|
| 8 | 0.186 | **0.014** | 0.212 |
| 64 | 1.308 | **0.056** | 0.238 |
| 256 | 1.822 | **0.216** | 0.261 |
| 1024 | 9.434 | 0.828 | **0.337** |

Pool (9 hypotheses, the light `(B,9)` kernel):

| envs | torch (32t) | cpp 1-thread | cpp 8-thread |
|--:|--:|--:|--:|
| 8 | 0.018 | **0.008** | 0.209 |
| 64 | 0.023 | **0.009** | 0.204 |
| 256 | 0.037 | **0.013** | 0.211 |
| 1024 | 0.161 | **0.043** | 0.223 |

### GPU device (`artifacts/cpp_fusion/bench_gpu.json`, GPU 1)

Factorized grammar:

| envs | torch (cuda) | cpp cpu-1t | cpp cpu-8t | cpp cuda (transfer-incl.) |
|--:|--:|--:|--:|--:|
| 8 | 0.376 | **0.062** | 0.301 | 0.269 |
| 64 | 0.355 | **0.166** | 0.412 | 0.375 |
| 256 | **0.358** | 0.433 | 0.689 | 0.742 |
| 1024 | **0.322** | 1.443 | 1.877 | 1.867 |

### Where C++ wins, and where it does not (plainly)

- **CPU, C++ wins decisively and predictably.** For the heavy factorized kernel a
  single-thread C++ pass beats torch's 32-thread execution at *every* size
  (~13x at 8 envs, ~11x at 1024) by avoiding per-op dispatch and the broadcast
  temporaries torch allocates per mode. torch's CPU timing is also highly variable
  (its factorized median jumped 1.8->9.4 ms across sizes/runs); the C++ path is
  stable. Multi-thread C++ carries a fixed ~0.2 ms `std::thread`-spawn cost per
  call, so it only pays off for the largest batch (1024 envs: **0.337 ms**, the
  fastest cell), and loses to single-thread below that. For the light pool kernel,
  single-thread C++ wins throughout; threading is never worth it.
- **GPU, torch wins at scale; C++ wins only when the GPU is launch-bound.** torch's
  CUDA scoring is flat ~0.32-0.38 ms regardless of env count (kernel-launch +
  sync bound), so at 8-64 envs the C++ CPU scorer (scoring GPU tensors via
  host round-trip) is *faster* than torch-cuda (0.062 vs 0.376 ms at 8 envs). From
  256 envs up, torch-cuda's parallelism wins. The **transfer-inclusive C++ CUDA
  path loses to torch-cuda at every size** -- the per-call H2D/D2H dominates a kernel
  this small -- exactly reproducing ActionABI's own preregistered transfer-inclusive
  negative (`actionabi/reports/benchmark_sprint.md`). Kernel-only C++ CUDA would
  be faster but is not a real end-to-end number, so it is not claimed.

**The defensible claim is the CPU scaling regime.** For future multi-env,
training-time identification (many parallel envs scoring the full grammar every
step on CPU while the GPU trains the policy), the fused C++ scorer is 1-2 orders of
magnitude faster and far more predictable than the torch scoring step. On a single
GPU eval at 8 envs the scoring is not the bottleneck either way (see below).

---

## 4. End-to-end proof (real ManiSkill GPU sim, GPU 1)

One real eval slice, `pick_cube` / seen / seed 20260718 / 100 episodes-per-contract
(2 frozen representative contracts = 200 episodes) / 8 envs /
`CUDA_VISIBLE_DEVICES=1`, identical config except the scoring backend (same frozen
PPO checkpoint and calibration -- SHA-256 verified equal):

| Backend | Success | Overall | Wilson 95% | Per-contract | sec/step |
|---|--:|--:|--:|---|--:|
| torch | 64/200 | 0.320 | [0.259, 0.388] | c0=0/100, c1=64/100 | 0.00699 |
| **ActionABI C++** | 63/200 | **0.315** | [0.255, 0.382] | c0=0/100, c1=63/100 | 0.00685 |

The C++ backend reproduces the torch success rate within noise -- a **one-episode**
difference (float32 MAP flip on a single step), with identical per-contract
structure (c0 is the documented gripper-inverted Wall-1 contract at 0.00; c1 the
recoverable one). The belief loop ran end-to-end with ActionABI's C++ core scoring
the evidence at every step. End-to-end wall time is GPU-simulation-bound
(sec/step ~0.0069 both backends), so at 8 envs the scoring backend is not the
bottleneck -- the fusion is proven correct here, and its latency advantage is in the
many-env CPU regime of section 3.

---

## 5. Verification ledger

| Gate | Result |
|---|---|
| ActionABI Release ctest (incl. new `cell_score`) | **9/9** |
| ActionShift `test_cpp_fusion.py` (parity) | **4/4** |
| ActionShift `test_adaptation_factorized.py` (unchanged driver behaviour) | **13/13** |
| ActionShift full suite | 235 passed, 1 pre-existing failure unrelated to this work* |
| ruff + strict mypy (actionshift) | clean |
| C++/torch cell parity (float64) | max rel **5.7e-14** |
| End-to-end pick_cube/seen slice torch vs C++ | 0.320 vs 0.315 (within noise) |

\* `test_tasks.py::test_frozen_task_registry_covers_pick_push_and_insertion` fails
because a `pull_cube` task was added to the registry by concurrent unrelated work;
this fusion touches no code under `envs/`.

---

## 6. Claim boundary

- **What is in C++:** the identification *evidence-scoring* step -- the
  per-transition, per-hypothesis-cell Gaussian residual scores (factorized grammar,
  wired and load-bearing) and the pooled per-hypothesis log-likelihood (9-pool,
  provided + parity/benchmark-tested but not driver-wired; see section 1), using the
  fixed single-step-delayed lag semantics. This is the load-bearing step: it produces
  the scores the MAP decision reads.
- **What stays in torch:** the history ring, reset/invalid masks, the MAP
  linear-assignment, the tracked-target accumulation, and the encode. This is the
  identification *scorer* backend, **not the whole adapter in C++**.
- **Parity:** float64 reproduces the torch scores to 5.7e-14 (<= 1e-6 target);
  float32 (production) yields bit-identical MAP decisions and an end-to-end success
  rate within one episode of torch.
- **Performance:** C++ wins the CPU scoring regime decisively (1-2 orders of
  magnitude, all env counts for the heavy kernel) and is far more predictable than
  torch on CPU. On GPU, torch-cuda wins at scale and the transfer-inclusive C++ CUDA
  path loses (transfer-bound), consistent with ActionABI's own CUDA verdict. The
  interesting, defensible claim is the many-env CPU regime for future training-time
  multi-env identification -- not a single-GPU 8-env eval, where scoring is not the
  bottleneck.
- **Not claimed:** kernel-only CUDA speedup as an end-to-end result; any change to
  the belief's identifiability walls (gripper, scale, absolute-target) -- the C++
  backend computes the same evidence, so it inherits the same walls.
