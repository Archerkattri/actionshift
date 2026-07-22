# Recurrent episode-length adapter: an honest negative that sharpens the information limit

Run date: 2026-07-21 UTC. Method: `src/actionshift/adaptation/recurrent_adapter.py`;
training `experiments/train_recurrent_real.py`; evaluation `experiments/run_recurrent_slice.py`;
artifacts `artifacts/adaptation/recurrent/result.json` and
`artifacts/adaptation/recurrent_slices/`. Frozen Gate 0 PPO backbone (`final_ckpt.pt`,
sha `3e6c95d6...`), real PickCube GPU simulation, GPU 2 only.

## The method the evidence motivated

The tournament left one motivated unprivileged move on the table. Passive short-window learned
identification (UP-OSI, window 14) failed honestly at 0.0, and the excitation ablation proved it is
INFORMATION-limited at short windows (weak response model, calibration R2 0.09-0.34, per-step
SNR ~1) — while pool-privileged exact belief reaches ~1.0 by accumulating evidence across the WHOLE
50-step episode with only 9 hypotheses. So the motivated method is EPISODE-LENGTH accumulation
without pool privilege: a recurrent adapter that integrates every `[raw, response]` transition since
the episode began and continuously refines a contract estimate.

Implementation (both proven inductive biases kept):

- a RUNNING ridge least-squares estimate of the lagged joint-regression maps, accumulated
  incrementally over the whole episode so far (the `X^T X` / `X^T Y` Gram accumulators of
  `training.history_features`, but growing step by step instead of over a fixed window, so the maps
  sharpen monotonically). A unit test asserts this running statistic reproduces `history_features`
  bit-for-bit when fed the same window;
- the permutation-equivariant head from `training.OsiRegressor` reading those running maps (the
  equivariant design was proven necessary — flat MLPs memorize permutations and generalize at chance);
- a GRU carried across the episode that refines the flags (target/frame/gripper) and lag heads from
  the running summary — the learned temporal-refinement component;
- an eval-time `RecurrentOsiAdapter` with the `ContractAdapter` protocol, copying `OsiAdapter`'s
  auto-reset timing exactly (per-env boundary reset of the accumulator, GRU hidden, tracked target,
  and estimate; the boundary-crossing transition is discarded as invalid evidence). The estimate
  updates every step; `encode` passes the canonical action through until an 8-step warmup, then
  encodes under the current per-environment MAP estimate.

Training: supervised on real collected FULL-EPISODE sequences (45 steps) under 96 training + 12
held-out contracts, hash-disjoint from every frozen evaluation contract (`evaluation_hashes()`),
with policy and random excitation mixed across the eight parallel environments. Deep supervision:
the OSI loss is applied at every timestep so early-step estimates also train. Collection budget
matched to the OSI run: **34,560 training + 4,320 held-out env-steps** (recorded); loss 4.72 -> 3.41
over 80 epochs; 286 s wall on one RTX 5090.

## Headline curve — held-out identification vs steps observed (12 contracts, 96 sequences)

| steps observed | permutation | sign | target flag | lag |
|---:|---:|---:|---:|---:|
| 5  | 0.111 | 0.569 | 0.594 | 0.750 |
| 15 | 0.370 | 0.573 | 0.615 | 0.917 |
| 30 | 0.333 | 0.576 | 0.677 | 0.927 |
| 45 | 0.335 | 0.559 | 0.688 | 0.906 |

The curve is the result. Episode-length accumulation demonstrably works for the DISCRETE, low-cardinality
fields: **lag identification climbs 0.75 -> 0.92** (a 4-way choice resolved by accumulation) and the
binary **target flag climbs 0.59 -> 0.69**. But the CONTINUOUS map fields saturate early and low:
**permutation plateaus at ~0.35 by step 15** (chance is 0.17 — there is real signal, but nowhere near
enough) and **sign never leaves chance (~0.56)**. More observed steps do not move them. This is exactly
where the information saturates, made visible: the weak real response model (SNR ~1) does not let a
passive learner recover the 6x6 permutation/sign/scale map, no matter how long the episode.

## End-to-end evaluation (pick_cube, seed 20260718, 100 episodes/contract, 8 envs)

| Task / split | recurrent (this work) | Wilson 95% | passive up_osi | exact_belief | oracle |
|---|---:|---:|---:|---:|---:|
| Pick / seen | **0/200 = 0.000** | [0.000, 0.019] | 0.0 | 0.928 | 1.000 |
| Pick / unseen comp. | **2/200 = 0.010** | [0.003, 0.036] | 0.0 | 0.958 | 0.995 |

A 16-episode probe inside the training run agreed (0/16 on seen[1], unseen[0], unseen[1]). The result
is <0.3, so per the preregistered rule no additional seeds were run — the method is not promising and
earns no promotion. The two unseen-composition successes are consistent with the ~1% base rate of a
wrong estimate that happens to encode benignly on an easy episode, not with identification.

## Verdict: the unprivileged challenge is NOT solved — and the negative is more informative than OSI's

Episode-length accumulation does not rescue passive learned identification on the real ActionShift
response model. The recurrent adapter ties passive UP-OSI at the task (0.000 vs 0.0 on seen) despite
seeing 3x the horizon, both excitation regimes, and a GRU. But it is a SHARPER negative than the
single 0.0 OSI reported, because the steps-observed curve localizes the limit per contract field:

- accumulation IS extracting information — lag 0.75 -> 0.92, target 0.59 -> 0.69 over the episode;
- the bottleneck is specifically the continuous permutation + sign map, which stays at ~0.35 / chance
  and does not improve with more steps (information-limited, not horizon-limited);
- an end-to-end encode needs essentially all of permutation, sign, scale, target, frame, lag, and
  gripper simultaneously; with per-channel permutation at 0.35 and sign at chance, the probability the
  full contract is recovered is negligible, hence ~0.0.

The discriminative crux this pins down: on the SAME real responses, pool-privileged exact belief
reaches 0.928-0.958 and six steps of bounded active probing reach ~1.0, because they SCORE 9 discrete
hypotheses (a classification among known candidates) rather than ESTIMATE a continuous 6x6 map. The
unprivileged recurrent learner must solve the harder estimation problem from the same weak signal, and
episode-length accumulation is not enough to close it. That is a real, reusable benchmark finding: the
difficulty of ActionShift's hidden interface for a deployable method is continuous unprivileged
identification under a weak response model, not episode horizon.

## What this does and does not change

- It does not overturn the tournament. The belief family remains pool-privileged; the recurrent
  method is the first unprivileged learned method trained end-to-end on real responses, and it fails
  honestly. No unprivileged method has yet cleared the task.
- It refines the open direction. Accumulation resolves the discrete fields (lag, target); the missing
  ingredient for the continuous fields is stronger evidence per step — active excitation folded into a
  learned identifier (the probe-augmented recurrent hybrid) or an outcome/grasp channel — not more
  passive horizon. This is the next motivated method.

## Claim boundary

- Unprivileged at evaluation: the adapter sees only its own raw actions and calibrated tcp responses;
  it never receives the true contract. A test asserts `RecurrentOsiAdapter.__init__` takes no contract
  argument. Supervision labels exist only at training time.
- Trained on disjoint contracts: 96 + 12 sampled training contracts, each hash-rejected against every
  frozen evaluation contract (seen, unseen_composition, long_lag). No evaluation contract was seen in
  training, and no tuning was done on evaluation contracts.
- Real simulation, matched budget: real PickCube GPU rollouts through the frozen Gate 0 PPO backbone;
  34,560 training + 4,320 held-out collection env-steps, comparable to the OSI run.
- One seed (20260718) at task scale, by the preregistered pruning rule (a <0.3 method is not promoted
  to three seeds). Held-out identification is a 12-contract / 96-sequence average.
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the identity-rotation
  wrapper; no external-benchmark or hardware claim is made. Gripper inversion and tool-frame remain
  unidentifiable from pose responses alone, a documented ceiling shared with the OSI baseline.
