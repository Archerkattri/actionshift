# Probe-augmented learned identification: the probe budget that scores nine hypotheses cannot estimate the map

Run date: 2026-07-21 UTC. Method: `src/actionshift/adaptation/probe_osi.py`
(`ProbeOsiAdapter`); training `experiments/train_probe_osi_real.py`; evaluation
`experiments/run_probe_osi_slice.py`; tests `tests/test_adaptation_probe_osi.py`.
Artifacts: `artifacts/adaptation/probe_osi/` (trained model + identification report)
and `artifacts/adaptation/probe_osi_slices/` (hash-addressed per-contract episode
JSONL + summaries). Frozen Gate 1 PPO backbones (same checkpoints as the
tournament/factorized runs), real ManiSkill GPU simulation, **GPU 0 only**.

## The method the evidence motivated

Two prior negatives set this up. Passive learned identification (UP-OSI, recurrent)
failed at 0.0 because smooth policy actions are weak excitation (per-step SNR ~1);
the continuous permutation/sign map never resolves (perm ~0.29–0.35, sign ~chance),
and episode-length accumulation does not help. Meanwhile the belief family clears the
same task at ~1.0 with **six fixed basis pulses** (amplitude 0.5): a basis pulse
isolates one raw channel per step, so the per-channel response is a near-direct read
of the contract column. The recurrent report's explicit recommendation was to give
the *learned* identifier that same excitation.

This method does exactly that, changing **one variable**. It reuses the recurrent
machinery unchanged — `RunningLagFeatures` (episode-length running least-squares of
the lagged joint-regression maps, so the strong probe evidence at the episode start is
never rolled out of a window) and the equivariant `RecurrentOsiRegressor` (equivariant
heads for the continuous map, a GRU for the discrete flags). The **only** change from
the recurrent negative is the excitation: the adapter spends the first `budget = 6`
steps sending the SAME fixed probe schedule the probe family uses, and the model is
trained on probe-excited transitions. That isolates bounded active probing as the
single lever, so the identification-by-field deltas measure exactly what six probes buy
a learned estimator.

Training: supervised on real probe-excited full-episode (45-step) sequences,
first 6 steps fixed basis pulses then frozen-policy pass-through, collected across
**both** eval tasks (pick_cube + push_cube) under each task's own contract-independent
calibration, 96 training + 12 held-out contracts, hash-disjoint from every frozen
evaluation contract (`evaluation_hashes()`). Deep supervision (the OSI loss at every
timestep). Collection budget **34,560 training + 4,320 held-out env-steps** (recorded),
matched to the OSI and recurrent runs; loss 4.72 → 4.00 over 80 epochs, 251 s on one
RTX 5090.

## Headline — held-out identification BY FIELD on probe-excited windows (96 sequences, both tasks)

| steps observed | permutation | sign | scale (±25%) | scale med \|Δlog\| | target | lag | gripper |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5  | 0.111 | 0.569 | 0.417 | 0.333 | 0.479 | 0.417 | 0.417 |
| 10 | 0.314 | 0.564 | 0.411 | 0.258 | 0.521 | 0.562 | 0.417 |
| 20 | **0.391** | 0.575 | 0.411 | 0.263 | **0.917** | 0.448 | 0.479 |
| 45 | 0.299 | 0.549 | 0.382 | 0.274 | 0.917 | 0.417 | 0.448 |

Read directly against the baselines on the same response model:

| field | passive OSI | random-excitation ablation | recurrent (episode-length) | **probe (this work, best)** |
|---|---:|---:|---:|---:|
| permutation | 0.29 | 0.52 | ~0.35 (plateau) | **0.39** |
| sign | 0.55 (chance) | chance | ~chance | **0.55 (chance)** |
| target | — | — | 0.59→0.69 | **0.48→0.92** |
| lag | — | 0.77 | 0.75→0.92 | 0.42→0.56 |

**The probe budget does not rescue the continuous map.** Six basis pulses lift
permutation only to **0.39** — above passive (0.29) and the recurrent plateau (0.35),
but **below the random-excitation ablation (0.52)**, and sign never leaves chance.
The strong result is on a DISCRETE field: **target identification jumps 0.48 → 0.92**
(the probe pulses cleanly expose the delta-vs-absolute previous-step signature). Scale
stays walled (±25% ≈ 0.40, consistent with the ~0.6× attenuation the factorized run and
the ActionABI bridge both measured) and gripper stays at chance (unobservable from a
pose response). Lag is weak here (0.42–0.56) because eval-relevant training used
`max_lag=2` and the probe pulses de-emphasise the long-lag ring.

**The decay from step 20 (0.391) to step 45 (0.299) is the mechanism, made visible.**
The running accumulator weights every transition; the 6 clean probe steps are
progressively **diluted** by 39 subsequent weak, correlated policy steps folded into the
same least-squares Gram. The probe advantage is therefore transient — it peaks right
after the probe phase and erodes — the opposite of the belief family, whose posterior is
already near-certain after the probes so weak evidence cannot move it.

## End-to-end evaluation (seed 20260718, 100 episodes/contract, 8 envs, GPU 0)

Every cell was < 0.3 at seed 20260718, so by the preregistered pruning rule none was
promoted to three seeds. Rates over 200 episodes (2 contracts × 100), 8 envs, probe
budget spent in full (mean probe steps = 6.00 on every contract).

| Task / split | **probe_osi (this work)** | passive learned (OSI/recurrent)\* | factorized grammar +probes\* | pool fixed_probes\* | oracle\* |
|---|---:|---:|---:|---:|---:|
| Pick / seen | **0/200 = 0.000** | 0.000 | 0.458 | 0.928 | 1.000 |
| Pick / unseen comp. | **1/200 = 0.005** | 0.010 | 0.005 | 0.970 | 0.995 |
| Push / seen | **3/200 = 0.015** | — | 0.673 | 0.982 | 1.000 |
| Push / unseen comp. | **0/200 = 0.000** | — | 0.000 | 0.980 | 1.000 |

\* Reference columns are the matched-setup numbers from `reports/adaptation_recurrent.md`
and `reports/adaptation_factorized.md`.

Per-contract, the negative is complete: on Pick/seen the CLEAN contract
(`gripper_inverted=False`, delta, lag 0) — the one the factorized-probe belief reaches
0.917 on — scores **0/100** here. The handful of non-zero episodes (Push/seen 3/200,
Pick/unseen 1/200) sit at the ~0.5–1.5% base rate of a wrong estimate that happens to
encode benignly on an easy episode, not identification. The result agrees with the
identification-by-field prediction exactly: with per-channel permutation at 0.39 and
sign at chance, the probability the full 6-channel contract is encoded correctly is
negligible, so the task collapses.

## Why it fails where the belief family succeeds — the sharp finding

The discriminative crux the recurrent report opened is now closed with a number. **The
same six-pulse budget that lets the belief family reach ~1.0 is fundamentally
insufficient for a learned estimator**, and the reason is dimensional, not incidental:

- The belief family **scores** nine discrete hypotheses. Six probes that each isolate a
  channel are far more than enough to pick the right one of nine — a classification
  among known candidates.
- The learned identifier must **estimate** a continuous map. The lagged least-squares
  regressor is 12-dimensional per channel (`[raw_t, raw_{t−1}]`), so recovering the full
  6×6 permutation/sign/scale map needs ≥12 well-conditioned samples. **Six basis pulses
  underdetermine it** — they span at most six regressor directions, so ridge fills the
  rest with bias, and permutation caps at ~0.39 while sign stays at chance.

This is a stronger, more useful statement of the open challenge than the passive 0.0:
the barrier is not "the learned method never got good excitation" (it now got the
belief family's exact excitation) but **"the bounded probe budget that suffices for
discrete scoring cannot supply the sample rank a continuous estimator needs."** Active
probing is not a free substitute for the pool/grammar privilege — it changes which
quantity is cheap (discrete target identification became easy: 0.92) without making the
continuous map identifiable within the fair budget.

## Ladder position

| method | privilege | Pick/seen | Push/seen | continuous map recovered? |
|---|---|---:|---:|---|
| passive OSI / recurrent (learned) | trained model only | ~0.000 | — | no (perm 0.29–0.35, sign chance) |
| **probe_osi (this work)** | **+ bounded probe budget** | **0.000** | **0.015** | **no (perm 0.39, sign chance)** |
| factorized grammar (+probes) | grammar knowledge | 0.458 | 0.673 | scoring, not estimation |
| pool exact_belief / fixed_probes | declared 9-pool | 0.928 | 0.982 | scoring, not estimation |
| delay-aware / oracle | true contract | ~1.0 | ~1.0 | given |

The probe-augmented learned method adds the probe budget to the trained-model-only
tier and stays with the passive learned methods at the bottom of the ladder. It does
**not** reach the grammar-knowledge belief (0.458/0.673) or the pool family
(0.93/0.98), because those convert their privilege into a discrete SCORING problem the
weak response model can solve, while this method must ESTIMATE the continuous map and
the fair probe budget cannot make it identifiable. It remains the first genuinely
unprivileged learned method to be trained end-to-end on the belief family's own
excitation — and it fails honestly, which sharpens rather than softens the benchmark
claim.

## Honest privilege statement

What this method assumes, and nothing more:

1. a **bounded probe budget** — six raw basis pulses of amplitude 0.5 at episode start.
   This is a privilege **SHARED with the probe family** (bounded active probing), not
   with the pool/grammar families;
2. the **contract-independent response calibration** (`alpha`/`sigma`), measured on the
   unwrapped identity environment and shared by every tournament method;
3. a **model trained on hash-disjoint contracts** (the UP-OSI privilege declared in the
   registry).

It does **not** use the declared 9-contract pool and does **not** use grammar knowledge:
the model regresses the continuous contract parameters, it never scores a declared
candidate set or enumerates a grid. The probe schedule is the agent's own known
excitation, so reading it back out of the accumulated evidence is legitimate
self-knowledge. A test asserts the eval adapter's constructor takes no
contract/pool argument, and in the runner the true contract configures the hidden
environment only.

## Claim boundary

- Unprivileged at evaluation: the adapter observes only its own raw actions (probe
  pulses, then encoded task actions) and calibrated tcp-pose responses; it never
  receives the true contract or a pool (test-enforced).
- Matched calibration, backbone, contracts, seed, and probe settings (budget 6,
  amplitude 0.5) with the belief probe family, for direct comparability. Grammar/pool
  are NOT used. No eval-contract tuning; probe settings and the training grammar
  (`max_lag=2`) are fixed a priori.
- Real PickCube/PushCube GPU rollouts through the frozen Gate 1 PPO backbones; 34,560
  training + 4,320 held-out collection env-steps, matched to the OSI/recurrent runs.
  Held-out identification is a 96-sequence, two-task average.
- Identifiability limits (declared, measured, not scored around): **scale** is
  non-identifiable under the weak `pd_ee_delta_pose` response (±25% ≈ 0.40; ~0.6×
  attenuation, ActionABI-confirmed); **gripper** inversion is unobservable from the pose
  response (~chance); **frame** is degenerate under the identity rotation.
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the
  identity-rotation wrapper; no external-benchmark or hardware claim is made. `long_lag`
  was not evaluated end-to-end (the reactive families collapse there; probing is not the
  lever on lag).
