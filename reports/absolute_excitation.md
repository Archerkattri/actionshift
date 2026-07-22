# Absolute-specific hold-probe excitation: breaching the discrete-identification wall

Run date: 2026-07-21 UTC. Method: a sustained **hold-probe** schedule
(`src/actionshift/adaptation/hold_probe.py`, new/additive) + a telescoped
integrated-window evidence term folded by `FactorizedGrammarDriver.update` under a
`hold_mask` (additive, off by default), scored downstream by the existing drift
scale corrector (`scale_corrector.py`). Runner
`experiments/run_factorized_slice.py` gains the `factorized_grammar_hold_probes`
method + `--hold-steps`/`--hold-rounds`/`--force` (resumable; the hold schedule
version enters the job hash only for the new method, so every prior artifact stays
bit-reproducible). Tests `tests/test_adaptation_hold_probe.py` (9 cases). Real
ManiSkill GPU simulation, **GPU 3 only** (`CUDA_VISIBLE_DEVICES=3`). Frozen Gate 1
PPO backbones, v2 calibration (`artifacts/adaptation/calibration/{task}_v2.json`),
100 episodes/contract, 8 envs, the two frozen representative contracts per split,
seeds 20260718/19/20. Hash-addressed outputs under
`artifacts/adaptation/absolute_excitation/` (hold-probe) and
`.../absolute_excitation_before/` (matched pulse-probe baseline). ruff + strict
mypy clean; additive only; grammar + calibration knowledge only (leakage
test-enforced).

## The wall this attacks

`reports/adaptation_scale_corrector.md` re-localized the residual unprivileged
difficulty, precisely and honestly: on the all-absolute `unseen_composition` split
the discrete belief **never locks the permutation and mis-identifies the
absolute-vs-delta target** (collapsing to a `delta`/0.5-floor reading by step ~15),
*upstream* of any scale refinement. Root cause: absolute modes are scored against a
**differenced** drive `raw(t) - raw(t-1)` (bit-faithful to
`CompleteActionDecoder`'s zero-initialized cumulative target), which halves an
already attenuated `pd_ee_delta_pose` response (`alpha ~ 0.02-0.09`, per-step
SNR ~1). The proven scale corrector waits downstream of a correct discrete MAP that
never forms. The prior levers (per-step pulse probe, magnitude gain, re-anchoring)
left the cells at **0.000-0.005** for both tasks.

## The idea: a held raw value separates absolute from delta and un-halves the signal

The per-step pulse probe is matched to a *delta* decode (its response is the
instantaneous drive). A **held** raw value separates the two targets and restores
the absolute signal, derived exactly from `contracts/transforms.py`:

- **Target discriminator.** Hold one raw channel at `+A` for several steps. For a
  *delta* decode the executed canonical is `decode_pose(raw) = A * sign * scale`
  every step -- the response *persists* (the tcp keeps moving). For an *absolute*
  decode the executed canonical is `decode_pose(raw_t) - decode_pose(raw_{t-1})`,
  which is `A * sign * scale` on the excursion step then **zero** while the hold
  continues (the target is constant): the response *decays to zero after the first
  step*.
- **Permutation SNR (telescoping).** Summing the observed response across a held
  window telescopes the differenced absolute drive back to the full undifferenced
  excursion: `sum_t (raw(t) - raw(t-1)) = raw(T) - raw(0) = A`. So the integrated
  observable at semantic channel `i` is `alpha_i * A * sign * scale` -- the full,
  un-halved signal -- restoring the permutation identifiability the per-step
  differenced score destroys.

**Evidence model.** For each held step the driver adds this step's per-mode drive
`base_m(t)` (bit-faithful: `raw(t-lag)` for delta, `raw(t-lag) - raw(t-lag-1)` for
absolute) and the observed pose into per-environment accumulators, and at each
window close folds **one integrated Gaussian term** into the same
`(modes, batch, 6[i], 6[j], signs, scales)` score tensor the MAP already resolves:

```
contribution[m,i,j,s,k] = -0.5 * ( (sum_t obs_i - alpha_i * (sum_t base_m,j) * sign_s * scale_k)
                                    / (sigma_i * sqrt(len)) )^2
```

The `sqrt(len)` is the summed-noise scale. The two targets give *distinct*
integrated predictions -- delta accumulates to `len * command`, absolute telescopes
to the single net excursion -- so this one term sharply separates target mode and,
with the full signal restored, the permutation. Per-step evidence is suppressed on
hold steps (only the integrated term counts, avoiding a correlated double count);
control steps keep the per-step path and alone feed the scale corrector. At the
probe/control transition the tracked absolute target is re-anchored to the achieved
tcp displacement (`_observed_cumulative`, task knowledge), since the probe did not
execute task intent. The schedule is a fixed sweep of bounded raw excitations and
takes no contract/pool argument (leakage test-enforced).

## Synthetic first -- the mechanism is exact (9 tests, all green)

`tests/test_adaptation_hold_probe.py` drives the bit-faithful synthetic hidden
wrapper (decode-then-lag, identity rotation):

- **Identification, clean.** On random *absolute* contracts the telescoped
  hold-window evidence recovers the exact **target, permutation, sign, and on-grid
  scale** from the probe phase alone, on every environment.
- **Identification under attenuation + noise.** With a per-channel, partly
  sign-flipped low gain and per-step noise, target + permutation are still
  recovered on essentially every trial (2 rounds).
- **Off-grid parity with the corrector.** On an off-grid absolute scale the grid
  cannot represent, the hold probe locks the discrete contract and the downstream
  scale corrector closes the residual so executed == intent (max err < 5e-2);
  without the corrector the identified-but-off-grid contract drifts (> 5e-2).
- **Delta preserved.** A delta contract is still identified as delta (no target
  regression).
- **Reproducibility + leakage.** `update` with no `hold_mask` leaves the per-step
  path and hold accumulators bit-untouched; schedule/adapter take no contract arg.

### Honest synthetic caveat that the real run overturns

A sweep at the **real** calibration `alpha` used *instantaneously* (fitting the
model's own gain to a single differenced step) predicts near-hopeless permutation
recovery (exact-perm 0.03-0.17 even at 36-48 probe steps; target 0.5-0.8). That
model is pessimistic: it applies `alpha ~ 0.03` per isolated step. A **coherent
sustained hold** drives the real PD controller far harder than an uncorrelated
random action does, so on the real environment the integrated response is much
stronger than the instantaneous-alpha model implies -- see the direct measurement
below.

## Real-environment verification -- identification is exact, and causal

Instrumenting a real Push/unseen base-frame absolute episode (true
`perm = (5,4,3,2,1,0)`, `target = absolute`), the belief after the 12-step hold
probe recovers the **exact** permutation and target on **all 8 environments**
(`exact_perm = 1.00`, `target_acc = 1.00`) -- versus the pulse-probe belief that
never locked either. So the discrete collapse the prior report diagnosed is gone.

Because a large probe could in principle solve a task by accident, three causal
controls establish that the belief control, not the probe motion, produces success
(Push/unseen, seed 20260718):

| control | Push/unseen |
|---|---:|
| probe-only (60-step hold flail, belief never controls) | **0.000** |
| 24-step hold flail, then identity (no-adapt) control | **0.000** |
| 24-step hold flail, then **belief** control (this method) | **0.725** |

Both tasks give 0.000 for flail-then-identity. Flailing alone does not solve the
task and the perturbation does not set up success; the belief-driven control after
the probe does. Combined with `exact_perm = 1.00`, the gain is genuine
identification, not a probe artifact.

## Before / after on the frozen cells (real GPU sim, 3 seeds, 600 eps/cell)

Before = matched per-step pulse probe (v2 calibration + scale corrector, same
setup). After = the hold-probe method (12-step primary schedule
`h2-r1`, v2 + scale corrector). Wilson 95% intervals.

| Cell | before (pulse) | after (hold-probe) | Δ |
|---|---:|---:|---:|
| **Push / unseen_comp** (all-absolute, gripper-agnostic) | 0.005 | **0.722 [0.684, 0.756]** | **+0.717** |
| **Pick / unseen_comp** (all-absolute + gripper-inverted) | 0.000 | **0.790 [0.756, 0.821]** | **+0.790** |
| Pick / seen (delta; regression) | 0.665 / ~0.715 | **0.710 [0.672, 0.745]** | ~0 (no regression) |

Per-contract (after, hold-probe `h2-r1`, pooled 3 seeds):

- Push/unseen: tool-frame absolute **1.000 [0.987, 1.0]**, base-frame absolute
  **0.443 [0.388, 0.5]**.
- Pick/unseen: tool-frame absolute **0.720 [0.667, 0.768]**, base-frame absolute
  **0.860 [0.816, 0.895]** (both gripper-inverted -- the v2 gripper channel + the
  hold-probe perm/target identification now co-operate).

Before, the pulse baseline is 0.01/0.00 (Push) and 0.00/0.00 (Pick) on the two
contracts -- a total collapse. (Under the default identity rotation the `frame=tool`
representative decodes identically to base, so the factorized base assumption is
correct for it; both contracts are genuine all-absolute cells.)

### The base-frame Push residual is execution, not identification

Push/unseen's base contract sits at 0.443 even though identification is *exact*
(`exact_perm = 1.00`). Its scale `(1.5,1.5,0.5,2.0,0.75,1.25)` and absolute control
over the ~38 post-probe steps are the residual: the scale corrector refines the
continuous scale online but does not always fully converge within the episode. This
is a downstream continuous-control limit on a correctly identified contract -- the
regime the scale corrector was built for -- not the discrete collapse the wall was
about.

### Probe length is the delta-regression knob (sensitivity)

| Cell | pulse (before) | hold `h2-r1` (12 steps) | hold `h2-r2` (24 steps) |
|---|---:|---:|---:|
| Push / unseen_comp | 0.005 | 0.722 [.684,.756] | 0.728 [.691,.762] |
| Pick / unseen_comp | 0.000 | 0.790 [.756,.821] | 0.697 [.659,.732] |
| Pick / seen (delta) | ~0.715 | **0.710 [.672,.745]** | **0.405 [.366,.445]** |

The longer 24-step probe identifies equally but **regresses the identifiable delta
seen cell to 0.405** -- 24 of 50 steps spent probing (and flailing) starves the
delta control that already worked. The 12-step probe removes the regression
(0.710, intervals overlapping the ~0.715 baseline) while still breaching both
absolute cells, because the coherent-hold response identifies the contract in as
few as 12 steps. `h2-r1` is therefore the operating point.

## Wall verdict -- BREACHED

- **The discrete absolute-identification wall is breached.** The permutation +
  absolute/delta target that collapsed on the differenced drive are now recovered
  exactly on the real environment (`exact_perm = 1.00`, `target_acc = 1.00`), and
  the all-absolute `unseen_composition` cells go from a total collapse to a clear
  majority success: **Push 0.005 -> 0.722**, **Pick 0.000 -> 0.790** (3 seeds,
  Wilson intervals far separated from the ~0 baseline), with **no regression** on
  the identifiable delta seen cell (0.710 vs ~0.715). This is the first time the
  unprivileged grammar-knowledge belief succeeds on the all-absolute split.
- **The cause is the hold-probe design, verified causally.** Sustained holds give a
  sharp target discriminator (persist vs decay) and, via telescoping, restore the
  full undifferenced permutation signal; the scale corrector then refines the
  continuous scale downstream exactly as it was proven to. Probe-only and
  flail-then-identity controls are both 0.000, so the gain is identification, not
  probe motion.
- **Honest residuals (moved, not everything solved).** (1) On a correctly
  identified base-frame absolute contract with a hard scale vector, execution still
  caps at ~0.44 within the episode -- a downstream continuous scale/absolute-control
  limit, not the discrete collapse. (2) A too-long probe trades the delta seen cell
  away; the schedule length must be bounded (12 steps here). A natural next move
  (not run) is an adaptive probe that stops once the target reads confidently
  absolute, keeping the breach while shortening the delta-cell probe further.

## Claim boundary

- Unprivileged at evaluation: the adapter sees only its own raw actions and the
  calibrated tcp-pose + finger-qpos responses; the schedule and every belief
  constructor take no contract/pool argument (test-enforced). The hold schedule is
  a fixed bounded excitation (amplitude 0.5, inside the declared bound).
- Matched calibration-family (v2) + backbone + contracts + seeds with the
  factorized / grasp-channel / scale-corrector cells; the only additions are the
  hold-probe method and the `hold_mask` window evidence. Prior artifacts are
  bit-reproducible (the new method + schedule version perturb the hash only for the
  hold-probe cells; `update` with no `hold_mask` is bit-unchanged).
- Three seeds (600 eps/cell) on all reported cells (every cell exceeds the 0.3
  promotion threshold). Per-episode + summary artifacts under
  `artifacts/adaptation/absolute_excitation/`.
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the
  default identity-rotation wrapper; no external-benchmark or hardware claim.

## Code / test summary

- `hold_probe.py` (new): `HoldProbeSchedule` (sustained per-channel holds, versioned
  for hashing) + `FactorizedGrammarHoldProbingAdapter` (probe phase routes evidence
  to the telescoped-window accumulator, withholds tracked-target accumulation while
  probing, re-anchors to the achieved displacement at probe end, feeds only control
  steps to the scale corrector).
- `factorized_grammar.py` (additive): `update` gains `hold_mask`/`window_close`;
  `_accumulate_hold` integrates a window and folds one telescoped Gaussian term at
  close; `_advance_history`/`_mode_bases` helpers; `map_encode` gains a per-env
  `target_mask`; `anchor_tracked_target` snaps the absolute target to the achieved
  tcp displacement. The per-step path is bit-unchanged when `hold_mask` is unused.
- `run_factorized_slice.py`: `factorized_grammar_hold_probes` method,
  `--hold-steps`/`--hold-rounds`/`--force`; hold schedule version in the job hash;
  resumable skip of a completed hash-addressed summary.
- `experiments/run_absolute_excitation.sh`: the resumable GPU-3 campaign (matched
  pulse baseline + primary 12-step schedule + 24-step sensitivity, 3 seeds).
- `tests/test_adaptation_hold_probe.py` (9 cases): schedule construction,
  clean/attenuated+noise identification of target + permutation + scale, off-grid
  parity with/without the corrector, delta preservation, no-`hold_mask`
  reproducibility, leakage guards. All green; ruff + strict mypy clean.
