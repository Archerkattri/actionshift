# Grasp-channel evidence: closing the gripper wall, testing the scale wall

Run date: 2026-07-21 UTC. Method: version-2 response calibration
(`src/actionshift/adaptation/calibration.py`, `response.py`) wired into both the
pool belief (`hypotheses.py`) and the full-grammar factorized belief
(`factorized_grammar.py`); runners `experiments/run_factorized_slice.py` /
`run_adaptation_slice.py` extended with a `--calibration-version {v1,v2}` flag
and a `--reanchor-period` flag; tests `tests/test_adaptation_grasp_channel.py`
(+ the existing calibration/factorized/stage1 suites). Real ManiSkill GPU
simulation, **GPU 2 only** (`CUDA_VISIBLE_DEVICES=2`). Seeds 20260718/19/20,
100 episodes/contract, 8 envs, the two frozen representative contracts per split,
frozen Gate 1 PPO backbones. Hash-addressed outputs under
`artifacts/adaptation/factorized_slices/` and `.../slices/` (v2 job ids carry a
`calibration_version` + `calibration_sha256` + `reanchor_period`, so no v1
artifact is overwritten). v2 calibrations: `.../calibration/{task}_v2.json`.

## The motivated attack

`reports/adaptation_factorized.md` localized the open unprivileged challenge to
two response-model walls the grammar belief cannot cross with pose evidence
alone:

- **Wall 1 — gripper.** Gripper inversion is invisible to the tcp-pose response,
  so the belief defaults `gripper_inverted=False` and grasps with the wrong sign
  on inverting contracts (Pick/inverted → **0.00**, a total kill).
- **Wall 2 — absolute-target + scale.** Scale is not identifiable from the weak
  `pd_ee_delta_pose` response (≈0.6× attenuation); under an absolute target the
  scale error integrates into unbounded drift, collapsing the all-absolute
  `unseen_composition` split (0.00–0.055) for both tasks.

Both are response-model limits, not hypothesis-space limits, so the lever is
richer evidence — all measured on the **unwrapped identity environment** and
therefore contract-independent task knowledge, exactly like the existing tcp
calibration. Three upgrades were built and are opt-in via calibration
versioning (v1 = the original pose-only linear model, bit-reproducible):

1. **Gripper evidence channel.** The calibration auto-locates the finger-joint
   block in the flat state obs (same `locate_contiguous_slice` discipline as the
   tcp slice, correlated against `agent.robot.get_qpos()` finger joints), fits the
   command→finger-qpos-delta model (sign + gain + R2), and exposes it as a 7th
   response channel. Wired into the pool belief (each hypothesis predicts the
   gripper channel through its `gripper_inverted` flag) and the factorized belief
   (a per-environment binary gripper-sign factor, independent of the pose
   assignment).
2. **Magnitude-dependent tracking gain.** A saturating per-channel gain
   `observed = alpha·c/(1+|c|/c0)` (c0 grid-fit per channel) replacing the
   constant-alpha linear model, opt-in and versioned; the linear R2 is recorded
   alongside for a direct comparison.
3. **Absolute-drift re-anchoring.** Periodic re-anchoring of the tracked absolute
   target's position channels to the observed tcp displacement (a running sum of
   the measured pose deltas — task knowledge), to bound integration drift.

Leakage guard: every calibration + belief constructor still takes **no**
contract/pool argument (test-enforced, extended to the new `reanchor_period`
knob); the v2 calibration is measured on the unwrapped env only.

## Calibration quality (unwrapped, 64 random steps, 8 envs, seed 20260720)

**Gripper channel — strong and identifiable.** Finger block located at obs
offset 7 (the two Panda finger joints, both control modes). Command→finger-delta
fit: **alpha = +0.0093, sigma = 0.0049, R2 = 0.545** (sign positive: +1 opens,
finger delta > 0). This is the highest-R2 channel in the whole calibration —
much sharper than the pose channels (R2 0.09–0.34) — which is exactly why gripper
inversion becomes identifiable when pose does not.

**Magnitude-dependent gain — negligible improvement (honest).** Per-channel pose
R2, linear vs saturating:

| Channel | 0 (x) | 1 (y) | 2 (z) | 3 (rx) | 4 (ry) | 5 (rz) |
|---|---:|---:|---:|---:|---:|---:|
| linear R2 | 0.281 | 0.110 | 0.287 | 0.341 | 0.090 | 0.263 |
| saturating R2 | 0.281 | 0.110 | 0.288 | 0.341 | 0.101 | 0.263 |
| fit `c0` | 5.0 | ∞ | 3.0 | ∞ | 0.3 | 3.0 |

The saturating model gives at most +0.011 R2 (one rotation channel) and is
otherwise identical; most channels fit `c0 → ∞` (the linear limit). An explicit
per-decile gain sweep confirms only a mild ~15–20% gain drop from small to large
commands (e.g. translation-x gain 0.039 at |c|≈0.1 vs 0.034 at |c|≈0.9). On this
response model the ≈0.6× attenuation is essentially **constant in magnitude**, so
a magnitude-dependent gain does not make scale more identifiable. It is retained,
versioned and opt-in, but it is not the lever.

## Per-cell results (before = v1 pose-only; after = v2 gripper+magnitude)

| Cell | v1 (before) | v2 (after) | Δ |
|---|---:|---:|---:|
| Pick / seen, factorized **probed** (3 seeds, 600 eps) | 0.458 | **0.715** [0.678, 0.750] | **+0.257** |
| — gripper_inverted = **True** sub-cell | 0.000 | **0.537** [0.480, 0.592] | **+0.537** |
| — gripper_inverted = False sub-cell | 0.917 | 0.893 [0.853, 0.923] | −0.024 (ns) |
| Pick / seen, factorized **passive** (1 seed) | 0.363 | **0.535** | +0.172 |
| Pick / seen, **pool exact_belief** (regression, 1 seed) | 0.928 | **0.930** | +0.002 |
| Pick / unseen_comp, probed (absolute+inverted) | 0.005 | 0.000–0.005 | ≈0 |
| Push / unseen_comp, probed (absolute) | 0.000 | 0.005 (0.045 w/ reanchor=8) | ≈0 |

v1 numbers are the matched-setup factorized report cells. v2 Pick/seen probed is
three seeds (20260718/19/20); the unseen and pool cells are one seed at the
promotion-pruning rule (all < 0.3 improvement or a regression check).

### Wall 1 (gripper) — substantially closed

The gripper evidence channel converts the Pick/seen gripper-inverting contract
from a **total kill (0.000) to 0.537** [0.480, 0.592] pooled over three seeds,
lifting the whole Pick/seen probed cell **0.458 → 0.715** with **no regression on
the non-inverting contract** (0.917 → 0.893, intervals overlap). The belief now
identifies the gripper sign from the finger-delta channel and grasps correctly.
The residual gap to the non-inverting 0.893 is **Wall 2 leaking in**: this
representative also carries the harder scale vector (0.5, 2.0, 1.5, 0.75, 1.25,
0.6), so its imperfect scale identification caps it below the clean cell — a
scale limit, no longer a gripper limit. The passive belief improves in lockstep
(0.363 → 0.535). This is the first time an unprivileged belief succeeds on a
gripper-inverting grasp contract.

### Pool regression — clean

The 9-pool `exact_belief` on Pick/seen holds at **0.930** (v1 0.928): the gripper
channel disambiguates the pool's gripper-inverted candidates by evidence instead
of pose alone and does not perturb the ceiling. No degradation.

### Wall 2 (absolute-target + scale) — unmoved

The all-absolute `unseen_composition` split stays collapsed for **both** tasks
under every v2 configuration tried: Pick 0.000–0.005 (both contracts are also
gripper-inverted, but the arm never reaches the object before drift fails, so the
gripper fix cannot express), Push 0.005 with the magnitude gain and 0.000–0.045
across re-anchoring periods {1, 8}. Diagnosis, honestly:

- The magnitude-dependent gain does not sharpen scale (R2 unchanged), so scale
  stays non-identifiable and the encoded absolute target stays mis-scaled.
- Re-anchoring bounds the *argument* growth of the cumulative target but not the
  scale *factor* error; with a wrong scale factor `r`, the wrapper's executed
  delta follows a recursion `E_t = r·E_{t-1} + r·Δc_t` that is not stabilized by
  re-anchoring (and is mildly destabilized at period 1 for the large-scale Push
  contracts). It is implemented, versioned and off by default, but it is not a
  usable lever on this response model.

Wall 2 is therefore **unmoved**: closing it needs scale to become identifiable,
which the pose response does not support at per-step SNR ~1 — consistent with the
factorized report and the ActionABI bridge.

## Updated open-challenge statement

Grammar knowledge plus a **gripper evidence channel** closes one of the two
localized walls: the gripper-sign field that the tcp-pose response cannot reveal
is now recovered from the finger-qpos response (R2 0.545), converting the
Pick/seen gripper-inverting kill (0.00) into 0.54 and the whole probed cell into
0.715 — with no pool and no regression on the pool belief. The **scale wall
stands**: on the weak `pd_ee_delta_pose` response the attenuation is constant in
magnitude, a magnitude-dependent gain buys ~0 R2, and neither it nor target
re-anchoring rescues absolute-target control; the all-absolute split remains ≈0.

So the residual unprivileged difficulty is now pinned to a **single** named,
response-model-limited quantity — **exact scale under absolute-target control** —
rather than to the gripper sign or the hypothesis-space size. What the pool
privilege still buys beyond grammar-plus-gripper-evidence is exactly the scale it
is handed for free. The motivated next move this points to (not run here): a
richer *pose* observable that makes scale identifiable (e.g. an object-relative
or goal-relative state channel, or a controller-internal target readout), not a
better tracking model of the same attenuating tcp-delta response.

## Wall verdict

- **Wall 1 (gripper): closed / substantially closed.** 0.00 → 0.54 on the killer
  sub-cell, +0.26 on the full Pick/seen probed cell, 3-seed Wilson intervals
  separated from the v1 zero, no pool regression.
- **Wall 2 (absolute + scale): unmoved.** Magnitude gain ≈0 fit gain;
  re-anchoring no usable lever; unseen-composition stays ≈0. Reported honestly.

## Claim boundary

- Unprivileged at evaluation: the adapter sees only its raw actions and calibrated
  tcp-pose + finger-qpos responses; constructors take no contract/pool argument
  (test-enforced). The gripper/magnitude/re-anchor knowledge is all measured on
  the unwrapped identity environment.
- Matched calibration-family + backbone + contracts + seeds with the v1 cells; the
  only change is the v2 evidence. v1 remains bit-reproducible (unversioned
  calibration path unchanged; v2 lives in separate `_v2.json` files and hash-
  addressed job ids).
- Three seeds (600 eps) on the promoted Pick/seen probed cell; one seed on the
  unmoved unseen cells and the pool regression by the preregistered pruning rule.
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the
  identity-rotation wrapper; no external-benchmark or hardware claim.
- An unrelated pre-existing registry test (`tests/envs/test_tasks.py`, a
  `pull_cube` task not in the frozen set) fails independently of this work; none
  of the files changed here touch the task registry.

## Code / test summary

- `calibration.py`: `ResponseCalibration` v2 fields (`version`, `gain_model`,
  `alpha_c0`, `saturating_fit_r2`, `gripper_start/alpha/sigma/fit_r2`, all
  defaulting to v1); `SupportsGripperProbe`; `fit_saturating_response`,
  `fit_gripper_response`; `calibrate_response(calibrate_gripper=, magnitude_gain=)`;
  `response_from_observations` appends the finger-delta channel.
- `response.py`: `ResponseModel` gains `alpha_c0` (saturating `expected()`) and
  `gripper_alpha/sigma` (`gripper_log_likelihood`).
- `hypotheses.py`: pool belief folds the gripper channel into the likelihood.
- `factorized_grammar.py` (additive, coexists with the C++ scorer backend): binary
  gripper-sign belief, magnitude-gain in the cell contribution, `reanchor_period`.
- `maniskill.py`: `ManiSkillPoseProbe.gripper_positions()`; v2 flags threaded.
- Runners: `--calibration-version`, `--reanchor-period`; v2 calibration path +
  hashing.
- Tests: `tests/test_adaptation_grasp_channel.py` (12 cases) — gripper
  locate/fit, saturating fit recovers c0, response extraction, factorized +
  pool gripper-inversion identification, gripper-belief boundary reset,
  re-anchoring, leakage guard. All green; ruff + strict mypy clean.
