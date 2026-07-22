# Drift-based scale corrector: an honest negative that re-localizes the last wall

Run date: 2026-07-21 UTC. Method: `src/actionshift/adaptation/scale_corrector.py`
(new, additive) wired into the full-grammar factorized belief
(`factorized_grammar.py`) behind a `scale_correction` flag; runner
`experiments/run_factorized_slice.py` gains `--scale-correction` (perturbs the job
hash only when on, so every prior artifact stays bit-reproducible); tests
`tests/test_adaptation_scale_corrector.py` (12 cases). Real ManiSkill GPU
simulation, **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`). Frozen Gate 1 PPO backbones,
v2 calibration (`artifacts/adaptation/calibration/{task}_v2.json`),
`factorized_grammar_probes` (budget 6, amplitude 0.5), 100 episodes/contract, 8
envs, the two frozen representative contracts per split, seed 20260718 (Pick/seen
promoted to seeds 20260718/19/20). Hash-addressed outputs under
`artifacts/adaptation/factorized_slices/` with `scale_correction` in the job hash.
ruff + strict mypy clean; additive only; no contract/pool leakage (test-enforced).

## The motivated attack and its mechanism

`reports/adaptation_factorized.md` and `reports/adaptation_grasp_channel.md` pinned
the last unprivileged wall to **exact scale under absolute-target control**: on the
all-absolute `unseen_composition` split the grammar belief collapses to ~0 for both
tasks, and the magnitude-gain / re-anchoring levers bought ~0. The insight this
work exploits: under absolute-target encoding with a wrong scale estimate the tcp
does not merely mis-track per step, it **drifts systematically**, and the drift
rate is itself a strong accumulating signal proportional to the per-channel scale
error. Nobody was using it.

**Derivation (from `contracts/transforms.py`).** With the MAP permutation and sign
matched and only the scale wrong, encoding a canonical intent `c_t` under a *fixed*
effective scale `eff_i` and executing it under the true scale `s_i` gives, per
semantic channel `i` and while `eff_i` is held constant (both delta and absolute
target),

```
executed_i(t) = (s_i / eff_i) * c_i(t),   obs_i(t) = alpha_i * executed_i(t) + noise.
```

For an absolute target the executed delta is `s * (S_t/eff_t − S_{t-1}/eff_{t-1})`
in the cumulative intent `S_t`, so a *change* in `eff` within a step injects a large
cumulative jump term `s * S_{t-1} * (1/eff_t − 1/eff_{t-1})`. Holding `eff` fixed
across an integration window kills that term, and the simple integrated ratio then
recovers the true scale, integration averaging the per-step noise that defeats the
instantaneous grid MAP:

```
s_hat_i = eff_i * ( Σ_t |obs_i(t)| ) / ( |alpha_i| * Σ_t |c_i(t)| )  →  s_i.
```

The corrector (`ScaleCorrector`) is a closed loop wrapped around the MAP encode:
after MAP selection it replaces the grid scale with `s_hat` per channel. Fail-safes:
the estimate is bounded to `[0.4, 2.5]`; a channel with integrated command below a
floor is frozen (thin evidence) and keeps deferring to the grid MAP; the window
auto-restarts whenever the effective scale changes and **drops the single jump step**
after any change (tracked via the previous step's `eff`), so every committed window
is measured at one fixed scale; the corrector resets per episode boundary; and probe
steps are excluded (their raw action is not the encoded task command). It uses only
task knowledge — the calibrated `alpha` and the adapter's own commanded intent,
observed response, and the scale it itself chose. Its constructors take **no**
contract/pool/true-scale argument (test-enforced).

## Synthetic convergence — exact, both targets

`tests/test_adaptation_scale_corrector.py` (12 tests, all green; ruff + strict mypy
clean) drives the bit-faithful synthetic hidden wrapper (decode-then-lag, identity
rotation) with **off-grid** true scales the grammar grid cannot represent, so only
the corrector can close the residual:

- **Closed-loop unit convergence.** From `corr = 1`, feeding `obs = alpha·(s/eff)·c`
  recovers the true scale per channel to `1e-3`, including an off-unit MAP base and
  negative `alpha` channels; windowed integration recovers it to within 10% under
  per-step Gaussian noise `σ = 0.05`.
- **End-to-end parity.** With the factorized adapter and `scale_correction=True` on
  an off-grid **absolute** contract (`scale = 1.1, 0.85, 1.4, 0.55, 1.7, 1.05`), the
  effective scale converges **exactly** to the truth and `executed == intent` (mean
  per-step error 0.0 after convergence); the same holds for the **delta** contract.
  Without the corrector the same off-grid absolute contract drifts (max error > 0.05).
- **Fail-safes.** A thin-evidence channel never forms an estimate and defers to the
  grid MAP; a ratio of 10 clamps to 2.5; an episode boundary resets the estimate.
- **Leakage guard.** `ScaleCorrector.__init__`, and the driver/adapter with
  `scale_correction`, take no `contract`/`true_contract`/`pool`/`scale` argument.

So the mechanism is correct: **where the discrete MAP (permutation, sign, target) is
right and only the continuous scale is biased, the corrector recovers scale exactly
and rescues absolute-target parity.**

## Before / after on the frozen cells (real GPU sim, 100 eps/contract, 8 envs)

| Cell | v2 probed (before) | v2 probed + corrector (after) | Δ |
|---|---:|---:|---:|
| **Push / unseen_comp** (both absolute, gripper-agnostic) | 0.005 | 0.005 | **0.000** |
| **Pick / unseen_comp** (both absolute + gripper-inverted) | 0.000 | 0.000 | 0.000 |
| Pick / seen, probed (delta; regression) — 3 seeds, 600 eps | 0.692 | 0.698 | +0.006 (ns) |

Unseen cells: seed 20260718 (1 seed, `<0.3` pruning rule). Pick/seen: three seeds
20260718/19/20 — corrector {0.705, 0.735, 0.655}, baseline {0.640, 0.725, 0.710}.

### Push / unseen_composition — the decisive gripper-agnostic scale test — unmoved

Push does not grasp, so its `unseen_composition` collapse is a **pure** absolute-scale
failure. The corrector leaves it exactly where it was: **0.005 → 0.005**. Pick's
unseen contracts are additionally gripper-inverted (Wall 1), so they stay at 0.000
whether or not scale is corrected; they cannot express the fix and are not the test.

### Pick / seen — no regression, neutral on the identifiable delta split

On the identifiable delta split the corrector is **neutral**: 0.692 → 0.698 over three
seeds (paired difference +0.006, within seed noise; the single-seed 20260718 read
+0.065 was one-seed noise), matching the v2 three-seed 0.715 and never regressing —
exactly the synthetic prediction that where the discrete MAP is already right, scale
correction neither helps nor hurts a delta contract whose per-step error the policy's
closed loop already absorbs.

## Why the wall is unmoved — the assumption that failed (diagnosed, not hidden)

The corrector's derivation assumes the belief's **discrete** MAP — permutation, sign,
and *target* — is correct and only the continuous scale is biased. Instrumenting a
real Push/unseen absolute episode (true `perm = (5,4,3,2,1,0)`, `scale =
(1.5,1.5,0.5,2.0,0.75,1.25)`, `target = absolute`) shows that precondition is **not
met on real absolute contracts** — and it fails the same way with the corrector off,
so this is intrinsic, not corrector-induced:

| step | MAP permutation | MAP scale | MAP target | tcp displacement |
|---:|---|---|---|---:|
| 6  | (4,3,1,2,5,0) | 0.5,2.0,0.5,0.5,0.5,0.75 | absolute | 0.068 |
| 15 | (5,2,1,3,0,4) | 0.5,0.5,0.5,0.5,1.5,0.5 | **delta** | 0.026 |
| 25 | (5,2,0,3,1,4) | 0.5,0.5,0.5,0.5,0.5,0.5 | **delta** | 0.021 |
| 44 | (2,4,1,3,5,0) | 0.5,0.5,0.5,0.5,0.5,0.5 | **delta** | 0.064 |

The belief **never locks the permutation** (it changes at every checkpoint and never
reaches the truth), **mis-identifies the target as `delta`** by step 15, and
**collapses every scale to the 0.5 grid floor**. The tcp barely moves (displacement
~0.02–0.07 throughout) — this is a *mis-identification*, not a scale-driven runaway
the corrector could catch. There is simply no correct absolute encoding underneath
for the scale corrector to refine; its estimate only validates at step 44, on top of
a wrong `delta`/0.5 contract, and is therefore meaningless.

**Root cause.** Absolute-target modes are scored against a *differenced* drive signal
`history[lag] − history[lag+1]` (bit-faithful to `CompleteActionDecoder`'s
zero-initialized cumulative target). Differencing halves an already weak signal — the
v2 calibration gains are tiny and partly sign-flipped (`alpha ≈
0.036, 0.016, 0.023, −0.035, −0.093, −0.048`, `σ ≈ 0.017–0.049`, per-step SNR ~1, fit
R² 0.09–0.34). At that SNR the discrete belief cannot separate the absolute modes'
permutation or even the absolute-vs-delta target, so it settles on the lower-variance
`delta` interpretation with collapsed scale. Scale is the *last* quantity to become
identifiable, but on real absolute contracts **permutation and target fail first** —
upstream of any scale refinement.

## Verdict on the open challenge

- **The last wall is NOT breached; the near-unprivileged tier is NOT complete.**
  The drift-based scale corrector does not recover the all-absolute
  `unseen_composition` split: Push (the pure, gripper-agnostic scale test) stays at
  0.005 → 0.005, Pick stays at 0.000.
- **But the wall is re-localized, more precisely and more honestly.** The prior
  reports framed the residual difficulty as a *single* missing quantity — exact scale
  under a correct absolute MAP. This diagnostic shows that framing was incomplete: on
  real absolute contracts the discrete belief also loses the **permutation** and the
  **target** itself, because the absolute mode's differenced drive halves the SNR. So
  the true residual is not "scale given a correct absolute MAP" but "the discrete
  permutation + absolute/delta identification collapses on the weak differenced
  absolute drive," which is upstream of scale.
- **The corrector is a proven-correct component where its precondition holds.** It
  recovers off-grid scale *exactly* in synthetic (both targets), and on the real
  identifiable delta split it is neutral with no regression (Pick/seen 0.692 → 0.698,
  ns, 3 seeds). It is retained, versioned, and off by default.
- **The next motivated move it points to** (not run here) is therefore not a better
  scale estimator but a way to make the *discrete* absolute-target identification
  survive the differenced-drive SNR loss — e.g. an absolute-mode-specific excitation
  that restores the cumulative signal, or a richer observable (object/goal-relative
  state) that reveals the absolute target directly — after which the scale corrector
  would become the exact downstream refinement it is proven to be.

## Claim boundary

- Unprivileged at evaluation: the adapter sees only its own raw actions and calibrated
  tcp-pose + finger-qpos responses; constructors take no contract/pool argument
  (test-enforced). The corrector's only inputs are the calibrated `alpha`, its own
  commanded intent, its own observed response, and the scale it itself chose.
- Matched calibration-family (v2) + backbone + contracts + seed with the factorized /
  grasp-channel cells; the only change is the additive `scale_correction` flag. Prior
  artifacts are bit-reproducible (the flag perturbs the job hash only when on).
- Single seed (20260718) on the unmoved unseen cells by the preregistered <0.3
  pruning rule; Pick/seen (>0.3) promoted to three seeds (20260718/19/20).
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the
  identity-rotation wrapper; no external-benchmark or hardware claim.

## Code / test summary

- `scale_corrector.py` (new): `ScaleCorrector` — per-env/per-channel windowed
  integrated-ratio scale estimation at a fixed effective scale, jump-step drop on
  effective-scale change, thin-evidence freeze, bounded estimate, per-episode reset.
- `factorized_grammar.py` (additive): `FactorizedGrammarDriver`/`FactorizedGrammarAdapter`
  gain `scale_correction`, `scale_window`, `scale_bounds`, `scale_command_floor`; the
  MAP encode applies the corrected effective scale; `update` feeds the corrector the
  commanded/observed transition and the effective scale used, with `corrector_mask`
  excluding probe steps; probing adapter passes `~last_probe_mask`.
- `run_factorized_slice.py`: `--scale-correction`, threaded into the adapter and the
  job hash (only when on).
- `tests/test_adaptation_scale_corrector.py` (12 cases): closed-loop convergence
  (noiseless + noise, off-unit base), fail-safes (thin evidence, bound, boundary
  reset), end-to-end synthetic parity (absolute recovered only with the corrector;
  delta identifiable unperturbed; delta off-grid also converges), leakage guards.
