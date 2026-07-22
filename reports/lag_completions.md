# Lag-completion extras: probing on a delay-aware backbone, frozen DualABI under lag, and PullCube pipeline-flush

Run date: 2026-07-21 UTC. GPU 2 only (`CUDA_VISIBLE_DEVICES=2`). Three cheap, targeted
studies that complete the lag story left open by the adaptation tournament (rounds 1-6) and the
delay-aware backbone report.

Runners: `experiments/run_delay_slice.py` (Study A, extended additively to run the probe family on
the delay-aware backbone), `experiments/run_adaptation_slice.py` (Studies B, C — unchanged).
Shared eval loop / probe accounting: `src/actionshift/adaptation/delay_aware.py`
(`evaluate_delay_aware_adapter`, now mirrors `adaptation.maniskill.evaluate_adapter`'s probe-cost
tracking). Aggregation: `experiments/analyze_delay_slices.py` (extended with a probe-cost column).
Artifacts: hash-addressed per-contract episode JSONL + summaries under
`artifacts/adaptation/delay_slices/` (Study A) and `artifacts/adaptation/slices/` (Studies B, C).
100 eps/contract, two frozen contracts per split; Wilson 95% cells; primary seed 20260718, cells
clearing 0.3 (or interesting-and-close) promoted to seeds 20260719/20 (600 eps pooled).

**Backbones.** Study A uses the *randomized-lag* delay-aware augmented-state PPO backbones (the
headline long-lag backbones from `reports/adaptation_delay_aware.md`): PickCube
`3523a7f717ddd9c2`, PushCube `d6d1a1ce8d1784af`. Studies B, C use the frozen Gate 0 backbones (the
tournament backbones). The probe family shares the same exact-belief driver, declared 9-contract
pool, budget (6), and amplitude (0.5) as the frozen tournament — matched privilege throughout.

**Honesty boundary (carried from the delay-aware report).** The delay-aware backbone is a NEW
policy; Study A cells are *not* a like-for-like method contest against the frozen-backbone
tournament rows. Study A's internal comparison (probe family vs delay-aware exact-belief / oracle,
all on the *same* delay-aware backbone) IS matched. Frozen-backbone reference numbers are printed
for context only.

---

## Study A — probe family on the delay-aware backbone (long-lag + seen sanity)

Does active probing still buy anything once the backbone itself plans through the pipeline? Long-lag
cells are 3-seed pooled (n=600, 2 contracts × 100 × 3 seeds); every probe cell cleared 0.3 on seed
20260718 and was promoted. Seen cells are 1-seed sanity (n=200), following the delay-aware report's
convention for the belief-family seen checks.

### Long-lag (lag 2 + lag 4), 3-seed pooled

| Task / method | success | Wilson 95% | lag2 | lag4 | probe steps | probe disp | delay-aware ref |
|---|---:|---|---:|---:|---:|---:|---|
| Pick / oracle-encode | 0.528 | [0.488, 0.568] | 0.657 | 0.400 | 0.00 | 0.0000 | (backbone ceiling) |
| Pick / exact-belief | 0.360 | [0.323, 0.399] | 0.453 | 0.267 | 0.00 | 0.0000 | (identify-then-act base) |
| Pick / fixed_probes | **0.377** | [0.339, 0.416] | 0.473 | 0.280 | 6.00 | 0.0531 | ≈ belief |
| Pick / entropy_probes | **0.377** | [0.339, 0.416] | 0.457 | 0.297 | 6.00 | 0.0123 | ≈ belief |
| Pick / dualabi | **0.402** | [0.363, 0.441] | 0.597 | 0.207 | **1.28** | 0.0000 | ≈ belief, ~1 probe step |
| Push / oracle-encode | 0.415 | [0.376, 0.455] | 0.530 | 0.300 | 0.00 | 0.0000 | (backbone ceiling) |
| Push / exact-belief | 0.387 | [0.349, 0.426] | 0.593 | 0.180 | 0.00 | 0.0000 | (identify-then-act base) |
| Push / fixed_probes | 0.362 | [0.324, 0.401] | 0.357 | 0.367 | 6.00 | 0.0532 | ≈ belief (slightly below) |
| Push / entropy_probes | **0.467** | [0.427, 0.507] | 0.623 | 0.310 | 6.00 | 0.0136 | **ABOVE belief** |
| Push / dualabi | 0.397 | [0.358, 0.436] | 0.517 | 0.277 | **1.11** | 0.0000 | ≈ belief, ~1 probe step |

Frozen-backbone long-lag references (3-seed, context only): Pick fixed 0.037 / entropy 0.025 /
belief 0.022; Push fixed 0.215 / entropy 0.100 / belief 0.117.

*Probe-displacement reads ~0 for DualABI under lag because it early-stops after a single probe step
and that lone probe's effect is still in the lag pipeline at the step where displacement is
measured (the observed response reflects the pre-probe action). Reported as measured.*

### Seen sanity (lag 0), 1-seed (n=200)

| Task / method | success | Wilson 95% | probe steps | probe disp |
|---|---:|---|---:|---:|
| Pick / oracle-encode | 0.715 | [0.649, 0.773] | 0.00 | 0.0000 |
| Pick / exact-belief | 0.475 | [0.407, 0.544] | 0.00 | 0.0000 |
| Pick / fixed_probes | 0.440 | [0.373, 0.509] | 6.00 | 0.0656 |
| Pick / entropy_probes | 0.330 | [0.269, 0.398] | 6.00 | 0.0164 |
| Pick / dualabi | 0.425 | [0.359, 0.494] | 2.72 | 0.0118 |
| Push / oracle-encode | 0.990 | [0.964, 0.997] | 0.00 | 0.0000 |
| Push / exact-belief | 0.370 | [0.306, 0.439] | 0.00 | 0.0000 |
| Push / fixed_probes | **0.040** | [0.020, 0.077] | 6.00 | 0.0656 |
| Push / entropy_probes | **0.020** | [0.008, 0.050] | 6.00 | 0.0259 |
| Push / dualabi | **0.050** | [0.027, 0.090] | 2.41 | 0.0109 |

### Study A answers to the three posed questions

1. **Does probing add anything once the backbone handles delay (vs delay-aware exact-belief
   0.360/0.387)?** *Mostly no, with one real exception.* On Pick, fixed/entropy 0.377 and dualabi
   0.402 all overlap belief 0.360 — probing buys nothing; identification was never the Pick
   bottleneck. On Push, **entropy_probes 0.467 [0.427, 0.507] is a genuine gain over belief 0.387
   [0.349, 0.426]** (separated Wilson intervals, +0.08), while fixed slightly regresses (0.362). So
   active probing on a delay-aware backbone is neutral-to-mildly-positive on long-lag, and only
   entropy converts.
2. **Does the frozen "fixed > entropy under lag" ordering persist or vanish?** *It vanishes and
   reverses.* On the frozen reactive backbone, fixed (0.215) beat entropy (0.100) on Push/long-lag
   — attributed to entropy's lag-blind stateless preview mis-informing selection. On the
   delay-aware backbone the order flips: Push **entropy 0.467 > fixed 0.362**, and Pick
   entropy ≈ fixed (0.377 = 0.377). The fixed-probe "pipeline-flush" edge was a property of the
   REACTIVE backbone, not of lag: once the controller plans through the delay, entropy's
   information-seeking is no longer penalized and is at least as good as scripted probing.
3. **Does DualABI's early-stop / probe-efficiency hold under lag?** *Yes, and here it is honest.*
   DualABI matches belief/entropy success (Pick 0.402, Push 0.397) while spending **~1.1-1.3 probe
   steps vs the full 6** — 73% (Pick) / 89% (Push) of episodes stop after a single probe step
   (histograms below). On the competent delay-aware backbone the early stop is efficient *without*
   being a false-confidence trap (contrast Study B).

**Seen-split disclosure.** On Pick/seen, probing modestly degrades vs belief (0.44/0.33/0.43 vs
0.475) but competence roughly holds. On **Push/seen probing collapses** (fixed 0.040, entropy 0.020,
dualabi 0.050 vs oracle 0.990): the near-perfect delay-hedged Push controller cannot absorb 6
disruptive probe steps on a short horizon (belief-seen was already only 0.370 — the delay report's
documented "delay-hedging smears the identification signal" — and prepending active probes makes it
far worse). An honest, disclosed hazard of probing on top of a delay-aware backbone.

### DualABI early-stop distribution — delay-aware backbone (3-seed, n=600)

| Task | mean steps | prob. steps histogram | success by stop-step |
|---|---:|---|---|
| Pick / dualabi / long-lag | 1.28 | {1: 438, 2: 155, 3: 7} | 1→0.363, 2→0.510, 3→0.43 |
| Push / dualabi / long-lag | 1.11 | {1: 535, 2: 65} | 1→0.398, 2→0.385 |

---

## Study B — frozen-backbone DualABI on long-lag (the missing tournament cells)

The tournament never evaluated DualABI under lag ("expected to bind DualABI equally"). These are the
missing cells (frozen Gate 0 backbone, seed 20260718, n=200 — collapse confirmed at 1 seed, and the
belief/fixed/entropy references it sits against are already 3-seed).

| Task / method | success | Wilson 95% | mean probe steps | frozen refs (3-seed) |
|---|---:|---|---:|---|
| Pick / dualabi / long-lag | **0.020** | [0.008, 0.050] | 1.61 | belief 0.022, fixed 0.037, entropy 0.025 |
| Push / dualabi / long-lag | **0.110** | [0.074, 0.161] | 1.28 | belief 0.117, fixed 0.215, entropy 0.100 |

### Early-stop distribution under model mismatch (frozen backbone, n=200)

| Task | histogram (probe steps) | success by stop-step |
|---|---|---|
| Pick / dualabi | {1: 109, 2: 62, 3: 26, 4: 3} | 1→3/109, 2→1/62, 3→0/26, 4→0/3 |
| Push / dualabi | {1: 151, 2: 41, 3: 8} | 1→16/151, 2→5/41, 3→1/8 |

**Verdict.** Frozen DualABI collapses under lag exactly like the rest of the belief family (Pick
0.020 ≈ belief 0.022; Push 0.110 ≈ belief 0.117, below fixed 0.215) — the round-1 expectation is
confirmed. The scientifically interesting part is the **early-stop behavior under model mismatch**:
55% (Pick) / 76% (Push) of lagged episodes stop after a *single* probe step (mean 1.3-1.6), yet
success stays ~0 regardless of when it stopped. This is an honest characterization of task-regret
probing under mismatch: the stateless, lag-blind regret preview reads low regret almost immediately
(the MAP contract looks task-adequate under an instantaneous preview that *ignores* lag), so DualABI
commits to task control early — a **false-confidence early stop into a doomed episode**, because the
reactive backbone cannot execute through the delay no matter how good the identification. Early
stopping is trustworthy only when paired with a backbone that can act on the identified contract
(Study A) — not under model mismatch.

---

## Study C — PullCube pipeline-flush generality (frozen backbone, 3-seed, n=600)

Does "scripted fixed probing flushes the lag pipeline and beats the passive oracle" replicate on a
third task? The pipeline-flush prediction: fixed should exceed the oracle 0.322; entropy should not.

| Method | success | Wilson 95% | probe steps | probe disp |
|---|---:|---|---:|---:|
| oracle (privileged, ref) | 0.322 | [0.286, 0.360] | 0.00 | — |
| exact_belief (passive) | 0.287 | [0.252, 0.324] | 0.00 | 0.0000 |
| **fixed_probes** | **0.260** | [0.227, 0.297] | 6.00 | 0.0532 |
| **entropy_probes** | **0.325** | [0.289, 0.363] | 6.00 | 0.0171 |
| no_adapt (ref) | 0.000 | [0.000, 0.006] | — | — |

**Verdict — the prediction fails; the finding does NOT generalize.** Fixed_probes 0.260 falls
**below** the oracle 0.322 *and* below passive exact-belief 0.287 — the opposite of Push, where
fixed (0.215) beat the oracle (0.153). Entropy_probes 0.325 ties the oracle and is the *strongest*
prober here. So on task 3 neither half of the pipeline-flush prediction holds: fixed does not
flush-beat the oracle (it is the worst method, even below passive belief), and it is *entropy*, not
fixed, that reaches the oracle. Interpretation: PullCube is intrinsically more lag-robust (oracle
0.322 vs Push 0.153), so the synchronization benefit of flushing the pipeline is small, while the 6
wasted fixed-probe steps and the highest probe displacement (0.053) cost more than they buy;
entropy at least spends its budget on informative pulses. **The pipeline-flush advantage is
Push-specific, not a general lag remedy.**

---

## What changed in the overall lag narrative

1. **Pipeline-flush is task-specific, not general.** The tournament's round-1 refinement already
   scoped it carefully ("scripted probing flushes the lag pipeline," not "probing helps under lag
   generically"). Study C tightens this further: it is also *not* a general property across tasks —
   on PullCube fixed probing is the *worst* method (0.260 < passive 0.287 < oracle 0.322), inverting
   the Push result. The "active probing beats the passive privileged oracle under lag" headline
   holds *only* on Push.
2. **The frozen "fixed > entropy under lag" ordering was a backbone artifact, not a lag law.** On a
   delay-aware backbone the ordering reverses (Push entropy 0.467 > fixed 0.362; Pick entropy ≈
   fixed). Entropy's lag-blind preview only hurt when the *backbone* could not plan through the lag.
3. **Once the backbone handles delay, extra active identification buys little and can hurt.** Probing
   over delay-aware exact-belief is ~neutral on the Pick identify-then-act path, delivers one real
   gain (Push entropy +0.08), and carries one real hazard (Push/seen collapse). This reinforces the
   delay-aware report's claim that *identification is not the bottleneck once the controller can plan
   through the lag* — the backbone, not the probe strategy, is the lever.
4. **DualABI early-stopping is efficient under lag, but its honesty is backbone-dependent.** With a
   competent backbone (Study A) the early stop is efficient *and* correct (~1 probe step, success ≈
   belief). With a reactive backbone under model mismatch (Study B) the *same* early firing is
   efficient *and wrong* — a false-confidence stop into a doomed episode driven by the lag-blind
   regret preview. A cross-cutting caveat for task-regret probing: early stopping must be gated on a
   backbone that can act on the identification.

The lag splits remain, overall, solved by delay-aware training and not by any reactive/probe method;
these three studies sharpen *where* probing helps (Push long-lag entropy), where it is neutral (Pick
long-lag), where it actively hurts (Push seen; PullCube fixed), and why the frozen pipeline-flush and
ordering results do not generalize.

## Claim boundary

- Study A cells are on a NEW (delay-aware) backbone and are never mixed with the frozen-backbone
  tournament as a method contest; Study A's *internal* probe-vs-belief-vs-oracle comparison is
  matched (same backbone, driver, pool, budget, seeds). Frozen references are context only.
- Studies B, C are on the frozen Gate 0 backbones with the exact-belief driver and declared
  9-contract pool — matched privilege, identical to the tournament.
- Study B long-lag is 1-seed (n=200): collapse is unambiguous and the belief/fixed/entropy
  references it is read against are 3-seed. Study A seen cells are 1-seed sanity. All long-lag probe
  cells and Study C are 3-seed (n=600).
- Separations are stated via non-overlapping Wilson 95% intervals; no paired bootstrap was run for
  the cross-backbone Study A cells (different adapters, reported as independent Wilson cells).
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the identity-rotation
  wrapper; no external-benchmark or hardware claim.

## Reproduce

```bash
# Study A: probe family on the delay-aware backbone (GPU 2)
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/run_delay_slice.py \
  --task pick_cube --method entropy_probes --split long_lag --seed 20260718 \
  --checkpoint artifacts/delay_aware/pick_cube/3523a7f717ddd9c2/final_ckpt.pt \
  --episodes 100 --num-envs 8 --output artifacts/adaptation/delay_slices
# ... {pick_cube:3523a7f717ddd9c2, push_cube:d6d1a1ce8d1784af} x {fixed,entropy,dualabi}_probes
#     x {long_lag,seen} x seeds {20260718,19,20 for long_lag}
.venv/bin/python experiments/analyze_delay_slices.py --slices artifacts/adaptation/delay_slices

# Study B: frozen DualABI long-lag
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/run_adaptation_slice.py \
  --task push_cube --method dualabi --split long_lag --seed 20260718 \
  --episodes 100 --num-envs 8 --output artifacts/adaptation/slices

# Study C: PullCube fixed/entropy long-lag (3 seeds)
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/run_adaptation_slice.py \
  --task pull_cube --method fixed_probes --split long_lag --seed 20260718 \
  --episodes 100 --num-envs 8 --output artifacts/adaptation/slices
```

## Code changes (additive)

- `src/actionshift/adaptation/delay_aware.py` — `evaluate_delay_aware_adapter` now tracks probe
  steps + probe displacement, mirroring `adaptation.maniskill.evaluate_adapter`. For non-probing
  adapters (oracle / exact-belief) `last_probe_mask` is absent, so both accumulators stay zero and
  the existing delay-aware oracle/belief slices are byte-unchanged.
- `experiments/run_delay_slice.py` — `_METHODS` extended with `fixed_probes` / `entropy_probes` /
  `dualabi`; `build_adapter` builds the shared `ExactBeliefDriver` + probe/DualABI adapters (same
  constants as `run_adaptation_slice.py`); probe-specific job fields and `mean_probe_steps` /
  `mean_probe_displacement` are added only for probe runs, so oracle/belief job hashes are undisturbed.
- `experiments/analyze_delay_slices.py` — episode-weighted probe-cost column (absent on
  oracle/belief slices → prints 0).

ruff + strict `mypy src` clean; `tests/test_delay_aware.py` + `tests/test_adaptation_dualabi.py`
pass (16). No new tests added — the probe-tracking loop is a faithful copy of the already-tested
`evaluate_adapter` accounting; the runner/analyzer changes are wiring, not new logic.
