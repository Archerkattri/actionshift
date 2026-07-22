# Adaptation tournament, round 1: belief-family methods at protocol strength

Run date: 2026-07-20 UTC. Runner: `experiments/run_adaptation_slice.py`; analysis:
`experiments/analyze_adaptation.py`; artifacts: `artifacts/adaptation/slices/` (hash-addressed
episode JSONL + summaries), `artifacts/adaptation/tournament.json`. Frozen Gate 0 PPO backbones;
100 episodes per contract, two frozen contracts per split; seeds 20260718/19/20; Wilson 95%
cells; paired bootstrap on episode pairs matched by (seed, contract, episode index).

## Success rates (pooled; 3 seeds / 600 episodes unless marked)

| Task / split | oracle* | entropy_probes | fixed_probes | exact_belief | random_probes | no_adapt* | up_osi (passive) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pick / seen | 1.000 | **0.987** | 0.928 | 0.928 | 0.925 | 0.000 | 0.0† |
| Pick / unseen comp. | 0.995 | 0.975 | 0.970 | 0.958 | 0.980 | 0.000 | 0.0† |
| Pick / long lag | 0.027 | 0.045¹ | 0.037 | 0.022 | — | 0.005 | — |
| Push / seen | 1.000 | **1.000** | 0.982 | 0.983 | 0.942 | 0.003 | — |
| Push / unseen comp. | 1.000 | 0.987 | 0.980 | 0.995 | 0.995 | 0.000 | — |
| Push / long lag | 0.153 | — | **0.215** | 0.117 | — | 0.000 | — |

\* Gate 1 reference (600-episode cells). † 16-episode probe cells. ¹ one seed / 200 episodes.

## Headline result — the first preregistered promotion-gate clear

**Entropy-guided probing beats fixed probing by +5.8 points [+3.7, +8.2] on Pick/seen** (paired
95% bootstrap, 3 matched seeds, 600 paired episodes, interval excludes zero) — clearing the
preregistered "five percentage points over fixed probes" promotion criterion — **with no safety
or cost regression** (entropy probes also incur ~24% less end-effector displacement at the same
6-step budget). The direction replicates on Push/seen (+1.8 [+0.8, +3.0], significant, below the
5-point margin). Entropy probing is also significantly above passive belief on both tasks' seen
split (+5.8 and +1.7, both 3-seed significant) and is the top method in every instantaneous cell.

## Other significant paired results (all 3-seed)

- Random probing can HURT: random − passive = −4.2 points [−6.3, −2.0] on Push/seen —
  uninformative probe steps waste episode time without buying identification.
- fixed − passive on Push/unseen = −1.5 [−2.8, −0.3] (marginal negative); on ceiling-height
  splits probing buys nothing over passive evidence.
- Everything else on unseen-composition is ns — all belief methods sit at the oracle ceiling.

## The discriminative ladder the benchmark now supports

1. Oracle (true contract): ~1.0 instantaneous; collapses under lag (0.027/0.153).
2. Pool-privileged belief family (exact belief, probes): matches the ceiling instantaneous;
   entropy > fixed = passive > random with significant separations; same lag collapse.
3. Passive learned system-ID (UP-OSI-style, unprivileged): 0.0 — held-out identification
   collapses on real responses (perm 0.29, sign ~chance). The random-excitation ablation
   doubles permutation accuracy (0.52) and lifts lag (0.77) but sign stays at chance and
   evaluation stays 0: SHORT-window passive identification is information-limited under the
   weak real response model (calibration R2 0.09–0.34), not merely excitation-limited.
4. No adaptation: ~0.0 everywhere.

## Lag finding (promoted to 3 seeds, 2026-07-21)

**On Push/long-lag, active probing exceeds the passive privileged oracle**: fixed probes
0.215 [0.184, 0.250] (3 seeds, 600 eps) vs oracle 0.153 [0.127, 0.184] (Wilson intervals
touch at the boundary) and clearly above passive exact-belief 0.117 [0.093, 0.145]
(separated intervals). Interpretation: the probe phase flushes the lag pipeline — during
probing, the delayed executions are bounded pulses rather than mis-timed task actions, so
task control begins with a synchronized pipeline. Contract knowledge is therefore NOT the
only lever on the lag split; timing structure matters. On Pick the effect is not
significant (0.037 [0.024, 0.055] vs oracle 0.027 [0.016, 0.043]) — lagged Pick control is
too fragile for any reactive method. Refinement (entropy lag cells, 3 seeds): the
advantage is specific to FIXED scripted probes — entropy probes score only 0.100
[0.078, 0.127] on Push/long-lag (vs fixed 0.215, passive 0.117), consistent with entropy's
documented stateless probe-selection preview ignoring lag, which mis-informs adaptive
selection exactly where lag hypotheses matter. The finding is therefore "scripted probing
flushes the lag pipeline," not "probing helps under lag" generically. The lag splits remain
collapsed overall and delay-aware training remains the open requirement.

## Claim boundary

- The entropy-vs-fixed/passive comparisons are **matched-privilege**: every belief-family
  method shares the same declared 9-contract pool, budget, amplitude, backbone, contracts, and
  seeds. The claim is about the value of information-SEEKING, not about unprivileged adaptation.
- No unprivileged method succeeds yet: passive OSI fails honestly; the recurrent
  (episode-length accumulation) and probe-augmented learned variants are the motivated next
  methods; DualABI and the remaining registry methods still have no trained end-to-end runs and
  earn no claims.
- Long-lag cells and the probe-lag observations are one-seed where marked.
- These are ActionShift-benchmark results under `pd_ee_delta_pose` state control with the
  identity-rotation wrapper; no external-benchmark or hardware claim is made.

## Round 2 (2026-07-21): DualABI, the recurrent challenge, and the bridge

**DualABI — probe-efficiency Pareto win (3 seeds, matched privilege, independently re-analyzed).**
The project's named method, wired end-to-end on the same driver/pool/budget as the probe family
with task-regret-aware probe selection + sticky early stopping, MATCHES entropy_probes' success
on every instantaneous cell (all paired deltas ns: -0.003..+0.008) while using **2.89 vs 6.00
mean probe steps (~52% fewer) and ~42% less probe displacement**, and is significantly above
fixed_probes/exact_belief on Pick/seen (+5.5 [3.2, 7.8]). This clears the preregistered
success-versus-action-cost Pareto promotion criterion. It does NOT beat entropy on raw success
(no such claim), and the week-one proxy negative (DualABI < fixed pulses on success) stands on
the record. Details: `reports/adaptation_dualabi.md`.

**Recurrent unprivileged challenge — honest negative, sharpened.** Episode-length accumulation
(GRU + running least-squares sufficient statistics, deep supervision, both excitations, budget-
matched) lifts DISCRETE fields with horizon (lag identification 0.75->0.92, target 0.59->0.69)
but the continuous permutation/sign map plateaus (~0.35 / ~chance) and end-to-end success stays
~0.0. The unprivileged wall is now precise: discrete hypothesis scoring succeeds; continuous
map ESTIMATION is information-limited at per-step SNR~1 under the weak real response model.
Next motivated method: probe-augmented recurrent identification. Details:
`reports/adaptation_recurrent.md`.

**Bridge to ActionABI.** The wrapper now exports labeled real-sim traces; ActionABI's C++ CLI
achieves 0.97/0.98 permutation/sign supervised accuracy on the lag-0/random-excitation stratum
(pre-fix scorer; the current post-fix OVERALL figures are 0.80/0.84 across all strata — cite those for aggregate claims), and
the run surfaced two real ActionABI defects (a lag-observable scorer bug; calibration not
robust to systematic response-model bias). Details: `../actionabi/reports/labeled_sim_traces.md`.

**Positioning.** Independent literature check (Related work section of README.md; full matrix in git history: docs/positioning_litcheck.md): no prior benchmark
isolating compositional action-interface shift found (checked vs RoboHiMan/ATOM-Bench/
LIBERO-Plus); frame claims as "not aware of", cite ExecSpec first, note Reflective VLA's
calibration-shift perturbation as nearest neighbor; venue fit: NeurIPS D&B / CoRL.

## Round 3 (2026-07-21): full-grammar belief — the privilege gap, localized

The full-grammar factorized belief adapter (`reports/adaptation_factorized.md`) replaces the
9-contract pool privilege with GRAMMAR KNOWLEDGE only (the benchmark's declared finite contract
space; 7-value scale grid covering every frozen contract, test-enforced). Results (probed
variant, 3 seeds on promoted cells): Pick/seen 0.458, Push/seen 0.673 — the first belief method
to succeed with no pool, converting the learned methods' ~0.0 into real success — and 0.990 on
the clean sub-cell (Push, delta-target, non-inverted gripper), matching the pool ceiling. The
unseen (all-absolute) split collapses (0.00-0.055) for a now-precise reason: scale is
non-identifiable under the weak `pd_ee_delta_pose` response (~0.6x attenuation, confirmed
independently by the ActionABI bridge), and scale error integrates into unbounded drift under
absolute-target encoding; the gripper flag is unobservable from pose responses entirely.

**The open challenge is now localized:** what the pool privilege buys beyond grammar knowledge
is exactly {exact scale, gripper sign, and via scale, reliable absolute-target control} — all
RESPONSE-MODEL-limited, not hypothesis-space-limited. Beating the open challenge therefore
requires richer evidence channels (gripper state, grasp outcomes, better tracking models), not
bigger hypothesis spaces or longer learned accumulation. Overhead: +4.7 ms/step adapter cost
over the pool belief (assignment-bound, cacheable); end-to-end remains GPU-sim-bound.

## Round 4 (2026-07-21): the long-lag split is solved

A delay-aware augmented-state PPO backbone (obs + last-4 canonical actions, trained under
per-episode randomized lag {0,1,2,4}; two documented deviations from the official ManiSkill PPO
config; the lag executor unit-tested bit-identical to the wrapper's ActionLag) solves the split
every reactive method failed. 3-seed pooled Wilson cells, 600 eps, vs frozen references:

| Long-lag cell | delay-aware | frozen reference | best frozen probe |
|---|---:|---:|---:|
| Pick, oracle-encode | **0.528** [0.488, 0.568] | 0.027 | 0.037 |
| Push, oracle-encode | **0.415** [0.376, 0.455] | 0.153 | 0.215 |
| Pick, exact-belief | **0.360** [0.323, 0.399] | 0.022 | — |
| Push, exact-belief | **0.387** [0.349, 0.426] | 0.117 | — |

Every interval sits entirely above its frozen reference and above the best frozen-backbone probe
result. The composition claim holds: identification alone cannot fix delay (Gate 1
falsification), delay-aware control alone cannot fix hidden semantics (no-adapt ~0), and the
combination solves both — with INFERRED contracts (belief rows), not only privileged ones.
Competence: oracle-seen Pick 0.715 / Push 0.990 (both clear the 0.5 floor); the curriculum
variant trades the axes (Pick seen 0.950, long-lag 0.455) — randomized-lag training is the
long-lag method, curriculum the better instantaneous policy. Honest nuance: belief-seen drops
under the delay-aware backbone (0.475/0.370) — delay-hedging control smears the within-episode
identification signal; a real method property, disclosed. Claim boundary: NEW backbone — these
cells are never mixed with the frozen-backbone tournament as a method contest; the claim is
that the SPLIT is solvable. Labeled "delay-aware augmented-state PPO (local)"
(Katsikopoulos/Walsh augmented-state reduction); DCAC/D-TRPO cited, not reproduced.
Details: reports/adaptation_delay_aware.md.

## Round 5 (2026-07-21): wave-3 results — fusion, gripper channel, third task

**C++ fusion (reports/cpp_fusion.md).** ActionABI's fixed C++ scoring core now runs as a live
evidence-scoring backend inside the factorized-grammar belief loop (pybind11; per-cell parity
5.7e-14 float64; MAP decisions bit-identical; end-to-end slice within one episode of the torch
backend). Honest benchmark: C++ wins the many-env CPU regime (up to ~13x over torch-32-thread),
torch-CUDA wins at GPU scale (launch-bound ~0.32 ms), and the transfer-inclusive C++ CUDA path
loses everywhere — reproducing ActionABI's own preregistered transfer-inclusive negative.

**Gripper evidence channel (reports/adaptation_grasp_channel.md).** Versioned v2 calibration
adds a contract-independent gripper channel (finger-qpos response, R2 0.545 — the sharpest
channel in the calibration) and a magnitude-dependent gain model (honestly negligible: the
~0.6x attenuation is magnitude-constant). Wall 1 (gripper) substantially closed: the Pick
gripper-inverted kill-cell 0.000 -> 0.537 [0.480, 0.592], full factorized cell 0.458 -> 0.715
[0.678, 0.750], pool belief unregressed (0.930). Wall 2 unmoved: unseen (all-absolute) stays
~0. **The open challenge is now a single named quantity: exact scale under absolute-target
control.**

**Third task (reports/third_task.md).** PullCube-v1 gated in per the preregistered protocol
(Gate 0: competence 1.00, parity diff 0.0000; disclosed success_at_end anomaly under the
identical success_once measure). Gate 1 premise replicates (oracle 1.000/0.998 vs no-adapt
0.000; long-lag 0.322 — more lag-robust than Pick/Push). Cross-task replication of the
load-bearing claims: entropy > fixed **+5.5 (significant) on unseen — a second independent
clear of the 5-point promotion gate** — and DualABI matches entropy at 2.5-2.9 vs 6.0 probe
steps (the Pareto win replicates). Ordering holds on seen (entropy > fixed ~ passive > random);
disclosed anomaly: random probes are benign on PullCube/unseen (0.997), opposite in sign to
Push/seen. The benchmark is now three competent tasks.

## Round 6 (2026-07-21): the unprivileged challenge, closed with a number

Probe-augmented learned identification (reports/adaptation_probe_osi.md) — the last motivated
unprivileged attempt with current evidence channels — is an honest negative that quantifies the
challenge. Same 6-pulse budget, same calibration, no pool, no grammar enumeration
(test-enforced): held-out permutation caps at 0.39 (below even the random-excitation ablation's
0.52), sign never leaves chance, and end-to-end success is ~0.0 on every cell, including the
clean contract the grammar-probe belief reaches 0.917 on. Mechanism: 6 basis pulses let a
belief SCORE a small hypothesis set to ~1.0 but dimensionally underdetermine the continuous
12-dim lagged map a learned method must ESTIMATE, and the running accumulator then dilutes the
6 clean probe steps with 39 weak policy steps. **Active probing is not a substitute for
hypothesis-space knowledge.** The benchmark's method ladder is complete: privileged pool ~1.0;
grammar knowledge 0.46-0.72 (gripper channel closes wall 1; scale-under-absolute remains);
learned methods (passive, recurrent, probe-augmented) ~0.0 — each tier's gap explained,
measured, and reproducible.

## Round 7 (2026-07-21): lag completions — self-correction

The lag-completion studies (reports/lag_completions.md) CORRECT two earlier readings:
(1) **Pipeline-flush is Push-specific, not a lag law** — the preregistered generality prediction
FAILED on PullCube (fixed probes 0.260 fall below both the 0.322 oracle and 0.287 passive
belief; entropy 0.325 ties the oracle). The Round-2/3 "scripted probing flushes the lag
pipeline" finding is hereby downgraded to a task-specific artifact. (2) **fixed > entropy under
lag was a reactive-backbone artifact** — on the delay-aware backbone it reverses (Push entropy
0.467 [0.427, 0.507] > fixed 0.362; the one probe method to beat delay-aware belief 0.387).
Further: probing buys little once the backbone plans through delay (most cells overlap belief),
DualABI's early-stop is efficient under lag on a competent backbone (~1.1-1.3 probe steps) but
false-fires under model mismatch on the frozen one (55-76% single-probe stops at ~0 success),
and probing on the delay-aware backbone HARMS Push/seen (0.02-0.05 vs 0.990) — a disclosed
hazard. Frozen DualABI long-lag cells filled (0.020/0.110, collapse as expected).

## Round 8 (2026-07-21): Peg retry — exclusion upheld honestly

Official-config PPO at 4x the prior budget (~41M steps (~55% of the TRUE official 75M-step Peg budget; an earlier reading cited 250M from the wrong config file — corrected per git history: docs/weakness_sota_recon.md §6); run ended early):
success_once 0.000 at all 17 evaluations and 0/100 on both the final and best-return
checkpoints — below the 0.20 floor. Peg stays excluded; the benchmark stays three competent
tasks. The interface itself is Peg-ready (tcp calibration auto-located, same response regime);
the blocker is backbone competence alone. Caveat recorded: this supports continued exclusion,
it does not refute the published 250M result. Details: reports/peg_retry.md.

## Round 9 (2026-07-21): the weakness wave — every hardware-free gap attacked

- **StackCube-v1 gated in (task 4)** (budget note: trained 25M/1024-envs; TRUE official config is 50M/4096 — competence verdict unaffected, efficiency claims barred; see fourth_task.md correction): mid-difficulty, persistent-placement backbone
  (success_at_end 0.875); ceiling 0.965 vs 0.000; the most lag-fragile task yet (oracle 0.000
  under lag). DualABI's Pareto win replicates a third time (2.9-3.5 vs 6.0 probe steps at
  matched success). Honest anomaly: the entropy>fixed ordering does NOT replicate here —
  passive belief already saturates near ceiling. reports/fourth_task.md.
- **Peg exclusion airtight**: 0/100 at 41M, 71.7M, and the full official 75.7M steps (budget
  corrected 250M->75M per git history: docs/weakness_sota_recon.md §6; segment provenance + determinism check).
  The benchmark is FOUR competent tasks (Pick/Push/Pull/Stack); Peg remains excluded on
  backbone competence alone. reports/peg_retry.md.
- **Real-rotation v2**: the frame axis is de-degenerated (v1 base==tool proven: no-adapt 1.000
  identity vs 0.000 real). Oracle inversion exact (1.000); the pool belief identifies frame
  end-to-end (0.900-0.997, tool cell, 3 seeds); entropy probing erases the small passive
  penalty. Structural finding: the factorized grammar cannot represent tool-frame under real
  rotation (rotation couples channels, breaking per-channel separability) — frame
  identification requires the pool privilege or a richer evidence model. reports/rotation_v2.md.
- **Scale corrector**: proven exact where its precondition holds (synthetic convergence to
  off-grid scales, both target modes; delta split unperturbed) but the real absolute-mode
  failure is UPSTREAM — discrete permutation/target identification collapses under the
  differenced absolute drive before scale is reachable. The open challenge re-localizes to
  "restore discrete absolute-mode identifiability"; the corrector stands ready as the proven
  downstream refinement. reports/adaptation_scale_corrector.md.
- **ActionABI documentation-agreement**: 27 documented field-labels across the 6 real datasets;
  0 contradictions; the single unique certification (PushT absolute) is documentation-correct;
  23 abstentions consistent with under-determination. Bonus real-world evidence: the
  LeRobot/OXE ecosystems actively disagree on gripper conventions (git history: docs/weakness_sota_recon.md §2)
  — the problem the pair addresses exists in the wild. ../actionabi/reports/documentation_agreement.md.

## Round 10 (2026-07-21): external anchors

- **DCAC delayed-RL comparison (reports/delayed_rl_external.md):** our augmented-state recipe is
  delay-ROBUST on DCAC's exact delay-5 MuJoCo setup (40-69% undelayed retention on
  HalfCheetah/Walker2d where the naive baseline is near-random) but NOT competitive with
  DC/AC-or-SAC absolute returns at 1M steps (the standard on-policy/off-policy gap; our
  undelayed PPO already trails their delayed SAC). Claim boundary set accordingly: the
  mechanism validates; no absolute-return claim vs DCAC. The delayed-MANIPULATION niche remains
  uncovered by their locomotion suite (3 near-misses cited).
- **Transformer-ICL conditional resolved (reports/transformer_icl_adjudication.md):** Vintix
  (the only runnable ICL candidate) adjudicated UNFAITHFUL for ActionShift on three
  architecture/protocol grounds (pinned commit + quotes); the registry exclusion now cites the
  evidence. A faithful ICL baseline requires local training on ManiSkill contract histories.
- **Graded claims consolidated (Claims section of README.md; the pre-merge standalone ledger is in git history: docs/SOTA_CLAIMS.md):** 7 STRONG claims across the pair under
  harsh grading; explicit DO-NOT-CLAIM list; two budget misreads (Peg 250M->75M, StackCube
  25M-vs-50M) caught by our own audits before any claim shipped.

## Round 11 (2026-07-22): the absolute-mode wall is breached

Hold-probe excitation (reports/absolute_excitation.md) — sustained per-channel holds scored with
one telescoped evidence term per window (the telescoping sum undoes the differenced absolute
drive that caused the discrete-identification collapse) — takes the dead all-absolute
unseen-composition cells from 0.005 to **0.722 [0.684, 0.756]** (Push) and 0.000 to
**0.790 [0.756, 0.821]** (Pick, gripper-inverted included), 3 seeds, with exact permutation
recovery (1.00 on all envs), causal controls (probe-only 0.000; flail-controls isolate
identification as the cause), and no delta-split regression (0.710). The grammar-knowledge tier
now succeeds on every split family. Honest residuals: one contract executes at 0.44 despite
exact identification (a continuous absolute-control limit downstream of the now-solved discrete
problem), and probe length is a real knob (24-step probes regress the delta cell; 12 is the
operating point). The open challenge narrows to: continuous control under identified absolute
contracts, and the fully-unprivileged learned tier.

## Round 12 (2026-07-22): imitation is exactly as brittle — and exactly as rescuable

Frozen Diffusion Policy backbones (official ManiSkill IL baseline, clean-interface competence
0.580/0.670, cross-checked against the official evaluator) collapse to ~0.00 under the same
hidden contracts that zero PPO — and the SAME policy-agnostic belief adapters restore them to
their oracle ceilings (exact_belief 0.62-0.735, DualABI 0.627/0.688 with its probe-efficiency
Pareto transferring intact). Headline: brittleness to hidden action-interface shift is a
property of the INTERFACE, not the learning paradigm, and the benchmark's adapters are
paradigm-agnostic rescues. Caveats: DP's native 100-step horizon vs PPO's 50 makes cross-
paradigm ABSOLUTE numbers cross-horizon (the load-bearing relative-collapse and restoration
comparisons are horizon-matched within backbone); demos control-mode-converted via the official
replay tool. Details: reports/imitation_brittleness.md.

## Round 13 (2026-07-22): the off-policy external anchor

Augmented-state SAC on DCAC's constant-delay-5 MuJoCo setup (reports/delayed_rl_sac.md, 15
runs, 3 seeds/env, fanned across all four GPUs, resume-verified): reaches DCAC's
augmented-SAC/RTAC BASELINE TIER — HalfCheetah 1951 vs their ~2000 read, Ant 716 vs ~600-700
(matches/slightly above), Walker2d ~half (under-tuned seed flagged) — while naive SAC collapses
to near-random on all three (reproducing their central finding). It does NOT reach DC/AC's
delay-correcting returns (their resampling mechanism, not implemented). Defensible sentence:
the simple augmented-state recipe is baseline-tier competitive on the established delayed-RL
benchmark; the delayed-MANIPULATION niche (where our ~20x lag-solve numbers live) remains
uncovered by that literature. Caveats: figure reads (no published table), v4-vs-v2 MuJoCo,
different codebase/tuning, n=3.
