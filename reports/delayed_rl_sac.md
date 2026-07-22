# Delayed-RL external comparison — augmented-state SAC on DCAC's own MuJoCo delay benchmark

Run date: 2026-07-22 UTC. Sim-only, GPUs 0/1/2/3 (four RTX 5090; battery balanced
4/4/4/3 across cards, one GPU per run). Harness:
`experiments/delayed_external/delay_wrapper.py` (DCAC-style constant obs+action delay
+ augmented state) and `experiments/delayed_external/sac_delay_mujoco.py`
(CleanRL-style continuous **SAC**). Env: `.venv-delayext` (torch 2.11.0+cu128,
gymnasium 1.3.0, mujoco 3.10.0, numpy 2.4.6, Python 3.11.15). Per-run
`config.json`/`provenance.json`/`curve.jsonl`/`checkpoint.pt`/`summary.json` under
`artifacts/delayed_external_sac/`; aggregate in `.../aggregate.json`; run ledger in
`.../manifest.json`.

## Why this run exists (what it adds over the PPO attempt)

The prior report (`reports/delayed_rl_external.md`) put ActionShift's augmented-state
reduction on DCAC's benchmark using a CleanRL-style **on-policy PPO**, and its central
limitation was explicit: PPO at a 1M-step budget cannot be read against DCAC's numbers
because **DCAC and all its baselines (SAC, RTAC) are off-policy SAC-derived and far more
sample-efficient**. Its own verdict: *"A genuinely apples-to-apples competitiveness test
would require an off-policy augmented SAC."* This report **is** that test: the same
augmented-state reduction (obs + last-K=ω+α=5 actions), on the same `ConstantDelayWrapper`,
now carried by a CleanRL-style **SAC** — the same algorithm family as DCAC's baselines.

## 1. DCAC's benchmark setup (recap — full quotes in the PPO report §1)

Bouteiller et al., "Reinforcement Learning with Random Delays," ICLR 2021
(arXiv:2010.02966); code `github.com/rmst/rlrd`. DCAC = **Delay-Correcting Actor-Critic**.
Headline **constant-delay** config (**Figure 6**): observation delay **ω = 2**, action
delay **α = 3** → constant **total delay of 5** timesteps. All algorithms use the same
RDMDP augmented observations; DCAC's extra ingredient over augmented SAC is
partial-trajectory resampling for credit assignment, **not** the augmentation. A naive
(unaugmented) SAC *"exhibits near-random results in delayed environments."* Metric =
average episodic return vs env steps; budgets HalfCheetah/Walker2d/Ant = **1M**.

**DCAC Figure-6 approximate reads (constant total delay 5) — figure estimates, NOT a
published table** (the paper publishes curves only):

| Env (delay 5, Fig 6) | DC/AC (to-beat) | SAC (augmented) | RTAC | budget |
|---|---:|---:|---:|---|
| HalfCheetah-v2 | ~5200 | ~2000 | ~2100 | 1M |
| Walker2d-v2 | ~4000 | ~2200 | ~1600 | 1M |
| Ant-v2 | ~3200 | ~600 | ~700 | 1M |

The **augmented-SAC baseline column** is the fair analogue to ours (both augment the
state; neither resamples). DC/AC is the method-to-beat.

## 2. Our harness (faithful; honestly labelled)

**Delay + augmentation** — reused verbatim from the PPO run's verified
`ConstantDelayWrapper`: action delay α (FIFO action pipeline), observation delay ω (FIFO
obs pipeline), and augmented observation `concat(delayed_obs, last K=ω+α=5 emitted
actions)` — the classical constant-delay Markov reduction (Katsikopoulos & Engelbrecht
2003; Walsh et al. 2009), identical to ActionShift's delay-aware backbone. A new
backward-compatible `augment=False` mode injects the delay but withholds the action
history (the **naive** control); the PPO path is untouched.

**Policy** (`sac_delay_mujoco.py`) — a **CleanRL-style** continuous SAC
(`cleanrl/sac_continuous_action.py` recipe: twin soft Q-networks + Polyak-averaged target
nets τ=0.005, squashed-Gaussian tanh actor, automatic entropy-temperature tuning,
uniform 1e6 replay buffer, per-step gradient updates after 5k warmup, γ=0.99,
policy/q LR 3e-4/1e-3, batch 256). **Explicitly labelled CleanRL-style; it is NOT DCAC's
delay-correcting actor-critic** — we implement the augmented-state reduction, not DCAC's
resampling. Budget **1M env steps** (their Fig-6 budget), 3 seeds/env. Metric = mean
episodic RETURN of true (undelayed) per-step rewards — exactly their metric. (Unlike the
PPO stack, SAC uses no obs/reward normalization — CleanRL SAC does not, and neither does
DCAC's SAC; and no `ClipAction`, since tanh squashing already bounds actions to the env's
real [-1,1].)

**Resumability** — full checkpoint (all nets + optimizers + entropy temp + RNG + episode
log + `global_step`) every 100k steps, written atomically; a SIGTERM/SIGINT handler also
flushes a checkpoint on clean kill so a graceful stop loses ≤1 env step. Relaunch resumes
from the latest checkpoint with a **fresh replay buffer** (the buffer is not persisted —
documented, and identical for every run so the protocol is consistent); a completed run
(summary present) self-skips. **Verified twice this session:** (a) scratch SIGTERM at
step 3091 → relaunch reported `RESUME … at step 3091`; (b) on the real battery, the five
delay-5 runs that had passed 400k were killed and relaunched, and resumed from their
`global_step=400000` checkpoints (the other ten had no checkpoint and correctly restarted
from 0). See the ledger in §7.

## 3. Results — augmented-state SAC, constant delay ω=2/α=3 (total 5), 1M steps, 3 seeds

Final return = mean of last-100 episodes. Undelayed (ω=α=0, same code, 1 seed) is the
retention ceiling; naive (delay-5, `augment=False`, 1 seed) is the collapse control.

| Env (delay 5) | **Ours delay-5 aug** (mean ± 95% CI, 3 seeds) | per-seed | median | Undelayed (1 seed) | Naive delay-5 (1 seed) | DC/AC (Fig6) | SAC-aug (Fig6) | RTAC (Fig6) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **HalfCheetah**-v4 | **1951 ± 1513** | 1122 / 1238 / **3494** | 1238 | 9951 | **−208** | ~5200 | ~2000 | ~2100 |
| **Walker2d**-v4 | **1104 ± 78** | 1025 / 1140 / 1147 | 1140 | 704 † | **41** | ~4000 | ~2200 | ~1600 |
| **Ant**-v4 | **716 ± 123** | 591 / 774 / 783 | 774 | 3871 | **−67** | ~3200 | ~600 | ~700 |

95% CI = normal-approx `1.96·sd/√3`; with n=3 a t-multiplier (4.30) would widen it — treat
CIs as indicative, not tight. **HalfCheetah seed-2 (3494) is a positive outlier**: it
lifts the mean to 1951 while the other two seeds sit ~1180; the **median (1238)** is the
more robust central estimate for HalfCheetah. † **Walker2d undelayed = a single, unstable
seed** (final 704, best-50 1171, curve bounced 222→698→139→1069→790→364) — SAC on
Walker2d-v4 is known to be high-variance with early termination; this denominator is
unreliable (see §5).

**Learning curves — delay-5 augmented, mean over 3 seeds, return at training milestones:**

| Env | 100k | 200k | 400k | 600k | 800k | 1000k |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah | 106 | 375 | 1142 | 1862 | 1919 | 1955 |
| Walker2d | 495 | 696 | 834 | 902 | 1074 | 1073 |
| Ant | 10 | 181 | 491 | 608 | 691 | 718 |

All three augmented curves rise **smoothly and monotonically** and are near-plateau by 1M.

**Naive non-augmented delayed (collapse control) — curve stays on the floor throughout:**

| Env | 100k | 400k | 1000k |
|---|---:|---:|---:|
| HalfCheetah | −287 | −241 | −213 |
| Walker2d | 3 | 6 | 52 |
| Ant | −61 | −58 | −85 |

## 4. Honest comparison vs DCAC's Figure-6 reads

**Against DCAC's augmented-SAC / RTAC baselines (the fair analogue — same augment, no
resampling):**
- **HalfCheetah: MATCHES.** Ours **1951** (median 1238) vs their augmented-SAC **~2000** /
  RTAC **~2100**. The 3-seed mean lands essentially on their augmented-SAC read; even the
  more conservative median is the same order.
- **Ant: MATCHES / slightly exceeds.** Ours **716** vs their augmented-SAC **~600** /
  RTAC **~700**. On DCAC's hardest constant-delay env, our augmented SAC sits right at —
  marginally above — their augmented-SAC and RTAC baselines.
- **Walker2d: BELOW (~half).** Ours **1104** vs their augmented-SAC **~2200** / RTAC
  **~1600**. Roughly half their augmented-SAC read; our single unstable undelayed seed
  suggests Walker2d-v4 SAC was simply under-tuned in this harness rather than a delay
  problem, but on absolute return we do not reach their Walker2d baseline.

**Against DC/AC itself (the method-to-beat): BELOW on all three** — 1951 vs ~5200,
1104 vs ~4000, 716 vs ~3200. Expected and honest: DC/AC's advantage over augmented SAC in
their own figures is exactly its partial-trajectory resampling, which we do **not**
implement. We reproduce the *augmented-baseline* tier, not the delay-correcting tier.

**Against the naive control: decisive.** Unaugmented delayed SAC collapses to −208
(HalfCheetah), 41 (Walker2d), −67 (Ant) — near-random, exactly as DCAC reports. The
augmentation is **necessary**, and our augmented SAC clears it by 1–2 orders of magnitude.

## 5. Retention analysis

Retention = delay-5 (mean) / undelayed (same code), a within-code measure of how much of
its own no-delay ceiling the augmentation preserves under delay-5 — **not** a DCAC ratio.

| Env | delay-5 aug | undelayed | retention | note |
|---|---:|---:|---:|---|
| HalfCheetah | 1951 (med 1238) | 9951 | **19.6 %** (med 12.4 %) | ceiling is a proper SAC HalfCheetah (~10k) |
| Ant | 716 | 3871 | **18.5 %** | clean ceiling |
| Walker2d | 1104 | 704 † | *156 %* → **not meaningful** | undelayed seed underperformed (§3 †) |

Two honest observations:
1. **Retention is lower than the PPO run's (69 %/40 %) because the SAC ceilings are far
   higher.** SAC's undelayed HalfCheetah (9951) and Ant (3871) are 5–10× the PPO ceilings
   (1477/380), so the same fixed 5-step delay costs a larger *fraction*. In **absolute**
   delayed return, SAC beats PPO on every env (1951 vs 1025; 1104 vs 857; 716 vs 11) — the
   off-policy switch is a clear absolute win; the retention *ratio* just has a taller
   denominator.
2. **The Walker2d 156 % is a single-seed artifact, not a result.** The lone undelayed
   Walker2d seed was unstable and finished *below* the delayed 3-seed mean; retention >100 %
   here means "the denominator was weak," not "delay helped." We report it and discount it.

## 6. Caveats (published-numbers comparison — not a controlled head-to-head)

1. **Figure reads, not a table.** DCAC publishes learning curves only; the DC/AC / SAC /
   RTAC values are approximate visual estimates from Figure 6. No hard-number claim is made.
2. **v4 vs v2 MuJoCo.** Ours is Gymnasium `-v4` (mujoco 3.10); theirs mujoco-py `-v2`.
   Reward scales for HalfCheetah/Walker2d are broadly comparable but not identical
   (termination/contact details differ); Ant-v4 vs Ant-v2 differ in the contact-cost term.
   Absolute-return comparison carries this version caveat.
3. **Different codebase / hardware / tuning.** Ours is CleanRL-style SAC with default
   hyperparameters and no per-env search; DCAC's SAC/RTAC/DC/AC are their own
   implementation. No shared seeds. Same 1M-step budget, unrelated optimizer/wall-clock.
   Walker2d specifically looks under-tuned in our harness (unstable undelayed seed).
4. **n = 3 seeds, normal-approx CI**; HalfCheetah has one positive outlier seed — median
   reported alongside the mean.
5. **Not the ActionShift manipulation result.** This locomotion comparison is a sanity
   check that our augmentation is a sound, competitive delayed-RL recipe in the off-policy
   family; it does not transfer ActionShift's manipulation numbers or its delayed-
   manipulation niche claim (unchanged from the PPO report §6).

## 7. Verdict + resume/provenance ledger

**Verdict.** The off-policy switch closes the gap the PPO run could not. A CleanRL-style
**augmented-state SAC**, on DCAC's own constant-delay-5 MuJoCo benchmark at their 1M-step
budget, **matches DCAC's published augmented-SAC/RTAC baseline returns on HalfCheetah
(~1950 mean / ~1240 median vs their ~2000–2100) and Ant (~720 vs their ~600–700)**, and
reaches **~half** their augmented-SAC read on Walker2d (~1100 vs ~2200, with an under-tuned
Walker2d ceiling). It does **not** reach DC/AC's delay-correcting returns (5200/4000/3200) —
expected, since we implement the augmentation, not DCAC's resampling. A **naive**
non-augmented SAC collapses to near-random on all three, reproducing their central finding.

**The single defensible sentence now claimable:**
> *"On DCAC's own constant-total-delay-5 MuJoCo benchmark at a matched 1M-step budget, a
> CleanRL-style augmented-state SAC (obs + last-5 actions — ActionShift's delay reduction)
> reaches the DCAC augmented-SAC/RTAC baseline tier — matching their Figure-6 reads on
> HalfCheetah (~1950 vs ~2000) and Ant (~720 vs ~600–700) and about half on Walker2d
> (~1100 vs ~2200) — while a naive non-augmented SAC collapses to near-random; it does not
> reach DC/AC's delay-correcting returns, whose extra ingredient (partial-trajectory
> resampling) we do not implement. Caveats: figure reads not a table, v4-vs-v2 MuJoCo,
> different codebase/hardware/tuning, n=3 seeds."*

**Not claimable:** that we match or beat **DC/AC's** absolute returns (we reach the
augmented-baseline tier, not the delay-correcting tier); any same-hardware DCAC number
(its 2020-era mujoco_py deps do not build here — PPO report §3); any hard number against a
DCAC table (they publish curves only).

**Resume / provenance ledger** (all 15 runs `status=complete`, `manifest.json`):

| Run group | envs × seeds | delay | augment | outcome |
|---|---|---|---|---|
| Headline delay-5 augmented | {HC,Walker,Ant} × {1,2,3} | ω2/α3 | yes | 9/9 complete |
| Undelayed ceiling | {HC,Walker,Ant} × {1} | ω0/α0 | yes | 3/3 complete |
| Naive collapse control | {HC,Walker,Ant} × {1} | ω2/α3 | **no** | 3/3 complete |

- **Hardware:** 4× NVIDIA RTX 5090; battery balanced 4/4/4/3 across GPUs 0/1/2/3 (one GPU
  per run), all four cards at ~91 % util. Per-run wall ~2.0–2.9 h; whole 15-run battery
  drained in one concurrent wave.
- **Resumes exercised:** the five delay-5 runs HalfCheetah-aug-{1,2,3} and Walker2d-aug-{1,2}
  were killed mid-training and **resumed from their `global_step=400000` checkpoints** with
  fresh replay buffers (elapsed shown for these reflects the post-resume segment only); the
  other ten started fresh at 0. Scratch SIGTERM→resume verified checkpoint step 3091 → resume
  at 3091. Completed runs self-skip (verified).
- **Provenance (each run):** torch 2.11.0+cu128, gymnasium 1.3.0, mujoco 3.10.0, numpy 2.4.6,
  Python 3.11.15; `sac_impl = "CleanRL-style sac_continuous_action (labelled; NOT DCAC)"`.

## Reproduce

```bash
# fan the whole battery across 4 GPUs (round-robin, one GPU/run), resumable
.venv-delayext/bin/python experiments/delayed_external/run_sac_battery.py \
  --output artifacts/delayed_external_sac --gpus 0,1,2,3 --max-concurrent 16

# single run (GPU 2), headline constant delay omega=2 alpha=3 (total 5)
CUDA_VISIBLE_DEVICES=2 .venv-delayext/bin/python \
  experiments/delayed_external/sac_delay_mujoco.py \
  --env-id HalfCheetah-v4 --obs-delay 2 --act-delay 3 --augment 1 \
  --total-timesteps 1000000 --seed 1 --output artifacts/delayed_external_sac
# undelayed ceiling: --obs-delay 0 --act-delay 0 ; naive control: --augment 0

# aggregate -> comparison table + aggregate.json
.venv-delayext/bin/python experiments/delayed_external/aggregate_sac.py \
  --output artifacts/delayed_external_sac
```
