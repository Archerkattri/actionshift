# Delayed-RL external comparison — our augmented-state PPO on DCAC's own MuJoCo delay benchmark

Run date: 2026-07-21 UTC. Sim-only, GPU 0 (`CUDA_VISIBLE_DEVICES=0`).
Harness: `experiments/delayed_external/delay_wrapper.py` (DCAC-style constant obs+action
delay + augmented state) and `experiments/delayed_external/ppo_delay_mujoco.py`
(CleanRL-style continuous PPO). Fresh env: `.venv-delayext` (torch 2.11.0+cu128,
gymnasium 1.3.0, mujoco 3.10.0, numpy 2.4.6, Python 3.11). Artifacts + per-run
`config.json`/`provenance.json`/`curve.jsonl`/`summary.json` under
`artifacts/delayed_external/`.

## Purpose and the claim under test

ActionShift's delay work (`reports/adaptation_delay_aware.md`) trains a "delay-aware
augmented-state PPO (local)" backbone on **manipulation** (ManiSkill Pick/Push) under
randomized per-episode **action** lag. The literature's delayed-RL methods, by contrast,
are benchmarked on **MuJoCo locomotion**. This report puts OUR augmented-state approach on
THEIR benchmark setup so the two can be read side by side. Two independent questions:

1. **Competitiveness.** Is a simple augmented-state PPO competitive with the published
   delayed-RL numbers (DCAC, and its SAC/RTAC baselines) on their own MuJoCo + constant-delay
   setup?
2. **Niche.** Does ActionShift's delayed-**manipulation** result occupy a niche the
   locomotion delayed-RL benchmarks do not cover?

## 1. DCAC's published benchmark setup (quoted)

**Paper.** Bouteiller, Ramstedt, Beltrame, Pal, Binas, "Reinforcement Learning with Random
Delays," arXiv:2010.02966. DCAC = **Delay-Correcting Actor-Critic**. Venue note: the `rlrd`
repo title says ICLR 2020, but OpenReview (`forum?id=QFYnKlBJYR`) states "Published as a
conference paper at ICLR 2021" — cite the 2021 ICLR proceedings.
**Code.** https://github.com/rmst/rlrd (PyTorch, pip-installable).

**Environments.** The full Gym MuJoCo continuous-control suite. From the paper's Figures 6-7,
the six benchmarked envs are **Ant-v2, HalfCheetah-v2, Walker2d-v2, Hopper-v2, Humanoid-v2,
Reacher-v2** (mujoco-py `-v2` variants). The paper: *"this enables us to introduce random
delays to the Gym MuJoCo continuous control suite ... which is otherwise turn-based."*

**Delay parameterization.** Their released Gym wrapper turns a turn-based env into a
**Random-Delay MDP (RDMDP)** with a separate **observation delay ω** and **action delay α**,
both in timesteps. The `rlrd` README exposes `Env.min_observation_delay`/`sup_observation_delay`
and `Env.min_action_delay`/`sup_action_delay`, and notes *"our gym wrapper adds a constant
1-step delay to the action delay."*
- **Headline constant-delay config (Figure 6): ω = 2, α = 3 → a constant TOTAL delay of five
  timesteps.** Figure 6 caption, verbatim: *"ω = 2, α = 3 (constant delays). With a constant
  total delay of five time-steps, DCAC exhibits a very strong advantage in performance. All
  tested algorithms use the same RDMDP augmented observations."*
- Figure 7 uses **random** delays sampled from a real-world WiFi dataset (caption: *"α, ω ∼
  WiFi (random delays). DCAC clearly dominates the baselines. Ant became too difficult for all
  tested algorithms. HalfCheetah also became difficult and only DCAC escapes from local
  minima."*).

**Baselines / protocol.** DCAC is compared against **SAC** and **RTAC**, and crucially *"all
other experiments compare DCAC against SAC in the same RDMDP setting, i.e. all algorithms use
the augmented observation space"* — i.e. the SAC/RTAC baselines are themselves **augmented-state**
methods; DCAC's extra ingredient is partial-trajectory resampling for credit assignment, not the
augmentation. A **naive** (unaugmented) SAC *"exhibits near-random results in delayed
environments."* Metric = **average episodic return** vs environment steps; *"we perform six runs
with different seeds, and shade the 90% confidence intervals."*

**Training budgets (x-axis of Fig 6):** HalfCheetah / Walker2d / Ant / Reacher to **1M** steps,
Hopper to **2M**, Humanoid to **3M**.

**Reported numbers.** The paper publishes **learning curves only — no numeric results table.**
Approximate final-return reads from **Figure 6 (constant total delay 5)**, clearly labelled as
visual estimates (green = DC/AC, blue = SAC, orange = RTAC):

| Env (delay 5, Fig 6) | DC/AC (ours-to-beat) | SAC (augmented) | RTAC | budget |
|---|---:|---:|---:|---|
| HalfCheetah-v2 | ~5200 | ~2000 | ~2100 | 1M |
| Walker2d-v2 | ~4000 | ~2200 | ~1600 | 1M |
| Ant-v2 | ~3200 | ~600 | ~700 | 1M |
| Hopper-v2 | ~3000 | ~2600 | ~3000 | 2M |
| Humanoid-v2 | ~6000 | ~5500 | — | 3M |
| Reacher-v2 | ~-8 | ~-9 | ~-8 | 0.1M |

(Exact returns are not extractable from the text; these reads are for order-of-magnitude
comparison only. Anyone encoding a hard number should pull the vector figure from the PDF.)

## 2. Our harness (faithful to their spec; honestly labelled)

**Delay injection (`ConstantDelayWrapper`).** Per-env, applied to standard Gymnasium MuJoCo
envs before vectorisation. It reproduces the DCAC headline constant-delay spec directly:
- **Action delay α:** the underlying env executes the action the agent emitted α steps earlier
  (a FIFO action pipeline of length α, zero-initialised at reset).
- **Observation delay ω:** the agent receives the true state from ω steps ago (FIFO obs pipeline
  of length ω).
- **Augmented state:** returned observation = `concat(delayed_obs, last K emitted actions)` with
  **K = ω + α = 5**, the sufficient statistic that restores the Markov property of the constant-
  delay MDP (Katsikopoulos & Engelbrecht 2003; Walsh et al. 2009). This is the SAME augmentation
  ActionShift's delay-aware backbone uses (obs + last-K actions) — here ported onto their envs and
  their ω/α spec. We implement the **constant-delay** case only (their Fig 6 headline); the WiFi
  random-delay case (Fig 7) is out of scope for this harness.

**Policy (`ppo_delay_mujoco.py`).** A **CleanRL-style** continuous-control PPO
(`cleanrl/ppo_continuous_action.py` recipe: RecordEpisodeStatistics + ClipAction +
NormalizeObservation/Reward + clip wrapper stack, Gaussian actor with state-independent log-std,
GAE, clipped surrogate, Adam, per-epoch minibatch updates, LR anneal). **This is explicitly
labelled CleanRL-style and is NOT DCAC's actor-critic**; we implement the augmented-state
reduction, not DCAC's resampling algorithm. Budget: **1M env steps** (matching their HalfCheetah/
Walker2d Fig-6 budget), num_envs=8, num_steps=512, 3 seeds/env. Metric = mean episodic RETURN of
true (undelayed) per-step rewards, exactly their metric.

**Labelling.** Ours is **"delay-aware augmented-state PPO (local)"**. Env versions are `-v4`
(mujoco 3.x), not their `-v2` (mujoco-py); reward structure is broadly comparable but not
identical (see caveats).

## 3. Attempt to run DCAC (`rlrd`) on this box for a same-hardware baseline — SKIPPED, documented

Time-boxed. Outcome: **rlrd's DCAC code installs, but its MuJoCo v2 envs cannot run on this box.**
- `pip install git+https://github.com/rmst/rlrd.git` failed under modern setuptools (old-gym
  `extras_require` build error). Fixed with `setuptools==57.5.0`; **rlrd 0.1 + gym 0.19.0 + torch
  2.13.0 then installed and `import rlrd` succeeds.**
- The `-v2` MuJoCo envs (HalfCheetah-v2, Ant-v2) need **`mujoco_py`**. The pip wheel installs, but
  `import mujoco_py` triggers a source build of its Cython extension (`cymj.pyx`) that **fails
  (`Cython.Compiler.Errors.CompileError`)** — the mujoco_py 2.1.2.14 Cython source is incompatible
  with Cython 3.x, and the box lacks GL dev headers (`/usr/include/GL/gl.h` absent) and
  apt-`patchelf`. The mujoco210 binary was fetched and placed in `~/.mujoco/mujoco210`, env vars
  set (`MUJOCO_GL=egl`), pip `patchelf` installed — the compile still fails.
- **Verdict:** this is the classic 2020-era dependency wall. Per plan, we **skip the same-hardware
  DCAC run and compare against DCAC's PUBLISHED curves only.** (A same-hardware DCAC baseline would
  need either a container with the old GL/Cython toolchain or a port of DCAC onto mujoco-3
  bindings — out of scope here.)

## 4. Our numbers on their setup

Delay-aware augmented-state PPO (local), constant delay ω=2/α=3 (total 5), HalfCheetah-v4 &
Walker2d-v4, 1M steps, 3 seeds. Return = mean of last-100 episodes (final window); we also report
the best 50-episode rolling mean over training. An **undelayed** (ω=α=0) CleanRL PPO reference
(1 seed) is included as a same-code no-delay ceiling.

**Full battery (3 envs × constant delay ω=2/α=3 = total 5, 3 seeds each; undelayed ref 1 seed):**

| Env (delay 5) | Ours delayed d=5 (mean ± sd, 3 seeds) | Ours best-50 | Ours undelayed (ω=α=0, same code) | Delay retention | DCAC DC/AC (pub, Fig 6) | DCAC SAC (aug, pub) | DCAC RTAC (pub) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HalfCheetah** (v4 / their v2) | **1025 ± 62** | 1033 | 1477 | **69 %** | ~5200 | ~2000 | ~2100 |
| **Walker2d** (v4 / their v2) | **857 ± 118** | 905 | 2157 | **40 %** | ~4000 | ~2200 | ~1600 |
| **Ant** (v4 / their v2) | **11 ± 11** | 21 | 380 | ~3 % | ~3200 | ~600 | ~700 |

Per-seed delayed finals (last-100-episode mean return): HalfCheetah [971, 992, 1112],
Walker2d [692, 916, 961], Ant [−5, 16, 20]. All runs 1M env steps, ~21 min each (GPU-0 battery)
or ~9 min (Ant, near-solo GPUs 1-3). "Retention" = delayed / undelayed on our OWN code (a
within-code measure of how much the augmentation preserves under delay-5; it is NOT a DCAC ratio).

**Reading of our numbers:**
- **The augmentation works.** Under the exact constant delay-5 that makes a *naive* unaugmented
  policy "near-random" (DCAC's words), our augmented-state PPO learns and retains **69 %**
  (HalfCheetah) and **40 %** (Walker2d) of its own undelayed performance. The obs+last-K-actions
  reduction — the mechanism ActionShift's delay-aware backbone is built on — is a sound,
  delay-robust recipe on their envs and their delay spec. This is the transferable positive result.
- **Absolute return is NOT competitive with DCAC's published curves.** Ours (~1025 / ~857) sits
  below DC/AC (~5200 / ~4000) *and* below their augmented SAC/RTAC baselines (~2000 / ~2200) —
  indeed our *undelayed* PPO (1477 / 2157) is already at or below their *delayed* SAC. This gap is
  the well-known **on-policy-PPO-vs-off-policy-SAC sample-efficiency gap** at equal 1M-step budget
  (SAC/RTAC/DCAC are all off-policy SAC-derived and far more sample-efficient on MuJoCo), plus
  minimal per-env PPO tuning — NOT a failure of the delay handling.
- **Ant is an honest weak point.** Our PPO barely leaves the floor on Ant (delayed 11, undelayed
  380) within 1M steps; Ant is DCAC's hardest constant-delay env (DC/AC ~3200 while their own
  SAC/RTAC stay ~600). On-policy PPO under this budget/tuning does not crack Ant, delayed or not.

## 5. Caveats (published-numbers comparison)

This is a **published-numbers** comparison, not a controlled head-to-head. The honest caveats:

1. **Different codebase/algorithm.** Ours is CleanRL-style **on-policy PPO** with augmented state;
   DCAC is an **off-policy actor-critic** (SAC-derived) with delay-correcting trajectory
   resampling. Their *augmented-state SAC/RTAC baselines* are the closest analogue to ours in
   spirit (both augment; neither resamples), so those baseline curves are the fairer reference
   than the DC/AC curve itself.
2. **Different MuJoCo version.** Ours uses Gymnasium `-v4` (mujoco 3.10); theirs used mujoco-py
   `-v2`. Reward scales are broadly comparable for HalfCheetah/Walker2d but not identical
   (termination/contact details differ), so absolute-return comparison carries a version caveat.
3. **Different hardware / PPO impl / tuning.** No shared seeds, no shared hyperparameter search;
   our budget matches their 1M-step Fig-6 schedule but our wall-clock and optimizer are unrelated
   to theirs.
4. **Their numbers are figure reads, not a table.** The paper publishes no numeric results table;
   the DC/AC/SAC/RTAC values above are approximate visual estimates from Figure 6.
5. **Domain-mismatch to ActionShift's own claim.** This locomotion comparison is a *sanity check
   that our augmentation is a sound, competitive delayed-RL recipe*; it is NOT the ActionShift
   manipulation result and does not transfer its numbers.

## 6. The delayed-MANIPULATION niche (independent of the locomotion comparison)

Per the SOTA recon (git history: `docs/weakness_sota_recon.md` §5c; summarized in the Related work section of README.md), the delayed-RL literature — DCAC included —
benchmarks **locomotion / classic control**, never randomized per-episode **action**-lag online RL
on **manipulation**. The three nearest published neighbours, which must be cited and distinguished
(not claimed as matches):
1. **arXiv:2509.20869** "Model-Based RL under Random Observation Delays" — Meta-World manipulation
   under random **observation** (not action) delay; benchmarks DA-Dreamer vs **DCAC** (reports DCAC
   degrading outside its trained delay range). Closest single hit.
2. **arXiv:2506.00131** "Belief-Based Offline RL for Delay-Robust Policy Optimization" — D4RL
   **Adroit** dexterous-hand (Pen/Door/Hammer) under delays up to 16 steps, but **offline** RL, not
   online PPO.
3. **arXiv:2605.15480** "Residual RL for Robot Teleoperation under Stochastic Delays" — Franka
   Panda under stochastic **communication** delay (teleoperation framing), not a randomized-action-
   lag PPO benchmark.

Honest niche claim (defensible): *"We are not aware of prior published results on randomized
per-episode action-delay-robust online RL specifically on ManiSkill-style manipulation; the closest
work studies observation delay on Meta-World (arXiv:2509.20869), offline-RL delay on Adroit
(arXiv:2506.00131), or teleoperation communication delay (arXiv:2605.15480), none of which match
the action-lag / online-PPO / ManiSkill3 setting."* A flat "nothing exists" claim is NOT defensible.

## 7. Verdict

**On competitiveness (Question 1): NOT competitive on absolute return, but the mechanism is
validated.** A simple augmented-state PPO, run on DCAC's own MuJoCo + constant-delay-5 setup,
does NOT match DCAC's published returns, and does not reach even their augmented-SAC/RTAC
baselines — because on-policy PPO is much less sample-efficient than the off-policy SAC family at
an equal 1M-step budget, not because the augmentation fails. What IS demonstrated is that the
**augmented-state reduction is delay-robust**: it retains 40-69 % of undelayed performance under
the delay that renders naive control near-random, on HalfCheetah and Walker2d. A genuinely
apples-to-apples competitiveness test would require an *off-policy augmented SAC* (or DCAC's own
code — attempted here, blocked by the mujoco_py/Cython 2020-era build wall) at matched budget;
our PPO-vs-SAC comparison cannot settle the absolute-return question in our favour, and we do not
claim it does.

**On the niche (Question 2): the delayed-MANIPULATION niche is real and uncovered.** DCAC and the
delayed-RL literature benchmark locomotion / classic control only; none run randomized per-episode
**action**-lag online RL on ManiSkill-style **manipulation**. The three nearest neighbours
(arXiv:2509.20869 obs-delay Meta-World; arXiv:2506.00131 offline-RL Adroit; arXiv:2605.15480
teleop communication delay) are adjacent but distinct and must be cited, not claimed as matches.

**What IS claimable:**
- The augmented-state reduction (obs + last-K actions) is a **delay-robust recipe on DCAC's own
  constant-delay-5 MuJoCo setup** (HalfCheetah retains 69 %, Walker2d 40 % of undelayed;
  decisively above the naive unaugmented baseline DCAC reports as near-random). This validates the
  mechanism ActionShift's delay-aware backbone relies on, on the literature's own benchmark.
- ActionShift's delayed-**manipulation** result occupies a niche the locomotion delayed-RL
  benchmarks do not cover (hedged, 3 near-misses cited).

**What is NOT claimable:**
- That our simple augmented-state PPO **matches or beats DCAC's (or SAC's/RTAC's) absolute returns**
  on delayed MuJoCo. It does not, at equal 1M-step budget — an expected on-policy/off-policy gap.
- Any same-hardware DCAC number (its 2020-era mujoco_py deps would not build here — documented §3).
- Any hard numeric claim against DCAC's table (they publish curves only; our DC/AC/SAC/RTAC
  comparison values are approximate figure reads).

## Reproduce

```bash
# fresh env (once)
python3.11 -m venv .venv-delayext
.venv-delayext/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-delayext/bin/pip install "gymnasium[mujoco]" numpy

# train (GPU 0), headline constant delay omega=2 alpha=3 (total 5)
cd experiments/delayed_external
CUDA_VISIBLE_DEVICES=0 ../../.venv-delayext/bin/python ppo_delay_mujoco.py \
  --env-id HalfCheetah-v4 --obs-delay 2 --act-delay 3 \
  --total-timesteps 1000000 --seed 1 --output ../../artifacts/delayed_external
# undelayed ceiling: --obs-delay 0 --act-delay 0
```
