# Fourth task: gating in a mid-difficulty task with StackCube-v1

Run date: 2026-07-21 UTC
Seeds: `20260718`, `20260719`, `20260720`
Hardware: NVIDIA RTX 5090, **GPU 2 only** (`CUDA_VISIBLE_DEVICES=2`)
Task setup: ManiSkill v3.0.1 source at `third_party/maniskill` (commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`), state observations, `pd_ee_delta_pose`, real GPU simulation.

## Decision

**StackCube-v1 is a Gate-0 pass and is admitted as the fourth ActionShift task**, extending the working
benchmark to four competent tasks (PickCube, PushCube, PullCube, StackCube). StackCube was the intended
mid-difficulty target and it verified on every requirement, so the PokeCube-v1 fallback was not needed.
Unlike the other three tasks, StackCube is a genuine **pick-place-stack** composition and its official PPO
budget is the largest of the cube tasks (25M steps vs 2M for Pull/Push), which is why it counts as a
mid-difficulty gate-in rather than another reach/push primitive.

## Task validity (pinned checkout)

- `StackCube-v1` exists in the pinned checkout with `max_episode_steps=50`,
  `SUPPORTED_ROBOTS=[panda_wristcam, panda, fetch]`, a `success` scalar in `evaluate()`, the shared
  `pd_ee_delta_pose` 7-dim controller, and `tcp_pose=self.agent.tcp.pose.raw_pose` in `_get_obs_extra`
  (the contract-independent stack decodes against a genuine tcp block) — the same interface Pick/Push/Pull use.
- The `HiddenContractWrapper` identity/oracle paths, calibration, and the belief-family adapters run on
  StackCube unchanged; no task-specific patch was added.

## Backbone: official-code PPO at the official StackCube config

Backbone: local reproduction of pinned official ManiSkill v3.0.1 PPO
(`examples/baselines/ppo/ppo.py`) using the **official StackCube-v1 configuration verbatim** from
`examples/baselines/ppo/examples.sh`:

```
python ppo.py --env_id="StackCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
```

### Deviations from the official config (target: zero)

Only the standard, benchmark-wide interface deviations already applied identically to Pick/Push/Pull/Peg —
no algorithm-hyperparameter deviation:

1. `--control_mode=pd_ee_delta_pose` (official default is `pd_joint_delta_pos`) — the **required ActionShift
   shared 7-dim controller**, identical across all tasks; the parity/wrapper/adaptation stack is defined on it.
2. `--seed=20260718` (official default `1`) — the benchmark seed.
3. `--no-capture-video` (IO only) and `--exp-name=stack_cube-ppo-25M-s20260718` (cosmetic run-dir name).

`num_steps`/`num_eval_steps` were left at the ppo.py defaults (50), which the official StackCube line does not
override and which exactly cover StackCube's 50-step horizon.

### Budget, wall time, and competence

- Throughput ~19,500 steps/s (num_envs=1024). **Full 25M-step budget completed in ~22.0 min** (06:52:23Z →
  07:14:24Z), **well within the 90-minute hard cap** — no early termination.
- Final evaluation: **`success_once = 1.00`**, **`success_at_end = 0.875`**, `return = 43.1`. Success emerges
  at ~7.7M steps (0.375), clears the 0.50 floor by ~9M, and holds `success_once ∈ [0.75, 1.0]` from 11.5M on.
- Frozen final checkpoint `runs/stack_cube-ppo-25M-s20260718/final_ckpt.pt`, sha256
  `e63cc8d8ffdca3d03553a21ea615c759b2b224493a7e7e12bee7efc29d5bad9c`
  (manifest: `artifacts/fourth_task/gate0/stack_cube-frozen-checkpoint.json`; training log:
  `artifacts/fourth_task/logs/stack_cube_ppo_train.log`).

**Note (contrast with PullCube):** StackCube does **not** show the Pull `success_once`/`success_at_end`
divergence — `success_at_end = 0.875` means the policy actually *holds* the stacked cube, so this is a
persistent-placement backbone, not a touch-and-release one.

## Gate 0: official backbone competence and wrapper parity

Success measure is `success_once` under `ignore_terminations=True` over the full 50-step horizon — identical to
Pick/Push/Pull in `ppo_parity.evaluate_ppo_checkpoint`. 100 episodes per condition, seed `20260718`.
Raw JSONL: `artifacts/fourth_task/gate0/parity/stack_cube-{unwrapped,identity,oracle_nonidentity}.jsonl`;
verdict `artifacts/fourth_task/gate0/stack_cube-gate0-verdict.json`.

| Task | Floor | Backbone | Unwrapped (100 ep) | Identity | Oracle nonidentity | Decision |
|---|---:|---|---:|---:|---:|---|
| StackCube-v1 | 0.50 | PPO (25M) | 0.98 | 1.00 | 0.96 | **advance** |

Both parity conditions satisfy the preregistered rule exactly: |identity − unwrapped| = 0.02 ≤ 0.02 tolerance
(with overlapping 95% Wilson intervals) and |oracle − unwrapped| = 0.02 ≤ 0.02 (overlapping intervals).
Competence 0.98 ≫ 0.50 floor.

## Gate 1: privileged-ceiling slice (oracle vs no-adapt)

Two frozen 6-pose-channel representative contracts per split, 100 episodes each, 3 seeds = **600 episodes per
cell**. Oracle encodes the canonical PPO action before the hidden wrapper; no-adapt sends the same output with
no contract knowledge. Verdict data: 18 files in `artifacts/fourth_task/gate1/`; ledger lines appended to
`artifacts/sprint/gate1/jobs.jsonl` (existing Pick/Push/Pull lines untouched).

| Split | Oracle success (95% Wilson) | No-adapt (95% Wilson) | Ceiling gap |
|---|---:|---:|---:|
| seen | 0.965 [0.947, 0.977] | 0.000 [0.000, 0.006] | 0.965 |
| unseen composition | 0.960 [0.941, 0.973] | 0.000 [0.000, 0.006] | 0.960 |
| long lag | 0.000 [0.000, 0.006] | 0.000 [0.000, 0.006] | 0.000 |

The seen and unseen-composition cells reproduce the benchmark premise on the fourth task: the same competent
policy is near-perfect when its action ABI is known and **always fails (0.000)** when the ABI is hidden,
including under the absolute/tool-frame composition encoder.

**Long-lag falsification is strongest yet on StackCube:** even the privileged oracle collapses to **0.000** —
more fragile than Pick (0.027), Push (0.153), or Pull (0.322). Precision pick-place-stack tolerates *no* action
lag even with full contract knowledge; contract knowledge alone remains insufficient under lag on all four tasks.

## Tournament slices (frozen StackCube PPO backbone)

Runner `experiments/run_adaptation_slice.py`; hash-addressed output pooled into `artifacts/adaptation/slices`
(so `experiments/analyze_adaptation.py` aggregates it automatically); the frozen-checkpoint lookup resolves
StackCube from the extended `artifacts/sprint/gate1/jobs.jsonl` ledger. Belief-family methods share one declared
9-contract pool, 6-step budget, amplitude, backbone, contracts, and seed (matched-privilege). Cells are
**seed 20260718 × 2 contracts × 100 = 200 episodes** (Wilson 95%). Aggregate:
`artifacts/adaptation/tournament.json`.

| Split | exact_belief | fixed_probes | entropy_probes | dualabi | random_probes |
|---|---:|---:|---:|---:|---:|
| seen | 0.945 [0.904, 0.969] | 0.945 [0.904, 0.969] | 0.910 [0.862, 0.942] | 0.945 [0.904, 0.969] | 0.925 [0.880, 0.954] |
| unseen composition | 0.960 [0.923, 0.980] | 0.915 [0.868, 0.946] | 0.940 [0.898, 0.965] | 0.960 [0.923, 0.980] | 0.930 [0.886, 0.958] |

Every belief-family method recovers **0.91–0.96** on both splits versus no-adapt's **0.000** — the core result
(identification recovers competence) replicates strongly on StackCube.

### Paired comparisons (seed 20260718, 200 paired episodes, 95% bootstrap)

**No pairwise success difference is significant at this single-seed budget** — all methods sit near ceiling and
every 95% CI includes zero:

- Unseen: entropy > fixed +2.5 [−2.5, +7.5] (ns; **does not clear the preregistered +5-point promotion gate**);
  dualabi > fixed +4.5 [−0.5, +9.5] (ns, closest to promotion); dualabi = exact_belief 0.0.
- Seen: entropy − fixed −3.5 [−8.5, +1.5] (ns) and entropy − exact_belief −3.5 [−9.0, +2.0] (ns) — entropy is
  the *lowest* cell on seen; dualabi = fixed = exact_belief 0.0.

Under a plain competence reading of "promote > 0.3", all five methods promote (all ≫ 0.3, and all ≫ the 0.000
no-adapt floor). Under the stricter preregistered "+5 over fixed on unseen" gate, **no method is promoted on
StackCube** (entropy +2.5 ns).

### DualABI probe-efficiency Pareto (replicates Pick/Push/Pull)

DualABI matches the top success on both splits (0.945 seen ties exact_belief/fixed; 0.960 unseen ties
exact_belief) while using **2.9 / 3.5 mean probe steps vs 6.0** for fixed/entropy/random (~42–51% fewer) and far
less probe displacement (0.011 / 0.017 vs fixed 0.065 / 0.068 and random 0.106 / 0.109). The success-versus-
action-cost Pareto promotion criterion is cleared on the fourth task, consistent with `reports/adaptation_dualabi.md`.

## Cross-task consistency check — does the ordering replicate?

**Method ordering `entropy > fixed ~ passive > random`:**

- **Seen: does NOT replicate.** entropy (0.910) is the *lowest* method, below fixed/passive/dualabi (0.945) and
  below random (0.925). Information-seeking does not help — and its probe perturbations slightly cost the precision
  stack.
- **Unseen composition: partially replicates the top of the order but not the premise.** entropy (0.940) does beat
  fixed (0.915) and random (0.930), but **passive exact_belief (0.960) is the top method**, not entropy, and no gap
  is significant.

**Verdict:** the load-bearing benchmark claims replicate on the fourth task — a known ABI recovers near-ceiling
competence (0.96) while a hidden ABI fails (0.000), and DualABI matches the best method at ~half the probe budget.
The tournament *sub-finding* that active information-seeking (entropy) is the top adaptation method **does not
generalize to StackCube**: StackCube's contract is already identifiable from the passive pass-through at
near-ceiling, so entropy adds nothing (and slightly hurts on the precision-sensitive seen split). This is the same
class of honest, task-dependent tournament anomaly recorded for the `> random` tail on PullCube — here it is the
`entropy on top` claim that is task-specific, not universal.

## Honest anomalies

1. **entropy is not the top adaptation method on StackCube** (above): on seen it is the lowest belief-family cell.
   Passive belief already saturates near ceiling, leaving no room for information-seeking to help. Single-seed, so
   no pairwise difference is significant; disclosed rather than smoothed.
2. **Long-lag oracle 0.000** — StackCube is the most lag-fragile of the four tasks; even the privileged oracle
   cannot complete a stack under lag. no-adapt is also 0.000, so the "identification alone cannot fix delay"
   conclusion holds a fortiori.
3. **Single-seed tournament** (seed 20260718 only, per the fourth-task spec) — 200 paired episodes per cell. All
   tournament cells are therefore reported without three-seed pooling; the Gate-1 ceiling cells are the full
   three-seed (600-episode) result.

## Claim boundary

Supported: official-code PPO integration and full-budget (25M) competence for StackCube; exact PPO wrapper/oracle
parity; a three-seed privileged-ceiling gap (0.96 vs 0.000 on seen/unseen); belief-family adaptation recovering
near-ceiling competence; and the DualABI probe-efficiency Pareto win. Not supported: published-score reproduction,
unprivileged adaptation success, algorithm superiority beyond the matched-privilege pool (no significant pairwise
tournament difference at one seed), the "entropy is best" ordering on StackCube, or any long-lag claim beyond the
disclosed 0.000 collapse. Peg remains excluded, so the original preregistered all-of-Pick/Push/Peg gate is
unchanged; StackCube adds a fourth *competent, mid-difficulty* task to the working benchmark. All raw episode
JSONL, checkpoint hashes, verdicts, and logs are under `artifacts/fourth_task/` and `artifacts/adaptation/`.

## Files and additive diffs

- Source (additive, ruff + strict mypy clean, registry tests green): `stack_cube` added to
  `benchmarking/gates.py::_FLOORS` (0.50), `benchmarking/baselines.py::_TASKS`,
  `benchmarking/ppo_parity.py::_ENVIRONMENT_IDS`, `envs/tasks.py::TASKS` (family `stack`,
  `max_episode_steps=50`), and the `--task` choices in `scripts/evaluate_gate1_slice.py`; the frozen
  task-registry test in `tests/envs/test_tasks.py` updated to cover five tasks.
- `run_adaptation_slice.py::frozen_checkpoint` behaviour is unchanged: StackCube resolves because its three-seed
  × two-method × three-split Gate-1 job entries were appended to `artifacts/sprint/gate1/jobs.jsonl` (existing
  Pick/Push/Pull lines untouched).
- Frozen checkpoint + hash: `artifacts/fourth_task/gate0/stack_cube-frozen-checkpoint.json`.


> **Budget correction (2026-07-21, post-hoc):** this run used 25M steps at num_envs=1024; the TRUE official StackCube-v1 baseline config is 50M steps at num_envs=4096 (`baselines.sh`; the 25M figure came from `examples.sh` — same error class as the corrected Peg budget). The Gate-0 competence verdict (0.98) stands on its own; do NOT make budget-efficiency claims vs the official baseline from this run. See the Claims section of README.md (Claim 8; pre-merge ledger in git history: docs/SOTA_CLAIMS.md).
