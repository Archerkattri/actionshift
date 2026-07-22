# Peg retry: PegInsertionSide-v1 at the full official PPO budget (75M)

Run date: 2026-07-21 UTC
Seed: `20260718`
Hardware: NVIDIA RTX 5090, GPU 3 only (`CUDA_VISIBLE_DEVICES=3`)
Task setup: ManiSkill v3.0.1 source at `third_party/maniskill` (commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`), state observations, `pd_ee_delta_pose`, real GPU simulation.

## Decision

**PegInsertionSide-v1 remains EXCLUDED at the full official budget. The working benchmark stays three-task
(PickCube, PushCube, PullCube).** Official-code PPO trained to **~75.7M cumulative steps** — at and beyond
the corrected official 75M Peg budget — produced **zero successful insertions** at every evaluation. The
final 75.7M-cumulative checkpoint and the two earlier milestone checkpoints (41M, 71.7M) each score
**0/100 `success_once`** on the honest 100-episode floor measure, far below the preregistered **0.20**
Peg floor. This is now an *airtight, full-budget* exclusion, not a truncated one: it confirms and extends
`reports/gate0.md` (Peg had no competent short-run backbone) all the way to the task's own official budget.
Peg is **not** converted into adaptation evidence; no Gate 1 ceiling or tournament cells are produced.

## Budget correction (explicit)

An earlier version of this report used **250M** steps as the "official" Peg budget. That figure came from
the **wrong config file**: `examples/baselines/ppo/examples.sh` (a `ppo.py` line that is not the source of
the published baseline numbers). The **true official budget is 75M steps**, from
`examples/baselines/ppo/baselines.sh` — the script whose runs generate ManiSkill's published PPO baseline
report. This report is corrected to the 75M budget and is consistent with the committed reports. The 75M
figure in `baselines.sh` is attached to the CUDA-graph `ppo_fast.py` variant; to preserve exact
**wrapper/oracle parity discipline** (the parity harness `ppo_parity.PpoAgent` replicates the `ppo.py`
network, and Pick/Push/Pull were all frozen from `ppo.py`), the retry uses the parity-compatible `ppo.py`
at the corrected 75M budget rather than switching scripts mid-benchmark.

## Backbone: official-code PPO at the corrected 75M budget

Backbone: local reproduction of pinned official ManiSkill v3.0.1 PPO
(`examples/baselines/ppo/ppo.py`), config as in `examples.sh` for PegInsertionSide (num_envs=1024,
update_epochs=8, num_minibatches=32, num_steps=100, num-eval-steps=100) with all other hyperparameters at
the `ppo.py` defaults the Peg line does not override (`learning_rate=3e-4`, `gamma=0.8`, `gae_lambda=0.9`,
`num_eval_envs=8`, `eval_freq=25`, `anneal_lr=False`), **total budget 75M** (corrected).

### Deviations from official `ppo.py` args

Only the three standard benchmark-interface deviations applied identically to Pick/Push/Pull, plus the
segmented-training note forced by the environment:

1. `--control_mode=pd_ee_delta_pose` (official default `pd_joint_delta_pos`) — the **required ActionShift
   shared 7-dim controller**, identical across all four tasks; the parity/wrapper/adaptation stack are all
   defined on it.
2. `--seed=20260718` (official default `1`) — the benchmark seed.
3. `--no-capture-video` (IO only, no learning effect) and `--exp-name` (cosmetic run-dir name).
4. **Segmented training with 2 optimizer resets.** A ~55-60-minute environment process-kill wall prevented
   a single uninterrupted 75M run (each launch was terminated around ~41M steps / ~60 min). Training was
   therefore completed in three segments carried across via `ppo.py --checkpoint` (weights preserved;
   `anneal_lr=False` so the constant 3e-4 LR has no schedule discontinuity; only the Adam moments reset at
   each boundary). This is a resource-imposed provenance note, not a hyperparameter change. All three
   segments are frozen with hashes.

Manifest with all segment hashes: `artifacts/peg_retry/gate0/peg_insertion_side-frozen-checkpoint.json`.
Final checkpoint `runs/peg_insertion_side-ppo-75M-topup-s20260718/final_ckpt.pt`
sha256 `4ebc3a7f084a3f7932191c36fa631808a80ae7c23f4038bc9b7953f100fc6175`.

## Training curve and budget used

Throughput ~14-16k steps/s early (single 5090), ~9.5k/s after the resumes (batch 102,400/iteration).

| Segment | Run dir | Steps (cumulative) | End cause |
|---|---|---|---|
| fresh 75M-config | `runs/peg_insertion_side-ppo-75M-s20260718` | 0 → ~41.0M | killed at ~60-min wall (iter 401) |
| resume | `runs/peg_insertion_side-ppo-75M-resume-s20260718` | ~41.0M → ~71.7M | killed by 55-min `timeout` (iter 301) |
| top-up | `runs/peg_insertion_side-ppo-75M-topup-s20260718` | ~71.7M → **~75.7M** | **clean completion, `final_ckpt.pt`** |

`success_once` and `success_at_end` are **0.000 at every one of the ~33 evaluations across all three
segments** (0 → 75.7M cumulative). Return climbs from 1.99 to a noisy ~40-65 plateau — the policy reliably
learns the dense-reward *approach/alignment* behaviour (hover the peg near the hole) but never discovers
the insertion. Segment-1 curve (0→41M) is **bit-identical** to the earlier 250M-labelled run (same seed +
`torch_deterministic=True`), a useful reproducibility check. Representative points:

| Cumulative steps | `success_once` | return |
|---:|---:|---:|
| 0 (seg1) | 0.000 | 1.99 |
| 10.2M | 0.000 | 23.94 |
| 23.0M | 0.000 | 55.18 |
| 38.4M | 0.000 | 61.80 |
| 41.0M (seg1 end) | 0.000 | 41.56 |
| 51.2M (resume) | 0.000 | 62.33 |
| 61.6M (resume) | 0.000 | 57.01 |
| 71.7M (resume end) | 0.000 | 41.92 |
| 74.3M (top-up) | 0.000 | 25.73 |
| ~75.7M (final) | 0.000 | — |

Raw scalars: `events.out.tfevents.*` in each of the three run dirs.

## Gate 0 floor evaluation (honest 100-episode measure)

Evaluated with the exact benchmark measure used for Pick/Push/Pull —
`ppo_parity.evaluate_ppo_checkpoint`, unwrapped, `success_once`, `ignore_terminations=True`, full 100-step
horizon, seed 20260718, 16 eval envs, 100 episodes:

| Checkpoint | Cumulative steps | Floor | `success_once` (100 ep) | Verdict |
|---|---:|---:|---:|---|
| 75M-run/ckpt_401 | ~41.0M | 0.20 | **0.00** (0/100) | below floor |
| resume/ckpt_301 | ~71.7M | 0.20 | **0.00** (0/100) | below floor |
| **topup/final_ckpt** | **~75.7M** | 0.20 | **0.00** (0/100) | **below floor** |

Every episode runs the full 100 steps (never terminates early), confirming zero insertions across the
whole budget. Raw JSONL: `artifacts/peg_retry/gate0/parity/peg75M-*-unwrapped.jsonl`. Because competence
fails at the full official budget, the identity/oracle parity conditions are moot (a 0.00-vs-0.00 parity is
mechanically consistent but cannot pass without competence); they were not run and no Gate 0 advance
verdict is issued.

## Ceiling gap and tournament cells

**Not applicable — Peg is gated out.** With no competent backbone there is no privileged oracle to bound
(the ceiling gap is undefined), and the belief-family tournament (exact_belief / fixed_probes /
entropy_probes / dualabi) has no meaningful base policy to adapt. No Gate 1 or tournament artifacts were
produced for Peg, and `artifacts/sprint/gate1/jobs.jsonl` was **not** extended with Peg entries (that
append is reserved for a gated-in task, per the PullCube template).

## tcp auto-calibration on Peg's observation layout (interface readiness)

Verified independently of competence. `load_or_run_calibration("peg_insertion_side", ...)` runs on Peg's
obs layout with **no task-specific patch**: the contract-independent calibration auto-locates the
contiguous tcp block at `position_start=18`, `quaternion_start=21`, and fits the same weak per-channel
linear response measured for Pick/Push/Pull (alpha ~0.017-0.036). So the adaptation/calibration interface
is Peg-ready should a competent Peg backbone ever exist; the blocker is purely backbone competence, not the
wrapper or calibration path. Output: `artifacts/peg_retry/calibration/peg_insertion_side.json`.

## Ordering-replication check vs the other three tasks

No Peg tournament exists, so the `entropy > fixed ~ passive > random` ordering **cannot be tested on Peg**.
The cross-task ordering replication remains established on the three competent tasks only (Pick/Push in
`reports/gate1.md` / `reports/adaptation_tournament.md`, and PullCube in `reports/third_task.md`). Peg
contributes no evidence for or against the ordering; its exclusion neither weakens nor extends the
benchmark's load-bearing tournament claims, which stand on three tasks.

## Honest anomalies and limitations

1. **Full official budget now covered.** Unlike the earlier truncated attempt, training reached ~75.7M
   cumulative steps ≥ the corrected 75M official budget, with a flat-zero success curve throughout and no
   upward inflection anywhere. The exclusion is therefore at (not short of) the task's official budget.
   Residual caveat: the run was *segmented* with two optimizer resets (below), so it is not identical to a
   single uninterrupted 75M optimizer trajectory; but the weights were carried across, the LR is constant,
   and success is flat zero across all segments including the clean final top-up.
2. **Segmented training / optimizer resets.** Forced by a ~55-60-min environment process-kill wall — no
   single launch survived past ~41M steps. Each resume reset the Adam moments (not the weights or LR). If a
   future environment allows an uninterrupted 75M run, that is the one remaining way this verdict could in
   principle change; the evidence here (0/100 at 41M, 71.7M and 75.7M; ~33 zero evals) makes that unlikely.
3. **Alignment-without-insertion policy.** Return climbs to ~40-65 while `success_once` stays exactly 0.
   The dense reward is dominated by reach/alignment terms; the policy exploits those and hovers near the
   hole. Every one of 100 evaluation episodes ran the full 100-step horizon without an early success
   termination — this is a genuine plateau, not a logging artifact.
4. **Reproducibility check passed.** The first 41M of the 75M run is bit-identical to the earlier
   250M-labelled run (same seed + deterministic), confirming the pipeline is deterministic and the zero is
   not a one-off.

## Claim boundary

Supported: official-code PPO integration for PegInsertionSide-v1 at its corrected official 75M budget; an
honest training curve to ~75.7M cumulative steps with full segment provenance and hashes; rigorous 0/100
`success_once` floor evaluations at 41M, 71.7M, and the final ~75.7M checkpoint; and verification that the
tcp calibration / wrapper interface is Peg-ready. Not supported: any Peg competence claim, any Peg
adaptation/ceiling/tournament conclusion, or a claim that Peg is unsolvable by PPO in general (only that it
did not clear the 0.20 floor at or beyond its own official 75M budget under this official config, in a
segmented run). All raw episode JSONL, checkpoint hashes, tfevents curves, the calibration record, and the
exclusion manifest remain under `artifacts/peg_retry/`.

## Files and additive diffs

- Source (additive, ruff + strict mypy clean): `peg_insertion_side` added to the `--task` choices in
  `scripts/evaluate_gate1_slice.py` (the one shared registry that still excluded it; `_FLOORS`, `_TASKS`,
  `_ENVIRONMENT_IDS`, `envs/tasks.py::TASKS`, and the frozen-registry test
  `tests/envs/test_tasks.py::test_frozen_task_registry_covers_pick_push_pull_and_insertion` already covered
  Peg from Gate 0). No change to `artifacts/sprint/gate1/jobs.jsonl` (Peg is not gated in).
- Artifacts: `artifacts/peg_retry/gate0/peg_insertion_side-frozen-checkpoint.json` (manifest + all segment
  hashes + budget correction); `artifacts/peg_retry/gate0/parity/peg75M-{ckpt401,resume-ckpt301,final}-unwrapped.jsonl`
  (100-ep floor evals); `artifacts/peg_retry/calibration/peg_insertion_side.json` (tcp calibration);
  training logs + checkpoint hashes under `artifacts/peg_retry/logs/`; checkpoints + tfevents under the
  three `runs/peg_insertion_side-ppo-75M{,-resume,-topup}-s20260718/` dirs.
</content>
