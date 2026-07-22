# Gate 0: official backbone competence and wrapper validity

Run date: 2026-07-18/19 UTC  
Seed: `20260718`  
Hardware: four NVIDIA RTX 5090 GPUs  
Task setup: ManiSkill v3.0.1 source at `a4a4f9272ad64b1564035874b605ceb687b63ed8`,
state observations, `pd_ee_delta_pose`, local short budgets

## Decision

Gate 0 is a **task-level partial pass**. PickCube and PushCube have competent, parity-safe PPO
checkpoints and advance to Gate 1. PegInsertionSide has no competent short-run backbone and is
excluded. The overall three-task gate therefore remains failed; the exclusion is not converted into
adaptation evidence.

| Task | Floor | Frozen backbone | Final 100-episode unwrapped | Identity | Oracle nonidentity | Decision |
|---|---:|---|---:|---:|---:|---|
| PickCube-v1 | 0.50 | PPO | 1.00 | 1.00 | 1.00 | advance |
| PushCube-v1 | 0.50 | PPO | 1.00 | 1.00 | 1.00 | advance |
| PegInsertionSide-v1 | 0.20 | none | 0.00 | 0.00 | 0.00 | exclude |

The parity records retain 100 individual episodes per condition. Pick and Push satisfy both the
two-percentage-point rule and overlapping Wilson-interval rule exactly. Peg's zero-versus-zero parity
is mechanically consistent but cannot pass without competence.

## Backbone runs

These are local reproductions of pinned official code under the common controller and short budgets,
not reproductions of each paper's tuned published configuration.

| Task | Method | Budget | Valid evaluation success | Status / note |
|---|---|---:|---:|---|
| Pick | PPO | 10M | 0.96875 during training; 1.00 final checkpoint | competent |
| Pick | SAC | 250k | 1.00 | competent; no wrapper-parity run |
| Pick | TD-MPC2 | 50k | 0.00 at 25,600 | integration only / not competent |
| Push | PPO | 2M | 0.875 during training; 1.00 final checkpoint | competent |
| Push | SAC | 250k | 0.94 | competent; no wrapper-parity run |
| Push | TD-MPC2 | 50k | 0.81 at 25,600 | competent; no wrapper-parity run |
| Peg | PPO | 10M | 0.00 | not competent |
| Peg | SAC | 250k | 0.00 | not competent |
| Peg | TD-MPC2 | 50k | 0.00 at 25,600 | not competent |
| All | FastTD3 | n/a | n/a | inapplicable: official repo has no ManiSkill adapter |

PPO is frozen even where another training log is competitive because PPO alone has direct
episode-level unwrapped/identity/oracle parity with the exact final checkpoint.

## Corrections and retained failures

- The first Peg SAC process failed because evaluation stopped at 50 steps on a 100-step task. Its
  immutable failure remains in `artifacts/sprint/gate0`; a separate corrected 100-step run completed
  at 0.00 success and return 39.65.
- The first TD-MPC2 runs only evaluated at step zero. Their later `train S:` values are rollout
  metrics, not evaluation, and are not used. Separate immutable runs added `eval_freq=25000` and 112
  evaluation episodes. Their step-25,600 results are the values in the table.
- TD-MPC2 v3.0.1 required the pinned semantic-neutral Gymnasium 1.3 compatibility patch in
  `patches/maniskill-tdmpc2-gymnasium-1.3.patch`; the patch only unwraps attribute access.
- FastTD3 remains structured as inapplicable rather than being replaced by an unofficial port.

## Claim boundary

Supported: official-code integration, short-budget competence for the listed task/method pairs, and
PPO wrapper/oracle parity for Pick and Push. Not supported: published-score reproduction, algorithm
superiority from one seed, Peg adaptation conclusions, or claims about methods without final-checkpoint
parity. Raw logs, immutable results, checkpoints, episode JSONL, and hardware/source provenance remain
under `artifacts/sprint/` and `third_party/` locally.
