# Third task: restoring three-task generality with PullCube-v1

Run date: 2026-07-21 UTC
Seeds: `20260718`, `20260719`, `20260720`
Hardware: NVIDIA RTX 5090, GPU 3 only (`CUDA_VISIBLE_DEVICES=3`)
Task setup: ManiSkill v3.0.1 source at `third_party/maniskill` (commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`), state observations, `pd_ee_delta_pose`, real GPU simulation.

## Decision

**PullCube-v1 is a Gate-0 pass and is admitted as the third ActionShift task**, restoring three-task
generality (PickCube, PushCube, PullCube). PegInsertionSide remains excluded (no competent short-run
backbone; unchanged from `reports/gate0.md`). PullCube was chosen over PokeCube because it verified
first on every requirement: it exists in the pinned checkout (`PullCube-v1`, `max_episode_steps=50`,
`SUPPORTED_ROBOTS=[panda, fetch]`), exposes a `success` scalar, uses the shared `pd_ee_delta_pose`
controller and 7-dim action, exposes `tcp_pose` in `_get_obs_extra` (so the contract-independent
calibration auto-locates the tcp block), and its official-code PPO cleared the competence floor with
exact wrapper parity. PokeCube was not needed as a fallback.

## Wrapper / interface validity

- The `HiddenContractWrapper` 7-dim action interface and the identity/oracle wrapping paths run on
  PullCube unchanged (`tests/envs/test_tasks.py::...smoke[pull_cube]` passes: real GPU oracle-decode
  path, finite observations, batch success shape `(1,)`).
- Calibration tcp-pose auto-location works on PullCube's observation layout: `locate_contiguous_slice`
  finds `position_start=18`, `quaternion_start=21` (contiguous 7-value tcp raw-pose block). Per-channel
  fit R2 = 0.09-0.34, the same weak-response regime measured for Pick/Push — no task-specific patch.

## Gate 0: official backbone competence and wrapper parity

Backbone: local reproduction of pinned official ManiSkill v3.0.1 PPO (`examples/baselines/ppo/ppo.py`)
under the common controller and short budget — not a reproduction of a tuned published configuration.
Command (identical arg pattern to the frozen PushCube run, only `--env_id` and budget differ):

```
ppo.py --env_id=PullCube-v1 --seed=20260718 --control_mode=pd_ee_delta_pose \
       --total_timesteps=2000000 --num_envs=1024 --num_eval_envs=16 \
       --num-eval-steps=100 --eval-freq=10 --no-capture-video
```

Budget 2,000,000 steps (matched to PushCube). Frozen final checkpoint
`runs/pull_cube-ppo-2M-s20260718/final_ckpt.pt`, sha256
`74ae6a09b9af5e9e50dc71944f2e99316a8b67b02f3a96ca45df4a6d53dc1bd7`
(manifest: `artifacts/third_task/gate0/pull_cube-frozen-checkpoint.json`). Success measure is
`success_once` under `ignore_terminations=True` over the full 50-step horizon — **identical to the
measure used for Pick and Push** in `ppo_parity.evaluate_ppo_checkpoint`.

| Task | Floor | Backbone | Unwrapped (100 ep) | Identity | Oracle nonidentity | Decision |
|---|---:|---|---:|---:|---:|---|
| PullCube-v1 | 0.50 | PPO (2M) | 1.00 [0.963, 1.000] | 1.00 | 1.00 | **advance** |

Both parity conditions satisfy the preregistered rule exactly: absolute difference 0.0000 and
overlapping 95% Wilson intervals for identity and oracle-nonidentity vs unwrapped. Raw 100-episode JSONL
per condition: `artifacts/third_task/gate0/parity/pull_cube-{unwrapped,identity,oracle_nonidentity}.jsonl`;
verdict `artifacts/third_task/gate0/pull_cube-gate0-verdict.json`.

**Disclosed anomaly (backbone property, not a gate failure).** PullCube's `success_once` reaches 1.00 but
`success_at_end` collapses to ~0 across training evals (return also decays 28.8 -> 13.5 over the last
epochs): the policy reliably pulls the cube into the goal region at least once but then drags it back out
before step 50. Because ActionShift scores `success_once` for every task, this is competent and parity-safe
under the same measure applied to Pick/Push; it is recorded here so the behaviour is not mistaken for a
persistent-placement policy.

## Gate 1: privileged-ceiling slice (oracle vs no-adapt)

Two frozen 6-pose-channel representative contracts per split, 100 episodes each, 3 seeds = 600 episodes
per cell. Oracle encodes the canonical PPO action before the hidden wrapper; no-adapt sends the same output
with no contract knowledge. Verdict: `artifacts/third_task/gate1/pull_cube-gate1-verdict.json`; raw JSONL:
18 files in `artifacts/third_task/gate1/`.

| Split | Oracle success (95% Wilson) | No-adapt (95% Wilson) | Ceiling gap |
|---|---:|---:|---:|
| seen | 1.000 [0.994, 1.000] | 0.000 [0.000, 0.006] | ~1.000 |
| unseen composition | 0.998 [0.991, 1.000] | 0.000 [0.000, 0.006] | 0.998 |
| long lag | 0.322 [0.286, 0.360] | 0.000 [0.000, 0.006] | 0.322 |

The seen and unseen-composition cells reproduce the benchmark premise on the third task: the same
competent policy is near-perfect when its action ABI is known and always fails (0.000) when the ABI is
hidden, including the absolute/tool-frame composition encoder. Long lag partially collapses even for the
privileged oracle (0.322) — the same falsification finding as Pick (0.027) and Push (0.153), but PullCube
is markedly more lag-robust: its coarse push-to-region success tolerates two- to four-step delay far better
than precision Pick/Push. Contract knowledge alone remains insufficient under lag on all three tasks.

## Tournament slices (frozen PullCube PPO backbone)

Runner `experiments/run_adaptation_slice.py`; hash-addressed output pooled into
`artifacts/adaptation/slices/` (so `experiments/analyze_adaptation.py` aggregates it automatically); the
frozen-checkpoint lookup resolves PullCube from the extended `artifacts/sprint/gate1/jobs.jsonl` ledger.
Belief-family methods share one declared 9-contract pool, 6-step budget, amplitude, backbone, contracts,
and seeds (matched-privilege). Cells are 3 seeds x 2 contracts x 100 = 600 episodes (Wilson 95%); long lag
is 1 seed / 200 episodes.

| Split | exact_belief | fixed_probes | entropy_probes | dualabi | random_probes |
|---|---:|---:|---:|---:|---:|
| seen | 0.952 [0.931, 0.966] | 0.970 [0.953, 0.981] | **0.975** [0.959, 0.985] | 0.968 [0.951, 0.980] | 0.943 [0.922, 0.959] |
| unseen composition | 0.935 [0.912, 0.952] | 0.927 [0.903, 0.945] | **0.982** [0.967, 0.990] | 0.977 [0.961, 0.986] | 0.997 [0.988, 0.999] |
| long lag | 0.275 [0.218, 0.341] | — | — | — | — |

Aggregate: `artifacts/adaptation/tournament.json` (now includes the PullCube cells alongside Pick/Push).

### Significant paired comparisons (3-seed, 600 paired episodes, 95% bootstrap)

- **Unseen composition, entropy > fixed +5.5 [+3.3, +7.8]** — clears the preregistered "five points over
  fixed probes" promotion criterion on the third task.
- Unseen composition: entropy > exact_belief (passive) +4.7 [+2.5, +7.0]; dualabi > fixed +5.0 [+2.8, +7.3];
  dualabi > exact_belief +4.2 [+1.8, +6.5] — all significant.
- Seen: entropy > exact_belief (passive) +2.3 [+0.2, +4.5], significant; entropy vs fixed +0.5, ns.
- entropy vs dualabi ns on both splits (matched success).

### DualABI probe-efficiency Pareto (replicates Pick/Push)

DualABI matches entropy on success (paired deltas ns) while using **2.9 / 2.5 mean probe steps vs 6.0**
for entropy and fixed (~52-58% fewer) and less probe displacement (0.013 / 0.024 vs entropy 0.036 / 0.044
and fixed 0.066 / 0.070). This clears the success-versus-action-cost Pareto promotion criterion on PullCube,
consistent with `reports/adaptation_dualabi.md`.

## Cross-task consistency check — does the ordering replicate?

**Method ordering `entropy > fixed ~ passive > random`:**

- **Seen: replicates exactly.** entropy 0.975 > fixed 0.970 ~ passive (exact_belief) 0.952 > random 0.943,
  with entropy significantly above passive. Entropy is the top method; the passive-vs-fixed gap is small,
  as on Push/seen.
- **Unseen composition: partially replicates.** entropy (0.982) is top-tier and significantly above both
  fixed and passive, and DualABI matches it at half the cost — the core finding holds. But the `> random`
  tail does **not** replicate: random_probes scores 0.997 (highest in the cell). This is the honest anomaly
  below.

**Verdict:** the load-bearing tournament claims replicate on the third task — information-seeking helps
(entropy is top or tied-top and significantly beats passive belief), entropy clears the 5-point gate over
fixed on the unseen split, and DualABI matches entropy at ~half the probe budget. The weaker sub-finding
that "random probing can hurt" (observed on Push/seen) is task- and split-specific and does not generalize
to PullCube.

## Honest anomalies

1. **`success_once` vs `success_at_end` divergence** (Gate 0, above): the PullCube backbone is competent by
   the benchmark's `success_once` measure but does not hold the cube in the goal. Disclosed; measure is
   identical to Pick/Push.
2. **random_probes 0.997 on unseen composition** — anomalously high, breaking the `> random` ordering tail.
   PullCube's coarse push-to-a-0.1m-region success under the absolute/tool-frame unseen contracts is not
   penalized by uninformative agitation, and random pulses can incidentally nudge the cube goalward. This is
   the opposite sign to Push/seen (where random hurt), confirming the random-hurts effect is split/task
   dependent, not universal.
3. **Long-lag oracle 0.322** — higher than Pick/Push; PullCube tolerates delay better, but no-adapt is still
   0.000 and exact_belief long-lag is 0.275 (identification succeeds within episode; residual lag limits it),
   so the "identification alone cannot fix delay" conclusion still holds. Long-lag cells are 1-seed (marked).

## Claim boundary

Supported: official-code PPO integration and short-budget competence for PullCube; exact PPO wrapper/oracle
parity; a three-seed privileged-ceiling gap; and matched-privilege belief-family tournament cells whose
ordering reproduces the Pick/Push tournament. Not supported: published-score reproduction, unprivileged
adaptation success, algorithm superiority beyond the matched-privilege pool, or any claim on the long-lag
split beyond the disclosed one-seed cell. Peg remains excluded, so the overall preregistered three-task gate
that required all of Pick/Push/Peg is unchanged; PullCube restores a third *competent* task to the working
benchmark. All raw episode JSONL, checkpoint hashes, verdicts, and calibration remain under
`artifacts/third_task/` and `artifacts/adaptation/`.

## Files and additive diffs

- Source (additive, ruff + strict mypy clean, full suite 236 passed): `pull_cube` added to
  `benchmarking/gates.py::_FLOORS`, `benchmarking/baselines.py::_TASKS`,
  `benchmarking/ppo_parity.py::_ENVIRONMENT_IDS`, `envs/tasks.py::TASKS`, and the `--task` choices in
  `scripts/evaluate_gate1_slice.py`; registry test updated to cover four tasks.
- `run_adaptation_slice.py::frozen_checkpoint` behaviour is unchanged: PullCube resolves because its three
  seeds x two methods x three splits Gate-1 job entries were appended to `artifacts/sprint/gate1/jobs.jsonl`
  (existing Pick/Push lines untouched).
- Frozen checkpoint + hash: `artifacts/third_task/gate0/pull_cube-frozen-checkpoint.json`.
