# Adaptation Stage 1: exact-belief adapter runs end-to-end on the real benchmark

Run date: 2026-07-20 UTC
Scope: machinery validation probe, NOT a tournament claim. One seed (`20260720`),
16 episodes per contract, 8 environments, one GPU. Frozen Gate 0 PPO backbone
(`final_ckpt.pt`, sha `3e6c95d6...`), PickCube-v1, real GPU simulation.

## What was built (commits `65347ba`, `4ab87dc`, this one)

The adapter framing from `the sprint plan (git history)
every method = frozen canonical PPO backbone + an interface adapter. Implemented:

- `adaptation/response.py` — calibrated per-channel linear response model (signed gains).
- `adaptation/calibration.py` — contract-independent calibration on the unwrapped env:
  auto-locates the tcp pose block in the flat state observation, fits per-channel
  gain/noise/R2 from random-action rollouts. Task knowledge, never contract knowledge.
- `adaptation/hypotheses.py` — per-hypothesis bit-faithful wrapper replicas
  (`CompleteActionDecoder` clones, decode-then-lag order, identity rotation) + exact
  per-environment Bayesian belief with correct auto-reset mask timing (boundary
  transition invalidated at step t; decoder reset lands at t+1, mirroring the wrapper's
  pending-reset semantics).
- `adaptation/adapters.py` — `ContractAdapter` protocol; oracle / no-adapt / exact-belief.
- `adaptation/maniskill.py` — calibration probe + adapter-generic evaluation loop that
  mirrors the Gate 1 episode accounting exactly (no clamp on encoded raw actions,
  matching the Gate 1 oracle path).
- 12 new CPU tests (171 total pass; ruff + strict mypy clean): synthetic convergence to
  the true contract, oracle parity after convergence, replica/wrapper bit-parity through
  stateful contracts and resets, no-truth-in-pool honesty check, leakage guard.

## Real calibration (PickCube, unwrapped, 64 random steps, 8 envs)

tcp position located at obs offset 19, quaternion at 22. Per-channel gains
(0.032, 0.016, 0.018, -0.035, -0.027, -0.038), sigmas ~0.02-0.05, fit R2
(0.28, 0.11, 0.29, 0.34, 0.09, 0.26). The rotation-channel gains are negative — a real
commanded-frame vs measured-frame convention difference the signed calibration absorbs.
The fit is honestly weak per step (random-action thrash under PD tracking); the belief
compensates by accumulating evidence across steps.

## Probe results (pool of 9 declared hypotheses; belief resets each episode)

| True contract | exact_belief | Gate 1 no-adapt | Gate 1 oracle |
|---|---:|---:|---:|
| seen[1] (delta/base, perm+sign+scale) | 15/16 = 0.938 | 0.000 | 1.000 |
| unseen[0] (absolute/tool, gripper inverted) | 16/16 = 1.000 | 0.000 | 0.995 |
| unseen[1] (absolute/base, gripper inverted) | 16/16 = 1.000 | 0.000 | 0.995 |

Within-episode adaptation from a uniform prior: the belief identifies the hidden
contract from calibrated tcp-response evidence in the first steps of each episode and
then encodes like the oracle, including through the stateful absolute-target and
tool-frame paths on the real simulator.

## Claim boundary (do not overclaim)

- exact_belief is the **pool-privileged algorithmic ceiling**: it knows the declared
  finite candidate pool and the wrapper's decode pipeline shape. It is not a deployable
  learned method and is labeled exactly as Gate 1's registry labels it.
- Probe scale only: one seed, 16 episodes, one task, three contracts. Tournament claims
  require the preregistered three-seed protocol through the job runner.
- Long-lag was not probed; Gate 1 falsification (contract knowledge does not fix delay)
  stands and is expected to bind exact_belief equally.
- The Gate 1 reference numbers are full-scale (600 episodes/cell); the probe numbers are
  16-episode cells. They are context, not a paired comparison.

## Next (per the plan)

Stage 2: probe adapters (fixed/random/entropy) on the same driver with budget/displacement
accounting. Stage 3: trained adapters (DR-static, recurrent GRU, UP-OSI-style regressor,
RMA-style distillation, factorized posterior_only) via a shared collection loop under a
documented training contract distribution disjoint from the frozen evaluation contracts.
Stage 4: DualABI heads; flip registry `runnable` flags only as each end-to-end path lands;
run the preregistered one-seed pruning then three-seed promotion through the sprint runner.

## Stage 2 addendum (2026-07-20, same probe-scale caveats)

Probe adapters landed: `adaptation/probes.py` (fixed / random / entropy strategies on the
exact-belief driver; bounded amplitude 0.5, gripper channel never probed, per-episode
budget with counter reset on auto-reset boundaries) plus probe-step and displacement
accounting threaded through `evaluate_adapter`. 5 new tests (176 total; ruff + strict
mypy clean). Real PickCube probe on seen[1] with budget 6: fixed probes 16/16, entropy
probes 16/16 (passive exact_belief was 15/16); entropy identification cost ~24% less
tcp displacement than fixed (0.046 vs 0.060). Entropy probe selection uses a stateless
instantaneous preview (lag/target state ignored for selection only; belief updates stay
exact) — documented approximation.

## Stage 3 foundation addendum (2026-07-20)

Trained-adapter foundation landed (`adaptation/training.py`): training-contract sampler
over the full 6-DoF grammar with hash rejection of every frozen evaluation contract;
history-window collection; and the UP-OSI-style regressor + eval-time `OsiAdapter`.
Two findings that shaped the design (both measured, kept honest):

1. A flat MLP on raw (raw, response) windows learns NOTHING transferable — held-out
   permutation accuracy stayed at chance (0.18) because the network memorizes training
   permutations instead of learning the identification computation.
2. The working design is sufficient-statistic features + equivariant heads: per
   candidate lag, a pinv least-squares joint regression of responses on
   [raw_t, raw_{t-1}] (block0 = the permutation/sign/scale map for BOTH target
   families; block1 ~ -block0 iff target=absolute), scored by one shared per-cell
   scorer and one shared per-row head, so permutation generalization holds by
   construction. Held-out identification on synthetic contracts: permutation 0.992,
   sign 0.986, target flag 0.992 (window 14).

Boundary: synthetic-environment supervision only so far. The real-ManiSkill matched-
budget collection + training run (frozen PPO under sampled training contracts, then
adapter evaluation against the frozen evaluation contracts) is the next work item and
the prerequisite for flipping `up_osi` runnable in the registry. Gripper inversion and
tool-frame (identity-rotation) remain unidentifiable from pose responses alone —
documented ceiling, needs a grasp-outcome evidence channel later.

## Real-data OSI run (2026-07-20) — honest negative: passive identification fails

First real-ManiSkill matched-budget training run (`experiments/train_osi_real.py`,
`artifacts/adaptation/osi/result.json`): 96 training + 12 held-out contracts
(hash-disjoint from evaluation), 34,560 collection env-steps of frozen-PPO
pass-through under hidden contracts, 6,144 windows, 80 epochs.

- Held-out identification on REAL responses collapsed vs synthetic: permutation
  0.29 (synthetic 0.99), sign 0.55 (~chance), lag 0.42; only target 0.94 survives.
- Adapter evaluation: 0/16 on all three evaluation contracts (a wrong estimate
  encodes like no-adapt; consistent).
- Diagnosis: the frozen PPO's smooth deterministic actions are poor excitation —
  raw_t ~ raw_{t-1} makes the joint regressors near-collinear — and per-step
  response SNR is ~1 (|response| ~ alpha*|action| ~ 0.03 vs sigma ~ 0.03). A
  ridge-regularized fit (added this run; also keeps synthetic accuracy at 0.99)
  tames conditioning but cannot create missing information.
- The discriminative benchmark story this sets up: on the SAME real responses,
  pool-privileged exact belief reaches ~1.0 and six steps of bounded active
  probing reach 1.0, while passive learned system-ID reaches 0.0 — the
  information is obtainable; passive observation under a task policy does not
  obtain it. Echoes the week-one proxy (probes strong). Next trained method:
  the probe-augmented OSI hybrid.
- Excitation ablation launched (random-action collection) to separate
  "excitation-limited" from "noise-limited" cleanly.
