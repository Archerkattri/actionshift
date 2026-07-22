# Rotation v2: de-degenerating the frame axis of the ActionShift benchmark

Run date: 2026-07-21 UTC. GPU 1 only (`CUDA_VISIBLE_DEVICES=1`). Frozen Gate-0/1 PPO
backbones. Seed 20260718 (belief unseen cells promoted to 20260718/19/20). Runners:
`experiments/run_rotation_v2_parity.py` (frame=tool ceiling cells),
`experiments/run_adaptation_slice.py --rotation-mode {identity,real}` (belief slices),
`experiments/run_rotation_v2_campaign.sh` (driver). Artifacts:
`artifacts/adaptation/v2_rotation_slices/` (hash-addressed; `rotation_mode` enters the
job hash only for the v2 real variant, so every v1 artifact keeps its hash).

## The scope weakness this fixes

The hidden-contract wrapper decodes `frame="tool"` contracts by rotating the pose twist
by the end-effector rotation `R` (`contracts/transforms.decode_complete_action`). But the
benchmark wrapper was constructed with **no `ee_rotation_provider`**, so `R = I` always.
Under identity rotation, `frame="base"` and `frame="tool"` decode to the *same* canonical
command — the frame axis was **observationally free**: no method, and no oracle, could tell
base from tool, and a no-adapt policy paid nothing for a tool contract. The frame degree of
freedom in the contract grammar was decorative.

v2 wires the **live tcp rotation** into the wrapper so tool-frame contracts are decoded
against a genuinely non-identity end-effector axis, and threads the same rotation through
the oracle path and the pool belief so both invert the wrapper exactly. Default stays
`identity`; `real` is the opt-in v2 variant.

## What changed mechanically

- **Rotation provider** (`benchmarking/ppo_parity.make_tcp_rotation_provider`,
  `contracts/transforms.quaternion_to_rotation_matrix`): a zero-arg callable reading
  `agent.tcp.pose.q` (ManiSkill `wxyz`) → batched on-device `(num_envs,3,3)` rotation,
  passed into `HiddenContractWrapper` at construction. Gated by
  `rotation_mode: "identity"|"real"` threaded through `_make_environment`,
  `evaluate_ppo_checkpoint`, `adaptation.maniskill.evaluate_adapter`, and
  `run_adaptation_slice.py`. Default `identity` → provider is `None` → the wrapper's
  identity path is bit-unchanged.
- **Oracle path (v2)** (`policy_action_for_condition`, `evaluate_ppo_checkpoint`): reads the
  pre-step tcp rotation from the same env at the same state the wrapper decodes against and
  inverts the tool-frame encode with that exact `R`. Same object, same step ⇒ `R_encode ==
  R_decode` ⇒ inversion exact.
- **Pool belief (v2)** (`adaptation.hypotheses`: `HypothesisSimulator.step`,
  `ExactBeliefDriver.update`/`map_encode`, `resolve_rotation`; consumed by
  `ExactBeliefAdapter`, `ProbingBeliefAdapter`, `DualABIProbeAdapter`): the 9-contract pool's
  per-hypothesis replicas now decode against the **observed rotation sequence** — recovered
  from the calibrated quaternion slice of the observation
  (`adaptation.maniskill.rotation_from_observation`), which is exactly the quaternion the
  wrapper's provider read — so each replica stays bit-faithful to the wrapper under a real
  end-effector frame, and `map_encode` encodes the MAP hypothesis against the current
  rotation. `ee_rotation=None` everywhere reproduces v1 identity behaviour.
- Every `ContractAdapter` gained an `ee_rotation` keyword (task knowledge from the
  observation, never the true contract; a leakage-guard test enforces this). The learned
  identifiers (`osi`/`recurrent`/`probe_osi`) and the factorized-grammar belief accept and
  ignore it — documented below.

Tests: `tests/test_rotation_v2.py` (provider quaternion→matrix + env factory; v2 oracle
inversion of a tool decode under real rotation; replica bit-faithfulness vs the wrapper
decoder under real rotation incl. lag/absolute/reset; belief identifies tool-only-with-real
rotation; MAP-encode round-trips through the wrapper under real rotation). ruff + strict
mypy clean; full suite green except one concurrent-agent WIP file (`scale_corrector`,
unrelated to rotation).

## Parity evidence — oracle inversion is exact under live rotation

Pure-frame contract (identity permutation/sign/scale, `frame="tool"`): the *only* difference
from the unwrapped policy is the tool-frame expression of the twist. 100 eps, num_envs 16,
seed 20260718.

| Cell | Pick | Push | reading |
|---|---:|---:|---|
| ceiling (identity condition, real) | 1.000 | 1.000 | frozen backbone clean ceiling |
| **oracle**, tool, **rotation=real** | **1.000** | **1.000** | inversion exact ⇒ full ceiling recovered |
| no_adapt, tool, rotation=real | 0.000 | 0.000 | the frame axis is now a real gap |
| oracle, tool, rotation=identity (v1) | 0.970 | 1.000 | v1 control |
| no_adapt, tool, rotation=identity (v1) | **1.000** | **1.000** | v1 DEGENERACY: tool == base, no cost |

The load-bearing pair: **`no_adapt` on a `frame=tool` contract is 1.000 under identity
(v1) and 0.000 under real rotation (v2)** on both tasks. v1 literally could not distinguish
base from tool; v2 makes the frame a full oracle-vs-no-adapt gap (1.000 vs 0.000). And
**`oracle` = 1.000 = ceiling under live rotation** proves the v2 oracle inverts the
wrapper's live-rotation decode exactly.

## v2 Gate-1-style ceiling (frame=tool)

Directly from the parity cells above: on a `frame=tool` contract with real rotation,
oracle 1.000 vs no_adapt 0.000 (both tasks). This is the Gate-1 premise (identification is
necessary and, for the frame axis, now non-trivial) re-established on a genuinely
non-degenerate frame axis.

## v1-vs-v2 belief cells (exact_belief, entropy_probes)

100 eps/contract, num_envs 8. **Seen** split at seed 20260718 (control — see below).
**Unseen** split promoted to 3 seeds (20260718/19/20; 600 eps/cell) with pooled Wilson 95%
intervals, since every cell exceeds the 0.3 promotion threshold. Overall = pooled over the
split's two representative contracts; the tool-frame contract is `c0`, base is `c1`.

Seen split (both contracts base) — 1-seed control:

| Task | method | v1 (identity) | v2 (real) |
|---|---|---:|---:|
| Pick / seen | exact_belief | 0.945 | 0.920 |
| Pick / seen | entropy_probes | 0.980 | 0.945 |
| Push / seen | exact_belief | 0.985 | 0.995 |
| Push / seen | entropy_probes | 1.000 | 1.000 |

Unseen_composition split (c0 = real tool contract) — 3-seed pooled, Wilson 95%:

| Task | method | overall v1 | overall v2 | **tool c0** v1 | **tool c0** v2 |
|---|---|---:|---:|---:|---:|
| Pick | exact_belief | 0.975 [.959,.985] | 0.938 [.916,.955] | 0.973 [.948,.986] | **0.900 [.861,.929]** |
| Pick | entropy_probes | 0.978 [.963,.987] | 0.965 [.947,.977] | 0.980 [.957,.991] | **0.977 [.953,.989]** |
| Push | exact_belief | 1.000 [.994,1.00] | 0.972 [.955,.982] | 1.000 [.987,1.00] | **0.943 [.911,.964]** |
| Push | entropy_probes | 0.982 [.967,.990] | 0.998 [.991,1.00] | 0.963 [.936,.979] | **0.997 [.981,.999]** |

Reading:

- **Seen split (both contracts base): v1 ≈ v2** everywhere — differences are within GPU-sim
  episode noise (see reproducibility note). Correct by construction: rotation only affects
  tool-frame decode, so base contracts are rotation-invariant. This is the built-in control.
- **Unseen split (contains a real tool contract, c0):** the belief family **still reaches the
  ceiling** with a genuinely non-identity frame. Passive `exact_belief` on the tool contract
  pays a small, statistically real cost (Pick 0.973→0.900, Push 1.000→0.943 — Wilson intervals
  separated) — the honest price of the now-live axis it must infer — while active
  `entropy_probes` fully absorbs it (Pick 0.980→0.977 ns, Push 0.963→0.997, real ≥ identity):
  bounded probing excites the rotation-sensitive channels and resolves the frame.

## Frame-identification verdict

**Frame identification works from real rotation evidence.** Two independent lines:

1. *Synthetic, exact* (`tests/test_rotation_v2.py::test_belief_identifies_tool_frame_only_with_real_rotation`):
   with a base and a tool hypothesis identical except frame, the pool belief concentrates
   >0.99 on the true tool contract within a few steps under real rotation, and stays exactly
   at the uniform prior under identity (the two are observationally identical there).
2. *End-to-end, real sim* (3-seed pooled): on the unseen split whose `c0` is a real tool
   contract, the pool belief's success on that contract stays 0.900 [.861,.929] (Pick) /
   0.943 [.911,.964] (Push) passively and 0.977 / 0.997 with entropy probing under real
   rotation — versus the no_adapt floor of ~0 for a tool contract (parity cells). The belief
   is decoding the tool hypothesis against the observed rotation and identifying the frame;
   it does not collapse.

So the answer to "does the belief family still reach the ceiling when frame is a REAL axis?"
is **yes for the privileged 9-contract pool** — with a small, measured passive penalty that
active probing erases.

## What v2 adds to the benchmark's scope (honest statement)

- v2 promotes `frame` from a **decorative** contract axis to a **real, identifiable** one.
  Before, base/tool were observationally identical (the parity table's v1 row: no_adapt on
  tool = 1.000); after, they are a full oracle-vs-no-adapt gap (1.000 vs 0.000). Any claim
  that a method "handles the frame axis" is now falsifiable; previously it was vacuous.
- The pool belief's privilege is unchanged (it knows the finite pool); what changed is that
  the pool now contains **genuinely distinct** base and tool hypotheses, and the belief must
  use rotation evidence to separate them — which it does.
- **Factorized-grammar belief limit (a finding, not a bug).** The full-grammar factorized
  belief factorizes the decode **per channel**, which holds only under identity rotation.
  A tool-frame decode multiplies the translation (and rotation-vector) sub-blocks by `R`,
  which **couples channels** (`obs_i` depends on several raw channels), breaking the
  per-channel separability the factorization requires. The factorized adapter therefore
  cannot represent tool-frame hypotheses under real rotation and correctly stays `frame=base`
  (its documented identifiability limit). Consequence: under v2, the *unprivileged*
  grammar-knowledge belief cannot identify a real tool frame — this is a genuine new gap the
  de-degenerated axis exposes, and the honest boundary of what grammar-knowledge alone buys.
  Frame identification under v2 currently requires either the pool privilege (works, above)
  or a richer-than-factorized evidence model.

## Backward compatibility (non-negotiable — verified)

- `rotation_mode="identity"` is the default on every entry point. The `rotation_mode` key is
  added to the slice job hash **only** when `!= "identity"`, so identity-mode job hashes are
  byte-identical to the pre-rotation runner. Verified: a `pick_cube exact_belief seen
  20260718` identity re-run reproduces job id `c68f13fa5492bbf0` with **no** `rotation_mode`
  field, and re-runs are stable at that hash.
- **Reproducibility note (honest):** the benchmark is reproducible at the **hash/config**
  level, not bit level — ManiSkill GPU physics is non-deterministic, so two identical-config
  runs give episode-level rates that vary within a couple of points (identity re-runs:
  0.915 / 0.920 same hash, byte-different episode logs). The seen-split v1≈v2 cells above and
  all prior rounds live inside that same noise band; that is why the tournament uses Wilson
  intervals and multiple seeds. The parity result (1.000 vs 0.000) is far outside it.
- Unrelated to rotation: existing v1 slice artifacts are schema `1.0`; the current runner
  emits schema `1.1` (a concurrent `calibration_version` change), so their *numeric* rates
  shifted slightly (0.935→~0.92) before this task touched anything. The rotation change is
  orthogonal and hash-invariant in identity mode.

## Provenance

Pick backbone sha256 `3e6c95d6…`, Push `a4a02198…` (frozen Gate-1 checkpoints). Pool sha
`b177b082…` (identity + 6 frozen representatives + 2 distractors, unchanged from the
tournament). Calibration `v1` (pose-only linear). Parity artifacts
`artifacts/adaptation/v2_rotation_slices/parity-{pick,push}_cube-*.json`; belief slices
`artifacts/adaptation/v2_rotation_slices/*.summary.json` (+ per-episode `*.jsonl`).
