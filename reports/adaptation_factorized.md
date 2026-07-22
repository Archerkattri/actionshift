# Full-grammar factorized belief: grammar knowledge without the pool

Run date: 2026-07-21 UTC. Method: `src/actionshift/adaptation/factorized_grammar.py`;
runner `experiments/run_factorized_slice.py`; tests `tests/test_adaptation_factorized.py`;
artifacts `artifacts/adaptation/factorized_slices/` (hash-addressed per-contract episode JSONL +
summaries). Frozen Gate 1 PPO backbones (same checkpoints as the tournament), real ManiSkill GPU
simulation, **GPU 1 only** (`CUDA_VISIBLE_DEVICES=1`). Seeds 20260718/19/20 (three seeds on every
promoted cell), 100 episodes/contract, 8 envs, two frozen representative contracts per split.

## The scientific move

Current belief methods score a declared **9-contract pool** (`run_adaptation_slice.declared_pool`).
That is a strong privilege: the true contract is one of nine known candidates. The recurrent negative
(`reports/adaptation_recurrent.md`) proved the crux — **discrete hypothesis scoring succeeds where
continuous map estimation fails** at per-step SNR ~1. So the motivated attack on the open unprivileged
challenge is to scale discrete scoring to the **entire declared finite grammar**, converting "pool
privilege" into the strictly weaker "grammar knowledge" (the benchmark's own declared contract space).

The joint grammar is enormous (720 permutations × 2⁶ signs × 7⁶ scales × 2 targets × 4 lags), but under
the benchmark's identity end-effector rotation the decode is **separable per channel**: semantic channel
`i` is produced by exactly one raw channel with one sign and one scale. So the joint factorizes into a
small grid of per-cell evidence scores accumulated as tensor ops, and a MAP contract is read out by a
linear-assignment over channels plus a best-global-mode pick. Per global mode `m = (target, lag)` the
state is a `(6 semantic × 6 raw × 2 signs × 7 scales)` evidence tensor — **4,032 cell-mode scores**,
trivially vectorized on GPU. This is the same decode-predict-compare as `hypotheses.py`, but computed
per channel over the whole grammar instead of over nine enumerated contracts. ActionABI's bridge run
already showed per-channel factorization recovers permutation/sign on this exact data
(`actionabi/reports/labeled_sim_traces.md`); this closes the loop by turning it into a *control*
method and measuring end-to-end task success.

### Grammar-coverage check (stated, test-enforced)

The scale grid `{0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0}` (7 values) covers **every** frozen evaluation
contract's scale. Verified against `benchmarking/gate1_eval.representative_contracts` for all three
splits and `run_adaptation_slice.declared_pool` (the `0.6` value appears in `oracle_contract`, the
`unseen_composition` contract, and the two distractors, so all seven values are required — a 6-value
grid would miss the frozen contracts). Enforced by
`test_adaptation_factorized.GrammarCoverageTest`, which also asserts every frozen `lag ∈ {0,1,2,4}` and
`target ∈ {delta, absolute}`.

## Synthetic convergence — bit-faithful, verified

`tests/test_adaptation_factorized.py` (13 tests, all green; ruff + strict mypy clean) drives the same
bit-faithful synthetic hidden environment as the Stage-1 pool replica (decode-then-lag, identity
rotation) with a **randomly sampled full-grammar contract** and no pool:

- **Convergence + oracle parity.** From uniform, the belief recovers a random full-grammar contract's
  permutation, sign, scale and target exactly and reaches `executed == intent` to `1e-4` (both delta and
  absolute target). The recovered permutation is always a valid bijection matching the truth.
- **Lag identification.** A lagged mode (`lag=2`) is identified as such.
- **Bit-faithful prediction.** The driver's per-cell prediction for the true contract equals the
  wrapper's decoded output channel-for-channel to `1e-5`, across lag and absolute-target modes (this
  validates the history-ring alignment and the `decode(raw_t) − decode(raw_{t-1})` absolute term).
- **Mask timing.** `invalid_mask` discards the boundary-crossing transition and resets that env's
  accumulated evidence and tracked target; `reset_mask` zeros the history ring before the current push.
  Timing is copied from `adapters.py` / `hypotheses.py` exactly (pending reset vs. this-step boundary).
- **Identifiability limits (asserted, not hidden).** A tool-frame contract is reported as `base` yet
  still reaches parity (frame is degenerate under identity rotation). A gripper-inverting contract is
  reported `gripper_inverted=False` and its **gripper channel comes out inverted** while the pose is
  recovered exactly — the unobservability made explicit.
- **Leakage guard.** `FactorizedGrammarAdapter.__init__`, `FactorizedGrammarDriver.__init__`, and the
  probing adapter take **no** `contract` / `true_contract` / `pool` argument (test-enforced). In the
  runner the true contract configures the hidden environment but is never passed to the adapter — it
  sees only its own raw actions and calibrated tcp-pose responses, exactly like the pool belief.

## End-to-end results (real GPU sim, 3 seeds, 600 episodes/cell on the seen split)

Cells that cleared the preregistered 0.3 promotion bar at seed 20260718 were promoted to three seeds.
The `unseen_composition` cells were all < 0.3 and are reported at one seed by the pruning rule.

| Task / split | factorized_grammar (passive) | factorized_grammar_probes | pool exact_belief\* | pool fixed_probes\* | learned (recurrent/OSI)\* | oracle\* |
|---|---:|---:|---:|---:|---:|---:|
| Pick / seen | **0.363** [0.326, 0.403] | **0.458** [0.419, 0.498] | 0.928 | 0.928 | 0.000 | 1.000 |
| Push / seen | **0.633** [0.594, 0.671] | **0.673** [0.635, 0.710] | 0.983 | 0.982 | — | 1.000 |
| Pick / unseen comp. | 0.000 | 0.005 | 0.958 | 0.970 | 0.010 | 0.995 |
| Push / unseen comp. | 0.055 | 0.000 | 0.995 | 0.980 | — | 1.000 |

\* Pool/oracle/learned columns are the matched-setup tournament + recurrent numbers
(`reports/adaptation_tournament.md`, `reports/adaptation_recurrent.md`). Wilson 95% intervals shown for
the three-seed factorized cells.

### The per-contract breakdown is the result — two identifiability walls, cleanly isolated

Pooling the 600 seen episodes by the true contract's `gripper_inverted` flag (each split has one
gripper-inverted and one non-inverted representative):

| Cell | gripper_inverted = False | gripper_inverted = True |
|---|---:|---:|
| Pick / passive | **0.723** (217/300) | 0.003 (1/300) |
| Pick / probes | **0.917** (275/300) | 0.000 (0/300) |
| Push / passive | 0.857 (257/300) | 0.410 (123/300) |
| Push / probes | **0.990** (297/300) | 0.357 (107/300) |

**Wall 1 — gripper (unobservable).** For Pick, the gripper-inverting contract goes to **0.003 / 0.000**
while the non-inverting one reaches **0.723 / 0.917**. Gripper inversion is not observable from the
tcp-pose response, so the belief defaults to `False` and grasps with the wrong gripper sign on inverting
contracts — a total kill for a grasping task. Push (which does not grasp) is unaffected by this on the
`False` contract (0.857 / 0.990) but its gripper-inverted representative also carries the harder
scale/permutation, so it sits lower (0.41 / 0.36) for a *different* reason (Wall 2), not gripper.

**Wall 2 — absolute-target + scale.** The entire `unseen_composition` split (both representatives are
`target=absolute`) collapses to ~0.0 for **both** tasks — including Push, which is gripper-agnostic. The
pool belief reaches 0.958–0.995 on the same contracts. The difference is exactly what the pool gets for
free: the exact scale and target. Per the ActionABI bridge on this data, **scale is not identifiable**
from the weak `pd_ee_delta_pose` response (it is systematically attenuated to ≈0.6× truth; accuracy
0.24–0.42), and **target identification is excitation-dependent**. Under a `delta` contract a scale
error only rescales each step (bounded per-step error, hence Push/seen still reaches 0.99). Under an
`absolute` contract the encode integrates through a cumulative target, so the same scale error
**accumulates into unbounded drift** and the episode fails. Absolute-target control is therefore the
sharp failure mode of grammar-only belief, and it is caused by scale non-identifiability, not by the
assignment.

**Probes help identification, not the walls.** Fixed probes lift the clean delta cells substantially
(Pick/False 0.723 → 0.917; Push/False 0.857 → 0.990 — the large hypothesis space rewards active
excitation) but cannot manufacture the gripper bit or the missing scale evidence, so they leave the two
walls in place (and slightly perturb the hard Push gripper-inverted cell, 0.410 → 0.357).

## Overhead

Adapter-only microbenchmark on GPU 1 (8 envs, 300 steps, warmup excluded, sim excluded):

- factorized grammar (passive): **13.2 ms/step**
- pool exact-belief (9 contracts): **8.5 ms/step**

So the full-grammar belief adds **≈ +4.7 ms/step (+55%)** over the nine-hypothesis pool. The overhead is
dominated by MAP extraction — 64 CPU `linear_sum_assignment` calls per step (8 modes × 8 envs, each a
6×6 assignment) — and is trivially reducible (the MAP need not be recomputed every step). End-to-end the
eval is GPU-simulation-bound: the measured `mean_seconds_per_step` is **0.0061–0.0076 s** across cells,
comparable to the pool runs; the grammar scoring is not the bottleneck.

## Honest privilege statement

The only knowledge this method uses is:

1. the **declared finite grammar** — the benchmark's own contract space (permutations of 6 channels,
   signs, the 7-value scale grid, `{delta, absolute}`, `{0,1,2,4}` lag); and
2. the **contract-independent response calibration** (`alpha` / `sigma`), measured on the *unwrapped
   identity* environment and shared by every tournament method (same `ResponseModel` as the pool belief,
   for direct comparability).

It never receives the true contract (test-enforced) and never enumerates or is given the 9-contract
pool. This is strictly weaker than the pool privilege: it is "knowledge of the space of possible
interfaces" rather than "knowledge of nine specific candidates one of which is true". No eval-contract
tuning was done; the grammar grids and probe settings (budget 6, amplitude 0.5) are fixed a priori.

**Identifiability limits (declared, not scored around):**
- **frame** — degenerate under the identity rotation (`base` ≡ `tool`); collapsed to `base`.
- **gripper** — unobservable from the tcp-pose response; fixed to `False` (marked unidentified). This
  caps grasping tasks on gripper-inverting contracts (measured: Pick/inverted → 0.00).
- **scale** — expressible in the grammar but **not identifiable** from the weak, attenuating
  `pd_ee_delta_pose` response (ActionABI-confirmed); the dominant cause of the sub-ceiling delta success
  and the absolute-target collapse.

## Verdict: grammar knowledge partially — not fully — closes the unprivileged gap

This is the first belief-family method to substantially succeed **without the pool**. It converts the
learned unprivileged methods' honest ~0.0 (passive OSI, recurrent) into **0.36–0.67 on the seen split**,
and reaches the pool ceiling on the clean sub-cell (Push, delta, non-inverted gripper: **0.990**). That
is real: discrete full-grammar scoring extracts genuine control signal from grammar knowledge alone,
confirming the recurrent report's thesis that *scoring* beats *estimating* on this response model.

But it does **not** fully close the gap. It stays well below the pool-privileged 0.93–0.98 and collapses
entirely on absolute-target and gripper-inverted contracts. The result **localizes exactly what the pool
privilege buys** beyond grammar knowledge: not "which of nine contracts", but specifically the three
quantities the weak pose response cannot reveal — **the exact scale, the gripper sign, and (through
scale) reliable absolute-vs-delta control**. That is the useful, honest finding, and per the honesty
rule it *strengthens* the open-challenge claim: the benchmark's hidden interface remains genuinely hard
for a deployable, unprivileged method, and the residual difficulty is now pinned to specific,
named, response-model-limited fields rather than to the size of the hypothesis space.

The motivated next moves it points to (not run here): fold the **magnitude-dependent controller gain**
into the calibrated response model (task knowledge, not contract knowledge) to make scale identifiable
and rescue absolute-target control; and add an **outcome / grasp-success channel** to identify the
gripper bit that pose responses cannot.

## Claim boundary

- Unprivileged at evaluation: the adapter sees only its own raw actions and calibrated tcp-pose
  responses; a test asserts none of its constructors take a contract/pool argument. The true contract
  configures the hidden environment only.
- Matched calibration + backbone + contracts + seeds with the pool belief; the only change is
  pool → full grammar. Three seeds (20260718/19/20, 600 episodes) on every promoted seen cell; one seed
  on the sub-0.3 unseen cells by the preregistered pruning rule.
- Scale is expressible in the grammar; its non-identifiability here is a **controller/response-model**
  property of `pd_ee_delta_pose`, consistent across this work and the ActionABI bridge, not a grammar
  limitation.
- ActionShift-benchmark results under `pd_ee_delta_pose` state control with the identity-rotation
  wrapper; no external-benchmark or hardware claim is made. `long_lag` was not evaluated (the pool
  family already collapses there; grammar knowledge is not the lever on lag).
