# PLUG-AND-VERIFY: the `actionshift-selftest` startup check

Run date: 2026-07-21 UTC. Tool: `actionshift-selftest` (console entry ->
`actionshift.selftest.cli:main`); module: `src/actionshift/selftest/`; tests:
`tests/test_selftest.py`; user docs: the `actionshift-selftest` section of `README.md`.

This packages the action-interface pair's load-bearing finding — a bounded probe
phase identifies a hidden action contract (`reports/adaptation_tournament.md`) —
as a practical, legible self-test: **"is this robot wired the way the policy
thinks?"** It answers with a fail-closed verdict and a scriptable exit code.

## Usage

```
actionshift-selftest [--demo NAME | --hidden-contract C] [--expected C]
                     [--strategy fixed|entropy] [--budget 6] [--amplitude 0.5]
                     [--real] [--json]
# exit codes: 0 = PASS, 1 = MISMATCH, 2 = INCONCLUSIVE
```

## Design

Four small, separable pieces (ruff + strict-mypy clean; additive only, plus the one
console-entry registration in `pyproject.toml`):

- **`library.py`** — a named library of plausible wirings (identity, `swapped-axes`,
  `sign-flip`, `miswired`, `scaled`, `reversed`, `lagged`) forming the *declared
  finite pool* (the exact-belief privilege), plus a demo registry that injects a
  known hidden contract (including `unmodeled`, deliberately outside the pool). All
  default pool members are delta/base/pose-only — the wirings a bounded pose probe
  can actually observe.
- **`identify.py`** — the probe phase. Reuses the tournament's exact stack verbatim:
  a bounded `ProbingBeliefAdapter` (fixed or entropy schedule, budget 6, amplitude
  0.5) folding responses into an `ExactBeliefDriver`. Drives one `ProbeEnvironment`:
  a bit-faithful **synthetic** stand-in (the same `CompleteActionDecoder` the real
  wrapper uses; CPU, no ManiSkill) or a **real** ManiSkill environment. Emits an
  `IdentificationResult`: MAP wiring, per-field marginal posterior mass, and a
  **misspecification statistic** (the MAP hypothesis's chi-square per DoF against the
  calibrated noise scale).
- **`verdict.py`** — a pure, torch-free `decide_verdict` (fail-closed; see below).
- **`cli.py`** — argument parsing, real/synthetic wiring, calibration auto-run
  (`load_or_run_calibration`), human + `--json` rendering, exit code.

Confidence is reported per field as the **marginal posterior mass** on the MAP
wiring's value for that field (summed over pool members sharing it), matching the
"per-field confidence (posterior mass)" requirement.

## Demo transcripts (synthetic, deterministic)

Recorded via `actionshift-selftest --demo <name>` (seed 20260720, 8 envs, entropy,
budget 6). These demonstrate all three verdicts and the fail-closed guard.

### 1. PASS — a correctly-wired robot (`--demo identity`)

```
VERDICT: PASS  (exit 0)
  permutation  (0, 1, 2, 3, 4, 5)   confidence 1.00
  sign         (1, 1, 1, 1, 1, 1)   confidence 1.00
  scale        (1.0 x6)             confidence 1.00
  target/frame/lag                  confidence 1.00
  gripper_inverted                  not checked (out of scope for a pose probe)
  fit residual: 1.00x calibrated noise scale (ok)
  every observable field resolved with high confidence and matches the expected wiring
```

### 2. MISMATCH — a swapped connector + flipped driver (`--demo miswired`)

```
VERDICT: MISMATCH  (exit 1)
  permutation  (1, 0, 2, 3, 4, 5)   confidence 1.00
  sign         (1, 1, 1, -1, 1, 1)  confidence 1.00
  fit residual: 1.00x calibrated noise scale (ok)
  channels 0 and 1 SWAPPED; channel 3 sign FLIPPED
```

The exact-diff string is generated from the identified-vs-expected contract, not
templated — `swapped-axes` alone prints `channels 0 and 1 SWAPPED`; `sign-flip`
alone prints `channel 3 sign FLIPPED`; `lagged` prints `actuation lag differs:
expected 0 step(s), wired 2 step(s)`; etc.

### 3. INCONCLUSIVE — an unmodeled wiring, fail-closed (`--demo unmodeled`)

```
VERDICT: INCONCLUSIVE  (exit 2)
  fit residual: 41.22x calibrated noise scale (TOO LARGE)
  response misspecification: the best hypothesis fits the probe responses at 41.2x
  the calibrated noise scale (threshold 4.0x) -- the true wiring is likely outside
  the declared pool. Abstaining.
```

The hidden contract here is a valid wiring that is **not** in the declared pool.
Rather than snap to the nearest pool member and emit a confident (and wrong)
PASS/MISMATCH, the misspecification guard fires and the tool abstains.

## Verdict-logic guarantees (fail-closed)

`decide_verdict` is a pure function of `(IdentificationResult, expected, floor,
misspec_ratio)`. The guarantees, each covered by a unit test in
`tests/test_selftest.py`:

1. **No false PASS on a bad fit.** If the MAP hypothesis fits the responses worse
   than `misspec_ratio` x the calibrated noise (default 4.0), the verdict is
   `INCONCLUSIVE` regardless of how the fields look — the true wiring is likely
   unmodeled. (`test_misspecification_forces_inconclusive`)
2. **No PASS without full resolution.** PASS requires **every** observable field to
   clear the confidence floor (default 0.9) *and* equal expected. One unresolved
   field -> `INCONCLUSIVE`, never a hopeful PASS. (`test_pass_requires_every_field_resolved`)
3. **No MISMATCH we don't trust.** A field that differs from expected but sits below
   the floor does **not** trigger MISMATCH; it makes the verdict abstain.
   (`test_inconclusive_when_differing_field_is_low_confidence`)
4. **MISMATCH only on confident evidence**, and it survives an unresolved sibling
   field (the confident diff is reported; the unresolved field is disclosed as a
   caveat). (`test_confident_mismatch_survives_an_unresolved_sibling_field`)
5. **Gripper is out of scope, always** — never contributes to PASS/MISMATCH; printed
   as "not checked." (`test_gripper_is_never_in_the_pose_field_diffs`)
6. **Distinct, meaningful exit codes** 0/1/2. (`test_exit_codes_are_distinct_and_meaningful`)

The through-line: the tool certifies PASS only when it can, flags MISMATCH only on
evidence it trusts, and otherwise abstains. It never guesses.

## Tests

`tests/test_selftest.py` — **20 assertions, all green** (CPU, ~2.3 s, no GPU):

- Verdict logic: the six guarantees above plus threshold validation.
- Diff descriptions: readable sentence per field; gripper excluded from pose diffs.
- Library: pool uniqueness, `resolve_pool` de-duplication, unknown-demo rejection.
- **CPU-synthetic identification smoke** (the real probe stack against the
  bit-faithful synthetic env): `swapped-axes` -> MISMATCH, identity -> PASS,
  `unmodeled` -> INCONCLUSIVE via misspecification, and the `fixed` strategy also
  identifies.
- CLI end-to-end: `main([...])` returns the right exit code per demo; `--json` runs.

## Real ManiSkill validation (honest)

The `--real` path runs end-to-end on the sim: it auto-runs response calibration,
drives the probe pulses through a real `PickCube-v1` hidden-contract environment,
and renders the verdict. Measured behavior (4 envs, `--demo swapped-axes`):

- **Budget 6 (the finding's minimal budget): `INCONCLUSIVE`.** The weak
  `pd_ee_delta_pose` response does not let the belief concentrate in six real steps
  (max field confidence ~0.64) — so the tool **fails closed**, exactly as designed.
- **Budget 40: still `INCONCLUSIVE`, but concentrating** — sign 0.99, lag/frame/
  target 1.00, permutation 0.82, scale 0.87, fit residual 0.17x (a good fit); the
  swapped/miswired pool pair splits the permutation marginal just under the strict
  0.90 floor.

This is consistent with the pair's documented result that **short-window real
identification is information-limited under the weak real response model**
(`reports/adaptation_tournament.md`, Round 6). It is a feature, not a defect: on the
real response the tool declines to guess. Synthetic/demo mode uses the calibrated
response model as the identification model, so the minimal 6-step budget is
sufficient and the verdict logic is demonstrated crisply and deterministically.

## Honest limits (full list in the `actionshift-selftest` section of `README.md`)

Cannot check: gripper direction (probe never actuates it), exact scale under
absolute-target control (non-identifiable — the pair's open challenge), lag
*correction* (identification only; correction needs a delay-aware backbone), tool
frame under identity rotation, and any wiring outside the declared pool (abstained
on via the misspecification guard). Within its pool and on the observable pose
fields, it verifies permutation, sign, scale (delta), target, frame, and lag.
