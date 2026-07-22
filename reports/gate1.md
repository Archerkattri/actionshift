# Gate 1: real hidden-contract PPO ceiling slice

Run date: 2026-07-19 UTC  
Seeds: `20260718`, `20260719`, `20260720`  
Episodes: 7,200  
Tasks: Gate 0-passing PickCube-v1 and PushCube-v1  
Backbones: frozen parity-safe official-code PPO checkpoints

## What was actually run

This is a real ManiSkill hidden-contract validity slice, not the full adaptation-method tournament.
For every task, seed, and split, two frozen six-pose-channel contracts were evaluated for 100 episodes
each. The privileged oracle path knows the contract and encodes the canonical PPO action before the
hidden wrapper; no-adapt sends the same PPO output without contract knowledge. Initial conditions are
matched by experimental seed, contract, and episode. Contract-specific environment seeds are retained
separately.

| Task / split | Oracle success (95% Wilson) | No-adapt success (95% Wilson) | Paired difference (95% bootstrap) |
|---|---:|---:|---:|
| Pick / seen | 1.000 [0.994, 1.000] | 0.000 [0.000, 0.006] | 1.000 [1.000, 1.000] |
| Pick / unseen composition | 0.995 [0.985, 0.998] | 0.000 [0.000, 0.006] | 0.995 [0.988, 1.000] |
| Pick / long lag | 0.027 [0.016, 0.043] | 0.005 [0.002, 0.015] | 0.022 [0.008, 0.037] |
| Push / seen | 1.000 [0.994, 1.000] | 0.003 [0.001, 0.012] | 0.997 [0.992, 1.000] |
| Push / unseen composition | 1.000 [0.994, 1.000] | 0.000 [0.000, 0.006] | 1.000 [1.000, 1.000] |
| Push / long lag | 0.153 [0.127, 0.184] | 0.000 [0.000, 0.006] | 0.153 [0.125, 0.182] |

Every cell contains 600 episodes (two contracts x 100 episodes x three seeds). The analyzer requires
identical pairing keys and uses intact episode pairs for the bootstrap.

## Interpretation

The seen and unseen-composition results validate the benchmark premise: the same competent policy is
nearly perfect when its action ABI is known and almost always fails when the ABI is hidden. The
absolute/tool-frame composition encoder also remains near-perfect, so these results are not explained
by a broken oracle path.

Long lag is different. Even the privileged encoder cannot make a frozen reactive PPO policy robust to
two- or four-step delay: Pick falls to 2.7% and Push to 15.3%. Contract knowledge alone is therefore
insufficient; long-lag success requires a policy trained for delayed dynamics or predictive control.
This is useful falsification evidence and prevents treating the oracle encoder as a universal ceiling.

## Methods not run and why

Only oracle and no-adapt currently have trained, checkpoint-compatible, end-to-end execution paths.
Domain randomization, recurrent robustness, UP-OSI-style, RMA-style, fixed/random probes,
posterior-only, entropy probing, exact belief, and DualABI have tested software components but no
trained matched-budget adapter checkpoints wired to these official PPO backbones. Transformer ICL and
SPACE/GLAM-style methods also lack a pinned faithful runnable implementation. They remain structured
exclusions; no proxy is renamed as a reproduction.

Consequently:

- this report supports a three-seed **privileged-ceiling gap**, not DualABI or adaptation-method
  superiority;
- no method earns the preregistered five-point, 30%-recovery, safety, or Pareto promotion gate;
- the earlier controlled proxy result where fixed probes beat DualABI remains the only direct
  fixed-probe comparison and is not overwritten;
- training the missing adapters is the highest-priority longer run, but results cannot exist before
  those checkpoints and matched training loops exist.

## Evidence and failure ledger

- `artifacts/sprint/gate1/jobs.jsonl`: 36 hash-addressed jobs with checkpoint hashes.
- `artifacts/sprint/gate1/gate1-verdict.json`: Wilson summaries and paired bootstrap intervals.
- Raw JSONL: 36 files, 200 episodes each, retained locally under `artifacts/sprint/gate1/`.
- PegInsertionSide is excluded because no Gate 0 short run met its competence floor.
- Safety violations, action cost, posterior calibration, probe displacement, and recovery are not
  emitted by this frozen-PPO slice and therefore are not imputed.
- SAPIEN used its bundled Vulkan ICD fallback; all jobs completed and GPU simulation remained active.
