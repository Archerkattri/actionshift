# Transformer in-context adaptation — feasibility adjudication (Vintix)

Date: 2026-07-21. This report resolves the ActionShift method registry's last conditional
baseline, `transformer_icl` (`src/actionshift/benchmarking/adaptation.py`). The registry rule
(project registry rule): *"Transformer ICL and SPACE/GLAM-style methods are conditional on a faithful,
pinned, runnable implementation — never rename an approximation as a reproduction."*

**Verdict: EXCLUDE. Vintix cannot be applied to ActionShift without new architecture heads and
full retraining, so any run would be a Vintix-derived approximation, not a faithful reproduction.
No install, shim, or evaluation was performed — the exclusion is decided on the architecture.**

---

## 1. Pinned source

| Field | Value |
|---|---|
| Repository | `https://github.com/dunnolab/vintix` |
| Pinned commit | `2b10276d49361e1b4e1440df63a2901c95671750` (2025-05-23, default branch head) |
| Paper | Polubarov et al., "Vintix: Action Model via In-Context Reinforcement Learning," ICML 2025, arXiv:2501.19400 |
| Code license | Apache-2.0 |
| Dataset license | CC BY-SA 4.0 |
| Pretrained checkpoint | `dunnolab/Vintix` on Hugging Face (`snapshot_download(repo_id="dunnolab/Vintix")`) |
| Follow-up | Vintix II (DPT, ICLR 2026), `github.com/dunnolab/vintix-II` — same head-group design, same conclusion |

Recon (git history: `docs/weakness_sota_recon.md` §4; summarized in the Related work section of README.md) had already flagged Vintix as the only runnable ICRL
candidate with real manipulation-suite exposure, but with hidden/compositional action-space eval
marked "No". This adjudication confirms and grounds that flag with the concrete architecture facts.

## 2. What Vintix requires from a task (evidence)

- **Dimension-grouped encoder/decoder heads.** Paper (§ architecture, arXiv:2501.19400):
  *"For each group, a separate encoder and decoder MLP head are created, enabling the model to map
  variable observation and action spaces into a shared embedding space."* The group key is the
  `(observation_dim, action_dim)` pair. The model is *"task-agnostic in the sense that it has access
  only to the dimensionality-based group identifier, but not to an individual task identifier."*
- **Cannot accept an unseen `(obs, action)` dimensionality without retraining.** The heads are
  created "based on observation and action space dimensionalities"; a new dimensionality pair "would
  require new MLP heads, necessitating retraining." This is the load-bearing incompatibility.
- **Training domains (no ManiSkill).** MuJoCo v4, Meta-World v2 (state `(39,)`, action `(4,)`),
  Bi-DexHands (state up to `(428,)`, action up to `(52,)`), and Industrial-Benchmark. ManiSkill is
  not among them, so no ManiSkill observation distribution, dynamics, or `pd_ee_delta_pose` action
  semantics were ever seen.
- **In-context competence is distilled per task.** Vintix is built by Algorithm Distillation over
  each training task's *learning histories*; at inference it reproduces that RL improvement operator
  by accumulating `(o, a, r)` triplets across episodes. It adapts toward the optimum of a task it was
  trained on. It has no mechanism for, and no training exposure to, recovering a hidden action-space
  remapping.
- **Inference call shape.** `model.reset_model(task_name, ...)` then
  `model.get_next_action(observation=obs, prev_reward=reward)`. `reset_model` selects the per-task
  head group and the running observation/reward normalization statistics fitted during training.
  It needs the reward channel in context (fine under ActionShift leakage discipline — the adapter
  protocol already exposes obs, raw action, and response/reward), but it also needs the task to be
  one whose head-group and normalization it holds.

## 3. What ActionShift requires from a baseline

The other tournament methods are `ContractAdapter`s (`src/actionshift/adaptation/adapters.py`): they
consume the same leakage-safe `(obs, raw action, response/reward)` stream from a frozen ManiSkill PPO
backbone under a hidden compositional contract, with no access to the contract or the belief pool.
The frozen PickCube setup is **observation_dim = 42, action_dim = 7** (`pd_ee_delta_pose`: 6-DoF EE
pose delta + 1 gripper; confirmed `reports/baseline_budgets.json`), state observations, ManiSkill3.

## 4. Feasibility adjudication (the three required questions)

**(a) Do the pretrained checkpoints cover anything compatible?** No. The nearest manipulation domain
is Meta-World, whose dimensionality group is `(39, 4)` — neither the obs dim (42) nor the action dim
(7) matches. No Vintix training group is `(42, 7)`: Meta-World `(39, 4)`; the MuJoCo action-dim-7
member is Pusher `(23, 7)` (right action dim, wrong obs dim); Bi-DexHands is far larger. There is no
head into which ActionShift's `(42, 7)` stream can be fed. **Fails.**

**(b) Does the architecture accept arbitrary obs/action dims, or is it tied to training domains?**
Tied. Heads exist only for dimensionality pairs present at training time; a new pair requires
constructing and training new encoder/decoder MLP heads. The architecture is explicitly not
dimension-agnostic. **Fails.**

**(c) Does its evaluation protocol match ActionShift adaptation?** No. Vintix's cross-episode
`(o, a, r)` accumulation adapts toward the optimum of a *known training task*; ActionShift's protocol
requires within/cross-episode identification of and recovery from a *hidden action-ABI contract* on a
task Vintix never saw. Even with matching dims, the per-task normalization and the distilled
improvement operator would target the wrong task's behavior, and there is no reason it would identify
a permutation/sign/scale/lag/gripper remap it was never trained to expect. **Fails.**

Any one of (a)–(c) is disqualifying; all three fail.

## 5. Why a run would violate the registry rule

To make Vintix produce actions for ActionShift PickCube one must: (i) add new `(42, 7)` encoder/
decoder MLP heads absent from the checkpoint (architectural surgery), and (ii) train them — and, for
non-trivial competence, distill an RL improvement operator — on ManiSkill learning histories under
the contract distribution (full retraining), because the checkpoint has zero ManiSkill exposure. That
is a new model trained on ActionShift, not the published Vintix. Labeling such a run "Vintix" (or even
"transformer ICL reproduction") is exactly the "rename an approximation as a reproduction" move the
handoff forbids. The honest structured exclusion is therefore the correct and fully successful
outcome, and no venv/GPU work was undertaken.

The registry `transformer_icl` reason string was updated (small additive diff, ruff + strict mypy
clean) to cite this adjudication, the pinned commit, and the concrete incompatibility.

## 6. What a future faithful ICL baseline would require

A faithful transformer-ICL baseline for ActionShift is a real, well-scoped project, not a checkpoint
drop-in. It would need at least one of:

1. **An ICRL model natively trained on ManiSkill under the ActionShift contract distribution.** Adopt
   Vintix's (or Vintix II / DPT's) Algorithm-Distillation recipe, but generate the training corpus
   from ManiSkill `pd_ee_delta_pose` learning histories with per-episode hidden contracts sampled from
   the same finite grammar the benchmark uses (permutation/sign/scale/target/frame/lag/gripper). Build
   the `(42, 7)` head group. Then the in-context adaptation target *is* contract recovery, and the run
   is a faithful "ICRL-trained-on-ActionShift" baseline (labeled as locally trained, not as Vintix).
2. **A dimension-agnostic tokenizer** (per-dimension or set-token embedding rather than per-group MLP
   heads) so a single model can ingest `(42, 7)` without new heads — but this is a different
   architecture than published Vintix and still needs training exposure to ManiSkill + contract shift
   to have any competence; it does not rescue the pretrained checkpoint.

Either path is a training run of comparable effort to the delay-aware backbone, budgeted and labeled
as a local method — consistent with how `delay-aware augmented-state PPO (local)` and the
`UP-OSI-style`/`RMA-style` baselines are already handled. Until such a model is trained, `transformer_icl`
correctly stays `runnable=False` with a documented, evidence-backed reason.

## 7. Provenance

- Sources fetched: `github.com/dunnolab/vintix` (README, license, commit list via GitHub API),
  `arxiv.org/abs/2501.19400` + `arxiv.org/html/2501.19400v1` (architecture/tokenization),
  `huggingface.co/dunnolab/Vintix` (checkpoint). WebFetch used for all; quotes above are from those
  fetches.
- ActionShift facts: `src/actionshift/adaptation/adapters.py` (adapter protocol),
  `reports/baseline_budgets.json` (obs 42), `src/actionshift/benchmarking/adaptation.py` (registry rule), Gate 0 frozen PickCube
  `pd_ee_delta_pose` setup.
- Registry diff: `src/actionshift/benchmarking/adaptation.py`, `transformer_icl.reason` (ruff + strict
  mypy clean).
- Time-boxed architecture adjudication; no venv created, no download, no GPU used.
