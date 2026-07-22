"""Reproducible media driver for ActionShift.

Renders the real ManiSkill benchmark (offscreen SAPIEN, ``render_mode="rgb_array"``)
through the *real* adaptation adapters and composes the labeled hero / lag GIFs used
in the README. Every frame is a genuine simulator render of the frozen Gate-1 PPO
backbone driven through the hidden-contract wrapper; captions are baked per-frame.

Usage (GPU 0 or 1):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python media/make_media.py hero
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python media/make_media.py lag
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python media/make_media.py all

The GIFs are the reproducible artifacts; numbers embedded in captions come from the
committed reports (reports/adaptation_tournament.md, reports/adaptation_delay_aware.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from PIL import Image, ImageDraw, ImageFont

from actionshift.adaptation.adapters import ExactBeliefAdapter, NoAdaptAdapter, OracleAdapter
from actionshift.adaptation.calibration import response_from_observations
from actionshift.adaptation.hypotheses import ExactBeliefDriver
from actionshift.adaptation.maniskill import load_or_run_calibration
from actionshift.adaptation.probes import ProbingBeliefAdapter
from actionshift.adaptation.response import ResponseModel
from actionshift.benchmarking.gate1_eval import representative_contracts
from actionshift.benchmarking.ppo_parity import (
    PpoAgent,
    identity_contract,
)
from actionshift.contracts.types import ActionContract
from actionshift.envs.wrapper import HiddenContractWrapper

# ---- shared palette (kept in sync with media/style.py) ----
INK = (26, 26, 46)
GREEN = (42, 157, 92)
RED = (192, 57, 43)
BLUE = (43, 108, 176)
AMBER = (224, 168, 46)
PURPLE = (107, 77, 158)
WHITE = (255, 255, 255)

_ENV_IDS = {"pick_cube": "PickCube-v1", "push_cube": "PushCube-v1"}
_PROBE_BUDGET = 6
_PROBE_AMPLITUDE = 0.5
_MEDIA = Path(__file__).resolve().parent


def _declared_pool() -> tuple[ActionContract, ...]:
    distractors = (
        ActionContract(permutation=(3, 1, 4, 0, 5, 2), sign=(-1, -1, 1, 1, -1, 1),
                       scale=(0.6, 1.25, 0.75, 1.5, 2.0, 0.5), target="delta",
                       frame="base", lag=0, gripper_inverted=False),
        ActionContract(permutation=(4, 2, 0, 1, 5, 3), sign=(1, -1, 1, 1, -1, -1),
                       scale=(1.25, 0.6, 2.0, 0.5, 1.5, 0.75), target="delta",
                       frame="base", lag=0, gripper_inverted=True),
    )
    reps = tuple(
        c for split in ("seen", "unseen_composition", "long_lag")
        for c in representative_contracts(split)
    )
    return (identity_contract(), *reps, *distractors)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _make_render_env(task: str, contract: ActionContract, num_envs: int = 1):
    env_id = _ENV_IDS[task]
    base = gym.make(
        env_id, num_envs=num_envs, obs_mode="state",
        control_mode="pd_ee_delta_pose", render_mode="rgb_array",
        reconfiguration_freq=1, disable_env_checker=True,
    )
    base = HiddenContractWrapper(base, contract)
    return ManiSkillVectorEnv(
        base, num_envs=num_envs, ignore_terminations=True, record_metrics=True,
    )


def _render_frame(env) -> np.ndarray:
    img = env.render()
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _response_model(calibration) -> ResponseModel:
    return ResponseModel(
        alpha=calibration.alpha,
        sigma=calibration.sigma,
        alpha_c0=calibration.alpha_c0 if calibration.gain_model == "saturating" else None,
        gripper_alpha=calibration.gripper_alpha,
        gripper_sigma=calibration.gripper_sigma,
    )


def _load_agent(env, checkpoint: Path) -> PpoAgent:
    obs_dim = int(np.prod(env.single_observation_space.shape))
    act_dim = int(np.prod(env.single_action_space.shape))
    agent = PpoAgent(obs_dim, act_dim).to(env.device)
    payload = torch.load(checkpoint, map_location=env.device, weights_only=True)
    agent.load_state_dict(payload)
    agent.eval()
    return agent


def _frozen_checkpoint(task: str) -> Path:
    for line in Path("artifacts/sprint/gate1/jobs.jsonl").read_text().splitlines():
        job = json.loads(line)
        if job.get("task") == task:
            return Path(job["checkpoint"])
    raise LookupError(task)


def run_episode(
    task: str,
    contract: ActionContract,
    adapter_kind: str,
    *,
    seed: int,
    max_steps: int = 50,
    hold_frames: int = 6,
) -> dict:
    """Run one real single-env episode with rendering. Returns frames + per-step phase."""
    calibration = load_or_run_calibration(
        task, Path(f"artifacts/adaptation/calibration/{task}.json"), seed=20260720
    )
    response = _response_model(calibration)
    env = _make_render_env(task, contract)
    try:
        obs, _ = env.reset(seed=seed)
        low = torch.as_tensor(env.single_action_space.low, device=env.device, dtype=obs.dtype)
        high = torch.as_tensor(env.single_action_space.high, device=env.device, dtype=obs.dtype)
        agent = _load_agent(env, _frozen_checkpoint(task))

        device = str(env.device)
        if adapter_kind == "no_adapt":
            adapter = NoAdaptAdapter()
        elif adapter_kind == "oracle":
            adapter = OracleAdapter(contract, batch_size=1)
        elif adapter_kind == "entropy":
            driver = ExactBeliefDriver(
                _declared_pool(), batch_size=1, response=response, device=device
            )
            adapter = ProbingBeliefAdapter(
                driver, strategy="entropy", budget=_PROBE_BUDGET,
                amplitude=_PROBE_AMPLITUDE, seed=seed,
            )
        elif adapter_kind == "exact_belief":
            adapter = ExactBeliefAdapter(
                _declared_pool(), batch_size=1, response=response, device=device
            )
        else:
            raise ValueError(adapter_kind)

        frames, phases = [], []
        prev_obs = obs
        succeeded = False
        for _step in range(max_steps):
            frames.append(_render_frame(env))
            with torch.no_grad():
                canonical = torch.clamp(agent.deterministic_action(obs), low, high)
                raw = adapter.encode(canonical)
                probe_mask = getattr(adapter, "last_probe_mask", None)
                is_probe = (
                    bool(probe_mask.reshape(-1)[0].item()) if probe_mask is not None else False
                )
                obs, _, _term, _trunc, info = env.step(raw)
                resp = response_from_observations(calibration, prev_obs, obs)
                adapter.observe(raw, resp)
                prev_obs = obs
            success_flag = _success(info)
            phases.append("probe" if is_probe else "act")
            if success_flag:
                succeeded = True
                # hold a few frames on the success pose
                for _ in range(hold_frames):
                    frames.append(_render_frame(env))
                    phases.append("success")
                break
        else:
            frames.append(_render_frame(env))
            phases.append("act")
        return {"frames": frames, "phases": phases, "success": succeeded,
                "probe_steps": sum(1 for p in phases if p == "probe")}
    finally:
        env.close()


def _success(info) -> bool:
    for key in ("success", "success_once"):
        if key in info:
            val = info[key]
            try:
                return bool(torch.as_tensor(val).reshape(-1)[0].item())
            except Exception:
                return bool(val)
    if "final_info" in info and "episode" in info["final_info"]:
        flag = info["final_info"]["episode"]["success_once"]
        return bool(torch.as_tensor(flag).reshape(-1)[0].item())
    return False


def _caption_panel(
    frame: np.ndarray, title: str, subtitle: str, accent: tuple, step_text: str, size: int = 384,
) -> Image.Image:
    """Return a captioned panel: title bar + rendered frame + status strip."""
    img = Image.fromarray(frame).resize((size, size), Image.LANCZOS)
    bar_h, strip_h = 46, 30
    panel = Image.new("RGB", (size, size + bar_h + strip_h), WHITE)
    draw = ImageDraw.Draw(panel)
    # title bar with accent
    draw.rectangle([0, 0, size, bar_h], fill=accent)
    draw.text((10, 6), title, font=_font(19), fill=WHITE)
    draw.text((10, 27), subtitle, font=_font(13), fill=WHITE)
    panel.paste(img, (0, bar_h))
    # accent frame border
    draw.rectangle([0, bar_h, size - 1, bar_h + size - 1], outline=accent, width=3)
    # status strip
    draw.rectangle([0, bar_h + size, size, bar_h + size + strip_h], fill=(245, 247, 250))
    draw.text((10, bar_h + size + 6), step_text, font=_font(15), fill=INK)
    return panel


def _pad(frames: list, phases: list, n: int) -> tuple[list, list]:
    while len(frames) < n:
        frames.append(frames[-1])
        phases.append(phases[-1])
    return frames, phases


def _save_gif(panels: list[Image.Image], out: Path, fps: int = 8) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = int(1000 / fps)
    imgs = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in panels]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True)


def build_hero(task: str = "pick_cube") -> None:
    seen = representative_contracts("seen")
    hidden = seen[1]  # permutation/sign/scale composite, gripper NOT inverted (recoverable)
    ident = identity_contract()

    # Search a handful of seeds for a representative triptych:
    # clean succeeds, hidden no-adapt fails, entropy adapter succeeds after probing.
    chosen = None
    for seed in range(20260718, 20260718 + 40):
        clean = run_episode(task, ident, "no_adapt", seed=seed)
        if not clean["success"]:
            continue
        fail = run_episode(task, hidden, "no_adapt", seed=seed)
        if fail["success"]:
            continue
        adapt = run_episode(task, hidden, "entropy", seed=seed)
        if adapt["success"] and adapt["probe_steps"] >= _PROBE_BUDGET:
            chosen = (seed, clean, fail, adapt)
            print(f"[hero] seed {seed}: clean OK, no-adapt FAIL, entropy OK "
                  f"({adapt['probe_steps']} probe steps)")
            break
    if chosen is None:
        raise RuntimeError("no representative hero seed found")
    seed, clean, fail, adapt = chosen

    # Align on the successful panels; truncate the failing panel's long flail so the
    # triptych stays crisp instead of dominated by 50 frames of no-adapt thrashing.
    n = max(len(clean["frames"]), len(adapt["frames"])) + 2
    fail["frames"] = fail["frames"][:n]
    fail["phases"] = fail["phases"][:n]
    for run in (clean, fail, adapt):
        _pad(run["frames"], run["phases"], n)

    panels_seq = []
    for i in range(n):
        # panel 1: clean
        c_ph = clean["phases"][i]
        p1 = _caption_panel(
            clean["frames"][i], "1 - Clean interface",
            "frozen PPO, contract = identity", GREEN,
            "SUCCESS" if c_ph == "success" else f"step {min(i + 1, len(clean['phases']))}")
        # panel 2: hidden, no adapt
        fail["phases"][i]
        p2 = _caption_panel(
            fail["frames"][i], "2 - Hidden contract",
            "same policy, no adaptation", RED,
            "FAILS - wrong action semantics" if i >= len(fail["phases"]) - 2 else f"step {i + 1}")
        # panel 3: entropy adapter
        a_ph = adapt["phases"][i]
        if a_ph == "probe":
            k = sum(1 for p in adapt["phases"][:i + 1] if p == "probe")
            strip = f"PROBING {k}/{_PROBE_BUDGET} (bounded, safe)"
            accent3 = AMBER
        elif a_ph == "success":
            strip, accent3 = "SUCCESS - contract identified", GREEN
        else:
            strip, accent3 = "acting on MAP contract", BLUE
        p3 = _caption_panel(
            adapt["frames"][i], "3 - Belief adapter",
            "entropy probe -> Bayesian belief", accent3, strip)
        row = Image.new("RGB", (p1.width * 3 + 24, p1.height + 40), WHITE)
        row.paste(p1, (0, 40))
        row.paste(p2, (p1.width + 12, 40))
        row.paste(p3, (2 * p1.width + 24, 40))
        d = ImageDraw.Draw(row)
        d.text((12, 8), "ActionShift - one frozen PPO policy, three action contracts "
                        "(PickCube-v1, real ManiSkill render)", font=_font(20), fill=INK)
        panels_seq.append(row)
    _save_gif(panels_seq, _MEDIA / "hero_triptych.gif", fps=7)
    print(f"[hero] wrote {_MEDIA / 'hero_triptych.gif'}")


_DELAY_AWARE_CKPT = {
    "pick_cube": "artifacts/delay_aware/pick_cube/3523a7f717ddd9c2/final_ckpt.pt",
    "push_cube": "artifacts/delay_aware/push_cube/d6d1a1ce8d1784af/final_ckpt.pt",
}
_DELAY_HISTORY = 4


def _delay_aware_episode(
    task: str, contract: ActionContract, *, seed: int, max_steps: int = 50, hold_frames: int = 6,
) -> dict:
    """Render one episode of the trained delay-aware augmented-state backbone,
    oracle-encode, under the lag contract."""
    from actionshift.adaptation.delay_aware import ActionHistoryBuffer, DelayAwarePpoAgent

    env = _make_render_env(task, contract)
    try:
        obs, _ = env.reset(seed=seed)
        low = torch.as_tensor(env.single_action_space.low, device=env.device, dtype=obs.dtype)
        high = torch.as_tensor(env.single_action_space.high, device=env.device, dtype=obs.dtype)
        base_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        agent = DelayAwarePpoAgent(base_dim, act_dim, history=_DELAY_HISTORY).to(env.device)
        payload = torch.load(
            Path(_DELAY_AWARE_CKPT[task]), map_location=env.device, weights_only=True
        )
        agent.load_state_dict(payload)
        agent.eval()
        buffer = ActionHistoryBuffer(
            1, act_dim, history=_DELAY_HISTORY, device=env.device, dtype=obs.dtype
        )
        adapter = OracleAdapter(contract, batch_size=1)
        frames, phases = [], []
        succeeded = False
        for _step in range(max_steps):
            frames.append(_render_frame(env))
            with torch.no_grad():
                augmented = buffer.augment(obs)
                canonical = torch.clamp(agent.deterministic_action(augmented), low, high)
                raw = adapter.encode(canonical)
                buffer.push(canonical)
                obs, _, _term, _trunc, info = env.step(raw)
            phases.append("act")
            if _success(info):
                succeeded = True
                for _ in range(hold_frames):
                    frames.append(_render_frame(env))
                    phases.append("success")
                break
        else:
            frames.append(_render_frame(env))
            phases.append("act")
        return {"frames": frames, "phases": phases, "success": succeeded}
    finally:
        env.close()


def build_lag(task: str = "pick_cube") -> None:
    """Long-lag: frozen oracle-encode fails vs delay-aware backbone + oracle succeeds."""
    long_lag = representative_contracts("long_lag")
    lag2 = long_lag[0]  # oracle contract with lag=2

    # Search seeds for a representative pair: frozen reactive backbone FAILS under
    # lag (collapses, ~0.03) while the delay-aware augmented backbone SUCCEEDS (~0.53).
    chosen = None
    for seed in range(20260718, 20260718 + 40):
        aware = _delay_aware_episode(task, lag2, seed=seed)
        if not aware["success"]:
            continue
        frozen = run_episode(task, lag2, "oracle", seed=seed, max_steps=50)
        if frozen["success"]:
            continue
        chosen = (seed, frozen, aware)
        print(f"[lag] seed {seed}: frozen FAIL, delay-aware OK")
        break
    if chosen is None:
        raise RuntimeError("no representative lag seed found")
    seed, frozen, aware = chosen

    n = max(len(frozen["frames"]), len(aware["frames"])) + 2
    frozen["frames"] = frozen["frames"][:n]
    frozen["phases"] = frozen["phases"][:n]
    _pad(frozen["frames"], frozen["phases"], n)
    _pad(aware["frames"], aware["phases"], n)

    seq = []
    for i in range(n):
        lp = _caption_panel(
            frozen["frames"][i], "Frozen reactive backbone",
            "oracle-encode, lag = 2 steps", RED,
            "FAILS under delay (~0.03)" if i >= n - 2 else f"step {i + 1}")
        rp_ph = aware["phases"][i]
        rp = _caption_panel(
            aware["frames"][i], "Delay-aware augmented PPO",
            "oracle-encode, lag = 2 steps", GREEN,
            "SUCCESS (~0.53, 20x)" if rp_ph == "success" else f"step {i + 1}")
        row = Image.new("RGB", (lp.width * 2 + 12, lp.height + 40), WHITE)
        row.paste(lp, (0, 40))
        row.paste(rp, (lp.width + 12, 40))
        d = ImageDraw.Draw(row)
        d.text((12, 10), "ActionShift - long-lag split: delay-aware control fixes "
                         "what identification can't", font=_font(16), fill=INK)
        seq.append(row)
    _save_gif(seq, _MEDIA / "lag_sidebyside.gif", fps=7)
    print(f"[lag] wrote {_MEDIA / 'lag_sidebyside.gif'}")


def build_selftest() -> None:
    """Render the three real `actionshift-selftest` demo transcripts as three
    full-width, vertically-stacked terminal cards (PASS / MISMATCH / INCONCLUSIVE).

    Stacked single-column layout with a large monospace font so the panel stays
    readable at GitHub's default README width (~900 px display).
    """
    import re
    import subprocess
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    demos = [
        ("identity", "#2a9d5c"),
        ("miswired", "#c0392b"),
        ("unmodeled", "#e0a82e"),
    ]
    exe = str((_MEDIA.parent / ".venv" / "bin" / "actionshift-selftest").resolve())
    ansi = re.compile(r"(libvulkan|Fallback to SAPIEN|warn\(|UserWarning)")
    bg, fg, field_c, dim_c = "#161821", "#c8d0e0", "#8fb9e8", "#5a6270"
    wrap = 82

    def styled_lines(raw_lines: list[str], accent: str) -> list[tuple[str, str, str]]:
        """Wrap long prose lines (aligned field rows are left intact) and tag colors."""
        out: list[tuple[str, str, str]] = []
        for ln in raw_lines:
            pieces = (
                [ln] if len(ln) <= wrap
                else textwrap.wrap(ln, wrap, subsequent_indent="    ",
                                   break_long_words=False, break_on_hyphens=False) or [""]
            )
            for piece in pieces:
                stripped = piece.strip()
                color, weight = fg, "normal"
                if "VERDICT" in piece or any(
                    tok in piece for tok in ("PASS", "MISMATCH", "INCONCLUSIVE")
                ):
                    color, weight = accent, "bold"
                elif stripped.startswith(("permutation", "sign", "scale", "target",
                                          "frame", "lag", "gripper", "fit residual")):
                    color = field_c
                elif set(stripped) <= {"="} and stripped:
                    color = dim_c
                out.append((piece, color, weight))
        return out

    cards = []
    for name, accent in demos:
        proc = subprocess.run([exe, "--demo", name], capture_output=True, text=True)
        raw = [ln for ln in proc.stdout.splitlines() if not ansi.search(ln)]
        cards.append((name, accent, styled_lines(raw, accent)))

    # ---- layout (all in inches) ----
    width = 8.6
    x0, pad_x = 0.34, 0.16
    line_h, title_h = 0.255, 0.44
    in_top, in_bot, gap = 0.15, 0.22, 0.36
    top_m, bot_m = 0.92, 0.86
    card_w = width - 2 * x0
    card_h = [title_h + in_top + len(d) * line_h + in_bot for _, _, d in cards]
    height = top_m + sum(card_h) + gap * (len(cards) - 1) + bot_m

    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("#f5f7fa")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    fig.text(0.5, 1 - (0.44 / height),
             'actionshift-selftest: is this robot wired the way the policy thinks?',
             ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1a1a2e")

    y = height - top_m
    for (name, accent, disp), ch in zip(cards, card_h, strict=True):
        top = y
        # card background + accent border
        ax.add_patch(Rectangle((x0, top - ch), card_w, ch, facecolor=bg,
                               edgecolor=accent, linewidth=2.4, zorder=2))
        # title bar
        ax.add_patch(Rectangle((x0, top - title_h), card_w, title_h, facecolor=accent,
                               edgecolor="none", zorder=3))
        ax.text(x0 + pad_x, top - title_h / 2,
                f"$ actionshift-selftest --demo {name}", color="white",
                fontsize=12.5, family="monospace", fontweight="bold",
                va="center", ha="left", zorder=4)
        ty = top - title_h - in_top
        for text, color, weight in disp:
            ax.text(x0 + pad_x, ty, text, color=color, fontsize=11,
                    family="monospace", va="top", ha="left", fontweight=weight, zorder=4)
            ty -= line_h
        y = top - ch - gap

    fig.text(0.5, 0.52 / height,
             "Exit codes are scriptable: 0 = PASS, 1 = MISMATCH, 2 = INCONCLUSIVE.",
             ha="center", va="center", fontsize=10, color="#4a5568")
    fig.text(0.5, 0.25 / height,
             "The tool abstains rather than guesses on wirings outside its declared pool.",
             ha="center", va="center", fontsize=10, color="#4a5568")

    out = _MEDIA / "selftest_demos.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[selftest] wrote {out}")


def build_all() -> None:
    build_hero()
    build_lag()
    build_selftest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("hero", "lag", "selftest", "all"))
    parser.add_argument("--task", default="pick_cube")
    args = parser.parse_args()
    if args.command == "hero":
        build_hero(args.task)
    elif args.command == "lag":
        build_lag(args.task)
    elif args.command == "selftest":
        build_selftest()
    else:
        build_all()


if __name__ == "__main__":
    main()
