"""Generate the two composite paper figures for ActionShift.

  fig_task_panel.png        -- the four frozen-backbone ManiSkill tasks, each shown
                               at the MOMENT OF TASK EXECUTION (grasp/push/pull/stack).
  fig_collapse_filmstrip.png -- collapse vs. restoration filmstrip (2 rows x 4 frames),
                               one frozen PPO policy on PickCube-v1 under a hidden contract.

Frame provenance (no simulator required; sanctioned crop-from-media path):
  * Task panel frames are extracted from the official ManiSkill3 environment-demo mp4s
    (third_party/maniskill/figures/environment_demos/<Env>-v1_rt.mp4), each at a hand-
    picked frame index where the manipulation action is unmistakable.
  * Filmstrip frames are cropped from repo/media/hero_triptych.gif (itself a real
    offscreen SAPIEN render of the frozen Gate-1 PPO backbone, regenerable via
    media/make_media.py hero).

Outputs are written to BOTH paper/figures/ and repo/media/ so the LaTeX source and the
repository README stay in sync.

Usage:
  python media/make_paper_figs.py            # build both figures
  python media/make_paper_figs.py panel      # task panel only
  python media/make_paper_figs.py filmstrip  # filmstrip only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------- palette
GREEN = (42, 157, 92)
BLUE = (43, 108, 176)
AMBER = (224, 168, 46)
RED = (192, 57, 43)
INK = (26, 26, 40)
WHITE = (255, 255, 255)
LIGHT = (244, 244, 246)

_HERE = Path(__file__).resolve().parent          # work/media (or repo/media)
_ROOT = _HERE.parent                             # work/ (or repo/)
_PROJECT = _ROOT.parent                          # projects/actionshift


def _find(rel: str) -> Path:
    """Locate an asset relative to this file, the project's work/, or repo/."""
    for base in (_ROOT, _PROJECT / "work", _PROJECT / "repo"):
        cand = base / rel
        if cand.exists():
            return cand
    return _ROOT / rel


_DEMOS = _find("third_party/maniskill/figures/environment_demos")
_TRIPTYCH = _find("media/hero_triptych.gif")

# where the compiled paper and the repo expect the figures
_OUT_DIRS = [
    _PROJECT / "paper" / "figures",              # projects/actionshift/paper/figures
    _PROJECT / "repo" / "media",                 # projects/actionshift/repo/media
]


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in ("/usr/share/fonts/truetype/dejavu/",):
        p = Path(base) / name
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _fit_font(draw, text, max_size, width_budget, bold=True, min_size=16):
    """Largest font <= max_size whose rendering of ``text`` fits ``width_budget`` px."""
    size = max_size
    while size > min_size:
        f = _font(size, bold=bold)
        left, _, right, _ = draw.textbbox((0, 0), text, font=f)
        if right - left <= width_budget:
            return f
        size -= 1
    return _font(min_size, bold=bold)


def _save(img: Image.Image, fname: str) -> None:
    for d in _OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / fname)
        print(f"wrote {d / fname}  ({img.size[0]}x{img.size[1]})")


def _extract_frame(video: Path, n: int) -> Image.Image:
    """Return frame index ``n`` of ``video`` as a PIL image (via ffmpeg)."""
    tmp = _HERE / f".__frame_{video.stem}_{n}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"select='eq(n\\,{n})'", "-vframes", "1", str(tmp)],
        check=True,
    )
    img = Image.open(tmp).convert("RGB")
    tmp.unlink(missing_ok=True)
    return img


def _text_center(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    draw.text((x0 + (x1 - x0 - tw) / 2 - left, y0 + (y1 - y0 - th) / 2 - top),
              text, font=font, fill=fill)


# ============================================================================ TASK PANEL
# Each entry: (env-id, demo mp4 stem, frame index @ action moment, header colour, caption)
# Frame indices chosen by scrubbing the official ManiSkill3 demo mp4s and reading the
# frames back: the gripper/arm is caught mid-manipulation, not hovering.
_TASKS = [
    ("PickCube-v1", "PickCube-v1_rt", 68, GREEN,
     "grasp cube, lift to the 3-D goal (green)"),
    ("PushCube-v1", "PushCube-v1_rt", 62, BLUE,
     "push cube onto the planar goal (target)"),
    ("PullCube-v1", "PullCube-v1_rt", 56, AMBER,
     "pull cube toward the planar goal (target)"),
    ("StackCube-v1", "StackCube-v1_rt", 96, RED,
     "place red cube onto the green cube"),
]


def build_task_panel() -> None:
    S = 336            # frame side
    bar_h = 40         # coloured header height
    cap_h = 46         # caption strip height
    gap = 12
    margin = 10
    n = len(_TASKS)
    W = margin * 2 + n * S + (n - 1) * gap
    H = bar_h + S + cap_h

    panel = Image.new("RGB", (W, H), WHITE)
    dr = ImageDraw.Draw(panel)
    f_title = _font(23)
    f_cap = _font(16, bold=False)

    for i, (env, stem, idx, colour, caption) in enumerate(_TASKS):
        x = margin + i * (S + gap)
        frame = _extract_frame(_DEMOS / f"{stem}.mp4", idx).resize((S, S), Image.LANCZOS)
        # header bar
        dr.rectangle([x, 0, x + S, bar_h], fill=colour)
        dr.text((x + 10, bar_h / 2 - 13), env, font=f_title, fill=WHITE)
        # frame + coloured border
        panel.paste(frame, (x, bar_h))
        dr.rectangle([x, bar_h, x + S - 1, bar_h + S - 1], outline=colour, width=3)
        # caption strip
        dr.text((x + 4, bar_h + S + 12), caption, font=f_cap, fill=INK)

    _save(panel, "fig_task_panel.png")


# ============================================================================ FILMSTRIP
# Panel crop windows inside a hero_triptych.gif frame (1176x500).
_P2 = (412, 98, 778, 468)   # middle panel  = hidden contract, NO adaptation
_P3 = (812, 98, 1168, 468)  # right panel   = belief adapter, RESTORED

# (gif frame index [1-based], panel, phase colour, per-frame caption)
_TOP = [   # no adaptation -> collapse
    (1, _P2, RED, "step 1"),
    (9, _P2, RED, "step 9 - drifting off"),
    (17, _P2, RED, "step 17"),
    (22, _P2, RED, "FAILS - wrong semantics"),
]
_BOT = [   # + belief adapter -> restored
    (5, _P3, AMBER, "probe 5/6 (bounded, safe)"),
    (8, _P3, BLUE, "acting on MAP contract"),
    (15, _P3, BLUE, "acting on MAP contract"),
    (22, _P3, GREEN, "SUCCESS - identified"),
]


def _gif_frame(n: int) -> Image.Image:
    """1-based frame ``n`` of the triptych gif."""
    tmp = _HERE / f".__gif_{n}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(_TRIPTYCH),
         "-vf", f"select='eq(n\\,{n - 1})'", "-vframes", "1", str(tmp)],
        check=True,
    )
    img = Image.open(tmp).convert("RGB")
    tmp.unlink(missing_ok=True)
    return img


def build_filmstrip() -> None:
    F = 462                 # frame side (large)
    gap = 16
    margin = 26
    cap_h = 56              # per-frame caption bar
    hdr_h = 60             # row header band
    title_h = 78
    n = 4
    W = margin * 2 + n * F + (n - 1) * gap
    H = title_h + 2 * (hdr_h + F + cap_h) + margin

    strip = Image.new("RGB", (W, H), WHITE)
    dr = ImageDraw.Draw(strip)

    title = ("Collapse vs. restoration: one frozen PPO policy on PickCube-v1 "
             "(seen split), real ManiSkill render")
    f_title = _fit_font(dr, title, 48, W - 40)
    _text_center(dr, (0, 6, W, title_h - 6), title, f_title, INK)

    rows = [
        (_TOP, RED,
         "TOP  -  no adaptation: policy acts on wrong action semantics -> 0.00 success"),
        (_BOT, GREEN,
         "BOTTOM  -  + belief adapter: bounded probe -> Bayesian belief -> act on MAP -> success"),
    ]

    y = title_h
    for frames, band, header in rows:
        # row header band (full width, large text)
        dr.rectangle([margin, y, W - margin, y + hdr_h], fill=band)
        f_hdr = _fit_font(dr, header, 40, W - 2 * margin - 32)
        _, t, _, b = dr.textbbox((0, 0), header, font=f_hdr)
        dr.text((margin + 16, y + (hdr_h - (b - t)) / 2 - t), header, font=f_hdr, fill=WHITE)
        y += hdr_h
        for i, (gi, box, colour, caption) in enumerate(frames):
            x = margin + i * (F + gap)
            crop = _gif_frame(gi).crop(box).resize((F, F), Image.LANCZOS)
            strip.paste(crop, (x, y))
            dr.rectangle([x, y, x + F - 1, y + F - 1], outline=colour, width=5)
            # caption bar in the phase colour, text auto-fit to the frame width
            dr.rectangle([x, y + F, x + F, y + F + cap_h], fill=colour)
            f_cap = _fit_font(dr, caption, 34, F - 20)
            _text_center(dr, (x, y + F, x + F, y + F + cap_h), caption, f_cap, WHITE)
        y += F + cap_h

    _save(strip, "fig_collapse_filmstrip.png")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "panel"):
        build_task_panel()
    if what in ("all", "filmstrip"):
        build_filmstrip()


if __name__ == "__main__":
    main()
