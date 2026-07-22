"""Regenerate ActionShift figures 1-5.

All numbers are hardcoded report-derived summary statistics (final).
Run: python media/make_plots.py   (from the actionshift/ dir, or anywhere)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from style import COLORS, apply_style, style_axis

HERE = os.path.dirname(os.path.abspath(__file__))


def _save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# FIG 1 — adaptation ladder
# ---------------------------------------------------------------------------
def fig1_method_ladder():
    apply_style()
    rows = [
        ("No adaptation\n(floor)", 0.000, 0.000, 0.006, "floor"),
        ("Learned identifier\n(passive UP-OSI)", 0.000, 0.000, 0.006, "learned"),
        ("Grammar belief\n(factorized + probes)", 0.458, 0.419, 0.498, "grammar"),
        ("Pool belief\n(entropy probes)", 0.987, 0.974, 0.993, "pool"),
        ("Privileged oracle\n(ceiling)", 1.000, 0.994, 1.000, "oracle"),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    lows = np.array([r[2] for r in rows])
    highs = np.array([r[3] for r in rows])
    colors = [COLORS[r[4]] for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(rows))
    yerr = np.vstack([vals - lows, highs - vals])
    ax.bar(x, vals, width=0.62, color=colors, edgecolor=COLORS["ink"],
                  linewidth=0.8, zorder=3)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor=COLORS["ink"],
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)

    for xi, v, hi in zip(x, vals, highs, strict=False):
        ax.annotate(f"{v:.3f}", (xi, hi + 0.018), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=COLORS["ink"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Task success (100 ep/contract, 3 seeds)")
    ax.set_title("The adaptation ladder (PickCube, seen split)", pad=26)
    ax.text(0.5, 1.045,
            "Same frozen PPO backbone throughout; only the adapter's privilege changes. "
            "n=600 episodes/cell (Wilson 95%).",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="#4a5568")
    style_axis(ax)
    _save(fig, "plot_method_ladder.png")


# ---------------------------------------------------------------------------
# FIG 2 — probe-cost Pareto
# ---------------------------------------------------------------------------
def fig2_probe_pareto():
    apply_style()
    data = {
        "dualabi": ("DualABI", "o", COLORS["dualabi"],
                    [(2.78, 0.983), (2.60, 1.000), (2.91, 0.968), (2.94, 0.945)]),
        "entropy": ("Entropy (champion)", "s", COLORS["entropy"],
                    [(6.00, 0.987), (6.00, 1.000), (6.00, 0.975), (6.00, 0.910)]),
        "fixed": ("Fixed schedule", "^", COLORS["fixed"],
                  [(6.00, 0.928), (6.00, 0.982), (6.00, 0.970), (6.00, 0.945)]),
        "random": ("Random", "x", COLORS["random"],
                   [(6.00, 0.925), (6.00, 0.942), (6.00, 0.943), (6.00, 0.925)]),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    # jitter x slightly for the overlapping x=6.00 clusters so markers are visible
    rng = np.random.default_rng(7)
    for key, (name, marker, color, pts) in data.items():
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts])
        if key != "dualabi":
            xs = xs + rng.uniform(-0.08, 0.08, size=xs.shape)
        ax.scatter(xs, ys, marker=marker, s=95, c=color,
                   edgecolors=COLORS["ink"] if marker != "x" else color,
                   linewidths=1.1 if marker != "x" else 2.0,
                   label=name, zorder=3, alpha=0.95)

    # Pareto-preferred arrow (up-left)
    ax.annotate("", xy=(1.9, 1.001), xytext=(4.6, 0.955),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["oracle"],
                                lw=2.2, alpha=0.65))
    ax.text(1.85, 1.003, "Pareto-preferred", color=COLORS["oracle"],
            fontsize=10, fontweight="bold", ha="left", va="bottom")

    ax.set_xlim(0, 6.3)
    ax.set_ylim(0.90, 1.005)
    ax.set_xlabel("Mean probe steps per episode (lower = cheaper)")
    ax.set_ylabel("Task success")
    ax.set_title("Probe-cost Pareto: same success, ~half the probing", pad=14)
    ax.legend(loc="lower left", ncol=1, handletextpad=0.4)
    ax.text(0.0, -0.19,
            "DualABI matches the entropy champion's success (all paired 95% intervals include 0) "
            "while spending ~52% fewer probe\nsteps and ~42% less displacement, replicated across "
            "4 tasks.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#4a5568")
    style_axis(ax, grid_axis="both")
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.9)
    _save(fig, "plot_probe_pareto.png")


# ---------------------------------------------------------------------------
# FIG 3 — long-lag split solved by delay-aware control
# ---------------------------------------------------------------------------
def fig3_lag_solve():
    apply_style()
    groups = ["PickCube", "PushCube"]
    frozen = [0.027, 0.153]
    delay = [0.528, 0.415]
    delay_lo = [0.488, 0.376]
    delay_hi = [0.568, 0.455]
    mult = ["~20x", "~2.7x"]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    x = np.arange(len(groups))
    w = 0.36
    ax.bar(x - w / 2, frozen, w, color=COLORS["floor"],
                edgecolor=COLORS["ink"], linewidth=0.8, zorder=3,
                label="Frozen reactive backbone (oracle-encode)")
    ax.bar(x + w / 2, delay, w, color=COLORS["oracle"],
                edgecolor=COLORS["ink"], linewidth=0.8, zorder=3,
                label="Delay-aware augmented-state PPO (oracle-encode)")
    yerr = np.vstack([np.array(delay) - np.array(delay_lo),
                      np.array(delay_hi) - np.array(delay)])
    ax.errorbar(x + w / 2, delay, yerr=yerr, fmt="none", ecolor=COLORS["ink"],
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)

    for xi, v in zip(x - w / 2, frozen, strict=False):
        ax.annotate(f"{v:.3f}", (xi, v + 0.012), ha="center", va="bottom",
                    fontsize=10, color=COLORS["ink"])
    # Value label centered in the bar body, well clear of the error-bar whisker.
    for xi, v, hi, m in zip(x + w / 2, delay, delay_hi, mult, strict=False):
        ax.annotate(f"{v:.3f}", (xi, v * 0.5), ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        ax.annotate(m, (xi, hi + 0.02), ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=COLORS["oracle"])

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("Task success on long_lag split")
    ax.set_title("The long-lag split: solved by delay-aware control, not identification", pad=14)
    ax.legend(loc="upper right")
    ax.text(0.0, -0.16,
            "Identification alone cannot fix delay; delay-aware control alone cannot fix hidden "
            "semantics; the combination solves\nboth. New backbone, not the frozen tournament "
            "backbone (n=600, 3 seeds).",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#4a5568")
    style_axis(ax)
    _save(fig, "plot_lag_solve.png")


# ---------------------------------------------------------------------------
# FIG 4 — imitation rescue
# ---------------------------------------------------------------------------
def fig4_imitation_rescue():
    apply_style()
    conds = ["Clean interface", "Hidden contract\n(no adaptation)",
             "Hidden contract\n+ belief adapter (rescued)"]
    dp = [0.580, 0.003, 0.620]
    ppo = [1.000, 0.000, 0.929]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    x = np.arange(len(conds))
    w = 0.36
    ax.bar(x - w / 2, dp, w, color=COLORS["grammar"], edgecolor=COLORS["ink"],
           linewidth=0.8, zorder=3, label="Diffusion Policy (imitation)")
    ax.bar(x + w / 2, ppo, w, color=COLORS["pool"], edgecolor=COLORS["ink"],
           linewidth=0.8, zorder=3, label="PPO (RL)")

    for xi, v in zip(x - w / 2, dp, strict=False):
        ax.annotate(f"{v:.3f}", (xi, v + 0.014), ha="center", va="bottom",
                    fontsize=10, color=COLORS["ink"])
    for xi, v in zip(x + w / 2, ppo, strict=False):
        ax.annotate(f"{v:.3f}", (xi, v + 0.014), ha="center", va="bottom",
                    fontsize=10, color=COLORS["ink"])

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("PickCube task success")
    ax.set_title("Brittleness is a property of the interface, not the learning paradigm", pad=14)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0.90))
    ax.text(0.0, -0.17,
            "A competent Diffusion Policy collapses ~0.58 -> ~0.003 under a hidden contract, the "
            "same 0.00 floor as PPO; the same\npolicy-agnostic belief adapters restore both. "
            "(Rescued = exact_belief/oracle ceiling.)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#4a5568")
    style_axis(ax)
    _save(fig, "plot_imitation_rescue.png")


# ---------------------------------------------------------------------------
# FIG 5 — architecture diagram
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, facecolor, edgecolor, textcolor=None, fontsize=10.5,
         fontweight="bold"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=textcolor or COLORS["ink"], zorder=4, wrap=True)
    return (x, y, w, h)


def _arrow(ax, p0, p1, color, label=None, lw=2.0, rad=0.0, label_dy=0.12,
           label_color=None, ls="-"):
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                          color=color, lw=lw, linestyle=ls,
                          connectionstyle=f"arc3,rad={rad}", zorder=2)
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.6,
                style="italic", color=label_color or "#4a5568", zorder=4)


def fig5_architecture():
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    h = 1.45
    y = 1.7
    w = 2.35
    gap = 1.05
    xs = [0.2]
    for _ in range(3):
        xs.append(xs[-1] + w + gap)

    _box(ax, (xs[0], y), w, h, "Frozen policy\n(PPO / Diffusion\nPolicy)",
         "#eef2f9", COLORS["pool"], fontsize=9.2)
    _box(ax, (xs[1], y), w, h, "Belief adapter\n(exact / entropy /\nDualABI)",
         "#f3eefa", COLORS["dualabi"], fontsize=9.2)
    _box(ax, (xs[2], y), w, h,
         "Hidden contract\nwrapper\n(perm - sign - scale\n- target - frame\n- lag - gripper)",
         "#fdeeee", COLORS["floor"], fontsize=8.6)
    _box(ax, (xs[3], y), w, h, "ManiSkill env\n(PickCube ...)",
         "#eaf5ef", COLORS["oracle"])

    yc = y + h / 2
    _arrow(ax, (xs[0] + w, yc), (xs[1], yc), COLORS["ink"], "canonical\naction", label_dy=0.1)
    _arrow(ax, (xs[1] + w, yc), (xs[2], yc), COLORS["ink"], "raw\naction", label_dy=0.1)
    _arrow(ax, (xs[2] + w, yc), (xs[3], yc), COLORS["ink"], "executed", label_dy=0.1)

    # feedback arrow env -> adapter (curved below)
    _arrow(ax, (xs[3] + w / 2, y), (xs[1] + w / 2, y), COLORS["oracle"],
           None, lw=1.8, rad=-0.35)
    ax.text((xs[3] + w / 2 + xs[1] + w / 2) / 2, y - 1.3,
            "calibrated response", ha="center", va="top", fontsize=9,
            style="italic", color=COLORS["oracle"],
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0), zorder=6)

    # PROBE PHASE band around adapter
    pad = 0.22
    probe = FancyBboxPatch((xs[1] - pad, y - pad), w + 2 * pad, h + 2 * pad,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.8, edgecolor=COLORS["probe"],
                           facecolor="none", linestyle=(0, (5, 3)), zorder=5)
    ax.add_patch(probe)
    ax.text(xs[1] + w / 2, y + h + pad + 0.16, "PROBE PHASE (bounded, safe)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color=COLORS["probe"])
    ax.text(xs[1] + w / 2, y - pad - 0.16,
            "6 bounded pulses -> Bayesian belief -> MAP contract, then act",
            ha="center", va="top", fontsize=8.4, style="italic", color="#8a6d1f")

    ax.set_title("ActionShift: policy-agnostic adaptation to a hidden action contract",
                 pad=16, y=1.02)
    _save(fig, "diagram_architecture.png")


def main():
    fig1_method_ladder()
    fig2_probe_pareto()
    fig3_lag_solve()
    fig4_imitation_rescue()
    fig5_architecture()


if __name__ == "__main__":
    main()
