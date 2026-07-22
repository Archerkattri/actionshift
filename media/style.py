"""Shared plotting style for ActionShift + ActionABI figures.

Both projects import (or copy) this module so every figure reads as one
visual system: same palette, same typography, same axis treatment.
"""

import matplotlib as mpl

# Canonical palette. Keyed by role so figures stay consistent across projects.
COLORS = {
    "oracle": "#2a9d5c",   # privileged ceiling / verified-correct path (green)
    "pool": "#2b6cb0",     # pool belief (blue)
    "grammar": "#2c7a7b",  # grammar belief (teal)
    "learned": "#9aa0a6",  # passive learned identifier (gray)
    "floor": "#c0392b",    # no-adaptation floor / refusal / contradiction (red)
    "dualabi": "#6b4d9e",  # DualABI adapter (purple)
    "entropy": "#dd7a2f",  # entropy probes (orange)
    "fixed": "#4a5568",    # fixed schedule (slate)
    "random": "#a0aec0",   # random baseline (light gray)
    "probe": "#e0a82e",    # probe phase / flagged (amber)
    "ink": "#1a1a2e",      # dark ink text
    "grid": "#e2e8f0",     # light grid
}


def apply_style():
    """Apply consistent rcParams for a clean, publication-grade look."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.titlecolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.9,
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "legend.frameon": False,
    })


def style_axis(ax, grid_axis="y"):
    """Common per-axis cleanup: subtle grid on one axis only."""
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.9)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax
