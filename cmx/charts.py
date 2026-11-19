"""One look for every chart: light background, thin grey grid, the same colours in the same order."""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
# A colour-blind-checked palette. Series always take colours in this order, never cycled.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_chart(width, height, title):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    style(ax, title)
    return fig, ax


def new_pair(width, height, left_title, right_title):
    """Two charts side by side (never two scales on one chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(width, height), facecolor=SURFACE)
    style(axes[0], left_title)
    style(axes[1], right_title)
    return fig, axes[0], axes[1]


def save_chart(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
