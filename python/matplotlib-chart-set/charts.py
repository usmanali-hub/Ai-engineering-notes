"""A small, consistent chart set with Matplotlib: one style, honest axes, no chart junk."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PALETTE = {"primary": "#1f6feb", "muted": "#8b949e", "accent": "#d1242f"}


def _clean_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def line_chart(x: np.ndarray, series: dict[str, np.ndarray], title: str, ylabel: str) -> Figure:
    """One line per series, the first in the primary colour and the rest muted, labelled directly."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colours = [PALETTE["primary"]] + [PALETTE["muted"]] * (len(series) - 1)
    for (name, y), colour in zip(series.items(), colours):
        ax.plot(x, y, color=colour, linewidth=2 if colour == PALETTE["primary"] else 1.2)
        ax.annotate(name, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=colour)
    ax.set(title=title, ylabel=ylabel)
    ax.margins(x=0.12)  # room for the direct labels
    ax.set_xticks(x)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def bar_chart(labels: list[str], values: list[float], title: str, xlabel: str) -> Figure:
    """Horizontal bars sorted by value; bars always start at zero so lengths are honest."""
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(labels) + 1.2))
    ax.barh(np.array(labels)[order], np.array(values)[order], color=PALETTE["primary"])
    ax.set(title=title, xlabel=xlabel, xlim=(0, max(values) * 1.1))
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def histogram(values: np.ndarray, title: str, xlabel: str, bins: int = 30) -> Figure:
    """Histogram with the median marked, which is more stable than the mean for skewed data."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(values, bins=bins, color=PALETTE["primary"], alpha=0.85)
    median = float(np.median(values))
    ax.axvline(median, color=PALETTE["accent"], linestyle="--")
    ax.annotate(f"median {median:.1f}", (median, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", color=PALETTE["accent"])
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def demo(out_dir: str = ".") -> list[str]:
    rng = np.random.default_rng(0)
    x = np.arange(12)
    files = []
    fig = line_chart(x, {"2026": np.cumsum(rng.normal(5, 2, 12)), "2025": np.cumsum(rng.normal(4, 2, 12))}, "Monthly sales", "units")
    fig.savefig(f"{out_dir}/line.png", dpi=110)
    files.append("line.png")
    fig = bar_chart(["Kuwait", "UAE", "Pakistan", "Oman"], [46.1, 33.6, 30.8, 12.4], "Revenue by country", "revenue (thousands)")
    fig.savefig(f"{out_dir}/bar.png", dpi=110)
    files.append("bar.png")
    fig = histogram(rng.lognormal(3, 0.6, 1000), "Order value distribution", "order value")
    fig.savefig(f"{out_dir}/hist.png", dpi=110)
    files.append("hist.png")
    plt.close("all")
    return files


if __name__ == "__main__":
    print(demo())
