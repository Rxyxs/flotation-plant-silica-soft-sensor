"""Figures for the README, written to ``outputs/figures/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.data import CONTROLS, TARGET  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
BLUE, ORANGE, GREY, RED, GREEN = "#2563eb", "#d97706", "#9ca3af", "#b91c1c", "#16a34a"


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_silica_series(hours: pl.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4))
    m = hours.filter(pl.col("measured"))
    i = hours.filter(~pl.col("measured"))
    ax.plot(m["date"].to_numpy(), m[TARGET].to_numpy(), ".", ms=2, color=BLUE, label="lab result")
    ax.plot(i["date"].to_numpy(), i[TARGET].to_numpy(), ".", ms=3, color=RED, label="interpolated, not used as a target")
    ax.set_ylabel("silica in the concentrate (%)")
    ax.set_title("Hourly silica in the iron concentrate, March-September 2017 (one 13-day gap in March)", fontsize=10)
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(alpha=0.25)
    return _save(fig, "silica_series.png")


def plot_protocols(r2: dict[str, float]) -> Path:
    labels = list(r2)
    values = [r2[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = [GREY, GREY, GREY, RED]
    bars = ax.barh(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("R² of the same XGBoost model (below 0 = worse than predicting the average)")
    ax.set_title("Where a high R² comes from: the validation protocol, not the model", fontsize=10)
    ax.set_xlim(min(values) - 0.3, 1.1)
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, "validation_protocols.png")


def plot_walk_forward(summary: dict[str, dict]) -> Path:
    items = sorted(summary.items(), key=lambda kv: kv[1]["rmse"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for row, (name, s) in enumerate(items):
        color = BLUE if "lab" in name and "Persistence" not in name else (ORANGE if "Persistence" in name else GREY)
        if name == "Training mean":
            color = "#111827"
        ax.scatter(s["rmse_per_fold"], [row] * len(s["rmse_per_fold"]), color=color, alpha=0.35, s=18)
        ax.scatter([s["rmse"]], [row], color=color, marker="D", s=60, zorder=3)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, silica percentage points (dots = folds, diamond = all folds pooled)")
    ax.set_title("Walk-forward, 5 folds, 24-hour gap: only models fed recent lab results beat persistence", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, "walk_forward.png")


def plot_recommendations(mean_change: dict[str, list[float]], agreement: dict[str, float]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    n_models = len(next(iter(mean_change.values())))
    width = 0.8 / n_models
    x = np.arange(len(CONTROLS))
    palette = [BLUE, ORANGE, GREEN, "#7c3aed"]
    for i in range(n_models):
        ax.bar(x + i * width - 0.4 + width / 2, [mean_change[c][i] for c in CONTROLS], width,
               color=palette[i % len(palette)], label=f"model trained on the first {25 * (i + 1)}% of the history")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nsame direction in {agreement[c]:.0%} of hours" for c in CONTROLS], fontsize=8)
    ax.set_ylabel("average recommended change (%)")
    ax.set_title("Same 40 high-silica hours, four models: they disagree on which way to move pH and density", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, "recommendations.png")


def plot_operator_reaction(df: pl.DataFrame, lag: int = 2) -> Path:
    d = df.with_columns(pl.col(TARGET).shift(lag).over("segment").alias("before")).drop_nulls("before")
    edges = np.quantile(d["before"].to_numpy(), np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(d["before"].to_numpy(), edges[1:-1]), 0, 9)
    centers = [d["before"].to_numpy()[bins == b].mean() for b in range(10)]
    amine = [d["Amina Flow"].to_numpy()[bins == b].mean() for b in range(10)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centers, amine, marker="o", color=BLUE)
    ax.set_xlabel(f"silica in the concentrate {lag} hours earlier (%), deciles")
    ax.set_ylabel("amine flow now (mean)")
    ax.set_title("The amine dose follows the silica that came before it", fontsize=10)
    ax.grid(alpha=0.25)
    return _save(fig, "operator_reaction.png")
