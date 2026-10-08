"""Reads the real flotation plant data and turns it into one row per hour.

Dataset: *Quality Prediction in a Mining Process* (Kaggle, ``edumagalhaes``, CC0-1.0): a
reverse cationic flotation plant for **iron ore**, March to September 2017, 737,453 rows.
Process variables are sampled every 20 seconds; the feed and concentrate assays come from
the lab. Three things in the file shape every result, all handled here:

- **The timestamps only carry the hour.** The 180 readings of an hour share one timestamp,
  so the honest unit of analysis is the hour (means and standard deviations of the
  20-second readings).
- **The silica in the concentrate is interpolated in places.** In 310 hours it changes
  within the hour, and 232 hours lie on perfectly straight segments between two values;
  together, 328 hours are not lab results and are never used as a target (``measured``
  is False).
- **The feed assay is not hourly.** It changes in 7.5% of the hours and once stays the
  same for 792 hours (33 days), so it enters as "last assay" plus how old that assay is.

There is also one gap of 319 hours (13 days) in March 2017; lags never cross it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CSV_NAME = "MiningProcess_Flotation_Plant_Database.csv"
KAGGLE_SLUG = "edumagalhaes/quality-prediction-in-a-mining-process"

FEED = ["% Iron Feed", "% Silica Feed"]
CONTROLS = ["Starch Flow", "Amina Flow", "Ore Pulp pH", "Ore Pulp Density"]
PULP_FLOW = "Ore Pulp Flow"
AIR = [f"Flotation Column 0{i} Air Flow" for i in range(1, 8)]
LEVEL = [f"Flotation Column 0{i} Level" for i in range(1, 8)]
PROCESS = CONTROLS + [PULP_FLOW] + AIR + LEVEL
IRON_CONC = "% Iron Concentrate"  # the same lab sample as the target: never a feature
TARGET = "% Silica Concentrate"
LINEAR_TOL = 1e-6


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Downloads the dataset with the Kaggle API, using the credentials of the Kaggle CLI
    configured in the user's home folder (never in this repository)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    raw_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(KAGGLE_SLUG, path=str(raw_dir), unzip=True, quiet=False)
    path = raw_dir / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(f"the Kaggle download did not produce {path}")
    return path


def read_raw(path: Path) -> pl.DataFrame:
    """The CSV as published: decimals with a comma, timestamps to the hour."""
    df = pl.read_csv(path, infer_schema=False)
    numeric = [c for c in df.columns if c != "date"]
    return df.with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        *[pl.col(c).str.replace(",", ".", literal=True).cast(pl.Float64) for c in numeric],
    )


def linear_segments(values: np.ndarray, tol: float = LINEAR_TOL) -> np.ndarray:
    """True for points in the middle of a straight run of three or more values that is not
    flat: what linear interpolation between two lab results looks like."""
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v), dtype=bool)
    if len(v) < 3:
        return out
    second = np.abs(np.diff(v, 2)) < tol
    moving = np.abs(np.diff(v)) > tol
    out[1:-1] = second & moving[1:]
    return out


def hourly(raw: pl.DataFrame) -> pl.DataFrame:
    """One row per hour: means of every variable, standard deviations of the process ones,
    the ``measured`` flag for the target, the feed assay's age and a ``segment`` id that
    changes after every gap in the hours."""
    h = (
        raw.group_by("date")
        .agg(
            [pl.len().alias("readings")]
            + [pl.col(c).mean() for c in FEED + PROCESS + [IRON_CONC, TARGET]]
            + [pl.col(c).std().alias(f"{c} sd") for c in PROCESS]
            + [pl.col(TARGET).n_unique().alias("_target_values")]
        )
        .sort("date")
    )
    target = h[TARGET].to_numpy()
    measured = (h["_target_values"].to_numpy() == 1) & ~linear_segments(target)
    gap = h["date"].diff().dt.total_hours().fill_null(1)
    changed = (pl.col("% Iron Feed").diff().abs() > LINEAR_TOL) | (pl.col("% Silica Feed").diff().abs() > LINEAR_TOL)
    h = h.with_columns(
        pl.Series("measured", measured),
        (gap > 1).cum_sum().alias("segment"),
        changed.fill_null(True).alias("_feed_changed"),
    ).drop("_target_values")
    # hours since the feed assay last changed, restarting after a gap
    feed_run = (pl.col("_feed_changed") | (pl.col("segment") != pl.col("segment").shift(1))).fill_null(True).cum_sum()
    return h.with_columns(feed_run.alias("_feed_run")).with_columns(
        pl.int_range(pl.len()).over("_feed_run").alias("feed_assay_age_h")
    ).drop(["_feed_changed", "_feed_run"])
