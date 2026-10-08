"""The two feature sets the models are compared on.

- ``process``: what the plant knows in real time in the hour being predicted (feed assay and
  its age, every process variable's mean and spread, hour of day). This is a pure soft
  sensor: no lab result of the concentrate at all.
- ``process + lab``: the same plus the last *measured* silica known ``LAB_DELAY`` or more
  hours earlier. The lab takes time to return a result; two hours is an assumption, stated
  because it matters: with a one-hour delay persistence would be much harder to beat.

The iron in the concentrate is never a feature: it comes from the same lab sample as the
silica (correlation -0.80) and is not known before it.
"""

from __future__ import annotations

import polars as pl

from src.data import CONTROLS, FEED, PROCESS, PULP_FLOW, TARGET

LAB_DELAY = 2
LAB_LAGS = (LAB_DELAY, LAB_DELAY + 1, LAB_DELAY + 2, 24)

PROCESS_FEATURES = (
    FEED
    + ["feed_assay_age_h"]
    + PROCESS
    + [f"{c} sd" for c in CONTROLS + [PULP_FLOW]]
    + ["hour_of_day"]
)
LAB_FEATURES = [f"silica_lab_lag{lag}" for lag in LAB_LAGS]
FEATURE_SETS = {"process": PROCESS_FEATURES, "process + lab": PROCESS_FEATURES + LAB_FEATURES}
PERSISTENCE = f"silica_lab_lag{LAB_DELAY}"


def build(hours: pl.DataFrame) -> pl.DataFrame:
    """Adds hour of day and the lab lags. A lag is the last *measured* value as of that many
    hours before, within the same continuous segment (never across the March gap), so an
    interpolated hour is never presented as a lab result."""
    last_measured = (
        pl.when(pl.col("measured")).then(pl.col(TARGET)).otherwise(None).forward_fill().over("segment")
    )
    df = hours.with_columns(pl.col("date").dt.hour().alias("hour_of_day"), last_measured.alias("_last_measured"))
    df = df.with_columns([pl.col("_last_measured").shift(lag).over("segment").alias(f"silica_lab_lag{lag}") for lag in LAB_LAGS])
    return df.drop("_last_measured").drop_nulls(subset=LAB_FEATURES)
