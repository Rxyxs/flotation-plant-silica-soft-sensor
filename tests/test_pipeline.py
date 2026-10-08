"""Offline tests for each step, on small fixtures shaped like the Kaggle file. The results
in the README come from the real data; these check that every step does what it says."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.data import FEED, IRON_CONC, PROCESS, TARGET, hourly, linear_segments, read_raw
from src.evaluation import diebold_mariano, walk_forward
from src.features import LAB_DELAY, PERSISTENCE, build
from src.prescription import operator_reaction, optimize_de, optimize_ga, trust_region

COLUMNS = ["date"] + FEED + PROCESS + [IRON_CONC, TARGET]


def _raw(hours: int = 60, per_hour: int = 3, seed: int = 0, gap_after: int | None = None) -> pl.DataFrame:
    """A small plant: ``per_hour`` readings per hour, silica measured once an hour."""
    rng = np.random.default_rng(seed)
    rows = []
    start = dt.datetime(2017, 3, 10, 1)
    t = start
    silica = 2.0
    for h in range(hours):
        if gap_after is not None and h == gap_after:
            t += dt.timedelta(hours=100)
        silica = float(np.clip(silica + rng.normal(0, 0.3), 0.6, 5.5))
        feed = 55.0 if h < hours // 2 else 57.0
        for _ in range(per_hour):
            values = {c: float(rng.normal(300, 10)) for c in PROCESS}
            values.update({"% Iron Feed": feed, "% Silica Feed": 70 - feed, IRON_CONC: 65.0, TARGET: silica})
            rows.append({"date": t, **values})
        t += dt.timedelta(hours=1)
    return pl.DataFrame(rows).select(COLUMNS)


def test_read_raw_parses_comma_decimals(tmp_path):
    path = tmp_path / "plant.csv"
    header = ",".join(COLUMNS)
    line = "2017-03-10 01:00:00," + ",".join(['"1,5"'] * (len(COLUMNS) - 1))
    path.write_text(header + "\n" + line + "\n", encoding="utf-8")
    df = read_raw(path)
    assert df["date"][0] == dt.datetime(2017, 3, 10, 1)
    assert df[TARGET][0] == 1.5


def test_linear_segments_flags_only_the_inside_of_straight_runs():
    values = np.array([2.0, 2.0, 2.5, 3.0, 3.5, 3.1, 3.1])
    assert linear_segments(values).tolist() == [False, False, True, True, False, False, False]


def test_hourly_flags_interpolated_hours_and_measures_feed_assay_age():
    raw = _raw(hours=40)
    # make hour 10 vary within the hour (interpolated at 20-second resolution)
    t10 = raw["date"].unique().sort()[10]
    raw = raw.with_columns(
        pl.when(pl.col("date") == t10).then(pl.col(TARGET) + pl.int_range(pl.len()) * 0.01).otherwise(pl.col(TARGET)).alias(TARGET)
    )
    h = hourly(raw)
    assert h.height == 40 and (h["readings"] == 3).all()
    assert not h["measured"][10] and h["measured"].sum() >= 38
    age = h["feed_assay_age_h"].to_list()
    assert age[0] == 0 and age[19] == 19 and age[20] == 0  # the feed assay changes at hour 20


def test_gap_starts_a_new_segment_and_lab_lags_never_cross_it():
    h = hourly(_raw(hours=50, gap_after=25))
    assert h["segment"].n_unique() == 2
    df = build(h)
    first_after_gap = df.filter(pl.col("segment") == 1)["date"].min()
    # the first rows after the gap are dropped until the longest lag (24 h) is available
    assert (first_after_gap - h.filter(pl.col("segment") == 1)["date"].min()).total_seconds() / 3600 >= 24


def test_lab_lag_is_the_last_measured_value_at_least_two_hours_old():
    h = hourly(_raw(hours=60)).with_columns(
        pl.when(pl.int_range(pl.len()) == 40).then(False).otherwise(pl.col("measured")).alias("measured")
    )
    df = build(h)
    row = df.filter(pl.col("date") == h["date"][42])
    # two hours before hour 42 is hour 40, which is not measured: the lag carries hour 39
    assert row[PERSISTENCE][0] == h[TARGET][39]
    assert LAB_DELAY == 2


def test_walk_forward_scores_only_measured_hours_and_every_model():
    rng = np.random.default_rng(3)
    h = hourly(_raw(hours=400, per_hour=2, seed=4))
    df = build(h).with_columns(pl.Series("measured", rng.random(build(h).height) > 0.1))
    out = walk_forward(df, n_splits=3, gap=5)
    assert len(out["actual"]) == len(out["rows"])
    assert df["measured"].to_numpy()[out["rows"]].all()
    for name in ("Training mean", "Persistence (lab, 2 h old)", "Ridge (process)", "CatBoost (process + lab)"):
        assert name in out["summary"] and len(out["summary"][name]["rmse_per_fold"]) == 3
    assert out["summary"]["Training mean"]["skill_vs_mean"] == 0.0


def test_diebold_mariano_prefers_the_better_forecast():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=3000)
    good = actual + rng.normal(0, 0.5, 3000)
    bad = actual + rng.normal(0, 1.0, 3000)
    r = diebold_mariano(good, bad, actual)
    assert r["statistic"] < 0 and r["p_value"] < 0.01


def test_trust_region_and_both_optimizers_find_the_same_minimum():
    bounds = trust_region(np.array([100.0, 10.0]), lo=np.array([50.0, 9.5]), hi=np.array([150.0, 10.5]))
    assert np.allclose(bounds, [(90.0, 110.0), (9.5, 10.5)])

    def objective(v):
        return (v[0] - 95.0) ** 2 + (v[1] - 12.0) ** 2  # the second minimum is outside the region

    de = optimize_de(objective, bounds)
    ga = optimize_ga(objective, bounds)
    assert de[0] == pytest.approx(95.0, abs=0.1) and de[1] == pytest.approx(10.5, abs=1e-3)
    assert np.allclose(ga, de, atol=0.5)
    assert all(a <= x <= b for x, (a, b) in zip(ga, bounds))


def test_trust_region_pins_a_value_outside_the_usual_range():
    assert trust_region(np.array([200.0]), lo=np.array([50.0]), hi=np.array([150.0])) == [(200.0, 200.0)]


def test_operator_reaction_detects_a_dose_that_follows_the_silica():
    rng = np.random.default_rng(1)
    n = 500
    silica = np.cumsum(rng.normal(0, 0.2, n)) + 3
    amine = 400 + 30 * np.roll(silica, 2) + rng.normal(0, 5, n)  # dose reacts 2 h later
    df = pl.DataFrame({"date": np.arange(n), "segment": np.zeros(n, dtype=int), "measured": np.ones(n, dtype=bool),
                       TARGET: silica, "Amina Flow": amine})
    r = operator_reaction(df)
    assert r["amine_vs_silica_2h_before"] > 0.9
