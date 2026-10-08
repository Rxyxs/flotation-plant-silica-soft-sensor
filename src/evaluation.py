"""How well can the silica be predicted, and how much does the validation protocol matter?

``walk_forward`` scores every model on five expanding folds in time, with a 24-hour gap
between training and test so the autocorrelation of the silica (0.78 at one hour) cannot
carry the answer across the boundary. Only ``measured`` hours are trained on or scored.

``protocol_comparison`` fits the same gradient-boosting model under four protocols, from
the one that leaks the most (random 20-second rows, iron in the concentrate as an input) to
the honest one (hourly, in time order, no iron), to show where a high R² comes from.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data import FEED, IRON_CONC, PROCESS, TARGET
from src.features import FEATURE_SETS, PERSISTENCE

SEED = 42
N_SPLITS = 5
GAP_HOURS = 24
RAW_SAMPLE = 200_000  # 20-second rows used in the random-split protocols


def xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1,
    )


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def catboost() -> CatBoostRegressor:
    return CatBoostRegressor(iterations=600, depth=5, learning_rate=0.03, random_seed=SEED, verbose=False)


MODELS = {"Ridge": ridge, "XGBoost": xgboost, "CatBoost": catboost}


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))


def walk_forward(df: pl.DataFrame, n_splits: int = N_SPLITS, gap: int = GAP_HOURS) -> dict:
    """RMSE per fold and pooled out-of-sample predictions for the training mean, persistence
    (last lab result known ``LAB_DELAY`` hours earlier) and every model on both feature sets."""
    y = df[TARGET].to_numpy()
    measured = df["measured"].to_numpy()
    folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(y))
    preds: dict[str, list[np.ndarray]] = {}
    per_fold: dict[str, list[float]] = {}
    test_rows = []

    def record(name: str, p: np.ndarray, actual: np.ndarray) -> None:
        preds.setdefault(name, []).append(np.asarray(p, dtype=float))
        per_fold.setdefault(name, []).append(rmse(p, actual))

    for train, test in folds:
        train, test = train[measured[train]], test[measured[test]]
        actual = y[test]
        test_rows.append(test)
        record("Training mean", np.full(len(test), y[train].mean()), actual)
        record("Persistence (lab, 2 h old)", df[PERSISTENCE].to_numpy()[test], actual)
        for set_name, features in FEATURE_SETS.items():
            X = df.select(features).to_numpy()
            for model_name, make in MODELS.items():
                model = make().fit(X[train], y[train])
                record(f"{model_name} ({set_name})", model.predict(X[test]), actual)

    rows = np.concatenate(test_rows)
    actual = y[rows]
    pooled = {name: np.concatenate(parts) for name, parts in preds.items()}
    summary = {}
    base = pooled["Training mean"]
    for name, p in pooled.items():
        summary[name] = {
            "rmse": rmse(p, actual),
            "rmse_per_fold": per_fold[name],
            "skill_vs_mean": 1 - rmse(p, actual) ** 2 / rmse(base, actual) ** 2,
        }
    return {"summary": summary, "pooled": pooled, "actual": actual, "rows": rows,
            "fold_sizes": [(len(a), len(b)) for a, b in folds]}


def diebold_mariano(pred_a: np.ndarray, pred_b: np.ndarray, actual: np.ndarray, max_lags: int = 24) -> dict:
    """Squared-error Diebold-Mariano test with a Newey-West variance; negative = A better."""
    d = (np.asarray(pred_a) - actual) ** 2 - (np.asarray(pred_b) - actual) ** 2
    fit = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return {"statistic": float(fit.tvalues[0]), "p_value": float(fit.pvalues[0]), "mean_difference": float(d.mean())}


def protocol_comparison(raw: pl.DataFrame, hours: pl.DataFrame) -> dict:
    """R² of the same XGBoost model under four validation protocols."""
    inputs = FEED + PROCESS
    out = {}
    sample = raw.sample(min(RAW_SAMPLE, raw.height), seed=SEED)
    y_raw = sample[TARGET].to_numpy()
    for name, cols in (("random 20-second rows, with iron in the concentrate", inputs + [IRON_CONC]),
                       ("random 20-second rows", inputs)):
        X = sample.select(cols).to_numpy()
        scores = [r2_score(y_raw[te], xgboost().fit(X[tr], y_raw[tr]).predict(X[te]))
                  for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X)]
        out[name] = float(np.mean(scores))
    measured = hours.filter(pl.col("measured"))
    X, y = measured.select(inputs).to_numpy(), measured[TARGET].to_numpy()
    scores = [r2_score(y[te], xgboost().fit(X[tr], y[tr]).predict(X[te]))
              for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X)]
    out["random hours"] = float(np.mean(scores))
    scores = [r2_score(y[te], xgboost().fit(X[tr], y[tr]).predict(X[te]))
              for tr, te in TimeSeriesSplit(N_SPLITS, gap=GAP_HOURS).split(X)]
    out["hours in time order"] = float(np.mean(scores))
    out["hours in time order, per fold"] = [float(s) for s in scores]
    return out
