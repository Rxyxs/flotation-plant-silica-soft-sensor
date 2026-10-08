"""Can a model of the silica recommend reagent and pH setpoints?

The first version of this project optimized setpoints on a simulated plant and showed that
two optimizers (a genetic algorithm and differential evolution) agree. On real data that
agreement is the wrong test: two optimizers searching the *same* model will find the same
minimum whether or not the model is right. The test here is whether models trained on
different stretches of time recommend the same move for the same hour.

For the ``N_HOURS`` hours with the most silica in the last 20% of the data, four surrogate
models (XGBoost on the process features, trained on expanding windows that all end before
those hours) are each optimized over the four controls (starch, amine, pH, pulp density)
inside a trust region: within ``MAX_STEP`` of the current value and inside the 5th-95th
percentile of what the plant ever ran. ``operator_reaction`` checks the most likely
reason the models disagree: operators move the reagents *because* the silica moved.
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl
from deap import base, creator, tools
from scipy.optimize import differential_evolution

from src.data import CONTROLS, TARGET
from src.evaluation import GAP_HOURS, SEED, xgboost
from src.features import PROCESS_FEATURES

N_SURROGATES = 4
N_HOURS = 40
MAX_STEP = 0.10  # at most a 10% move from the current setpoint
EVAL_SHARE = 0.20


def surrogates(df: pl.DataFrame, n_models: int = N_SURROGATES) -> tuple[list, np.ndarray]:
    """Models trained on expanding windows that all end ``GAP_HOURS`` before the evaluation
    block (the last ``EVAL_SHARE`` of the hours); returns them and the block's row indices."""
    X = df.select(PROCESS_FEATURES).to_numpy()
    y = df[TARGET].to_numpy()
    measured = df["measured"].to_numpy()
    start = int(len(df) * (1 - EVAL_SHARE))
    models = []
    for k in range(1, n_models + 1):
        end = int((start - GAP_HOURS) * k / n_models)
        rows = np.arange(end)
        rows = rows[measured[rows]]
        models.append(xgboost().fit(X[rows], y[rows]))
    block = np.arange(start, len(df))
    return models, block[measured[block]]


def trust_region(x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, step: float = MAX_STEP) -> list[tuple[float, float]]:
    bounds = []
    for v, a, b in zip(x0, lo, hi):
        low, high = max(a, v * (1 - step)), min(b, v * (1 + step))
        if low > high:  # the current value sits outside the usual range: allow only it
            low = high = v
        bounds.append((float(low), float(high)))
    return bounds


def optimize_de(objective, bounds, seed: int = SEED) -> np.ndarray:
    if all(a == b for a, b in bounds):
        return np.array([a for a, _ in bounds])
    free = [i for i, (a, b) in enumerate(bounds) if a < b]
    fixed = np.array([a for a, _ in bounds], dtype=float)

    def f(v):
        x = fixed.copy()
        x[free] = v
        return objective(x)

    result = differential_evolution(f, [bounds[i] for i in free], seed=seed, maxiter=40, popsize=12, tol=1e-7)
    out = fixed.copy()
    out[free] = result.x
    return out


def optimize_ga(objective, bounds, seed: int = SEED, generations: int = 40, population: int = 40) -> np.ndarray:
    """A plain genetic algorithm (DEAP): tournament selection, blend crossover, Gaussian
    mutation, everything clipped to the trust region."""
    rng = random.Random(seed)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    lo = np.array([a for a, _ in bounds])
    hi = np.array([b for _, b in bounds])
    span = np.maximum(hi - lo, 1e-12)

    def clip(ind):
        for i in range(len(ind)):
            ind[i] = float(min(max(ind[i], lo[i]), hi[i]))
        return ind

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual([rng.uniform(a, b) for a, b in bounds]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (objective(np.array(ind)),))
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=population)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    best = tools.selBest(pop, 1)[0]
    for _ in range(generations):
        offspring = [creator.Individual(list(ind)) for ind in toolbox.select(pop, len(pop))]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                tools.cxBlend(a, b, alpha=0.3)
        for ind in offspring:
            if rng.random() < 0.3:
                for i in range(len(ind)):
                    ind[i] += rng.gauss(0, 0.1 * span[i])
            clip(ind)
            ind.fitness.values = toolbox.evaluate(ind)
        pop = offspring
        candidate = tools.selBest(pop, 1)[0]
        if candidate.fitness.values[0] < best.fitness.values[0]:
            best = creator.Individual(list(candidate))
            best.fitness.values = candidate.fitness.values
    return np.array(best)


def recommend(df: pl.DataFrame, n_hours: int = N_HOURS) -> dict:
    models, block = surrogates(df)
    X = df.select(PROCESS_FEATURES).to_numpy()
    y = df[TARGET].to_numpy()
    controls = [PROCESS_FEATURES.index(c) for c in CONTROLS]
    lo = np.percentile(X[:, controls], 5, axis=0)
    hi = np.percentile(X[:, controls], 95, axis=0)
    hours = block[np.argsort(-y[block], kind="stable")[:n_hours]]

    changes = np.zeros((len(models), len(hours), len(controls)))  # relative change, DE
    predicted_drop = np.zeros((len(models), len(hours)))
    ga_gap = []
    for i, model in enumerate(models):
        for j, row in enumerate(hours):
            x0 = X[row].copy()

            def objective(v, x0=x0, model=model):
                x = x0.copy()
                x[controls] = v
                return float(model.predict(x[None, :])[0])

            bounds = trust_region(x0[controls], lo, hi)
            best = optimize_de(objective, bounds)
            changes[i, j] = (best - x0[controls]) / x0[controls]
            predicted_drop[i, j] = objective(x0[controls]) - objective(best)
            if i == len(models) - 1:  # cross-check the optimizer on the last surrogate
                ga = optimize_ga(objective, bounds)
                ga_gap.append(objective(ga) - objective(best))

    signs = np.sign(np.round(changes, 6))
    agreement = {c: float(np.mean(np.all(signs[:, :, k] == signs[0, :, k], axis=0))) for k, c in enumerate(CONTROLS)}
    return {
        "hours": len(hours),
        "surrogates": len(models),
        "mean_change_pct": {c: [float(v) for v in changes[:, :, k].mean(axis=1) * 100] for k, c in enumerate(CONTROLS)},
        "direction_agreement": agreement,
        "predicted_drop_median": [float(np.median(p)) for p in predicted_drop],
        "ga_minus_de_median": float(np.median(ga_gap)),
        "ga_minus_de_max_abs": float(np.max(np.abs(ga_gap))),
        "changes": changes,
    }


def operator_reaction(df: pl.DataFrame, lag: int = 2) -> dict:
    """Correlation of the amine dose with the silica measured ``lag`` hours *earlier*, and
    with the silica in the same hour. If operators add amine when silica rises, the dose
    follows the silica, and a model reads high amine as a sign of high silica."""
    d = df.filter(pl.col("measured")).sort("date")
    same = d.select(pl.corr("Amina Flow", TARGET)).item()
    before = df.select(pl.corr("Amina Flow", pl.col(TARGET).shift(lag).over("segment"))).item()
    step = df.with_columns(
        pl.col("Amina Flow").diff().over("segment").alias("d_amine"),
        pl.col(TARGET).diff().over("segment").shift(1).alias("d_silica_before"),
    ).select(pl.corr("d_amine", "d_silica_before")).item()
    return {"amine_vs_silica_same_hour": float(same), f"amine_vs_silica_{lag}h_before": float(before),
            "amine_change_vs_previous_silica_change": float(step)}
