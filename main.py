"""
Silica in the concentrate of a real flotation plant: what can be predicted, and what can
be prescribed.

    python main.py --download   # once: the Kaggle dataset into data/raw/ (Kaggle API key needed)
    python main.py              # every analysis, outputs/results.json and outputs/figures/
"""

from __future__ import annotations

import argparse
import json
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl

from src.data import CSV_NAME, RAW_DIR, TARGET, download, hourly, read_raw
from src.evaluation import diebold_mariano, protocol_comparison, walk_forward
from src.features import LAB_DELAY, build
from src.plots import (
    plot_operator_reaction,
    plot_protocols,
    plot_recommendations,
    plot_silica_series,
    plot_walk_forward,
)
from src.prescription import MAX_STEP, operator_reaction, recommend

RESULTS = RAW_DIR.parent.parent / "outputs" / "results.json"
BEST = "Ridge (process + lab)"
PERSISTENCE = "Persistence (lab, 2 h old)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download the dataset from Kaggle")
    args = parser.parse_args()
    if args.download:
        download()

    raw = read_raw(RAW_DIR / CSV_NAME)
    hours = hourly(raw)
    df = build(hours)
    target = hours[TARGET].to_numpy()
    data = {
        "rows": raw.height,
        "first_hour": str(hours["date"].min()),
        "last_hour": str(hours["date"].max()),
        "hours": hours.height,
        "measured_hours": int(hours["measured"].sum()),
        "interpolated_hours": int((~hours["measured"]).sum()),
        "largest_gap_hours": int(hours["date"].diff().dt.total_hours().max()),
        "longest_unchanged_feed_assay_hours": int(hours["feed_assay_age_h"].max() + 1),
        "silica_mean": float(target.mean()),
        "silica_sd": float(target.std()),
        "silica_autocorrelation_1h": float(np.corrcoef(target[:-1], target[1:])[0, 1]),
        "iron_vs_silica_concentrate_corr": float(np.corrcoef(hours["% Iron Concentrate"].to_numpy(), target)[0, 1]),
    }
    print(data)

    protocols = protocol_comparison(raw, hours)
    print(protocols)
    wf = walk_forward(df)
    dm = {
        "ridge_lab_vs_persistence": diebold_mariano(wf["pooled"][BEST], wf["pooled"][PERSISTENCE], wf["actual"]),
        "ridge_process_vs_mean": diebold_mariano(
            wf["pooled"]["Ridge (process)"], wf["pooled"]["Training mean"], wf["actual"]
        ),
    }
    for name, s in sorted(wf["summary"].items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{name:<30} {s['rmse']:.3f}")
    reaction = operator_reaction(df)
    rec = recommend(df)
    print({k: v for k, v in rec.items() if k != "changes"})

    plot_silica_series(hours)
    plot_protocols({k: v for k, v in protocols.items() if not k.endswith("per fold")})
    plot_walk_forward(wf["summary"])
    plot_recommendations(rec["mean_change_pct"], rec["direction_agreement"])
    plot_operator_reaction(df)

    results = {
        "data": data,
        "lab_delay_hours": LAB_DELAY,
        "validation_protocols_r2": protocols,
        "walk_forward": {"summary": wf["summary"], "fold_sizes": wf["fold_sizes"], "scored_hours": len(wf["actual"])},
        "diebold_mariano": dm,
        "operator_reaction": reaction,
        "recommendations": {**{k: v for k, v in rec.items() if k != "changes"}, "max_step": MAX_STEP},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {RESULTS}")


if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()
