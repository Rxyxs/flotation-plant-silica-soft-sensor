"""Every number in the READMEs' results table must come from `outputs/results.json`, the
file `main.py` writes. If the pipeline changes and a README is not updated, this fails."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.json"
READMES = [("en", "README.md"), ("es", "README.es.md")]
LABELS_ES = {
    "Ridge (process + lab)": "Ridge (proceso + laboratorio)",
    "CatBoost (process + lab)": "CatBoost (proceso + laboratorio)",
    "Persistence (lab, 2 h old)": "Persistencia (laboratorio de hace 2 h)",
    "XGBoost (process + lab)": "XGBoost (proceso + laboratorio)",
    "Ridge (process)": "Ridge (proceso)",
    "Training mean": "Promedio del entrenamiento",
    "CatBoost (process)": "CatBoost (proceso)",
    "XGBoost (process)": "XGBoost (proceso)",
}


def _fmt(x: float, decimals: int, lang: str) -> str:
    s = f"{x:.{decimals}f}"
    return s.replace(".", ",") if lang == "es" else s


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("outputs/results.json not found: run `python main.py` first")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang, readme", READMES)
def test_walk_forward_table_matches_results(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for name, s in results["walk_forward"]["summary"].items():
        label = LABELS_ES[name] if lang == "es" else name
        skill = f"{s['skill_vs_mean'] * 100:+.1f}%"
        row = f"| {label} | {_fmt(s['rmse'], 3, lang)} | {skill.replace('.', ',') if lang == 'es' else skill} |"
        assert row in text, f"{readme}: expected {row}"


@pytest.mark.parametrize("lang, readme", READMES)
def test_headline_numbers_match_results(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    r2 = results["validation_protocols_r2"]
    for key in ("random 20-second rows, with iron in the concentrate", "random 20-second rows", "random hours", "hours in time order"):
        assert _fmt(r2[key], 2, lang).lstrip("-") in text, f"{readme}: R² {key}"
    data = results["data"]
    for value in (data["interpolated_hours"], data["largest_gap_hours"], data["longest_unchanged_feed_assay_hours"]):
        assert str(value) in text
    agree = results["recommendations"]["direction_agreement"]
    for value in agree.values():
        assert f"{round(value * 100)}%" in text
