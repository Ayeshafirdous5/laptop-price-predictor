"""Feature-importance audit for the saved Laptop Price pipeline.

Read-only analysis: loads the existing fitted Pipeline, reproduces the
deterministic cleaning/split used by scripts/evaluate_model.py, and computes
two global importance estimates *without* retraining, tuning, or refitting
the model.

Output: reports/model_evaluation/feature_importance.json
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reuse the canonical cleaning/split helpers from evaluate_model.py; when this
# script is run directly, its own directory is sys.path[0], so the sibling
# module imports cleanly regardless of the current working directory.
import evaluate_model as em

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "laptop_data.csv"
PIPE_PATH = ROOT / "models" / "laptop_price_pipeline.pkl"
OUT_DIR = ROOT / "reports" / "model_evaluation"
OUT_PATH = OUT_DIR / "feature_importance.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_MEANING = {
    "RAM_GB": "RAM in GB",
    "SSD_GB": "SSD capacity in GB",
    "HDD_GB": "HDD capacity in GB",
    "Flash_GB": "Flash capacity in GB",
    "Hybrid_GB": "Hybrid capacity in GB",
    "Inches": "Screen size in inches",
    "Weight_kg": "Weight in kg",
    "CPU_Freq_GHz": "CPU clock frequency in GHz",
    "Resolution_X": "Horizontal resolution (px)",
    "Resolution_Y": "Vertical resolution (px)",
    "Pixels": "Total pixels (derived: X*Y)",
    "PPI": "Pixels per inch (derived from resolution + inches)",
    "Touchscreen": "1 if screen is touch, else 0",
    "IPS": "1 if IPS panel, else 0",
    "Company": "Laptop manufacturer",
    "TypeName": "Laptop type (e.g. Gaming)",
    "OpSys": "Operating system",
    "CPU_Brand": "CPU manufacturer tier",
    "CPU_Tier": "CPU family tier",
    "GPU_Brand": "GPU manufacturer",
    "GPU_Series": "GPU series",
}


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fitted_pipeline() -> Pipeline:
    """Load the saved Pipeline without retraining. Raises on bad data."""
    before = sha256(PIPE_PATH)
    with open(PIPE_PATH, "rb") as f:
        pipe = pickle.load(f)
    after = sha256(PIPE_PATH)
    if before != after:
        raise RuntimeError("model file changed during read: SHA-256 mismatch")
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"unexpected pipeline type: {type(pipe).__name__}")
    return pipe

def clean_and_split():
    """Reproduce the deterministic 80/20 split from evaluate_model.py."""
    raw = pd.read_csv(DATA_PATH)
    df = em.build_frame(raw)
    X = df[em.FEATURE_COLS].copy()
    y = df["Price"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def _grouped_impurity(imp: np.ndarray, ct: ColumnTransformer) -> pd.Series:
    """Aggregate 74 transformed importances back to the original features.

    Walks the fitted ColumnTransformer's positional transformers structure and
    expands each OneHotEncoder's categories_ per original column; never splits
    get_feature_names_out strings on underscores.
    """
    base_names: list[str] = []
    for _name, _trans, _cols in ct.transformers_:
        if isinstance(_trans, OneHotEncoder):
            for _col, _cats in zip(_cols, _trans.categories_):
                base_names.extend([_col] * len(_cats))
        elif isinstance(_trans, StandardScaler):
            base_names.extend(_cols)
        else:
            raise TypeError(f"unexpected transformer type: {type(_trans).__name__}")

    if len(base_names) != imp.shape[0]:
        raise ValueError(
            f"importance length {imp.shape[0]} != mapped columns {len(base_names)}"
        )

    per_feature = pd.Series(imp, index=base_names)
    grouped = per_feature.groupby(level=0).sum().sort_values(ascending=False)
    return grouped


def compute_impurity_importance(pipe: Pipeline) -> pd.Series:
    """Return impurity importance (unitless, sums to 1) per original feature."""
    model = pipe.named_steps["model"]
    ct = pipe.named_steps["preprocess"]

    if not isinstance(model, GradientBoostingRegressor):
        raise TypeError("model is not GradientBoostingRegressor")

    imp = np.asarray(model.feature_importances_, dtype=float)
    return _grouped_impurity(imp, ct)


def mae_in_rupees(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rupee MAE for permutation importance.

    The model predicts log1p(price) while y_true is raw rupee price. We expm1
    the predictions only (never the true values) and score in INR.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    pred_rupee = np.clip(np.expm1(y_pred), 0, None)
    return float(mean_absolute_error(y_true, pred_rupee))


def compute_permutation_importance(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Permutation importance (mean +/- std, 10 repeats, rupee MAE units)."""
    from sklearn.inspection import permutation_importance

    scorer = make_scorer(mae_in_rupees, greater_is_better=False)
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=scorer,
        n_jobs=1,
    )
    out = pd.DataFrame(
        {
            "feature": em.FEATURE_COLS,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return out


def build_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    impurity: pd.Series,
    permutation: pd.DataFrame,
    versions: dict,
) -> dict:
    """Assemble the machine-readable report with units, flags, and limitations."""
    importance_report = {
        "metric": "impurity-based tree importance (mean decrease in MSE)",
        "unit": "unitless; normalized to sum=1 over the 74 transformed columns",
        "aggregation_method": (
            "sum over one-hot levels per original feature, via fitted "
            "ColumnTransformer positional mapping"
        ),
        "features": [
            {
                "feature": feat,
                "importance": float(round(imp, 6)),
                "unit": FEATURE_MEANING.get(feat, ""),
                "derived": feat in ("Pixels", "PPI"),
            }
            for feat, imp in impurity.items()
        ],
    }

    permutation_report = {
        "metric": "permutation importance: held-out rupee MAE increase",
        "units": "Indian Rupees (INR)",
        "scorer": "mae_in_rupees (expm1 only on predictions; y_true raw)",
        "greater_is_better": False,
        "n_repeats": 10,
        "random_state": RANDOM_STATE,
        "aggregation_method": "mean +/- std across repeats, per original feature",
        "features": [
            {
                "feature": row["feature"],
                "importance_mean": float(round(row["importance_mean"], 2)),
                "importance_std": float(round(row["importance_std"], 2)),
                "unit": "INR",
                "derived": row["feature"] in ("Pixels", "PPI"),
            }
            for _, row in permutation.iterrows()
        ],
    }


    return {
        "title": "Feature importance report (read-only audit)",
        "summary": {
            "note": (
                "Historical laptop listings, not verified current-market "
                "prices. Importance is predictive association, not causation."
            ),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(y_test)),
            "split": "train_test_split(test_size=0.20, random_state=42)",
            "target_transformation": "log1p -> expm1 to rupees, clip>=0",
            "model": "Pipeline(preprocess + GradientBoostingRegressor)",
            "model_sha256": sha256(PIPE_PATH),
        },
        "split_details": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "impurity_importance": importance_report,
        "permutation_importance": permutation_report,
        "derived_flags": {
            "Pixels": (
                "Total pixels = Resolution_X * Resolution_Y; an exact function "
                "of resolution and therefore carries no independent "
                "information beyond the split into the two resolution "
                "dimensions."
            ),
            "PPI": (
                "Pixels per inch = hypot(Resolution_X, Resolution_Y) / Inches; "
                "an exact function of screen size and resolution, so it "
                "carries no independent information."
            ),
            "guidance": (
                "Both derived columns are flagged so that readers do not "
                "attribute causal effect to them; correlated features share "
                "importance structurally."
            ),
        },
        "limitations": [
            (
                "Global (population-level) importance. It does not explain "
                "individual laptops; per-row attribution would require a local "
                "method (e.g. SHAP TreeExplainer) at inference time."
            ),
            (
                "Impurity importance is in-sample, biased toward continuous/"
                "high-cardinality features, and shares importance "
                "structurally among correlated features (e.g. "
                "Pixels/PPI/Resolution_X/Y)."
            ),
            (
                "Permutation importance measures held-out metric degradation "
                "when a feature is shuffled; it reflects association used by "
                "the model, not causal effect."
            ),
            (
                "Data are historical listings; prices may differ from the "
                "current market."
            ),
        ],
        "versions": versions,
    }


def main() -> int:
    """Run the read-only audit and write the JSON report."""
    import platform

    import sklearn

    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }

    print("Loading fitted pipeline ...")
    pipe = load_fitted_pipeline()

    print("Reproducing clean + split ...")
    X_train, X_test, y_train, y_test = clean_and_split()

    print("Computing impurity importance ...")
    impurity = compute_impurity_importance(pipe)

    print("Computing permutation importance (10 repeats) ...")
    permutation = compute_permutation_importance(pipe, X_test, y_test)

    report = build_report(
        X_train, X_test, y_test, impurity, permutation, versions
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {OUT_PATH}")

    print("\n=== Impurity importance (unitless, sum=1) per original feature ===")
    for feat, imp in impurity.items():
        print(f"  {feat:<16} {imp:.6f}")
    print("\n=== Permutation importance (INR MAE increase, mean +/- std) ===")
    print(permutation.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
