"""Analysis-only evaluation harness for Laptop Price Intelligence.

Reproduces the notebook protocol read-only: deterministic cleaning + FE
(mirrors notebook, no fitting), train_test_split(test_size=0.20,
random_state=42), predict-only on saved pipeline, expm1 to rupees.
Descriptive test metrics only. Writes ONLY under reports/model_evaluation/.
Historical listings, not current-market prices.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "laptop_data.csv"
PIPE_PATH = ROOT / "models" / "laptop_price_pipeline.pkl"
META_PATH = ROOT / "models" / "model_metadata.json"
OUT_DIR = ROOT / "reports" / "model_evaluation"

FEATURE_COLS = [
    "Company", "TypeName", "OpSys", "CPU_Brand", "CPU_Tier",
    "GPU_Brand", "GPU_Series", "Inches", "RAM_GB", "Weight_kg",
    "Resolution_X", "Resolution_Y", "Touchscreen", "IPS",
    "Pixels", "PPI", "CPU_Freq_GHz", "SSD_GB", "HDD_GB",
    "Flash_GB", "Hybrid_GB",
]
SMALL_GROUP_N = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def parse_memory_nb(mem: str) -> dict:
    out = {"SSD_GB": 0.0, "HDD_GB": 0.0, "Flash_GB": 0.0, "Hybrid_GB": 0.0}
    for part in str(mem).split("+"):
        part = part.strip()
        m = re.match(r"([\d.]+)\s*(TB|GB)\s*(.*)", part, re.I)
        assert m, f"unparseable memory part: {part!r}"
        gb = float(m.group(1)) * (1000.0 if m.group(2).upper() == "TB" else 1.0)
        rest = m.group(3).upper()
        if "FLASH" in rest:
            out["Flash_GB"] += gb
        elif "HYBRID" in rest:
            out["Hybrid_GB"] += gb
        elif "SSD" in rest:
            out["SSD_GB"] += gb
        elif "HDD" in rest:
            out["HDD_GB"] += gb
    return out


def build_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Mirror notebooks/laptop_price_analysis.ipynb cleaning + FE exactly."""
    idx_col = [c for c in df_raw.columns if c.startswith("Unnamed")]
    assert idx_col == ["Unnamed: 0"], f"unexpected id cols: {idx_col}"
    df = df_raw.drop(columns=idx_col)
    df = df.drop_duplicates().reset_index(drop=True)
    df["RAM_GB"] = df["Ram"].str.replace("GB", "", regex=False).astype(int)
    df["Weight_kg"] = df["Weight"].str.replace("kg", "", regex=False).astype(float)
    res = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)")
    assert res.notna().all().all(), "resolution regex failed"
    df["Resolution_X"] = res[0].astype(int)
    df["Resolution_Y"] = res[1].astype(int)
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    mem = df["Memory"].apply(parse_memory_nb)
    for k in ("SSD_GB", "HDD_GB", "Flash_GB", "Hybrid_GB"):
        df[k] = [d[k] for d in mem]
    assert (df["SSD_GB"] + df["HDD_GB"] + df["Flash_GB"] + df["Hybrid_GB"] > 0).all()

    def cpu_tier(cpu: str) -> str:
        for t in ("Core i3", "Core i5", "Core i7", "Core i9"):
            if t in cpu:
                return t
        b = cpu.split()[0]
        if b == "Intel":
            for t in ("Celeron", "Pentium", "Atom"):
                if t in cpu:
                    return t
            return "Intel Other"
        if b == "AMD":
            return "AMD Ryzen" if "Ryzen" in cpu else "AMD Other"
        return "Other"

    df["CPU_Brand"] = df["Cpu"].apply(lambda s: s.split()[0])
    df["CPU_Tier"] = df["Cpu"].apply(cpu_tier)
    freq = df["Cpu"].str.extract(r"([\d.]+)\s*GHz")[0]
    assert freq.notna().all()
    df["CPU_Freq_GHz"] = freq.astype(float)

    def gpu_series(g: str) -> str:
        if "GeForce" in g or g.startswith("GTX"):
            return "GeForce"
        if "Quadro" in g:
            return "Quadro"
        if "Radeon" in g:
            return "Radeon"
        if "FirePro" in g:
            return "FirePro"
        if "Iris" in g:
            return "Iris"
        if "UHD" in g:
            return "UHD"
        if "HD" in g:
            return "HD"
        if "Mali" in g:
            return "Mali"
        return "Other"

    df["GPU_Brand"] = df["Gpu"].apply(lambda s: s.split()[0])
    df["GPU_Series"] = df["Gpu"].apply(gpu_series)
    df["Pixels"] = df["Resolution_X"] * df["Resolution_Y"]
    df["PPI"] = np.hypot(df["Resolution_X"], df["Resolution_Y"]) / df["Inches"]
    assert df[FEATURE_COLS].notna().all().all()
    return df.reset_index(drop=True)


def metrics_rupees(y_true, y_pred) -> dict:
    yp = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    yt = np.asarray(y_true, dtype=float)
    ae = np.abs(yt - yp)
    ape = ae / yt * 100.0
    return {
        "n": int(len(yt)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "MAPE_pct": float(np.mean(ape)),
        "MedAPE_pct": float(np.median(ape)),
        "within_10_pct": float(np.mean(ape <= 10) * 100),
        "within_20_pct": float(np.mean(ape <= 20) * 100),
        "within_30_pct": float(np.mean(ape <= 30) * 100),
    }


def group_table(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for name, g in frame.groupby(col):
        ae = (g["predicted"] - g["actual"]).abs()
        ape = ae / g["actual"] * 100
        rows.append({col: name, "n": int(len(g)), "MAE": float(ae.mean()),
                     "MedAPE_pct": float(ape.median()),
                     "small_group_warning": bool(len(g) < SMALL_GROUP_N)})
    return pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)
def main() -> None:
    assert DATA_PATH.exists() and PIPE_PATH.exists() and META_PATH.exists()
    before_hash = sha256(PIPE_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert meta["test_size"] == 0.2 and meta["random_state"] == 42
    assert meta["target_transformation"] == "log1p"
    df_raw = pd.read_csv(DATA_PATH)
    df = build_frame(df_raw)
    assert len(df) == meta["dataset_rows_after_cleaning"] == 1274
    X = df[FEATURE_COLS].copy()
    y = df["Price"].copy()
    assert list(X.columns) == meta["features"] == FEATURE_COLS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    assert len(X_test) == meta["test_rows"] == 255
    assert len(X_train) == meta["train_rows"] == 1019
    with open(PIPE_PATH, "rb") as f:
        pipe = pickle.load(f)
    assert list(getattr(pipe, "feature_names_in_", [])) == FEATURE_COLS
    raw_pred = np.asarray(pipe.predict(X_test), dtype=float)
    assert np.all(np.isfinite(raw_pred)), "non-finite log-space predictions"
    y_pred = np.clip(np.expm1(raw_pred), 0, None)
    assert np.all(np.isfinite(y_pred)) and (y_pred > 0).all()
    m = metrics_rupees(y_test.values, y_pred)
    stored = meta["test_metrics"]
    abs_diff = {"MAE": abs(m["MAE"] - stored["MAE"]),
                "RMSE": abs(m["RMSE"] - stored["RMSE"]),
                "R2": abs(m["R2"] - stored["R2"]),
                "MAPE_pct": abs(m["MAPE_pct"] - stored["MAPE_%"])}
    err = X_test.copy()
    err["actual"] = y_test.values
    err["predicted"] = y_pred
    err["residual"] = err["predicted"] - err["actual"]
    err["abs_error"] = err["residual"].abs()
    err["ape_pct"] = err["abs_error"] / err["actual"] * 100
    bands = [0, 30000, 60000, 100000, 200000, float("inf")]
    labels = ["<30k", "30-60k", "60-100k", "100-200k", "200k+"]
    err["price_band"] = pd.cut(err["actual"], bins=bands, labels=labels,
                               right=False)
    band_rows = []
    for b in labels:
        g = err[err["price_band"] == b]
        row = {"price_band": b, "n": int(len(g)),
               "small_group_warning": bool(len(g) < SMALL_GROUP_N)}
        if len(g):
            for k, v in metrics_rupees(g["actual"].values,
                                       g["predicted"].values).items():
                if k != "n":
                    row[k] = v
        band_rows.append(row)
    band_df = pd.DataFrame(band_rows)
    brand_df = group_table(err, "Company")
    type_df = group_table(err, "TypeName")
    worst10 = err.sort_values("abs_error", ascending=False).head(10)[
        ["Company", "TypeName", "Inches", "RAM_GB", "CPU_Tier",
         "CPU_Freq_GHz", "GPU_Series", "SSD_GB", "HDD_GB", "Flash_GB",
         "Hybrid_GB", "Weight_kg", "actual", "predicted",
         "abs_error", "ape_pct"]].rename(columns={"ape_pct": "pct_error"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    err.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    band_df.to_csv(OUT_DIR / "metrics_by_price_band.csv", index=False)
    brand_df.to_csv(OUT_DIR / "metrics_by_brand.csv", index=False)
    type_df.to_csv(OUT_DIR / "metrics_by_type.csv", index=False)
    worst10.to_csv(OUT_DIR / "largest_10_errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(err["actual"], err["predicted"], alpha=0.6, s=30)
    lo = float(min(err["actual"].min(), err["predicted"].min()))
    hi = float(max(err["actual"].max(), err["predicted"].max()))
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    ax.set_xlabel("Actual price (INR, historical listings)")
    ax.set_ylabel("Predicted price (INR)")
    ax.set_title("Actual vs predicted (held-out test, descriptive only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(err["predicted"], err["residual"], alpha=0.6, s=30)
    ax.axhline(0, color="red", ls="--")
    ax.set_xlabel("Predicted price (INR)")
    ax.set_ylabel("Residual = predicted - actual (INR)")
    ax.set_title("Residuals vs predicted price (held-out test)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residuals_vs_predicted.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(err["residual"], bins=30)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("Residual (INR)")
    ax.set_title(f"Residual distribution (mean {err['residual'].mean():+,.0f} INR)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residual_distribution.png", dpi=150)
    plt.close(fig)
    summary = {
        "note": "Historical listings only; not current-market prices. "
                "Descriptive test analysis; no tuning on test data.",
        "dataset": str(DATA_PATH.name),
        "rows_raw": int(len(df_raw)), "rows_clean": int(len(df)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "split": "train_test_split(test_size=0.20, random_state=42)",
        "target_transformation": "log1p -> expm1 to rupees, clip>=0",
        "metrics_rupees": m,
        "stored_test_metrics": stored,
        "comparison_abs_diff": abs_diff,
        "small_group_threshold": SMALL_GROUP_N,
        "model_sha256_before": before_hash,
        "model_sha256_after": sha256(PIPE_PATH),
    }
    assert summary["model_sha256_before"] == summary["model_sha256_after"]
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    summary["outputs"] = sorted(p.name for p in OUT_DIR.iterdir())
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
