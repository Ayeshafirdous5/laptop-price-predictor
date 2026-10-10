# Laptop Price Intelligence

A full-stack machine learning project that estimates laptop prices in Indian Rupees (₹) from hardware specifications. A Flask web application serves a scikit-learn pipeline trained on **1,274 historical laptop listings**; the same feature-engineering module powers both training and inference, so the app and the model can never disagree about what the inputs mean. The repository also includes an automated test suite and read-only evaluation tooling whose outputs live in `reports/model_evaluation/`.

The web page itself is branded "Laptop Price Predictor" (see the HTML `<title>`).

> **Historical-data disclaimer:** every prediction is an estimate learned from past listings. It is not a guaranteed current market price, and it must not be used as the sole basis for buying or selling decisions.

## Problem statement and objectives

Laptop prices vary with many interacting specifications — RAM, CPU tier, storage mix, screen quality, brand and more. This project aims to:

- estimate a laptop's price from its specifications with a reproducible machine-learning pipeline;
- keep training and serving logic in exact sync (one shared feature-engineering module, plus a schema check when the app loads the model);
- evaluate the model honestly on a held-out test set, including per-segment breakdowns with explicit small-sample warnings;
- fail gracefully: friendly validation messages instead of server errors, and a health endpoint that reports model readiness;
- document limitations clearly rather than overstate accuracy.

## Key application features

All verified in `app.py` and `templates/index.html`:

- **Prediction form** at `GET/POST /` with dataset-derived options: 19 brands, 6 laptop types, 9 CPUs, 9 GPUs, 9 operating systems, 9 RAM sizes, 15 storage capacities, plus screen size, resolution, weight, touchscreen and IPS controls.
- **Server-side validation** with user-safe messages (invalid screen size, resolution, weight, storage, missing fields, …). Malformed input never returns a traceback or an HTTP 500 — covered by the test suite.
- **Graceful model handling:** if the model file is missing or unreadable, the app still starts and explains the problem instead of crashing. On load it compares the pipeline's expected features against the 21 columns defined in `src/feature_engineering.py` and refuses to predict on a schema mismatch.
- **Health endpoint** at `GET /health` returning JSON: `status` (`ok`/`degraded`), `model_loaded`, `schema_matches`, `feature_count` (21) and `target_transformation` (`log1p`).
- **Accessible, responsive UI:** skip-to-form link, labelled fieldsets, an `aria-live` result panel, and disclaimers next to every displayed price (header trust note, result note and page footer).
- **Rupee predictions** obtained by inverting the training target (`expm1` of the log-space prediction), with guards against non-finite or non-positive outputs. Prices are rounded to whole rupees.
- **Clear button** that resets the form, and preserved form values after a failed submission.

## Technology stack

| Layer | Technology |
| --- | --- |
| Web framework | Flask (Jinja2 templates) |
| Language | Python — developed and verified on Python 3.12.7 |
| Data handling | pandas 2.2.2, numpy 1.26.4 |
| Machine learning | scikit-learn 1.8.0: `Pipeline` + `ColumnTransformer` (one-hot encoding + standard scaling) feeding a `GradientBoostingRegressor` |
| Evaluation plots | matplotlib (evaluation script only) |
| Tests | Python standard-library `unittest` (no extra dependencies) |

Application runtime dependencies, listed by name in `requirements.txt`: `flask`, `numpy`, `pandas`, `scikit-learn`.

## Dataset

`laptop_data.csv` (project root) contains **1,303 raw laptop listings** with columns such as Company, TypeName, Inches, ScreenResolution, Ram, Cpu, Gpu, Memory, OpSys, Weight and Price, plus an exported row-index column. These are historical listings from an older market snapshot — not live or current pricing.

Cleaning and preparation (implemented in `scripts/evaluate_model.py::build_frame`, mirroring the training notebook):

1. Drop the exported index column (`Unnamed: 0`).
2. **Remove exact duplicate rows** — 1,303 → 1,274 rows (29 duplicates removed).
3. Parse unit-bearing strings: `Ram` `"8GB"` → `RAM_GB = 8`; `Weight` `"1.37kg"` → `Weight_kg = 1.37`.
4. Extract the resolution (`"1920x1080"` → `Resolution_X` / `Resolution_Y`) and derive `Touchscreen` and `IPS` flags from the same string.
5. Parse `Memory` strings such as `"128GB SSD +  1TB HDD"` into SSD / HDD / Flash / Hybrid capacities in GB, using the decimal TB → 1000 GB convention that the training notebook applied.
6. Parse `Cpu` strings into brand, family tier (Core i3 / i5 / i7 / i9 priority order) and clock frequency; parse `Gpu` into brand and series — with `UHD` checked **before** `HD` so "Intel UHD Graphics 620" is never misread as series HD.
7. Derive `Pixels` (= Resolution_X × Resolution_Y) and `PPI` (= √(X² + Y²) / inches).
8. Apply `log1p` to the `Price` target for training; at inference `expm1` converts predictions back to rupees (clipped at ≥ 0).

## Feature engineering and the prediction pipeline

The fitted artifact `models/laptop_price_pipeline.pkl` is a scikit-learn `Pipeline` that one-hot encodes the categorical features, standard-scales the numeric features (74 transformed columns in total), and feeds them to a `GradientBoostingRegressor`. It expects exactly these **21 input features**:

- **Categorical (7):** Company, TypeName, OpSys, CPU_Brand, CPU_Tier, GPU_Brand, GPU_Series
- **Numeric (14):** Inches, RAM_GB, Weight_kg, Resolution_X, Resolution_Y, Touchscreen, IPS, Pixels, PPI, CPU_Freq_GHz, SSD_GB, HDD_GB, Flash_GB, Hybrid_GB

`src/feature_engineering.py` converts raw form values into that exact schema (column order enforced) and contains **no fitting and no hand-written category dictionaries** — all categorical encoding lives inside the fitted pipeline. Because the Flask app imports the same module, training-time and serving-time transformations cannot drift apart; the module mirrors `notebooks/laptop_price_analysis.ipynb` exactly.

## Model choice and evaluation results

- **Model:** `Pipeline(preprocess + GradientBoostingRegressor)`, chosen by 5-fold cross-validation on the training split. Best parameters (recorded in `models/model_metadata.json`): `learning_rate=0.1`, `max_depth=3`, `n_estimators=500`.
- **Protocol:** a deterministic 80/20 hold-out — `train_test_split(test_size=0.20, random_state=42)` → 1,019 train rows and 255 test rows. No tuning was performed on the test set, and both evaluation scripts verify the model file's SHA-256 before and after each run to prove the artifact was never modified.

### Held-out test-set metrics (n = 255, rupees)

| Metric | Value |
| --- | --- |
| MAE | ₹11,262.96 (≈ ₹11,263) |
| RMSE | ₹20,699.69 (≈ ₹20,700) |
| R² | 0.750 |
| MAPE | 16.19 % |
| Median absolute percentage error (MedAPE) | 10.54 % |
| Predictions within ±10 % / ±20 % / ±30 % | 47.1 % / 75.3 % / 87.5 % |

These numbers are results on a held-out test set — **not a guarantee of accuracy for any individual laptop**. For reference, the 5-fold cross-validation results on the training split (MAE ≈ ₹9,154, RMSE ≈ ₹14,295, R² ≈ 0.843, MAPE ≈ 15.11 %) are recorded in `models/model_metadata.json` and are consistent with the hold-out numbers.

### Where the model is weaker (from `reports/model_evaluation/`)

- **Expensive laptops:** the ₹200k+ price band contains only 4 test samples (flagged `small_group_warning=True`) with MAE ≈ ₹70,300 — aggregate metrics are not reliable for this segment.
- **Rare brands:** Razer (3 test rows), Xiaomi (3), Microsoft (1) and similar brands are flagged as small groups; Razer's median absolute error exceeds 100 %. High-volume brands behave much better (e.g. HP n=57, MAE ≈ ₹8,629; Dell n=51, MAE ≈ ₹8,875).
- **Laptop type:** Workstations (n=9, flagged) and Gaming laptops (n=48, MAE ≈ ₹21,384) show larger absolute errors than mass-market Notebooks (n=145, MAE ≈ ₹5,624) — see `metrics_by_type.csv`.

### Feature-importance audit (read-only)

`reports/model_evaluation/feature_importance.json` reports two global importance estimates aggregated back to the 21 original features. Both methods agree on the top drivers: **RAM_GB** (impurity importance 0.46 of 1.0; shuffling it on the test set worsens MAE by ≈ ₹5,337 ± 777), **SSD_GB** (≈ +₹2,873 ± 584) and **CPU_Tier** (≈ +₹2,592 ± 659). Importance describes what the model relies on, not what makes laptops expensive; the derived `Pixels` and `PPI` columns are explicitly flagged in the report because they are exact functions of resolution and screen size.

## Getting started (Windows)

Requirements: Python 3.12 (the version this project and the model metadata were verified with) plus the packages in `requirements.txt`. From the project root:

```powershell
cd "C:\Users\Farha Sultana\OneDrive\Desktop\LaptopPricePrediction"
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000/> in a browser; the readiness check is available at <http://127.0.0.1:5000/health>. The application runs **locally only** — it is not deployed online, and the development server it starts is intended for local use, not production hosting.

> **Note on regenerating reports:** `scripts/evaluate_model.py` additionally imports matplotlib for its diagnostic plots. matplotlib is *not* listed in `requirements.txt`; install it separately (`python -m pip install matplotlib`) if you want to rewrite the reports. Running the Flask app and the test suite does not need it.

## Running the automated tests

From the project root (with the virtual environment active):

```powershell
python -m unittest discover -s tests -v
```

No third-party test runner is needed — the suite uses the standard library `unittest` only. What it covers:

- `tests/test_feature_engineering.py` — RAM/weight/resolution/memory parsing (including the TB → 1000 GB rule), CPU and GPU extraction (with the UHD-before-HD priority), yes/no flags, and the exact 21-column schema with derived `PPI`/`Pixels` values.
- `tests/test_app.py` — the `/health` payload, page rendering, a realistic prediction that must return a finite positive rupee price, and malformed inputs (bad inches, resolution, weight; empty post; all-zero storage) that must produce friendly error messages with no traceback.

## Reproducing the evaluation reports

Both scripts are read-only with respect to the model and the dataset — they (re)write only files under `reports/model_evaluation/`. Run them from the project root:

```powershell
# Held-out evaluation: metrics, per-segment CSVs, residual plots, summary.json
python scripts/evaluate_model.py

# Feature-importance audit: impurity + permutation importance for all 21 features
python scripts/analyze_feature_importance.py
```

Outputs already generated under `reports/model_evaluation/` include `summary.json`, `metrics_by_price_band.csv`, `metrics_by_brand.csv`, `metrics_by_type.csv`, `largest_10_errors.csv`, `test_predictions.csv`, three PNG plots and `feature_importance.json`.

## Project structure

```
LaptopPricePrediction/
├── app.py                              # Flask app: routes / and /health
├── src/
│   ├── __init__.py
│   └── feature_engineering.py          # shared 21-feature parsing & validation
├── models/
│   ├── laptop_price_pipeline.pkl       # fitted Pipeline used by the app
│   └── model_metadata.json             # training metadata, parameters, metrics
├── scripts/
│   ├── evaluate_model.py               # held-out evaluation harness
│   └── analyze_feature_importance.py   # read-only feature-importance audit
├── tests/
│   ├── test_app.py                     # Flask route and smoke tests
│   └── test_feature_engineering.py     # feature-engineering unit tests
├── notebooks/
│   └── laptop_price_analysis.ipynb     # EDA + training notebook
├── reports/
│   └── model_evaluation/               # generated metrics, CSVs, plots, importance JSON
├── templates/
│   └── index.html                      # prediction form UI
├── laptop_data.csv                     # dataset (1,303 raw rows)
├── requirements.txt
├── .gitignore
└── README.md
```

Legacy artifacts kept for provenance but **not used by the application**: the root-level `laptop_price_model.pkl` (superseded by `models/laptop_price_pipeline.pkl`) and the `main.ipynb/` scratch directory (superseded by `notebooks/laptop_price_analysis.ipynb`).

## Limitations and responsible use

- **Historical data:** the model learned from past listings; prices of current or newly released laptops may differ substantially. The UI repeats this disclaimer beside every prediction.
- **Aggregate metrics only:** MAE ≈ ₹11.3 k and MAPE ≈ 16.2 % describe average behaviour on 255 historical test rows. Individual predictions can be off by considerably more, especially in under-represented segments (expensive machines, rare brands, Workstations).
- **Small evaluation samples:** any per-brand or per-band row flagged `small_group_warning=True` (fewer than 10 test rows) is directional at best — see the CSVs in `reports/model_evaluation/`.
- **Associations, not causes:** feature importances describe what the model uses, not what makes laptops expensive; correlated and derived features (Pixels, PPI, resolution) share importance structurally.
- **No uncertainty estimate:** the app returns a single point estimate rounded to whole rupees — not a confidence or prediction interval.
- **Static model:** the pipeline is a frozen artifact trained on 2026-10-08 (per `models/model_metadata.json`); it does not retrain or update as the market moves.

Treat the output as an informative historical-data estimate for learning and portfolio purposes — not as retail, purchasing or financial advice.

## Future improvements (planned — not yet implemented)

- Local per-prediction attribution (e.g. SHAP TreeExplainer) so users can see which features drove a specific estimate.
- Prediction intervals instead of point estimates.
- Refreshing the training data with more recent listings and periodic retraining.
- Continuous integration for the `unittest` suite and the evaluation scripts.
- A production deployment story (none exists today — the app currently runs locally only).

## Contact

**Ayesha Firdous** — Computer Science Engineering Student

For queries or feedback: ayeshafirdous0509@gmail.com