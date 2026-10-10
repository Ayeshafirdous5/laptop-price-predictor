from flask import Flask, render_template, request, jsonify
from pathlib import Path
import math
import pickle

import numpy as np

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FeatureEngineeringError,
    build_feature_frame,
    compose_memory,
    parse_capacity,
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "laptop_price_pipeline.pkl"

MODEL_LOAD_ERROR = None
TARGET_TRANSFORM = "log1p"
SCHEMA_OK = False

try:
    with open(MODEL_PATH, "rb") as model_file:
        PIPELINE = pickle.load(model_file)
except FileNotFoundError as exc:
    PIPELINE = None
    MODEL_LOAD_ERROR = f"Prediction model is unavailable: {exc.filename or MODEL_PATH.name}."
except (OSError, ValueError, TypeError, pickle.UnpicklingError) as exc:
    PIPELINE = None
    MODEL_LOAD_ERROR = "Prediction model is unavailable."
    app.logger.exception("Failed to load laptop price pipeline: %s", exc)

if PIPELINE is not None:
    SCHEMA_OK = list(getattr(PIPELINE, "feature_names_in_", [])) == list(FEATURE_COLUMNS)

# Dropdown values use representative dataset-derived options. They are labels
# only; the fitted pipeline encodes every category internally.
companies = ["Acer", "Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google", "HP",
             "Huawei", "LG", "Lenovo", "MSI", "Mediacom", "Microsoft", "Razer",
             "Samsung", "Toshiba", "Vero", "Xiaomi"]
typenames = ["2 in 1 Convertible", "Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"]
cpus = [
    "Intel Core i3 2.0GHz",
    "Intel Core i5 7200U 2.5GHz",
    "Intel Core i5 8250U 1.6GHz",
    "Intel Core i7 7700HQ 2.8GHz",
    "Intel Core i7 8550U 1.8GHz",
    "Intel Celeron Dual Core N3060 1.6GHz",
    "Intel Pentium Quad Core N4200 1.1GHz",
    "AMD Ryzen 5 2500U 2.0GHz",
    "AMD A9-Series 9420 2.9GHz",
]
gpus = [
    "Intel HD Graphics 620",
    "Intel UHD Graphics 620",
    "Intel Iris Plus Graphics 640",
    "Nvidia GeForce GTX 1050",
    "Nvidia GeForce GTX 1060",
    "Nvidia Quadro M2200",
    "AMD Radeon Pro 455",
    "AMD FirePro W5130M",
    "AMD R4 Graphics",
]
oses = ["Android", "Chrome OS", "Linux", "Mac OS X", "No OS", "Windows 10",
        "Windows 10 S", "Windows 7", "macOS"]
rams = ["2", "4", "6", "8", "12", "16", "24", "32", "64"]
storages = ["0", "8", "16", "32", "64", "128", "180", "240", "256", "500",
            "508", "512", "1000", "1024", "2000"]


def _missing_required_fields(form_data):
    required = ("company", "typename", "inches", "touchscreen", "ips",
                "resolution", "cpu", "ram", "gpu", "os", "weight")
    return [name for name in required if not str(form_data.get(name, "") or "").strip()]


def _storage_capacities(form_data):
    """Map submitted storage controls onto GB capacities without inventing data.

    The legacy form submits HDD/SSD dropdown numbers plus optional
    hybrid/flash checkboxes. The newer expected fields are explicit capacity
    inputs named flash_gb/hybrid_gb. A checked box moves the already-supplied
    HDD or SSD capacity into the corresponding special storage type rather
    than creating a fictitious value.
    """
    hdd = parse_capacity(form_data["hdd"], "HDD")
    ssd = parse_capacity(form_data["ssd"], "SSD")
    raw_flash = str(form_data.get("flash_gb", "") or "").strip()
    raw_hybrid = str(form_data.get("hybrid_gb", "") or "").strip()
    flash_checked = form_data.get("flash") == "on"
    hybrid_checked = form_data.get("hybrid") == "on"

    flash_gb = parse_capacity(raw_flash, "Flash Storage") if raw_flash else 0.0
    hybrid_gb = parse_capacity(raw_hybrid, "Hybrid") if raw_hybrid else 0.0

    if flash_gb == 0 and flash_checked:
        if ssd > 0:
            flash_gb, ssd = ssd, 0.0
        elif hdd > 0:
            flash_gb, hdd = hdd, 0.0

    if hybrid_gb == 0 and hybrid_checked:
        if hdd > 0:
            hybrid_gb, hdd = hdd, 0.0
        elif ssd > 0:
            hybrid_gb, ssd = ssd, 0.0

    return ssd, hdd, flash_gb, hybrid_gb


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        form_data = request.form

        if PIPELINE is None:
            prediction = f"Error: {MODEL_LOAD_ERROR or 'Prediction model is unavailable.'}"
        elif not SCHEMA_OK:
            prediction = "Error: Prediction model is unavailable."
        else:
            try:
                missing = _missing_required_fields(form_data)
                if missing:
                    raise FeatureEngineeringError(
                        "Please fill in all fields (missing: "
                        + ", ".join(missing) + ")."
                    )

                ssd, hdd, flash_gb, hybrid_gb = _storage_capacities(form_data)
                memory = compose_memory(ssd, hdd, flash_gb, hybrid_gb)

                input_frame = build_feature_frame(
                    company=form_data.get("company", ""),
                    typename=form_data.get("typename", ""),
                    opsys=form_data.get("os", ""),
                    inches=form_data.get("inches", ""),
                    ram=form_data.get("ram", ""),
                    weight=form_data.get("weight", ""),
                    resolution=form_data.get("resolution", ""),
                    touchscreen=form_data.get("touchscreen", ""),
                    ips=form_data.get("ips", ""),
                    cpu=form_data.get("cpu", ""),
                    gpu=form_data.get("gpu", ""),
                    memory=memory,
                )

                log_price = float(PIPELINE.predict(input_frame)[0])
                price = float(np.expm1(log_price)) if TARGET_TRANSFORM == "log1p" else log_price
                if not math.isfinite(price) or price <= 0:
                    raise ValueError(f"Non-positive prediction: {price!r}")
                # Historical-data estimate, rounded to whole rupees.
                prediction = int(round(price))
            except FeatureEngineeringError as exc:
                prediction = f"Error: {exc}"
            except Exception as exc:  # never leak tracebacks to the page
                app.logger.exception("Prediction failed: %s", exc)
                prediction = "Error: Could not compute a prediction for those inputs."

    return render_template('index.html',
                           companies=companies,
                           typenames=typenames,
                           cpus=cpus,
                           gpus=gpus,
                           oses=oses,
                           rams=rams,
                           storages=storages,
    prediction=prediction)


@app.route('/health', methods=['GET'])
def health():
    ok = PIPELINE is not None and SCHEMA_OK
    return jsonify(
        status="ok" if ok else "degraded",
        model_loaded=PIPELINE is not None,
        schema_matches=bool(SCHEMA_OK),
        feature_count=len(FEATURE_COLUMNS),
        target_transformation=TARGET_TRANSFORM,
    )


if __name__ == '__main__':
    app.run(debug=False)