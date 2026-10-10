"""Deterministic feature engineering for the laptop price pipeline.

Every function here mirrors ``notebooks/laptop_price_analysis.ipynb`` EXACTLY --
the fitted artifact ``models/laptop_price_pipeline.pkl`` was trained on the
output of this logic, so the parsing rules (priorities, units, conventions) are
compatibility-critical:

* RAM ``"8GB"``              -> ``8``   (int)
* Weight ``"1.37kg"``        -> ``1.37`` (float)
* Resolution ``"1920x1080"`` -> ``Resolution_X/Y`` (+ Pixels, PPI from Inches)
* Memory ``"128GB SSD +  1TB HDD"`` -> SSD/HDD/Flash/Hybrid capacities in GB
  (TB -> 1000 GB decimal convention, matching training)
* CPU string -> brand / tier / frequency (priority order as in the notebook)
* GPU string -> brand / series (UHD is checked BEFORE HD)

No LabelEncoder, no hand-written category dictionaries, no OneHotEncoder here:
categorical encoding lives INSIDE the fitted pipeline. This module only performs
deterministic, side-effect-free string/number transformations.
"""

from __future__ import annotations

import math
import re
from typing import Any, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Exact raw-input schema expected by models/laptop_price_pipeline.pkl
# (verified against pipeline.feature_names_in_ and model_metadata.json).
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    # categorical (encoded inside the pipeline)
    "Company",
    "TypeName",
    "OpSys",
    "CPU_Brand",
    "CPU_Tier",
    "GPU_Brand",
    "GPU_Series",
    # numeric
    "Inches",
    "RAM_GB",
    "Weight_kg",
    "Resolution_X",
    "Resolution_Y",
    "Touchscreen",
    "IPS",
    "Pixels",
    "PPI",
    "CPU_Freq_GHz",
    "SSD_GB",
    "HDD_GB",
    "Flash_GB",
    "Hybrid_GB",
]

CATEGORICAL_FEATURES = [
    "Company", "TypeName", "OpSys",
    "CPU_Brand", "CPU_Tier", "GPU_Brand", "GPU_Series",
]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

# Plausible input bounds (dataset spans: inches 10.1-18.4, weight 0.69-4.7 kg,
# resolution 1366x768 .. 3840x2160, RAM 2-64 GB). Bounds are deliberately wide:
# they exist to reject typos, not to restrict legitimate laptops.
_INCHES_RANGE = (5.0, 30.0)
_WEIGHT_RANGE = (0.05, 30.0)
_PIXEL_DIM_RANGE = (100, 16384)


class FeatureEngineeringError(ValueError):
    """Raised for invalid raw inputs; message is safe to show to end users."""


# ---------------------------------------------------------------------------
# Simple unit-bearing parsers
# ---------------------------------------------------------------------------
def parse_ram(value: Any) -> int:
    """``"8GB"`` / ``"8 GB"`` / ``8`` -> ``8`` (int GB)."""
    if isinstance(value, bool):
        raise FeatureEngineeringError(f"Invalid RAM value: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(" ", "").upper().replace("GB", "")
        try:
            number = float(text)
        except ValueError as exc:
            raise FeatureEngineeringError(
                f"Invalid RAM value: {value!r} (expected e.g. '8GB' or 8)."
            ) from exc
    if not number > 0 or number > 1024:
        raise FeatureEngineeringError(
            f"Invalid RAM value: {value!r} (expected a positive number of GB)."
        )
    return int(number)


def parse_weight(value: Any) -> float:
    """``"1.37kg"`` / ``"1.37"`` / ``1.37`` -> ``1.37`` (float kg)."""
    if isinstance(value, bool):
        raise FeatureEngineeringError(f"Invalid weight value: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower().replace("kg", "").strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise FeatureEngineeringError(
                f"Invalid weight: {value!r} (expected e.g. '1.37' or '1.37kg')."
            ) from exc
    low, high = _WEIGHT_RANGE
    if not low <= number <= high:
        raise FeatureEngineeringError(
            f"Invalid weight: {value!r} (expected {low}-{high} kg)."
        )
    return number


def parse_inches(value: Any) -> float:
    """Screen size in inches, validated against a plausible range."""
    if isinstance(value, bool):
        raise FeatureEngineeringError(f"Invalid screen size: {value!r}")
    try:
        number = float(str(value).strip().replace('"', "").replace("in", ""))
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Invalid screen size: {value!r} (expected e.g. 15.6)."
        ) from exc
    low, high = _INCHES_RANGE
    if not low <= number <= high:
        raise FeatureEngineeringError(
            f"Invalid screen size: {value!r} (expected {low}-{high} inches)."
        )
    return number


def parse_resolution(value: Any) -> Tuple[int, int]:
    """``"1920x1080"`` -> ``(1920, 1080)``.

    Mirrors the notebook regex ``(\\d+)x(\\d+)`` (first match anywhere in the
    string), with case-insensitive ``x`` so "1920X1080" also works.
    """
    match = re.search(r"(\d+)\s*x\s*(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        raise FeatureEngineeringError(
            f"Invalid resolution: {value!r} (expected e.g. 1920x1080)."
        )
    x_res, y_res = int(match.group(1)), int(match.group(2))
    low, high = _PIXEL_DIM_RANGE
    if not (low <= x_res <= high and low <= y_res <= high):
        raise FeatureEngineeringError(
            f"Invalid resolution: {value!r} (components must be {low}-{high} pixels)."
        )
    return x_res, y_res


def parse_yes_no(value: Any) -> int:
    """``"Yes"`` -> 1, ``"No"`` -> 0 (the form's Touchscreen/IPS selects)."""
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return 1
    if text in {"no", "n", "false", "0"}:
        return 0
    raise FeatureEngineeringError(f"Expected Yes/No, got: {value!r}")


# ---------------------------------------------------------------------------
# Memory / storage parsing (exact notebook logic, incl. TB -> 1000 GB)
# ---------------------------------------------------------------------------
def parse_memory(mem: Any) -> dict:
    """Parse a dataset-style Memory string into per-type GB capacities.

    Handles combinations such as ``"128GB SSD +  1TB HDD"``, ``"1TB HDD"``,
    ``"1.0TB Hybrid"``, dual-SSD rows and Flash Storage. TB is converted with
    the notebook's decimal convention: 1 TB -> 1000 GB.

    Returns ``{"SSD_GB", "HDD_GB", "Flash_GB", "Hybrid_GB"}`` as floats.
    Raises FeatureEngineeringError if nothing parseable is found.
    """
    caps = {"SSD_GB": 0.0, "HDD_GB": 0.0, "Flash_GB": 0.0, "Hybrid_GB": 0.0}
    for part in str(mem).split("+"):
        part = part.strip()
        match = re.match(r"([\d.]+)\s*(TB|GB)", part, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1)) * (1000.0 if match.group(2).upper() == "TB" else 1.0)
        if "SSD" in part:
            caps["SSD_GB"] += value
        elif "HDD" in part:
            caps["HDD_GB"] += value
        elif "Flash" in part:
            caps["Flash_GB"] += value
        elif "Hybrid" in part:
            caps["Hybrid_GB"] += value
    if sum(caps.values()) <= 0:
        raise FeatureEngineeringError(
            f"Invalid storage: {mem!r} (expected e.g. '256GB SSD' or "
            f"'128GB SSD +  1TB HDD')."
        )
    return caps


def compose_memory(ssd_gb: float, hdd_gb: float, flash_gb: float,
                   hybrid_gb: float) -> str:
    """Build a dataset-style Memory string from individual capacities.

    Lets the web form (separate capacity fields) flow through the *same*
    ``parse_memory`` parser used for raw dataset values -- one code path.
    """
    def _fmt(value: float) -> str:
        number = float(value)
        return str(int(number)) if number == int(number) else str(number)

    parts = []
    if ssd_gb:
        parts.append(f"{_fmt(ssd_gb)}GB SSD")
    if hdd_gb:
        parts.append(f"{_fmt(hdd_gb)}GB HDD")
    if flash_gb:
        parts.append(f"{_fmt(flash_gb)}GB Flash Storage")
    if hybrid_gb:
        parts.append(f"{_fmt(hybrid_gb)}GB Hybrid")
    return " + ".join(parts)


def parse_capacity(value: Any, label: str) -> float:
    """Non-negative capacity in GB (form dropdown values are plain numbers)."""
    if isinstance(value, bool):
        raise FeatureEngineeringError(f"Invalid {label}: {value!r}")
    try:
        number = float(str(value).strip().replace("GB", "").replace(" ", ""))
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Invalid {label}: {value!r} (expected a number of GB)."
        ) from exc
    if number < 0 or number > 8000:
        raise FeatureEngineeringError(
            f"Invalid {label}: {value!r} (expected 0-8000 GB)."
        )
    return number


# ---------------------------------------------------------------------------
# CPU / GPU structured features (exact notebook priority order)
# ---------------------------------------------------------------------------
def cpu_brand(cpu: Any) -> str:
    """First token of the raw CPU string (Intel / AMD / Samsung ...)."""
    text = str(cpu).strip()
    if not text:
        raise FeatureEngineeringError("CPU is empty.")
    return text.split()[0]


def cpu_features(cpu: Any) -> Tuple[str, str, float]:
    """Return ``(CPU_Brand, CPU_Tier, CPU_Freq_GHz)`` from a raw CPU string.

    Priority matches the notebook exactly: Core i3/i5/i7/i9 first, then for
    Intel Celeron/Pentium/Atom else 'Intel Other'; for AMD Ryzen else
    'AMD Other'; anything else -> 'Other'. Frequency from ``<n>GHz``.
    """
    text = str(cpu).strip()
    if not text:
        raise FeatureEngineeringError("CPU is empty.")

    tier = None
    for candidate in ("Core i3", "Core i5", "Core i7", "Core i9"):
        if candidate in text:
            tier = candidate
            break
    if tier is None:
        brand = text.split()[0]
        if brand == "Intel":
            for candidate in ("Celeron", "Pentium", "Atom"):
                if candidate in text:
                    tier = candidate
                    break
            else:
                tier = "Intel Other"
        elif brand == "AMD":
            tier = "AMD Ryzen" if "Ryzen" in text else "AMD Other"
        else:
            tier = "Other"

    match = re.search(r"([\d.]+)\s*GHz", text, flags=re.IGNORECASE)
    if not match:
        raise FeatureEngineeringError(
            f"Invalid CPU: {text!r} (expected a clock speed like '2.5GHz')."
        )
    freq = float(match.group(1))
    if not 0.1 <= freq <= 6.0:
        raise FeatureEngineeringError(f"Invalid CPU frequency: {text!r}")
    return text.split()[0], tier, freq


def gpu_features(gpu: Any) -> Tuple[str, str]:
    """Return ``(GPU_Brand, GPU_Series)`` from a raw GPU string.

    Keyword priority matches the notebook: GeForce/Quadro/Radeon/FirePro/Iris
    first, then 'UHD' BEFORE 'HD' so e.g. "Intel UHD Graphics 620" is never
    mis-read as series 'HD'. Anything else -> 'Other'.
    """
    text = str(gpu).strip()
    if not text:
        raise FeatureEngineeringError("GPU is empty.")
    brand = text.split()[0]
    if "GeForce" in text or text.startswith("GTX"):
        series = "GeForce"
    elif "Quadro" in text:
        series = "Quadro"
    elif "Radeon" in text:
        series = "Radeon"
    elif "FirePro" in text:
        series = "FirePro"
    elif "Iris" in text:
        series = "Iris"
    elif "UHD" in text:
        series = "UHD"
    elif "HD" in text:
        series = "HD"
    elif "Mali" in text:
        series = "Mali"
    else:
        series = "Other"
    return brand, series


# ---------------------------------------------------------------------------
# The 21-feature frame
# ---------------------------------------------------------------------------
def build_feature_frame(
    *,
    company: Any,
    typename: Any,
    opsys: Any,
    inches: Any,
    ram: Any,
    weight: Any,
    resolution: Any,
    touchscreen: Any,
    ips: Any,
    cpu: Any,
    gpu: Any,
    memory: Any,
) -> pd.DataFrame:
    """Convert raw laptop form values into the exact 21-feature DataFrame.

    The returned DataFrame has precisely the columns (and order) in
    ``FEATURE_COLUMNS`` -- what ``pipeline.predict`` was fitted on.
    All parsing/validation errors raise :class:`FeatureEngineeringError` with
    an end-user-safe message.
    """
    company_s = str(company or "").strip()
    typename_s = str(typename or "").strip()
    opsys_s = str(opsys or "").strip()
    if not company_s or not typename_s or not opsys_s:
        missing = [n for n, v in (("Company", company_s), ("Type", typename_s),
                                  ("Operating system", opsys_s)) if not v]
        raise FeatureEngineeringError(
            f"Missing required field(s): {', '.join(missing)}."
        )

    inches_f = parse_inches(inches)
    ram_gb = parse_ram(ram)
    weight_f = parse_weight(weight)
    x_res, y_res = parse_resolution(resolution)
    touch = parse_yes_no(touchscreen)
    ips_flag = parse_yes_no(ips)
    cpu_b, cpu_t, cpu_f = cpu_features(cpu)
    gpu_b, gpu_s = gpu_features(gpu)
    storage = parse_memory(memory)

    row = {
        "Company": company_s,
        "TypeName": typename_s,
        "OpSys": opsys_s,
        "CPU_Brand": cpu_b,
        "CPU_Tier": cpu_t,
        "GPU_Brand": gpu_b,
        "GPU_Series": gpu_s,
        "Inches": inches_f,
        "RAM_GB": ram_gb,
        "Weight_kg": weight_f,
        "Resolution_X": x_res,
        "Resolution_Y": y_res,
        "Touchscreen": touch,
        "IPS": ips_flag,
        "Pixels": x_res * y_res,
        # PPI is always DERIVED from resolution + screen size, never user-entered.
        "PPI": math.hypot(x_res, y_res) / inches_f,
        "CPU_Freq_GHz": cpu_f,
        "SSD_GB": storage["SSD_GB"],
        "HDD_GB": storage["HDD_GB"],
        "Flash_GB": storage["Flash_GB"],
        "Hybrid_GB": storage["Hybrid_GB"],
    }
    # columns= enforces the exact schema/order the fitted pipeline expects.
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
