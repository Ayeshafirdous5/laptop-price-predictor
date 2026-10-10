"""Smoke tests for the Flask integration (stdlib unittest -- no pytest needed).

Covers: health endpoint, a realistic prediction, pipeline smoke test
(finite positive rupee price) and malformed-input handling (user-friendly
errors, never an unhandled traceback/500).

Run from the project root:
    python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402  (loads pipeline once at import)
from src.feature_engineering import build_feature_frame  # noqa: E402

# A realistic gaming laptop as the HTML form would submit it.
REALISTIC_FORM = {
    "company": "Dell",
    "typename": "Gaming",
    "inches": "15.6",
    "touchscreen": "No",
    "ips": "Yes",
    "resolution": "1920x1080",
    "cpu": "Intel Core i7 7700HQ 2.8GHz",
    "ram": "16",
    "gpu": "Nvidia GeForce GTX 1060",
    "os": "Windows 10",
    "weight": "2.8",
    "hdd": "1000",
    "ssd": "512",
    "flash_gb": "0",
    "hybrid_gb": "0",
}


class TestPipelineSmoke(unittest.TestCase):
    """Raw input -> feature frame -> fitted pipeline -> finite positive INR."""

    def test_realistic_input_predicts_finite_positive_price(self):
        frame = build_feature_frame(
            company="Dell", typename="Gaming", opsys="Windows 10",
            inches="15.6", ram="16GB", weight="2.8kg",
            resolution="1920x1080", touchscreen="No", ips="Yes",
            cpu="Intel Core i7 7700HQ 2.8GHz", gpu="Nvidia GeForce GTX 1060",
            memory="512GB SSD +  1TB HDD",
        )
        self.assertEqual(len(frame.columns), 21)
        self.assertIsNotNone(app_module.PIPELINE, app_module.MODEL_LOAD_ERROR)

        log_price = float(app_module.PIPELINE.predict(frame)[0])
        price = (math.expm1(log_price)
                 if app_module.TARGET_TRANSFORM == "log1p" else log_price)

        self.assertTrue(math.isfinite(price))
        self.assertGreater(price, 0)
        # Sanity band: this configuration should be a plausible laptop price.
        self.assertGreater(price, 10_000)
        self.assertLess(price, 500_000)


class TestFlaskRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model_loaded"])
        self.assertTrue(payload["schema_matches"])
        self.assertEqual(payload["feature_count"], 21)
        self.assertEqual(payload["target_transformation"], "log1p")

    def test_get_index_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Laptop Price Predictor", response.data)

    def test_valid_post_returns_prediction(self):
        response = self.client.post("/", data=REALISTIC_FORM)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predicted Price", response.data)
        self.assertNotIn(b"Traceback", response.data)

    def test_invalid_inches_shows_friendly_error_not_500(self):
        payload = dict(REALISTIC_FORM, inches="abc")
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid screen size", response.data)
        self.assertNotIn(b"Traceback", response.data)

    def test_malformed_resolution_shows_friendly_error(self):
        payload = dict(REALISTIC_FORM, resolution="garbage")
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid resolution", response.data)
        self.assertNotIn(b"Traceback", response.data)

    def test_invalid_weight_shows_friendly_error(self):
        payload = dict(REALISTIC_FORM, weight="heavy")
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid weight", response.data)

    def test_empty_post_shows_friendly_error(self):
        response = self.client.post("/", data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"fill in all fields", response.data)
        self.assertNotIn(b"Traceback", response.data)

    def test_all_storage_zero_shows_friendly_error(self):
        payload = dict(REALISTIC_FORM, ssd="0", hdd="0")
        response = self.client.post("/", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid storage", response.data)


if __name__ == "__main__":
    unittest.main()
