"""Unit tests for src/feature_engineering.py (stdlib unittest -- no pytest needed).

Run from the project root:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

# Make the project root importable regardless of how the tests are launched.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    FeatureEngineeringError,
    build_feature_frame,
    compose_memory,
    cpu_features,
    gpu_features,
    parse_capacity,
    parse_memory,
    parse_ram,
    parse_resolution,
    parse_weight,
    parse_yes_no,
)


class TestRamParsing(unittest.TestCase):
    def test_gb_string(self):
        self.assertEqual(parse_ram("8GB"), 8)

    def test_gb_string_with_space(self):
        self.assertEqual(parse_ram(" 16 GB "), 16)

    def test_plain_number_string(self):
        self.assertEqual(parse_ram("32"), 32)

    def test_int_passthrough(self):
        self.assertEqual(parse_ram(64), 64)

    def test_invalid_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            parse_ram("lots")
        with self.assertRaises(FeatureEngineeringError):
            parse_ram("0")
        with self.assertRaises(FeatureEngineeringError):
            parse_ram("")


class TestWeightParsing(unittest.TestCase):
    def test_kg_string(self):
        self.assertAlmostEqual(parse_weight("1.37kg"), 1.37)

    def test_plain_number_string(self):
        self.assertAlmostEqual(parse_weight("1.86"), 1.86)

    def test_float_passthrough(self):
        self.assertAlmostEqual(parse_weight(2.5), 2.5)

    def test_invalid_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            parse_weight("heavy")
        with self.assertRaises(FeatureEngineeringError):
            parse_weight("-1")
        with self.assertRaises(FeatureEngineeringError):
            parse_weight("999")


class TestResolutionParsing(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(parse_resolution("1920x1080"), (1920, 1080))

    def test_4k(self):
        self.assertEqual(parse_resolution("3840x2160"), (3840, 2160))

    def test_case_insensitive_x(self):
        self.assertEqual(parse_resolution("1920X1080"), (1920, 1080))

    def test_garbage_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            parse_resolution("full hd please")

    def test_empty_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            parse_resolution("")


class TestStorageParsing(unittest.TestCase):
    """The dataset's real Memory formats, incl. TB -> 1000 GB convention."""

    def test_single_ssd(self):
        caps = parse_memory("256GB SSD")
        self.assertEqual(caps, {"SSD_GB": 256.0, "HDD_GB": 0.0,
                                "Flash_GB": 0.0, "Hybrid_GB": 0.0})

    def test_ssd_plus_hdd_combo(self):
        caps = parse_memory("128GB SSD +  1TB HDD")
        self.assertEqual(caps["SSD_GB"], 128.0)
        self.assertEqual(caps["HDD_GB"], 1000.0)   # TB -> 1000 GB (notebook rule)

    def test_single_tb_hdd(self):
        self.assertEqual(parse_memory("1TB HDD")["HDD_GB"], 1000.0)

    def test_decimal_tb_hybrid(self):
        self.assertEqual(parse_memory("1.0TB Hybrid")["Hybrid_GB"], 1000.0)

    def test_dual_ssd_sums(self):
        self.assertEqual(parse_memory("256GB SSD +  256GB SSD")["SSD_GB"], 512.0)

    def test_flash_storage_combo(self):
        caps = parse_memory("64GB Flash Storage +  1TB HDD")
        self.assertEqual(caps["Flash_GB"], 64.0)
        self.assertEqual(caps["HDD_GB"], 1000.0)

    def test_2tb_hdd(self):
        self.assertEqual(parse_memory("2TB HDD")["HDD_GB"], 2000.0)

    def test_unparseable_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            parse_memory("mystery drive")

    def test_compose_parse_roundtrip(self):
        memory = compose_memory(ssd_gb=512, hdd_gb=1000, flash_gb=0, hybrid_gb=1000)
        caps = parse_memory(memory)
        self.assertEqual(caps["SSD_GB"], 512.0)
        self.assertEqual(caps["HDD_GB"], 1000.0)
        self.assertEqual(caps["Hybrid_GB"], 1000.0)

    def test_parse_capacity(self):
        self.assertEqual(parse_capacity("512", "SSD size"), 512.0)
        self.assertEqual(parse_capacity("0", "HDD size"), 0.0)
        with self.assertRaises(FeatureEngineeringError):
            parse_capacity("abc", "SSD size")


class TestCpuParsing(unittest.TestCase):
    """Priority order must match the training notebook exactly."""

    def test_core_i7(self):
        self.assertEqual(cpu_features("Intel Core i7 7700HQ 2.8GHz"),
                         ("Intel", "Core i7", 2.8))

    def test_core_i5(self):
        self.assertEqual(cpu_features("Intel Core i5 7200U 2.5GHz"),
                         ("Intel", "Core i5", 2.5))

    def test_core_i3(self):
        self.assertEqual(cpu_features("Intel Core i3-7100U 2.4GHz"),
                         ("Intel", "Core i3", 2.4))

    def test_celeron(self):
        brand, tier, freq = cpu_features("Intel Celeron Dual Core N3060 1.6GHz")
        self.assertEqual((brand, tier), ("Intel", "Celeron"))
        self.assertAlmostEqual(freq, 1.6)

    def test_amd_ryzen(self):
        self.assertEqual(cpu_features("AMD Ryzen 5 2500U 2.0GHz"),
                         ("AMD", "AMD Ryzen", 2.0))

    def test_amd_non_ryzen_is_amd_other(self):
        self.assertEqual(cpu_features("AMD A9-Series 9420 2.9GHz"),
                         ("AMD", "AMD Other", 2.9))

    def test_other_intel_is_intel_other(self):
        self.assertEqual(cpu_features("Intel Core2 Duo T6670 2.2GHz"),
                         ("Intel", "Intel Other", 2.2))

    def test_missing_frequency_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            cpu_features("Intel Core i7")


class TestGpuParsing(unittest.TestCase):
    def test_geforce(self):
        self.assertEqual(gpu_features("Nvidia GeForce GTX 1050"), ("Nvidia", "GeForce"))

    def test_quadro(self):
        self.assertEqual(gpu_features("Nvidia Quadro M2200"), ("Nvidia", "Quadro"))

    def test_uhd_checked_before_hd(self):
        # Regression guard: 'UHD' contains 'HD' -- priority must pick UHD.
        self.assertEqual(gpu_features("Intel UHD Graphics 620"), ("Intel", "UHD"))

    def test_plain_hd(self):
        self.assertEqual(gpu_features("Intel HD Graphics 620"), ("Intel", "HD"))

    def test_iris(self):
        self.assertEqual(gpu_features("Intel Iris Plus Graphics 640"),
                         ("Intel", "Iris"))

    def test_radeon(self):
        self.assertEqual(gpu_features("AMD Radeon Pro 455"), ("AMD", "Radeon"))

    def test_firepro(self):
        self.assertEqual(gpu_features("AMD FirePro W6170M"), ("AMD", "FirePro"))

    def test_arm_mali(self):
        self.assertEqual(gpu_features("ARM Mali-T720"), ("ARM", "Mali"))

    def test_empty_raises(self):
        with self.assertRaises(FeatureEngineeringError):
            gpu_features("")


def _realistic_frame(**overrides):
    """A realistic gaming-laptop input shared by the frame tests."""
    args = dict(
        company="Dell", typename="Gaming", opsys="Windows 10",
        inches="15.6", ram="16GB", weight="2.8kg",
        resolution="1920x1080", touchscreen="No", ips="Yes",
        cpu="Intel Core i7 7700HQ 2.8GHz", gpu="Nvidia GeForce GTX 1060",
        memory="512GB SSD +  1TB HDD",
    )
    args.update(overrides)
    return build_feature_frame(**args)


class TestBuildFeatureFrame(unittest.TestCase):
    """The frame must match the fitted pipeline's exact 21-column schema."""

    def test_columns_and_order_match_pipeline_schema(self):
        frame = _realistic_frame()
        self.assertEqual(list(frame.columns), FEATURE_COLUMNS)
        self.assertEqual(len(frame.columns), 21)

    def test_engineered_values(self):
        row = _realistic_frame().iloc[0]
        self.assertEqual(row["RAM_GB"], 16)
        self.assertEqual(row["Weight_kg"], 2.8)
        self.assertEqual(row["Resolution_X"], 1920)
        self.assertEqual(row["Resolution_Y"], 1080)
        self.assertEqual(row["Pixels"], 1920 * 1080)
        # PPI is derived, never user-entered: hypot(X, Y) / Inches
        self.assertAlmostEqual(
            row["PPI"], (1920 ** 2 + 1080 ** 2) ** 0.5 / 15.6, places=6)
        self.assertEqual(row["Touchscreen"], 0)
        self.assertEqual(row["IPS"], 1)
        self.assertEqual(row["CPU_Brand"], "Intel")
        self.assertEqual(row["CPU_Tier"], "Core i7")
        self.assertAlmostEqual(row["CPU_Freq_GHz"], 2.8)
        self.assertEqual(row["GPU_Brand"], "Nvidia")
        self.assertEqual(row["GPU_Series"], "GeForce")
        self.assertEqual(row["SSD_GB"], 512.0)
        self.assertEqual(row["HDD_GB"], 1000.0)
        self.assertEqual(row["Flash_GB"], 0.0)
        self.assertEqual(row["Hybrid_GB"], 0.0)

    def test_categorical_values_are_raw_strings_not_codes(self):
        row = _realistic_frame().iloc[0]
        self.assertEqual(row["Company"], "Dell")     # no numeric mapping
        self.assertEqual(row["OpSys"], "Windows 10")

    def test_invalid_inputs_raise_user_safe_errors(self):
        for bad_field, bad_value in [
            ("inches", "abc"), ("inches", ""),
            ("resolution", "garbage"), ("weight", "heavy"),
            ("ram", "lots"), ("memory", ""),
        ]:
            with self.assertRaises(FeatureEngineeringError, msg=bad_field):
                _realistic_frame(**{bad_field: bad_value})


class TestYesNo(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_yes_no("Yes"), 1)
        self.assertEqual(parse_yes_no("No"), 0)
        with self.assertRaises(FeatureEngineeringError):
            parse_yes_no("maybe")


if __name__ == "__main__":
    unittest.main()
