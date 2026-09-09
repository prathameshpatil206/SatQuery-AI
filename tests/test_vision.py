"""
Unit test suite for the SatQuery-AI Vision & Satellite Specialist Layer.
"""

import unittest
from pathlib import Path
import numpy as np
import rasterio

from vision import (
    analyze_optical_sar,
    caption,
    create_synthetic_geotiff,
    detect_change,
    detect_modality,
    geo_to_pixel,
    ground,
    load_image,
    pixel_to_geo,
    read_geotiff,
    to_display_rgb,
    validate_spatial_pair,
    vqa,
    CaptionResult,
    ChangeResult,
    GroundingResult,
    Modality,
    OpticalSARResult,
    VQAResult,
)


class TestGeospatialEngine(unittest.TestCase):
    """Test rasterio reading, coordinate transformation, and normalization."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("sample/test_artifacts")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.tif_path = create_synthetic_geotiff(
            cls.test_dir / "test_multispectral.tif",
            width=128,
            height=128,
            bands=4,
            crs="EPSG:32632",
        )

    def test_read_geotiff(self):
        img, meta = read_geotiff(self.tif_path)
        self.assertEqual(img.shape, (4, 128, 128))
        self.assertEqual(meta["bands"], 4)
        self.assertEqual(meta["crs"], "EPSG:32632")
        self.assertTrue(meta["is_geospatial"])

    def test_coordinate_roundtrip(self):
        img, meta = read_geotiff(self.tif_path)
        transform = meta["transform"]
        orig_x, orig_y = 42, 67
        geo = pixel_to_geo(transform, orig_x, orig_y)
        pixel = geo_to_pixel(transform, geo["lon"], geo["lat"])
        self.assertAlmostEqual(pixel["x"], orig_x, delta=1)
        self.assertAlmostEqual(pixel["y"], orig_y, delta=1)

    def test_to_display_rgb(self):
        img, _ = read_geotiff(self.tif_path)
        rgb = to_display_rgb(img, bands=(0, 1, 2))
        self.assertEqual(rgb.shape, (128, 128, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreaterEqual(rgb.min(), 0)
        self.assertLessEqual(rgb.max(), 255)

    def test_load_image_numpy(self):
        arr = np.zeros((3, 64, 64), dtype=np.uint8)
        img, meta = load_image(arr)
        self.assertEqual(img.shape, (3, 64, 64))
        self.assertFalse(meta["is_geospatial"])

    def test_detect_modality(self):
        # 4 bands -> multispectral
        img, meta = read_geotiff(self.tif_path)
        self.assertEqual(detect_modality(img, meta), Modality.MULTISPECTRAL)
        # 1 band float -> SAR
        sar_arr = np.random.randn(1, 64, 64).astype(np.float32)
        self.assertEqual(detect_modality(sar_arr), Modality.SAR)

    def test_spatial_pair_validation(self):
        img, meta1 = read_geotiff(self.tif_path)
        meta2 = dict(meta1)
        report = validate_spatial_pair(meta1, meta2)
        self.assertTrue(report["compatible"])
        self.assertTrue(report["crs_match"])
        self.assertTrue(report["bounds_overlap"])


class TestDetectionSpecialists(unittest.TestCase):
    """Test single-image specialists: VQA, captioning, grounding."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("sample/test_artifacts")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.tif_path = create_synthetic_geotiff(
            cls.test_dir / "test_detection.tif",
            width=128,
            height=128,
            bands=4,
            crs="EPSG:32632",
        )

    def test_vqa(self):
        res = vqa(self.tif_path, "What is the dominant land cover?")
        self.assertIsInstance(res, VQAResult)
        self.assertTrue(len(res.answer) > 0)
        self.assertGreater(res.confidence, 0.0)
        self.assertIsNotNone(res.visual_evidence)
        self.assertIsNotNone(res.execution_trace)

    def test_caption(self):
        res = caption(self.tif_path)
        self.assertIsInstance(res, CaptionResult)
        self.assertTrue(len(res.caption) > 0)
        self.assertGreater(res.confidence, 0.0)
        self.assertTrue(len(res.detected_features) > 0)

    def test_ground(self):
        res = ground(self.tif_path, "water body")
        self.assertIsInstance(res, GroundingResult)
        self.assertEqual(res.target_query, "water body")
        self.assertIsNotNone(res.visual_evidence)


class TestChangeDetection(unittest.TestCase):
    """Test bi-temporal change detection."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("sample/test_artifacts")
        cls.t1_path = create_synthetic_geotiff(cls.test_dir / "t1_test.tif", width=128, height=128)
        cls.t2_path = create_synthetic_geotiff(cls.test_dir / "t2_test.tif", width=128, height=128)

        # Alter T2 to simulate real modification
        with rasterio.open(cls.t2_path, "r+") as dst:
            arr = dst.read()
            arr[:, 40:80, 40:80] = 3900
            dst.write(arr)

    def test_detect_change(self):
        res = detect_change(self.t1_path, self.t2_path, prompt="Check deforestation")
        self.assertIsInstance(res, ChangeResult)
        self.assertGreater(res.change_ratio, 0.0)
        self.assertIsNotNone(res.change_mask)
        self.assertIsNotNone(res.difference_heatmap)
        self.assertTrue(len(res.summary) > 0)
        self.assertIn("Check deforestation", res.summary)


class TestOpticalSARAnalysis(unittest.TestCase):
    """Test paired Optical + SAR analysis."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("sample/test_artifacts")
        cls.opt_path = create_synthetic_geotiff(cls.test_dir / "opt_test.tif", width=128, height=128, bands=4)
        cls.sar_path = cls.test_dir / "sar_test.tif"

        with rasterio.open(cls.opt_path) as src:
            meta = src.meta.copy()
            meta.update({"count": 1, "dtype": "float32"})
            data = np.random.exponential(scale=20.0, size=(1, 128, 128)).astype(np.float32)
            with rasterio.open(cls.sar_path, "w", **meta) as dst:
                dst.write(data)

    def test_analyze_optical_sar(self):
        res = analyze_optical_sar(self.opt_path, self.sar_path, query="Detect flooded areas")
        self.assertIsInstance(res, OpticalSARResult)
        self.assertTrue(len(res.analysis_summary) > 0)
        self.assertTrue(len(res.complementary_insights) > 0)
        self.assertEqual(res.visual_evidence.evidence_type, "composite")
        self.assertEqual(res.visual_evidence.preview_rgb.shape, (128, 128, 3))


if __name__ == "__main__":
    unittest.main()
