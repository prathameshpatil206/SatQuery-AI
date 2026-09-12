"""
Unit tests for the scientifically robust remote-sensing water detection module.
Verifies raw spectral NDWI calculations, suppression of dark forest false positives,
band mapping resolution, and API contracts.
"""

import unittest
from pathlib import Path
import numpy as np
import rasterio

from vision.geospatial import (
    create_synthetic_geotiff,
    get_spectral_band_mapping,
    load_image,
    read_geotiff,
)
from vision.detection import (
    BaselineRemoteSensingSpecialist,
    detect_water_candidates,
    ground,
    run_detection,
    vqa,
)


class TestWaterDetection(unittest.TestCase):
    """Test suite for physical water detection and forest rejection."""

    def test_sample_tiffs_water_and_forest(self):
        """
        Verify on sample TIFFs (valid_sample.tif, t1.tif, t2.tif):
        1. Water patch [50:100, 50:150] is detected.
        2. Forest patch [120:200, 100:220] has ZERO water false positives.
        """
        samples = ["sample/valid_sample.tif", "sample/t1.tif", "sample/t2.tif"]

        for s_path in samples:
            if not Path(s_path).exists():
                continue

            img, meta = load_image(s_path)
            boxes, debug = detect_water_candidates(img, metadata=meta, return_debug=True)

            print(f"\n[TEST VALIDATION] {s_path}")
            print(f"  Method: {debug['detector_method']}")
            print(f"  Band Mapping: {debug['band_mapping_used']['method']}")
            print(f"  Detected Water Coverage: {debug['detected_water_pixel_percentage']:.2f}%")
            print(f"  NDWI Range: [{debug['ndwi_stats']['min']:.3f}, {debug['ndwi_stats']['max']:.3f}] (Mean: {debug['ndwi_stats']['mean']:.3f})")
            print(f"  Confidence: {debug['confidence']}")
            print(f"  Detected Boxes: {boxes}")

            # Water region must be detected
            self.assertGreater(len(boxes), 0, f"Expected water box in {s_path}")
            w_box = boxes[0]
            # Check bounding box matches expected water patch [50, 50, 101, 151]
            self.assertAlmostEqual(w_box[0], 50.0, delta=5.0)
            self.assertAlmostEqual(w_box[1], 50.0, delta=5.0)
            self.assertAlmostEqual(w_box[2], 100.0, delta=5.0)
            self.assertAlmostEqual(w_box[3], 150.0, delta=5.0)

            # Forest region [120:200, 100:220] must NOT be inside any water box
            for b in boxes:
                ymin, xmin, ymax, xmax = b
                # Overlap with forest rectangle
                f_ymin, f_xmin, f_ymax, f_xmax = 120, 100, 200, 220
                inter_ymin = max(ymin, f_ymin)
                inter_xmin = max(xmin, f_xmin)
                inter_ymax = min(ymax, f_ymax)
                inter_xmax = min(xmax, f_xmax)

                has_overlap = inter_ymin < inter_ymax and inter_xmin < inter_xmax
                self.assertFalse(
                    has_overlap,
                    f"Water box {b} overlaps with forest patch in {s_path}!"
                )

    def test_pure_forest_no_water_scene(self):
        """
        A scene containing ONLY dense forest and land (no water) must return 0 boxes
        and NOT falsely detect dark vegetation as water.
        """
        np.random.seed(123)
        img = np.random.randint(200, 3500, size=(4, 256, 256), dtype=np.uint16)
        # Dense dark forest with strong chlorophyll absorption (low Red, high NIR):
        img[0, 40:220, 40:220] = 300   # Red
        img[1, 40:220, 40:220] = 700   # Green
        img[2, 40:220, 40:220] = 350   # Blue
        img[3, 40:220, 40:220] = 3400  # NIR (high mesophyll reflectance)

        boxes, debug = detect_water_candidates(img, return_debug=True)
        self.assertEqual(len(boxes), 0, f"Expected 0 water boxes on pure forest, got {boxes}")
        self.assertEqual(debug["detector_method"], "raw_multispectral_ndwi")
        self.assertEqual(debug["detected_water_pixel_percentage"], 0.0)
        self.assertEqual(debug["confidence"], 0.92)

    def test_get_spectral_band_mapping(self):
        """Verify band mapping resolver across all modalities and metadata formats."""
        arr4 = np.zeros((4, 64, 64), dtype=np.uint16)

        # 1. Custom mapping
        m_custom = get_spectral_band_mapping(arr4, custom_mapping={"green": 2, "nir": 1})
        self.assertEqual(m_custom["green_band_index"], 2)
        self.assertEqual(m_custom["nir_band_index"], 1)
        self.assertEqual(m_custom["method"], "custom_mapping")
        self.assertTrue(m_custom["is_confident"])

        # 2. Metadata descriptions (Sentinel-2 style)
        s2_meta = {"descriptions": ["B02 - Blue", "B03 - Green", "B04 - Red", "B08 - NIR"]}
        m_s2 = get_spectral_band_mapping(arr4, metadata=s2_meta)
        self.assertEqual(m_s2["blue_band_index"], 0)
        self.assertEqual(m_s2["green_band_index"], 1)
        self.assertEqual(m_s2["red_band_index"], 2)
        self.assertEqual(m_s2["nir_band_index"], 3)
        self.assertEqual(m_s2["method"], "metadata_descriptions")
        self.assertTrue(m_s2["is_confident"])

        # 3. Standard 4-band package default
        m_def = get_spectral_band_mapping(arr4)
        self.assertEqual(m_def["green_band_index"], 1)
        self.assertEqual(m_def["nir_band_index"], 3)
        self.assertEqual(m_def["method"], "standard_4band_default")
        self.assertTrue(m_def["is_confident"])

        # 4. Standard 3-band RGB
        arr3 = np.zeros((3, 64, 64), dtype=np.uint8)
        m_rgb = get_spectral_band_mapping(arr3)
        self.assertEqual(m_rgb["green_band_index"], 1)
        self.assertIsNone(m_rgb["nir_band_index"])
        self.assertEqual(m_rgb["method"], "standard_rgb")
        self.assertFalse(m_rgb["is_confident"])

    def test_conservative_rgb_fallback(self):
        """Verify that 3-band RGB fallback rejects dark vegetation and only flags blue water."""
        # 3-band image (RGB)
        rgb_img = np.zeros((3, 100, 100), dtype=np.uint8)
        # Forest region: dark green (Green > Blue and Green > Red)
        rgb_img[0, 10:40, 10:40] = 30   # R
        rgb_img[1, 10:40, 10:40] = 70   # G
        rgb_img[2, 10:40, 10:40] = 35   # B

        # Blue water region: Blue dominates over Red and Green
        rgb_img[0, 60:90, 60:90] = 20   # R
        rgb_img[1, 60:90, 60:90] = 60   # G
        rgb_img[2, 60:90, 60:90] = 120  # B (strong blue)

        boxes, debug = detect_water_candidates(rgb_img, return_debug=True)
        self.assertEqual(debug["detector_method"], "rgb_spectral_contrast_fallback")
        self.assertEqual(len(boxes), 1)
        # Forest region [10:40, 10:40] must NOT be detected
        box = boxes[0]
        self.assertGreaterEqual(box[0], 55.0)
        self.assertGreaterEqual(box[1], 55.0)

    def test_vqa_and_grounding_integration(self):
        """Verify VQA and Grounding public interfaces with water queries."""
        s_path = "sample/valid_sample.tif"
        if not Path(s_path).exists():
            return

        # VQA test
        res_vqa = vqa(s_path, "Is there surface water?")
        self.assertIn("Yes", res_vqa.answer)
        self.assertIn("raw multispectral NDWI", res_vqa.answer)
        self.assertGreaterEqual(res_vqa.confidence, 0.90)

        # Grounding test
        res_ground = ground(s_path, "water body")
        self.assertGreater(len(res_ground.boxes), 0)
        self.assertEqual(res_ground.boxes[0].label, "Water Body")
        self.assertGreaterEqual(res_ground.boxes[0].score, 0.90)

        # Unified run_detection test
        res_det = run_detection(s_path, query="highlight the water body", task="grounding")
        self.assertEqual(res_det["task"], "grounding")
        self.assertIn("spatial_data", res_det)
        self.assertGreater(len(res_det["spatial_data"]["features"]), 0)


if __name__ == "__main__":
    unittest.main()
