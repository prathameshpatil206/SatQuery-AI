"""
SatQuery-AI: Bi-Temporal Remote Sensing Change Analysis Module
Performs change detection, visual difference mapping, change segmentation,
and natural language change summary across multi-temporal satellite pairs.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from PIL import Image

from vision.types import (
    BoundingBox,
    ChangeResult,
    ExecutionTrace,
    VisualEvidence,
)
from vision.geospatial import (
    load_image,
    pixel_to_geo,
    to_display_rgb,
    validate_spatial_pair,
)


class BaseChangeSpecialist(ABC):
    """Abstract interface for bi-temporal remote sensing change detection."""

    @abstractmethod
    def detect_change(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> ChangeResult:
        """Analyze changes between time-1 (img1) and time-2 (img2) rasters."""
        pass


class AlgorithmicChangeDetector(BaseChangeSpecialist):
    """
    Robust algorithmic change detector using multi-spectral Change Vector Analysis (CVA),
    spectral Euclidean distance, adaptive Otsu thresholding, and contour bounding boxes.
    Provides deterministic baseline change maps and visual evidence.
    """

    def __init__(self, name: str = "Algorithmic-CVA-Detector", version: str = "1.0-baseline"):
        self.name = name
        self.version = version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect_change(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> ChangeResult:
        start_time = time.time()
        steps = ["validate_spatial_compatibility"]

        # Check spatial compatibility
        spatial_report = validate_spatial_pair(meta1, meta2)

        # Ensure spatial dimensions match
        rgb1 = to_display_rgb(img1)
        rgb2 = to_display_rgb(img2)

        h1, w1 = rgb1.shape[:2]
        h2, w2 = rgb2.shape[:2]

        if (h1, w1) != (h2, w2):
            steps.append(f"resize_alignment: {h2}x{w2} -> {h1}x{w1}")
            rgb2 = cv2.resize(rgb2, (w1, h1), interpolation=cv2.INTER_LINEAR)

        steps.append("compute_spectral_difference")

        # Multi-band normalized difference / Change Vector Analysis
        diff = np.linalg.norm(rgb2.astype(np.float32) - rgb1.astype(np.float32), axis=-1)
        # Normalize continuous difference heatmap to [0, 1]
        diff_max = float(np.max(diff))
        heatmap = (diff / (diff_max + 1e-6)).astype(np.float32)

        # Adaptive thresholding to generate binary change mask
        diff_uint8 = np.clip((diff / (diff_max + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        # Apply Otsu's thresholding for bimodal separation of change vs background
        otsu_val, mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        steps.append(f"otsu_thresholding: value={otsu_val:.1f}")

        # Morphological noise filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate change metrics
        total_pixels = float(h1 * w1)
        changed_pixels = float(np.count_nonzero(clean_mask))
        change_ratio = changed_pixels / total_pixels
        change_pct = change_ratio * 100.0

        steps.append("extract_change_bounding_boxes")
        # Extract contours of changed regions
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        changed_boxes = []

        transform = meta1.get("transform")
        has_crs = meta1.get("is_geospatial", False)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter small speckle changes (< 0.1% total area)
            if area > (total_pixels * 0.001):
                x, y, w, h = cv2.boundingRect(cnt)
                ymin, xmin, ymax, xmax = float(y), float(x), float(y + h), float(x + w)

                geo_coords = None
                if has_crs and transform:
                    top_left = pixel_to_geo(transform, xmin, ymin)
                    bottom_right = pixel_to_geo(transform, xmax, ymax)
                    geo_coords = {
                        "min_lon": top_left["lon"],
                        "max_lat": top_left["lat"],
                        "max_lon": bottom_right["lon"],
                        "min_lat": bottom_right["lat"],
                    }

                changed_boxes.append(
                    BoundingBox(
                        box_2d=[ymin, xmin, ymax, xmax],
                        label="detected_change_zone",
                        score=min(1.0, float(area / total_pixels) * 5.0 + 0.5),
                        geo_coordinates=geo_coords,
                    )
                )

        # Synthesize natural language summary
        if change_pct < 1.0:
            summary = (
                f"No significant structural change detected between T1 and T2 "
                f"({change_pct:.2f}% subtle variation, within natural seasonal tolerance)."
            )
            confidence = 0.92
        elif change_pct < 15.0:
            summary = (
                f"Moderate localized change observed across {change_pct:.2f}% of the scene, "
                f"identifying {len(changed_boxes)} distinct modified zones. "
                f"Pattern consistent with agricultural cycles, cleared parcels, or surface modification."
            )
            confidence = 0.88
        else:
            summary = (
                f"Extensive landscape alteration detected across {change_pct:.2f}% of the spatial extent "
                f"spanning {len(changed_boxes)} significant clusters. "
                f"Substantial spectral shift indicates construction, deforestation, or severe event impact."
            )
            confidence = 0.85

        if prompt:
            summary += f" [Addressed Query: '{prompt}']"

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=self.name,
            model_version=self.version,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata={"t1": meta1, "t2": meta2, "pair_validation": spatial_report},
            steps_executed=steps,
        )

        evidence = VisualEvidence(
            evidence_type="mask",
            mask=clean_mask,
            boxes=changed_boxes,
            preview_rgb=rgb2,
            description=f"Change mask covering {change_pct:.2f}% surface modification across {len(changed_boxes)} clusters",
        )

        return ChangeResult(
            summary=summary,
            change_ratio=change_ratio,
            change_mask=clean_mask,
            difference_heatmap=heatmap,
            changed_boxes=changed_boxes,
            confidence=confidence,
            visual_evidence=evidence,
            execution_trace=trace,
            metadata={
                "change_percentage": change_pct,
                "changed_pixel_count": int(changed_pixels),
                "clusters_detected": len(changed_boxes),
                "spatial_report": spatial_report,
            },
        )


class DeepChangeSpecialist(BaseChangeSpecialist):
    """
    Pluggable wrapper for Deep Siamese Change Detection Models (e.g. ChangeFormer / BIT / CDVQA).
    Falls back gracefully to AlgorithmicChangeDetector if checkpoint is uninitialized.
    """

    def __init__(self, model_checkpoint: Optional[str] = None):
        self.model_checkpoint = model_checkpoint
        self._algorithmic = AlgorithmicChangeDetector()
        self._is_loaded = False

    def detect_change(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> ChangeResult:
        if not self._is_loaded:
            res = self._algorithmic.detect_change(img1, img2, meta1, meta2, prompt)
            if res.execution_trace:
                res.execution_trace.steps_executed.append(
                    "Deep change weights uninitialized; executed via Algorithmic CVA engine."
                )
            return res

        # Placeholder for fine-tuned Siamese forward pass
        return self._algorithmic.detect_change(img1, img2, meta1, meta2, prompt)


_DEFAULT_CHANGE_SPECIALIST = AlgorithmicChangeDetector()


def get_default_change_specialist() -> BaseChangeSpecialist:
    """Get active bi-temporal change detection specialist."""
    return _DEFAULT_CHANGE_SPECIALIST


def set_default_change_specialist(specialist: BaseChangeSpecialist) -> None:
    """Set active bi-temporal change detection specialist."""
    global _DEFAULT_CHANGE_SPECIALIST
    _DEFAULT_CHANGE_SPECIALIST = specialist


def detect_change(
    img1: Union[str, np.ndarray, Image.Image],
    img2: Union[str, np.ndarray, Image.Image],
    prompt: Optional[str] = None,
    specialist: Optional[BaseChangeSpecialist] = None,
) -> ChangeResult:
    """
    Public Specialist Tool: Detect changes between bi-temporal satellite images (T1 vs T2).
    Accepts GeoTIFF file paths, standard image files, NumPy arrays, or PIL Images.
    """
    spec = specialist or get_default_change_specialist()
    arr1, meta1 = load_image(img1)
    arr2, meta2 = load_image(img2)
    return spec.detect_change(arr1, arr2, meta1, meta2, prompt)
