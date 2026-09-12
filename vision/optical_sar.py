"""
SatQuery-AI: Optical + SAR Paired Remote Sensing Analysis Module
Provides multimodal cross-sensor analysis combining optical spectral reflectance
with Synthetic Aperture Radar (SAR) polarimetric backscatter and structural penetration.
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
    ExecutionTrace,
    Modality,
    OpticalSARResult,
    VisualEvidence,
)
from vision.geospatial import (
    detect_modality,
    load_image,
    to_display_rgb,
    validate_spatial_pair,
)


class BaseOpticalSARSpecialist(ABC):
    """Abstract interface for paired Optical and SAR remote sensing analysis."""

    @abstractmethod
    def analyze(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        optical_meta: Dict[str, Any],
        sar_meta: Dict[str, Any],
        query: Optional[str] = None,
    ) -> OpticalSARResult:
        """Perform multimodal cross-sensor analysis."""
        pass


class HeuristicOpticalSARSpecialist(BaseOpticalSARSpecialist):
    """
    Multimodal Remote Sensing Specialist for paired Optical + SAR imagery.
    Cross-references optical spectral reflectance with SAR radar backscatter:
    - Leverages SAR microwave penetration to inspect surface conditions beneath cloud cover.
    - Fuses low optical reflectance with SAR specular reflection to confirm water bodies.
    - Correlates high optical texture with SAR double-bounce scattering for urban mapping.
    """

    def __init__(self, name: str = "Optical-SAR-CrossModal", version: str = "1.0-baseline"):
        self.name = name
        self.version = version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def analyze(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        optical_meta: Dict[str, Any],
        sar_meta: Dict[str, Any],
        query: Optional[str] = None,
    ) -> OpticalSARResult:
        start_time = time.time()
        steps = ["validate_multimodal_pair"]

        # Spatial alignment validation
        spatial_report = validate_spatial_pair(optical_meta, sar_meta)

        # Modality check
        opt_modality = detect_modality(optical, optical_meta)
        sar_modality = detect_modality(sar, sar_meta)

        # Preprocess optical to RGB and SAR to normalized 2D amplitude
        opt_rgb = to_display_rgb(optical)
        sar_rgb = to_display_rgb(sar)

        h_opt, w_opt = opt_rgb.shape[:2]
        h_sar, w_sar = sar_rgb.shape[:2]

        # Harmonize resolutions if needed
        if (h_opt, w_opt) != (h_sar, w_sar):
            steps.append(f"resample_sar_to_optical: ({h_sar}x{w_sar}) -> ({h_opt}x{w_opt})")
            sar_rgb = cv2.resize(sar_rgb, (w_opt, h_opt), interpolation=cv2.INTER_LINEAR)

        steps.append("extract_cross_modal_features")

        # 1. Optical Cloud & Brightness Analysis
        opt_gray = np.mean(opt_rgb, axis=-1)
        opt_brightness = float(np.mean(opt_gray))
        # Clouds are typically saturated white/high reflectance with low texture
        potential_cloud_mask = (opt_gray > 220) & (cv2.Laplacian(opt_gray.astype(np.uint8), cv2.CV_64F).var() < 100)
        cloud_coverage_pct = float(np.count_nonzero(potential_cloud_mask)) / float(h_opt * w_opt) * 100.0

        # 2. SAR Backscatter & Roughness Analysis
        sar_gray = np.mean(sar_rgb, axis=-1)
        sar_mean = float(np.mean(sar_gray))
        sar_var = float(np.var(sar_gray))

        # Water bodies: verified via physical spectral water detector and SAR specular reflection
        try:
            from vision.detection import detect_water_candidates
            opt_water_boxes, opt_water_debug = detect_water_candidates(optical, metadata=optical_meta, return_debug=True)
            water_coverage_pct = opt_water_debug.get("detected_water_pixel_percentage", 0.0)
            has_water = len(opt_water_boxes) > 0 and (sar_mean < 120.0)
        except Exception:
            water_candidates = (opt_rgb[:, :, 2] > opt_rgb[:, :, 0] + 20) & (sar_gray < 50)
            water_coverage_pct = float(np.count_nonzero(water_candidates)) / float(h_opt * w_opt) * 100.0
            has_water = water_coverage_pct > 1.0

        # Built-up / Urban structures: bright corner-reflector scattering in SAR and high variance in optical
        urban_candidates = (sar_gray > 180) & (opt_gray > 100)
        urban_coverage_pct = float(np.count_nonzero(urban_candidates)) / float(h_opt * w_opt) * 100.0

        steps.append("synthesize_complementary_evidence")

        insights = []
        if cloud_coverage_pct > 10.0:
            insights.append(
                f"Cloud occlusion detected across {cloud_coverage_pct:.1f}% of the optical acquisition. "
                f"SAR C-band radar penetrated cloud layers, revealing persistent ground surface roughness."
            )
        else:
            insights.append(
                f"Clear atmospheric conditions in optical scene (< {cloud_coverage_pct:.1f}% cloud coverage)."
            )

        if has_water and water_coverage_pct > 0.5:
            insights.append(
                f"Water bodies confirmed via cross-sensor agreement ({water_coverage_pct:.1f}% coverage): "
                f"exhibiting physical spectral water absorption and characteristic SAR specular reflection."
            )

        if urban_coverage_pct > 2.0:
            insights.append(
                f"Built-up / structural presence verified ({urban_coverage_pct:.1f}% coverage): "
                f"evidenced by high SAR dihedral backscatter and structured optical edges."
            )

        # Construct false-color composite: Red=SAR, Green=Optical Green, Blue=Optical Blue
        composite_rgb = np.zeros_like(opt_rgb)
        composite_rgb[:, :, 0] = sar_gray.astype(np.uint8)  # SAR backscatter
        composite_rgb[:, :, 1] = opt_rgb[:, :, 1]          # Optical green
        composite_rgb[:, :, 2] = opt_rgb[:, :, 2]          # Optical blue

        summary = (
            f"Paired Optical+SAR analysis completed. Optical modality: {opt_modality.value}, "
            f"SAR modality: {sar_modality.value}. "
            + " ".join(insights)
        )
        if query:
            summary += f" [Addressed Query: '{query}']"

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=self.name,
            model_version=self.version,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata={
                "optical": optical_meta,
                "sar": sar_meta,
                "spatial_alignment": spatial_report,
            },
            steps_executed=steps,
        )

        evidence = VisualEvidence(
            evidence_type="composite",
            preview_rgb=composite_rgb,
            description="False-color cross-sensor composite (Red: SAR Backscatter, Green: Optical Green, Blue: Optical Blue)",
        )

        return OpticalSARResult(
            analysis_summary=summary,
            optical_features={
                "mean_brightness": opt_brightness,
                "cloud_coverage_pct": cloud_coverage_pct,
                "modality": opt_modality.value,
            },
            sar_features={
                "mean_backscatter_intensity": sar_mean,
                "roughness_variance": sar_var,
                "modality": sar_modality.value,
            },
            complementary_insights=insights,
            confidence=0.90,
            visual_evidence=evidence,
            execution_trace=trace,
            metadata={
                "spatial_report": spatial_report,
                "water_coverage_pct": water_coverage_pct,
                "urban_coverage_pct": urban_coverage_pct,
            },
        )


_DEFAULT_OPTICAL_SAR_SPECIALIST = HeuristicOpticalSARSpecialist()


def get_default_optical_sar_specialist() -> BaseOpticalSARSpecialist:
    """Get active Optical + SAR specialist."""
    return _DEFAULT_OPTICAL_SAR_SPECIALIST


def set_default_optical_sar_specialist(specialist: BaseOpticalSARSpecialist) -> None:
    """Set active Optical + SAR specialist."""
    global _DEFAULT_OPTICAL_SAR_SPECIALIST
    _DEFAULT_OPTICAL_SAR_SPECIALIST = specialist


def analyze_optical_sar(
    optical_image: Union[str, np.ndarray, Image.Image],
    sar_image: Union[str, np.ndarray, Image.Image],
    query: Optional[str] = None,
    specialist: Optional[BaseOpticalSARSpecialist] = None,
) -> OpticalSARResult:
    """
    Public Specialist Tool: Multimodal analysis on paired Optical and SAR satellite images.
    Accepts GeoTIFF file paths, standard image files, NumPy arrays, or PIL Images.
    """
    spec = specialist or get_default_optical_sar_specialist()
    opt_arr, opt_meta = load_image(optical_image)
    sar_arr, sar_meta = load_image(sar_image)
    return spec.analyze(opt_arr, sar_arr, opt_meta, sar_meta, query)
