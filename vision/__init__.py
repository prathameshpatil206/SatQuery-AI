"""
SatQuery-AI: Vision & Satellite Specialist Layer
Exposes modular remote-sensing specialist tools for agentic query answering.
"""

from vision.types import (
    BoundingBox,
    CaptionResult,
    ChangeResult,
    ExecutionTrace,
    GroundingResult,
    Modality,
    OpticalSARResult,
    VisualEvidence,
    VQAResult,
    boxes_to_geojson,
)
from vision.geospatial import (
    create_synthetic_geotiff,
    detect_modality,
    geo_to_pixel,
    load_image,
    pixel_to_geo,
    read_geotiff,
    to_display_rgb,
    validate_spatial_pair,
)
from vision.detection import (
    BaseDetectionSpecialist,
    BaselineRemoteSensingSpecialist,
    DeepVisionLanguageSpecialist,
    caption,
    get_default_specialist,
    ground,
    run_detection,
    set_default_specialist,
    vqa,
)
from vision.change_detection import (
    AlgorithmicChangeDetector,
    BaseChangeSpecialist,
    DeepChangeSpecialist,
    detect_change,
    get_default_change_specialist,
    set_default_change_specialist,
)
from vision.optical_sar import (
    BaseOpticalSARSpecialist,
    HeuristicOpticalSARSpecialist,
    analyze_optical_sar,
    get_default_optical_sar_specialist,
    set_default_optical_sar_specialist,
)

__all__ = [
    # Primary Entry Point for Atharv (Backend)
    "run_detection",
    # Core Specialist Tools for SatQuery Agent
    "vqa",
    "caption",
    "ground",
    "detect_change",
    "analyze_optical_sar",
    # Geospatial Engine
    "load_image",
    "read_geotiff",
    "pixel_to_geo",
    "geo_to_pixel",
    "to_display_rgb",
    "detect_modality",
    "validate_spatial_pair",
    "create_synthetic_geotiff",
    # Data Contracts & Schemas
    "VQAResult",
    "CaptionResult",
    "GroundingResult",
    "ChangeResult",
    "OpticalSARResult",
    "BoundingBox",
    "VisualEvidence",
    "ExecutionTrace",
    "Modality",
    "boxes_to_geojson",
    # Base Specialist Interfaces for Model Extension
    "BaseDetectionSpecialist",
    "BaselineRemoteSensingSpecialist",
    "DeepVisionLanguageSpecialist",
    "BaseChangeSpecialist",
    "AlgorithmicChangeDetector",
    "DeepChangeSpecialist",
    "BaseOpticalSARSpecialist",
    "HeuristicOpticalSARSpecialist",
    # Configuration Hooks
    "get_default_specialist",
    "set_default_specialist",
    "get_default_change_specialist",
    "set_default_change_specialist",
    "get_default_optical_sar_specialist",
    "set_default_optical_sar_specialist",
]
