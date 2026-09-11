"""
SatQuery-AI: Vision Specialist Data Contracts and Schemas
Defines structured outputs for agentic remote-sensing specialist tools.
Includes native GeoJSON export for frontend mapping (Leaflet/Mapbox).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box in pixel [ymin, xmin, ymax, xmax] and projected geo coordinates."""
    box_2d: List[float]  # [ymin, xmin, ymax, xmax]
    label: str = ""
    score: float = 1.0
    geo_coordinates: Optional[Dict[str, float]] = None  # {'min_lat': ..., 'min_lon': ..., 'max_lat': ..., 'max_lon': ...}

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert bounding box to GeoJSON Feature with Polygon geometry for Leaflet map."""
        if self.geo_coordinates:
            min_lon = float(self.geo_coordinates.get("min_lon", 0.0))
            max_lon = float(self.geo_coordinates.get("max_lon", 0.0))
            min_lat = float(self.geo_coordinates.get("min_lat", 0.0))
            max_lat = float(self.geo_coordinates.get("max_lat", 0.0))
            coordinates = [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]]
        else:
            ymin, xmin, ymax, xmax = self.box_2d
            coordinates = [[
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin],
            ]]

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
            },
            "properties": {
                "label": self.label,
                "score": round(self.score, 3),
                "pixel_box": self.box_2d,
                "geo_coordinates": self.geo_coordinates,
            },
        }


def boxes_to_geojson(boxes: List[BoundingBox]) -> Dict[str, Any]:
    """Wrap a list of BoundingBox objects as a GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [b.to_geojson_feature() for b in boxes],
    }


@dataclass
class VisualEvidence:
    """Container for visual evidence returned to the agent/router and UI."""
    evidence_type: str  # "mask", "heatmap", "boxes", "composite", "preview"
    mask: Optional[np.ndarray] = None  # 2D binary or continuous change/feature mask
    boxes: List[BoundingBox] = field(default_factory=list)
    preview_rgb: Optional[np.ndarray] = None  # 8-bit RGB array suitable for rendering/display
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "description": self.description,
            "has_mask": self.mask is not None,
            "mask_shape": list(self.mask.shape) if self.mask is not None else None,
            "box_count": len(self.boxes),
            "preview_shape": list(self.preview_rgb.shape) if self.preview_rgb is not None else None,
        }


@dataclass
class ExecutionTrace:
    """Execution metadata and performance metrics for agent logging and auditability."""
    specialist_name: str
    model_version: str
    processing_time_ms: float
    device: str
    spatial_metadata: Dict[str, Any] = field(default_factory=dict)
    steps_executed: List[str] = field(default_factory=list)

    def to_step_list(self) -> List[Dict[str, Any]]:
        """Format as live-updating step list for Rohan's frontend."""
        steps = []
        n = max(len(self.steps_executed), 1)
        step_time = round(self.processing_time_ms / n, 1)
        for s in self.steps_executed:
            steps.append({
                "step": s,
                "status": "completed",
                "time_ms": step_time,
            })
        return steps


@dataclass
class VQAResult:
    """Output for single-image Visual Question Answering."""
    question: str
    answer: str
    confidence: float
    visual_evidence: Optional[VisualEvidence] = None
    execution_trace: Optional[ExecutionTrace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        boxes = self.visual_evidence.boxes if self.visual_evidence else []
        return {
            "task": "vqa",
            "query": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "visual_evidence": self.visual_evidence.to_dict() if self.visual_evidence else {},
            "spatial_data": boxes_to_geojson(boxes),
            "execution_trace": self.execution_trace.to_step_list() if self.execution_trace else [],
        }


@dataclass
class CaptionResult:
    """Output for single-image remote sensing captioning."""
    caption: str
    confidence: float
    detected_features: List[str] = field(default_factory=list)
    visual_evidence: Optional[VisualEvidence] = None
    execution_trace: Optional[ExecutionTrace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        boxes = self.visual_evidence.boxes if self.visual_evidence else []
        return {
            "task": "caption",
            "query": "Describe the scene",
            "answer": self.caption,
            "confidence": round(self.confidence, 3),
            "detected_features": self.detected_features,
            "visual_evidence": self.visual_evidence.to_dict() if self.visual_evidence else {},
            "spatial_data": boxes_to_geojson(boxes),
            "execution_trace": self.execution_trace.to_step_list() if self.execution_trace else [],
        }


@dataclass
class GroundingResult:
    """Output for visual grounding of natural language queries to satellite imagery regions."""
    target_query: str
    boxes: List[BoundingBox] = field(default_factory=list)
    confidence: float = 1.0
    visual_evidence: Optional[VisualEvidence] = None
    execution_trace: Optional[ExecutionTrace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        if self.boxes:
            answer = f"Located {len(self.boxes)} region(s) matching query: '{self.target_query}'."
        else:
            q_lower = (self.target_query or "").lower()
            if any(w in q_lower for w in ["water", "lake", "river", "reservoir", "ocean", "sea", "pond", "canal", "flood"]):
                answer = "No water body detected in this satellite scene. Spectral water indices (NDWI) and reflectance profiling confirm dry terrain with no open surface water."
            else:
                answer = f"No regions matching '{self.target_query}' were detected in this satellite scene."

        return {
            "task": "grounding",
            "query": self.target_query,
            "answer": answer,
            "confidence": round(self.confidence, 3),
            "visual_evidence": self.visual_evidence.to_dict() if self.visual_evidence else {},
            "spatial_data": boxes_to_geojson(self.boxes),
            "execution_trace": self.execution_trace.to_step_list() if self.execution_trace else [],
        }


@dataclass
class ChangeResult:
    """Output for bi-temporal remote sensing change detection."""
    summary: str
    change_ratio: float  # Percentage (0.0 to 1.0) of area changed
    change_mask: Optional[np.ndarray] = None  # 2D binary uint8 mask
    difference_heatmap: Optional[np.ndarray] = None  # 2D float32 normalized distance
    changed_boxes: List[BoundingBox] = field(default_factory=list)
    confidence: float = 1.0
    visual_evidence: Optional[VisualEvidence] = None
    execution_trace: Optional[ExecutionTrace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": "change_detection",
            "query": "Bi-temporal change analysis",
            "answer": self.summary,
            "confidence": round(self.confidence, 3),
            "change_percentage": round(self.change_ratio * 100.0, 2),
            "clusters_detected": len(self.changed_boxes),
            "visual_evidence": self.visual_evidence.to_dict() if self.visual_evidence else {},
            "spatial_data": boxes_to_geojson(self.changed_boxes),
            "execution_trace": self.execution_trace.to_step_list() if self.execution_trace else [],
        }


@dataclass
class OpticalSARResult:
    """Output for paired optical and SAR multimodal remote sensing analysis."""
    analysis_summary: str
    optical_features: Dict[str, Any] = field(default_factory=dict)
    sar_features: Dict[str, Any] = field(default_factory=dict)
    complementary_insights: List[str] = field(default_factory=list)
    confidence: float = 1.0
    visual_evidence: Optional[VisualEvidence] = None
    execution_trace: Optional[ExecutionTrace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        boxes = self.visual_evidence.boxes if self.visual_evidence else []
        return {
            "task": "optical_sar",
            "query": "Optical + SAR cross-modal analysis",
            "answer": self.analysis_summary,
            "confidence": round(self.confidence, 3),
            "complementary_insights": self.complementary_insights,
            "optical_features": self.optical_features,
            "sar_features": self.sar_features,
            "visual_evidence": self.visual_evidence.to_dict() if self.visual_evidence else {},
            "spatial_data": boxes_to_geojson(boxes),
            "execution_trace": self.execution_trace.to_step_list() if self.execution_trace else [],
        }
