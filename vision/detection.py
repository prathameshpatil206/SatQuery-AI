"""
SatQuery-AI: Single-Image Remote Sensing Specialist Module
Provides VQA, Captioning, and Visual Grounding for satellite imagery.
Follows a modular architecture allowing swappable model backends.
Includes zero-shot Florence-2 inference with projected GeoJSON coordinate mapping.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image

from vision.types import (
    BoundingBox,
    CaptionResult,
    ExecutionTrace,
    GroundingResult,
    VisualEvidence,
    VQAResult,
    boxes_to_geojson,
)
from vision.geospatial import geo_to_pixel, load_image, pixel_to_geo, to_display_rgb


class BaseDetectionSpecialist(ABC):
    """Abstract interface for satellite image VQA, captioning, and grounding."""

    @abstractmethod
    def vqa(self, image: np.ndarray, question: str, metadata: Optional[Dict[str, Any]] = None) -> VQAResult:
        """Answer a natural language question regarding the satellite image."""
        pass

    @abstractmethod
    def caption(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> CaptionResult:
        """Generate a natural language description/caption for the satellite image."""
        pass

    @abstractmethod
    def ground(self, image: np.ndarray, text: str, metadata: Optional[Dict[str, Any]] = None) -> GroundingResult:
        """Locate bounding boxes for entities matching the text query."""
        pass


class BaselineRemoteSensingSpecialist(BaseDetectionSpecialist):
    """
    Deterministic Remote-Sensing Baseline Specialist.
    Analyzes spectral channels, vegetation indices (NDVI), brightness, and spatial variance
    to generate grounded evidence, scene descriptions, and answer land-cover/scene queries.
    """

    def __init__(self, name: str = "Baseline-RS-Analytic", version: str = "1.0-baseline"):
        self.name = name
        self.version = version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _extract_scene_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract multi-spectral/visual statistical features for reasoning."""
        rgb = to_display_rgb(image)
        h, w = rgb.shape[:2]

        gray = np.mean(rgb, axis=-1).astype(np.float32)
        mean_brightness = float(np.mean(gray))
        spatial_variance = float(np.var(gray))

        ndvi_mean = None
        if image.ndim == 3 and image.shape[0] >= 4:
            red = image[0].astype(np.float32)
            nir = image[3].astype(np.float32)
            denom = nir + red + 1e-6
            ndvi = (nir - red) / denom
            ndvi_mean = float(np.mean(ndvi))

        r_mean = float(np.mean(rgb[:, :, 0]))
        g_mean = float(np.mean(rgb[:, :, 1]))
        b_mean = float(np.mean(rgb[:, :, 2]))

        dominant_type = "mixed terrain"
        if ndvi_mean is not None and ndvi_mean > 0.3:
            dominant_type = "dense vegetation / agricultural area"
        elif g_mean > r_mean and g_mean > b_mean:
            dominant_type = "vegetation / forest canopy"
        elif b_mean > r_mean and b_mean > g_mean:
            dominant_type = "water body / coastal marine"
        elif mean_brightness < 45:
            dominant_type = "water body or shadowed terrain"
        elif spatial_variance > 1800:
            dominant_type = "urban / built-up infrastructure"
        elif mean_brightness > 190:
            dominant_type = "arid / bare soil / bright structures"

        return {
            "rgb": rgb,
            "width": w,
            "height": h,
            "mean_brightness": mean_brightness,
            "spatial_variance": spatial_variance,
            "ndvi_mean": ndvi_mean,
            "dominant_type": dominant_type,
        }

    def vqa(self, image: np.ndarray, question: str, metadata: Optional[Dict[str, Any]] = None) -> VQAResult:
        start_time = time.time()
        stats = self._extract_scene_stats(image)
        q_lower = question.lower()

        if "what" in q_lower and ("type" in q_lower or "terrain" in q_lower or "land" in q_lower or "cover" in q_lower):
            answer = f"The predominant land cover is {stats['dominant_type']}."
            confidence = 0.85
        elif "water" in q_lower:
            has_water = stats["mean_brightness"] < 50 or stats["dominant_type"] == "water body / coastal marine"
            answer = "Yes, water or low-reflectance bodies are present." if has_water else "No prominent water bodies are detected."
            confidence = 0.82
        elif "vegetation" in q_lower or "forest" in q_lower or "green" in q_lower or "crop" in q_lower:
            has_veg = (stats["ndvi_mean"] is not None and stats["ndvi_mean"] > 0.2) or "vegetation" in stats["dominant_type"]
            answer = "Significant vegetation coverage is observed in this scene." if has_veg else "Vegetation cover is sparse or absent."
            confidence = 0.85
        elif "urban" in q_lower or "building" in q_lower or "city" in q_lower:
            is_urban = stats["spatial_variance"] > 1600
            answer = "Built-up structures or high-texture urban elements are evident." if is_urban else "No major urban structures detected."
            confidence = 0.80
        elif "count" in q_lower or "how many" in q_lower:
            answer = "Multiple distinct land parcels or structural features are present."
            confidence = 0.70
        else:
            answer = f"The satellite observation depicts a {stats['dominant_type']} with average brightness {stats['mean_brightness']:.1f}."
            confidence = 0.75

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=self.name,
            model_version=self.version,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata=metadata or {},
            steps_executed=["load_rgb", "compute_spectral_statistics", "query_heuristic_matching"],
        )

        evidence = VisualEvidence(
            evidence_type="preview",
            preview_rgb=stats["rgb"],
            description=f"Normalized RGB rendering of {stats['dominant_type']}",
        )

        return VQAResult(
            question=question,
            answer=answer,
            confidence=confidence,
            visual_evidence=evidence,
            execution_trace=trace,
            metadata={"stats": stats},
        )

    def caption(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> CaptionResult:
        start_time = time.time()
        stats = self._extract_scene_stats(image)

        features = [stats["dominant_type"]]
        if stats["ndvi_mean"] is not None:
            features.append(f"NDVI: {stats['ndvi_mean']:.2f}")
        if stats["spatial_variance"] > 1600:
            features.append("High spatial texture / built-up features")

        caption_text = (
            f"Remote sensing imagery depicting {stats['dominant_type']}, "
            f"characterized by mean scene reflectance of {stats['mean_brightness']:.1f} "
            f"across an area of {stats['width']}x{stats['height']} pixels."
        )

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=self.name,
            model_version=self.version,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata=metadata or {},
            steps_executed=["radiometric_stretch", "feature_synthesis", "caption_generation"],
        )

        evidence = VisualEvidence(
            evidence_type="preview",
            preview_rgb=stats["rgb"],
            description="Radiometrically calibrated RGB preview",
        )

        return CaptionResult(
            caption=caption_text,
            confidence=0.88,
            detected_features=features,
            visual_evidence=evidence,
            execution_trace=trace,
            metadata={"stats": stats},
        )

    def ground(self, image: np.ndarray, text: str, metadata: Optional[Dict[str, Any]] = None) -> GroundingResult:
        start_time = time.time()
        stats = self._extract_scene_stats(image)
        rgb = stats["rgb"]
        h, w = rgb.shape[:2]

        boxes = []
        gray = np.mean(rgb, axis=-1)
        t_lower = text.lower()

        if "water" in t_lower or "dark" in t_lower:
            mask = gray < 60
        elif "bright" in t_lower or "urban" in t_lower:
            mask = gray > 180
        else:
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx**2 + gy**2)
            mask = grad_mag > np.percentile(grad_mag, 85)

        if np.any(mask):
            coords = np.argwhere(mask)
            if len(coords) > 0:
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)

                geo_coords = None
                if metadata and metadata.get("transform") and metadata.get("crs"):
                    transform = metadata["transform"]
                    top_left = pixel_to_geo(transform, xmin, ymin)
                    bottom_right = pixel_to_geo(transform, xmax, ymax)
                    geo_coords = {
                        "min_lon": top_left["lon"],
                        "max_lat": top_left["lat"],
                        "max_lon": bottom_right["lon"],
                        "min_lat": bottom_right["lat"],
                    }

                boxes.append(
                    BoundingBox(
                        box_2d=[float(ymin), float(xmin), float(ymax), float(xmax)],
                        label=text,
                        score=0.82,
                        geo_coordinates=geo_coords,
                    )
                )

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=self.name,
            model_version=self.version,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata=metadata or {},
            steps_executed=["spectral_segmentation", "bounding_box_extraction"],
        )

        evidence = VisualEvidence(
            evidence_type="boxes",
            boxes=boxes,
            preview_rgb=rgb,
            description=f"Grounded bounding boxes for query: '{text}'",
        )

        return GroundingResult(
            target_query=text,
            boxes=boxes,
            confidence=0.82 if boxes else 0.40,
            visual_evidence=evidence,
            execution_trace=trace,
        )


class DeepVisionLanguageSpecialist(BaseDetectionSpecialist):
    """
    Production Vision-Language Specialist powered by Microsoft Florence-2-base on CUDA (FP16).
    Performs zero-shot grounding, captioning, and open-vocabulary remote sensing detection.
    Maps pixel bboxes directly to geographic coordinates (lon, lat) using geospatial affine transforms.
    """

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._is_loaded = False
        self._baseline = BaselineRemoteSensingSpecialist()

    def load_model(self) -> None:
        """Lazy load Florence-2 weights onto GPU."""
        if self._is_loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._is_loaded = True
        except Exception as e:
            self._is_loaded = False

    def vqa(self, image: np.ndarray, question: str, metadata: Optional[Dict[str, Any]] = None) -> VQAResult:
        base_res = self._baseline.vqa(image, question, metadata)
        self.load_model()
        if not self._is_loaded:
            return base_res

        start_time = time.time()
        rgb = to_display_rgb(image)
        pil_img = Image.fromarray(rgb)
        w, h = pil_img.size

        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=1,
            )

        gen_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(gen_text, task=prompt, image_size=(w, h))
        vlm_desc = parsed.get(prompt, gen_text).strip()

        answer = f"{base_res.answer} (Visual Context: {vlm_desc})"

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=f"Florence-2-VQA ({self.device.upper()})",
            model_version=self.model_id,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata=metadata or {},
            steps_executed=["geospatial_rgb_stretch", "florence2_scene_perception", "spectral_reasoning_fusion"],
        )

        return VQAResult(
            question=question,
            answer=answer,
            confidence=0.92,
            visual_evidence=VisualEvidence(evidence_type="preview", preview_rgb=rgb),
            execution_trace=trace,
            metadata=base_res.metadata,
        )

    def caption(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> CaptionResult:
        self.load_model()
        if not self._is_loaded:
            return self._baseline.caption(image, metadata)

        start_time = time.time()
        rgb = to_display_rgb(image)
        pil_img = Image.fromarray(rgb)
        w, h = pil_img.size

        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=1,
            )

        gen_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(gen_text, task=prompt, image_size=(w, h))
        caption_text = parsed.get(prompt, gen_text)

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=f"Florence-2-Captioner ({self.device.upper()})",
            model_version=self.model_id,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata=metadata or {},
            steps_executed=["radiometric_stretch", "florence2_detailed_caption"],
        )

        return CaptionResult(
            caption=caption_text,
            confidence=0.94,
            detected_features=["scene context", "satellite visual elements"],
            visual_evidence=VisualEvidence(evidence_type="preview", preview_rgb=rgb),
            execution_trace=trace,
        )

    def ground(self, image: np.ndarray, text: str, metadata: Optional[Dict[str, Any]] = None) -> GroundingResult:
        self.load_model()
        if not self._is_loaded:
            return self._baseline.ground(image, text, metadata)

        start_time = time.time()
        rgb = to_display_rgb(image)
        pil_img = Image.fromarray(rgb)
        w, h = pil_img.size

        # Use Phrase Grounding or Object Detection
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {text}"
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=1,
            )

        gen_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            gen_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(w, h)
        )

        data = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        raw_boxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        # Fallback to OD if phrase grounding returned nothing
        if not raw_boxes:
            prompt_od = "<OD>"
            inputs_od = self.processor(text=prompt_od, images=pil_img, return_tensors="pt").to(
                self.device, torch.float16 if self.device == "cuda" else torch.float32
            )
            with torch.no_grad():
                gen_ids_od = self.model.generate(
                    input_ids=inputs_od["input_ids"],
                    pixel_values=inputs_od["pixel_values"],
                    max_new_tokens=100,
                    num_beams=1,
                )
            gen_text_od = self.processor.batch_decode(gen_ids_od, skip_special_tokens=False)[0]
            parsed_od = self.processor.post_process_generation(gen_text_od, task=prompt_od, image_size=(w, h))
            data_od = parsed_od.get("<OD>", {})
            raw_boxes = data_od.get("bboxes", [])
            labels = data_od.get("labels", [])

        # Transform pixel boxes to projected geo coordinates
        boxes = []
        transform = metadata.get("transform") if metadata else None
        has_crs = metadata.get("is_geospatial", False) if metadata else False

        for idx, b in enumerate(raw_boxes):
            ymin, xmin, ymax, xmax = [float(v) for v in b]
            label = labels[idx] if idx < len(labels) else text

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

            boxes.append(
                BoundingBox(
                    box_2d=[ymin, xmin, ymax, xmax],
                    label=label,
                    score=0.88,
                    geo_coordinates=geo_coords,
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        trace = ExecutionTrace(
            specialist_name=f"Florence-2-Grounding ({self.device.upper()})",
            model_version=self.model_id,
            processing_time_ms=elapsed_ms,
            device=self.device,
            spatial_metadata=metadata or {},
            steps_executed=[
                "geospatial_rgb_stretch",
                "florence2_phrase_grounding",
                "pixel_to_geo_projection",
            ],
        )

        evidence = VisualEvidence(
            evidence_type="boxes",
            boxes=boxes,
            preview_rgb=rgb,
            description=f"Grounded {len(boxes)} region(s) for '{text}'",
        )

        return GroundingResult(
            target_query=text,
            boxes=boxes,
            confidence=0.90 if boxes else 0.50,
            visual_evidence=evidence,
            execution_trace=trace,
        )


# Global deep specialist instance
_DEFAULT_DETECTION_SPECIALIST = DeepVisionLanguageSpecialist()


def get_default_specialist() -> BaseDetectionSpecialist:
    """Return the currently configured single-image detection specialist."""
    return _DEFAULT_DETECTION_SPECIALIST


def set_default_specialist(specialist: BaseDetectionSpecialist) -> None:
    """Set the active detection specialist backend."""
    global _DEFAULT_DETECTION_SPECIALIST
    _DEFAULT_DETECTION_SPECIALIST = specialist


def vqa(
    image: Union[str, np.ndarray, Image.Image],
    question: str,
    specialist: Optional[BaseDetectionSpecialist] = None,
) -> VQAResult:
    """Public Specialist Tool: Visual Question Answering on single satellite image."""
    spec = specialist or get_default_specialist()
    img_arr, meta = load_image(image)
    return spec.vqa(img_arr, question, meta)


def caption(
    image: Union[str, np.ndarray, Image.Image],
    specialist: Optional[BaseDetectionSpecialist] = None,
) -> CaptionResult:
    """Public Specialist Tool: Captioning for satellite image."""
    spec = specialist or get_default_specialist()
    img_arr, meta = load_image(image)
    return spec.caption(img_arr, meta)


def ground(
    image: Union[str, np.ndarray, Image.Image],
    text: str,
    specialist: Optional[BaseDetectionSpecialist] = None,
) -> GroundingResult:
    """Public Specialist Tool: Visual grounding / phrase localization in satellite image."""
    spec = specialist or get_default_specialist()
    img_arr, meta = load_image(image)
    return spec.ground(img_arr, text, meta)


def run_detection(
    image_path: Union[str, np.ndarray, Image.Image],
    query: Optional[str] = None,
    task: str = "auto",
) -> Dict[str, Any]:
    """
    Primary unified entry point for Atharv (Backend / FastAPI router).
    Executes Florence-2 or baseline specialists and returns the standard JSON contract
    with GeoJSON spatial_data for Rohan's Leaflet frontend.

    Args:
        image_path: Path to GeoTIFF, PNG, JPEG or in-memory array
        query: User text query (e.g., 'highlight the water body', 'describe the scene')
        task: 'auto', 'grounding', 'vqa', or 'caption'
    """
    q = (query or "").strip()
    q_lower = q.lower()

    # Automatic task intent detection if task == 'auto'
    if task == "auto":
        if not q or "describe" in q_lower or "caption" in q_lower or "overview" in q_lower:
            task = "caption"
        elif "highlight" in q_lower or "locate" in q_lower or "ground" in q_lower or "find" in q_lower or "where" in q_lower:
            task = "grounding"
        else:
            task = "vqa"

    if task == "caption":
        res = caption(image_path)
        return res.to_dict()
    elif task == "grounding":
        # Extract phrase to ground
        target = q
        for prefix in ["highlight", "locate", "ground", "find", "where is", "where are", "the"]:
            if target.lower().startswith(prefix):
                target = target[len(prefix):].strip()
        target = target or "salient objects"
        res = ground(image_path, target)
        return res.to_dict()
    else:  # VQA
        res = vqa(image_path, q or "What is visible in this satellite imagery?")
        return res.to_dict()
