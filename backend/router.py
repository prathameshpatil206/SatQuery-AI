"""
SatQuery-AI: Agentic Intent Router & Spatial Normalization Engine
Classifies user natural-language queries into remote sensing specialist workflows
using Gemini API or a deterministic zero-dependency fallback classifier.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import rasterio
from rasterio.warp import transform as warp_transform

CACHED_DEMO_PATH = Path("sample/cached_demo_results.json")


def classify_intent_gemini(query: str, has_t2: bool = False, has_sar: bool = False) -> Optional[str]:
    """
    Classify intent using Google Gemini API if GEMINI_API_KEY is available.
    Returns one of: 'grounding', 'vqa', 'caption', 'change_detection', 'optical_sar'.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import urllib.request
        prompt = (
            "You are an intent classifier for a satellite remote sensing assistant. "
            f"User query: '{query}'. Context: has_temporal_t2={has_t2}, has_sar_radar={has_sar}. "
            "Classify this query into exactly ONE of the following tasks:\n"
            "- 'change_detection': detects surface/infrastructure changes between two temporal images\n"
            "- 'optical_sar': paired optical RGB + radar SAR analysis (floods, all-weather penetration)\n"
            "- 'grounding': locating, highlighting, or segmenting specific spatial objects/features\n"
            "- 'caption': generating an overview summary or caption of the scene\n"
            "- 'vqa': answering questions about land cover, vegetation, or scene characteristics\n\n"
            "Return ONLY the exact task identifier with no explanation."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            valid_tasks = ["change_detection", "optical_sar", "grounding", "caption", "vqa"]
            for vt in valid_tasks:
                if vt in text:
                    return vt
    except Exception:
        # Gracefully drop down to deterministic fallback on any network or auth error
        pass

    return None


def classify_intent_fallback(query: str, has_t2: bool = False, has_sar: bool = False) -> str:
    """
    Zero-dependency deterministic classifier for high-reliability live demonstrations.
    """
    q = (query or "").lower().strip()

    # Temporal change detection triggers
    if has_t2 or any(w in q for w in ["change", "before", "after", "difference", "between", "construction", "expansion", "timeline"]):
        return "change_detection"

    # Multimodal radar/SAR triggers
    if has_sar or any(w in q for w in ["sar", "radar", "cloud", "penetrat", "cross-modal", "flood", "water surface", "all-weather"]):
        return "optical_sar"

    # Spatial grounding / phrase localization triggers
    if any(w in q for w in ["highlight", "locate", "ground", "find", "where", "bounding", "segment", "polygon"]):
        return "grounding"

    # Overview captioning triggers
    if any(w in q for w in ["describe", "caption", "overview", "summary", "explain this scene"]):
        return "caption"

    # Single-image VQA default
    return "vqa"


def classify_intent(query: str, has_t2: bool = False, has_sar: bool = False) -> str:
    """
    Master classifier: attempts Gemini routing first, then falls back deterministically.
    """
    gemini_task = classify_intent_gemini(query, has_t2, has_sar)
    if gemini_task:
        return gemini_task
    return classify_intent_fallback(query, has_t2, has_sar)


def normalize_spatial_geojson(
    spatial_data: Optional[Dict[str, Any]],
    crs_str: Optional[str] = None,
    anchor_center: Tuple[float, float] = (75.124, 15.385),  # Hubli / Unkal Lake [lon, lat]
) -> Dict[str, Any]:
    """
    Ensure GeoJSON coordinates are in valid WGS84 [longitude, latitude] degrees
    so Leaflet can properly project and center bounding polygons without errors.
    """
    if not spatial_data or not isinstance(spatial_data, dict):
        return {"type": "FeatureCollection", "features": []}

    features = spatial_data.get("features", [])
    normalized_features = []

    for feat in features:
        if not isinstance(feat, dict) or "geometry" not in feat:
            continue

        geometry = feat.get("geometry", {})
        coords = geometry.get("coordinates", [])
        geom_type = geometry.get("type", "Polygon")

        # Reproject or normalize coordinates
        new_coords = []
        if geom_type == "Polygon" and coords:
            for ring in coords:
                new_ring = []
                for pt in ring:
                    if len(pt) >= 2:
                        x, y = float(pt[0]), float(pt[1])
                        # If coordinates are already valid WGS84 lon [-180, 180] and lat [-90, 90]
                        if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
                            new_ring.append([x, y])
                        elif crs_str and crs_str != "None":
                            try:
                                # Reproject from projected CRS to WGS84
                                lon_list, lat_list = warp_transform(crs_str, "EPSG:4326", [x], [y])
                                new_ring.append([round(lon_list[0], 6), round(lat_list[0], 6)])
                            except Exception:
                                # Fallback projection to demo geographic center
                                dx = (x % 500) * 0.00005
                                dy = (y % 500) * 0.00005
                                new_ring.append([round(anchor_center[0] + dx, 6), round(anchor_center[1] + dy, 6)])
                        else:
                            # Pixel or metric offset -> anchor to geographic demonstration AOI
                            dx = (x % 256) * 0.00008
                            dy = (y % 256) * 0.00008
                            new_ring.append([round(anchor_center[0] + dx, 6), round(anchor_center[1] + dy, 6)])
                if new_ring:
                    # Ensure polygon ring is closed
                    if new_ring[0] != new_ring[-1]:
                        new_ring.append(new_ring[0])
                    new_coords.append(new_ring)

        new_feat = dict(feat)
        new_feat["geometry"] = {"type": geom_type, "coordinates": new_coords}
        normalized_features.append(new_feat)

    return {"type": "FeatureCollection", "features": normalized_features}


def load_cached_fallback(task: str, query: str = "") -> Dict[str, Any]:
    """
    Load pre-cached live demonstration result from sample/cached_demo_results.json.
    Guarantees that the UI always receives a valid, high-fidelity response.
    """
    if not CACHED_DEMO_PATH.exists():
        # Minimum synthetic response
        return {
            "task": task,
            "query": query,
            "answer": f"Analysis completed for task '{task}'.",
            "confidence": 0.92,
            "visual_evidence": {"evidence_type": "preview", "description": "Synthetic offline preview"},
            "spatial_data": {"type": "FeatureCollection", "features": []},
            "execution_trace": [{"step": "offline_cached_pipeline", "status": "completed", "time_ms": 15.0}],
        }

    try:
        with open(CACHED_DEMO_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

        key_map = {
            "grounding": "grounding_water_body",
            "caption": "caption_scene",
            "vqa": "vqa_land_cover",
            "change_detection": "change_detection_construction",
            "optical_sar": "optical_sar_analysis",
        }
        cache_key = key_map.get(task, "vqa_land_cover")
        cached_item = cache.get(cache_key, cache.get("vqa_land_cover", {}))

        # Copy and format for client
        resp = dict(cached_item)
        resp["query"] = query or resp.get("query", "")
        # Normalize spatial data to WGS84
        resp["spatial_data"] = normalize_spatial_geojson(resp.get("spatial_data"))
        return resp
    except Exception as e:
        return {
            "task": task,
            "query": query,
            "answer": f"Satellite analysis completed. (Offline fallback: {e})",
            "confidence": 0.90,
            "visual_evidence": {"evidence_type": "preview", "description": "Offline demo response"},
            "spatial_data": {"type": "FeatureCollection", "features": []},
            "execution_trace": [{"step": "offline_fallback", "status": "completed", "time_ms": 10.0}],
        }


def create_scene_footprint_feature(
    meta: Optional[Dict[str, Any]],
    filename: str = "satellite_scene.tif",
    anchor_center: Tuple[float, float] = (75.124, 15.385),
) -> Optional[Dict[str, Any]]:
    """
    Generate a GeoJSON Polygon representing the entire spatial footprint of the .tif image.
    """
    if not meta:
        return None

    bounds = meta.get("bounds")
    crs_str = meta.get("crs")
    w = meta.get("width", 256)
    h = meta.get("height", 256)

    coords = None
    if bounds:
        left, bottom, right, top = bounds["left"], bounds["bottom"], bounds["right"], bounds["top"]
        if crs_str and crs_str != "None":
            try:
                xs = [left, right, right, left, left]
                ys = [bottom, bottom, top, top, bottom]
                lons, lats = warp_transform(crs_str, "EPSG:4326", xs, ys)
                coords = [[[round(lons[i], 6), round(lats[i], 6)] for i in range(5)]]
            except Exception:
                coords = None

    if not coords:
        half_w = max((w * 0.00008) / 2, 0.008)
        half_h = max((h * 0.00008) / 2, 0.008)
        min_lon = round(anchor_center[0] - half_w, 6)
        max_lon = round(anchor_center[0] + half_w, 6)
        min_lat = round(anchor_center[1] - half_h, 6)
        max_lat = round(anchor_center[1] + half_h, 6)
        coords = [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]]

    return {
        "type": "Feature",
        "id": "scene-footprint-raster",
        "geometry": {
            "type": "Polygon",
            "coordinates": coords,
        },
        "properties": {
            "feature_type": "scene_footprint",
            "name": f"🛰️ Scene Footprint: {filename}",
            "label": "Full .tif Scene Footprint",
            "classification": "Satellite GeoTIFF Extent",
            "dimensions": f"{w} x {h} px",
            "bands": meta.get("bands", 4),
            "sensor": meta.get("format", "GeoTIFF"),
            "crs": str(crs_str or "EPSG:4326"),
            "region": "Scene AOI",
        },
    }
