# Prompt & Guide for Atharv's Antigravity AI Agent

> **Instructions for Atharv:** Copy and paste the prompt below directly into your Antigravity AI chat session to scaffold and wire your FastAPI Agentic Router.

---

```markdown
You are working with me on the SIH project "SatQuery-AI" (Problem Statement: SIH26167).

==================================================
PROJECT CONTEXT & TEAM ARCHITECTURE
==================================================

SatQuery AI is an agentic vision-language assistant for multimodal remote-sensing and satellite image analysis through natural-language queries.

Team Roles:
- Rohan: Frontend (Next.js, dark-mode split-screen, chat on left, Leaflet map on right).
- Atharv (Me): Backend & Agentic Router (FastAPI, Gemini intent routing, tool calling, JSON contract formatting).
- Prathamesh: Vision Engine / Satellite Specialist Layer (pushed to branch `prathamesh`).

Your goal is to build our production FastAPI backend (`backend/main.py` and `backend/router.py`) that:
1. Exposes `GET /health` and `POST /query`.
2. Employs an Agentic Router (Gemini API with an ultra-reliable regex/keyword fallback) to classify user query intent and select the appropriate specialist tool.
3. Invokes Prathamesh's `vision` package functions to execute the requested satellite task.
4. Returns a strictly validated JSON response that includes:
   - `answer`: Natural language findings
   - `confidence`: Float score (0.0 to 1.0)
   - `spatial_data`: GeoJSON FeatureCollection (for Rohan's Leaflet map: `L.geoJSON(data.spatial_data).addTo(map)`)
   - `execution_trace`: Step-by-step progress list
   - `visual_evidence`: Evidence metadata (masks, heatmaps, previews)
5. Never crashes or returns HTTP 500 during live presentation (graceful try/except fallback using pre-cached responses in `sample/cached_demo_results.json`).

==================================================
STEP 1: MERGE PRATHAMESH'S VISION PACKAGE
==================================================

First, fetch and merge Prathamesh's completed vision package from `origin/prathamesh`:

```bash
git fetch origin prathamesh
git merge origin/prathamesh -m "merge: integrate vision specialist package from prathamesh"
```

Verify that the following modules exist and import properly:
- `vision/__init__.py`
- `vision/geospatial.py`
- `vision/detection.py` (provides `run_detection`)
- `vision/change_detection.py` (provides `detect_change`)
- `vision/optical_sar.py` (provides `analyze_optical_sar`)
- `sample/cached_demo_results.json` (offline backup)

Test the import:
```bash
python -c "from vision import run_detection, detect_change, analyze_optical_sar; print('Vision specialists imported successfully!')"
```

==================================================
STEP 2: PRATHAMESH'S VISION SPECIALIST API
==================================================

Prathamesh has exposed clean, high-level specialist functions that return dictionaries formatted for our contract:

1. Unified Detection (Grounding, VQA, Captioning):
   ```python
   from vision import run_detection

   # Returns complete dict with GeoJSON spatial_data and execution_trace
   result_dict = run_detection(image_path, query="Highlight the water body", task="auto")
   ```

2. Bi-Temporal Change Detection (Mandatory SIH task):
   ```python
   from vision import detect_change

   change_res = detect_change(image_t1_path, image_t2_path, prompt=query)
   result_dict = change_res.to_dict()
   ```

3. Optical + SAR Paired Analysis (Mandatory SIH task):
   ```python
   from vision import analyze_optical_sar

   opt_sar_res = analyze_optical_sar(optical_path, sar_path, query=query)
   result_dict = opt_sar_res.to_dict()
   ```

==================================================
STEP 3: THE EXACT CONTRACT EXPECTED BY ROHAN (FRONTEND)
==================================================

Rohan's Next.js frontend calls `POST /query` and directly renders:
- `data.spatial_data` on Leaflet map: `L.geoJSON(data.spatial_data).addTo(map)`
- `data.execution_trace` as an animated live step list
- `data.answer` in the chat message bubble

All functions return a dictionary adhering to this shared contract. 

You can also inspect `sample/cached_demo_results.json` in this repository for full live samples of every task.

---

### Response Format 1: Grounding (Bounding Boxes -> GeoJSON)
```json
{
  "task": "grounding",
  "query": "Highlight the water body",
  "answer": "Located 2 region(s) matching query: 'water body'.",
  "confidence": 0.90,
  "visual_evidence": {
    "evidence_type": "boxes",
    "description": "Grounded 2 region(s) for 'water body'",
    "has_mask": false,
    "box_count": 2,
    "preview_shape": [256, 256, 3]
  },
  "spatial_data": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [500006.28, 5197438.84],
              [502561.16, 5197438.84],
              [502561.16, 5199993.72],
              [500006.28, 5199993.72],
              [500006.28, 5197438.84]
            ]
          ]
        },
        "properties": {
          "label": "water body",
          "score": 0.88,
          "pixel_box": [0.13, 0.13, 255.62, 255.62],
          "geo_coordinates": {
            "min_lon": 500006.28,
            "max_lat": 5199993.72,
            "max_lon": 502561.16,
            "min_lat": 5197438.84
          }
        }
      }
    ]
  },
  "execution_trace": [
    {"step": "geospatial_rgb_stretch", "status": "completed", "time_ms": 12.1},
    {"step": "florence2_phrase_grounding", "status": "completed", "time_ms": 145.3},
    {"step": "pixel_to_geo_projection", "status": "completed", "time_ms": 1.2}
  ]
}
```

---

### Response Format 2: Single-Image VQA & Captioning
```json
{
  "task": "vqa",
  "query": "What is the predominant land cover here?",
  "answer": "The predominant land cover is vegetation / forest canopy. (Visual Context: The image shows agricultural parcels surrounded by dense green vegetation.)",
  "confidence": 0.92,
  "visual_evidence": {
    "evidence_type": "preview",
    "description": "Radiometrically calibrated RGB preview",
    "has_mask": false,
    "box_count": 0,
    "preview_shape": [256, 256, 3]
  },
  "spatial_data": {
    "type": "FeatureCollection",
    "features": []
  },
  "execution_trace": [
    {"step": "geospatial_rgb_stretch", "status": "completed", "time_ms": 8.0},
    {"step": "florence2_scene_perception", "status": "completed", "time_ms": 110.4},
    {"step": "spectral_reasoning_fusion", "status": "completed", "time_ms": 4.2}
  ]
}
```

---

### Response Format 3: Bi-Temporal Change Detection
```json
{
  "task": "change_detection",
  "query": "Find new construction between T1 and T2",
  "answer": "Moderate localized change observed across 5.04% of the scene, identifying 1 distinct modified zones. Pattern consistent with agricultural cycles, cleared parcels, or surface modification. [Addressed Query: 'Find new construction between T1 and T2']",
  "confidence": 0.88,
  "change_percentage": 5.04,
  "clusters_detected": 1,
  "visual_evidence": {
    "evidence_type": "mask",
    "description": "Change mask covering 5.04% surface modification across 1 clusters",
    "has_mask": true,
    "mask_shape": [256, 256],
    "box_count": 1,
    "preview_shape": [256, 256, 3]
  },
  "spatial_data": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [501000.0, 5198400.0],
              [501600.0, 5198400.0],
              [501600.0, 5199000.0],
              [501000.0, 5199000.0],
              [501000.0, 5198400.0]
            ]
          ]
        },
        "properties": {
          "label": "detected_change_zone",
          "score": 0.75,
          "pixel_box": [100.0, 100.0, 160.0, 160.0]
        }
      }
    ]
  },
  "execution_trace": [
    {"step": "validate_spatial_compatibility", "status": "completed", "time_ms": 5.2},
    {"step": "compute_spectral_difference", "status": "completed", "time_ms": 14.1},
    {"step": "otsu_thresholding: value=83.0", "status": "completed", "time_ms": 3.4},
    {"step": "extract_change_bounding_boxes", "status": "completed", "time_ms": 6.8}
  ]
}
```

---

### Response Format 4: Optical + SAR Paired Analysis
```json
{
  "task": "optical_sar",
  "query": "Assess flood impact and urban presence",
  "answer": "Paired Optical+SAR analysis completed. Optical modality: multispectral, SAR modality: sar. Clear atmospheric conditions in optical scene (< 0.0% cloud coverage). Water bodies confirmed via cross-sensor agreement (9.6% coverage): exhibiting low optical reflectance and characteristic SAR specular reflection. Built-up / structural presence verified (4.4% coverage): evidenced by high SAR dihedral backscatter and structured optical edges.",
  "confidence": 0.90,
  "complementary_insights": [
    "Clear atmospheric conditions in optical scene (< 0.0% cloud coverage).",
    "Water bodies confirmed via cross-sensor agreement (9.6% coverage): exhibiting low optical reflectance and characteristic SAR specular reflection.",
    "Built-up / structural presence verified (4.4% coverage): evidenced by high SAR dihedral backscatter and structured optical edges."
  ],
  "optical_features": {
    "mean_brightness": 118.6,
    "cloud_coverage_pct": 0.0,
    "modality": "multispectral"
  },
  "sar_features": {
    "mean_backscatter_intensity": 68.2,
    "roughness_variance": 1240.5,
    "modality": "sar"
  },
  "visual_evidence": {
    "evidence_type": "composite",
    "description": "False-color cross-sensor composite (Red: SAR Backscatter, Green: Optical Green, Blue: Optical Blue)",
    "preview_shape": [256, 256, 3]
  },
  "spatial_data": {
    "type": "FeatureCollection",
    "features": []
  },
  "execution_trace": [
    {"step": "validate_multimodal_pair", "status": "completed", "time_ms": 4.1},
    {"step": "extract_cross_modal_features", "status": "completed", "time_ms": 18.3},
    {"step": "synthesize_complementary_evidence", "status": "completed", "time_ms": 5.7}
  ]
}
```

==================================================
STEP 4: AGENTIC ROUTER & INTENT CLASSIFIER
==================================================

Build `backend/router.py`:
1. Primary Intent Classifier (Google Gemini API):
   - Prompt Gemini with the query and available tool descriptions.
   - Return one of: `"vqa"`, `"caption"`, `"grounding"`, `"change_detection"`, `"optical_sar"`.
2. Fallback Classifier (Zero-Dependency Regex/Keyword):
   If `GEMINI_API_KEY` is unset, rate-limited, or network drops at the hackathon venue, immediately use this deterministic fallback:

```python
def classify_intent_fallback(query: str, has_t2: bool = False, has_sar: bool = False) -> str:
    q = (query or "").lower()
    if has_t2 or any(w in q for w in ["change", "before", "after", "difference", "between"]):
        return "change_detection"
    if has_sar or any(w in q for w in ["sar", "radar", "cloud", "penetrat", "cross-modal"]):
        return "optical_sar"
    if any(w in q for w in ["highlight", "locate", "ground", "find", "where"]):
        return "grounding"
    if any(w in q for w in ["describe", "caption", "overview", "summary"]):
        return "caption"
    return "vqa"
```

==================================================
STEP 5: FASTAPI IMPLEMENTATION
==================================================

Create `backend/main.py`:
1. CORS Middleware:
   Allow origins `["*"]` so Rohan's Next.js dev server on `http://localhost:3000` can connect without restriction.
2. Endpoints:
   - `GET /health` $\rightarrow$ `{"status": "ok", "service": "SatQuery-AI Router"}`
   - `POST /query`:
     - Accepts `image: UploadFile`, `query: str = Form(...)`, optional `image_t2: Optional[UploadFile] = None`, and optional `image_sar: Optional[UploadFile] = None`.
     - Saves uploaded temporary images to `data/uploads/`.
     - Routes query to the proper specialist tool.
3. Fail-Safe Demo Wrapper:
   Wrap every tool call in a `try...except` block. If an error occurs, load the corresponding pre-cached response from `sample/cached_demo_results.json` and return HTTP 200 so the UI never displays an error during evaluation!
4. Launch command:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

Please inspect the codebase, merge `origin/prathamesh`, and implement the FastAPI router now.
```
