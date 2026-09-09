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
- `data.execution_trace` as an animated step list
- `data.answer` in the chat bubble

Here is the exact JSON structure your FastAPI endpoint MUST return:

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
    "box_count": 2
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
              [77.209, 28.613],
              [77.215, 28.613],
              [77.215, 28.618],
              [77.209, 28.618],
              [77.209, 28.613]
            ]
          ]
        },
        "properties": {
          "label": "water body",
          "score": 0.88,
          "pixel_box": [50.0, 50.0, 150.0, 200.0]
        }
      }
    ]
  },
  "execution_trace": [
    {"step": "Load GeoTIFF & extract CRS (EPSG:32632)", "status": "completed", "time_ms": 12.4},
    {"step": "2%-98% radiometric contrast stretching", "status": "completed", "time_ms": 7.8},
    {"step": "Florence-2 zero-shot grounding inference", "status": "completed", "time_ms": 135.2},
    {"step": "Convert pixel bboxes to GeoJSON WGS84 coordinates", "status": "completed", "time_ms": 1.5}
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
