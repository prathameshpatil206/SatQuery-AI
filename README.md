# SatQuery-AI: Agentic Multimodal Remote-Sensing Assistant

SatQuery AI is an agentic vision-language assistant for multimodal remote-sensing and satellite image analysis through natural-language queries (Problem Statement: SIH26167).

This integrated system unifies:
1. **Frontend Geospatial Dashboard (Rohan)**: Next.js 14, Tailwind CSS (Slate-950 Dark Theme), Leaflet interactive map with GeoJSON overlays, split-screen UI, and live trace animations.
2. **Backend & Agentic Router (Atharv)**: FastAPI application, Gemini intent classifier with deterministic keyword fallback, SQLite query history persistence, and JSON contract validation.
3. **Vision & Satellite Specialist Layer (Prathamesh)**: Florence-2 zero-shot grounding, VQA, scene captioning, OpenCV CVA bi-temporal change detection, and optical+SAR paired cross-modal analysis.

---

## 🌟 Key Features

- **Split-Screen Interactive Workspace**: 40% query controls & animated execution trace sidebar with 60% interactive Leaflet GIS map.
- **Multimodal Remote Sensing**: Supports single-band, optical RGB, multi-spectral (Sentinel-2), SAR radar (Sentinel-1), and bi-temporal image pairs ($T_1$, $T_2$).
- **Agentic Intent Routing**: Automatically classifies queries into Visual Grounding, Land-cover VQA, Scene Captioning, Bi-temporal Change Detection, or Optical+SAR cross-modal analysis.
- **Projected GeoJSON Output**: Converts pixel bounding boxes and segmented features into projected GeoJSON `FeatureCollection` overlays for direct Leaflet rendering.
- **SQLite Persistence**: Automatically records every query, spatial GeoJSON feature, execution trace, and confidence metric for auditability and replay.
- **Venue-Ready Fail-Safe**: Try/except fallback to pre-computed cached responses (`sample/cached_demo_results.json`), guaranteeing zero 500 errors during live hackathon demos.

---

## 🏗️ Architecture

```
                       User Natural Language Query
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │         Next.js Split-Screen UI           │
             │   (Query Form, File Uploads, Leaflet Map) │
             └─────────────────────┬─────────────────────┘
                                   │ POST /query (FormData)
                                   ▼
             ┌───────────────────────────────────────────┐
             │       FastAPI Backend & Router            │
             │  • Gemini / Deterministic Intent Routing  │
             │  • SQLite Logging & Query History         │
             └─────────────────────┬─────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Single-Image VQA │     │   Bi-Temporal    │     │   Optical+SAR    │
│  & Grounding     │     │ Change Detection │     │ Cross-Modal      │
│  (Florence-2)    │     │   (CVA + Otsu)   │     │ (Dual-Sensor)    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Standardized JSON Contract & GeoJSON      │
             │ (answer, confidence, spatial_data, trace) │
             └───────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (using `.venv`)
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install backend dependencies (if needed)
pip install fastapi uvicorn python-multipart

# Start the FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
# Install frontend dependencies
npm install

# Start Next.js development server
npm run dev
```

Visit **`http://localhost:3000`** in your browser.

### 4. Or Run Both Concurrently
```bash
./start_services.sh
```

---

## 🧪 Verification & Testing

Run the vision specialist unit tests:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```
