# SatQuery AI — Remote Sensing Visual Question Answering (VQA)

SatQuery AI is a high-performance, dark-mode satellite imagery analysis and Visual Question Answering (VQA) platform. Built with Next.js, Tailwind CSS, and Leaflet, it provides an intuitive split-screen workspace to inspect satellite imagery (Optical, SAR, and Temporal) alongside AI-driven spatial execution traces.

---

## 🌟 Key Features

- **Split-Screen Workspace:** Responsive 40% query controls & execution sidebar with 60% interactive map workspace.
- **Multi-Modal Imagery Support:** Accepts standard RGB optical feeds, multi-temporal optical imagery (`t2`), and Synthetic Aperture Radar (`SAR`) bands.
- **Interactive Spatial Visualization:** Integrated Leaflet map with GeoJSON overlays to highlight queried regions, bounding boxes, and geospatial detections.
- **Transparent Execution Trace:** Granular timing breakdown (`time_ms`), step-by-step model reasoning, and confidence scoring for every query.
- **Client-Side Fallback:** Resilient mock backend toggle (`USE_MOCK_FALLBACK`) allowing seamless frontend testing and UI inspection when offline or during API maintenance.

---

## 🏗️ Architecture & Tech Stack

- **Framework:** [Next.js](https://nextjs.org/) (App Router, TypeScript)
- **Styling:** [Tailwind CSS](https://tailwindcss.com/) (Slate-950 Dark Mode Theme)
- **Geospatial Workspace:** [React-Leaflet](https://react-leaflet.js.org/) & Leaflet
- **Backend Service Contract:** FastAPI endpoint (`/query`) accepting multi-modal raster uploads and returning structured spatial GeoJSON payload.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Node.js (v18+ recommended) and `npm` installed on your machine.

### Installation

1. **Clone the repository and switch to the `rohan` branch:**
   ```bash
   git clone [https://github.com/prathameshpatil206/SatQuery-AI.git](https://github.com/prathameshpatil206/SatQuery-AI.git)
   cd SatQuery-AI
   git checkout rohan
