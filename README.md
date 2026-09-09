# SatQuery-AI: Vision & Satellite Specialist Layer

SatQuery AI is an agentic vision-language assistant for multimodal remote-sensing and satellite image analysis through natural-language queries.

This package provides the **Vision / Satellite Specialist Layer** — an engine of specialized tools designed to be called by the SatQuery Agent / Router.

---

## Architecture Overview

```
User Query
    ↓
SatQuery Agent / Router
    ↓
Select Specialist Tool:
  ├── vqa(image, question)
  ├── caption(image)
  ├── ground(image, text)
  ├── detect_change(img1, img2, prompt)
  └── analyze_optical_sar(optical_image, sar_image, query)
    ↓
Structured Response:
  Evidence + Answer + Confidence + Execution Trace
```

---

## Specialist Tool Interfaces

All tools accept either **GeoTIFF file paths**, **standard benchmark images (PNG/JPEG)**, **NumPy arrays**, or **PIL Images**.

```python
from vision import (
    vqa,
    caption,
    ground,
    detect_change,
    analyze_optical_sar,
)

# 1. Single-Image Visual Question Answering (Mandatory)
vqa_result = vqa("path/to/image.tif", "What type of land cover dominates this region?")
print(vqa_result.answer)
print(vqa_result.confidence)
print(vqa_result.execution_trace)

# 2. Remote Sensing Captioning & Grounding
caption_result = caption("path/to/image.tif")
print(caption_result.caption)

grounding_result = ground("path/to/image.tif", "water body")
for box in grounding_result.boxes:
    print(box.box_2d, box.geo_coordinates)

# 3. Bi-Temporal Change Detection (Mandatory)
change_result = detect_change("t1.tif", "t2.tif", prompt="Detect deforestation")
print(change_result.summary)
print(f"Area changed: {change_result.change_ratio * 100:.2f}%")
print("Binary Mask:", change_result.change_mask.shape)
print("Difference Heatmap:", change_result.difference_heatmap.shape)

# 4. Optical + SAR Paired Analysis (Mandatory)
optical_sar_result = analyze_optical_sar("optical.tif", "sar.tif", query="Assess flood impact")
print(optical_sar_result.analysis_summary)
print("Complementary Insights:", optical_sar_result.complementary_insights)
print("Composite False-Color Image:", optical_sar_result.visual_evidence.preview_rgb.shape)
```

---

## Structured Output Contracts

All specialist functions return typed, structured dataclasses (`VQAResult`, `CaptionResult`, `GroundingResult`, `ChangeResult`, `OpticalSARResult`) that include:
- **`answer` / `summary`**: Natural-language findings.
- **`confidence`**: Float confidence score (0.0 to 1.0).
- **`visual_evidence`**: Containing binary masks, continuous heatmaps, bounding boxes with lat/lon coordinates, and displayable 8-bit RGB previews.
- **`execution_trace`**: Processing duration (ms), device used (`cuda` / `cpu`), model version, and spatial verification reports.

---

## Geospatial & Raster Engine

Located in `vision/geospatial.py`:
- `read_geotiff(path)`: Reads GeoTIFF raster arrays and extracts CRS, bounds, and affine transform.
- `pixel_to_geo(transform, x, y)` / `geo_to_pixel(transform, lon, lat)`: Bidirectional coordinate transformation.
- `to_display_rgb(array, bands=(0,1,2), percentile_clip=(2, 98))`: 2%-98% percentile contrast stretching from 12-bit/16-bit multi-spectral or float SAR to normalized 8-bit RGB.
- `detect_modality(array, metadata)`: Identifies `OPTICAL`, `MULTISPECTRAL`, or `SAR`.
- `validate_spatial_pair(meta1, meta2)`: Verifies CRS alignment, bounding box intersection, and resolution matching.

---

## Running Verification Tests

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```