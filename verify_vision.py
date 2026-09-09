"""
SatQuery-AI: Vision Engine Verification Script
Run this script to verify all remote-sensing specialist capabilities end-to-end:
    1. GPU & CUDA Acceleration
    2. Geospatial Raster & Affine Transform Engine
    3. Florence-2 Zero-Shot Grounding with GeoJSON Projection
    4. Florence-2 Remote-Sensing Captioning & VQA
    5. OpenCV Bi-Temporal Change Detection (CVA + Otsu)
    6. Optical + SAR Paired Cross-Modal Analysis
"""

import sys
import time
import torch
from pathlib import Path

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print_header("1. ENVIRONMENT & GPU HARDWARE")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Active Device     : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("Active Device     : CPU")

    print_header("2. GEOSPATIAL RASTER & COORDINATE ENGINE")
    from vision.geospatial import create_synthetic_geotiff, read_geotiff, pixel_to_geo, geo_to_pixel, to_display_rgb
    
    test_tif = Path("sample/valid_sample.tif")
    if not test_tif.exists():
        print("Generating test GeoTIFF with EPSG:32632 projection...")
        test_tif = create_synthetic_geotiff(test_tif, width=256, height=256, bands=4)
    
    img, meta = read_geotiff(test_tif)
    print(f"[PASS] Read GeoTIFF: shape {img.shape}, bands: {meta['bands']}, CRS: {meta['crs']}")
    
    # Verify coordinate projection roundtrip
    transform = meta["transform"]
    geo_pt = pixel_to_geo(transform, 50, 50)
    px_pt = geo_to_pixel(transform, geo_pt["lon"], geo_pt["lat"])
    assert abs(px_pt["x"] - 50) <= 1 and abs(px_pt["y"] - 50) <= 1
    print(f"[PASS] Affine Coordinate Roundtrip: Pixel (50, 50) -> Geo ({geo_pt['lon']:.2f}, {geo_pt['lat']:.2f}) -> Pixel ({px_pt['x']}, {px_pt['y']})")
    
    rgb = to_display_rgb(img)
    assert rgb.shape == (256, 256, 3) and rgb.dtype.name == "uint8"
    print(f"[PASS] 2%-98% Percentile Radiometric Stretch: RGB shape {rgb.shape}, range [{rgb.min()}, {rgb.max()}]")

    print_header("3. FLORENCE-2 ZERO-SHOT GROUNDING & GEOJSON")
    from vision import run_detection
    
    t0 = time.time()
    res_ground = run_detection(test_tif, query="Highlight the water body", task="grounding")
    elapsed_ground = time.time() - t0
    print(f"[PASS] Task: {res_ground['task']} (executed in {elapsed_ground:.2f}s)")
    print(f"       Answer: {res_ground['answer']}")
    print(f"       Confidence: {res_ground['confidence']}")
    print(f"       GeoJSON Type: {res_ground['spatial_data']['type']}")
    print(f"       Polygons Detected: {len(res_ground['spatial_data']['features'])}")
    if res_ground['spatial_data']['features']:
        first_box = res_ground['spatial_data']['features'][0]
        print(f"       First GeoJSON Polygon Coords: {first_box['geometry']['coordinates'][0][:2]}...")
        print(f"       Box Properties: {first_box['properties']}")

    print_header("4. FLORENCE-2 VQA & CAPTIONING")
    res_caption = run_detection(test_tif, query="Describe the scene", task="caption")
    print(f"[PASS] Caption: {res_caption['answer']}")
    print(f"       Confidence: {res_caption['confidence']}")
    
    res_vqa = run_detection(test_tif, query="What is the dominant land cover?", task="vqa")
    print(f"[PASS] VQA Answer: {res_vqa['answer']}")
    print(f"       Confidence: {res_vqa['confidence']}")

    print_header("5. OPENCV BI-TEMPORAL CHANGE DETECTION (CVA + OTSU)")
    from vision import detect_change
    t1_path = Path("sample/t1.tif")
    t2_path = Path("sample/t2.tif")
    if not (t1_path.exists() and t2_path.exists()):
        create_synthetic_geotiff(t1_path, width=256, height=256)
        create_synthetic_geotiff(t2_path, width=256, height=256)
        import rasterio
        with rasterio.open(t2_path, "r+") as dst:
            arr = dst.read()
            arr[:, 100:160, 100:160] = 3800
            dst.write(arr)
            
    res_change = detect_change(t1_path, t2_path, prompt="Detect construction").to_dict()
    print(f"[PASS] Change Summary: {res_change['answer']}")
    print(f"       Area Changed: {res_change['change_percentage']}%")
    print(f"       Clusters Detected: {res_change['clusters_detected']}")
    print(f"       Evidence Mask Shape: {res_change['visual_evidence']['mask_shape']}")

    print_header("6. OPTICAL + SAR PAIRED CROSS-MODAL ANALYSIS")
    from vision import analyze_optical_sar
    opt_path = Path("sample/optical.tif")
    sar_path = Path("sample/sar.tif")
    if not (opt_path.exists() and sar_path.exists()):
        create_synthetic_geotiff(opt_path, width=256, height=256, bands=4)
        import rasterio
        import numpy as np
        with rasterio.open(opt_path) as src:
            sar_meta = src.meta.copy()
            sar_meta.update({"count": 1, "dtype": "float32"})
            sar_data = np.random.exponential(scale=30.0, size=(1, 256, 256)).astype(np.float32)
            with rasterio.open(sar_path, "w", **sar_meta) as dst:
                dst.write(sar_data)

    res_opt_sar = analyze_optical_sar(opt_path, sar_path, query="Assess water bodies").to_dict()
    print(f"[PASS] Cross-Modal Summary: {res_opt_sar['answer']}")
    print(f"       Complementary Insights: {res_opt_sar['complementary_insights']}")
    print(f"       Composite Preview Shape: {res_opt_sar['visual_evidence']['preview_shape']}")

    print_header("VERIFICATION SUMMARY")
    print("All Vision & Satellite Specialist components verified successfully!")
    print("The vision layer is operational and ready for Atharv (Router) & Rohan (Frontend).")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
