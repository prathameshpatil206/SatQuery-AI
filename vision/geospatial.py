"""
SatQuery-AI: Geospatial & Remote Sensing Raster Engine
Provides robust loading, coordinate transforms, dynamic range stretching,
and spatial alignment verification for satellite imagery.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol, xy
from PIL import Image

from vision.types import Modality


def read_geotiff(image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a GeoTIFF and return basic metadata and image data.
    Preserves backward compatibility with the initial vision interface.
    """
    with rasterio.open(image_path) as src:
        image = src.read()
        metadata = {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "dtypes": src.dtypes,
            "nodata": src.nodatavals,
            "is_geospatial": src.crs is not None,
        }

    return image, metadata


def pixel_to_geo(transform: Affine, x: float, y: float) -> Dict[str, float]:
    """
    Convert pixel coordinates (column x, row y) to geographic/projected coordinates.
    """
    lon, lat = xy(transform, y, x)
    return {
        "lat": lat,
        "lon": lon,
    }


def geo_to_pixel(transform: Affine, lon: float, lat: float) -> Dict[str, int]:
    """
    Convert geographic/projected coordinates (lon/easting, lat/northing) to pixel coordinates.
    """
    row, col = rowcol(transform, lon, lat)
    return {
        "x": int(col),
        "y": int(row),
    }


def load_image(
    source: Union[str, Path, np.ndarray, Image.Image]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Polymorphic loader for both geospatial rasters (GeoTIFF/TIFF) and benchmark datasets (PNG/JPEG).

    Returns:
        image: NumPy array in shape (C, H, W)
        metadata: Standardized metadata dictionary
    """
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Image source not found: {source_path}")

        # Attempt geospatial raster loading first
        try:
            with rasterio.open(source_path) as src:
                image = src.read()
                metadata = {
                    "source": str(source_path),
                    "format": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs) if src.crs else None,
                    "transform": src.transform,
                    "bounds": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                    },
                    "dtypes": src.dtypes,
                    "nodata": src.nodatavals,
                    "is_geospatial": src.crs is not None,
                }
                return image, metadata
        except Exception:
            # Fallback to standard image reading (PNG, JPEG, etc.)
            with Image.open(source_path) as pil_img:
                arr = np.array(pil_img)
                # Ensure shape is (C, H, W)
                if arr.ndim == 2:
                    image = arr[np.newaxis, :, :]
                elif arr.ndim == 3:
                    image = np.transpose(arr, (2, 0, 1))
                else:
                    raise ValueError(f"Unexpected image shape: {arr.shape}")

                h, w = image.shape[1], image.shape[2]
                metadata = {
                    "source": str(source_path),
                    "format": pil_img.format or "STANDARD_IMAGE",
                    "width": w,
                    "height": h,
                    "bands": image.shape[0],
                    "crs": None,
                    "transform": None,
                    "bounds": None,
                    "dtypes": (str(image.dtype),) * image.shape[0],
                    "nodata": None,
                    "is_geospatial": False,
                }
                return image, metadata

    elif isinstance(source, Image.Image):
        arr = np.array(source)
        if arr.ndim == 2:
            image = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            image = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected PIL image shape: {arr.shape}")

        metadata = {
            "source": "PIL_Image",
            "format": source.format or "MEMORY_IMAGE",
            "width": image.shape[2],
            "height": image.shape[1],
            "bands": image.shape[0],
            "crs": None,
            "transform": None,
            "bounds": None,
            "dtypes": (str(image.dtype),) * image.shape[0],
            "nodata": None,
            "is_geospatial": False,
        }
        return image, metadata

    elif isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            image = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            # If channels are last, transpose to (C, H, W)
            if arr.shape[2] <= 16 and arr.shape[0] > 16:
                image = np.transpose(arr, (2, 0, 1))
            else:
                image = arr
        else:
            raise ValueError(f"Unexpected NumPy array dimensions: {arr.shape}")

        metadata = {
            "source": "numpy_array",
            "format": "NUMPY",
            "width": image.shape[2],
            "height": image.shape[1],
            "bands": image.shape[0],
            "crs": None,
            "transform": None,
            "bounds": None,
            "dtypes": (str(image.dtype),) * image.shape[0],
            "nodata": None,
            "is_geospatial": False,
        }
        return image, metadata

    else:
        raise TypeError(f"Unsupported image input type: {type(source)}")


def to_display_rgb(
    image_array: np.ndarray,
    bands: Tuple[int, int, int] = (0, 1, 2),
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Convert raw remote-sensing raster (12-bit, 16-bit, float SAR, or multi-band)
    into an 8-bit RGB array suitable for vision models and UI visualization.

    Applies robust 2%-98% percentile contrast stretching to avoid saturation or dark artifacts.

    Returns:
        rgb: uint8 array of shape (H, W, 3)
    """
    # Ensure shape is (C, H, W)
    if image_array.ndim == 2:
        image = image_array[np.newaxis, :, :]
    elif image_array.ndim == 3:
        if image_array.shape[2] <= 16 and image_array.shape[0] > 16:
            image = np.transpose(image_array, (2, 0, 1))
        else:
            image = image_array
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {image_array.shape}")

    c, h, w = image.shape

    # Select RGB bands
    if c >= 3:
        b_idx = [min(b, c - 1) for b in bands]
        selected = image[b_idx, :, :].astype(np.float32)
    elif c == 1:
        # Replicate single band across 3 channels (e.g., panchromatic or SAR)
        selected = np.repeat(image.astype(np.float32), 3, axis=0)
    else:  # c == 2 (e.g., dual-pol SAR VV/VH)
        ch1 = image[0:1, :, :].astype(np.float32)
        ch2 = image[1:2, :, :].astype(np.float32)
        ratio = np.clip(ch1 / (ch2 + 1e-6), 0, 10)
        selected = np.concatenate([ch1, ch2, ratio], axis=0)

    # Apply percentile contrast stretching channel-by-channel
    rgb_channels = []
    p_low, p_high = percentile_clip

    for i in range(3):
        channel = selected[i]
        valid_mask = np.isfinite(channel)
        if not np.any(valid_mask):
            rgb_channels.append(np.zeros((h, w), dtype=np.uint8))
            continue

        v_min = np.percentile(channel[valid_mask], p_low)
        v_max = np.percentile(channel[valid_mask], p_high)

        if v_max <= v_min:
            stretched = np.zeros_like(channel, dtype=np.uint8)
        else:
            stretched = np.clip((channel - v_min) / (v_max - v_min), 0.0, 1.0)
            stretched = (stretched * 255.0).astype(np.uint8)

        rgb_channels.append(stretched)

    rgb = np.stack(rgb_channels, axis=-1)  # Shape (H, W, 3)
    return rgb


def detect_modality(image_array: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Modality:
    """
    Identify imagery modality (Optical, Multispectral, SAR) based on bands, dtype, and statistics.
    """
    bands = image_array.shape[0] if image_array.ndim == 3 else 1

    # Check metadata hints
    if metadata:
        meta_str = str(metadata).lower()
        if "sar" in meta_str or "sentinel-1" in meta_str or "vv" in meta_str or "vh" in meta_str:
            return Modality.SAR
        if "sentinel-2" in meta_str or "landsat" in meta_str:
            return Modality.MULTISPECTRAL

    # Heuristics based on band count and dynamic range
    if bands > 3:
        return Modality.MULTISPECTRAL
    elif bands in (1, 2) and (np.issubdtype(image_array.dtype, np.floating) or "float" in str(image_array.dtype)):
        # High likelihood of SAR backscatter float values
        return Modality.SAR
    elif bands == 3:
        return Modality.OPTICAL
    else:
        return Modality.OPTICAL


def validate_spatial_pair(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """
    Validate spatial compatibility between two images (for bi-temporal change or Optical+SAR pairing).

    Checks:
        1. CRS matching
        2. Bounding box intersection / overlap
        3. Pixel resolution equivalence
    """
    report = {
        "compatible": False,
        "is_geospatial": False,
        "crs_match": False,
        "bounds_overlap": False,
        "resolution_match": False,
        "overlap_percentage": 0.0,
        "warnings": [],
    }

    geo1 = meta1.get("is_geospatial", False)
    geo2 = meta2.get("is_geospatial", False)

    if not geo1 or not geo2:
        # Benchmark image pair (PNG/JPEG) without geospatial tags
        # Dimensional matching check
        dim1 = (meta1.get("width"), meta1.get("height"))
        dim2 = (meta2.get("width"), meta2.get("height"))
        if dim1 == dim2:
            report["compatible"] = True
            report["warnings"].append("Images lack CRS/GeoTIFF tags; verified identical pixel dimensions.")
        else:
            report["warnings"].append(f"Images lack CRS and have unequal dimensions: {dim1} vs {dim2}.")
        return report

    report["is_geospatial"] = True

    # 1. CRS comparison
    crs1 = str(meta1.get("crs"))
    crs2 = str(meta2.get("crs"))
    if crs1 == crs2 and crs1 != "None":
        report["crs_match"] = True
    else:
        report["warnings"].append(f"CRS mismatch: {crs1} vs {crs2}.")

    # 2. Bounding box overlap check
    b1 = meta1.get("bounds")
    b2 = meta2.get("bounds")
    if b1 and b2:
        inter_left = max(b1["left"], b2["left"])
        inter_right = min(b1["right"], b2["right"])
        inter_bottom = max(b1["bottom"], b2["bottom"])
        inter_top = min(b1["top"], b2["top"])

        if inter_left < inter_right and inter_bottom < inter_top:
            report["bounds_overlap"] = True
            inter_area = (inter_right - inter_left) * (inter_top - inter_bottom)
            area1 = (b1["right"] - b1["left"]) * (b1["top"] - b1["bottom"])
            area2 = (b2["right"] - b2["left"]) * (b2["top"] - b2["bottom"])
            union_area = area1 + area2 - inter_area
            report["overlap_percentage"] = (inter_area / union_area) * 100.0 if union_area > 0 else 0.0
        else:
            report["warnings"].append("Bounding boxes do not overlap spatially.")

    # 3. Resolution check
    t1 = meta1.get("transform")
    t2 = meta2.get("transform")
    if t1 and t2:
        res1 = (abs(t1[0]), abs(t1[4]))
        res2 = (abs(t2[0]), abs(t2[4]))
        if abs(res1[0] - res2[0]) < tolerance and abs(res1[1] - res2[1]) < tolerance:
            report["resolution_match"] = True
        else:
            report["warnings"].append(f"Pixel resolution mismatch: {res1} vs {res2}.")

    # Overarching compatibility
    report["compatible"] = report["crs_match"] and report["bounds_overlap"]
    return report


def create_synthetic_geotiff(
    output_path: Union[str, Path],
    width: int = 256,
    height: int = 256,
    bands: int = 4,
    crs: str = "EPSG:32632",
    pixel_size: float = 10.0,
    origin: Tuple[float, float] = (500000.0, 5200000.0),
) -> Path:
    """
    Generate a valid multi-band synthetic GeoTIFF for automated verification and testing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = Affine(pixel_size, 0.0, origin[0], 0.0, -pixel_size, origin[1])

    # Synthesize realistic multi-spectral satellite values (12-bit range: 0 - 4095)
    np.random.seed(42)
    data = np.random.randint(200, 3500, size=(bands, height, width), dtype=np.uint16)

    # Add artificial patterns (e.g. simulated river, agricultural fields)
    data[:, 50:100, 50:150] = 500   # Water-like low reflectance
    data[3, 120:200, 100:220] = 3200  # High NIR (vegetation)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return output_path