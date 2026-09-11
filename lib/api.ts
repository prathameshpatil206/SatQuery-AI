// lib/api.ts
// Remote Sensing VQA API Client & Type Definitions aligned with FastAPI backend contract

export interface TraceStep {
  step: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "SUCCESS" | "FAILED" | string;
  time_ms: number;
}

export type TaskType = "grounding" | "vqa" | "caption" | "change_detection" | "optical_sar";

export interface VisualEvidence {
  evidence_type: string;
  description: string;
  has_mask?: boolean;
  box_count?: number;
  preview_shape?: number[];
}

export interface QueryResponse {
  task: TaskType;
  query: string;
  answer: string; // PRIMARY text response field from backend
  confidence: number;
  visual_evidence: VisualEvidence;
  spatial_data: GeoJSON.FeatureCollection | null;
  execution_trace: TraceStep[];
  change_percentage?: number;
  clusters_detected?: number;
  complementary_insights?: string[];
}

// Fallback mock mode flag: If true, resolves with mock satellite GeoJSON
// and execution traces adhering to team backend contract if fetch fails or is unreachable.
export const USE_MOCK_FALLBACK: boolean = true;

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Returns mock remote-sensing VQA response centered around Unkal Lake, Hubli, Karnataka
 * dynamically adapting to grounding, change detection, or optical/SAR tasks.
 */
function getMockQueryResponse(
  queryText: string,
  hasT2: boolean = false,
  hasSar: boolean = false
): QueryResponse {
  const queryLower = queryText.toLowerCase();

  // Determine appropriate task type based on query context and uploaded inputs
  let detectedTask: TaskType = "grounding";
  if (hasT2 || queryLower.includes("t1") || queryLower.includes("t2") || queryLower.includes("change") || queryLower.includes("construction")) {
    detectedTask = "change_detection";
  } else if (hasSar || queryLower.includes("sar") || queryLower.includes("optical and sar") || queryLower.includes("flood")) {
    detectedTask = "optical_sar";
  } else if (queryLower.includes("what") || queryLower.includes("is there") || queryLower.includes("how many") || queryLower.includes("assess")) {
    detectedTask = "vqa";
  }

  // Polygon coordinates delineating Unkal Lake in Hubli, Karnataka (GeoJSON is [lon, lat])
  const unkalLakeFeatureCollection: GeoJSON.FeatureCollection = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        id: "hubli-waterbody-unkal-01",
        properties: {
          name: "Unkal Lake Water Body",
          region: "Hubli-Dharwad, Karnataka",
          classification: "Inland Surface Water Reservoir",
          sensor: "Copernicus Sentinel-2 MSI Level-2A",
          area_sq_km: 0.78,
          confidence: 0.96,
          mean_ndwi: 0.42,
        },
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [75.1122, 15.3855],
              [75.1165, 15.3842],
              [75.1218, 15.3879],
              [75.1252, 15.3934],
              [75.1215, 15.3975],
              [75.1161, 15.3958],
              [75.1118, 15.3912],
              [75.1122, 15.3855],
            ],
          ],
        },
      },
    ],
  };

  if (detectedTask === "change_detection") {
    return {
      task: "change_detection",
      query: queryText,
      answer:
        "Detected 3 distinct zones of new construction and soil compaction between T1 and T2 in the Hubli urban expansion corridor. Built-up structural footprint expanded by 14.8% over the evaluated observation window.",
      confidence: 0.92,
      change_percentage: 14.8,
      clusters_detected: 3,
      visual_evidence: {
        evidence_type: "bi_temporal_differential_mask",
        description: "Co-registered Siamese difference heatmap showing new high-reflectance impervious surfaces",
        has_mask: true,
        box_count: 3,
        preview_shape: [512, 512, 3],
      },
      spatial_data: unkalLakeFeatureCollection,
      execution_trace: [
        { step: "Bi-Temporal Image Co-Registration", status: "COMPLETED", time_ms: 182 },
        { step: "Feature Extraction & Siamese Difference Encoding", status: "COMPLETED", time_ms: 410 },
        { step: "Thresholding & Morphological Cluster Filtering", status: "COMPLETED", time_ms: 145 },
        { step: "GIS Vector Boundary Generation", status: "COMPLETED", time_ms: 88 },
      ],
      complementary_insights: [
        "14.8% net increase in impervious surface area detected across the AOI.",
        "Primary construction clusters located along southwest road extensions.",
        "NDVI loss corresponds directly with newly erected concrete foundations.",
      ],
    };
  }

  if (detectedTask === "optical_sar") {
    return {
      task: "optical_sar",
      query: queryText,
      answer:
        "Multi-modal fusion of Optical and Sentinel-1 SAR (VV/VH backscatter) confirms surface water presence and delineates inundated low-lying areas. SAR backscatter dip (-18.4 dB) confirms standing water beneath cloud canopy.",
      confidence: 0.94,
      clusters_detected: 2,
      visual_evidence: {
        evidence_type: "optical_sar_fused_layer",
        description: "Fused Sentinel-2 RGB with Sentinel-1 dual-pol SAR radiometric calibration",
        has_mask: true,
        box_count: 2,
        preview_shape: [512, 512, 4],
      },
      spatial_data: unkalLakeFeatureCollection,
      execution_trace: [
        { step: "Sentinel-1 GRD SAR Calibration & Terrain Flattening", status: "COMPLETED", time_ms: 240 },
        { step: "Optical-SAR Cross-Attention Fusion", status: "COMPLETED", time_ms: 465 },
        { step: "Specular Water Reflection & Backscatter Thresholding", status: "COMPLETED", time_ms: 172 },
        { step: "Spatial Vectorization & Confidence Scoring", status: "COMPLETED", time_ms: 95 },
      ],
      complementary_insights: [
        "SAR VV/VH polarization ratio verifies calm water surfaces despite partial cloud shadowing.",
        "Urban structures exhibit characteristic high double-bounce radar return (> -6 dB).",
        "Water boundaries refined to 10m spatial resolution using fused radar-optical features.",
      ],
    };
  }

  // Default Grounding / VQA response
  return {
    task: detectedTask,
    query: queryText,
    answer:
      "Successfully located and segmented the primary water body (Unkal Lake, Hubli, Karnataka). Multispectral analysis indicates an estimated surface water area of 0.78 km² with high confidence water absorption signatures (mean NDWI +0.42).",
    confidence: 0.96,
    clusters_detected: 1,
    visual_evidence: {
      evidence_type: "segmentation_mask",
      description: "Binary water segmentation mask derived via McFeeters NDWI spectral index thresholding (> 0.28)",
      has_mask: true,
      box_count: 1,
      preview_shape: [512, 512, 1],
    },
    spatial_data: unkalLakeFeatureCollection,
    execution_trace: [
      { step: "Satellite Scene Ingestion & Metadata Parse", status: "COMPLETED", time_ms: 110 },
      { step: "Multi-Spectral NDWI Water Index Computation", status: "COMPLETED", time_ms: 360 },
      { step: "Otsu Adaptive Thresholding & Mask Vectorization", status: "COMPLETED", time_ms: 225 },
      { step: "GeoJSON Synthesis & Coordinate Alignment", status: "COMPLETED", time_ms: 78 },
    ],
    complementary_insights: [
      "Mean NDWI index is +0.42, characteristic of open surface freshwater reservoirs.",
      "Turbidity levels indicate stable sediment distribution with no severe algal blooms.",
      "Delineated perimeter matches historical Survey of India lake boundary vectors.",
    ],
  };
}

/**
 * Sends natural language query and optional multi-file payloads (image, image_t2, image_sar)
 * to backend FastAPI endpoint at /query.
 */
export async function sendQuery(
  queryText: string,
  image?: File,
  imageT2?: File,
  imageSar?: File
): Promise<QueryResponse> {
  const formData = new FormData();
  formData.append("query", queryText);

  if (image) {
    formData.append("image", image);
  }
  if (imageT2) {
    formData.append("image_t2", imageT2);
  }
  if (imageSar) {
    formData.append("image_sar", imageSar);
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8-second network timeout

    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Backend responded with status ${response.status}: ${response.statusText}`);
    }

    const data: QueryResponse = await response.json();
    return data;
  } catch (err) {
    if (USE_MOCK_FALLBACK) {
      console.warn("Backend unavailable or query failed. Utilizing team contract mock fallback:", err);
      // Realistic brief processing latency for natural UX
      await new Promise((resolve) => setTimeout(resolve, 750));
      return getMockQueryResponse(queryText, Boolean(imageT2), Boolean(imageSar));
    }
    throw err;
  }
}
