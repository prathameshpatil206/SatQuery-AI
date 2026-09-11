"use client";

import React, { useEffect, useCallback, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, ImageOverlay, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { RasterOverlay } from "@/lib/api";

// Fix default Leaflet icon asset resolution in Next.js client environments
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

/**
 * Internal useMap() helper that automatically calculates raster or GeoJSON bounds
 * and calls map.fitBounds() with padding on data updates.
 */
function MapBoundsFitter({
  geojsonData,
  rasterOverlay,
}: {
  geojsonData: GeoJSON.FeatureCollection | null;
  rasterOverlay?: RasterOverlay | null;
}) {
  const map = useMap();

  useEffect(() => {
    try {
      if (rasterOverlay?.bounds && rasterOverlay.bounds.length === 2) {
        const b = L.latLngBounds(rasterOverlay.bounds[0], rasterOverlay.bounds[1]);
        if (b.isValid()) {
          map.fitBounds(b, {
            padding: [45, 45],
            maxZoom: 16,
            animate: true,
          });
          return;
        }
      }

      if (geojsonData && geojsonData.features && geojsonData.features.length > 0) {
        const geoJsonLayer = L.geoJSON(geojsonData as any);
        const bounds = geoJsonLayer.getBounds();
        if (bounds && bounds.isValid()) {
          map.fitBounds(bounds, {
            padding: [45, 45],
            maxZoom: 16,
            animate: true,
          });
        }
      }
    } catch (err) {
      console.warn("Could not calculate spatial bounds for map fitting:", err);
    }
  }, [geojsonData, rasterOverlay, map]);

  return null;
}

/**
 * Overlay control for quick recentering
 */
function MapNavigationControls({
  rasterOverlay,
  spatialData,
}: {
  rasterOverlay?: RasterOverlay | null;
  spatialData: GeoJSON.FeatureCollection | null;
}) {
  const map = useMap();

  const handleRecenterHubli = useCallback(() => {
    map.flyTo([15.3647, 75.124], 12, { duration: 1.0 });
  }, [map]);

  const handleZoomToTif = useCallback(() => {
    try {
      if (rasterOverlay?.bounds) {
        const b = L.latLngBounds(rasterOverlay.bounds[0], rasterOverlay.bounds[1]);
        if (b.isValid()) {
          map.fitBounds(b, { padding: [40, 40], maxZoom: 16, animate: true });
          return;
        }
      }
      if (spatialData?.features?.length) {
        const geoJsonLayer = L.geoJSON(spatialData as any);
        const b = geoJsonLayer.getBounds();
        if (b.isValid()) {
          map.fitBounds(b, { padding: [40, 40], maxZoom: 16, animate: true });
        }
      }
    } catch (err) {
      console.warn("Could not fly to .tif bounds:", err);
    }
  }, [map, rasterOverlay, spatialData]);

  const hasRasterTarget = Boolean(rasterOverlay?.bounds || spatialData?.features?.length);

  return (
    <div className="leaflet-top leaflet-right" style={{ pointerEvents: "auto", margin: "14px", display: "flex", gap: "8px", flexDirection: "column", alignItems: "flex-end" }}>
      {hasRasterTarget && (
        <button
          type="button"
          onClick={handleZoomToTif}
          className="bg-amber-950/90 hover:bg-amber-900 text-amber-300 border border-amber-600/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono shadow-2xl transition-all hover:border-amber-400 flex items-center gap-1.5 active:scale-95"
          title="Zoom directly to .tif scene coverage"
        >
          <span>🛰️ Zoom to .tif Extent</span>
        </button>
      )}
      <button
        type="button"
        onClick={handleRecenterHubli}
        className="bg-slate-900/90 hover:bg-slate-800 text-blue-400 border border-slate-700/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono shadow-2xl transition-all hover:border-blue-500 flex items-center gap-1.5 active:scale-95"
        title="Recenter view to Hubli / Unkal Lake [15.3647, 75.124]"
      >
        <svg className="w-3.5 h-3.5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
        <span>Recenter Hubli</span>
      </button>
    </div>
  );
}

export interface MapProps {
  spatialData: GeoJSON.FeatureCollection | null;
  rasterOverlay?: RasterOverlay | null;
}

export default function Map({ spatialData, rasterOverlay }: MapProps) {
  // Basemap switcher state: default to high-res Satellite imagery (ESRI)
  const [basemap, setBasemap] = useState<"satellite" | "dark">("satellite");

  // Layer toggles
  const [showRaster, setShowRaster] = useState<boolean>(true);
  const [rasterOpacity, setRasterOpacity] = useState<number>(0.88);
  const [showFootprint, setShowFootprint] = useState<boolean>(true);
  const [showDetections, setShowDetections] = useState<boolean>(true);

  // Hubli / Unkal Lake, Karnataka coordinates
  const defaultCenter: [number, number] = [15.3647, 75.124];
  const defaultZoom = 12;

  // Dual polygon styling: Amber dashed line for .tif scene footprint, Cyan/Blue solid for AI detections
  const polygonStyle = (feature: any) => {
    const isFootprint =
      feature?.properties?.feature_type === "scene_footprint" ||
      feature?.properties?.type === "scene_footprint" ||
      feature?.id === "scene-footprint" ||
      feature?.id === "scene-footprint-raster" ||
      feature?.id === "scene-footprint-hubli";

    if (isFootprint) {
      if (!showFootprint) {
        return { stroke: false, fill: false, opacity: 0 };
      }
      return {
        stroke: true,
        color: "#f59e0b", // Amber-500
        weight: 3,
        dashArray: "8, 6",
        opacity: 0.95,
        fillColor: "#fbbf24", // Amber-400
        fillOpacity: 0.05,
      };
    }

    if (!showDetections) {
      return { stroke: false, fill: false, opacity: 0 };
    }

    return {
      stroke: true,
      color: "#06b6d4", // Vibrant Cyan-500
      weight: 2.5,
      opacity: 0.95,
      fillColor: "#38bdf8", // Sky-400
      fillOpacity: 0.2,
    };
  };

  // Attach interactive dark popups to GeoJSON features
  const onEachFeature = (feature: any, layer: L.Layer) => {
    if (!feature || !feature.properties) return;

    const props = feature.properties;
    const isFootprint =
      props.feature_type === "scene_footprint" ||
      props.type === "scene_footprint" ||
      feature.id === "scene-footprint" ||
      feature.id === "scene-footprint-raster" ||
      feature.id === "scene-footprint-hubli";

    const scoreVal = props.confidence ?? props.score;
    const title = props.name || props.label || (isFootprint ? "🛰️ Satellite Scene Footprint" : "AI Detected Region");
    const headerColor = isFootprint ? "#f59e0b" : "#38bdf8";

    const popupContent = `
      <div style="font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: #f1f5f9; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; min-width: 230px;">
        <div style="font-weight: 700; color: ${headerColor}; font-size: 13px; margin-bottom: 6px; border-bottom: 1px solid #1e293b; padding-bottom: 4px;">
          ${title}
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Role:</span>
          <span style="font-weight: 500; color: #e2e8f0;">${isFootprint ? "Entire .tif Scene Coverage" : "AI Localized Feature"}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Classification:</span>
          <span style="font-weight: 600; color: ${isFootprint ? "#f59e0b" : "#38bdf8"};">${props.classification || props.label || "Segmented Feature"}</span>
        </div>
        ${props.dimensions ? `
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Raster Extent:</span>
          <span style="font-weight: 600; color: #34d399;">${props.dimensions}</span>
        </div>` : ""}
        ${props.area_sq_km ? `
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Area Extent:</span>
          <span style="font-weight: 600; color: #34d399;">${props.area_sq_km} km²</span>
        </div>` : ""}
        ${scoreVal !== undefined ? `
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Confidence:</span>
          <span style="font-weight: 600; color: #facc15;">${Math.round(scoreVal * 100)}%</span>
        </div>` : ""}
        ${props.crs ? `
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">CRS:</span>
          <span style="font-weight: 500; color: #94a3b8;">${props.crs}</span>
        </div>` : ""}
        ${props.sensor ? `
        <div style="font-size: 10px; color: #64748b; margin-top: 6px; border-top: 1px solid #1e293b; padding-top: 4px;">
          Sensor / Source: ${props.sensor}
        </div>` : ""}
      </div>
    `;

    layer.bindPopup(popupContent, {
      className: "geoagent-dark-popup",
    });
  };

  return (
    <div className="relative h-full w-full bg-slate-950">
      <MapContainer
        center={defaultCenter}
        zoom={defaultZoom}
        className="h-full w-full bg-slate-950"
        zoomControl={true}
        scrollWheelZoom={true}
      >
        {/* Basemap: ESRI High-Resolution Satellite or Watermark-Free Dark Vector */}
        {basemap === "satellite" ? (
          <TileLayer
            key="esri-satellite"
            attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and GIS Community'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            maxZoom={19}
          />
        ) : (
          <TileLayer
            key="osm-dark-vector"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            className="dark-tile-layer"
            maxZoom={19}
          />
        )}

        {/* Real GeoTIFF Satellite Raster Image Overlay */}
        {rasterOverlay && rasterOverlay.url && showRaster && (
          <ImageOverlay
            key={`raster-overlay-${rasterOverlay.filename || "tif"}-${rasterOverlay.url.slice(-30)}`}
            url={rasterOverlay.url}
            bounds={rasterOverlay.bounds}
            opacity={rasterOpacity}
            zIndex={300}
          />
        )}

        {/* Spatial GeoJSON Vector Polygons */}
        {spatialData && spatialData.features && spatialData.features.length > 0 && (
          <GeoJSON
            key={`geojson-${JSON.stringify(spatialData).length}-${showFootprint ? "fp" : "nofp"}-${showDetections ? "det" : "nodet"}-${Date.now()}`}
            data={spatialData as any}
            style={polygonStyle}
            onEachFeature={onEachFeature}
          />
        )}

        {/* Dynamic Bounds Fitter */}
        <MapBoundsFitter geojsonData={spatialData} rasterOverlay={rasterOverlay} />

        {/* Recenter & Zoom Controls */}
        <MapNavigationControls rasterOverlay={rasterOverlay} spatialData={spatialData} />

        {/* Top-Right Basemap Switcher */}
        <div className="leaflet-top leaflet-right" style={{ pointerEvents: "auto", margin: "14px 280px 14px 14px" }}>
          <button
            type="button"
            onClick={() => setBasemap(basemap === "satellite" ? "dark" : "satellite")}
            className="bg-slate-900/90 hover:bg-slate-800 text-blue-400 border border-slate-700/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono shadow-2xl transition-all hover:border-blue-500 flex items-center gap-1.5 active:scale-95"
            title="Toggle between Satellite Imagery and Dark Vector Map"
          >
            <span>{basemap === "satellite" ? "🛰️ Satellite Feed" : "🗺️ Dark Vector"}</span>
          </button>
        </div>
      </MapContainer>

      {/* Floating HUD status overlay */}
      <div className="absolute bottom-5 left-5 pointer-events-none z-[1000] flex flex-col gap-1.5">
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-800 text-slate-300 px-3 py-2 rounded-xl text-xs font-mono shadow-2xl flex items-center gap-2.5">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>{basemap === "satellite" ? "ESRI World Imagery" : "OpenStreetMap"}</span>
          <span className="text-slate-700">|</span>
          {rasterOverlay ? (
            <span className="text-amber-400 font-semibold flex items-center gap-1">
              <span>🛰️ .tif Active:</span>
              <span className="text-slate-200">{rasterOverlay.filename || "satellite.tif"}</span>
              {rasterOverlay.dimensions && <span className="text-slate-500 text-[10px]">({rasterOverlay.dimensions})</span>}
            </span>
          ) : (
            <span className="text-slate-500">No Raster Loaded</span>
          )}
          <span className="text-slate-700">|</span>
          <span className="text-blue-400">
            {spatialData?.features?.length ? `${spatialData.features.length} GeoJSON Layer(s)` : "0 Vector Features"}
          </span>
        </div>
      </div>

      {/* Floating Interactive Map Legend & GIS Layer Toggles */}
      <div className="absolute bottom-5 right-5 pointer-events-none z-[1000] flex flex-col gap-1.5">
        <div className="bg-slate-900/95 backdrop-blur-md border border-slate-800 text-slate-300 p-3 rounded-2xl text-xs font-mono shadow-2xl flex flex-col gap-2.5 pointer-events-auto min-w-[240px]">
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
            <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
              GIS Layer Controls
            </span>
            <span className="text-[10px] text-blue-400 font-semibold">Interactive</span>
          </div>

          {/* Raster Overlay Toggle + Opacity */}
          <div className="flex flex-col gap-1 bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
            <label className="flex items-center justify-between cursor-pointer select-none">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showRaster}
                  onChange={(e) => setShowRaster(e.target.checked)}
                  className="rounded border-slate-700 text-amber-500 focus:ring-amber-400 h-3.5 w-3.5 bg-slate-900 cursor-pointer"
                />
                <span className="text-amber-300 font-medium text-[11px]">.tif Raster Overlay</span>
              </div>
              <span className="text-[10px] text-slate-500">{Math.round(rasterOpacity * 100)}%</span>
            </label>
            {showRaster && (
              <div className="flex items-center gap-2 mt-1">
                <span className="text-[9px] text-slate-500">Opacity</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={rasterOpacity}
                  onChange={(e) => setRasterOpacity(parseFloat(e.target.value))}
                  className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>
            )}
          </div>

          {/* Scene Footprint Polygon Toggle */}
          <label className="flex items-center justify-between cursor-pointer select-none px-1">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showFootprint}
                onChange={(e) => setShowFootprint(e.target.checked)}
                className="rounded border-slate-700 text-amber-500 focus:ring-amber-400 h-3.5 w-3.5 bg-slate-900 cursor-pointer"
              />
              <span className="w-3 h-3 border-2 border-dashed border-amber-500 bg-amber-500/10 rounded-sm shrink-0" />
              <span className="text-amber-300/90 text-[11px]">Scene Footprint (Coverage)</span>
            </div>
          </label>

          {/* AI Delineations Polygon Toggle */}
          <label className="flex items-center justify-between cursor-pointer select-none px-1">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showDetections}
                onChange={(e) => setShowDetections(e.target.checked)}
                className="rounded border-slate-700 text-cyan-500 focus:ring-cyan-400 h-3.5 w-3.5 bg-slate-900 cursor-pointer"
              />
              <span className="w-3 h-3 border-2 border-cyan-400 bg-sky-500/20 rounded-sm shrink-0" />
              <span className="text-cyan-300 text-[11px]">AI Feature Delineation</span>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}
