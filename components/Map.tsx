"use client";

import React, { useEffect, useCallback, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix default Leaflet icon asset resolution in Next.js client environments
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

/**
 * Internal useMap() helper that automatically calculates GeoJSON bounds
 * and calls map.fitBounds() with [50, 50] padding on spatial data changes.
 */
function MapBoundsFitter({ geojsonData }: { geojsonData: GeoJSON.FeatureCollection | null }) {
  const map = useMap();

  useEffect(() => {
    if (!geojsonData || !geojsonData.features || geojsonData.features.length === 0) {
      return;
    }

    try {
      const geoJsonLayer = L.geoJSON(geojsonData as any);
      const bounds = geoJsonLayer.getBounds();

      if (bounds && bounds.isValid()) {
        map.fitBounds(bounds, {
          padding: [50, 50],
          maxZoom: 16,
          animate: true,
        });
      }
    } catch (err) {
      console.warn("Could not calculate GeoJSON bounds:", err);
    }
  }, [geojsonData, map]);

  return null;
}

/**
 * Overlay control for quick recentering to default Hubli / Unkal Lake coordinates
 */
function MapRecenterControl() {
  const map = useMap();

  const handleRecenter = useCallback(() => {
    map.flyTo([15.3647, 75.124], 12, { duration: 1.0 });
  }, [map]);

  return (
    <div className="leaflet-top leaflet-right" style={{ pointerEvents: "auto", margin: "14px" }}>
      <button
        type="button"
        onClick={handleRecenter}
        className="bg-slate-900/90 hover:bg-slate-800 text-blue-400 border border-slate-700/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono shadow-2xl transition-all hover:border-blue-500 flex items-center gap-1.5 active:scale-95"
        title="Recenter view to Hubli / Unkal Lake"
      >
        <svg className="w-3.5 h-3.5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
        <span>Recenter Hubli [15.3647, 75.124]</span>
      </button>
    </div>
  );
}

export interface MapProps {
  spatialData: GeoJSON.FeatureCollection | null;
}

export default function Map({ spatialData }: MapProps) {
  // Basemap switcher state: default to high-res Satellite imagery (ESRI)
  const [basemap, setBasemap] = useState<"satellite" | "dark">("satellite");

  // Hubli / Unkal Lake, Karnataka coordinates
  const defaultCenter: [number, number] = [15.3647, 75.124];
  const defaultZoom = 12;

  // Blue highlight styling (#3b82f6 stroke, #60a5fa fill at 0.35 opacity)
  const polygonStyle = () => ({
    stroke: true,
    color: "#3b82f6", // Stroke: blue-500
    weight: 2.5,
    opacity: 0.95,
    fillColor: "#60a5fa", // Fill: blue-400
    fillOpacity: 0.35, // FillOpacity: 0.35
  });

  // Attach interactive dark popups to GeoJSON features
  const onEachFeature = (feature: any, layer: L.Layer) => {
    if (!feature || !feature.properties) return;

    const props = feature.properties;
    const scoreVal = props.confidence ?? props.score;
    const popupContent = `
      <div style="font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; color: #f1f5f9; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; min-width: 220px;">
        <div style="font-weight: 700; color: #60a5fa; font-size: 13px; margin-bottom: 6px; border-bottom: 1px solid #1e293b; padding-bottom: 4px;">
          ${props.name || props.label || "Remote Sensing Detected Polygon"}
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Region:</span>
          <span style="font-weight: 500; color: #e2e8f0;">${props.region || "Hubli AOI"}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="color: #94a3b8;">Classification:</span>
          <span style="font-weight: 600; color: #38bdf8;">${props.classification || props.label || "Segmented Feature"}</span>
        </div>
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
        ${props.sensor ? `
        <div style="font-size: 10px; color: #64748b; margin-top: 6px; border-top: 1px solid #1e293b; padding-top: 4px;">
          Sensor: ${props.sensor}
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

        {/* Spatial GeoJSON Vector Polygons */}
        {spatialData && spatialData.features && spatialData.features.length > 0 && (
          <>
            <GeoJSON
              key={`geojson-${JSON.stringify(spatialData).length}-${Date.now()}`}
              data={spatialData as any}
              style={polygonStyle}
              onEachFeature={onEachFeature}
            />
            {/* Bounds fitter using useMap() */}
            <MapBoundsFitter geojsonData={spatialData} />
          </>
        )}

        <MapRecenterControl />

        {/* Top-Right Basemap Switcher Control */}
        <div className="leaflet-top leaflet-right" style={{ pointerEvents: "auto", margin: "14px 170px 14px 14px" }}>
          <button
            type="button"
            onClick={() => setBasemap(basemap === "satellite" ? "dark" : "satellite")}
            className="bg-slate-900/90 hover:bg-slate-800 text-blue-400 border border-slate-700/80 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono shadow-2xl transition-all hover:border-blue-500 flex items-center gap-1.5 active:scale-95"
            title="Toggle between Satellite Imagery and Dark Vector Map"
          >
            <span>{basemap === "satellite" ? "🛰️ Satellite Imagery" : "🗺️ Dark Vector Map"}</span>
          </button>
        </div>
      </MapContainer>

      {/* Floating HUD status overlay */}
      <div className="absolute bottom-5 left-5 pointer-events-none z-[1000] flex flex-col gap-1.5">
        <div className="bg-slate-900/85 backdrop-blur-md border border-slate-800 text-slate-300 px-3 py-1.5 rounded-lg text-xs font-mono shadow-2xl flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>{basemap === "satellite" ? "ESRI World Satellite Feed" : "OpenStreetMap Dark Vector"}</span>
          <span className="text-slate-600">|</span>
          <span className="text-blue-400">
            {spatialData?.features?.length ? `${spatialData.features.length} GeoJSON Feature Active` : "Awaiting GeoJSON"}
          </span>
        </div>
      </div>
    </div>
  );
}
