"use client";

import React, { useState, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  sendQuery,
  fetchQueryHistory,
  QueryResponse,
  HistoryRecord,
  TraceStep,
  RasterOverlay,
  USE_MOCK_FALLBACK,
} from "@/lib/api";

// Dynamically import Leaflet with ssr: false to completely eliminate Next.js SSR window errors
const LeafletMap = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full bg-slate-950 flex flex-col items-center justify-center text-slate-500 font-mono gap-3 p-8">
      <div className="relative flex items-center justify-center">
        <div className="w-12 h-12 rounded-full border-2 border-blue-500/20 border-t-blue-500 animate-spin" />
        <div className="absolute w-6 h-6 rounded-full bg-blue-500/15 animate-ping" />
      </div>
      <span className="text-sm tracking-wide text-slate-400">Initializing GIS Leaflet Engine...</span>
      <span className="text-xs text-slate-600">Loading CARTO Dark Matter vector tiles</span>
    </div>
  ),
});

export default function GeoAgentDashboard() {
  const [query, setQuery] = useState("");
  const [primaryImage, setPrimaryImage] = useState<File | null>(null);
  const [t2Image, setT2Image] = useState<File | null>(null);
  const [sarImage, setSarImage] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [traces, setTraces] = useState<TraceStep[]>([]);
  const [spatialData, setSpatialData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [rasterOverlay, setRasterOverlay] = useState<RasterOverlay | null>(null);

  const [historyList, setHistoryList] = useState<HistoryRecord[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const primaryInputRef = useRef<HTMLInputElement>(null);
  const t2InputRef = useRef<HTMLInputElement>(null);
  const sarInputRef = useRef<HTMLInputElement>(null);

  const loadHistory = async () => {
    setLoadingHistory(true);
    try {
      const records = await fetchQueryHistory(20);
      setHistoryList(records);
    } catch (err) {
      console.warn("Failed to load history:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleSelectHistory = (record: HistoryRecord) => {
    setQuery(record.query_text);
    setQueryResult({
      task: record.task,
      query: record.query_text,
      answer: record.answer,
      confidence: record.confidence,
      visual_evidence: record.visual_evidence,
      spatial_data: record.spatial_data,
      execution_trace: record.execution_trace,
      query_id: record.id,
    });
    setTraces(record.execution_trace || []);
    setSpatialData(record.spatial_data);
    setRasterOverlay(record.raster_overlay || null);
    setShowHistory(false);
  };

  // Preset query chips specified in backend contract requirements
  const presetQueries = [
    {
      id: "water_body",
      label: "Highlight the water body",
      queryText: "Highlight the water body",
      description: "Segment lake/reservoir boundaries using spectral NDWI indexing",
    },
    {
      id: "change_detection",
      label: "Find new construction between T1 and T2",
      queryText: "Find new construction between T1 and T2",
      description: "Bi-temporal differential change detection across timestamps",
    },
    {
      id: "optical_sar",
      label: "Assess flood impact and urban presence using optical and SAR",
      queryText: "Assess flood impact and urban presence using optical and SAR",
      description: "Multi-modal optical RGB + Sentinel-1 SAR backscatter fusion",
    },
  ];

  const handleSelectPreset = (presetText: string) => {
    setQuery(presetText);
  };

  const handleFileChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    setter: (file: File | null) => void
  ) => {
    if (e.target.files && e.target.files[0]) {
      setter(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setQueryResult(null);
    setTraces([]);
    setRasterOverlay(null);

    try {
      const data: QueryResponse = await sendQuery(
        query,
        primaryImage || undefined,
        t2Image || undefined,
        sarImage || undefined
      );

      setQueryResult(data);
      setTraces(data.execution_trace || []);
      setSpatialData(data.spatial_data);
      setRasterOverlay(data.raster_overlay || null);
      loadHistory();
    } catch (err: any) {
      console.error("Query execution failed:", err);
      // Fallback display if mock flag was toggled off and backend unreachable
      setQueryResult({
        task: "grounding",
        query: query,
        answer: err.message || "Failed to connect to backend server at http://localhost:8000/query",
        confidence: 0,
        visual_evidence: {
          evidence_type: "error",
          description: "Connection error",
        },
        spatial_data: null,
        execution_trace: [
          { step: "Network Request Dispatch", status: "FAILED", time_ms: 50 },
        ],
      });
    } finally {
      setLoading(false);
    }
  };

  // Helper for status badge styling
  const getStatusBadge = (status: string) => {
    const s = status.toUpperCase();
    if (s === "COMPLETED" || s === "SUCCESS") {
      return (
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          COMPLETED
        </span>
      );
    }
    if (s === "RUNNING" || s === "IN_PROGRESS") {
      return (
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/30 animate-pulse">
          RUNNING
        </span>
      );
    }
    if (s === "FAILED" || s === "ERROR") {
      return (
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30">
          FAILED
        </span>
      );
    }
    return (
      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">
        {status}
      </span>
    );
  };

  return (
    <main className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans select-none">
      {/* ========================================================================= */}
      {/* LEFT PANEL: Control Sidebar, Multi-Image Upload, Traces (40% width)      */}
      {/* ========================================================================= */}
      <div className="w-[40%] min-w-[400px] max-w-[640px] h-full flex flex-col border-r border-slate-800 bg-slate-900/90 backdrop-blur-md overflow-hidden z-10 shadow-2xl">
        
        {/* Header */}
        <header className="px-6 py-4 border-b border-slate-800 bg-slate-900/95 flex justify-between items-center shrink-0">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/25 border border-blue-400/30">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
                  GeoAgent <span className="text-blue-400 font-extrabold">VQA</span>
                </h1>
                <span className="px-1.5 py-0.5 text-[10px] font-mono font-medium rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">Agentic Remote Sensing Intelligence</p>
            </div>
          </div>

          {/* Header Controls: History & Status */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setShowHistory(!showHistory);
                if (!showHistory) loadHistory();
              }}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium border transition-all flex items-center gap-1.5 ${
                showHistory
                  ? "bg-blue-600 text-white border-blue-400 shadow-md shadow-blue-500/20"
                  : "bg-slate-950/80 text-slate-300 border-slate-800 hover:border-slate-700 hover:bg-slate-900"
              }`}
              title="View SQLite Query History"
            >
              <svg className="w-3.5 h-3.5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>History</span>
              {historyList.length > 0 && (
                <span className="text-[10px] bg-blue-500/20 text-blue-300 px-1.5 py-0.2 rounded-full border border-blue-500/30">
                  {historyList.length}
                </span>
              )}
            </button>

            {/* Live Status Badge */}
            <div className="flex items-center gap-2 bg-slate-950/80 border border-slate-800 px-3 py-1 rounded-full shadow-inner">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[11px] font-mono text-emerald-300 font-medium">
                {USE_MOCK_FALLBACK ? "Backend / Fallback Ready" : "Backend Connected"}
              </span>
            </div>
          </div>
        </header>

        {/* Query History Overlay / Drawer */}
        {showHistory && (
          <div className="px-6 py-3 bg-slate-950 border-b border-slate-800 shadow-inner max-h-64 overflow-y-auto custom-scrollbar">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
                <span>SQLite Query History</span>
                <span className="text-[9px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">
                  {historyList.length} records
                </span>
              </span>
              <button
                type="button"
                onClick={loadHistory}
                disabled={loadingHistory}
                className="text-[10px] font-mono text-slate-400 hover:text-blue-300 transition-colors"
              >
                {loadingHistory ? "Refreshing..." : "Refresh"}
              </button>
            </div>

            {historyList.length === 0 ? (
              <div className="text-center py-4 text-xs text-slate-500 font-mono">
                No past queries recorded in SQLite database yet.
              </div>
            ) : (
              <div className="space-y-1.5">
                {historyList.map((rec) => (
                  <div
                    key={rec.id}
                    onClick={() => handleSelectHistory(rec)}
                    className="p-2 bg-slate-900/90 border border-slate-800 hover:border-blue-500/60 rounded-xl cursor-pointer transition-all hover:bg-slate-850 flex flex-col gap-1 text-xs group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 uppercase font-bold">
                          {rec.task.replace("_", " ")}
                        </span>
                        <span className="text-[10px] font-mono text-slate-500">
                          #{rec.id} • {new Date(rec.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                      <span className="text-[10px] text-emerald-400 font-mono">
                        {Math.round(rec.confidence * 100)}%
                      </span>
                    </div>
                    <div className="text-slate-200 font-medium truncate group-hover:text-blue-300">
                      {rec.query_text}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Scrollable Controls Container */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 custom-scrollbar">

          {/* Preset Query Chips */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-400">
                Preset Query Chips
              </span>
              <span className="text-[10px] text-slate-500 font-mono">1-Click Test</span>
            </div>
            <div className="flex flex-col gap-1.5">
              {presetQueries.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handleSelectPreset(preset.queryText)}
                  className={`text-left text-xs px-3 py-2 rounded-xl border transition-all flex items-center justify-between group ${
                    query === preset.queryText
                      ? "bg-blue-600/20 text-blue-300 border-blue-500/80 shadow-md"
                      : "bg-slate-950/60 text-slate-300 border-slate-800 hover:border-slate-700 hover:bg-slate-900"
                  }`}
                >
                  <span className="font-medium truncate">{preset.label}</span>
                  <svg className="w-3.5 h-3.5 text-slate-500 group-hover:text-blue-400 shrink-0 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              ))}
            </div>
          </div>

          {/* Query Formulation & Multi-Image Form */}
          <form onSubmit={handleSubmit} className="bg-slate-950/70 border border-slate-800 rounded-2xl p-4 space-y-3.5 shadow-md">
            
            {/* Multi-Image Upload Area */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-400">
                  Multi-Image Upload Payloads
                </span>
                <span className="text-[10px] text-slate-500 font-mono">FastAPI Contract</span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                
                {/* 1. Primary Image (`image`) */}
                <div>
                  <input
                    type="file"
                    ref={primaryInputRef}
                    onChange={(e) => handleFileChange(e, setPrimaryImage)}
                    accept="image/*,.tif,.tiff,.geojson"
                    className="hidden"
                    id="primary-image"
                  />
                  <label
                    htmlFor="primary-image"
                    className={`h-16 flex flex-col items-center justify-center p-2 rounded-xl border border-dashed cursor-pointer transition-all text-center ${
                      primaryImage
                        ? "bg-blue-950/40 border-blue-500/60 text-blue-300"
                        : "bg-slate-900/60 border-slate-700 hover:border-slate-500 text-slate-400"
                    }`}
                  >
                    <svg className="w-4 h-4 mb-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="text-[10px] font-mono font-bold truncate w-full">
                      {primaryImage ? primaryImage.name : "Primary Image"}
                    </span>
                    <span className="text-[8px] text-slate-500 uppercase">Optical T1</span>
                  </label>
                  {primaryImage && (
                    <button
                      type="button"
                      onClick={() => {
                        setPrimaryImage(null);
                        if (primaryInputRef.current) primaryInputRef.current.value = "";
                      }}
                      className="text-[9px] text-rose-400 hover:underline w-full text-center mt-0.5"
                    >
                      Clear
                    </button>
                  )}
                </div>

                {/* 2. T2 Image (`image_t2` for change detection) */}
                <div>
                  <input
                    type="file"
                    ref={t2InputRef}
                    onChange={(e) => handleFileChange(e, setT2Image)}
                    accept="image/*,.tif,.tiff,.geojson"
                    className="hidden"
                    id="t2-image"
                  />
                  <label
                    htmlFor="t2-image"
                    className={`h-16 flex flex-col items-center justify-center p-2 rounded-xl border border-dashed cursor-pointer transition-all text-center ${
                      t2Image
                        ? "bg-emerald-950/40 border-emerald-500/60 text-emerald-300"
                        : "bg-slate-900/60 border-slate-700 hover:border-slate-500 text-slate-400"
                    }`}
                  >
                    <svg className="w-4 h-4 mb-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-[10px] font-mono font-bold truncate w-full">
                      {t2Image ? t2Image.name : "T2 Image"}
                    </span>
                    <span className="text-[8px] text-slate-500 uppercase">Change T2</span>
                  </label>
                  {t2Image && (
                    <button
                      type="button"
                      onClick={() => {
                        setT2Image(null);
                        if (t2InputRef.current) t2InputRef.current.value = "";
                      }}
                      className="text-[9px] text-rose-400 hover:underline w-full text-center mt-0.5"
                    >
                      Clear
                    </button>
                  )}
                </div>

                {/* 3. SAR Image (`image_sar` for optical/SAR fusion) */}
                <div>
                  <input
                    type="file"
                    ref={sarInputRef}
                    onChange={(e) => handleFileChange(e, setSarImage)}
                    accept="image/*,.tif,.tiff,.geojson"
                    className="hidden"
                    id="sar-image"
                  />
                  <label
                    htmlFor="sar-image"
                    className={`h-16 flex flex-col items-center justify-center p-2 rounded-xl border border-dashed cursor-pointer transition-all text-center ${
                      sarImage
                        ? "bg-purple-950/40 border-purple-500/60 text-purple-300"
                        : "bg-slate-900/60 border-slate-700 hover:border-slate-500 text-slate-400"
                    }`}
                  >
                    <svg className="w-4 h-4 mb-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
                    </svg>
                    <span className="text-[10px] font-mono font-bold truncate w-full">
                      {sarImage ? sarImage.name : "SAR Image"}
                    </span>
                    <span className="text-[8px] text-slate-500 uppercase">Sentinel-1 SAR</span>
                  </label>
                  {sarImage && (
                    <button
                      type="button"
                      onClick={() => {
                        setSarImage(null);
                        if (sarInputRef.current) sarInputRef.current.value = "";
                      }}
                      className="text-[9px] text-rose-400 hover:underline w-full text-center mt-0.5"
                    >
                      Clear
                    </button>
                  )}
                </div>

              </div>
            </div>

            {/* Textarea Input */}
            <div className="space-y-1.5">
              <label htmlFor="query-textarea" className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-400">
                Natural Language Query
              </label>
              <textarea
                id="query-textarea"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter satellite query or click a preset chip above..."
                rows={2}
                className="w-full p-3 bg-slate-900 border border-slate-700 rounded-xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all resize-none shadow-inner font-sans"
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="w-full py-2.5 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 text-white font-semibold text-xs rounded-xl transition-all shadow-lg shadow-blue-600/20 disabled:shadow-none flex items-center justify-center gap-2 active:scale-[0.99]"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Executing Query Pipeline...</span>
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                  <span>Execute Query</span>
                </>
              )}
            </button>
          </form>

          {/* Agent Execution Trace Card */}
          <section className="border border-slate-800 bg-slate-950/80 rounded-2xl p-4 shadow-sm flex flex-col gap-2.5">
            <div className="flex justify-between items-center border-b border-slate-800/80 pb-2">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-blue-500 animate-ping" />
                <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300">
                  Agent Execution Trace
                </h2>
              </div>
              <span className="text-[10px] font-mono text-slate-500">
                {traces.length > 0 ? `${traces.length} steps recorded` : "Idle"}
              </span>
            </div>

            <div className="space-y-2 max-h-52 overflow-y-auto pr-1 custom-scrollbar font-mono text-xs">
              {loading && traces.length === 0 && (
                <div className="p-3 border border-blue-500/30 bg-blue-950/20 rounded-xl flex items-center gap-3">
                  <div className="w-3.5 h-3.5 rounded-full border-2 border-blue-400 border-t-transparent animate-spin shrink-0" />
                  <span className="text-blue-300 text-xs">
                    Dispatching multi-band remote sensing agent pipeline...
                  </span>
                </div>
              )}

              {traces.length === 0 && !loading ? (
                <div className="py-4 text-center text-slate-600 italic text-xs">
                  No execution traces yet. Submit a query to inspect agent latency and pipeline steps.
                </div>
              ) : (
                traces.map((trace, idx) => (
                  <div
                    key={`${trace.step}-${idx}`}
                    className="p-2.5 border border-slate-800/80 bg-slate-900/90 rounded-xl flex items-center justify-between shadow-sm hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <span className="text-slate-500 text-[10px]">#{idx + 1}</span>
                      <span className="text-slate-200 font-medium truncate text-xs">
                        {trace.step}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-slate-400 text-[10px] bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                        {trace.time_ms} ms
                      </span>
                      {getStatusBadge(trace.status)}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* Output Summary Card */}
          {queryResult && (
            <section className="bg-gradient-to-br from-slate-900 to-slate-950 border border-blue-500/40 rounded-2xl p-4 shadow-xl space-y-2.5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 font-mono font-bold text-[10px] uppercase border border-blue-500/30">
                    {queryResult.task.replace("_", " ")}
                  </span>
                  <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                    {Math.round(queryResult.confidence * 100)}% Confidence
                  </span>
                </div>

                {queryResult.change_percentage !== undefined && (
                  <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                    +{queryResult.change_percentage}% Change
                  </span>
                )}
              </div>

              {/* Primary Answer field from backend */}
              <div className="text-xs text-slate-200 leading-relaxed font-sans">
                <p className="font-normal">{queryResult.answer}</p>
              </div>

              {/* Visual Evidence Metadata & Raster Thumbnail */}
              {queryResult.visual_evidence && (queryResult.visual_evidence.evidence_type || queryResult.visual_evidence.description || queryResult.visual_evidence.preview_data_url) && (
                <div className="bg-slate-950/70 rounded-xl p-2.5 border border-slate-800 flex flex-col gap-2 text-[11px] font-mono">
                  {/* Raster Preview Image from .tif */}
                  {queryResult.visual_evidence.preview_data_url && (
                    <div className="rounded-lg overflow-hidden border border-amber-500/30 bg-slate-900 flex flex-col">
                      <div className="bg-amber-950/40 px-2.5 py-1 text-[10px] text-amber-300 font-medium flex items-center justify-between border-b border-amber-500/20">
                        <span className="flex items-center gap-1.5">
                          <span>🛰️ Active GeoTIFF:</span>
                          <span className="text-slate-200">{queryResult.raster_overlay?.filename || "satellite_scene.tif"}</span>
                        </span>
                        {queryResult.raster_overlay?.dimensions && (
                          <span className="text-slate-400">{queryResult.raster_overlay.dimensions}</span>
                        )}
                      </div>
                      <div className="relative w-full h-40 bg-slate-950 flex items-center justify-center overflow-hidden">
                        <img
                          src={queryResult.visual_evidence.preview_data_url}
                          alt="GeoTIFF Raster Scene"
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute bottom-1.5 right-1.5 bg-slate-900/90 text-amber-300 text-[9px] px-1.5 py-0.5 rounded border border-amber-500/30 backdrop-blur-sm">
                          Raster Band Composite
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-between items-center text-slate-400">
                    <span className="text-slate-500 uppercase tracking-wider text-[9px]">Evidence Modality:</span>
                    <span className="text-blue-400 font-semibold px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/20">
                      {queryResult.visual_evidence.evidence_type}
                    </span>
                  </div>
                  {queryResult.visual_evidence.description && (
                    <div className="text-slate-300 text-[11px] font-sans leading-relaxed">
                      {queryResult.visual_evidence.description}
                    </div>
                  )}
                  <div className="flex flex-wrap gap-3 text-[10px] text-slate-400 pt-1 border-t border-slate-800/80">
                    {queryResult.visual_evidence.box_count !== undefined && queryResult.visual_evidence.box_count > 0 && (
                      <span>Detections: <strong className="text-slate-200">{queryResult.visual_evidence.box_count}</strong></span>
                    )}
                    {queryResult.visual_evidence.has_mask && (
                      <span>Mask: <strong className="text-emerald-400">Generated</strong></span>
                    )}
                    {queryResult.spatial_data?.features && queryResult.spatial_data.features.length > 0 && (
                      <span>GeoJSON Polygons: <strong className="text-blue-400">{queryResult.spatial_data.features.length}</strong></span>
                    )}
                    {queryResult.raster_overlay && (
                      <span className="text-amber-400">Raster Overlay: <strong>Active</strong></span>
                    )}
                  </div>
                </div>
              )}

              {/* Complementary Insights bullet list */}
              {queryResult.complementary_insights && queryResult.complementary_insights.length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-1.5">
                  <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-blue-400">
                    Complementary Remote Sensing Insights:
                  </span>
                  <ul className="space-y-1 text-[11px] text-slate-300 pl-3 list-disc marker:text-blue-500">
                    {queryResult.complementary_insights.map((insight, idx) => (
                      <li key={idx} className="leading-snug">
                        {insight}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

        </div>

        {/* Footer Coordinate Readout */}
        <footer className="px-6 py-2.5 border-t border-slate-800/80 bg-slate-950/80 flex justify-between items-center text-[10px] font-mono text-slate-500 shrink-0">
          <span>AOI: Unkal Lake [15.3647° N, 75.1240° E]</span>
          <span className="text-blue-500">Hubli Sector Grid</span>
        </footer>

      </div>

      {/* ========================================================================= */}
      {/* RIGHT PANEL: Full-Screen Leaflet Map (60% width)                          */}
      {/* ========================================================================= */}
      <div className="w-[60%] h-full relative overflow-hidden bg-slate-950">
        <LeafletMap spatialData={spatialData} rasterOverlay={rasterOverlay} />
      </div>
    </main>
  );
}
