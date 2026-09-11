"""
SatQuery-AI: SQLite Database & Query History Persistence Engine
Manages local SQLite database for query logging, audit trail, execution traces,
and historical GeoJSON spatial outputs.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path("data")
DB_PATH = DB_DIR / "satquery.db"


def get_db_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables if they do not exist."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    task TEXT NOT NULL,
                    answer TEXT,
                    confidence REAL,
                    spatial_data TEXT,
                    visual_evidence TEXT,
                    execution_trace TEXT,
                    image_metadata TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_timestamp
                ON query_logs(timestamp DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_task
                ON query_logs(task);
                """
            )
    finally:
        conn.close()


def log_query(
    query_text: str,
    task: str,
    answer: str,
    confidence: float,
    spatial_data: Optional[Dict[str, Any]] = None,
    visual_evidence: Optional[Dict[str, Any]] = None,
    execution_trace: Optional[List[Dict[str, Any]]] = None,
    image_metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Log a satellite query transaction into the database.
    Returns the newly inserted record ID.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO query_logs (
                    timestamp,
                    query_text,
                    task,
                    answer,
                    confidence,
                    spatial_data,
                    visual_evidence,
                    execution_trace,
                    image_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    query_text,
                    task,
                    answer,
                    float(confidence),
                    json.dumps(spatial_data) if spatial_data is not None else None,
                    json.dumps(visual_evidence) if visual_evidence is not None else None,
                    json.dumps(execution_trace) if execution_trace is not None else None,
                    json.dumps(image_metadata) if image_metadata is not None else None,
                ),
            )
            return int(cursor.lastrowid)
    finally:
        conn.close()


def get_recent_queries(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve recent queries ordered chronologically (newest first)."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, timestamp, query_text, task, answer, confidence,
                   spatial_data, visual_evidence, execution_trace, image_metadata
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            meta_dict = json.loads(r["image_metadata"]) if r["image_metadata"] else {}
            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "query_text": r["query_text"],
                "task": r["task"],
                "answer": r["answer"],
                "confidence": r["confidence"],
                "spatial_data": json.loads(r["spatial_data"]) if r["spatial_data"] else None,
                "visual_evidence": json.loads(r["visual_evidence"]) if r["visual_evidence"] else {},
                "execution_trace": json.loads(r["execution_trace"]) if r["execution_trace"] else [],
                "image_metadata": meta_dict,
                "raster_overlay": meta_dict.get("raster_overlay"),
            })
        return results
    finally:
        conn.close()


def get_query_by_id(query_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single query record by its ID."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, timestamp, query_text, task, answer, confidence,
                   spatial_data, visual_evidence, execution_trace, image_metadata
            FROM query_logs
            WHERE id = ?
            """,
            (query_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        meta_dict = json.loads(row["image_metadata"]) if row["image_metadata"] else {}
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "query_text": row["query_text"],
            "task": row["task"],
            "answer": row["answer"],
            "confidence": row["confidence"],
            "spatial_data": json.loads(row["spatial_data"]) if row["spatial_data"] else None,
            "visual_evidence": json.loads(row["visual_evidence"]) if row["visual_evidence"] else {},
            "execution_trace": json.loads(row["execution_trace"]) if row["execution_trace"] else [],
            "image_metadata": meta_dict,
            "raster_overlay": meta_dict.get("raster_overlay"),
        }
    finally:
        conn.close()
