#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting SatQuery-AI Integrated Services (temp_full_project)"
echo "=========================================================="

# 1. Check Python virtual environment
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python"
fi

# 2. Start FastAPI Backend in background
echo "[1/2] Starting FastAPI Backend on http://localhost:8000 ..."
$PYTHON_BIN -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cleanup() {
    echo ""
    echo "Shutting down SatQuery-AI services..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Give backend a moment to initialize SQLite and bind port
sleep 2

# 3. Start Next.js Frontend
echo "[2/2] Starting Next.js Frontend on http://localhost:3000 ..."
npm run dev
