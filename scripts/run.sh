#!/usr/bin/env bash
# Production-style single-process: build the dashboard, serve API + UI on :8000.
set -e
cd "$(dirname "$0")/.."
HOST="${BACKEND_HOST:-127.0.0.1}"
PORT="${BACKEND_PORT:-8000}"
echo "[run] Building frontend..."
( cd frontend && npm run build )
echo "[run] Serving API + dashboard at http://$HOST:$PORT"
exec python3 -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
