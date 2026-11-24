#!/usr/bin/env bash
# Dev mode: FastAPI (reload) + Vite dev server with hot reload.
set -e
cd "$(dirname "$0")/.."
HOST="${BACKEND_HOST:-127.0.0.1}"
PORT="${BACKEND_PORT:-8000}"
python3 -m uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload &
API_PID=$!
trap "kill $API_PID 2>/dev/null" EXIT
( cd frontend && npm run dev )
