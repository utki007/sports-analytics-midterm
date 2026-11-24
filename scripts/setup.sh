#!/usr/bin/env bash
# One-time setup: Python deps + frontend deps.
set -e
cd "$(dirname "$0")/.."
echo "[setup] Python dependencies..."
python3 -m pip install -r requirements.txt
echo "[setup] Frontend dependencies..."
cd frontend && npm install
echo "[setup] Done. Run ./scripts/run.sh (prod) or ./scripts/dev.sh (dev)."
