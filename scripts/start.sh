#!/bin/bash
# scripts/start.sh — One-click start for development

set -e

# Activate venv
source .venv/bin/activate

# Start FastAPI
echo "Starting FastAPI on :8080..."
uvicorn web.server.main:app --host 0.0.0.0 --port 8080 --reload &
API_PID=$!

# Start React dev server
echo "Starting React on :3000..."
cd web/frontend && npm run dev &
UI_PID=$!

echo "API: http://localhost:8080"
echo "UI:  http://localhost:3000"
echo "Press Ctrl+C to stop"

trap "kill $API_PID $UI_PID 2>/dev/null" EXIT
wait
