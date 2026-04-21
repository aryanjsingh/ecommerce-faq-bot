#!/bin/bash
# Start E-Commerce FAQ Bot (API + UI)
echo "🛒 Starting E-Commerce FAQ Bot..."

# Start Python API on port 8000
cd "$(dirname "$0")"
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "✅ API running at http://localhost:8000 (PID: $API_PID)"

# Start Next.js UI on port 3000
cd ui
export PATH="$PATH:/opt/homebrew/bin"
npm run dev &
UI_PID=$!
echo "✅ UI running at http://localhost:3000 (PID: $UI_PID)"

echo ""
echo "🚀 E-Commerce Bot ready!"
echo "   UI  →  http://localhost:3000"
echo "   API →  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

trap "kill $API_PID $UI_PID 2>/dev/null; echo 'Stopped.'" EXIT INT
wait
