#!/bin/bash
# start.sh — Start the Vera bot server
# Usage: ./start.sh [port]

PORT=${1:-8080}

cd "$(dirname "$0")"
source venv/bin/activate

echo "Starting Vera bot on port $PORT..."
echo "Bot URL: http://localhost:$PORT"
echo "Judge run: python run_judge.py [warmup|phase2_short|all|full_evaluation]"
echo ""

uvicorn app:app --host 0.0.0.0 --port "$PORT" --log-level info
