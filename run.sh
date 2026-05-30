#!/bin/bash
# run.sh — Start the Quiz Scanner web application
# Usage: bash run.sh

set -e

echo "=================================================="
echo "  QuizScanner AI — Automated Quiz Grading System"
echo "=================================================="

# ── Check Python ───────────────────────────────────────────────────────
python3 --version || { echo "Python3 not found"; exit 1; }

# ── Install dependencies (first run only) ──────────────────────────────
if [ ! -f ".deps_installed" ]; then
    echo ""
    echo "[1/3] Installing Python dependencies..."
    pip install -r requirements.txt -q
    touch .deps_installed
    echo "     Done."
fi

# ── System library check ───────────────────────────────────────────────
if command -v dpkg &> /dev/null; then
    if ! dpkg -l libzbar0 &> /dev/null; then
        echo ""
        echo "[!] libzbar0 not found. Installing..."
        sudo apt-get install -y libzbar0 libgl1 &> /dev/null || true
    fi
fi

# ── Generate sample images (first run only) ────────────────────────────
if [ ! "$(ls samples/*.jpg 2>/dev/null)" ]; then
    echo ""
    echo "[2/3] Generating sample quiz images..."
    python3 src/generate_sample.py 5
    echo "     Done."
fi

# ── Launch Flask app ───────────────────────────────────────────────────
echo ""
echo "[3/3] Starting web server..."
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │  Open in browser:  http://localhost:5000     │"
echo "  │  Press Ctrl+C to stop                        │"
echo "  └─────────────────────────────────────────────┘"
echo ""

cd src
FLASK_ENV=development PYTHONPATH=. python3 app.py
