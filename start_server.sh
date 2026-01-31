#!/bin/bash
# Script to start Halalificator with Gunicorn
source .venv/bin/activate

# We use 1 worker because the processing (CLIP/Demucs) is very resource intensive.
# Multiple workers might exhaust RAM/VRAM.
# --timeout 600 allows for long-running startup/upload if needed.
gunicorn --workers 1 \
         --worker-class sync \
         --bind 0.0.0.0:5000 \
         --timeout 600 \
         app:app
