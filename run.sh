#!/usr/bin/env bash
# Run any command inside the project venv with a clean Python environment.
# This machine's shell profile sources ROS2 (humble), which prepends ~13 ROS
# packages to PYTHONPATH ahead of the venv — we strip that here.
#
#   ./run.sh python3 scripts/test_embedding.py
#   ./run.sh uvicorn app.main:app --host 0.0.0.0 --port 8000
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 HF_HUB_DISABLE_TELEMETRY=1 "$@"
