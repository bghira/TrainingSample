#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -c 'import cv2, trainingsample, pytest_benchmark'
exec .venv/bin/python -m pytest tests/test_competitive_benchmarks.py \
    --benchmark-only --benchmark-columns=median,iqr --benchmark-group-by=group "$@"
