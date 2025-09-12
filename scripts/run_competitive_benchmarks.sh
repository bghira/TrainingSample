#!/bin/bash
# Run competitive benchmarks locally for development

set -e

echo "🏁 Running competitive benchmarks against OpenCV/NumPy"
echo "======================================================"

# Check if dependencies are available
.venv/bin/python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__} available')" || {
    echo "❌ OpenCV not available. Install with: .venv/bin/pip install opencv-python"
    exit 1
}

.venv/bin/python3 -c "import trainingsample; print('✅ TrainingSample available')" || {
    echo "❌ TrainingSample not available. Build with: maturin develop --features python-bindings,simd"
    exit 1
}

# Run with memory profiling and detailed output
echo ""
echo "🧠 Memory usage will be monitored during benchmarks"
echo "⚡ SIMD optimizations: $(.venv/bin/python3 -c "import trainingsample as tsr; print('Enabled' if hasattr(tsr, 'get_x86_cpu_features') else 'Basic')")"
echo ""

# Run the benchmarks with verbose output
.venv/bin/python3 -m pytest tests/test_competitive_benchmarks.py -v -s --tb=short \
    --benchmark-disable \
    --durations=10

echo ""
echo "🎯 Benchmark complete! Check output above for performance comparisons."
echo "📊 For detailed benchmarking with pytest-benchmark, run:"
echo "    .venv/bin/python3 -m pytest tests/test_competitive_benchmarks.py --benchmark-only"
