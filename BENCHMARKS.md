# Competitive Performance Benchmarks

This library includes comprehensive benchmarks against industry-standard libraries (OpenCV, NumPy) to ensure competitive performance for real-world SFT (Supervised Fine-Tuning) workloads.

## Benchmark Categories

### 🖼️ High-Resolution Image Processing

**Target Workload**: 5120×5120 → 1024×1024 image processing pipeline
- **Input**: 5120×5120×3 images (26.2M pixels, ~78MB each)
- **Pipeline**: Center crop → Resize → Luminance calculation
- **Batch sizes**: 2-4 images (memory constrained)

### 📊 Performance Targets

| Operation | Input Size | Target Performance | Baseline |
|-----------|------------|-------------------|----------|
| **Resize** | 5120×5120 → 1024×1024 | Match OpenCV bilinear | `cv2.resize()` |
| **Center Crop** | 5120×5120 → 2048×2048 | Match/exceed NumPy | Array slicing |
| **Luminance** | 1024×1024 | 1.5x+ faster than NumPy | Vectorized math |
| **Full Pipeline** | 5120×5120 → 1024×1024 | >0.5 images/sec | Combined ops |

### 🎯 Quality Targets

- **Resize Quality**: PSNR >30dB vs OpenCV (excellent similarity)
- **Crop Accuracy**: Bit-exact match with NumPy center crop
- **Luminance Precision**: <0.1 difference vs NumPy reference

## Running Benchmarks

### Local Development
```bash
# Install dependencies
pip install opencv-python pytest-benchmark psutil

# Build with optimizations
maturin develop --release --features "python-bindings,simd"

# Run competitive benchmarks
./scripts/run_competitive_benchmarks.sh
```

### CI/CD Integration

Benchmarks run automatically in CI for:
- **Pull requests**: Performance regression detection
- **Main branch**: Performance tracking over time
- **Weekly schedule**: Long-term performance monitoring

## Benchmark Architecture

### Memory Efficiency
- Monitors RSS memory usage vs OpenCV
- Tests batch processing memory scaling
- Validates no memory leaks in pipelines

### SIMD Optimization Validation
- Compares SIMD-enabled vs scalar fallback performance
- Tests x86-64 AVX2/AVX-512 and ARM64 NEON paths
- Validates CPU feature detection accuracy

### Real-World Scenarios
- **SFT Data Processing**: High-res → training resolution pipeline
- **Batch Processing**: Multiple images with different operations
- **Memory Constraints**: Large images with limited RAM

## Performance Philosophy

### Why These Benchmarks Matter

1. **Real-World Relevance**: SFT workloads use 5120×5120+ images, not toy 224×224
2. **Competitive Pressure**: OpenCV and NumPy are highly optimized incumbents
3. **User Experience**: Poor performance = adoption barriers
4. **Resource Efficiency**: Training infrastructure costs scale with throughput

### Performance vs Quality Tradeoffs

- **Resize**: Bilinear interpolation for speed, good quality balance
- **SIMD**: Aggressive optimization while maintaining numerical accuracy
- **Memory**: Batch processing for throughput vs memory pressure balance

## Interpreting Results

### Good Performance Indicators
- ✅ Resize: 1-2 images/sec for 5120×5120 → 1024×1024
- ✅ Crop: 10+ images/sec for 5120×5120 → 2048×2048
- ✅ Luminance: 1.5x+ faster than NumPy with SIMD
- ✅ Pipeline: >0.5 complete transformations/sec

### Red Flags
- ❌ Slower than OpenCV resize (indicates poor SIMD utilization)
- ❌ Slower than NumPy crop (indicates unnecessary overhead)
- ❌ Memory usage >2x OpenCV (indicates memory leaks/inefficiency)
- ❌ Quality degradation (PSNR <30dB vs reference)

## Future Enhancements

### Planned Improvements
- GPU acceleration benchmarks (Metal/CUDA)
- More interpolation methods (bicubic, lanczos)
- Video processing pipeline benchmarks
- Multi-threaded batch processing optimization

### Performance Tracking
- Historical performance database
- Regression detection and alerting
- Performance comparison across different hardware configurations
- Automated performance optimization recommendations

---

**Goal**: Be the fastest, highest-quality image processing library for ML/SFT workloads while maintaining competitive memory usage and numerical accuracy.
