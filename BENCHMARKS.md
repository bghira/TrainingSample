# Performance Benchmarks

TrainingSample includes benchmarks for common preprocessing operations: crop, resize, luminance, resize-plus-luminance pipelines, and video frame resizing. The benchmarks are meant to catch regressions and provide workload-specific guidance, not to guarantee universal speedups over OpenCV or NumPy.

## Running Benchmarks

Use the repository virtual environment when available:

```bash
.venv/bin/python -m pytest tests/test_performance_benchmarks.py -q -s
```

To run every Python test and benchmark marker in the repo:

```bash
.venv/bin/python -m pytest -q
```

For a fresh source build before measuring:

```bash
env -u OPENCV_LINK_LIBS -u OPENCV_LINK_PATHS -u OPENCV_INCLUDE_PATHS \
    -u LIBCLANG_PATH -u LLVM_CONFIG_PATH \
    .venv/bin/maturin develop --release
```

The OpenCV Rust binding needs a discoverable OpenCV and Clang installation. On this development host, stale macOS-style OpenCV and LLVM environment variables had to be unset before the build could probe the system OpenCV installation.

## Current Local Snapshot

Last measured command:

```bash
.venv/bin/python -m pytest tests/test_performance_benchmarks.py -q -s
```

Environment:

- Linux x86_64
- CPython 3.13
- NumPy 2.3.4
- system OpenCV 4.11 via the Rust `opencv` crate
- release build installed with `maturin develop --release`

Point-in-time scenario timings from the benchmark output:

| Scenario | Before optimization | After optimization | Comparison after optimization |
|----------|---------------------|--------------------|-------------------------------|
| Crop batch, 16 images | 22.9 ms | 0.4 ms | NumPy slicing was still faster because it returns views |
| Mixed-shape crop, 8 images | 50.2 ms | 3.3 ms | NumPy slicing loop was near-zero because it returns views |
| Resize, 4 mixed-size images | 4.1 ms | 0.4 ms | OpenCV loop: 2.6 ms |
| Luminance, 4 mixed-size images | 10.4 ms | 0.6 ms | OpenCV loop: 0.9 ms |
| Resize + luminance pipeline, 4 images | 5.9 ms | 0.6 ms | OpenCV loop: 2.1 ms |
| Mixed-shape luminance, 6 images | 78.3 ms | 3.3 ms | NumPy loop: 19.4 ms |

Pytest-benchmark means from the same focused run:

| Benchmark | Mean |
|-----------|------|
| Center crop | 55.2 us |
| Resize operations | 353.1 us |
| Luminance calculation | 417.2 us |
| Crop operations | 583.8 us |
| Pipeline | 3.44 ms |
| Video processing | 2.85 ms |

A full `pytest -q` run also passed and produced similar benchmark ordering, with normal run-to-run variance.

## What Changed in the Latest Optimization

- Owned Rust `ndarray` outputs are transferred into NumPy with `from_owned_array_bound`, avoiding an additional copy in Python-facing result conversion.
- Contiguous luminance inputs use a channel-sum fast path. Instead of computing weighted luminance per pixel, it sums R, G, and B separately and applies the weights once at the end.
- Non-contiguous arrays still use the general ndarray path for correctness.

## Benchmark Categories

### Image Operations

- `batch_crop_images`
- `batch_center_crop_images`
- `batch_random_crop_images`
- `batch_resize_images`
- `batch_calculate_luminance`

### Pipeline Operations

- resize followed by luminance
- crop followed by resize
- mixed input sizes and output sizes

### Video Operations

- `batch_resize_videos` with frame batches shaped `(T, H, W, 3)`

## Interpreting Results

Use these benchmarks to answer practical questions:

- Is a change adding extra Rust-to-NumPy copies?
- Are contiguous arrays staying on the fast path?
- Is resize dominated by OpenCV work or Python binding overhead?
- Does a mixed-shape batch still behave reasonably?
- Is a video processing change accidentally introducing per-frame Python overhead?

Some comparisons need context:

- NumPy crop by slicing often returns a view, so it can be much faster than any function that returns owned cropped arrays.
- Very small images can be dominated by Python call overhead.
- Large images can be dominated by memory bandwidth rather than arithmetic.
- OpenCV performance varies by build options, CPU features, and linked libraries.

## Quality Checks

The tests validate basic output behavior alongside timing:

- Crop outputs have expected shape and match NumPy slicing where ownership differences do not matter.
- Resize outputs have expected shape and are close to OpenCV output for the configured interpolation.
- Luminance stays within a small tolerance of NumPy/OpenCV-style references.
- Non-contiguous arrays are accepted by safe luminance paths and rejected by strict zero-copy crop/resize paths.

## Regression Signals

Investigate if a change causes:

- Public batch crop to return to multi-millisecond timings for small batches.
- Luminance on contiguous RGB arrays to lose the channel-sum fast path.
- Resize benchmarks to add large overhead beyond OpenCV work.
- Video resizing to scale with per-frame Python object churn.
- Memory usage to grow unexpectedly for repeated batch calls.

## Future Benchmark Work

- Store historical benchmark results by commit and host.
- Add explicit memory allocation tracking for Python-facing APIs.
- Separate view-returning crop comparisons from owned-output crop comparisons.
- Add more video pipeline benchmarks.
- Document hardware and OpenCV build details in benchmark artifacts.
