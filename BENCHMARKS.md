# Benchmarks

## Commands

Build the current checkout before measuring:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/maturin develop --release
```

Run the performance suite:

```bash
.venv/bin/python -m pytest tests/test_performance_benchmarks.py -q -s
```

Run pytest-benchmark cases only:

```bash
.venv/bin/python -m pytest \
  tests/test_performance_benchmarks.py::TestDetailedBenchmarks \
  --benchmark-only
```

Save machine-readable results:

```bash
.venv/bin/python -m pytest \
  tests/test_performance_benchmarks.py::TestDetailedBenchmarks \
  --benchmark-only \
  --benchmark-json benchmark.json
```

## Measured operations

| Case | TrainingSample call | Reference |
|---|---|---|
| Crop | `batch_crop_images` | NumPy slicing |
| Resize | `batch_resize_images` | direct `cv2.resize(..., INTER_LINEAR)` loop |
| Luminance | `batch_calculate_luminance` | `cv2.cvtColor(..., COLOR_RGB2GRAY)` plus `numpy.mean` |
| Pipeline | resize, then luminance | equivalent OpenCV loop |
| Video resize | `batch_resize_videos` | no external baseline |
| Center crop | `batch_center_crop_images` | no external baseline |

## Comparison constraints

| Topic | Constraint |
|---|---|
| Crop ownership | TrainingSample returns owned arrays; plain NumPy slicing returns views |
| Resize color order | Resize is applied directly to the same RGB byte arrays; no RGB/BGR conversion is included |
| Resize interpolation | Both resize paths use linear interpolation |
| Luminance input | Both paths interpret the input as RGB |
| Batch shape | Mixed-shape inputs are processed one image at a time by the OpenCV reference loop |
| Build mode | Measure release builds only |

NumPy view-returning crop timings do not measure the cost of producing owned,
contiguous output. Use `.copy()` when owned-output cost is the subject of the
comparison.

## Result metadata

Record these fields with timing results:

```text
git commit
operating system and architecture
CPU model
Python version
NumPy version
TrainingSample version
Rust OpenCV version and link mode
Python cv2 version
OpenCV thread count
input shapes and dtypes
batch size
target sizes
warm-up count
sample count
median and dispersion
```

## Test behavior

The performance file contains both benchmarks and assertions.

| Assertion type | Examples |
|---|---|
| Correctness | output count, shape, dtype, pixel equality, luminance tolerance |
| Resource behavior | repeated calls, concurrent calls, memory cleanup |
| Broad timing guard | completion limits and scaling bounds intended for CI |
| Statistical timing | pytest-benchmark cases in `TestDetailedBenchmarks` |

Single-run `time.perf_counter` output is diagnostic. It is not a stable result
across hosts or OpenCV builds.

## Published numbers

This repository does not currently store benchmark artifacts keyed by commit,
host, and OpenCV configuration. Fixed timing tables are therefore not included
in this document.
