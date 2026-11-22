# Benchmarks

## Commands

Build the checkout with the project Rust toolchain and a discoverable native
OpenCV installation. The Python comparison uses the headless OpenCV wheel.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'maturin>=1.16,<2' opencv-python-headless
maturin develop --release --extras dev
```

Run the paired comparisons and save timings, IQRs and per-round data.

```bash
mkdir -p output
scripts/run_competitive_benchmarks.sh --benchmark-json=output/competitive.json
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

Compare the native luminance entry points on deterministic RGB/RGBA images and
strided views:

```bash
cargo run --release --example luminance_benchmark > luminance.csv
```

This reports median, minimum, and maximum microseconds per call over eleven
rounds of twenty calls, after ten warmups. The RGB channel means are exactly
127.5 for every case, so the reported luminance should also be 127.5. Run before
and after a change on the same idle host; record the commit and CPU alongside
the CSV. To compare the fallback build, add `--no-default-features`.

## Measured operations

The competitive suite measures public Python calls on seeded, contiguous
`uint8` inputs. It checks outputs before each timed case. Image sizes below
use height × width.

| Case | Input and result | Reference |
|---|---|---|
| Crop | Four RGB images, owned 100×100 crops at (10, 10) | NumPy slice and copy |
| Resize | RGB images at 480×640, 1024×768, 256×256 and 128×512, all resized to 224×224 | Loop over `cv2.resize` with `INTER_LINEAR` |
| Luminance | Same four input images, one mean luminance per image | `cv2.cvtColor(COLOR_RGB2GRAY)` then `cv2.mean` |
| Pipeline | Resize the four images, then return their luminances | Same resize and luminance loops |
| Video resize | 30 RGB frames at 224×224, returned as one 112×112 clip | Loop over `cv2.resize` into slices of one newly allocated output |
| Add or remove alpha | One 1024×1024 RGB or RGBA image, added alpha 255 | `cv2.cvtColor` with `COLOR_RGB2RGBA` or `COLOR_RGBA2RGB` |

## Comparison constraints

| Topic | Constraint |
|---|---|
| Crop ownership | The competitive runner copies the NumPy crop to match TSR ownership. The older performance tests also time NumPy views. |
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

The large-image luminance scaling check uses deterministic inputs and validates
results against NumPy. It compares medians from seven interleaved timing rounds,
with five calls per batch size in each round. Its CI guard allows up to twice
the per-image cost as batches grow; it does not require a parallel speedup on
shared runners or across CPU cache sizes.

## Apple M3 Max measurements, 2026-11-21

TSR 0.4.0 from `86fe3a4` supplied the baseline. The working-tree build changed
the two format conversion loops. Both used Rust 1.89.0 in release mode with
SIMD enabled, dynamically linked to Homebrew OpenCV 5.0.0. The host ran macOS
15.7.7 arm64, CPython 3.12.15 and NumPy 2.5.5. The OpenCV thread count was 16.
The Python wheels were `opencv-python-headless` 5.0.0.93 and 5.1.0.95.

Timings are median µs per complete call, including output allocation, over
21 rounds of 10 calls after 3 warmup rounds of 10. Each column uses the first
complete run for that configuration. The TSR columns used the 5.0.0 Python
reference. Crop uses the same NumPy reference in both wheel configurations.

| Operation | TSR baseline | TSR optimized | Reference 5.0 | Reference 5.1 |
|---|---:|---:|---:|---:|
| Crop | 4.5 | 4.4 | 5.5 | 5.6 |
| Resize | 217.6 | 210.0 | 248.2 | 237.5 |
| Luminance | 81.7 | 90.2 | 166.6 | 186.5 |
| Resize then luminance | 228.7 | 220.7 | 292.1 | 279.6 |
| Video resize | 384.2 | 351.4 | 541.9 | 74.0 |
| RGB → RGBA | 509.2 | 182.6 | 217.3 | 221.8 |
| RGBA → RGB | 553.1 | 144.4 | 171.7 | 172.0 |

**Color conversion.** In exploratory 2-second native samples of the original
loops (Rust 1.97 release), 595 of 1,719 main-thread RGB → RGBA samples landed
in `memset`. The reverse conversion spent 1,689 top-of-stack samples inside
the inlined native call. The old loops filled every output byte with alpha or
repeatedly extended the destination by 24 bytes. Writing whole pixels into one
sized output reduced the medians by 2.79× and 3.83×. The respective IQRs after
the change were 9.8 and 8.7 µs. A second matched-version run gave 182.9 and
141.9 µs. Both conversions passed byte-for-byte comparison.

**Video resize.** In two paired runs with cv2 5.1, the 30-frame half-size resize
took 74–78 µs in cv2 and 346–360 µs in TSR. Native samples show the wheel using
`kleidicv::neon::resize_to_quarter_u8` while the Homebrew library uses
`kleidicv_resize_generic_stripe_u8` and TBB. In an interleaved control run,
changing the target from 112×112 to 111×111 or 113×113 brought cv2 to
502–535 µs and TSR to 381–383 µs. The next comparison is the TSR binding
linked to an OpenCV build with the same half-scale kernel. These results
measure one host and two independently built OpenCV distributions.

Raw pytest-benchmark JSON, native stack samples and the adjacent-size control
timings from this run are in the local `output/cv2-2026-11-21/` directory.
