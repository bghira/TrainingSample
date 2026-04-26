# trainingsample

[![Crates.io](https://img.shields.io/crates/v/trainingsample.svg)](https://crates.io/crates/trainingsample)
[![PyPI](https://img.shields.io/pypi/v/trainingsample.svg)](https://pypi.org/project/trainingsample/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TrainingSample provides Rust-backed Python bindings for common image and video preprocessing operations used in ML data pipelines. It combines OpenCV-backed resizing with Rust implementations for batching, cropping, luminance calculation, format conversion, and video helpers.

The project is designed for workloads where Python-side loops and repeated boundary crossings become visible. It is not a blanket replacement for all of `cv2`, and performance depends on image size, batch shape, CPU, OpenCV build, and memory bandwidth.

## install

```bash
# python
pip install trainingsample

# rust
cargo add trainingsample
```

## python usage

```python
import numpy as np
import trainingsample as tsr

images = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(8)
]

crop_boxes = [(50, 50, 200, 200)] * len(images)
cropped = tsr.batch_crop_images(images, crop_boxes)

target_sizes = [(224, 224)] * len(images)
resized = tsr.batch_resize_images(images, target_sizes)

luminances = tsr.batch_calculate_luminance(resized)
```

OpenCV-compatible helpers are also exported for common operations:

```python
decoded = tsr.imdecode(image_bytes, tsr.IMREAD_COLOR)
gray = tsr.cvt_color(decoded, tsr.COLOR_RGB2GRAY)
edges = tsr.canny(decoded, threshold1=50, threshold2=150)
resized = tsr.resize(decoded, (224, 224), interpolation=tsr.INTER_LINEAR)
```

## rust usage

```rust
use ndarray::Array3;
use trainingsample::{
    batch_calculate_luminance_arrays, batch_crop_image_arrays, batch_resize_image_arrays,
};

let images: Vec<Array3<u8>> = (0..10)
    .map(|_| Array3::zeros((480, 640, 3)))
    .collect();

let crop_boxes = vec![(50, 50, 200, 200); 10]; // (x, y, width, height)
let cropped = batch_crop_image_arrays(&images, &crop_boxes);

let target_sizes = vec![(224, 224); 10]; // (width, height)
let resized = batch_resize_image_arrays(&images, &target_sizes);

let luminances = batch_calculate_luminance_arrays(&images);
```

## api reference

### `batch_crop_images(images, crop_boxes)`

- `images`: list of NumPy arrays shaped `(H, W, C)` with `uint8` data
- `crop_boxes`: list of `(x, y, width, height)` tuples
- returns: list of cropped NumPy arrays
- notes: output arrays are owned by NumPy without an extra copy from the owned Rust array

### `batch_center_crop_images(images, target_sizes)`

- `images`: list of NumPy arrays shaped `(H, W, C)` with `uint8` data
- `target_sizes`: list of `(width, height)` tuples
- returns: list of center-cropped NumPy arrays

### `batch_random_crop_images(images, target_sizes)`

- `images`: list of NumPy arrays shaped `(H, W, C)` with `uint8` data
- `target_sizes`: list of `(width, height)` tuples
- returns: list of randomly cropped NumPy arrays

### `batch_resize_images(images, target_sizes)`

- `images`: list of NumPy arrays shaped `(H, W, 3)` with `uint8` data
- `target_sizes`: list of `(width, height)` tuples
- returns: list of resized NumPy arrays
- implementation: OpenCV-backed resize with Rust/PyO3 conversion handling

### `batch_calculate_luminance(images)`

- `images`: list of NumPy arrays shaped `(H, W, C)` with `uint8` data
- returns: list of float luminance values
- notes: contiguous RGB/RGBA-like arrays use a channel-sum fast path; strided arrays fall back to the general ndarray path

### `batch_resize_videos(videos, target_sizes)`

- `videos`: list of NumPy arrays shaped `(T, H, W, 3)` with `uint8` data
- `target_sizes`: list of `(width, height)` tuples
- returns: list of resized video NumPy arrays

## current benchmark snapshot

These numbers are from the local benchmark run after the latest Python-interface optimizations:

```bash
.venv/bin/python -m pytest tests/test_performance_benchmarks.py -q -s
```

Environment: Linux x86_64, CPython 3.13, NumPy 2.3.4, system OpenCV 4.11 through the Rust `opencv` crate. Treat these as a point-in-time reference, not a cross-machine guarantee.

| Benchmark | Before | After | Notes |
|-----------|--------|-------|-------|
| Crop batch, 16 images | 22.9 ms | 0.4 ms | Public `batch_crop_images` path |
| Mixed-shape crop, 8 images | 50.2 ms | 3.3 ms | Mixed input and output sizes |
| Luminance batch, 4 mixed images | 10.4 ms | 0.6 ms | Now faster than the OpenCV comparison in this run |
| Mixed-shape luminance, 6 images | 78.3 ms | 3.3 ms | NumPy comparison was 19.4 ms in this run |
| Complete resize + luminance pipeline | 5.9 ms | 0.6 ms | Four mixed-size inputs to 224x224 |

Pytest-benchmark means from the same suite:

| Benchmark | Mean after |
|-----------|------------|
| Center crop | 55.2 us |
| Resize operations | 353.1 us |
| Luminance calculation | 417.2 us |
| Crop operations | 583.8 us |
| Pipeline | 3.44 ms |
| Video processing | 2.85 ms |

## architecture

TrainingSample uses different implementations for different operation types:

- Cropping: Rust/ndarray implementation with owned-array transfer into NumPy.
- Luminance: Rust channel-sum fast path for contiguous arrays, with a general ndarray fallback for non-contiguous inputs.
- Resize: OpenCV-backed implementation for image quality and mature interpolation behavior.
- Video resize: OpenCV-backed frame resizing with batched Python binding output.
- Format conversion: Rust SIMD implementation where the `simd` feature is enabled.

The optimized path generally requires contiguous `uint8` arrays. Views such as `image[:, ::2, :]` remain supported by safe public APIs, but they may use slower fallback paths.

## features

- Python bindings through PyO3 and rust-numpy
- Batch APIs for images and videos
- OpenCV-compatible constants and helper functions for common operations
- Optional SIMD feature for format conversion and selected numeric paths
- Error handling for invalid dimensions, unsupported channels, and invalid crop bounds
- Source build support for dynamic or static OpenCV configurations

## building from source

```bash
pip install maturin
maturin develop --release
```

The OpenCV Rust bindings need to find a working OpenCV and Clang installation. If the environment has stale OpenCV or LLVM variables, unset them before building:

```bash
env -u OPENCV_LINK_LIBS -u OPENCV_LINK_PATHS -u OPENCV_INCLUDE_PATHS \
    -u LIBCLANG_PATH -u LLVM_CONFIG_PATH \
    maturin develop --release
```

See [docs/BUILDING_STATIC_OPENCV.md](docs/BUILDING_STATIC_OPENCV.md) for static OpenCV bundle notes.

## license

MIT. See [LICENSE](LICENSE).
