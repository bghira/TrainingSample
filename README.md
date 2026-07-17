# trainingsample

[![Crates.io](https://img.shields.io/crates/v/trainingsample.svg)](https://crates.io/crates/trainingsample)
[![PyPI](https://img.shields.io/pypi/v/trainingsample.svg)](https://pypi.org/project/trainingsample/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Rust image and video operations exposed through Python and Rust APIs.

| Item | Value |
|---|---|
| Python package | `trainingsample` |
| Rust crate | `trainingsample` |
| Python version | 3.11 or newer |
| Python array type | NumPy `uint8` |
| Image layout | `(height, width, channels)` |
| Video layout | `(frames, height, width, channels)` |
| Size tuple order | `(width, height)` |
| License | MIT |

## Install

```bash
python -m pip install trainingsample
cargo add trainingsample
```

## Python example

```python
import numpy as np
import trainingsample as tsr

images = [
    np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    for _ in range(8)
]

cropped = tsr.batch_crop_images(
    images,
    [(50, 50, 320, 320)] * len(images),
)
resized = tsr.batch_resize_images(
    cropped,
    [(224, 224)] * len(cropped),
)
luminance = tsr.batch_calculate_luminance(resized)
```

## Python functions

| Function | Input | Output |
|---|---|---|
| `load_image_batch(paths)` | file paths | `list[bytes \| None]` |
| `batch_crop_images(images, boxes)` | images; `(x, y, width, height)` per image | owned images |
| `batch_center_crop_images(images, sizes)` | images; target size per image | owned images |
| `batch_random_crop_images(images, sizes)` | images; target size per image | owned images |
| `batch_resize_images(images, sizes)` | RGB images; target size per image | owned RGB images |
| `batch_resize_videos(videos, sizes)` | RGB videos; target size per video | owned RGB videos |
| `batch_calculate_luminance(images)` | images | `list[float]` |
| `rgb_to_rgba_optimized(image, alpha)` | RGB image; `uint8` alpha | RGBA image and elapsed time |
| `rgba_to_rgb_optimized(image)` | RGBA image | RGB image and elapsed time |

Specialized entry points:

| Function | Constraint |
|---|---|
| `batch_crop_images_zero_copy` | C-contiguous input |
| `batch_center_crop_images_zero_copy` | C-contiguous input |
| `batch_resize_images_zero_copy` | C-contiguous, three-channel input |
| `batch_resize_images_iterator` | C-contiguous, three-channel input |
| `batch_calculate_luminance_zero_copy` | accepts ndarray views |

The compatibility helpers use names such as `imdecode_py`, `cvt_color_py`, and
`resize_py`. See [the exact Python export table](docs/API_COMPAT_CV2.md).

## Rust example

```rust
use ndarray::Array3;
use trainingsample::crop_image_array;

let image = Array3::<u8>::zeros((480, 640, 3));
let cropped = crop_image_array(&image.view(), 50, 50, 320, 320).unwrap();
assert_eq!(cropped.dim(), (320, 320, 3));
```

## Implementations

| Operation | Implementation |
|---|---|
| Decode | `image` crate; JPEG, PNG, and WebP features enabled |
| Crop | Rust row copies for contiguous arrays; ndarray fallback for strided views |
| Luminance | Rust contiguous fast path; ndarray fallback |
| Image resize | OpenCV for `batch_resize_images` and named OpenCV resize helpers |
| Video resize | OpenCV frame resize into an owned four-dimensional output |
| Color conversion | Rust; SIMD feature used by optimized RGB/RGBA functions |
| Canny helper | `imageproc` |

The default Cargo feature set is `simd`. Python wheels are built with
`python-bindings`, `opencv`, and `simd`; macOS release wheels also enable
`metal`.

## Source build

```bash
python -m pip install 'maturin>=1,<2'
maturin develop --release
```

Build and test commands used by CI:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --no-default-features \
  --features python-bindings,simd,opencv -- -D warnings
cargo test --no-default-features --features simd,opencv
python -m pytest -q
```

OpenCV and libclang must be discoverable for source builds with the `opencv`
feature. Static wheel configuration is documented in
[docs/BUILDING_STATIC_OPENCV.md](docs/BUILDING_STATIC_OPENCV.md).

## Documentation

- [Benchmark definitions and commands](BENCHMARKS.md)
- [Python compatibility helpers](docs/API_COMPAT_CV2.md)
- [Static OpenCV build](docs/BUILDING_STATIC_OPENCV.md)

## Limits

- The Python API is not a drop-in replacement for `cv2`.
- Crop and resize functions return owned arrays.
- OpenCV resize paths require three-channel input.
- Strict zero-copy crop and resize entry points reject non-contiguous input.
- Runtime depends on input dimensions, batch size, host CPU, memory bandwidth,
  OpenCV build flags, and OpenCV thread settings.
