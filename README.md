# trainingsample

[![Crates.io](https://img.shields.io/crates/v/trainingsample.svg)](https://crates.io/crates/trainingsample)
[![PyPI](https://img.shields.io/pypi/v/trainingsample.svg)](https://pypi.org/project/trainingsample/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

fast rust reimplementation of image/video processing ops that don't suck at parallelism

## install

```bash
# python (recommended)
pip install trainingsample

# rust
cargo add trainingsample
```

## what it does

hybrid high-performance image processing that uses the best implementation for each operation:
- **TSR-optimized**: cropping, luminance calculation (SIMD parallelized)  
- **OpenCV-powered**: resizing operations (industry-standard performance)
- **unified API**: single Python/Rust interface, static wheels with all dependencies

batch operations that actually release the GIL and use all your cores. zero-copy numpy integration when possible.

## python usage

```python
import numpy as np
import trainingsample as ts

# load some images
images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]

# batch crop (x, y, width, height)
cropped = ts.batch_crop_images(images, [(50, 50, 200, 200)] * 10)

# center crop to square
center_cropped = ts.batch_center_crop_images(images, [(224, 224)] * 10)

# random crops
random_cropped = ts.batch_random_crop_images(images, [(256, 256)] * 10)

# resize (width, height)
resized = ts.batch_resize_images(images, [(224, 224)] * 10)

# luminance calculation
luminances = ts.batch_calculate_luminance(images)  # returns list of floats

# video processing (frames, height, width, channels)
video = np.random.randint(0, 255, (30, 480, 640, 3), dtype=np.uint8)
resized_video = ts.batch_resize_videos([video], [(224, 224)])
```

## rust usage

```rust
use trainingsample::{
    batch_crop_image_arrays, batch_resize_image_arrays,
    batch_calculate_luminance_arrays
};
use ndarray::Array3;

// create some test data
let images: Vec<Array3<u8>> = (0..10)
    .map(|_| Array3::zeros((480, 640, 3)))
    .collect();

// batch operations
let crop_boxes = vec![(50, 50, 200, 200); 10]; // (x, y, width, height)
let cropped = batch_crop_image_arrays(&images, &crop_boxes);

let target_sizes = vec![(224, 224); 10]; // (width, height)
let resized = batch_resize_image_arrays(&images, &target_sizes);

let luminances = batch_calculate_luminance_arrays(&images);
```

## api reference

### python functions

#### `batch_crop_images(images, crop_boxes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `crop_boxes`: list of (x, y, width, height) tuples
- returns: list of cropped numpy arrays
- **implementation**: TSR-optimized for mixed-shape batching

#### `batch_center_crop_images(images, target_sizes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of center-cropped numpy arrays
- **implementation**: TSR-optimized for mixed-shape batching

#### `batch_random_crop_images(images, target_sizes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of randomly cropped numpy arrays
- **implementation**: TSR-optimized for mixed-shape batching

#### `batch_resize_images(images, target_sizes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of resized numpy arrays
- **implementation**: OpenCV for optimal performance

#### `batch_calculate_luminance(images)`
- `images`: list of numpy arrays (H, W, 3) uint8
- returns: list of float luminance values
- **implementation**: TSR SIMD-optimized (10-35x faster than NumPy)

#### `batch_resize_videos(videos, target_sizes)`
- `videos`: list of numpy arrays (T, H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of resized video numpy arrays

### rust functions

same signatures but with `ndarray::Array3<u8>` and `ndarray::Array4<u8>` instead of numpy arrays. check the docs for details.

## architecture

TSR uses a **best-of-breed hybrid approach** for optimal performance:

### operation selection
- **cropping operations**: TSR implementation
  - mixed-shape batching (8 different input shapes → 7 different output shapes)
  - single API call: `tsr.batch_crop_images(mixed_images, mixed_crops)`
  - vs competitor: individual loops required for each shape combination
  
- **luminance calculation**: TSR SIMD implementation  
  - **18x faster** than NumPy for mixed-shape batches
  - **35x faster** than NumPy for uniform batches
  - vectorized across different image sizes in single batch call

- **resize operations**: OpenCV implementation
  - industry-standard performance and quality
  - highly optimized C++ implementations  
  - **7-25x faster** than TSR resize implementations

### static wheel distribution
- OpenCV **statically linked** into wheel (no external dependencies)
- single `pip install trainingsample` - no opencv-python conflicts
- consistent performance across platforms
- ~50MB wheel includes all optimizations

## features

- **hybrid architecture**: best implementation for each operation
- parallel processing with rayon (actually uses your cores) 
- zero-copy numpy integration via rust-numpy
- proper error handling (no silent failures)
- **static OpenCV** bundled (no external dependencies)
- no python threading nonsense, GIL is released
- memory efficient batch operations
- supports both images and videos

## performance

tested on realistic mixed-shape datasets because toy data means nothing:

### hybrid architecture benchmarks

#### luminance calculation (TSR-optimized)
- **mixed-shape batch** (6 different sizes): **18.19x faster** than NumPy
- **uniform batch** (16 × 1024×1024): **35.25x faster** than NumPy  
- **throughput**: 5,434 images/sec vs NumPy's 298 images/sec
- **key advantage**: single batch call handles different image sizes

#### resize operations (OpenCV-powered)
- **performance**: OpenCV **25x faster** than TSR implementations
- **quality**: industry-standard algorithms (bilinear, Lanczos, etc.)
- **mixed shapes**: handles different input/output sizes efficiently
- **integration**: seamless within TSR batch operations

#### cropping operations (TSR-optimized)
- **mixed-shape advantage**: 8 different input shapes → 7 different output shapes
- **API simplicity**: `tsr.batch_crop_images(mixed_images, mixed_crops)`
- **vs competitors**: no loops needed, single batch call
- **memory efficiency**: zero-copy operations where possible

### threading reality check
spoiler: ThreadPoolExecutor won't save you. the rust bindings don't release the GIL as effectively as you'd hope (1.08x speedup vs expected 4x). just use batch processing - it's 6x faster than threading anyway.

### batch sizes that matter
- **luminance**: 8-16 images for best throughput/memory balance
- **resizing**: 4-8 images optimal (OpenCV-optimized)
- **cropping**: benefits from larger batches due to mixed-shape handling
- **memory usage**: ~78MB per 5120x5120 image, plan accordingly

## Apple Silicon Performance (M3 Max)

Optimized SIMD implementations with concrete benchmarks:

| Operation | Algorithm | Implementation | Speedup | Performance |
|-----------|-----------|----------------|---------|-------------|
| **Image Resize** | Bilinear | Multi-core NEON | **10.2x** | 1,412 MPx/s |
| **Image Resize** | Lanczos4 | Metal GPU | **11.8x** | 112 MPx/s |
| **Format Conversion** | RGB→RGBA | Portable SIMD | **4.4x** | 1,500 MPx/s |
| **Format Conversion** | RGBA→RGB | Portable SIMD | **2.6x** | 1,651 MPx/s |
| **Luminance Calc** | RGB→Y | NEON SIMD | **4.7x** | 545 images/sec |

**Key Insights:**
- **CPU SIMD** (multi-core NEON) optimal for memory-bound operations like bilinear resize
- **GPU Metal** dominates compute-intensive algorithms like Lanczos4 interpolation
- **Unified memory** architecture enables zero-copy GPU operations
- **Automatic selection** between CPU/GPU based on algorithm characteristics

Tested on Apple Silicon M3 Max (12 P-cores, 38-core GPU, 400 GB/s unified memory).

## why this hybrid approach

### vs pure opencv/pil
- **OpenCV alone**: excellent resize performance, but poor mixed-shape batching
- **PIL**: slow, GIL-bound, no batch operations
- **TSR hybrid**: combines OpenCV's resize speed with TSR's batch/SIMD advantages

### vs pure rust implementations  
- **TSR resize**: slower than OpenCV's highly-optimized C++ (7-25x difference)
- **TSR luminance**: faster than NumPy due to SIMD (18-35x speedup)
- **best of both**: use optimal implementation for each operation

### static distribution advantage
- **no dependency conflicts**: opencv-python version compatibility issues eliminated
- **consistent performance**: same optimized OpenCV across all platforms
- **simple deployment**: single wheel, no system dependencies

## building from source

```bash
# for python
pip install maturin
maturin develop --release

# for rust
cargo build --release
```

requires rust 1.70+ and python 3.11+ if you want the python bindings.

## license

MIT. do whatever you want with it, leave attribution in-tact.
