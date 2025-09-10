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

batch image operations that actually release the GIL and use all your cores. crop, resize, luminance calc, video frame processing. zero-copy numpy integration when possible.

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

#### `batch_center_crop_images(images, target_sizes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of center-cropped numpy arrays

#### `batch_random_crop_images(images, target_sizes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of randomly cropped numpy arrays

#### `batch_resize_images(images, target_sizes)`
- `images`: list of numpy arrays (H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of resized numpy arrays

#### `batch_calculate_luminance(images)`
- `images`: list of numpy arrays (H, W, 3) uint8
- returns: list of float luminance values

#### `batch_resize_videos(videos, target_sizes)`
- `videos`: list of numpy arrays (T, H, W, 3) uint8
- `target_sizes`: list of (width, height) tuples
- returns: list of resized video numpy arrays

### rust functions

same signatures but with `ndarray::Array3<u8>` and `ndarray::Array4<u8>` instead of numpy arrays. check the docs for details.

## features

- parallel processing with rayon (actually uses your cores)
- zero-copy numpy integration via rust-numpy
- proper error handling (no silent failures)
- works with opencv, pil, whatever
- no python threading nonsense, GIL is released
- memory efficient batch operations
- supports both images and videos

## why not opencv/pil/whatever

because they're slow, don't parallelize properly, or hold the GIL. this uses full parallelism and doesn't care about python threading limitations.

## building from source

```bash
# for python
pip install maturin
maturin develop --release

# for rust
cargo build --release
```

requires rust 1.70+ and python 3.9+ if you want the python bindings.

## license

MIT. do whatever you want with it, leave attribution in-tact.
