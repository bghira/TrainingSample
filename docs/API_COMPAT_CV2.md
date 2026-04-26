# OpenCV API Compatibility Guide

TrainingSample exposes a subset of OpenCV-style image APIs plus batch-oriented helpers. The goal is to reduce Python loop overhead for common preprocessing workloads, not to implement the full `cv2` surface.

## Quick Start

```python
import cv2
import numpy as np
import trainingsample as tsr
```

Use TrainingSample where a matching helper exists:

```python
resized = tsr.resize(image, (224, 224), interpolation=tsr.INTER_LINEAR)
gray = tsr.cvt_color(image, tsr.COLOR_RGB2GRAY)
edges = tsr.canny(image, threshold1=50, threshold2=150)
```

For batches, prefer the batch APIs instead of a Python loop:

```python
images = [load_image(path) for path in paths]
sizes = [(224, 224)] * len(images)

resized = tsr.batch_resize_images(images, sizes)
luminances = tsr.batch_calculate_luminance(resized)
```

## Supported OpenCV-Style Operations

### Image Decoding

```python
with open("image.jpg", "rb") as f:
    img_bytes = f.read()

img = tsr.imdecode(img_bytes, tsr.IMREAD_COLOR)
img_gray = tsr.imdecode(img_bytes, tsr.IMREAD_GRAYSCALE)
```

### Color Space Conversion

```python
gray = tsr.cvt_color(image, tsr.COLOR_RGB2GRAY)
bgr = tsr.cvt_color(image, tsr.COLOR_RGB2BGR)
```

### Edge Detection

```python
edges = tsr.canny(image, threshold1=50, threshold2=150)
```

### Image Resizing

```python
resized = tsr.resize(image, (width, height), interpolation=tsr.INTER_LINEAR)
```

Supported interpolation constants:

```python
tsr.INTER_NEAREST
tsr.INTER_LINEAR
tsr.INTER_CUBIC
tsr.INTER_LANCZOS4
```

## Batch Operations

### Cropping

```python
crop_boxes = [(x, y, width, height) for image in images]
cropped = tsr.batch_crop_images(images, crop_boxes)
```

Center and random crop helpers use target sizes:

```python
target_sizes = [(224, 224)] * len(images)
center_cropped = tsr.batch_center_crop_images(images, target_sizes)
random_cropped = tsr.batch_random_crop_images(images, target_sizes)
```

### Resizing

```python
target_sizes = [(224, 224)] * len(images)
resized = tsr.batch_resize_images(images, target_sizes)
```

The public resize API returns a list of owned NumPy arrays. Current implementation uses OpenCV-backed resize internally and transfers owned Rust arrays into NumPy without an additional copy.

### Luminance

```python
luminances = tsr.batch_calculate_luminance(images)
```

For contiguous arrays, luminance uses a channel-sum fast path. Non-contiguous arrays are accepted by the safe public API but may run through a slower ndarray fallback.

### Video Resizing

```python
videos = [video_array]  # shape: (frames, height, width, 3)
target_sizes = [(224, 224)]
resized_videos = tsr.batch_resize_videos(videos, target_sizes)
```

## Zero-Copy Entry Points

Some lower-level APIs expose stricter zero-copy behavior:

```python
cropped = tsr.batch_crop_images_zero_copy(images, crop_boxes)
luminances = tsr.batch_calculate_luminance_zero_copy(images)
resized = tsr.batch_resize_images_zero_copy(images, target_sizes)
```

These functions are intended for contiguous arrays. Unsafe zero-copy crop and resize paths reject non-contiguous views with a `ValueError`.

## Video Capture and Writing

```python
cap = tsr.VideoCapture("video.mp4")

if cap.is_opened():
    ret, frame = cap.read()
    if ret:
        luminance = tsr.batch_calculate_luminance([frame])

cap.release()
```

```python
fourcc = tsr.fourcc("M", "J", "P", "G")
writer = tsr.VideoWriter("output.avi", fourcc, 30.0, (width, height))

for frame in frames:
    writer.write(frame)

writer.release()
```

## Object Detection

```python
classifier = tsr.CascadeClassifier("haarcascade_frontalface_alt.xml")
faces = classifier.detect_multi_scale(image)
```

## Benchmark Snapshot

The following numbers came from the local benchmark suite after the latest Python-interface optimization work:

```bash
.venv/bin/python -m pytest tests/test_performance_benchmarks.py -q -s
```

Environment: Linux x86_64, CPython 3.13, NumPy 2.3.4, system OpenCV 4.11 through the Rust `opencv` crate.

| Scenario | TrainingSample | Comparison in same run |
|----------|----------------|------------------------|
| Batch resize, 4 mixed-size images | 0.4 ms | OpenCV loop: 2.6 ms |
| Batch luminance, 4 mixed-size images | 0.6 ms | OpenCV loop: 0.9 ms |
| Resize + luminance pipeline, 4 mixed-size images | 0.6 ms | OpenCV loop: 2.1 ms |
| Mixed-shape luminance, 6 images | 3.3 ms | NumPy loop: 19.4 ms |
| Mixed-shape crop, 8 images | 3.3 ms | NumPy slicing loop: near-zero because slicing returns views |

The crop comparison is intentionally caveated: NumPy slicing can be effectively free when it returns a view. TrainingSample returns owned output arrays, which is the right comparison when the next stage needs independent contiguous buffers.

## Migration Notes

### Prefer Batch APIs for Repeated Work

```python
# OpenCV loop
results = [cv2.resize(img, (224, 224)) for img in images]

# TrainingSample batch call
results = tsr.batch_resize_images(images, [(224, 224)] * len(images))
```

### Keep Inputs Contiguous When Performance Matters

```python
if not image.flags["C_CONTIGUOUS"]:
    image = np.ascontiguousarray(image)
```

Public safe APIs accept many strided views, but contiguous arrays are usually faster and are required by strict zero-copy paths.

### Validate Your Own Workload

Image shape, interpolation, batch size, and memory bandwidth can change results. Benchmark the exact pipeline you intend to ship:

```python
import time

start = time.perf_counter()
results = tsr.batch_resize_images(images, sizes)
duration = time.perf_counter() - start

print(f"{len(images) / duration:.1f} images/sec")
```

## Limitations

- This is not a complete `cv2` replacement.
- Batch APIs allocate owned output arrays.
- Small inputs can be dominated by call overhead.
- Zero-copy functions require contiguous arrays for crop and resize paths.
- System OpenCV and wheel build configuration can affect performance and available codecs.
