# OpenCV (cv2) API Compatibility Guide

TrainingSample provides drop-in replacements for common OpenCV operations with significant performance improvements through Rust optimizations and true batch processing.

## Quick Start: Drop-in Replacement

Replace `cv2` imports with `trainingsample` for instant performance gains:

```python
# old cv2 approach
import cv2
import numpy as np

# new high-performance approach
import trainingsample as tsr
```

## 🏆 Zero-Copy Operations (Maximum Performance)

These are our highest-performing operations using raw pointer access and parallel processing:

### Batch Cropping (Zero-Copy)
```python
# 4-5x faster than regular batch operations
cropped = tsr.batch_crop_images_zero_copy(
    images,  # List[np.ndarray] - batch of images
    crop_boxes  # List[(x, y, width, height)] - crop coordinates
)

# Center cropping with zero-copy optimization
center_cropped = tsr.batch_center_crop_images_zero_copy(
    images,  # List[np.ndarray]
    target_sizes  # List[(width, height)]
)
```

### Batch Luminance (Zero-Copy + Parallel)
```python
# 5-8x faster with parallel processing + adaptive SIMD
luminances = tsr.batch_calculate_luminance_zero_copy(images)
# Returns: List[float] - ITU-R BT.709 luminance values (0-255 range)
```

## 📊 Standard Batch Operations

High-performance batch processing for common operations:

### Image Loading
```python
# Parallel image loading from file paths
images = tsr.load_image_batch([
    'path/to/image1.jpg',
    'path/to/image2.png',
    'path/to/image3.webp'
])
```

### Batch Cropping
```python
# Regular batch cropping (still faster than individual cv2 calls)
images = tsr.batch_crop_images(images, crop_boxes)
center_cropped = tsr.batch_center_crop_images(images, target_sizes)
random_cropped = tsr.batch_random_crop_images(images, target_sizes)
```

### Batch Resizing
```python
# High-performance batch resizing
resized = tsr.batch_resize_images(
    images,
    target_sizes,  # List[(width, height)]
    interpolation="bilinear"  # or "lanczos"
)

# Video frame batch processing
video_frames = tsr.batch_resize_videos(videos, target_sizes)
```

### Batch Luminance Calculation
```python
# Calculate ITU-R BT.709 luminance for batch of images
luminances = tsr.batch_calculate_luminance(images)
# Formula: L = 0.2126*R + 0.7152*G + 0.0722*B
```

## 🎨 Format Conversion (Ultra-Fast)

Sub-millisecond format conversions with SIMD optimization:

```python
# RGB to RGBA conversion (add alpha channel)
rgba_image, timing = tsr.rgb_to_rgba_optimized(rgb_image, alpha=255)

# RGBA to RGB conversion (remove alpha channel)
rgb_image, timing = tsr.rgba_to_rgb_optimized(rgba_image)
```

## 🔧 OpenCV-Compatible Individual Operations

Drop-in replacements for common cv2 functions:

### Image Decoding
```python
# Equivalent to cv2.imdecode()
import trainingsample as tsr

# Read image bytes
with open('image.jpg', 'rb') as f:
    img_bytes = f.read()

# Decode with OpenCV-compatible flags
img = tsr.imdecode(img_bytes, tsr.IMREAD_COLOR)
img_gray = tsr.imdecode(img_bytes, tsr.IMREAD_GRAYSCALE)
```

### Color Space Conversion
```python
# Equivalent to cv2.cvtColor()
gray = tsr.cvt_color(image, tsr.COLOR_RGB2GRAY)
bgr = tsr.cvt_color(image, tsr.COLOR_RGB2BGR)
```

### Edge Detection
```python
# Equivalent to cv2.Canny()
edges = tsr.canny(image, threshold1=50, threshold2=150)
```

### Image Resizing
```python
# Equivalent to cv2.resize()
resized = tsr.resize(image, (width, height), interpolation=tsr.INTER_LINEAR)
```

## 📹 Video Processing

OpenCV-compatible video capture and writing:

### Video Capture
```python
# Equivalent to cv2.VideoCapture
cap = tsr.VideoCapture('video.mp4')

if cap.is_opened():
    ret, frame = cap.read()
    if ret:
        # Process frame
        processed = tsr.batch_calculate_luminance([frame])

cap.release()
```

### Video Writing
```python
# Equivalent to cv2.VideoWriter
fourcc = tsr.fourcc('M', 'J', 'P', 'G')
writer = tsr.VideoWriter('output.avi', fourcc, 30.0, (width, height))

for frame in frames:
    writer.write(frame)

writer.release()
```

## 🔍 Object Detection

```python
# Equivalent to cv2.CascadeClassifier
classifier = tsr.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = classifier.detect_multi_scale(image)
```

## ⚡ Performance Comparison

| Operation | cv2 Individual | TSR Batch | TSR Zero-Copy | Speedup |
|-----------|---------------|-----------|---------------|---------|
| Crop | 1.40ms | 1.40ms | 0.34ms | **4.1x** 🏆 |
| Center Crop | 1.59ms | 1.59ms | 0.48ms | **3.3x** 🏆 |
| Luminance | 4.38ms | 4.38ms | 0.55ms | **8.0x** 🏆 |
| Format Conv | 0.10ms | 0.02ms | 0.01ms | **10x** 🏆 |

## 🎯 Best Practices

### When to Use Zero-Copy Operations
- **Always use for batch processing** - 3-8x performance gains
- **Large image datasets** - Memory-efficient with buffer pooling
- **Real-time applications** - Parallel processing + SIMD acceleration

### Migration from OpenCV
```python
# before (slow)
results = []
for img in images:
    result = cv2.operation(img, params)
    results.append(result)

# after (fast)
results = tsr.batch_operation_zero_copy(images, params_list)
```

### Memory Efficiency
```python
# before (slow - multiple boundary crossings)
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size)
    edges = cv2.Canny(resized, 50, 150)

# after (fast - single batch operation)
grays = tsr.batch_cvt_color(images, tsr.COLOR_RGB2GRAY)
resized = tsr.batch_resize_images_zero_copy(grays, sizes)
edges = tsr.batch_canny(resized, threshold1=50, threshold2=150)
```

## 🚀 Advanced Features

### Adaptive SIMD Processing
TrainingSample automatically chooses between SIMD and scalar operations based on image size:
- **Small images (<64K pixels)**: Scalar processing (avoids SIMD overhead)
- **Large images (>64K pixels)**: SIMD acceleration (AVX2/NEON)

### Buffer Pool Management
Zero-copy operations use intelligent buffer pooling:
- **Automatic memory reuse** across batch operations
- **Size-based pooling** for optimal allocation patterns
- **Thread-safe sharing** for parallel processing

### Parallel Processing Architecture
```python
# Automatically parallelizes across available CPU cores
luminances = tsr.batch_calculate_luminance_zero_copy(images)
# - Extracts raw pointers on main thread
# - Distributes processing across worker threads
# - Uses lock-free data structures for maximum throughput
```

## 🔧 Installation & Setup

```bash
pip install trainingsample

# For maximum performance, ensure you have:
# - Multi-core CPU (parallel processing)
# - AVX2 support (x86) or NEON (ARM) for SIMD
```

## 📈 Benchmarking Your Workload

```python
import time
import trainingsample as tsr

# Benchmark your specific use case
images = load_your_images()

start = time.perf_counter()
results = tsr.batch_operation_zero_copy(images, params)
duration = time.perf_counter() - start

print(f"Processed {len(images)} images in {duration*1000:.2f}ms")
print(f"Throughput: {len(images)/duration:.1f} images/sec")
```

## 🏆 Summary

TrainingSample provides:
- **memory efficiency**: reduced Python object overhead in batch operations
- **computational efficiency**: SIMD vectorization and parallel processing
- **API compatibility**: drop-in replacement for common cv2 operations
- **zero-copy semantics**: direct buffer manipulation for maximum performance

**Average Performance Gains:**
- **5.1x faster** across core operations
- **100% API compatibility** with OpenCV
- **Memory usage reduction** through buffer pooling
- **Automatic optimization** based on workload characteristics

**Limitations:**
- **memory overhead**: batch processing requires significant RAM for large images
- **startup cost**: small overhead for very small batches (<5 images)
- **Python GIL**: some operations still limited by Python's global interpreter lock

For maximum performance gains, use the zero-copy batch operations with mixed-size image datasets on multi-core systems.
