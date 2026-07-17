# Python compatibility helpers

TrainingSample 0.3.0 exposes a small set of OpenCV-style operations. Exported
names and accepted integer codes are listed below.

The module is not a drop-in `cv2` replacement. It does not export `IMREAD_*`,
`COLOR_*`, `VideoCapture`, `VideoWriter`, or `CascadeClassifier` under those
names.

## Exported functions

| Function | Signature |
|---|---|
| Decode | `imdecode_py(buf, flags)` |
| Color conversion | `cvt_color_py(src, code)` |
| Canny | `canny_py(image, threshold1, threshold2)` |
| Compatibility resize | `resize_py(src, dsize, interpolation=None)` |
| OpenCV bilinear resize | `resize_bilinear_opencv(image, target_width, target_height)` |
| OpenCV Lanczos resize | `resize_lanczos4_opencv(image, target_width, target_height)` |
| FourCC | `fourcc_py(c1, c2, c3, c4)` |
| OpenCV data path | `get_opencv_data_path_py()` |

## Decode

```python
with open("image.jpg", "rb") as file:
    encoded = file.read()

rgb = tsr.imdecode_py(encoded, 1)
gray_rgb = tsr.imdecode_py(encoded, 0)
unchanged = tsr.imdecode_py(encoded, -1)
```

| Flag | Meaning |
|---:|---|
| `-1` | unchanged channel count where supported |
| `0` | grayscale replicated into `(height, width, 3)` |
| `1` | RGB |

Decode uses the Rust `image` crate. Cargo explicitly enables JPEG, PNG, and
WebP; the crate's default features are not disabled. Python `bytes` input is
retained without cloning while the GIL is released; `bytearray` input is copied
by PyO3 before decode.

## Color conversion

```python
gray = tsr.cvt_color_py(rgb, 7)
hsv = tsr.cvt_color_py(rgb, 41)
```

| Code | Conversion |
|---:|---|
| `4` | BGR to RGB |
| `5` | RGB to BGR |
| `7` | RGB to grayscale |
| `8` | grayscale to RGB |
| `41` | RGB to HSV |
| `55` | HSV to RGB |

## Resize

```python
resized = tsr.resize_py(rgb, (224, 224), tsr.INTER_LINEAR)
batch = tsr.batch_resize_images([rgb], [(224, 224)])
```

| Export | Value |
|---|---:|
| `INTER_NEAREST` | `0` |
| `INTER_LINEAR` | `1` |
| `INTER_CUBIC` | `2` |
| `INTER_LANCZOS4` | `4` |

`batch_resize_images`, `resize_bilinear_opencv`, and
`resize_lanczos4_opencv` use OpenCV. `resize_py` uses the Rust compatibility
implementation in `cv_compat`.

## Canny

```python
edges = tsr.canny_py(rgb, 50.0, 150.0)
```

`canny_py` uses `imageproc` and returns a three-dimensional `uint8` NumPy
array.

## Video classes

| Class | Constructor |
|---|---|
| `PyVideoCapture` | `PyVideoCapture(filename)` |
| `PyVideoWriter` | `PyVideoWriter(filename, fourcc_str, fps, frame_size)` |

```python
capture = tsr.PyVideoCapture("input.mp4")
opened = capture.is_opened()
ok, frame = capture.read()
capture.release()

fourcc = tsr.fourcc_py("M", "J", "P", "G")
writer = tsr.PyVideoWriter("output.avi", fourcc, 30.0, (640, 480))
if ok and frame is not None:
    writer.write(frame)
writer.release()
```

`PyVideoCapture.from_bytes(source, suffix=".mp4")` accepts bytes-like or
file-like input and stores it in a temporary file for the lifetime of the
capture object.

Supported `PyVideoCapture.get` property codes:

| Code | Property |
|---:|---|
| `3` | frame width |
| `4` | frame height |
| `5` | frames per second |
| `7` | frame count |

Other property codes return `0.0`.

## Cascade class

```python
classifier = tsr.PyCascadeClassifier("cascade.xml")
detections = classifier.detect_multi_scale(image)
```

| Method | Signature |
|---|---|
| `empty` | `empty()` |
| `detect_multi_scale` | `detect_multi_scale(image, scale_factor=None, min_neighbors=None)` |

## Batch and specialized APIs

| Function | Input requirement | Return |
|---|---|---|
| `batch_crop_images` | ndarray image | list of owned arrays |
| `batch_center_crop_images` | ndarray image | list of owned arrays |
| `batch_random_crop_images` | ndarray image | list of owned arrays |
| `batch_resize_images` | C-contiguous RGB | list of owned arrays |
| `batch_resize_videos` | contiguous RGB frames | list of owned arrays |
| `batch_calculate_luminance` | ndarray image | list of floats |
| `batch_crop_images_zero_copy` | C-contiguous | list of owned arrays |
| `batch_center_crop_images_zero_copy` | C-contiguous | list of owned arrays |
| `batch_resize_images_zero_copy` | C-contiguous RGB | array for single input; list for batch input |
| `batch_resize_images_iterator` | C-contiguous RGB | `ResizeIterator` |

All crop boxes use `(x, y, width, height)`. All target sizes use
`(width, height)`.

## Exported class names

```text
PyBatchProcessor
PyCascadeClassifier
PyTrueBatchProcessor
PyVideoCapture
PyVideoWriter
ResizeIterator
```
