"""Paired public-API benchmarks. Input generation and parity checks are untimed."""

from importlib.metadata import version

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
tsr = pytest.importorskip("trainingsample")
pytest.importorskip("pytest_benchmark")


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(42)
    images = [
        rng.integers(256, size=(h, w, 3), dtype=np.uint8)
        for h, w in [(480, 640), (1024, 768), (256, 256), (128, 512)]
    ]
    rgba = rng.integers(256, size=(1024, 1024, 4), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    video = rng.integers(256, size=(30, 224, 224, 3), dtype=np.uint8)
    return images, rgb, rgba, video


@pytest.fixture(scope="module")
def cases(inputs):
    images, rgb, rgba, video = inputs
    target = (224, 224)
    sizes = [target] * len(images)
    boxes = [(10, 10, 100, 100)] * len(images)

    def resize():
        return [
            cv2.resize(img, target, interpolation=cv2.INTER_LINEAR) for img in images
        ]

    def luminance(batch):
        return [cv2.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0] for img in batch]

    def resize_video():
        output = np.empty((len(video), 112, 112, 3), dtype=np.uint8)
        for frame, dst in zip(video, output):
            cv2.resize(frame, (112, 112), dst=dst, interpolation=cv2.INTER_LINEAR)
        return [output]

    return {
        "crop": (
            lambda: tsr.batch_crop_images(images, boxes),
            lambda: [img[10:110, 10:110].copy() for img in images],
        ),
        "resize": (lambda: tsr.batch_resize_images(images, sizes), resize),
        "luminance": (
            lambda: tsr.batch_calculate_luminance(images),
            lambda: luminance(images),
        ),
        "pipeline": (
            lambda: tsr.batch_calculate_luminance(
                tsr.batch_resize_images(images, sizes)
            ),
            lambda: luminance(resize()),
        ),
        "video": (lambda: tsr.batch_resize_videos([video], [(112, 112)]), resize_video),
        "rgb_to_rgba": (
            lambda: tsr.rgb_to_rgba_optimized(rgb, 255)[0],
            lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA),
        ),
        "rgba_to_rgb": (
            lambda: tsr.rgba_to_rgb_optimized(rgba)[0],
            lambda: cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB),
        ),
    }


@pytest.mark.parametrize("engine", ["tsr", "reference"])
@pytest.mark.parametrize(
    "operation",
    ["crop", "resize", "luminance", "pipeline", "video", "rgb_to_rgba", "rgba_to_rgb"],
)
def test_competitive(benchmark, cases, operation, engine):
    native, reference = cases[operation]
    actual, expected = native(), reference()
    if operation in {"pipeline", "luminance"}:
        # OpenCV rounds each pixel to uint8 before averaging.
        tolerance = 0.5 if operation == "pipeline" else 0.05
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    elif operation in {"resize", "video"}:
        for a, b in zip(actual, expected, strict=True):
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            # Linear interpolation can round differently across OpenCV builds.
            error = np.abs(a.astype(np.int16) - b)
            assert error.max() <= 2 and error.mean() < 1
    else:
        np.testing.assert_array_equal(actual, expected)
    del actual, expected
    benchmark.extra_info.update(
        trainingsample=version("trainingsample"),
        numpy=np.__version__,
        cv2=cv2.__version__,
        cv2_threads=cv2.getNumThreads(),
    )
    benchmark.group = operation
    benchmark.pedantic(
        native if engine == "tsr" else reference,
        rounds=21,
        iterations=10,
        warmup_rounds=3,
    )
