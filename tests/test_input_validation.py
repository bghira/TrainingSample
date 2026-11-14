"""Invalid inputs must fail before raw buffer access or large allocations."""

import numpy as np
import pytest
import trainingsample as ts


@pytest.mark.parametrize("name", ["batch_crop_images", "batch_random_crop_images"])
@pytest.mark.parametrize("image_count,argument_count", [(1, 0), (0, 1), (1, 2)])
def test_crop_batch_lengths(name, image_count, argument_count):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    argument = (0, 0, 2, 2) if name == "batch_crop_images" else (2, 2)
    with pytest.raises(ValueError, match="length mismatch"):
        getattr(ts, name)([image] * image_count, [argument] * argument_count)


@pytest.mark.parametrize(
    "box", [(2**64 - 1, 0, 2, 2), (0, 2**64 - 1, 2, 2), (0, 0, 0, 2), (0, 0, 2, 0)]
)
def test_zero_copy_crop_rejects_invalid_bounds(box):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises((ValueError, OverflowError)):
        ts.batch_crop_images_zero_copy([image], [box])
    # A rejected input must not poison global state for subsequent valid calls.
    np.testing.assert_array_equal(
        ts.batch_crop_images_zero_copy([image], [(0, 0, 2, 2)])[0], image[:2, :2]
    )


@pytest.mark.parametrize("size", [(0, 2), (2, 0)])
def test_zero_copy_center_crop_rejects_empty_size(size):
    with pytest.raises(ValueError):
        ts.batch_center_crop_images_zero_copy(
            [np.zeros((4, 4, 3), dtype=np.uint8)], [size]
        )


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("channels", [0, 1, 2, 4])
@pytest.mark.parametrize(
    "api", ["batch_resize_images_zero_copy", "batch_resize_images_iterator"]
)
def test_strict_resize_rejects_non_rgb(api, channels, batch_size):
    image = np.zeros((4, 4, channels), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        getattr(ts, api)([image] * batch_size, [(2, 2)] * batch_size)


@pytest.mark.parametrize("mode", ["single", "batch", "iterator"])
@pytest.mark.parametrize(
    "size", [(0, 2), (2, 0), (2**31, 2), (2, 2**32), (2**64 - 1, 2)]
)
def test_strict_resize_rejects_invalid_dimensions(mode, size):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises((ValueError, OverflowError)):
        if mode == "single":
            ts.batch_resize_images_zero_copy(image, size)
        elif mode == "batch":
            ts.batch_resize_images_zero_copy([image], [size])
        else:
            ts.batch_resize_images_iterator([image], [size])


@pytest.mark.parametrize("channels", [0, 1, 2, 4])
def test_rgb_to_rgba_rejects_invalid_channels(channels):
    with pytest.raises(ValueError, match="3-channel"):
        ts.rgb_to_rgba_optimized(np.zeros((2, 2, channels), dtype=np.uint8), 255)
