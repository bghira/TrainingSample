"""Regressions for image layout, batch validation, and color conversion."""

import io

import numpy as np
import pytest
import trainingsample as tsr
from PIL import Image


class TestBatchRegressions:
    @pytest.mark.parametrize("width", [1, 7, 8, 9, 15, 16, 17])
    @pytest.mark.parametrize("reversed_view", [False, True])
    def test_rgba_conversion_handles_vector_tails(self, width, reversed_view):
        image = np.arange(4 * width * 3, dtype=np.uint8).reshape(4, width, 3)
        if reversed_view:
            image = image[::-1, ::-1, ::-1]
        actual, _ = tsr.rgb_to_rgba_optimized(image, 173)
        expected = np.concatenate(
            [image, np.full((4, width, 1), 173, dtype=np.uint8)], axis=2
        )
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "layout", ["stepped", "reversed", "transposed", "broadcast"]
    )
    def test_true_batch_views_match_contiguous_copies(self, layout):
        image = np.arange(4 * 17 * 3, dtype=np.uint8).reshape(4, 17, 3)
        views = {
            "stepped": image[::2, ::2],
            "reversed": image[::-1, ::-1, ::-1],
            "transposed": image.transpose(1, 0, 2),
            "broadcast": np.broadcast_to(image[:1, :1], (4, 17, 3)),
        }
        view = views[layout]
        copy = np.ascontiguousarray(view)
        processor = tsr.PyTrueBatchProcessor()
        for code in [4, 5, 7]:
            actual = processor.true_batch_cvt_color([view], code)[0]
            expected = processor.true_batch_cvt_color([copy], code)[0]
            np.testing.assert_array_equal(actual, expected)
        for stride in [0, 1, 3]:
            assert processor.strided_luminance([view], stride) == (
                processor.strided_luminance([copy], stride)
            )
        for interpolation in [0, 1, 2, 4]:
            actual = processor.true_batch_resize([view], [(5, 3)], interpolation)[0]
            expected = processor.true_batch_resize([copy], [(5, 3)], interpolation)[0]
            np.testing.assert_array_equal(actual, expected)

    def test_true_batch_gray_to_rgb(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)[::-1]
        actual = tsr.PyTrueBatchProcessor().true_batch_cvt_color([gray], 8)[0]
        np.testing.assert_array_equal(actual, np.repeat(gray, 3, axis=2))

    @pytest.mark.parametrize("parallel", [False, True])
    def test_zero_chunk_size_and_empty_pipeline(self, parallel):
        processor = tsr.PyBatchProcessor.with_config(parallel, 0)
        assert processor.batch_preprocess_pipeline([], []) == []
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        actual = processor.batch_cvt_color([image], 4)[0]
        np.testing.assert_array_equal(actual, image[:, :, ::-1])
        assert processor.batch_canny([image], 50.0, 100.0)[0].shape == (4, 5, 1)
        encoded = io.BytesIO()
        Image.fromarray(image).save(encoded, format="PNG")
        assert processor.batch_preprocess_pipeline([encoded.getvalue()], [(3, 7)])[
            0
        ].shape == (7, 3, 3)

    @pytest.mark.parametrize("interpolation", [0, 1, 2, 4])
    def test_resize_returns_exact_dimensions(self, interpolation):
        image = np.zeros((4, 17, 3), dtype=np.uint8)
        assert tsr.resize_py(image, (3, 8), interpolation).shape == (8, 3, 3)
        assert tsr.PyBatchProcessor().batch_resize([image], [(3, 8)], interpolation)[
            0
        ].shape == (8, 3, 3)

    def test_true_batch_invalid_sizes(self):
        processor = tsr.PyTrueBatchProcessor()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="mismatch"):
            processor.true_batch_resize([], [(2, 2)])
        for size in [(0, 2), (2, 0), (2**31, 2), (2, 2**31)]:
            with pytest.raises(RuntimeError, match="Target dimensions"):
                processor.true_batch_resize([image], [size])
