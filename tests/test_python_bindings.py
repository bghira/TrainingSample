"""Tests for Python bindings of training_sample_rust."""

import importlib.util

import numpy as np
import pytest

try:
    import trainingsample as tsr

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

HAS_BENCHMARK = importlib.util.find_spec("pytest_benchmark") is not None

pytestmark = pytest.mark.skipif(
    not HAS_BINDINGS, reason="Python bindings not available"
)


@pytest.fixture
def sample_images():
    """Create sample RGB images for testing."""
    return [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]


@pytest.fixture
def sample_video():
    """Create sample RGB video for testing."""
    return np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)


class TestImageOperations:
    """Test image processing operations."""

    def test_batch_crop_images(self, sample_images):
        """Test batch cropping with custom coordinates."""
        crop_boxes = [(10, 10, 50, 50), (20, 20, 40, 40), (5, 5, 60, 60)]
        results = tsr.batch_crop_images(sample_images, crop_boxes)

        assert len(results) == len(sample_images)
        assert results[0].shape == (50, 50, 3)
        assert results[1].shape == (40, 40, 3)
        assert results[2].shape == (60, 60, 3)

    def test_batch_center_crop_images(self, sample_images):
        """Test batch center cropping."""
        target_sizes = [(50, 50), (60, 60), (40, 40)]
        results = tsr.batch_center_crop_images(sample_images, target_sizes)

        assert len(results) == len(sample_images)
        assert results[0].shape == (50, 50, 3)
        assert results[1].shape == (60, 60, 3)
        assert results[2].shape == (40, 40, 3)

    def test_batch_random_crop_images(self, sample_images):
        """Test batch random cropping."""
        target_sizes = [(50, 50)] * len(sample_images)
        results = tsr.batch_random_crop_images(sample_images, target_sizes)

        assert len(results) == len(sample_images)
        for result in results:
            assert result.shape == (50, 50, 3)

    def test_batch_resize_images(self, sample_images):
        """Test batch image resizing."""
        target_sizes = [(64, 64), (128, 128), (32, 32)]
        results = tsr.batch_resize_images(sample_images, target_sizes)

        assert len(results) == len(sample_images)
        assert results[0].shape == (64, 64, 3)
        assert results[1].shape == (128, 128, 3)
        assert results[2].shape == (32, 32, 3)

    def test_batch_calculate_luminance(self, sample_images):
        """Test batch luminance calculation."""
        results = tsr.batch_calculate_luminance(sample_images)

        assert len(results) == len(sample_images)
        for luminance in results:
            assert isinstance(luminance, float)
            assert 0.0 <= luminance <= 255.0


class TestVideoOperations:
    """Test video processing operations."""

    def test_batch_resize_videos(self, sample_video):
        """Test batch video resizing."""
        videos = [sample_video]
        target_sizes = [(64, 64)]
        results = tsr.batch_resize_videos(videos, target_sizes)

        assert len(results) == 1
        assert results[0].shape == (10, 64, 64, 3)


class TestZeroCopyBindings:
    """Test zero-copy binding behavior and safety checks."""

    def test_zero_copy_luminance_matches_standard_path(self):
        """Zero-copy luminance should match the standard implementation."""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        expected = tsr.batch_calculate_luminance([image])[0]
        actual = tsr.batch_calculate_luminance_zero_copy([image])[0]

        assert actual == pytest.approx(expected, abs=1e-6)

    def test_zero_copy_luminance_accepts_non_contiguous_views(self):
        """Zero-copy luminance should handle strided arrays safely."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        strided = image[:, ::2, :]

        expected = tsr.batch_calculate_luminance([strided])[0]
        actual = tsr.batch_calculate_luminance_zero_copy([strided])[0]

        assert actual == pytest.approx(expected, abs=1e-6)

    def test_zero_copy_crop_rejects_non_contiguous_array(self):
        """Unsafe zero-copy crop path should reject non-contiguous input."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        strided = image[:, ::2, :]

        with pytest.raises(ValueError, match="C-contiguous"):
            tsr.batch_crop_images_zero_copy([strided], [(0, 0, 16, 16)])

    def test_zero_copy_center_crop_rejects_non_contiguous_array(self):
        """Unsafe zero-copy center crop path should reject non-contiguous input."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        strided = image[::2, :, :]

        with pytest.raises(ValueError, match="C-contiguous"):
            tsr.batch_center_crop_images_zero_copy([strided], [(16, 16)])

    def test_zero_copy_resize_rejects_non_contiguous_array(self):
        """Unsafe zero-copy resize path should reject non-contiguous input."""
        if not hasattr(tsr, "batch_resize_images_zero_copy"):
            pytest.skip("OpenCV zero-copy resize bindings not available")

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        strided = image[:, ::2, :]

        with pytest.raises(ValueError, match="C-contiguous"):
            tsr.batch_resize_images_zero_copy(strided, (16, 16))

    def test_zero_copy_resize_iterator_rejects_non_contiguous_array(self):
        """Unsafe zero-copy iterator resize path should reject non-contiguous input."""
        if not hasattr(tsr, "batch_resize_images_iterator"):
            pytest.skip("OpenCV resize iterator bindings not available")

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        strided = image[:, ::2, :]

        with pytest.raises(ValueError, match="C-contiguous"):
            tsr.batch_resize_images_iterator([strided], [(16, 16)])


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_crop_coordinates(self, sample_images):
        """Test cropping with invalid coordinates."""
        # Crop box exceeds image bounds
        crop_boxes = [(90, 90, 50, 50)]  # 90+50 > 100

        with pytest.raises(ValueError, match="Cropping failed"):
            tsr.batch_crop_images([sample_images[0]], crop_boxes)

    def test_empty_input_lists(self):
        """Test operations with empty input lists."""
        assert tsr.batch_crop_images([], []) == []
        assert tsr.batch_resize_images([], []) == []
        assert tsr.batch_calculate_luminance([]) == []

    def test_mismatched_input_lengths(self, sample_images):
        """Test operations with mismatched input lengths."""
        crop_boxes = [(10, 10, 50, 50)]  # Only one box for multiple images

        # This should fail or handle gracefully
        # The exact behavior depends on implementation
        try:
            results = tsr.batch_crop_images(sample_images, crop_boxes)
            # If it doesn't raise, check it processes correctly
            assert len(results) == 1
        except (RuntimeError, ValueError):
            # Expected for mismatched lengths
            pass


@pytest.mark.benchmark
class TestPerformance:
    """Benchmark tests for performance verification."""

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
    def test_luminance_performance(self, benchmark, sample_images):
        """Benchmark luminance calculation."""
        # Create larger batch for meaningful benchmark
        large_batch = sample_images * 10

        result = benchmark(tsr.batch_calculate_luminance, large_batch)
        assert len(result) == len(large_batch)

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
    def test_resize_performance(self, benchmark, sample_images):
        """Benchmark image resizing."""
        target_sizes = [(128, 128)] * len(sample_images)

        results = benchmark(tsr.batch_resize_images, sample_images, target_sizes)
        assert len(results) == len(sample_images)

    def test_luminance_performance_simple(self, sample_images):
        """Simple performance test without benchmark fixture."""
        import time

        # Create larger batch for meaningful timing
        large_batch = sample_images * 20

        start_time = time.time()
        result = tsr.batch_calculate_luminance(large_batch)
        end_time = time.time()

        assert len(result) == len(large_batch)
        # Should complete in reasonable time (less than 1 second for small images)
        assert (end_time - start_time) < 1.0

    def test_resize_performance_simple(self, sample_images):
        """Simple performance test without benchmark fixture."""
        import time

        target_sizes = [(128, 128)] * len(sample_images)

        start_time = time.time()
        results = tsr.batch_resize_images(sample_images, target_sizes)
        end_time = time.time()

        assert len(results) == len(sample_images)
        # Should complete in reasonable time
        assert (end_time - start_time) < 2.0


@pytest.mark.integration
class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_crop_then_resize_pipeline(self, sample_images):
        """Test a pipeline of crop followed by resize."""
        # First crop
        crop_boxes = [(10, 10, 80, 80)] * len(sample_images)
        cropped = tsr.batch_crop_images(sample_images, crop_boxes)

        # Then resize
        target_sizes = [(64, 64)] * len(cropped)
        resized = tsr.batch_resize_images(cropped, target_sizes)

        assert len(resized) == len(sample_images)
        for result in resized:
            assert result.shape == (64, 64, 3)

    def test_full_processing_pipeline(self, sample_images):
        """Test a complete processing pipeline."""
        # Center crop
        target_sizes = [(80, 80)] * len(sample_images)
        cropped = tsr.batch_center_crop_images(sample_images, target_sizes)

        # Resize
        resize_sizes = [(64, 64)] * len(cropped)
        resized = tsr.batch_resize_images(cropped, resize_sizes)

        # Calculate luminance
        luminances = tsr.batch_calculate_luminance(resized)

        assert len(resized) == len(sample_images)
        assert len(luminances) == len(sample_images)
        for result in resized:
            assert result.shape == (64, 64, 3)
        for luminance in luminances:
            assert isinstance(luminance, float)
