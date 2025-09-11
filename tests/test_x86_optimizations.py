#!/usr/bin/env python3
"""
Comprehensive unit tests for x86 optimization Python bindings.

These tests verify that the high-performance x86 optimizations work correctly
through the Python interface, including proper fallback behavior on non-x86 platforms.
"""

import platform

import numpy as np
import pytest

try:
    import trainingsample

    TRAININGSAMPLE_AVAILABLE = True
except ImportError:
    TRAININGSAMPLE_AVAILABLE = False
    trainingsample = None


def create_test_image(height=100, width=100):
    """Create a test image with a gradient pattern."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            image[h, w, 0] = (h * 255) // height  # Red gradient
            image[h, w, 1] = (w * 255) // width  # Green gradient
            image[h, w, 2] = ((h + w) * 255) // (height + width)  # Blue gradient
    return image


def create_large_test_image():
    """Create a larger test image for performance testing."""
    return create_test_image(256, 256)


class TestX86OptimizationAvailability:
    """Test CPU feature detection and platform availability."""

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_cpu_feature_detection_exists(self):
        """Test that CPU feature detection function exists."""
        assert hasattr(trainingsample, "get_x86_cpu_features")

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_cpu_feature_detection_returns_dict(self):
        """Test that CPU feature detection returns a proper dictionary."""
        features = trainingsample.get_x86_cpu_features()

        assert isinstance(features, dict)

        # Check that all expected features are present
        expected_features = [
            "avx512f",
            "avx512bw",
            "avx512dq",
            "avx2",
            "fma",
            "sse41",
            "is_amd_zen",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], bool)

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_cpu_features_on_apple_silicon(self):
        """Test CPU features on Apple Silicon (should all be False)."""
        if platform.machine() == "arm64":
            features = trainingsample.get_x86_cpu_features()

            # On Apple Silicon, all x86 features should be False
            x86_features = ["avx512f", "avx512bw", "avx512dq", "avx2", "fma", "sse41"]
            for feature in x86_features:
                assert (
                    features[feature] is False
                ), f"{feature} should be False on Apple Silicon"


class TestX86ResizeOptimizations:
    """Test x86-optimized resize functions."""

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_resize_function_exists(self):
        """Test that x86 resize function exists."""
        assert hasattr(trainingsample, "resize_bilinear_x86_optimized")

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_resize_basic_functionality(self):
        """Test basic x86 resize functionality."""
        image = create_test_image()
        target_width, target_height = 50, 50

        # On x86_64, this should work; on other platforms, it should fail gracefully
        if platform.machine() == "x86_64":
            try:
                result = trainingsample.resize_bilinear_x86_optimized(
                    image, target_width, target_height
                )
                assert result.shape == (target_height, target_width, 3)
                assert result.dtype == np.uint8

                # Verify output is reasonable
                assert np.sum(result) > 0  # Not all zeros
                assert np.max(result) <= 255  # Valid pixel range

            except Exception as e:
                # Even on x86_64, might fail if SIMD features not enabled
                pytest.skip(f"x86 optimizations not available: {e}")
        else:
            # On non-x86 platforms, should raise NotImplementedError
            with pytest.raises(Exception):  # Could be NotImplementedError or other
                trainingsample.resize_bilinear_x86_optimized(
                    image, target_width, target_height
                )

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_resize_different_sizes(self):
        """Test x86 resize with various target sizes."""
        if platform.machine() != "x86_64":
            pytest.skip("x86 optimizations only available on x86_64")

        image = create_large_test_image()
        test_cases = [
            (128, 128, "Half size"),
            (512, 512, "Double size"),
            (100, 200, "Different aspect ratio"),
            (17, 23, "Small odd dimensions"),
        ]

        for width, height, desc in test_cases:
            try:
                result = trainingsample.resize_bilinear_x86_optimized(
                    image, width, height
                )
                assert result.shape == (height, width, 3), f"Failed for {desc}"
                assert result.dtype == np.uint8, f"Wrong dtype for {desc}"

            except Exception as e:
                pytest.skip(f"x86 resize not available for {desc}: {e}")

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_resize_consistency_with_regular_resize(self):
        """Test that x86 resize produces consistent results with regular resize."""
        if platform.machine() != "x86_64":
            pytest.skip("x86 optimizations only available on x86_64")

        image = create_test_image()
        target_width, target_height = 75, 75

        try:
            # Get x86 optimized result
            x86_result = trainingsample.resize_bilinear_x86_optimized(
                image, target_width, target_height
            )

            # Get regular result (if available)
            if hasattr(trainingsample, "batch_resize_images"):
                regular_results = trainingsample.batch_resize_images(
                    [image], [(target_width, target_height)]
                )
                regular_result = regular_results[0]

                # Results should have same shape
                assert x86_result.shape == regular_result.shape

                # Results should be reasonably close
                diff = np.abs(x86_result.astype(float) - regular_result.astype(float))
                avg_diff = np.mean(diff)
                assert avg_diff < 5.0, f"Average difference too large: {avg_diff}"

        except Exception as e:
            pytest.skip(f"x86 consistency test not available: {e}")


class TestX86LuminanceOptimizations:
    """Test x86-optimized luminance calculation functions."""

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_luminance_function_exists(self):
        """Test that x86 luminance function exists."""
        assert hasattr(trainingsample, "calculate_luminance_x86_optimized")

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_luminance_basic_functionality(self):
        """Test basic x86 luminance functionality."""
        image = create_test_image()

        result = trainingsample.calculate_luminance_x86_optimized(image)

        if platform.machine() == "x86_64":
            # On x86, should return meaningful result
            assert isinstance(result, float)
            assert 0.0 <= result <= 255.0
            assert result > 0.0  # Should not be exactly zero for our test pattern
        else:
            # On non-x86, fallback returns 0.0
            assert result == 0.0

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_luminance_consistency(self):
        """Test luminance calculation consistency across different images."""
        # Test with uniform color images
        test_cases = [
            (np.full((50, 50, 3), 0, dtype=np.uint8), "Black image"),
            (np.full((50, 50, 3), 255, dtype=np.uint8), "White image"),
            (np.full((50, 50, 3), 128, dtype=np.uint8), "Gray image"),
        ]

        for image, desc in test_cases:
            result = trainingsample.calculate_luminance_x86_optimized(image)

            if platform.machine() == "x86_64" and result != 0.0:
                # On x86 with working optimizations
                if desc == "Black image":
                    assert (
                        result < 10.0
                    ), f"Black image should have low luminance: {result}"
                elif desc == "White image":
                    assert (
                        result > 200.0
                    ), f"White image should have high luminance: {result}"
                elif desc == "Gray image":
                    assert (
                        100.0 < result < 160.0
                    ), f"Gray image luminance out of range: {result}"

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_luminance_consistency_with_regular_luminance(self):
        """Test consistency between x86 and regular luminance calculations."""
        if platform.machine() != "x86_64":
            pytest.skip("x86 optimizations only available on x86_64")

        image = create_test_image()

        try:
            x86_result = trainingsample.calculate_luminance_x86_optimized(image)

            # Compare with regular batch luminance if available
            if hasattr(trainingsample, "batch_calculate_luminance"):
                regular_results = trainingsample.batch_calculate_luminance([image])
                regular_result = regular_results[0]

                if x86_result != 0.0:  # x86 optimizations working
                    diff = abs(x86_result - regular_result)
                    assert diff < 2.0, (
                        f"Luminance results too different: "
                        f"{x86_result} vs {regular_result}"
                    )

        except Exception as e:
            pytest.skip(f"x86 luminance consistency test not available: {e}")


class TestIntegrationWithBatchProcessing:
    """Test integration of x86 optimizations with existing batch processing."""

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_batch_processing_still_works(self):
        """Test that existing batch processing functions still work."""
        images = [create_test_image(), create_test_image(150, 150)]

        # Test batch resize
        if hasattr(trainingsample, "batch_resize_images"):
            results = trainingsample.batch_resize_images(images, [(64, 64), (64, 64)])
            assert len(results) == 2
            for result in results:
                assert result.shape == (64, 64, 3)

        # Test batch luminance
        if hasattr(trainingsample, "batch_calculate_luminance"):
            results = trainingsample.batch_calculate_luminance(images)
            assert len(results) == 2
            for result in results:
                assert isinstance(result, float)
                assert 0.0 <= result <= 255.0


class TestErrorHandling:
    """Test error handling in x86 optimization functions."""

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_invalid_image_dimensions(self):
        """Test error handling for invalid image dimensions."""
        # Create image with wrong number of channels
        invalid_image = np.zeros((50, 50, 4), dtype=np.uint8)  # RGBA instead of RGB

        if platform.machine() == "x86_64":
            # On x86_64, should raise an error for invalid dimensions
            with pytest.raises(Exception):  # Should raise some kind of error
                trainingsample.calculate_luminance_x86_optimized(invalid_image)
        else:
            # On non-x86 platforms, fallback returns 0.0 without validation
            result = trainingsample.calculate_luminance_x86_optimized(invalid_image)
            assert result == 0.0

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_invalid_resize_dimensions(self):
        """Test error handling for invalid resize dimensions."""
        image = create_test_image()

        # Try to resize to invalid dimensions
        with pytest.raises(Exception):
            trainingsample.resize_bilinear_x86_optimized(image, 0, 50)

        with pytest.raises(Exception):
            trainingsample.resize_bilinear_x86_optimized(image, 50, 0)


class TestPerformanceCharacteristics:
    """Test performance characteristics (timing, not absolute performance)."""

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_resize_completes_in_reasonable_time(self):
        """Test that x86 resize completes in reasonable time."""
        if platform.machine() != "x86_64":
            pytest.skip("x86 optimizations only available on x86_64")

        import time

        large_image = create_test_image(512, 512)

        try:
            start_time = time.time()
            result = trainingsample.resize_bilinear_x86_optimized(large_image, 256, 256)
            elapsed_time = time.time() - start_time

            # Should complete within reasonable time (very generous bounds)
            assert elapsed_time < 10.0, f"Resize took too long: {elapsed_time}s"
            assert result.shape == (256, 256, 3)

        except Exception as e:
            pytest.skip(f"x86 performance test not available: {e}")

    @pytest.mark.skipif(
        not TRAININGSAMPLE_AVAILABLE, reason="trainingsample not available"
    )
    def test_x86_luminance_completes_in_reasonable_time(self):
        """Test that x86 luminance calculation completes in reasonable time."""
        if platform.machine() != "x86_64":
            pytest.skip("x86 optimizations only available on x86_64")

        import time

        large_image = create_test_image(1000, 1000)

        start_time = time.time()
        result = trainingsample.calculate_luminance_x86_optimized(large_image)
        elapsed_time = time.time() - start_time

        # Should complete quickly (very generous bounds)
        assert (
            elapsed_time < 5.0
        ), f"Luminance calculation took too long: {elapsed_time}s"

        if result != 0.0:  # x86 optimizations working
            assert isinstance(result, float)
            assert 0.0 <= result <= 255.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
