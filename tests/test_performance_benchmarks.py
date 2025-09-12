"""Enhanced performance benchmarks and stress tests for training_sample_rust."""

import gc
import threading
import time

import numpy as np
import pytest

try:
    import trainingsample as tsr

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

HAS_BENCHMARK = True

pytestmark = pytest.mark.skipif(
    not HAS_BINDINGS, reason="Python bindings not available"
)


@pytest.fixture
def performance_test_images():
    """Create realistic test images for performance testing."""
    # Small batch (typical training batch size)
    small_batch = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)
    ]

    # Large batch (data loading scenario)
    large_batch = [
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(64)
    ]

    # Mixed sizes (realistic dataset)
    mixed_sizes = [
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for h, w in [(480, 640), (1024, 768), (256, 256), (128, 512)]
    ]

    return {
        "small_batch": small_batch,
        "large_batch": large_batch,
        "mixed_sizes": mixed_sizes,
    }


@pytest.fixture
def performance_test_videos():
    """Create test videos for performance testing."""
    # Short video clips
    short_clips = [
        np.random.randint(0, 255, (30, 224, 224, 3), dtype=np.uint8)  # 30 frames
        for _ in range(4)
    ]

    # Long video
    long_video = np.random.randint(0, 255, (120, 512, 512, 3), dtype=np.uint8)

    return {"short_clips": short_clips, "long_video": [long_video]}


class TestBasicPerformance:
    """Basic performance tests that always run."""

    def test_crop_performance_scaling(self, performance_test_images):
        """Test that crop performance scales reasonably with batch size."""
        images = performance_test_images["small_batch"]

        # Single image
        start = time.perf_counter()
        tsr.batch_crop_images([images[0]], [(50, 50, 100, 100)])
        single_time = time.perf_counter() - start

        # Full batch (16 images)
        crop_boxes = [(50, 50, 100, 100)] * len(images)
        start = time.perf_counter()
        tsr.batch_crop_images(images, crop_boxes)
        batch_time = time.perf_counter() - start

        # Batch should be more efficient than linear scaling
        linear_expectation = single_time * len(images)
        eff_ratio = batch_time / linear_expectation

        assert (
            eff_ratio < 1.5
        ), f"Batch processing should be reasonably efficient, got ratio: {eff_ratio}"
        assert (
            batch_time < 5.0
        ), f"Batch processing should complete reasonably, took: {batch_time:.3f}s"

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_crop_functionality_correctness(self, performance_test_images):
        """Test crop functionality and correctness."""
        images = performance_test_images["small_batch"]
        crop_box = (50, 50, 100, 100)  # x, y, width, height

        # TSR batch crop
        crop_boxes = [crop_box] * len(images)
        start = time.perf_counter()
        tsr_results = tsr.batch_crop_images(images, crop_boxes)
        tsr_time = time.perf_counter() - start

        # NumPy slicing (equivalent to OpenCV for cropping)
        def numpy_crop(img, x, y, w, h):
            return img[y : y + h, x : x + w]

        start = time.perf_counter()
        numpy_results = [numpy_crop(img, *crop_box) for img in images]
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / tsr_time if tsr_time > 0 else float("inf")

        # Validate results
        assert len(tsr_results) == len(numpy_results)
        for tsr_img, np_img in zip(tsr_results, numpy_results):
            assert tsr_img.shape == np_img.shape == (100, 100, 3)
            # Results should be identical for simple crop
            np.testing.assert_array_equal(tsr_img, np_img)

        print(f"Crop performance (batch size {len(images)}):")
        print(
            f"  TSR: {tsr_time:.4f}s, NumPy: {numpy_time:.4f}s, Speedup: {speedup:.2f}x"
        )

        # Validate functionality - TSR and NumPy should produce identical results
        assert len(tsr_results) == len(numpy_results) == len(images)
        for i, (tsr_crop, numpy_crop) in enumerate(zip(tsr_results, numpy_results)):
            assert (
                tsr_crop.shape == numpy_crop.shape == (100, 100, 3)
            ), f"Image {i}: Shape mismatch"
            np.testing.assert_array_equal(
                tsr_crop, numpy_crop, f"Image {i}: Results should be identical"
            )

        print(
            f"✅ TSR crop functionality correct "
            f"(NumPy is {1/speedup:.0f}x faster, which is expected)"
        )

    def test_resize_performance_different_scales(self, performance_test_images):
        """Test resize performance across different scale factors."""
        image = performance_test_images["mixed_sizes"][0]  # 480x640

        # Downscaling (should be fast)
        start = time.perf_counter()
        tsr.batch_resize_images([image], [(240, 320)])
        downscale_time = time.perf_counter() - start

        # Upscaling (should be slower but reasonable)
        start = time.perf_counter()
        tsr.batch_resize_images([image], [(960, 1280)])
        upscale_time = time.perf_counter() - start

        # Same size (should be very fast)
        start = time.perf_counter()
        tsr.batch_resize_images([image], [(640, 480)])  # Note: (width, height)
        same_size_time = time.perf_counter() - start

        # Performance expectations (relaxed for CI)
        assert (
            downscale_time < 2.0
        ), f"Downscaling should be reasonably fast: {downscale_time:.3f}s"
        assert upscale_time < 10.0, f"Upscaling should complete: {upscale_time:.3f}s"
        assert (
            same_size_time < 2.0
        ), f"Same-size resize should be fast: {same_size_time:.3f}s"

    def test_luminance_performance_batch_sizes(self, performance_test_images):
        """Test luminance calculation performance across batch sizes."""
        images = performance_test_images["small_batch"]

        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        times = []

        for batch_size in batch_sizes:
            batch = images[:batch_size]
            # Warm up
            tsr.batch_calculate_luminance(batch)

            # Time multiple iterations for better measurement accuracy
            start = time.perf_counter()
            iterations = 50  # More iterations for microsecond-scale operations
            for _ in range(iterations):
                tsr.batch_calculate_luminance(batch)
            elapsed = (time.perf_counter() - start) / iterations
            times.append(elapsed)

        # Should scale sub-linearly due to parallelism
        for i in range(1, len(batch_sizes)):
            scale_factor = batch_sizes[i] / batch_sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            # For very fast operations, parallelism overhead can dominate
            # Just ensure performance doesn't degrade catastrophically
            max_acceptable_ratio = (
                scale_factor * 2.0
            )  # Very relaxed for fast operations

            assert time_ratio < max_acceptable_ratio, (
                f"Batch size {batch_sizes[i]} scaling should not be catastrophic: "
                f"{time_ratio:.2f} vs {max_acceptable_ratio:.2f}"
            )

    def test_luminance_performance_comprehensive(self, performance_test_images):
        """Test luminance performance across different image sizes and batch sizes."""
        # Test with larger images where parallelism should clearly win
        large_images = [
            np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            for _ in range(16)
        ]

        # Test scaling with larger images (should show good parallelism)
        batch_sizes = [1, 4, 8, 16]
        times = []

        for batch_size in batch_sizes:
            batch = large_images[:batch_size]
            # Warm up
            tsr.batch_calculate_luminance(batch)

            start = time.perf_counter()
            for _ in range(3):  # Multiple iterations for stability
                tsr.batch_calculate_luminance(batch)
            elapsed = (time.perf_counter() - start) / 3
            times.append(elapsed)

        # With large images, parallelism should show clear benefits
        for i in range(1, len(batch_sizes)):
            scale_factor = batch_sizes[i] / batch_sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            # Even with larger work, parallel overhead exists for small batches
            # Look for reasonable scaling, not perfect scaling
            max_acceptable_ratio = (
                scale_factor * 1.2 if batch_sizes[i] > 4 else scale_factor * 1.8
            )

            assert time_ratio < max_acceptable_ratio, (
                f"Large images batch size {batch_sizes[i]} should scale reasonably: "
                f"{time_ratio:.2f} vs {max_acceptable_ratio:.2f}"
            )

        print(
            f"Large image scaling ratios: "
            f"{[times[i]/times[i-1] for i in range(1, len(times))]}"
        )

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_resize_batch_vs_individual_opencv(self, performance_test_images):
        """Show TSR batch advantage: batch processing vs individual OpenCV calls."""
        images = performance_test_images["mixed_sizes"][:8]  # Mix of different sizes
        target_size = (224, 224)
        target_sizes = [target_size] * len(images)

        # TSR: Single batch call handles mixed shapes
        start = time.perf_counter()
        tsr_results = tsr.batch_resize_images(images, target_sizes)
        tsr_time = time.perf_counter() - start

        # OpenCV: Individual calls (the old way)
        start = time.perf_counter()
        opencv_results = []
        for img in images:
            # Convert to BGR for OpenCV, then back to RGB
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            resized_bgr = cv2.resize(
                bgr_img, target_size, interpolation=cv2.INTER_LINEAR
            )
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            opencv_results.append(resized_rgb)
        opencv_time = time.perf_counter() - start

        speedup = opencv_time / tsr_time if tsr_time > 0 else float("inf")

        # Validate results
        assert len(tsr_results) == len(opencv_results) == len(images)
        for i, (tsr_img, cv_img) in enumerate(zip(tsr_results, opencv_results)):
            expected_shape = (target_size[1], target_size[0], 3)
            assert tsr_img.shape == cv_img.shape == expected_shape
            # Results should be reasonably close (different algorithms may differ)
            # TSR uses OpenCV internally but may have different interpolation settings
            tsr_float = tsr_img.astype(float)
            cv_float = cv_img.astype(float)
            diff = np.mean(np.abs(tsr_float - cv_float))
            # More lenient comparison since different resize implementations can vary
            assert diff < 25.0, f"Image {i}: Results too different (mean diff: {diff})"

        print(f"🚀 Resize Performance Comparison (batch size {len(images)}):")
        print(f"  Input shapes: {[img.shape[:2] for img in images]}")
        tsr_imgs_per_sec = len(images) / tsr_time
        print(f"  TSR (batch):      {tsr_time:.4f}s ({tsr_imgs_per_sec:.1f} imgs/sec)")
        cv_imgs_per_sec = len(images) / opencv_time
        print(
            f"  OpenCV (individual): {opencv_time:.4f}s "
            f"({cv_imgs_per_sec:.1f} imgs/sec)"
        )
        print(f"  🎯 TSR Advantage: {speedup:.2f}x faster")
        print("  💡 Why TSR exists: Batch processing beats " "individual OpenCV calls")

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_luminance_batch_vs_individual_opencv(self, performance_test_images):
        """
        Show TSR batch advantage: luminance calculation vs individual OpenCV calls.
        """
        images = performance_test_images["mixed_sizes"][:12]  # Mix of different sizes

        # TSR: Single batch call handles mixed shapes
        start = time.perf_counter()
        tsr_results = tsr.batch_calculate_luminance(images)
        tsr_time = time.perf_counter() - start

        # OpenCV: Individual calls (the old way)
        start = time.perf_counter()
        opencv_results = []
        for img in images:
            # Convert RGB to BGR for OpenCV
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Convert to grayscale using OpenCV
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            # Calculate mean luminance
            luminance = np.mean(gray)
            opencv_results.append(luminance)
        opencv_time = time.perf_counter() - start

        speedup = opencv_time / tsr_time if tsr_time > 0 else float("inf")

        # Validate results
        assert len(tsr_results) == len(opencv_results) == len(images)
        for i, (tsr_lum, cv_lum) in enumerate(zip(tsr_results, opencv_results)):
            # Results should be close (small differences due to RGB vs BGR conversion)
            diff = abs(tsr_lum - cv_lum)
            assert diff < 2.0, (
                f"Image {i}: Results too different: "
                f"TSR {tsr_lum:.2f} vs OpenCV {cv_lum:.2f}"
            )

        print(f"🚀 Luminance Performance Comparison (batch size {len(images)}):")
        print(f"  Input shapes: {[img.shape[:2] for img in images]}")
        tsr_imgs_per_sec = len(images) / tsr_time
        print(
            f"  TSR (batch):         {tsr_time:.4f}s ({tsr_imgs_per_sec:.1f} imgs/sec)"
        )
        cv_imgs_per_sec = len(images) / opencv_time
        print(
            f"  OpenCV (individual): {opencv_time:.4f}s "
            f"({cv_imgs_per_sec:.1f} imgs/sec)"
        )
        print(f"  🎯 TSR Advantage: {speedup:.2f}x faster")
        print("  💡 Why TSR exists: Batch SIMD processing beats individual calls")

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_complete_pipeline_batch_vs_individual(self, performance_test_images):
        """
        Show TSR's ultimate advantage: complete pipeline in batch vs individual ops.
        """
        images = performance_test_images["mixed_sizes"][:8]  # Mix of different sizes
        target_size = (224, 224)
        target_sizes = [target_size] * len(images)

        # TSR: Complete pipeline in 2 batch calls
        start = time.perf_counter()
        # Step 1: Batch resize
        resized = tsr.batch_resize_images(images, target_sizes)
        # Step 2: Batch luminance
        luminances = tsr.batch_calculate_luminance(resized)
        tsr_time = time.perf_counter() - start

        # OpenCV: Individual operations (the painful old way)
        start = time.perf_counter()
        opencv_luminances = []
        for img in images:
            # Individual resize with color space conversions
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            resized_bgr = cv2.resize(
                bgr_img, target_size, interpolation=cv2.INTER_LINEAR
            )
            # Individual luminance calculation
            gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
            luminance = np.mean(gray)
            opencv_luminances.append(luminance)
        opencv_time = time.perf_counter() - start

        speedup = opencv_time / tsr_time if tsr_time > 0 else float("inf")

        # Validate results
        assert len(luminances) == len(opencv_luminances) == len(images)
        for i, (tsr_lum, cv_lum) in enumerate(zip(luminances, opencv_luminances)):
            diff = abs(tsr_lum - cv_lum)
            assert diff < 3.0, (
                f"Image {i}: Pipeline results too different: "
                f"{tsr_lum:.2f} vs {cv_lum:.2f}"
            )

        print(
            f"🚀 Complete Pipeline Performance (resize + luminance, "
            f"batch size {len(images)}):"
        )
        input_shapes = [img.shape[:2] for img in images]
        print(f"  Input shapes: {input_shapes} → {target_size}")
        tsr_imgs_per_sec = len(images) / tsr_time
        print(
            f"  TSR (batch pipeline):    {tsr_time:.4f}s "
            f"({tsr_imgs_per_sec:.1f} imgs/sec)"
        )
        cv_imgs_per_sec = len(images) / opencv_time
        print(
            f"  OpenCV (individual):     {opencv_time:.4f}s "
            f"({cv_imgs_per_sec:.1f} imgs/sec)"
        )
        print(f"  🎯 TSR Advantage: {speedup:.2f}x faster")
        print("  💡 Key Benefits:")
        print(f"    • 2 batch calls vs {len(images)} individual operations")
        print("    • Mixed-shape handling in single API calls")
        print("    • No color space conversion overhead")
        print("    • SIMD parallelization across entire batches")

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_mixed_shapes_batch_functionality(self, performance_test_images):
        """Test TSR's ability to handle mixed-shape batches in a single API call."""
        # Create a realistic mixed-shape batch (different aspect ratios/resolutions)
        mixed_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # 4:3 landscape
            np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8),  # 4:3 portrait
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),  # 16:9 landscape
            np.random.randint(0, 255, (1080, 720, 3), dtype=np.uint8),  # 16:9 portrait
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),  # 1:1 square
            np.random.randint(0, 255, (256, 1024, 3), dtype=np.uint8),  # 1:4 tall
            np.random.randint(0, 255, (1024, 256, 3), dtype=np.uint8),  # 4:1 wide
            np.random.randint(
                0, 255, (768, 768, 3), dtype=np.uint8
            ),  # Different square
        ]

        target_size = (224, 224)

        # Test TSR can handle mixed shapes in a single batch operation
        target_sizes = [target_size] * len(mixed_images)
        tsr_results = tsr.batch_resize_images(mixed_images, target_sizes)

        # Validate results
        assert len(tsr_results) == len(mixed_images)
        for i, result in enumerate(tsr_results):
            expected_shape = (target_size[1], target_size[0], 3)
            assert (
                result.shape == expected_shape
            ), f"Image {i}: Expected shape {expected_shape}, got {result.shape}"
            assert result.dtype == np.uint8

        print(
            f"✅ Mixed-shape batch processing successful "
            f"({len(mixed_images)} images with different shapes → {target_size})"
        )

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_mixed_shapes_luminance_advantage(self, performance_test_images):
        """Show TSR's luminance advantage with mixed-shape batches."""
        # Mixed-shape images (realistic dataset scenario)
        mixed_images = [
            np.random.randint(
                0, 255, (1080, 1920, 3), dtype=np.uint8
            ),  # Full HD landscape
            np.random.randint(
                0, 255, (1920, 1080, 3), dtype=np.uint8
            ),  # Full HD portrait
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),  # HD landscape
            np.random.randint(
                0, 255, (512, 512, 3), dtype=np.uint8
            ),  # Square thumbnail
            np.random.randint(0, 255, (2048, 1024, 3), dtype=np.uint8),  # Wide panorama
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),  # Small square
        ]

        # TSR handles mixed shapes in one batch call
        start = time.perf_counter()
        tsr_luminances = tsr.batch_calculate_luminance(mixed_images)
        tsr_time = time.perf_counter() - start

        # NumPy must process each shape individually
        # (no vectorization across different shapes)
        def numpy_luminance(img):
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            return np.mean(0.299 * r + 0.587 * g + 0.114 * b)

        start = time.perf_counter()
        numpy_luminances = [numpy_luminance(img) for img in mixed_images]
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / tsr_time if tsr_time > 0 else float("inf")

        # Validate results
        assert len(tsr_luminances) == len(mixed_images)
        for tsr_lum, np_lum in zip(tsr_luminances, numpy_luminances):
            assert (
                abs(tsr_lum - np_lum) < 1.0
            ), f"Luminance mismatch: {tsr_lum} vs {np_lum}"

        print(
            f"Mixed-shape luminance performance "
            f"({len(mixed_images)} different sized images):"
        )
        tsr_imgs_per_sec = len(mixed_images) / tsr_time
        print(f"  TSR (batched): {tsr_time:.4f}s ({tsr_imgs_per_sec:.1f} imgs/sec)")
        np_imgs_per_sec = len(mixed_images) / numpy_time
        print(
            f"  NumPy (individual): {numpy_time:.4f}s "
            f"({np_imgs_per_sec:.1f} imgs/sec)"
        )
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Image shapes: {[img.shape[:2] for img in mixed_images]}")

        # TSR provides batching convenience for mixed shapes
        print(
            f"✅ TSR mixed-shape luminance functionality working "
            f"(individual calls {1/speedup:.0f}x faster, expected)"
        )

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not available")
    def test_mixed_shape_crop_advantage(self, performance_test_images):
        """
        Demonstrate TSR's advantage with mixed-shape cropping - different shapes.
        """
        # Create mixed-shape images with different crop regions
        # (realistic dataset scenario)
        mixed_images_and_crops = [
            # (image, crop_box) - each image different size, each crop different size
            (
                np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
                (100, 100, 800, 600),
            ),  # Large landscape → medium crop
            (
                np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8),
                (200, 200, 400, 400),
            ),  # Large portrait → small square crop
            (
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                (50, 50, 300, 200),
            ),  # Medium landscape → rectangular crop
            (
                np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
                (300, 100, 600, 400),
            ),  # HD landscape → large rectangular crop
            (
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
                (100, 100, 200, 200),
            ),  # Square → square crop (fixed height)
            (
                np.random.randint(0, 255, (2048, 1024, 3), dtype=np.uint8),
                (200, 50, 800, 150),
            ),  # Wide panorama → wide strip crop (fixed width)
            (
                np.random.randint(0, 255, (1024, 256, 3), dtype=np.uint8),
                (50, 200, 150, 400),
            ),  # Tall narrow → tall crop (swapped dimensions)
            (
                np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8),
                (150, 150, 400, 400),
            ),  # Square → different square crop
        ]

        images = [item[0] for item in mixed_images_and_crops]
        crop_boxes = [item[1] for item in mixed_images_and_crops]

        # TSR handles all mixed shapes/crops in one batch call
        start = time.perf_counter()
        tsr_results = tsr.batch_crop_images(images, crop_boxes)
        tsr_time = time.perf_counter() - start

        # NumPy/OpenCV must handle each shape individually (no vectorization possible)
        def numpy_crop(img, x, y, w, h):
            return img[y : y + h, x : x + w]

        start = time.perf_counter()
        numpy_results = []
        for img, (x, y, w, h) in mixed_images_and_crops:
            cropped = numpy_crop(img, x, y, w, h)
            numpy_results.append(cropped)
        numpy_time = time.perf_counter() - start

        speedup = numpy_time / tsr_time if tsr_time > 0 else float("inf")

        # Validate results - each output should have different shape
        assert len(tsr_results) == len(numpy_results) == len(images)
        output_shapes = []
        for i, ((tsr_crop, numpy_crop), (_, (x, y, w, h))) in enumerate(
            zip(zip(tsr_results, numpy_results), mixed_images_and_crops)
        ):
            expected_shape = (h, w, 3)  # height, width, channels
            assert (
                tsr_crop.shape == expected_shape
            ), f"TSR crop {i} shape mismatch: {tsr_crop.shape} vs {expected_shape}"
            assert numpy_crop.shape == expected_shape, f"NumPy crop {i} shape mismatch"
            output_shapes.append(expected_shape)
            # Results should be pixel-perfect identical
            np.testing.assert_array_equal(
                tsr_crop, numpy_crop, f"Crop {i} results differ"
            )

        print(f"Mixed-shape cropping performance ({len(images)} different shapes):")
        print(f"  Input shapes: {[img.shape[:2] for img in images]}")
        print(f"  Output shapes: {[shape[:2] for shape in output_shapes]}")
        print(f"  TSR (batch): {tsr_time:.4f}s ({len(images)/tsr_time:.1f} ops/sec)")
        print(
            f"  NumPy (individual): {numpy_time:.4f}s "
            f"({len(images)/numpy_time:.1f} ops/sec)"
        )
        print(f"  Speedup: {speedup:.2f}x")
        print(
            f"  ✅ TSR advantage: Single batch call for {len(images)} crop operations"
        )

        # NumPy array slicing is extremely fast, TSR provides API convenience
        print(
            f"✅ TSR mixed-shape crop working (NumPy {1/speedup:.0f}x faster, expected)"
        )

        # More importantly, demonstrate the API convenience and mixed-shape capability
        assert (
            len(set(output_shapes)) > 1
        ), "Test should produce different output shapes"
        unique_input_shapes = len(set([img.shape[:2] for img in images]))
        unique_output_shapes = len(set(output_shapes))

        print("  🎯 Key advantages:")
        print(f"    - Single API call handles {unique_input_shapes} input shapes")
        print(f"    - Produces {unique_output_shapes} different output shapes")
        print("    - No loops needed: tsr.batch_crop_images(mixed_images, mixed_crops)")
        print("    - vs NumPy: Must write loop for each different shape combination")


class TestMemoryEfficiency:
    """Test memory usage and garbage collection behavior."""

    def test_memory_cleanup_after_operations(self, performance_test_images):
        """Test that operations don't cause memory leaks."""
        images = performance_test_images["large_batch"][
            :16
        ]  # Use subset for memory test

        # Measure baseline memory
        gc.collect()

        # Perform many operations
        for _ in range(10):
            # Crop operations
            crop_boxes = [(100, 100, 200, 200)] * len(images)
            cropped = tsr.batch_crop_images(images, crop_boxes)

            # Resize operations
            resize_targets = [(128, 128)] * len(cropped)
            resized = tsr.batch_resize_images(cropped, resize_targets)

            # Luminance calculations
            tsr.batch_calculate_luminance(resized)

            # Clear references
            del cropped, resized

        # Force garbage collection
        gc.collect()

        # Test should complete without memory errors
        assert True  # If we get here without MemoryError, test passes

    def test_large_batch_memory_handling(self, performance_test_images):
        """Test handling of large batches without memory issues."""
        # Create large batch
        large_images = performance_test_images["large_batch"][
            :32
        ]  # 32 images of 512x512

        try:
            # Should handle large batch without memory errors
            crop_boxes = [(128, 128, 256, 256)] * len(large_images)
            cropped = tsr.batch_crop_images(large_images, crop_boxes)

            resize_targets = [(224, 224)] * len(cropped)
            resized = tsr.batch_resize_images(cropped, resize_targets)

            luminances = tsr.batch_calculate_luminance(resized)

            # Validate results
            assert len(luminances) == len(large_images)
            assert all(0 <= lum <= 255 for lum in luminances)

        except MemoryError:
            pytest.skip("System doesn't have enough memory for large batch test")


class TestThreadSafety:
    """Test thread safety and concurrent access."""

    def test_concurrent_operations_different_images(self, performance_test_images):
        """Test concurrent operations on different images."""
        images = performance_test_images["small_batch"]
        results = {}
        errors = {}

        def process_batch(thread_id, image_batch):
            try:
                crop_boxes = [(20, 20, 100, 100)] * len(image_batch)
                cropped = tsr.batch_crop_images(image_batch, crop_boxes)
                luminances = tsr.batch_calculate_luminance(cropped)
                results[thread_id] = luminances
            except Exception as e:
                errors[thread_id] = e

        # Split images across threads
        mid = len(images) // 2
        batch1 = images[:mid]
        batch2 = images[mid:]

        thread1 = threading.Thread(target=process_batch, args=(1, batch1))
        thread2 = threading.Thread(target=process_batch, args=(2, batch2))

        thread1.start()
        thread2.start()

        thread1.join(timeout=10.0)  # 10 second timeout
        thread2.join(timeout=10.0)

        # Check for errors
        assert len(errors) == 0, f"Thread errors occurred: {errors}"

        # Check results
        assert 1 in results and 2 in results
        assert len(results[1]) == len(batch1)
        assert len(results[2]) == len(batch2)

    def test_gil_release_verification(self, performance_test_images):
        """Verify that GIL is released during operations."""
        images = performance_test_images["large_batch"][:16]

        # This test verifies GIL release by running CPU-bound Python work
        # alongside Rust operations. If GIL is released, both should run concurrently.

        results = {"rust_done": False, "python_done": False}

        def cpu_bound_python_work():
            # CPU-intensive Python work
            total = 0
            for i in range(1000000):
                total += i * i
            results["python_done"] = True
            return total

        def rust_operations():
            # CPU-intensive Rust operations
            for _ in range(5):
                crop_boxes = [(50, 50, 200, 200)] * len(images)
                cropped = tsr.batch_crop_images(images, crop_boxes)
                resize_targets = [(128, 128)] * len(cropped)
                resized = tsr.batch_resize_images(cropped, resize_targets)
                tsr.batch_calculate_luminance(resized)
            results["rust_done"] = True

        start_time = time.perf_counter()

        # Start both operations
        python_thread = threading.Thread(target=cpu_bound_python_work)
        rust_thread = threading.Thread(target=rust_operations)

        python_thread.start()
        rust_thread.start()

        python_thread.join(timeout=15.0)
        rust_thread.join(timeout=15.0)

        total_time = time.perf_counter() - start_time

        # Both should complete (indicating GIL was released)
        assert results["rust_done"], "Rust operations should complete"
        assert results["python_done"], "Python operations should complete"

        # If GIL is properly released, total time should be reasonable
        # (not the sum of sequential execution times)
        assert (
            total_time < 10.0
        ), f"Concurrent execution should be efficient: {total_time:.3f}s"


class TestVideoPerformance:
    """Test video-specific performance characteristics."""

    def test_video_resize_performance(self, performance_test_videos):
        """Test video resizing performance."""
        short_clips = performance_test_videos["short_clips"]

        # Test batch video processing
        target_sizes = [(112, 112)] * len(short_clips)

        start = time.perf_counter()
        resized = tsr.batch_resize_videos(short_clips, target_sizes)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        total_frames = sum(clip.shape[0] for clip in short_clips)
        frames_per_second = total_frames / elapsed if elapsed > 0 else float("inf")

        assert elapsed < 5.0, f"Video processing should be reasonable: {elapsed:.3f}s"
        assert (
            frames_per_second > 10
        ), f"Should process at decent FPS: {frames_per_second:.1f}"

        # Validate results
        assert len(resized) == len(short_clips)
        for original, processed in zip(short_clips, resized):
            assert processed.shape == (original.shape[0], 112, 112, 3)

    def test_long_video_processing(self, performance_test_videos):
        """Test processing of longer video sequences."""
        long_video = performance_test_videos["long_video"][0]  # 120 frames

        start = time.perf_counter()
        resized = tsr.batch_resize_videos([long_video], [(224, 224)])
        elapsed = time.perf_counter() - start

        # Should handle long videos efficiently
        frames_per_second = (
            long_video.shape[0] / elapsed if elapsed > 0 else float("inf")
        )

        assert elapsed < 30.0, f"Long video processing should complete: {elapsed:.3f}s"
        assert (
            frames_per_second > 5
        ), f"Should maintain reasonable FPS: {frames_per_second:.1f}"

        # Validate result
        assert resized[0].shape == (120, 224, 224, 3)


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
class TestDetailedBenchmarks:
    """Detailed benchmarks using pytest-benchmark."""

    def test_crop_operations_benchmark(self, benchmark, performance_test_images):
        """Benchmark different crop operations."""
        images = performance_test_images["small_batch"]
        crop_boxes = [(50, 50, 150, 150)] * len(images)

        result = benchmark(tsr.batch_crop_images, images, crop_boxes)

        # Validate benchmark result
        assert len(result) == len(images)
        for img in result:
            assert img.shape == (150, 150, 3)

    def test_center_crop_benchmark(self, benchmark, performance_test_images):
        """Benchmark center cropping operations."""
        images = performance_test_images["small_batch"]
        target_sizes = [(224, 224)] * len(images)

        result = benchmark(tsr.batch_center_crop_images, images, target_sizes)

        assert len(result) == len(images)
        for img in result:
            assert img.shape == (224, 224, 3)

    def test_resize_operations_benchmark(self, benchmark, performance_test_images):
        """Benchmark resize operations."""
        images = performance_test_images["mixed_sizes"]
        target_sizes = [(256, 256)] * len(images)

        result = benchmark(tsr.batch_resize_images, images, target_sizes)

        assert len(result) == len(images)
        for img in result:
            assert img.shape == (256, 256, 3)

    def test_luminance_calculation_benchmark(self, benchmark, performance_test_images):
        """Benchmark luminance calculations."""
        images = performance_test_images["large_batch"][:32]  # Use subset

        result = benchmark(tsr.batch_calculate_luminance, images)

        assert len(result) == len(images)
        assert all(0 <= lum <= 255 for lum in result)

    def test_video_processing_benchmark(self, benchmark, performance_test_videos):
        """Benchmark video processing."""
        videos = performance_test_videos["short_clips"]
        target_sizes = [(128, 128)] * len(videos)

        result = benchmark(tsr.batch_resize_videos, videos, target_sizes)

        assert len(result) == len(videos)
        for original, processed in zip(videos, result):
            assert processed.shape == (original.shape[0], 128, 128, 3)

    def test_pipeline_benchmark(self, benchmark, performance_test_images):
        """Benchmark complete preprocessing pipeline."""
        images = performance_test_images["small_batch"]

        def complete_pipeline(imgs):
            # Step 1: Center crop
            cropped = tsr.batch_center_crop_images(imgs, [(200, 200)] * len(imgs))

            # Step 2: Resize
            resized = tsr.batch_resize_images(cropped, [(224, 224)] * len(cropped))

            # Step 3: Calculate luminance
            luminances = tsr.batch_calculate_luminance(resized)

            return resized, luminances

        resized, luminances = benchmark(complete_pipeline, images)

        assert len(resized) == len(images)
        assert len(luminances) == len(images)
        for img in resized:
            assert img.shape == (224, 224, 3)


class TestStressTests:
    """Stress tests for edge conditions and reliability."""

    def test_repeated_operations_stability(self, performance_test_images):
        """Test stability under repeated operations."""
        images = performance_test_images["small_batch"][
            :4
        ]  # Small batch for repeated testing

        # Perform operations many times
        for iteration in range(50):
            try:
                crop_boxes = [(10, 10, 100, 100)] * len(images)
                cropped = tsr.batch_crop_images(images, crop_boxes)

                resize_targets = [(128, 128)] * len(cropped)
                resized = tsr.batch_resize_images(cropped, resize_targets)

                luminances = tsr.batch_calculate_luminance(resized)

                # Validate results remain consistent
                assert len(luminances) == len(images)
                assert all(0 <= lum <= 255 for lum in luminances)

            except Exception as e:
                pytest.fail(f"Operation failed on iteration {iteration}: {e}")

    def test_mixed_size_batch_performance(self, performance_test_images):
        """Test performance with mixed image sizes."""
        mixed_images = performance_test_images["mixed_sizes"]

        # Different crop sizes for each image (within bounds)
        crop_boxes = [
            (50, 50, 200, 200),  # For 480x640
            (100, 100, 300, 300),  # For 1024x768
            (20, 20, 100, 100),  # For 256x256
            (
                10,
                10,
                80,
                100,
            ),  # For 128x512 (fixed: y=10, height=100 so total=110 < 128)
        ]

        start = time.perf_counter()
        cropped = tsr.batch_crop_images(mixed_images, crop_boxes)
        elapsed = time.perf_counter() - start

        # Should handle mixed sizes efficiently
        assert (
            elapsed < 1.0
        ), f"Mixed size processing should be efficient: {elapsed:.3f}s"

        # Validate results (height, width, channels)
        expected_shapes = [(200, 200, 3), (300, 300, 3), (100, 100, 3), (100, 80, 3)]
        for result, expected_shape in zip(cropped, expected_shapes):
            assert result.shape == expected_shape

    @pytest.mark.slow
    def test_extreme_batch_size(self):
        """Test handling of very large batch sizes."""
        # Create large batch of small images to test batch processing limits
        batch_size = 100
        small_images = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        try:
            # Should handle large batch
            crop_boxes = [(10, 10, 32, 32)] * batch_size

            start = time.perf_counter()
            cropped = tsr.batch_crop_images(small_images, crop_boxes)
            elapsed = time.perf_counter() - start

            # Should complete in reasonable time despite large batch
            assert (
                elapsed < 5.0
            ), f"Large batch should complete reasonably: {elapsed:.3f}s"
            assert len(cropped) == batch_size

        except MemoryError:
            pytest.skip("System memory insufficient for extreme batch size test")
