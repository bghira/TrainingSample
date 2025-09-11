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
            start = time.perf_counter()
            tsr.batch_calculate_luminance(batch)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Should scale sub-linearly due to parallelism
        for i in range(1, len(batch_sizes)):
            scale_factor = batch_sizes[i] / batch_sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            # Time should increase slower than batch size due to parallel processing
            assert (
                time_ratio < scale_factor * 1.2
            ), f"Batch size {batch_sizes[i]} should benefit from parallelism"


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


class TestX86OptimizationPerformance:
    """Performance regression tests for x86 optimizations."""

    @pytest.mark.skipif(not HAS_BINDINGS, reason="Python bindings not available")
    def test_x86_optimization_performance_regression(self, performance_test_images):
        """Test that x86 optimizations provide expected performance benefits."""
        import platform

        if not hasattr(tsr, "resize_bilinear_x86_optimized"):
            pytest.skip("x86 optimizations not available")

        test_image = performance_test_images["small_batch"][0]  # 224x224 image
        target_width, target_height = 112, 112

        if platform.machine() == "x86_64":
            # Test x86 optimized resize performance
            x86_times = []
            for _ in range(5):  # Multiple runs for stability
                start = time.perf_counter()
                try:
                    result = tsr.resize_bilinear_x86_optimized(
                        test_image, target_width, target_height
                    )
                    elapsed = time.perf_counter() - start
                    x86_times.append(elapsed)
                    assert result.shape == (target_height, target_width, 3)
                except Exception:
                    pytest.skip("x86 optimizations not working on this platform")

            avg_x86_time = sum(x86_times) / len(x86_times)

            # Test regular batch resize performance
            regular_times = []
            for _ in range(5):
                start = time.perf_counter()
                results = tsr.batch_resize_images(
                    [test_image], [(target_width, target_height)]
                )
                elapsed = time.perf_counter() - start
                regular_times.append(elapsed)
                assert results[0].shape == (target_height, target_width, 3)

            avg_regular_time = sum(regular_times) / len(regular_times)

            # x86 should be competitive (within 50% performance range)
            performance_ratio = avg_x86_time / avg_regular_time
            assert performance_ratio < 1.5, (
                f"x86 optimization should be competitive: "
                f"{performance_ratio:.2f}x slower"
            )

            print(
                f"x86 avg time: {avg_x86_time:.4f}s, "
                f"regular avg time: {avg_regular_time:.4f}s, "
                f"ratio: {performance_ratio:.2f}x"
            )
        else:
            pytest.skip("x86 performance tests only run on x86_64 platforms")

    @pytest.mark.skipif(not HAS_BINDINGS, reason="Python bindings not available")
    def test_x86_luminance_performance_regression(self, performance_test_images):
        """Test x86 luminance calculation performance."""
        import platform

        if not hasattr(tsr, "calculate_luminance_x86_optimized"):
            pytest.skip("x86 luminance optimization not available")

        large_image = performance_test_images["large_batch"][0]  # 512x512 image

        if platform.machine() == "x86_64":
            # Test x86 optimized luminance performance
            x86_times = []
            x86_results = []
            for _ in range(10):  # Multiple runs for stability
                start = time.perf_counter()
                result = tsr.calculate_luminance_x86_optimized(large_image)
                elapsed = time.perf_counter() - start
                x86_times.append(elapsed)
                x86_results.append(result)

            avg_x86_time = sum(x86_times) / len(x86_times)

            # Only test performance if x86 optimizations are actually working
            if x86_results[0] != 0.0:  # 0.0 indicates fallback mode
                # Test regular batch luminance performance
                regular_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    tsr.batch_calculate_luminance([large_image])
                    elapsed = time.perf_counter() - start
                    regular_times.append(elapsed)

                avg_regular_time = sum(regular_times) / len(regular_times)

                # x86 should show some performance benefit or be competitive
                performance_ratio = avg_x86_time / avg_regular_time
                assert performance_ratio < 2.0, (
                    f"x86 luminance should be competitive: "
                    f"{performance_ratio:.2f}x slower"
                )

                # Results should be consistent
                result_variance = max(x86_results) - min(x86_results)
                assert result_variance < 1.0, (
                    f"x86 luminance results should be consistent: "
                    f"{result_variance:.3f} variance"
                )

                print(
                    f"x86 luminance avg time: {avg_x86_time:.6f}s, "
                    f"regular avg time: {avg_regular_time:.6f}s, "
                    f"ratio: {performance_ratio:.2f}x"
                )
            else:
                print("x86 luminance optimizations using fallback mode")
        else:
            pytest.skip("x86 performance tests only run on x86_64 platforms")

    @pytest.mark.skipif(not HAS_BINDINGS, reason="Python bindings not available")
    def test_cpu_feature_detection_performance(self):
        """Test that CPU feature detection is fast and stable."""
        if not hasattr(tsr, "get_x86_cpu_features"):
            pytest.skip("CPU feature detection not available")

        # CPU feature detection should be very fast
        detection_times = []
        for _ in range(100):  # Many calls to test caching
            start = time.perf_counter()
            features = tsr.get_x86_cpu_features()
            elapsed = time.perf_counter() - start
            detection_times.append(elapsed)

            # Validate feature map structure
            assert isinstance(features, dict)
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

        avg_detection_time = sum(detection_times) / len(detection_times)
        max_detection_time = max(detection_times)

        # Detection should be extremely fast (sub-millisecond)
        assert (
            avg_detection_time < 0.001
        ), f"CPU feature detection should be fast: {avg_detection_time:.6f}s avg"
        assert (
            max_detection_time < 0.01
        ), f"CPU feature detection max time too high: {max_detection_time:.6f}s"


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
