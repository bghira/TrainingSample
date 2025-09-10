"""Comprehensive and meaningful tests for training_sample_rust."""

import numpy as np
import pytest

try:
    import trainingsample as tsr

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

pytestmark = pytest.mark.skipif(
    not HAS_BINDINGS, reason="Python bindings not available"
)


@pytest.fixture
def real_test_images():
    """Create realistic test images with known patterns."""
    # Create a gradient image
    gradient = np.zeros((200, 300, 3), dtype=np.uint8)
    for y in range(200):
        for x in range(300):
            gradient[y, x] = [
                int(255 * x / 299),  # Red gradient left to right
                int(255 * y / 199),  # Green gradient top to bottom
                128,  # Constant blue
            ]

    # Create a checkerboard pattern
    checkerboard = np.zeros((128, 128, 3), dtype=np.uint8)
    for y in range(128):
        for x in range(128):
            if (x // 16 + y // 16) % 2 == 0:
                checkerboard[y, x] = [255, 255, 255]  # White
            else:
                checkerboard[y, x] = [0, 0, 0]  # Black

    # Create a solid color image
    solid_red = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)

    return [gradient, checkerboard, solid_red]


@pytest.fixture
def edge_case_images():
    """Create edge case images for thorough testing."""
    # Very small image
    tiny = np.random.randint(0, 255, (2, 2, 3), dtype=np.uint8)

    # Very large image (but manageable for testing)
    large = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)

    # Square image
    square = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Extreme aspect ratios
    wide = np.random.randint(0, 255, (50, 500, 3), dtype=np.uint8)
    tall = np.random.randint(0, 255, (500, 50, 3), dtype=np.uint8)

    return [tiny, large, square, wide, tall]


class TestDataValidation:
    """Test data integrity and validation."""

    def test_preserves_data_type(self, real_test_images):
        """Test that operations preserve uint8 data type."""
        for img in real_test_images:
            # Test cropping preserves dtype
            crop_boxes = [(10, 10, 50, 50)]
            cropped = tsr.batch_crop_images([img], crop_boxes)
            assert cropped[0].dtype == np.uint8

            # Test resizing preserves dtype
            target_sizes = [(64, 64)]
            resized = tsr.batch_resize_images([img], target_sizes)
            assert resized[0].dtype == np.uint8

    def test_data_range_preservation(self, real_test_images):
        """Test that pixel values stay in valid range [0, 255]."""
        for img in real_test_images:
            # Test center cropping
            cropped = tsr.batch_center_crop_images([img], [(50, 50)])
            assert np.all(cropped[0] >= 0) and np.all(cropped[0] <= 255)

            # Test resizing
            resized = tsr.batch_resize_images([img], [(100, 100)])
            assert np.all(resized[0] >= 0) and np.all(resized[0] <= 255)

    def test_channel_integrity(self, real_test_images):
        """Test that RGB channels are preserved correctly."""
        solid_red = real_test_images[2]  # Known to be solid red

        # Crop and check red channel is preserved
        cropped = tsr.batch_center_crop_images([solid_red], [(32, 32)])
        assert np.all(cropped[0][:, :, 0] == 255)  # Red channel
        assert np.all(cropped[0][:, :, 1] == 0)  # Green channel
        assert np.all(cropped[0][:, :, 2] == 0)  # Blue channel


class TestGeometricOperations:
    """Test geometric correctness of operations."""

    def test_center_crop_precision(self, real_test_images):
        """Test center crop calculates coordinates correctly."""
        gradient = real_test_images[0]  # 200x300 gradient

        # Center crop to 100x100 should start at (50, 100) for (height, width)
        cropped = tsr.batch_center_crop_images([gradient], [(100, 100)])

        # Check that we got the center portion
        expected_center_pixel = gradient[100, 150]  # Center of original
        actual_center_pixel = cropped[0][50, 50]  # Center of cropped
        np.testing.assert_array_equal(actual_center_pixel, expected_center_pixel)

    def test_crop_coordinate_accuracy(self, real_test_images):
        """Test that custom crop coordinates are precise."""
        gradient = real_test_images[0]  # 200x300 gradient

        # Crop from (25, 50) with size 75x100
        crop_boxes = [(50, 25, 100, 75)]  # (x, y, width, height)
        cropped = tsr.batch_crop_images([gradient], crop_boxes)

        # Top-left corner should match original at (25, 50)
        expected = gradient[25, 50]
        actual = cropped[0][0, 0]
        np.testing.assert_array_equal(actual, expected)

        # Bottom-right corner should match original at (25+74, 50+99)
        expected = gradient[98, 149]  # 25+74-1, 50+99-1 (inclusive)
        actual = cropped[0][73, 99]  # Last pixel in crop
        np.testing.assert_array_equal(actual, expected)

    def test_resize_aspect_ratio_handling(self, real_test_images):
        """Test how resizing handles different aspect ratios."""
        gradient = real_test_images[0]  # 200x300 aspect ratio 1.5

        # Resize to square - should not distort, but interpolate
        resized = tsr.batch_resize_images([gradient], [(150, 150)])
        assert resized[0].shape == (150, 150, 3)

        # Resize to different aspect ratio
        resized = tsr.batch_resize_images([gradient], [(100, 200)])
        assert resized[0].shape == (200, 100, 3)  # (height, width, channels)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_images(self, edge_case_images):
        """Test operations on very small images."""
        tiny = edge_case_images[0]  # 2x2 image

        # Should handle 1x1 crop
        cropped = tsr.batch_crop_images([tiny], [(0, 0, 1, 1)])
        assert cropped[0].shape == (1, 1, 3)

        # Should handle resize to larger size
        resized = tsr.batch_resize_images([tiny], [(10, 10)])
        assert resized[0].shape == (10, 10, 3)

    def test_extreme_aspect_ratios(self, edge_case_images):
        """Test operations on extreme aspect ratios."""
        wide = edge_case_images[3]  # 50x500
        tall = edge_case_images[4]  # 500x50

        # Test cropping extreme ratios
        wide_cropped = tsr.batch_center_crop_images([wide], [(25, 25)])
        assert wide_cropped[0].shape == (25, 25, 3)

        tall_cropped = tsr.batch_center_crop_images([tall], [(25, 25)])
        assert tall_cropped[0].shape == (25, 25, 3)

        # Test resizing to square
        wide_resized = tsr.batch_resize_images([wide], [(64, 64)])
        tall_resized = tsr.batch_resize_images([tall], [(64, 64)])
        assert wide_resized[0].shape == (64, 64, 3)
        assert tall_resized[0].shape == (64, 64, 3)

    def test_boundary_crops(self, real_test_images):
        """Test crops at image boundaries."""
        gradient = real_test_images[0]  # 200x300

        # Crop from corner to corner (full image)
        full_crop = tsr.batch_crop_images([gradient], [(0, 0, 300, 200)])
        np.testing.assert_array_equal(full_crop[0], gradient)

        # Crop single pixel from corner
        corner_crop = tsr.batch_crop_images([gradient], [(0, 0, 1, 1)])
        np.testing.assert_array_equal(corner_crop[0][0, 0], gradient[0, 0])

        # Crop from opposite corner
        opposite_crop = tsr.batch_crop_images([gradient], [(299, 199, 1, 1)])
        np.testing.assert_array_equal(opposite_crop[0][0, 0], gradient[199, 299])


class TestLuminanceAccuracy:
    """Test luminance calculation accuracy."""

    def test_known_luminance_values(self):
        """Test luminance calculation against known values."""
        # Pure white should have luminance ~255
        white = np.full((10, 10, 3), [255, 255, 255], dtype=np.uint8)
        luminance = tsr.batch_calculate_luminance([white])
        assert abs(luminance[0] - 255.0) < 1.0

        # Pure black should have luminance ~0
        black = np.full((10, 10, 3), [0, 0, 0], dtype=np.uint8)
        luminance = tsr.batch_calculate_luminance([black])
        assert abs(luminance[0] - 0.0) < 1.0

        # Pure red should have luminance ~76 (0.299 * 255)
        red = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        luminance = tsr.batch_calculate_luminance([red])
        expected_red_luminance = 0.299 * 255
        assert abs(luminance[0] - expected_red_luminance) < 1.0

        # Pure green should have luminance ~150 (0.587 * 255)
        green = np.full((10, 10, 3), [0, 255, 0], dtype=np.uint8)
        luminance = tsr.batch_calculate_luminance([green])
        expected_green_luminance = 0.587 * 255
        assert abs(luminance[0] - expected_green_luminance) < 1.0

        # Pure blue should have luminance ~29 (0.114 * 255)
        blue = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)
        luminance = tsr.batch_calculate_luminance([blue])
        expected_blue_luminance = 0.114 * 255
        assert abs(luminance[0] - expected_blue_luminance) < 1.0

    def test_luminance_gradient(self, real_test_images):
        """Test luminance calculation on gradient image."""
        gradient = real_test_images[0]  # Has known gradient pattern

        # Calculate luminance for the gradient
        luminance = tsr.batch_calculate_luminance([gradient])

        # Should be somewhere in middle range due to gradient
        assert 50 < luminance[0] < 200

        # Compare with manual calculation on a subset
        manual_luminance = np.mean(
            0.299 * gradient[:, :, 0]
            + 0.587 * gradient[:, :, 1]
            + 0.114 * gradient[:, :, 2]
        )
        assert abs(luminance[0] - manual_luminance) < 1.0


class TestBatchConsistency:
    """Test that batch operations are consistent with individual operations."""

    def test_batch_vs_individual_crops(self, real_test_images):
        """Test batch cropping gives same results as individual crops."""
        images = real_test_images[:2]
        crop_boxes = [(10, 20, 50, 60), (25, 35, 40, 50)]

        # Batch operation
        batch_results = tsr.batch_crop_images(images, crop_boxes)

        # Individual operations
        individual_results = []
        for img, crop_box in zip(images, crop_boxes):
            result = tsr.batch_crop_images([img], [crop_box])
            individual_results.append(result[0])

        # Results should be identical
        for batch_result, individual_result in zip(batch_results, individual_results):
            np.testing.assert_array_equal(batch_result, individual_result)

    def test_batch_vs_individual_resize(self, real_test_images):
        """Test batch resizing gives same results as individual resizes."""
        images = real_test_images[:2]
        target_sizes = [(64, 64), (128, 96)]

        # Batch operation
        batch_results = tsr.batch_resize_images(images, target_sizes)

        # Individual operations
        individual_results = []
        for img, size in zip(images, target_sizes):
            result = tsr.batch_resize_images([img], [size])
            individual_results.append(result[0])

        # Results should be identical
        for batch_result, individual_result in zip(batch_results, individual_results):
            np.testing.assert_array_equal(batch_result, individual_result)


class TestVideoOperations:
    """Test video-specific operations."""

    def test_video_frame_consistency(self):
        """Test that video operations maintain frame relationships."""
        # Create a video where each frame is different
        video = np.zeros((5, 64, 64, 3), dtype=np.uint8)
        for f in range(5):
            video[f, :, :, 0] = f * 50  # Each frame has different red intensity

        # Resize video
        resized_video = tsr.batch_resize_videos([video], [(32, 32)])
        resized = resized_video[0]

        # Check that frame differences are preserved
        assert resized.shape == (5, 32, 32, 3)

        # Frame 0 should be darker than frame 4
        assert np.mean(resized[0, :, :, 0]) < np.mean(resized[4, :, :, 0])

        # Frame relationships should be maintained
        for f in range(4):
            assert np.mean(resized[f, :, :, 0]) <= np.mean(resized[f + 1, :, :, 0])


class TestRealWorldScenarios:
    """Test scenarios that mirror real-world usage."""

    def test_image_preprocessing_pipeline(self, real_test_images):
        """Test a complete image preprocessing pipeline."""
        original_images = real_test_images

        # Step 1: Center crop to square
        cropped = tsr.batch_center_crop_images(
            original_images, [(128, 128)] * len(original_images)
        )

        # Step 2: Resize to training size
        resized = tsr.batch_resize_images(cropped, [(224, 224)] * len(cropped))

        # Step 3: Calculate luminance for filtering
        luminances = tsr.batch_calculate_luminance(resized)

        # Validate pipeline results
        assert all(img.shape == (224, 224, 3) for img in resized)
        assert len(luminances) == len(original_images)
        assert all(0 <= lum <= 255 for lum in luminances)

        # Check that processing didn't create invalid data
        for img in resized:
            assert np.all(img >= 0) and np.all(img <= 255)
            assert img.dtype == np.uint8

    def test_data_augmentation_pipeline(self, real_test_images):
        """Test random augmentation pipeline."""
        original_image = real_test_images[0]

        # Generate multiple random crops of same image
        num_augmentations = 5
        crop_size = (64, 64)

        augmented_crops = []
        for _ in range(num_augmentations):
            cropped = tsr.batch_random_crop_images([original_image], [crop_size])
            augmented_crops.append(cropped[0])

        # All should have same target size
        assert all(crop.shape == (64, 64, 3) for crop in augmented_crops)

        # At least some should be different (very high probability)
        all_same = all(
            np.array_equal(augmented_crops[0], crop) for crop in augmented_crops[1:]
        )
        assert not all_same  # Random crops should produce different results

    def test_batch_efficiency_validation(self, real_test_images):
        """Test that batch operations handle realistic batch sizes."""
        # Simulate a realistic training batch
        batch_size = 16
        images = real_test_images * 6  # Repeat to get 18 images
        images = images[:batch_size]  # Take exactly 16

        # Process entire batch
        crop_sizes = [(128, 128)] * batch_size
        resize_targets = [(224, 224)] * batch_size

        # Should handle batch without issues
        cropped = tsr.batch_center_crop_images(images, crop_sizes)
        resized = tsr.batch_resize_images(cropped, resize_targets)
        luminances = tsr.batch_calculate_luminance(resized)

        # Validate batch results
        assert len(cropped) == batch_size
        assert len(resized) == batch_size
        assert len(luminances) == batch_size

        # All outputs should be valid
        for img in resized:
            assert img.shape == (224, 224, 3)
            assert img.dtype == np.uint8
            assert np.all(img >= 0) and np.all(img <= 255)


class TestErrorHandlingAndRobustness:
    """Test error conditions and robustness."""

    def test_invalid_crop_coordinates(self, real_test_images):
        """Test handling of invalid crop coordinates."""
        image = real_test_images[0]  # 200x300 image

        # Test crop that exceeds image bounds
        with pytest.raises((RuntimeError, ValueError)):
            tsr.batch_crop_images([image], [(250, 150, 100, 100)])  # x exceeds width

        with pytest.raises((RuntimeError, ValueError)):
            tsr.batch_crop_images(
                [image], [(50, 180, 100, 100)]
            )  # y+height exceeds height

    def test_zero_size_operations(self, real_test_images):
        """Test handling of zero-size operations."""
        image = real_test_images[0]

        # Zero-width crop should fail gracefully
        with pytest.raises((RuntimeError, ValueError)):
            tsr.batch_crop_images([image], [(10, 10, 0, 50)])

        # Zero-height crop should fail gracefully
        with pytest.raises((RuntimeError, ValueError)):
            tsr.batch_crop_images([image], [(10, 10, 50, 0)])

        # Zero-size resize should fail gracefully
        with pytest.raises((RuntimeError, ValueError)):
            tsr.batch_resize_images([image], [(0, 100)])

    def test_mismatched_batch_sizes(self, real_test_images):
        """Test handling of mismatched input batch sizes."""
        images = real_test_images[:2]

        # Too few crop boxes
        crop_boxes = [(10, 10, 50, 50)]  # Only 1 box for 2 images

        # This should either handle gracefully or raise clear error
        try:
            result = tsr.batch_crop_images(images, crop_boxes)
            # If it succeeds, should process only as many as crop boxes provided
            assert len(result) == 1
        except (RuntimeError, ValueError, IndexError):
            # Clear error is also acceptable
            pass

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        # Empty image list should return empty results
        result = tsr.batch_crop_images([], [])
        assert len(result) == 0

        result = tsr.batch_resize_images([], [])
        assert len(result) == 0

        result = tsr.batch_calculate_luminance([])
        assert len(result) == 0
