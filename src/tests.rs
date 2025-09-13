#[allow(unused_imports)]
use super::*;
use ndarray::{Array3, Array4};

fn create_test_image() -> Array3<u8> {
    Array3::from_shape_fn((100, 100, 3), |(y, x, c)| ((x + y + c * 50) % 256) as u8)
}

#[allow(dead_code)]
fn create_test_video() -> Array4<u8> {
    Array4::from_shape_fn((5, 50, 50, 3), |(f, y, x, c)| {
        ((x + y + c * 50 + f * 10) % 256) as u8
    })
}

#[cfg(test)]
mod x86_optimization_tests {
    #[allow(unused_imports)]
    use super::*;
    use ndarray::Array3;

    #[allow(dead_code)]
    fn create_large_test_image() -> Array3<u8> {
        Array3::from_shape_fn((256, 256, 3), |(y, x, c)| ((x + y + c * 85) % 256) as u8)
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_integration_with_batch_functions() {
        use crate::core::{batch_calculate_luminance_arrays, batch_resize_image_arrays};

        let images = vec![create_test_image(), create_large_test_image()];

        // Test batch resize
        let target_sizes = [(64u32, 64u32); 2];
        let batch_resize_result = batch_resize_image_arrays(&images, &target_sizes);
        assert_eq!(batch_resize_result.len(), 2);

        for result in batch_resize_result {
            assert!(result.is_ok(), "Batch resize should succeed");
            let resized = result.unwrap();
            assert_eq!(resized.dim(), (64, 64, 3));
        }

        // Test batch luminance
        let batch_luminance_result = batch_calculate_luminance_arrays(&images);
        assert_eq!(batch_luminance_result.len(), 2);

        for luminance in batch_luminance_result {
            assert!(
                luminance > 0.0 && luminance < 255.0,
                "Luminance should be valid"
            );
        }
    }
}

#[cfg(test)]
mod cropping_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_crop_image_array() {
        let img = create_test_image();
        let result = crop_image_array(&img.view(), 10, 10, 50, 50).unwrap();
        assert_eq!(result.dim(), (50, 50, 3));

        // Check that cropped values match original
        assert_eq!(result[[0, 0, 0]], img[[10, 10, 0]]);
        assert_eq!(result[[25, 25, 1]], img[[35, 35, 1]]);
    }

    #[test]
    fn test_crop_out_of_bounds() {
        let img = create_test_image();
        let result = crop_image_array(&img.view(), 90, 90, 50, 50);
        assert!(result.is_err());
    }

    #[test]
    fn test_center_crop_image_array() {
        let img = create_test_image();
        let result = center_crop_image_array(&img.view(), 60, 60).unwrap();
        assert_eq!(result.dim(), (60, 60, 3));

        // Center crop should start at (20, 20) for 100x100 -> 60x60
        assert_eq!(result[[0, 0, 0]], img[[20, 20, 0]]);
    }

    #[test]
    fn test_random_crop_image_array() {
        let img = create_test_image();
        let result = random_crop_image_array(&img.view(), 50, 50).unwrap();
        assert_eq!(result.dim(), (50, 50, 3));
    }

    #[test]
    fn test_center_crop_larger_than_image() {
        let img = create_test_image();
        let result = center_crop_image_array(&img.view(), 150, 150).unwrap();
        // Should return the original image size when target is larger
        assert_eq!(result.dim(), (100, 100, 3));
    }
}

#[cfg(test)]
mod luminance_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_calculate_luminance_array() {
        let img = create_test_image();
        let luminance = calculate_luminance_array(&img.view());

        // Luminance should be a reasonable value between 0 and 255
        assert!(luminance >= 0.0);
        assert!(luminance <= 255.0);
    }

    #[test]
    fn test_luminance_pure_white() {
        let mut img = Array3::<u8>::zeros((10, 10, 3));
        img.fill(255);
        let luminance = calculate_luminance_array(&img.view());

        // Pure white should have high luminance
        assert!((luminance - 255.0).abs() < 1.0);
    }

    #[test]
    fn test_luminance_pure_black() {
        let img = Array3::<u8>::zeros((10, 10, 3));
        let luminance = calculate_luminance_array(&img.view());

        // Pure black should have zero luminance
        assert!((luminance - 0.0).abs() < 1.0);
    }
}

#[cfg(test)]
mod resize_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "opencv")]
    fn test_resize_image_array() {
        let img = create_test_image();
        let result = resize_image_array(&img.view(), 64, 64).unwrap();
        assert_eq!(result.dim(), (64, 64, 3));
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_resize_video_array() {
        let video = create_test_video();
        let result = resize_video_array(&video.view(), 32, 32).unwrap();
        assert_eq!(result.dim(), (5, 32, 32, 3));
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_resize_upscale() {
        let img = Array3::from_shape_fn((10, 10, 3), |(y, x, c)| (x + y + c) as u8);
        let result = resize_image_array(&img.view(), 20, 20).unwrap();
        assert_eq!(result.dim(), (20, 20, 3));
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_resize_downscale() {
        let img = create_test_image();
        let result = resize_image_array(&img.view(), 25, 25).unwrap();
        assert_eq!(result.dim(), (25, 25, 3));
    }
}

#[cfg(test)]
mod batch_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_batch_crop_image_arrays() {
        let images = vec![create_test_image(), create_test_image()];
        let crop_boxes = vec![(10, 10, 50, 50), (20, 20, 40, 40)];

        let results = batch_crop_image_arrays(&images, &crop_boxes);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());

        assert_eq!(results[0].as_ref().unwrap().dim(), (50, 50, 3));
        assert_eq!(results[1].as_ref().unwrap().dim(), (40, 40, 3));
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_batch_resize_image_arrays() {
        let images = vec![create_test_image(), create_test_image()];
        let target_sizes = vec![(64, 64), (32, 32)];

        let results = batch_resize_image_arrays(&images, &target_sizes);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());

        assert_eq!(results[0].as_ref().unwrap().dim(), (64, 64, 3));
        assert_eq!(results[1].as_ref().unwrap().dim(), (32, 32, 3));
    }

    #[test]
    fn test_batch_calculate_luminance_arrays() {
        let images = vec![create_test_image(), create_test_image()];
        let results = batch_calculate_luminance_arrays(&images);

        assert_eq!(results.len(), 2);
        for luminance in results {
            assert!((0.0..=255.0).contains(&luminance));
        }
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_batch_resize_video_arrays() {
        let videos = vec![create_test_video()];
        let target_sizes = vec![(25, 25)];

        let results = batch_resize_video_arrays(&videos, &target_sizes);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        assert_eq!(results[0].as_ref().unwrap().dim(), (5, 25, 25, 3));
    }
}

#[cfg(test)]
mod integration_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "opencv")]
    fn test_crop_then_resize_pipeline() {
        let img = create_test_image();

        // First crop
        let cropped = crop_image_array(&img.view(), 20, 20, 60, 60).unwrap();
        assert_eq!(cropped.dim(), (60, 60, 3));

        // Then resize
        let resized = resize_image_array(&cropped.view(), 32, 32).unwrap();
        assert_eq!(resized.dim(), (32, 32, 3));
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_full_processing_pipeline() {
        let img = create_test_image();

        // Center crop
        let cropped = center_crop_image_array(&img.view(), 80, 80).unwrap();

        // Resize
        let resized = resize_image_array(&cropped.view(), 64, 64).unwrap();

        // Calculate luminance
        let luminance = calculate_luminance_array(&resized.view());

        assert_eq!(resized.dim(), (64, 64, 3));
        assert!((0.0..=255.0).contains(&luminance));
    }
}

#[cfg(test)]
mod edge_case_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_single_pixel_image() {
        let img = Array3::from_shape_fn((1, 1, 3), |_| 128);
        let luminance = calculate_luminance_array(&img.view());
        assert!((luminance - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_crop_exact_size() {
        let img = create_test_image();
        let result = crop_image_array(&img.view(), 0, 0, 100, 100).unwrap();
        assert_eq!(result.dim(), (100, 100, 3));

        // Should be identical to original
        for ((y, x, c), &original_val) in img.indexed_iter() {
            assert_eq!(result[[y, x, c]], original_val);
        }
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_resize_same_size() {
        let img = create_test_image();
        let result = resize_image_array(&img.view(), 100, 100).unwrap();
        assert_eq!(result.dim(), (100, 100, 3));
    }
}

// OpenCV specific tests
#[cfg(test)]
mod opencv_tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(feature = "opencv")]
    use crate::opencv_ops::OpenCVBatchProcessor;

    #[test]
    #[cfg(feature = "opencv")]
    fn test_opencv_batch_resize() {
        let processor = OpenCVBatchProcessor::new();
        let test_image = create_test_image();

        let images = vec![test_image.view()];
        let target_sizes = vec![(128, 128)];

        let results = processor.batch_resize_lanczos4(&images, &target_sizes);
        assert!(results.is_ok());

        let resized = results.unwrap();
        assert_eq!(resized.len(), 1);
        assert_eq!(resized[0].dim(), (128, 128, 3));
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_opencv_batch_luminance() {
        let processor = OpenCVBatchProcessor::new();
        let test_image = create_test_image();

        let images = vec![test_image.view()];
        let results = processor.batch_calculate_luminance_opencv(&images);
        assert!(results.is_ok());

        let luminance = results.unwrap();
        assert_eq!(luminance.len(), 1);
        assert!(luminance[0] >= 0.0 && luminance[0] <= 255.0);
    }
}
