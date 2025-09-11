use super::*;
use ndarray::{Array3, Array4};

fn create_test_image() -> Array3<u8> {
    Array3::from_shape_fn((100, 100, 3), |(y, x, c)| ((x + y + c * 50) % 256) as u8)
}

fn create_test_video() -> Array4<u8> {
    Array4::from_shape_fn((5, 50, 50, 3), |(f, y, x, c)| {
        ((x + y + c * 50 + f * 10) % 256) as u8
    })
}

#[cfg(test)]
mod x86_optimization_tests {
    use super::*;
    use ndarray::Array3;

    fn create_large_test_image() -> Array3<u8> {
        Array3::from_shape_fn((256, 256, 3), |(y, x, c)| ((x + y + c * 85) % 256) as u8)
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_x86_resize_engine_creation() {
        use crate::resize_x86_optimized::X86ResizeEngine;

        let result = X86ResizeEngine::new();
        assert!(result.is_ok(), "Should be able to create X86ResizeEngine");

        let engine = result.unwrap();
        let features = engine.cpu_features();

        // On Apple Silicon, all features should be false, but that's expected
        println!("Detected x86 features:");
        println!("  AVX-512F: {}", features.has_avx512f);
        println!("  AVX-512BW: {}", features.has_avx512bw);
        println!("  AVX2: {}", features.has_avx2);
        println!("  FMA: {}", features.has_fma);
        println!("  SSE4.1: {}", features.has_sse41);
        println!("  AMD Zen: {}", features.is_amd_zen);

        // Features should all be false on Apple Silicon, which is correct
        #[cfg(target_arch = "aarch64")]
        {
            assert!(!features.has_avx512f);
            assert!(!features.has_avx2);
            assert!(!features.has_fma);
            assert!(!features.has_sse41);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_x86_resize_basic_functionality() {
        use crate::resize_x86_optimized::X86ResizeEngine;

        let engine = X86ResizeEngine::new().unwrap();
        let image = create_test_image();

        let result = engine.resize_bilinear(&image.view(), 50, 50);
        assert!(result.is_ok(), "x86 resize should succeed");

        let resized = result.unwrap();
        assert_eq!(resized.dim(), (50, 50, 3));

        // Verify cores were used for processing
        let cores_used = engine.cores_used();
        assert!(cores_used > 0, "Should use at least one core");
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_x86_resize_different_sizes() {
        use crate::resize_x86_optimized::X86ResizeEngine;

        let engine = X86ResizeEngine::new().unwrap();
        let image = create_large_test_image();

        // Test various resize targets
        let test_cases = vec![
            (128, 128, "Half size"),
            (512, 512, "Double size"),
            (100, 200, "Different aspect ratio"),
            (17, 23, "Small odd dimensions"),
        ];

        for (width, height, desc) in test_cases {
            let result = engine.resize_bilinear(&image.view(), width, height);
            assert!(result.is_ok(), "x86 resize should succeed for {}", desc);

            let resized = result.unwrap();
            assert_eq!(resized.dim(), (height as usize, width as usize, 3));

            // Verify output is reasonable (not all zeros or all same value)
            let sum: u32 = resized.iter().map(|&x| x as u32).sum();
            assert!(
                sum > 0,
                "Resized image should not be all zeros for {}",
                desc
            );
            assert!(
                sum < (width * height * 3 * 255) as u32,
                "Resized image should not be all max values for {}",
                desc
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_x86_luminance_optimizations() {
        use crate::luminance_x86_optimized::{
            calculate_luminance_x86_optimized, calculate_luminance_x86_parallel,
        };

        let image = create_test_image();

        // Test single-threaded optimization
        let luminance1 = calculate_luminance_x86_optimized(&image.view());
        assert!(
            luminance1 > 0.0 && luminance1 < 255.0,
            "Luminance should be in valid range"
        );

        // Test multi-threaded optimization
        let result = calculate_luminance_x86_parallel(&image.view());
        assert!(
            result.is_ok(),
            "Parallel luminance calculation should succeed"
        );

        let luminance2 = result.unwrap();
        assert!(
            luminance2 > 0.0 && luminance2 < 255.0,
            "Parallel luminance should be in valid range"
        );

        // Results should be very close (allowing for floating point precision)
        let diff = (luminance1 - luminance2).abs();
        assert!(
            diff < 0.1,
            "Single and parallel luminance should be nearly identical, got diff: {}",
            diff
        );
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_x86_resize_consistency_with_fallback() {
        use crate::resize_simd::resize_bilinear_simd;
        use crate::resize_x86_optimized::resize_bilinear_x86_optimized;

        let image = create_test_image();
        let target_width = 75;
        let target_height = 75;

        // Get x86 optimized result
        let x86_result = resize_bilinear_x86_optimized(&image.view(), target_width, target_height);
        assert!(x86_result.is_ok(), "x86 resize should succeed");

        // Get SIMD fallback result
        let simd_result = resize_bilinear_simd(&image.view(), target_width, target_height);
        assert!(simd_result.is_ok(), "SIMD resize should succeed");

        let x86_resized = x86_result.unwrap();
        let (simd_resized, _) = simd_result.unwrap();

        // Both should have same dimensions
        assert_eq!(x86_resized.dim(), simd_resized.dim());

        // Results should be reasonably close (different algorithms may have slight differences)
        let mut total_diff = 0u64;
        let mut pixel_count = 0u64;

        for h in 0..target_height as usize {
            for w in 0..target_width as usize {
                for c in 0..3 {
                    let x86_val = x86_resized[[h, w, c]] as i16;
                    let simd_val = simd_resized[[h, w, c]] as i16;
                    total_diff += (x86_val - simd_val).abs() as u64;
                    pixel_count += 1;
                }
            }
        }

        let avg_diff = total_diff as f64 / pixel_count as f64;
        assert!(
            avg_diff < 2.0,
            "Average pixel difference should be small, got: {}",
            avg_diff
        );
    }

    #[test]
    fn test_x86_fallback_functions_on_non_x86() {
        use crate::luminance_x86_optimized::{
            calculate_luminance_x86_optimized, calculate_luminance_x86_parallel,
        };
        use crate::resize_x86_optimized::resize_bilinear_x86_optimized;

        let image = create_test_image();

        // On non-x86 platforms (like Apple Silicon), these should return appropriate errors/defaults
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            let resize_result = resize_bilinear_x86_optimized(&image.view(), 50, 50);
            assert!(
                resize_result.is_err(),
                "x86 resize should fail gracefully on non-x86"
            );

            let luminance1 = calculate_luminance_x86_optimized(&image.view());
            assert_eq!(
                luminance1, 0.0,
                "x86 luminance should return 0.0 on non-x86"
            );

            let luminance2 = calculate_luminance_x86_parallel(&image.view());
            assert!(
                luminance2.is_err(),
                "x86 parallel luminance should fail gracefully on non-x86"
            );
        }

        // On x86 platforms, these should work
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let resize_result = resize_bilinear_x86_optimized(&image.view(), 50, 50);
            assert!(resize_result.is_ok(), "x86 resize should succeed on x86");

            let luminance1 = calculate_luminance_x86_optimized(&image.view());
            assert!(luminance1 > 0.0, "x86 luminance should work on x86");

            let luminance2 = calculate_luminance_x86_parallel(&image.view());
            assert!(
                luminance2.is_ok(),
                "x86 parallel luminance should work on x86"
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_x86_performance_characteristics() {
        use crate::resize_x86_optimized::X86ResizeEngine;
        use std::time::Instant;

        let engine = X86ResizeEngine::new().unwrap();
        let large_image = create_large_test_image();

        // Time a resize operation
        let start = Instant::now();
        let result = engine.resize_bilinear(&large_image.view(), 128, 128);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Large image resize should succeed");
        assert!(
            elapsed.as_millis() < 5000,
            "Resize should complete in reasonable time"
        );

        // Verify cores were utilized efficiently
        let cores_used = engine.cores_used();
        let available_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        // Should use at least some cores, but not necessarily all (depends on workload)
        assert!(cores_used > 0, "Should use at least one core");
        assert!(
            cores_used <= available_cores,
            "Should not use more cores than available"
        );
    }

    #[test]
    fn test_x86_integration_with_batch_functions() {
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
    use super::*;

    #[test]
    fn test_resize_image_array() {
        let img = create_test_image();
        let result = resize_image_array(&img.view(), 64, 64).unwrap();
        assert_eq!(result.dim(), (64, 64, 3));
    }

    #[test]
    fn test_resize_video_array() {
        let video = create_test_video();
        let result = resize_video_array(&video.view(), 32, 32).unwrap();
        assert_eq!(result.dim(), (5, 32, 32, 3));
    }

    #[test]
    fn test_resize_upscale() {
        let img = Array3::from_shape_fn((10, 10, 3), |(y, x, c)| (x + y + c) as u8);
        let result = resize_image_array(&img.view(), 20, 20).unwrap();
        assert_eq!(result.dim(), (20, 20, 3));
    }

    #[test]
    fn test_resize_downscale() {
        let img = create_test_image();
        let result = resize_image_array(&img.view(), 25, 25).unwrap();
        assert_eq!(result.dim(), (25, 25, 3));
    }
}

#[cfg(test)]
mod batch_tests {
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
    use super::*;

    #[test]
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
    fn test_resize_same_size() {
        let img = create_test_image();
        let result = resize_image_array(&img.view(), 100, 100).unwrap();
        assert_eq!(result.dim(), (100, 100, 3));
    }
}
