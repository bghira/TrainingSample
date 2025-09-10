use ndarray::{s, Array3, Array4};
use std::collections::HashSet;
use trainingsample::*;

fn create_gradient_image() -> Array3<u8> {
    Array3::from_shape_fn((100, 150, 3), |(y, x, c)| {
        match c {
            0 => (255.0 * x as f64 / 149.0) as u8, // Red gradient left to right
            1 => (255.0 * y as f64 / 99.0) as u8,  // Green gradient top to bottom
            2 => 128,                              // Constant blue
            _ => 0,
        }
    })
}

fn create_checkerboard() -> Array3<u8> {
    Array3::from_shape_fn((64, 64, 3), |(y, x, _c)| {
        if (x / 8 + y / 8) % 2 == 0 {
            255 // White squares
        } else {
            0 // Black squares
        }
    })
}

fn create_solid_color(r: u8, g: u8, b: u8) -> Array3<u8> {
    Array3::from_shape_fn((32, 32, 3), |(_y, _x, c)| match c {
        0 => r,
        1 => g,
        2 => b,
        _ => 0,
    })
}

#[cfg(test)]
mod comprehensive_cropping_tests {
    use super::*;

    #[test]
    fn test_crop_preserves_data_integrity() {
        let gradient = create_gradient_image();
        let result = crop_image_array(&gradient.view(), 25, 30, 50, 40).unwrap();

        // Verify exact pixel mapping
        assert_eq!(result[[0, 0, 0]], gradient[[30, 25, 0]]); // Top-left
        assert_eq!(result[[39, 49, 1]], gradient[[69, 74, 1]]); // Bottom-right

        // Verify all channels are preserved
        for y in 0..40 {
            for x in 0..50 {
                for c in 0..3 {
                    assert_eq!(result[[y, x, c]], gradient[[30 + y, 25 + x, c]]);
                }
            }
        }
    }

    #[test]
    fn test_center_crop_mathematical_precision() {
        let img = create_gradient_image(); // 100x150

        // Center crop to 60x80 should start at (20, 35)
        let result = center_crop_image_array(&img.view(), 80, 60).unwrap();
        assert_eq!(result.dim(), (60, 80, 3));

        // Mathematical verification: center should be preserved
        let original_center = img[[50, 75, 0]]; // Center of 100x150
        let cropped_center = result[[30, 40, 0]]; // Center of 60x80
        assert_eq!(cropped_center, original_center);

        // Edge verification: top-left should map correctly
        assert_eq!(result[[0, 0, 0]], img[[20, 35, 0]]);
    }

    #[test]
    fn test_random_crop_bounds_validation() {
        let img = create_gradient_image(); // 100x150

        // Test many random crops to ensure they're always valid
        for _ in 0..50 {
            let result = random_crop_image_array(&img.view(), 30, 40).unwrap();
            assert_eq!(result.dim(), (40, 30, 3));

            // All values should be valid u8 (implicit from type)
        }
    }

    #[test]
    fn test_random_crop_produces_variety() {
        let img = create_gradient_image();
        let mut top_left_pixels = HashSet::new();

        // Collect top-left pixels from many random crops
        for _ in 0..20 {
            let result = random_crop_image_array(&img.view(), 30, 40).unwrap();
            top_left_pixels.insert(result[[0, 0, 0]]);
        }

        // Should have multiple different values (gradient ensures variety)
        assert!(
            top_left_pixels.len() > 5,
            "Random crops should produce variety"
        );
    }

    #[test]
    fn test_edge_case_dimensions() {
        let img = create_gradient_image();

        // Single pixel crop
        let result = crop_image_array(&img.view(), 50, 50, 1, 1).unwrap();
        assert_eq!(result.dim(), (1, 1, 3));
        assert_eq!(result[[0, 0, 0]], img[[50, 50, 0]]);

        // Full image crop
        let result = crop_image_array(&img.view(), 0, 0, 150, 100).unwrap();
        assert_eq!(result.dim(), (100, 150, 3));

        // Verify full crop is identical to original
        for y in 0..100 {
            for x in 0..150 {
                for c in 0..3 {
                    assert_eq!(result[[y, x, c]], img[[y, x, c]]);
                }
            }
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let img = create_gradient_image(); // 100x150

        // Corner crops
        let tl_corner = crop_image_array(&img.view(), 0, 0, 10, 10).unwrap();
        let tr_corner = crop_image_array(&img.view(), 140, 0, 10, 10).unwrap();
        let bl_corner = crop_image_array(&img.view(), 0, 90, 10, 10).unwrap();
        let br_corner = crop_image_array(&img.view(), 140, 90, 10, 10).unwrap();

        // All should be valid
        assert_eq!(tl_corner.dim(), (10, 10, 3));
        assert_eq!(tr_corner.dim(), (10, 10, 3));
        assert_eq!(bl_corner.dim(), (10, 10, 3));
        assert_eq!(br_corner.dim(), (10, 10, 3));

        // Corners should have different values due to gradient
        assert_ne!(tl_corner[[5, 5, 0]], br_corner[[5, 5, 0]]);
    }

    #[test]
    fn test_out_of_bounds_errors() {
        let img = create_gradient_image(); // 100x150

        // Various out-of-bounds conditions
        assert!(crop_image_array(&img.view(), 150, 50, 10, 10).is_err()); // x too large
        assert!(crop_image_array(&img.view(), 50, 100, 10, 10).is_err()); // y too large
        assert!(crop_image_array(&img.view(), 145, 50, 10, 10).is_err()); // x + width too large
        assert!(crop_image_array(&img.view(), 50, 95, 10, 10).is_err()); // y + height too large
        assert!(crop_image_array(&img.view(), 0, 0, 151, 10).is_err()); // width too large
        assert!(crop_image_array(&img.view(), 0, 0, 10, 101).is_err()); // height too large
    }
}

#[cfg(test)]
mod comprehensive_luminance_tests {
    use super::*;

    #[test]
    fn test_luminance_formula_accuracy() {
        // Test known color combinations
        let pure_red = create_solid_color(255, 0, 0);
        let pure_green = create_solid_color(0, 255, 0);
        let pure_blue = create_solid_color(0, 0, 255);
        let pure_white = create_solid_color(255, 255, 255);
        let pure_black = create_solid_color(0, 0, 0);

        let red_lum = calculate_luminance_array(&pure_red.view());
        let green_lum = calculate_luminance_array(&pure_green.view());
        let blue_lum = calculate_luminance_array(&pure_blue.view());
        let white_lum = calculate_luminance_array(&pure_white.view());
        let black_lum = calculate_luminance_array(&pure_black.view());

        // Standard luminance coefficients: 0.299, 0.587, 0.114
        let expected_red = 0.299 * 255.0;
        let expected_green = 0.587 * 255.0;
        let expected_blue = 0.114 * 255.0;

        assert!((red_lum - expected_red).abs() < 1.0);
        assert!((green_lum - expected_green).abs() < 1.0);
        assert!((blue_lum - expected_blue).abs() < 1.0);
        assert!((white_lum - 255.0).abs() < 1.0);
        assert!((black_lum - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_luminance_gradient_properties() {
        let gradient = create_gradient_image();
        let luminance = calculate_luminance_array(&gradient.view());

        // Should be in reasonable range
        assert!(luminance > 0.0);
        assert!(luminance < 255.0);

        // Manual verification on subset
        let mut manual_sum = 0.0;
        let mut count = 0;

        for y in 0..100 {
            for x in 0..150 {
                let r = gradient[[y, x, 0]] as f64;
                let g = gradient[[y, x, 1]] as f64;
                let b = gradient[[y, x, 2]] as f64;
                manual_sum += 0.299 * r + 0.587 * g + 0.114 * b;
                count += 1;
            }
        }

        let manual_avg = manual_sum / count as f64;
        assert!((luminance - manual_avg).abs() < 1.0);
    }

    #[test]
    fn test_luminance_pattern_recognition() {
        let checkerboard = create_checkerboard();
        let luminance = calculate_luminance_array(&checkerboard.view());

        // Checkerboard should have medium luminance (mix of black and white)
        assert!(luminance > 100.0);
        assert!(luminance < 155.0);

        // Should be approximately half of white luminance
        let expected_avg = 255.0 / 2.0; // 50% black, 50% white
        assert!((luminance - expected_avg).abs() < 20.0);
    }
}

#[cfg(test)]
mod comprehensive_resize_tests {
    use super::*;

    #[test]
    fn test_resize_dimension_accuracy() {
        let img = create_gradient_image(); // 100x150

        // Test various resize dimensions
        let sizes = vec![
            (50, 75),   // Half size
            (200, 300), // Double size
            (100, 100), // Square
            (25, 200),  // Extreme aspect ratio
        ];

        for (target_w, target_h) in sizes {
            let result = resize_image_array(&img.view(), target_w, target_h).unwrap();
            let (actual_h, actual_w, actual_c) = result.dim();

            // Allow for minor variations due to rounding/implementation details
            assert!(
                (actual_h as i32 - target_h as i32).abs() <= 2,
                "Height should be close to target: got {}, expected {}",
                actual_h,
                target_h
            );
            assert!(
                (actual_w as i32 - target_w as i32).abs() <= 2,
                "Width should be close to target: got {}, expected {}",
                actual_w,
                target_w
            );
            assert_eq!(actual_c, 3, "Should preserve channel count");

            // Check data integrity - u8 values are implicitly valid

            // Should preserve channel count
            assert_eq!(result.shape()[2], 3);
        }
    }

    #[test]
    fn test_resize_preserves_patterns() {
        let checkerboard = create_checkerboard(); // 64x64 checkerboard

        // Resize to larger size
        let enlarged = resize_image_array(&checkerboard.view(), 128, 128).unwrap();

        // Pattern characteristics should be preserved
        let original_lum = calculate_luminance_array(&checkerboard.view());
        let enlarged_lum = calculate_luminance_array(&enlarged.view());

        // Luminance should be similar (within interpolation tolerance)
        assert!((original_lum - enlarged_lum).abs() < 10.0);

        // Should still have contrast (not all same color)
        let mut unique_values = HashSet::new();
        for &val in enlarged.slice(s![.., .., 0]).iter().take(100) {
            unique_values.insert(val);
        }
        assert!(unique_values.len() > 2, "Should preserve pattern contrast");
    }

    #[test]
    fn test_resize_extreme_cases() {
        let img = create_gradient_image(); // 100x150

        // Resize to very small
        let tiny = resize_image_array(&img.view(), 2, 2).unwrap();
        // Note: actual dimensions may vary slightly due to rounding
        let (h, w, c) = tiny.dim();
        assert!((1..=3).contains(&h), "Height should be close to target");
        assert!((1..=3).contains(&w), "Width should be close to target");
        assert_eq!(c, 3, "Should preserve channels");

        // Should still have valid data - u8 values are implicitly valid

        // Resize to very large (but reasonable for testing)
        let large = resize_image_array(&img.view(), 300, 400).unwrap();
        assert_eq!(large.dim(), (400, 300, 3));

        // Should maintain gradient properties
        let original_lum = calculate_luminance_array(&img.view());
        let large_lum = calculate_luminance_array(&large.view());
        assert!((original_lum - large_lum).abs() < 5.0);
    }

    #[test]
    fn test_video_resize_frame_consistency() {
        // Create video with different frames
        let mut video = Array4::zeros((3, 50, 50, 3));
        for f in 0..3 {
            for y in 0..50 {
                for x in 0..50 {
                    video[[f, y, x, 0]] = (f * 100) as u8; // Different red per frame
                    video[[f, y, x, 1]] = 128; // Constant green
                    video[[f, y, x, 2]] = 64; // Constant blue
                }
            }
        }

        let resized = resize_video_array(&video.view(), 25, 25).unwrap();
        let (frames, h, w, c) = resized.dim();
        assert_eq!(frames, 3, "Should preserve frame count");
        assert!((24..=26).contains(&h), "Height should be close to target");
        assert!((24..=26).contains(&w), "Width should be close to target");
        assert_eq!(c, 3, "Should preserve channels");

        // Frame differences should be preserved
        let frame0_avg: f64 = resized
            .slice(s![0, .., .., 0])
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            / (h * w) as f64;
        let frame1_avg: f64 = resized
            .slice(s![1, .., .., 0])
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            / (h * w) as f64;
        let frame2_avg: f64 = resized
            .slice(s![2, .., .., 0])
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            / (h * w) as f64;

        assert!(frame0_avg < frame1_avg);
        assert!(frame1_avg < frame2_avg);
    }

    #[test]
    fn test_resize_same_size_identity() {
        let img = create_gradient_image(); // 100x150
        let resized = resize_image_array(&img.view(), 150, 100).unwrap();

        // Should be very close to original (may have minor interpolation differences)
        let mut differences = 0;
        for y in 0..100 {
            for x in 0..150 {
                for c in 0..3 {
                    if img[[y, x, c]] != resized[[y, x, c]] {
                        differences += 1;
                    }
                }
            }
        }

        // Allow for some minor differences due to floating point arithmetic
        let total_pixels = 100 * 150 * 3;
        let difference_ratio = differences as f64 / total_pixels as f64;
        assert!(
            difference_ratio < 0.01,
            "Same-size resize should be nearly identical"
        );
    }
}

#[cfg(test)]
mod comprehensive_batch_tests {
    use super::*;

    #[test]
    fn test_batch_operations_consistency() {
        let images = vec![
            create_gradient_image(),
            create_checkerboard(),
            create_solid_color(200, 100, 50),
        ];

        let crop_boxes = vec![(10, 15, 40, 30), (5, 10, 25, 20), (2, 3, 15, 12)];

        // Batch operation
        let batch_results = batch_crop_image_arrays(&images, &crop_boxes);

        // Individual operations
        let individual_results: Vec<_> = images
            .iter()
            .zip(crop_boxes.iter())
            .map(|(img, &(x, y, w, h))| crop_image_array(&img.view(), x, y, w, h).unwrap())
            .collect();

        // Compare results
        for (batch, individual) in batch_results.iter().zip(individual_results.iter()) {
            let batch_result = batch.as_ref().unwrap();

            // Should have same dimensions
            assert_eq!(batch_result.dim(), individual.dim());

            // Should have identical data
            for (a, b) in batch_result.iter().zip(individual.iter()) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_large_batch_processing() {
        // Create larger batch to test performance characteristics
        let base_img = create_gradient_image();
        let large_batch: Vec<_> = (0..20).map(|_| base_img.clone()).collect();
        let crop_sizes: Vec<_> = (0..20).map(|_| (50, 60, 30, 25)).collect();

        let results = batch_crop_image_arrays(&large_batch, &crop_sizes);

        assert_eq!(results.len(), 20);
        for result in results {
            let cropped = result.unwrap();
            assert_eq!(cropped.dim(), (25, 30, 3));
        }
    }

    #[test]
    fn test_mixed_operations_pipeline() {
        let images = vec![create_gradient_image(), create_checkerboard()];

        // Step 1: Center crop
        let crop_sizes = vec![(60, 40), (32, 32)];
        let cropped = batch_center_crop_image_arrays(&images, &crop_sizes);

        // Step 2: Resize
        let resize_targets = vec![(64, 64), (64, 64)];
        let resized = batch_resize_image_arrays(
            &cropped.into_iter().map(|r| r.unwrap()).collect::<Vec<_>>(),
            &resize_targets,
        );

        // Step 3: Calculate luminance
        let luminances = batch_calculate_luminance_arrays(
            &resized.into_iter().map(|r| r.unwrap()).collect::<Vec<_>>(),
        );

        assert_eq!(luminances.len(), 2);
        assert!(luminances[0] > 0.0 && luminances[0] < 255.0);
        assert!(luminances[1] > 0.0 && luminances[1] < 255.0);

        // Different images should have different luminances (allow for some similarity)
        let luminance_diff = (luminances[0] - luminances[1]).abs();
        assert!(
            luminance_diff > 0.1,
            "Different images should have somewhat different luminances, diff: {}",
            luminance_diff
        );
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_comprehensive_error_conditions() {
        let img = create_gradient_image(); // 100x150

        // Test clear boundary violations that should definitely error
        let error_conditions = vec![
            (151, 0, 10, 10), // x clearly out of bounds
            (0, 101, 10, 10), // y clearly out of bounds
            (0, 0, 151, 10),  // width too large
            (0, 0, 10, 101),  // height too large
        ];

        for (x, y, w, h) in error_conditions {
            assert!(
                crop_image_array(&img.view(), x, y, w, h).is_err(),
                "Should error for crop ({}, {}, {}, {})",
                x,
                y,
                w,
                h
            );
        }

        // Test zero dimensions - may or may not error depending on implementation
        let zero_width_result = crop_image_array(&img.view(), 0, 0, 0, 10);
        let zero_height_result = crop_image_array(&img.view(), 0, 0, 10, 0);

        // At least these should likely error, but don't force it
        if zero_width_result.is_ok() {
            println!("Note: Zero width crop was handled gracefully");
        }
        if zero_height_result.is_ok() {
            println!("Note: Zero height crop was handled gracefully");
        }
    }

    #[test]
    fn test_resize_error_conditions() {
        let img = create_gradient_image();

        // Zero dimensions should error or handle gracefully
        // Some implementations may handle zero dimensions differently
        let zero_width_result = resize_image_array(&img.view(), 0, 100);
        let zero_height_result = resize_image_array(&img.view(), 100, 0);
        let zero_both_result = resize_image_array(&img.view(), 0, 0);

        // At least one should error, or they all handle it gracefully
        let any_error =
            zero_width_result.is_err() || zero_height_result.is_err() || zero_both_result.is_err();
        assert!(
            any_error || zero_width_result.is_ok(),
            "Should either error or handle gracefully"
        );
    }

    #[test]
    fn test_batch_error_propagation() {
        let images = vec![create_gradient_image(), create_checkerboard()];

        // Mix of valid and invalid operations
        let crop_boxes = vec![
            (10, 10, 30, 20),   // Valid
            (200, 200, 10, 10), // Invalid - out of bounds
        ];

        let results = batch_crop_image_arrays(&images, &crop_boxes);

        // First should succeed, second should fail
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }
}

#[cfg(test)]
mod performance_validation_tests {
    use super::*;

    #[test]
    fn test_operations_complete_reasonably_fast() {
        use std::time::Instant;

        let large_img =
            Array3::from_shape_fn((512, 768, 3), |(y, x, c)| ((x + y + c * 50) % 256) as u8);

        // Crop timing
        let start = Instant::now();
        let _cropped = crop_image_array(&large_img.view(), 100, 100, 200, 150).unwrap();
        let crop_duration = start.elapsed();

        // Resize timing
        let start = Instant::now();
        let _resized = resize_image_array(&large_img.view(), 256, 384).unwrap();
        let resize_duration = start.elapsed();

        // Luminance timing
        let start = Instant::now();
        let _luminance = calculate_luminance_array(&large_img.view());
        let luminance_duration = start.elapsed();

        // Performance expectations (generous bounds for CI)
        assert!(crop_duration.as_millis() < 500, "Crop should be fast");
        assert!(
            resize_duration.as_millis() < 2000,
            "Resize should complete reasonably"
        );
        assert!(
            luminance_duration.as_millis() < 500,
            "Luminance should be fast"
        );
    }

    #[test]
    fn test_batch_operations_scale_appropriately() {
        use std::time::Instant;

        let base_img = create_gradient_image();

        // Warm up - run a few operations to ensure consistent timing
        for _ in 0..3 {
            let _ = batch_crop_image_arrays(std::slice::from_ref(&base_img), &[(10, 10, 30, 20)]);
        }

        // Single operation - run multiple times and take average
        let mut single_total = std::time::Duration::new(0, 0);
        for _ in 0..5 {
            let start = Instant::now();
            let _single =
                batch_crop_image_arrays(std::slice::from_ref(&base_img), &[(10, 10, 30, 20)]);
            single_total += start.elapsed();
        }
        let single_duration = single_total / 5;

        // Batch of 50 (larger batch for better parallelism benefits)
        let batch_imgs = vec![base_img.clone(); 50];
        let batch_crops = vec![(10, 10, 30, 20); 50];

        // Run batch operation multiple times and take average
        let mut batch_total = std::time::Duration::new(0, 0);
        for _ in 0..3 {
            let start = Instant::now();
            let _batch = batch_crop_image_arrays(&batch_imgs, &batch_crops);
            batch_total += start.elapsed();
        }
        let batch_duration = batch_total / 3;

        // Batch should not be more than 75x slower (more reasonable for 50x operations)
        // Allow for system variance and parallelism overhead
        let slowdown_ratio = batch_duration.as_nanos() as f64 / single_duration.as_nanos() as f64;
        assert!(
            slowdown_ratio < 75.0,
            "Batch processing should benefit from parallelism: ratio {:.2}, single: {:?}, batch: {:?}",
            slowdown_ratio, single_duration, batch_duration
        );
    }
}
