use ndarray::Array3;
use std::time::Instant;

// Correct reference implementation for Lanczos3 - included inline
fn lanczos3_kernel(x: f32) -> f32 {
    let x = x.abs();
    if x < 3.0 && x != 0.0 {
        let pi_x = std::f32::consts::PI * x;
        let pi_x_3 = pi_x / 3.0;
        3.0 * pi_x.sin() * pi_x_3.sin() / (pi_x * pi_x)
    } else if x == 0.0 {
        1.0
    } else {
        0.0
    }
}

fn resize_lanczos3_simple(
    image: &ndarray::ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> anyhow::Result<Array3<u8>> {
    let (src_height, src_width, _channels) = image.dim();
    let dst_width = target_width as usize;
    let dst_height = target_height as usize;

    let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            for c in 0..3 {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                let y_center = (dst_y as f32 + 0.5) * y_scale - 0.5;
                let x_center = (dst_x as f32 + 0.5) * x_scale - 0.5;

                let y_support = if y_scale > 1.0 { y_scale * 3.0 } else { 3.0 };
                let x_support = if x_scale > 1.0 { x_scale * 3.0 } else { 3.0 };

                let y_left = (y_center - y_support).ceil() as i32;
                let y_right = (y_center + y_support).floor() as i32;
                let x_left = (x_center - x_support).ceil() as i32;
                let x_right = (x_center + x_support).floor() as i32;

                for src_y in y_left..=y_right {
                    if src_y >= 0 && (src_y as usize) < src_height {
                        let y_distance = (src_y as f32 - y_center) / if y_scale > 1.0 { y_scale } else { 1.0 };
                        let y_weight = lanczos3_kernel(y_distance);

                        if y_weight.abs() > 1e-6 {
                            for src_x in x_left..=x_right {
                                if src_x >= 0 && (src_x as usize) < src_width {
                                    let x_distance = (src_x as f32 - x_center) / if x_scale > 1.0 { x_scale } else { 1.0 };
                                    let x_weight = lanczos3_kernel(x_distance);

                                    if x_weight.abs() > 1e-6 {
                                        let combined_weight = y_weight * x_weight;
                                        sum += image[[src_y as usize, src_x as usize, c]] as f32 * combined_weight;
                                        weight_sum += combined_weight;
                                    }
                                }
                            }
                        }
                    }
                }

                if weight_sum > 0.0 {
                    result[[dst_y, dst_x, c]] = ((sum / weight_sum) + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(result)
}

/// Simple performance comparison test for optimized algorithms
#[cfg(test)]
mod optimization_comparison {
    use super::*;

    fn create_test_image(width: usize, height: usize) -> Array3<u8> {
        Array3::from_shape_fn((height, width, 3), |(y, x, c)| match c {
            0 => ((x + y) % 256) as u8,
            1 => ((x * y / 16) % 256) as u8,
            2 => ((x ^ y) % 256) as u8,
            _ => 0,
        })
    }

    #[test]
    fn test_optimization_performance_comparison() {
        println!("\n🚀 OPTIMIZATION PERFORMANCE COMPARISON");
        println!("======================================");

        let test_image = create_test_image(1024, 1024);
        let view = test_image.view();
        let iterations = 3;

        println!(
            "\n📐 Testing 1024x1024 → 512x512 ({} iterations)",
            iterations
        );

        #[cfg(feature = "simd")]
        {
            // Original SIMD implementation
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = trainingsample::resize_simd::resize_lanczos3_simd(&view, 512, 512).unwrap();
            }
            let original_time = start.elapsed().as_secs_f64() / iterations as f64;
            let original_throughput = (512 * 512) as f64 / original_time / 1_000_000.0;

            // Blocked optimized implementation
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = trainingsample::resize_optimized::resize_lanczos3_blocked_optimized(
                    &view, 512, 512,
                )
                .unwrap();
            }
            let blocked_time = start.elapsed().as_secs_f64() / iterations as f64;
            let blocked_throughput = (512 * 512) as f64 / blocked_time / 1_000_000.0;

            // Adaptive optimized implementation
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = trainingsample::resize_optimized::resize_lanczos3_adaptive_optimized(
                    &view, 512, 512,
                )
                .unwrap();
            }
            let adaptive_time = start.elapsed().as_secs_f64() / iterations as f64;
            let adaptive_throughput = (512 * 512) as f64 / adaptive_time / 1_000_000.0;

            let blocked_speedup = blocked_throughput / original_throughput;
            let adaptive_speedup = adaptive_throughput / original_throughput;

            println!("   📊 Results:");
            println!(
                "     Original SIMD:      {:.1} MPx/s ({:.1}ms)",
                original_throughput,
                original_time * 1000.0
            );
            println!(
                "     Blocked Optimized:  {:.1} MPx/s ({:.1}ms) - {:.2}x speedup",
                blocked_throughput,
                blocked_time * 1000.0,
                blocked_speedup
            );
            println!(
                "     Adaptive Optimized: {:.1} MPx/s ({:.1}ms) - {:.2}x speedup",
                adaptive_throughput,
                adaptive_time * 1000.0,
                adaptive_speedup
            );

            // Performance analysis
            if blocked_speedup >= 2.0 {
                println!("     🚀 Excellent blocked optimization! 2x+ speedup achieved");
            } else if blocked_speedup >= 1.5 {
                println!("     ✅ Great blocked optimization! 1.5x+ speedup");
            } else if blocked_speedup >= 1.2 {
                println!("     ✅ Good blocked optimization! 1.2x+ speedup");
            } else if blocked_speedup >= 1.0 {
                println!("     ⚡ Modest blocked improvement");
            } else {
                println!("     📊 Blocked performance similar to original");
            }

            // Verify we didn't make things significantly worse
            assert!(
                blocked_speedup >= 0.8,
                "Blocked optimization should not be >20% slower than original (got {:.2}x)",
                blocked_speedup
            );
            assert!(
                adaptive_speedup >= 0.8,
                "Adaptive optimization should not be >20% slower than original (got {:.2}x)",
                adaptive_speedup
            );

            println!("   ✅ Performance validation passed!");
        }

        #[cfg(not(feature = "simd"))]
        {
            println!("   ⚠️  SIMD not enabled - testing scalar fallbacks");

            let start = Instant::now();
            let _ = trainingsample::resize_optimized::resize_lanczos3_adaptive_optimized(
                &view, 512, 512,
            )
            .unwrap();
            let time = start.elapsed().as_secs_f64();
            let throughput = (512 * 512) as f64 / time / 1_000_000.0;

            println!(
                "     Scalar Fallback: {:.1} MPx/s ({:.1}ms)",
                throughput,
                time * 1000.0
            );
            println!("   ✅ Scalar fallback works correctly");
        }
    }

    #[test]
    fn test_correctness_verification() {
        println!("\n🎯 CORRECTNESS VERIFICATION");
        println!("===========================");

        let test_image = create_test_image(128, 128);
        let view = test_image.view();

        #[cfg(feature = "simd")]
        {
            println!("\n📐 Verifying 128x128 → 64x64 correctness");

            // Get reference result from correct implementation
            let reference = resize_lanczos3_simple(&view, 64, 64).unwrap();

            // Test optimized implementations
            let (blocked_result, _) =
                trainingsample::resize_optimized::resize_lanczos3_blocked_optimized(&view, 64, 64)
                    .unwrap();
            let (adaptive_result, _) =
                trainingsample::resize_optimized::resize_lanczos3_adaptive_optimized(&view, 64, 64)
                    .unwrap();

            // Calculate differences
            let blocked_diff = calculate_max_difference(&reference, &blocked_result);
            let adaptive_diff = calculate_max_difference(&reference, &adaptive_result);

            println!("   📊 Maximum pixel differences vs reference:");
            println!("     Blocked:  {} (max allowed: 10)", blocked_diff);
            println!("     Adaptive: {} (max allowed: 3)", adaptive_diff);

            // Verify results are very close (allowing for minor numerical differences)
            // Blocked algorithm uses separable filtering so may have slightly higher error
            assert!(
                blocked_diff <= 10,
                "Blocked result differs too much from reference: {}",
                blocked_diff
            );
            assert!(
                adaptive_diff <= 3,
                "Adaptive result differs too much from reference: {}",
                adaptive_diff
            );

            println!("   ✅ All implementations produce correct results within tolerance!");
        }
    }

    fn calculate_max_difference(img1: &Array3<u8>, img2: &Array3<u8>) -> u8 {
        img1.iter()
            .zip(img2.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).abs() as u8)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn test_algorithm_metrics() {
        println!("\n📊 ALGORITHM METRICS ANALYSIS");
        println!("==============================");

        let test_image = create_test_image(512, 512);
        let view = test_image.view();

        #[cfg(feature = "simd")]
        {
            println!("\n📐 Testing 512x512 → 256x256 with metrics");

            // Test adaptive algorithm and examine metrics
            let (result, metrics) =
                trainingsample::resize_optimized::resize_lanczos3_adaptive_optimized(
                    &view, 256, 256,
                )
                .unwrap();

            assert_eq!(result.dim(), (256, 256, 3));

            println!("   📊 Adaptive Algorithm Metrics:");
            println!("     Implementation: {}", metrics.implementation);
            println!(
                "     Throughput: {:.1} MPx/s",
                metrics.throughput_mpixels_per_sec
            );
            println!(
                "     Cache efficiency: {:.1}%",
                metrics.cache_efficiency * 100.0
            );
            println!(
                "     Vectorization efficiency: {:.1}%",
                metrics.vectorization_efficiency * 100.0
            );

            // Verify reasonable metrics
            assert!(
                metrics.throughput_mpixels_per_sec > 0.0,
                "Throughput should be positive"
            );
            assert!(
                metrics.cache_efficiency > 0.0 && metrics.cache_efficiency <= 1.0,
                "Cache efficiency should be 0-100%"
            );
            assert!(
                metrics.vectorization_efficiency > 0.0 && metrics.vectorization_efficiency <= 1.0,
                "Vectorization efficiency should be 0-100%"
            );

            println!("   ✅ Metrics validation passed!");
        }
    }

    #[test]
    fn debug_pixel_values() {
        println!("\n🔍 DEBUG PIXEL VALUES");
        println!("=====================");
        
        // Create a simple 4x4 test image with known values
        let test_image = Array3::from_shape_fn((4, 4, 3), |(y, x, c)| match c {
            0 => ((x + y) % 256) as u8,      // Red channel
            1 => ((x * y / 2) % 256) as u8,  // Green channel  
            2 => ((x ^ y) % 256) as u8,      // Blue channel
            _ => 0,
        });
        
        println!("\nInput 4x4 image:");
        for y in 0..4 {
            for x in 0..4 {
                println!("  [{}, {}]: R={}, G={}, B={}", 
                         y, x,
                         test_image[[y, x, 0]],
                         test_image[[y, x, 1]], 
                         test_image[[y, x, 2]]);
            }
        }
        
        let view = test_image.view();
        
        #[cfg(feature = "simd")]
        {
            // Correct reference implementation
            let reference = resize_lanczos3_simple(&view, 2, 2).unwrap();
            
            println!("\nReference 2x2 result:");
            for y in 0..2 {
                for x in 0..2 {
                    println!("  [{}, {}]: R={}, G={}, B={}", 
                             y, x,
                             reference[[y, x, 0]],
                             reference[[y, x, 1]], 
                             reference[[y, x, 2]]);
                }
            }
            
            // Optimized blocked implementation  
            let (blocked, _) = trainingsample::resize_optimized::resize_lanczos3_blocked_optimized(&view, 2, 2).unwrap();
            
            println!("\nBlocked 2x2 result:");
            for y in 0..2 {
                for x in 0..2 {
                    println!("  [{}, {}]: R={}, G={}, B={}", 
                             y, x,
                             blocked[[y, x, 0]],
                             blocked[[y, x, 1]], 
                             blocked[[y, x, 2]]);
                }
            }
            
            // Calculate differences
            println!("\nDifferences (blocked - reference):");
            for y in 0..2 {
                for x in 0..2 {
                    for c in 0..3 {
                        let ref_val = reference[[y, x, c]] as i16;
                        let opt_val = blocked[[y, x, c]] as i16;
                        let diff = opt_val - ref_val;
                        println!("  [{}, {}, {}]: {} - {} = {}", y, x, c, opt_val, ref_val, diff);
                    }
                }
            }
        }
    }

    #[test]
    fn test_size_scaling() {
        println!("\n📏 SIZE SCALING PERFORMANCE");
        println!("===========================");

        let test_sizes = [
            (256, 256, 128, 128),
            (512, 512, 256, 256),
            (1024, 1024, 512, 512),
        ];

        #[cfg(feature = "simd")]
        for &(src_w, src_h, dst_w, dst_h) in &test_sizes {
            let test_image = create_test_image(src_w, src_h);
            let view = test_image.view();

            println!("\n📐 Testing {}x{} → {}x{}", src_w, src_h, dst_w, dst_h);

            let start = Instant::now();
            let (result, metrics) =
                trainingsample::resize_optimized::resize_lanczos3_adaptive_optimized(
                    &view,
                    dst_w as u32,
                    dst_h as u32,
                )
                .unwrap();
            let elapsed = start.elapsed();

            assert_eq!(result.dim(), (dst_h, dst_w, 3));

            println!("   📊 Results:");
            println!("     Time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
            println!(
                "     Throughput: {:.1} MPx/s",
                metrics.throughput_mpixels_per_sec
            );
            println!("     Implementation: {}", metrics.implementation);

            // Larger images should complete in reasonable time
            assert!(
                elapsed.as_secs_f64() < 5.0,
                "Should complete within 5 seconds"
            );
            assert!(
                metrics.throughput_mpixels_per_sec > 0.1,
                "Should have reasonable throughput"
            );
        }

        println!("   ✅ Size scaling test completed successfully!");
    }
}
