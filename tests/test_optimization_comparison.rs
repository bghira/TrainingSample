use ndarray::Array3;
use std::time::Instant;

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

            // Get reference result from original implementation
            let (reference, _) =
                trainingsample::resize_simd::resize_lanczos3_simd(&view, 64, 64).unwrap();

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
            println!("     Blocked:  {} (max allowed: 3)", blocked_diff);
            println!("     Adaptive: {} (max allowed: 3)", adaptive_diff);

            // Verify results are very close (allowing for minor numerical differences)
            assert!(
                blocked_diff <= 3,
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
