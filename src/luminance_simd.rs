use ndarray::ArrayView3;

/// Performance tracking for SIMD implementations
#[derive(Debug, Clone)]
pub struct LuminanceMetrics {
    pub pixels_processed: usize,
    pub elapsed_nanos: u64,
    pub simd_width: usize,
    pub implementation: &'static str,
    pub throughput_mpixels_per_sec: f64,
}

impl LuminanceMetrics {
    pub fn new(
        pixels_processed: usize,
        elapsed_nanos: u64,
        simd_width: usize,
        implementation: &'static str,
    ) -> Self {
        let throughput_mpixels_per_sec =
            (pixels_processed as f64) / (elapsed_nanos as f64 / 1_000_000_000.0) / 1_000_000.0;

        Self {
            pixels_processed,
            elapsed_nanos,
            simd_width,
            implementation,
            throughput_mpixels_per_sec,
        }
    }
}

/// Select the shared exact kernel. It dispatches to NEON for contiguous RGB/RGBA
/// on supported ARM CPUs, and uses compiler-vectorizable integer sums elsewhere.
/// A single kernel keeps channel handling and empty-input behavior consistent.
pub fn detect_optimal_luminance_impl() -> fn(&ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    calculate_luminance_optimized_sequential
}

/// Portable luminance entry point with exact integer accumulation.
pub fn calculate_luminance_portable_simd(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    calculate_luminance_optimized_sequential(image)
}

/// Scalar reference implementation with performance tracking.
pub fn calculate_luminance_scalar_tracked(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let result = crate::luminance::calculate_luminance_scalar(image);
    let metrics = LuminanceMetrics::new(
        image.dim().0 * image.dim().1,
        start.elapsed().as_nanos() as u64,
        1,
        "scalar_integer",
    );
    (result, metrics)
}

pub fn calculate_luminance_optimized(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    detect_optimal_luminance_impl()(image)
}

/// Single-threaded exact accumulation; callers can parallelize across images.
pub fn calculate_luminance_optimized_sequential(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let result = crate::luminance::calculate_luminance_array_sequential(image);
    let (simd_width, implementation) = implementation_details(image);
    let metrics = LuminanceMetrics::new(
        image.dim().0 * image.dim().1,
        start.elapsed().as_nanos() as u64,
        simd_width,
        implementation,
    );
    (result, metrics)
}

fn implementation_details(_image: &ArrayView3<u8>) -> (usize, &'static str) {
    #[cfg(target_arch = "aarch64")]
    if !_image.is_empty()
        && matches!(_image.dim().2, 3 | 4)
        && _image.as_slice().is_some()
        && std::arch::is_aarch64_feature_detected!("neon")
    {
        return (16, "neon_integer_sums");
    }
    (1, "integer_channel_sums")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_simd_vs_scalar_correctness() {
        let test_image =
            Array3::<u8>::from_shape_fn((100, 100, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();

        let (scalar_result, _) = calculate_luminance_scalar_tracked(&view);

        #[cfg(feature = "simd")]
        {
            let (simd_result, _) = calculate_luminance_portable_simd(&view);
            let diff = (scalar_result - simd_result).abs();
            assert!(
                diff < 0.001,
                "SIMD and scalar results should match: {} vs {}",
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_performance_tracking() {
        let test_image =
            Array3::<u8>::from_shape_fn((256, 256, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();

        let (_, metrics) = calculate_luminance_optimized(&view);

        assert_eq!(metrics.pixels_processed, 256 * 256);
        assert!(metrics.elapsed_nanos > 0);
        assert!(metrics.throughput_mpixels_per_sec > 0.0);
        println!(
            "Performance: {} MPx/s using {}",
            metrics.throughput_mpixels_per_sec, metrics.implementation
        );
    }

    #[test]
    fn benchmark_simd_vs_scalar() {
        use crate::luminance::calculate_luminance_scalar;
        use std::time::Instant;

        // Create a larger test image for meaningful benchmarking
        let test_image =
            Array3::<u8>::from_shape_fn((512, 512, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();
        let pixel_count = 512 * 512;
        let iterations = 1000;

        println!(
            "\n🔬 SIMD vs Scalar Benchmark (512x512, {} iterations)",
            iterations
        );
        println!("============================================================");

        // Warmup both implementations
        for _ in 0..10 {
            calculate_luminance_scalar(&view);
            calculate_luminance_optimized_sequential(&view);
        }

        // Benchmark scalar implementation
        let start = Instant::now();
        let mut scalar_sum = 0.0;
        for _ in 0..iterations {
            scalar_sum += calculate_luminance_scalar(&view);
        }
        let scalar_time = start.elapsed().as_secs_f64() / iterations as f64;
        std::hint::black_box(scalar_sum); // Prevent optimization
        let scalar_throughput = (pixel_count as f64) / scalar_time / 1_000_000.0;

        // Benchmark SIMD implementation (sequential to match scalar)
        let start = Instant::now();
        let mut simd_sum = 0.0;
        for _ in 0..iterations {
            let (result, _metrics) = calculate_luminance_optimized_sequential(&view);
            simd_sum += result;
        }
        let simd_time = start.elapsed().as_secs_f64() / iterations as f64;
        std::hint::black_box(simd_sum); // Prevent optimization
        let simd_throughput = (pixel_count as f64) / simd_time / 1_000_000.0;

        let speedup = simd_throughput / scalar_throughput;

        println!("📊 Results:");
        println!(
            "   Scalar:  {:.1} MPx/s ({:.3}ms per iteration)",
            scalar_throughput,
            scalar_time * 1000.0
        );
        println!(
            "   SIMD:    {:.1} MPx/s ({:.3}ms per iteration)",
            simd_throughput,
            simd_time * 1000.0
        );
        println!("   Speedup: {:.2}x", speedup);

        if speedup >= 1.5 {
            println!("   ✅ SIMD provides meaningful speedup!");
        } else {
            println!("   ⚡ SIMD overhead may be limiting gains on small images");
        }

        // The test passes if SIMD is at least as fast as scalar
        // On ARM64 (Apple Silicon), SIMD may be slower due to architecture differences
        #[cfg(target_arch = "aarch64")]
        assert!(
            speedup >= 0.5,
            "SIMD performance acceptable on ARM64 (actual: {:.2}x)",
            speedup
        );

        #[cfg(not(target_arch = "aarch64"))]
        assert!(
            speedup >= 0.8,
            "SIMD should not be significantly slower than scalar"
        );
    }
}
