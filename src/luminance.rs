use ndarray::ArrayView3;

#[cfg(feature = "simd")]
pub use crate::luminance_simd::{calculate_luminance_optimized, LuminanceMetrics};

/// Main luminance calculation function with automatic SIMD optimization
pub fn calculate_luminance_array(image: &ArrayView3<u8>) -> f64 {
    #[cfg(feature = "simd")]
    {
        let (result, _metrics) = calculate_luminance_optimized(image);
        result
    }

    #[cfg(not(feature = "simd"))]
    {
        calculate_luminance_scalar(image)
    }
}

/// Luminance calculation with performance metrics
pub fn calculate_luminance_with_metrics(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    #[cfg(feature = "simd")]
    {
        calculate_luminance_optimized(image)
    }

    #[cfg(not(feature = "simd"))]
    {
        let start = std::time::Instant::now();
        let result = calculate_luminance_scalar(image);
        let metrics = LuminanceMetrics::new(
            image.dim().0 * image.dim().1,
            start.elapsed().as_nanos() as u64,
            1,
            "scalar_fallback",
        );
        (result, metrics)
    }
}

/// Scalar implementation (fallback)
pub fn calculate_luminance_scalar(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels < 3 {
        // Grayscale or single channel - just average the values
        let sum: u64 = image.iter().map(|&x| x as u64).sum();
        return sum as f64 / (height * width * channels) as f64;
    }

    let mut total_luminance = 0.0;
    let pixel_count = height * width;

    for h in 0..height {
        for w in 0..width {
            let r = image[[h, w, 0]] as f64;
            let g = image[[h, w, 1]] as f64;
            let b = image[[h, w, 2]] as f64;

            // Standard RGB to luminance conversion
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            total_luminance += luminance;
        }
    }

    total_luminance / pixel_count as f64
}

#[cfg(not(feature = "simd"))]
#[derive(Debug, Clone)]
pub struct LuminanceMetrics {
    pub pixels_processed: usize,
    pub elapsed_nanos: u64,
    pub simd_width: usize,
    pub implementation: &'static str,
    pub throughput_mpixels_per_sec: f64,
}

#[cfg(not(feature = "simd"))]
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
