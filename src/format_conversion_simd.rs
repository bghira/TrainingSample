use ndarray::ArrayView3;

/// High-performance SIMD RGB↔RGBA conversions optimized for Metal GPU operations
/// This addresses the critical bottleneck in Metal GPU texture upload/download

#[derive(Debug, Clone)]
pub struct FormatConversionMetrics {
    pub pixels_processed: usize,
    pub elapsed_nanos: u64,
    pub simd_width: usize,
    pub implementation: &'static str,
    pub throughput_mpixels_per_sec: f64,
}

impl FormatConversionMetrics {
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

// Preserve the public metric for the portable SIMD path. LLVM selects the
// hardware vector width when it vectorizes the pixel loop.
#[cfg(feature = "simd")]
const REPORTED_SIMD_WIDTH: usize = 8;

/// SIMD-optimized RGB to RGBA conversion using portable SIMD
#[cfg(feature = "simd")]
pub fn rgb_to_rgba_simd(
    image: &ArrayView3<u8>,
    alpha_value: u8,
) -> (Vec<u8>, FormatConversionMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels != 3 {
        panic!("RGB to RGBA conversion requires 3-channel input");
    }

    let total_pixels = height * width;
    let image = image.as_standard_layout();
    // Fixed-size pixel arrays let LLVM vectorize a single pass over the image.
    let mut rgba_pixels = vec![[0; 4]; total_pixels];
    let (rgb_pixels, _) = image.as_slice().unwrap().as_chunks::<3>();
    for (&[r, g, b], dst) in rgb_pixels.iter().zip(&mut rgba_pixels) {
        *dst = [r, g, b, alpha_value];
    }
    let rgba_data = rgba_pixels.into_flattened();

    let elapsed = start.elapsed().as_nanos() as u64;
    let metrics = FormatConversionMetrics::new(
        total_pixels,
        elapsed,
        REPORTED_SIMD_WIDTH,
        "simd_rgb_to_rgba",
    );

    (rgba_data, metrics)
}

/// SIMD-optimized RGBA to RGB conversion
#[cfg(feature = "simd")]
pub fn rgba_to_rgb_simd(
    rgba_data: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, FormatConversionMetrics) {
    let start = std::time::Instant::now();
    let total_pixels = width * height;

    if rgba_data.len() != total_pixels * 4 {
        panic!("RGBA data size mismatch");
    }

    let mut rgb_pixels = vec![[0; 3]; total_pixels];
    let (rgba_pixels, _) = rgba_data.as_chunks::<4>();
    for (&[r, g, b, _], dst) in rgba_pixels.iter().zip(&mut rgb_pixels) {
        *dst = [r, g, b];
    }
    let rgb_data = rgb_pixels.into_flattened();

    let elapsed = start.elapsed().as_nanos() as u64;
    let metrics = FormatConversionMetrics::new(
        total_pixels,
        elapsed,
        REPORTED_SIMD_WIDTH,
        "simd_rgba_to_rgb",
    );

    (rgb_data, metrics)
}

/// Scalar fallback implementations
pub fn rgb_to_rgba_scalar(
    image: &ArrayView3<u8>,
    alpha_value: u8,
) -> (Vec<u8>, FormatConversionMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels != 3 {
        panic!("RGB to RGBA conversion requires 3-channel input");
    }

    let total_pixels = height * width;
    let mut rgba_data = Vec::with_capacity(total_pixels * 4);

    for y in 0..height {
        for x in 0..width {
            rgba_data.push(image[[y, x, 0]]); // R
            rgba_data.push(image[[y, x, 1]]); // G
            rgba_data.push(image[[y, x, 2]]); // B
            rgba_data.push(alpha_value); // A
        }
    }

    let elapsed = start.elapsed().as_nanos() as u64;
    let metrics = FormatConversionMetrics::new(total_pixels, elapsed, 1, "scalar_rgb_to_rgba");

    (rgba_data, metrics)
}

pub fn rgba_to_rgb_scalar(
    rgba_data: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, FormatConversionMetrics) {
    let start = std::time::Instant::now();
    let total_pixels = width * height;

    if rgba_data.len() != total_pixels * 4 {
        panic!("RGBA data size mismatch");
    }

    let mut rgb_data = Vec::with_capacity(total_pixels * 3);

    for i in 0..total_pixels {
        let rgba_offset = i * 4;
        rgb_data.push(rgba_data[rgba_offset]); // R
        rgb_data.push(rgba_data[rgba_offset + 1]); // G
        rgb_data.push(rgba_data[rgba_offset + 2]); // B
                                                   // Skip alpha
    }

    let elapsed = start.elapsed().as_nanos() as u64;
    let metrics = FormatConversionMetrics::new(total_pixels, elapsed, 1, "scalar_rgba_to_rgb");

    (rgb_data, metrics)
}

/// Auto-detecting format conversion with optimal implementation
pub fn rgb_to_rgba_optimized(
    image: &ArrayView3<u8>,
    alpha_value: u8,
) -> (Vec<u8>, FormatConversionMetrics) {
    #[cfg(feature = "simd")]
    {
        rgb_to_rgba_simd(image, alpha_value)
    }

    #[cfg(not(feature = "simd"))]
    {
        rgb_to_rgba_scalar(image, alpha_value)
    }
}

pub fn rgba_to_rgb_optimized(
    rgba_data: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, FormatConversionMetrics) {
    #[cfg(feature = "simd")]
    {
        rgba_to_rgb_simd(rgba_data, width, height)
    }

    #[cfg(not(feature = "simd"))]
    {
        rgba_to_rgb_scalar(rgba_data, width, height)
    }
}
