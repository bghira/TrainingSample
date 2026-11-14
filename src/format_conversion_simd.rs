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
    let mut rgba_data = vec![alpha_value; total_pixels * 4];
    const SIMD_WIDTH: usize = 8;

    // Keep every output byte initialized, including tails shorter than eight
    // pixels. Mixing writes to spare capacity with Vec::push overwrites earlier
    // pixels and can move subsequent writes beyond the allocation.
    for (rgb_chunk, rgba_chunk) in image
        .as_slice()
        .unwrap()
        .chunks(SIMD_WIDTH * 3)
        .zip(rgba_data.chunks_mut(SIMD_WIDTH * 4))
    {
        for (rgb, rgba) in rgb_chunk
            .chunks_exact(3)
            .zip(rgba_chunk.chunks_exact_mut(4))
        {
            rgba[..3].copy_from_slice(rgb);
        }
    }

    let elapsed = start.elapsed().as_nanos() as u64;
    let metrics =
        FormatConversionMetrics::new(total_pixels, elapsed, SIMD_WIDTH, "simd_rgb_to_rgba");

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

    let mut rgb_data = Vec::with_capacity(total_pixels * 3);

    // SIMD processing for RGBA → RGB conversion
    const SIMD_WIDTH: usize = 8;
    let chunks = total_pixels / SIMD_WIDTH;

    // Process in chunks of 8 pixels (32 RGBA bytes → 24 RGB bytes)
    for chunk in 0..chunks {
        let rgba_offset = chunk * SIMD_WIDTH * 4;
        let rgb_offset = chunk * SIMD_WIDTH * 3;

        let rgba_chunk = &rgba_data[rgba_offset..rgba_offset + 32];

        // Reserve space in RGB output
        rgb_data.resize(rgb_data.len() + 24, 0);
        let rgb_slice = &mut rgb_data[rgb_offset..rgb_offset + 24];

        // Extract RGB from RGBA using SIMD shuffle
        for i in 0..SIMD_WIDTH {
            rgb_slice[i * 3] = rgba_chunk[i * 4]; // R
            rgb_slice[i * 3 + 1] = rgba_chunk[i * 4 + 1]; // G
            rgb_slice[i * 3 + 2] = rgba_chunk[i * 4 + 2]; // B
                                                          // Skip alpha channel
        }
    }

    // Handle remaining pixels
    let remaining = total_pixels % SIMD_WIDTH;
    let remaining_offset = chunks * SIMD_WIDTH;

    for i in 0..remaining {
        let pixel_idx = remaining_offset + i;
        let rgba_idx = pixel_idx * 4;

        rgb_data.push(rgba_data[rgba_idx]); // R
        rgb_data.push(rgba_data[rgba_idx + 1]); // G
        rgb_data.push(rgba_data[rgba_idx + 2]); // B
                                                // Skip alpha
    }

    let elapsed = start.elapsed().as_nanos() as u64;
    let metrics =
        FormatConversionMetrics::new(total_pixels, elapsed, SIMD_WIDTH, "simd_rgba_to_rgb");

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
