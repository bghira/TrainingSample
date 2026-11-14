use ndarray::ArrayView3;

#[cfg(feature = "simd")]
pub use crate::luminance_simd::{calculate_luminance_optimized, LuminanceMetrics};

/// Main luminance calculation function with automatic SIMD optimization
pub fn calculate_luminance_array(image: &ArrayView3<u8>) -> f64 {
    if let Some(result) = calculate_luminance_contiguous(image) {
        return result;
    }

    calculate_luminance_scalar(image)
}

/// Single-threaded luminance calculation to avoid nested parallelism in batch operations
pub fn calculate_luminance_array_sequential(image: &ArrayView3<u8>) -> f64 {
    if let Some(result) = calculate_luminance_contiguous(image) {
        return result;
    }

    calculate_luminance_scalar(image)
}

pub(crate) fn batch_luminance(images: &[ArrayView3<u8>]) -> Vec<f64> {
    use rayon::prelude::*;

    let pixels = images.iter().fold(0usize, |total, image| {
        total.saturating_add(image.dim().0.saturating_mul(image.dim().1))
    });
    // Small batches finish faster than the cost of scheduling Rayon workers.
    // Keep large batches parallel and leave individual kernels single-threaded.
    if images.len() > 1 && pixels > 1024 * 1024 {
        images
            .par_iter()
            .map(calculate_luminance_array_sequential)
            .collect()
    } else {
        images
            .iter()
            .map(calculate_luminance_array_sequential)
            .collect()
    }
}

fn calculate_luminance_contiguous(image: &ArrayView3<u8>) -> Option<f64> {
    let (height, width, channels) = image.dim();
    let data = image.as_slice()?;

    if height == 0 || width == 0 || channels == 0 {
        return Some(0.0);
    }

    if channels < 3 {
        let sum: u64 = data.iter().map(|&x| x as u64).sum();
        return Some(sum as f64 / data.len() as f64);
    }

    let sums = sum_channels(data, channels);
    Some(weighted_average(sums, height * width))
}

fn weighted_average(sums: [u64; 3], pixels: usize) -> f64 {
    (0.299 * sums[0] as f64 + 0.587 * sums[1] as f64 + 0.114 * sums[2] as f64) / pixels as f64
}

fn sum_channels(data: &[u8], channels: usize) -> [u64; 3] {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if matches!(channels, 3 | 4) && std::arch::is_aarch64_feature_detected!("neon") {
        // Every load is bounded by an exact chunk of interleaved pixels.
        return unsafe { sum_channels_neon(data, channels) };
    }
    // Fixed chunk sizes allow the compiler to vectorize RGB and RGBA separately.
    match channels {
        3 => sum_channel_chunks::<3>(data),
        4 => sum_channel_chunks::<4>(data),
        _ => {
            let mut sums = [0u64; 3];
            for pixel in data.chunks_exact(channels) {
                for c in 0..3 {
                    sums[c] += pixel[c] as u64;
                }
            }
            sums
        }
    }
}

fn sum_channel_chunks<const CHANNELS: usize>(data: &[u8]) -> [u64; 3] {
    let mut sums = [0u64; 3];
    for pixel in data.chunks_exact(CHANNELS) {
        sums[0] += pixel[0] as u64;
        sums[1] += pixel[1] as u64;
        sums[2] += pixel[2] as u64;
    }
    sums
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn sum_channels_neon(data: &[u8], channels: usize) -> [u64; 3] {
    use std::arch::aarch64::*;
    let mut sums = [0u64; 3];
    let mut chunks = data.chunks_exact(16 * channels);
    if channels == 3 {
        for chunk in &mut chunks {
            let rgb = vld3q_u8(chunk.as_ptr());
            sums[0] += vaddlvq_u8(rgb.0) as u64;
            sums[1] += vaddlvq_u8(rgb.1) as u64;
            sums[2] += vaddlvq_u8(rgb.2) as u64;
        }
    } else {
        for chunk in &mut chunks {
            let rgba = vld4q_u8(chunk.as_ptr());
            sums[0] += vaddlvq_u8(rgba.0) as u64;
            sums[1] += vaddlvq_u8(rgba.1) as u64;
            sums[2] += vaddlvq_u8(rgba.2) as u64;
        }
    }
    for pixel in chunks.remainder().chunks_exact(channels) {
        for c in 0..3 {
            sums[c] += pixel[c] as u64;
        }
    }
    sums
}

/// Ultra-fast adaptive luminance calculation with automatic SIMD/scalar selection
///
/// Uses SIMD for large images (>64K pixels) and scalar for smaller ones to avoid SIMD overhead.
///
/// # Safety
/// - `rgb_ptr` must be valid for reads of at least `width * height * channels` bytes
/// - `width`, `height`, and `channels` must accurately represent the buffer layout
/// - The buffer must contain valid pixel data in RGB format
/// - `channels` should be 3 for RGB data; other values will return 0.0
pub unsafe fn calculate_luminance_raw_buffer(
    rgb_ptr: *const u8,
    width: usize,
    height: usize,
    channels: usize,
) -> f64 {
    if channels != 3 || width == 0 || height == 0 {
        return 0.0;
    }

    // Use scalar for small images or when SIMD is not available
    calculate_luminance_raw_buffer_scalar(rgb_ptr, width, height, channels)
}

/// High-performance scalar luminance calculation
///
/// # Safety
/// Same safety requirements as `calculate_luminance_raw_buffer`
unsafe fn calculate_luminance_raw_buffer_scalar(
    rgb_ptr: *const u8,
    width: usize,
    height: usize,
    channels: usize,
) -> f64 {
    let mut sum = 0.0f64;
    let pixel_count = width * height;

    // Process pixels with optimized scalar loop
    for i in 0..pixel_count {
        let pixel_offset = i * channels;
        let r = *rgb_ptr.add(pixel_offset) as f64;
        let g = *rgb_ptr.add(pixel_offset + 1) as f64;
        let b = *rgb_ptr.add(pixel_offset + 2) as f64;

        // ITU-R BT.709 luminance formula
        let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum += luminance;
    }

    sum / pixel_count as f64
}

#[cfg(feature = "simd")]
/// SIMD-optimized luminance calculation for large images
///
/// # Safety
/// Same safety requirements as `calculate_luminance_raw_buffer`
#[allow(dead_code)]
unsafe fn calculate_luminance_raw_buffer_simd(
    rgb_ptr: *const u8,
    width: usize,
    height: usize,
    _channels: usize,
) -> f64 {
    let pixel_count = width * height;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return calculate_luminance_raw_avx2(rgb_ptr, pixel_count);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        calculate_luminance_raw_neon(rgb_ptr, pixel_count)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback to scalar on non-ARM architectures
        calculate_luminance_raw_buffer_scalar(rgb_ptr, width, height, _channels)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn calculate_luminance_raw_avx2(rgb_ptr: *const u8, pixel_count: usize) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = 0.0f64;
    let remaining = pixel_count % 8;
    let simd_pixels = pixel_count - remaining;

    // AVX2 constants for ITU-R BT.709
    let r_coeff = _mm256_set1_ps(0.2126);
    let g_coeff = _mm256_set1_ps(0.7152);
    let b_coeff = _mm256_set1_ps(0.0722);
    let mut accumulator = _mm256_setzero_ps();

    // Process 8 pixels at a time with AVX2
    for i in 0..(simd_pixels / 8) {
        let base_offset = i * 24; // 8 pixels * 3 channels

        // Load 24 bytes (8 RGB pixels)
        let rgb_bytes = _mm256_loadu_si256(rgb_ptr.add(base_offset) as *const __m256i);

        // Convert to f32 and extract R, G, B channels
        let rgb_lo = _mm256_unpacklo_epi8(rgb_bytes, _mm256_setzero_si256());
        let _rgb_hi = _mm256_unpackhi_epi8(rgb_bytes, _mm256_setzero_si256());

        let rgb_lo_f32 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rgb_lo, _mm256_setzero_si256()));
        let rgb_hi_f32 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rgb_lo, _mm256_setzero_si256()));

        // FMA: luminance = r * 0.2126 + g * 0.7152 + b * 0.0722
        let r_contrib = _mm256_mul_ps(rgb_lo_f32, r_coeff);
        let g_contrib = _mm256_mul_ps(rgb_hi_f32, g_coeff);
        let b_contrib = _mm256_mul_ps(rgb_lo_f32, b_coeff); // Simplified for now

        let luminance_vec = _mm256_add_ps(_mm256_add_ps(r_contrib, g_contrib), b_contrib);
        accumulator = _mm256_add_ps(accumulator, luminance_vec);
    }

    // Horizontal sum of accumulator
    let sum_vec = _mm256_hadd_ps(accumulator, accumulator);
    let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
    sum += (sum_array[0] + sum_array[4]) as f64;

    // Process remaining pixels with scalar
    for i in simd_pixels..pixel_count {
        let pixel_offset = i * 3;
        let r = *rgb_ptr.add(pixel_offset) as f64;
        let g = *rgb_ptr.add(pixel_offset + 1) as f64;
        let b = *rgb_ptr.add(pixel_offset + 2) as f64;
        sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    sum / pixel_count as f64
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn calculate_luminance_raw_neon(rgb_ptr: *const u8, pixel_count: usize) -> f64 {
    use std::arch::aarch64::*;

    let mut sum = 0.0f64;
    let remaining = pixel_count % 4;
    let simd_pixels = pixel_count - remaining;

    // NEON constants for ITU-R BT.709
    let r_coeff = vdupq_n_f32(0.2126);
    let g_coeff = vdupq_n_f32(0.7152);
    let b_coeff = vdupq_n_f32(0.0722);
    let mut accumulator = vdupq_n_f32(0.0);

    // Process 4 pixels at a time with NEON
    for i in 0..(simd_pixels / 4) {
        let base_offset = i * 12; // 4 pixels * 3 channels

        // Load 12 bytes (4 RGB pixels)
        let rgb_bytes = vld1q_u8(rgb_ptr.add(base_offset));

        // Convert to f32 and apply luminance formula
        let rgb_u16 = vmovl_u8(vget_low_u8(rgb_bytes));
        let rgb_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(rgb_u16)));

        // Extract R, G, B (simplified - need proper deinterleaving)
        let r_f32 = vmulq_f32(rgb_f32, r_coeff);
        let g_f32 = vmulq_f32(rgb_f32, g_coeff);
        let b_f32 = vmulq_f32(rgb_f32, b_coeff);

        let luminance_vec = vaddq_f32(vaddq_f32(r_f32, g_f32), b_f32);
        accumulator = vaddq_f32(accumulator, luminance_vec);
    }

    // Horizontal sum
    let sum_f32 = vaddvq_f32(accumulator);
    sum += sum_f32 as f64;

    // Process remaining pixels with scalar
    for i in simd_pixels..pixel_count {
        let pixel_offset = i * 3;
        let r = *rgb_ptr.add(pixel_offset) as f64;
        let g = *rgb_ptr.add(pixel_offset + 1) as f64;
        let b = *rgb_ptr.add(pixel_offset + 2) as f64;
        sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    sum / pixel_count as f64
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
    if image.is_empty() {
        return 0.0;
    }
    if channels < 3 {
        let sum: u64 = image.iter().map(|&x| x as u64).sum();
        return sum as f64 / image.len() as f64;
    }
    let mut sums = [0u64; 3];
    for y in 0..height {
        for x in 0..width {
            sums[0] += image[[y, x, 0]] as u64;
            sums[1] += image[[y, x, 1]] as u64;
            sums[2] += image[[y, x, 2]] as u64;
        }
    }
    weighted_average(sums, height * width)
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
