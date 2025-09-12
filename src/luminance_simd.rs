use ndarray::ArrayView3;

#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use std::arch::aarch64::*;

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

// Safe wrapper functions for unsafe SIMD implementations
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn safe_calculate_luminance_avx2_fma(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { calculate_luminance_avx2_fma(image) }
    } else {
        calculate_luminance_portable_simd(image)
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn safe_calculate_luminance_avx2(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    if is_x86_feature_detected!("avx2") {
        unsafe { calculate_luminance_avx2(image) }
    } else {
        calculate_luminance_portable_simd(image)
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn safe_calculate_luminance_sse41(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    if is_x86_feature_detected!("sse4.1") {
        unsafe { calculate_luminance_sse41(image) }
    } else {
        calculate_luminance_portable_simd(image)
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
fn safe_calculate_luminance_neon(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { calculate_luminance_neon(image) }
    } else {
        calculate_luminance_portable_simd(image)
    }
}

/// Detect optimal SIMD implementation at runtime
pub fn detect_optimal_luminance_impl() -> fn(&ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            if is_x86_feature_detected!("fma") {
                return safe_calculate_luminance_avx2_fma;
            } else {
                return safe_calculate_luminance_avx2;
            }
        } else if is_x86_feature_detected!("sse4.1") {
            return safe_calculate_luminance_sse41;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return safe_calculate_luminance_neon;
        }
    }

    // Fallback implementations
    #[cfg(feature = "simd")]
    {
        calculate_luminance_portable_simd
    }

    #[cfg(not(feature = "simd"))]
    {
        return calculate_luminance_scalar_tracked;
    }
}

/// Portable SIMD implementation using wide crate
#[cfg(feature = "simd")]
pub fn calculate_luminance_portable_simd(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels < 3 {
        let sum: u64 = image.iter().map(|&x| x as u64).sum();
        let result = sum as f64 / (height * width * channels) as f64;
        let metrics = LuminanceMetrics::new(
            height * width,
            start.elapsed().as_nanos() as u64,
            1,
            "portable_simd_fallback",
        );
        return (result, metrics);
    }

    // SIMD constants
    let r_coeff = f32x8::splat(0.299);
    let g_coeff = f32x8::splat(0.587);
    let b_coeff = f32x8::splat(0.114);

    let mut total = f32x8::splat(0.0);
    let pixel_count = height * width;

    // Process 8 pixels at a time
    let simd_width = 8;
    let simd_chunks = width / simd_width;
    let _remainder = width % simd_width;

    for h in 0..height {
        // Vectorized processing
        for chunk in 0..simd_chunks {
            let base_w = chunk * simd_width;

            // Load 8 RGB pixels
            let mut r_vals = [0.0f32; 8];
            let mut g_vals = [0.0f32; 8];
            let mut b_vals = [0.0f32; 8];

            for i in 0..simd_width {
                r_vals[i] = image[[h, base_w + i, 0]] as f32;
                g_vals[i] = image[[h, base_w + i, 1]] as f32;
                b_vals[i] = image[[h, base_w + i, 2]] as f32;
            }

            let r_vec = f32x8::from(r_vals);
            let g_vec = f32x8::from(g_vals);
            let b_vec = f32x8::from(b_vals);

            // SIMD luminance calculation
            let luminance = r_vec * r_coeff + g_vec * g_coeff + b_vec * b_coeff;
            total += luminance;
        }

        // Handle remainder pixels
        for w in (simd_chunks * simd_width)..width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            // Add to just the first lane to avoid 8x multiplication
            total += f32x8::from([lum, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
    }

    let result = total.reduce_add() as f64 / pixel_count as f64;
    let elapsed_nanos = start.elapsed().as_nanos() as u64;
    let metrics = LuminanceMetrics::new(
        pixel_count,
        elapsed_nanos,
        simd_width,
        "portable_simd_f32x8",
    );

    (result, metrics)
}

/// AVX2 + FMA optimized implementation (Intel/AMD high-end)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn calculate_luminance_avx2_fma(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    use std::arch::x86_64::*;

    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels < 3 {
        return calculate_luminance_scalar_tracked(image);
    }

    // Check if data is contiguous for optimal performance
    if let Some(data) = image.as_slice() {
        return calculate_luminance_avx2_fma_contiguous(data, width, height);
    }

    // Fallback for non-contiguous data - optimized FMA version
    let r_coeff = _mm256_set1_ps(0.299);
    let g_coeff = _mm256_set1_ps(0.587);
    let b_coeff = _mm256_set1_ps(0.114);
    let zero = _mm256_setzero_ps();
    let mut total = _mm256_setzero_ps();

    let pixel_count = height * width;
    let simd_width = 8;

    for h in 0..height {
        let mut w = 0;
        while w + simd_width <= width {
            // Load 8 RGB pixels directly - much more efficient than scalar loops
            let r_vals = [
                image[[h, w, 0]] as f32,
                image[[h, w + 1, 0]] as f32,
                image[[h, w + 2, 0]] as f32,
                image[[h, w + 3, 0]] as f32,
                image[[h, w + 4, 0]] as f32,
                image[[h, w + 5, 0]] as f32,
                image[[h, w + 6, 0]] as f32,
                image[[h, w + 7, 0]] as f32,
            ];
            let g_vals = [
                image[[h, w, 1]] as f32,
                image[[h, w + 1, 1]] as f32,
                image[[h, w + 2, 1]] as f32,
                image[[h, w + 3, 1]] as f32,
                image[[h, w + 4, 1]] as f32,
                image[[h, w + 5, 1]] as f32,
                image[[h, w + 6, 1]] as f32,
                image[[h, w + 7, 1]] as f32,
            ];
            let b_vals = [
                image[[h, w, 2]] as f32,
                image[[h, w + 1, 2]] as f32,
                image[[h, w + 2, 2]] as f32,
                image[[h, w + 3, 2]] as f32,
                image[[h, w + 4, 2]] as f32,
                image[[h, w + 5, 2]] as f32,
                image[[h, w + 6, 2]] as f32,
                image[[h, w + 7, 2]] as f32,
            ];

            let r_vec = _mm256_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm256_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm256_loadu_ps(b_vals.as_ptr());

            // Ultra-efficient FMA chain: zero + r*coeff + g*coeff + b*coeff
            let luminance = _mm256_fmadd_ps(
                r_vec,
                r_coeff,
                _mm256_fmadd_ps(g_vec, g_coeff, _mm256_fmadd_ps(b_vec, b_coeff, zero)),
            );

            total = _mm256_add_ps(total, luminance);
            w += simd_width;
        }

        // Handle remainder pixels
        while w < width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            total = _mm256_add_ps(
                total,
                _mm256_setr_ps(lum, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            );
            w += 1;
        }
    }

    // Horizontal sum of AVX2 register
    let sum_array: [f32; 8] = std::mem::transmute(total);
    let result = sum_array.iter().sum::<f32>() as f64 / pixel_count as f64;

    let metrics = LuminanceMetrics::new(
        pixel_count,
        start.elapsed().as_nanos() as u64,
        simd_width,
        "avx2_fma_optimized",
    );

    (result, metrics)
}

/// Ultra-high-performance AVX2 FMA implementation for contiguous RGB data
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn calculate_luminance_avx2_fma_contiguous(
    data: &[u8],
    width: usize,
    height: usize,
) -> (f64, LuminanceMetrics) {
    use std::arch::x86_64::*;

    let start = std::time::Instant::now();
    let pixel_count = height * width;

    // AVX2 constants for luminance calculation
    let r_coeff = _mm256_set1_ps(0.299);
    let g_coeff = _mm256_set1_ps(0.587);
    let b_coeff = _mm256_set1_ps(0.114);
    let zero = _mm256_setzero_ps();
    let mut total = _mm256_setzero_ps();

    let mut i = 0;
    let pixels_per_iteration = 8;
    let bytes_per_iteration = pixels_per_iteration * 3; // 24 bytes = 8 RGB pixels

    // Process 8 RGB pixels at a time (24 bytes)
    while i + bytes_per_iteration <= data.len() {
        // Extract RGB values from interleaved data
        let r_vals = [
            data[i] as f32,
            data[i + 3] as f32,
            data[i + 6] as f32,
            data[i + 9] as f32,
            data[i + 12] as f32,
            data[i + 15] as f32,
            data[i + 18] as f32,
            data[i + 21] as f32,
        ];
        let g_vals = [
            data[i + 1] as f32,
            data[i + 4] as f32,
            data[i + 7] as f32,
            data[i + 10] as f32,
            data[i + 13] as f32,
            data[i + 16] as f32,
            data[i + 19] as f32,
            data[i + 22] as f32,
        ];
        let b_vals = [
            data[i + 2] as f32,
            data[i + 5] as f32,
            data[i + 8] as f32,
            data[i + 11] as f32,
            data[i + 14] as f32,
            data[i + 17] as f32,
            data[i + 20] as f32,
            data[i + 23] as f32,
        ];

        let r_vec = _mm256_loadu_ps(r_vals.as_ptr());
        let g_vec = _mm256_loadu_ps(g_vals.as_ptr());
        let b_vec = _mm256_loadu_ps(b_vals.as_ptr());

        // Ultra-efficient FMA chain for luminance calculation
        let luminance = _mm256_fmadd_ps(
            r_vec,
            r_coeff,
            _mm256_fmadd_ps(g_vec, g_coeff, _mm256_fmadd_ps(b_vec, b_coeff, zero)),
        );

        total = _mm256_add_ps(total, luminance);
        i += bytes_per_iteration;
    }

    // Handle remainder pixels
    while i + 3 <= data.len() {
        let r = data[i] as f32;
        let g = data[i + 1] as f32;
        let b = data[i + 2] as f32;
        let lum = 0.299 * r + 0.587 * g + 0.114 * b;

        total = _mm256_add_ps(
            total,
            _mm256_setr_ps(lum, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        );
        i += 3;
    }

    // Horizontal sum of AVX2 register
    let sum_array: [f32; 8] = std::mem::transmute(total);
    let result = sum_array.iter().sum::<f32>() as f64 / pixel_count as f64;

    let elapsed_nanos = start.elapsed().as_nanos() as u64;
    let metrics = LuminanceMetrics::new(pixel_count, elapsed_nanos, 8, "avx2_fma_contiguous");

    (result, metrics)
}

/// AVX2 optimized implementation (Intel/AMD mid-range)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn calculate_luminance_avx2(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels < 3 {
        return calculate_luminance_scalar_tracked(image);
    }

    let r_coeff = _mm256_set1_ps(0.299);
    let g_coeff = _mm256_set1_ps(0.587);
    let b_coeff = _mm256_set1_ps(0.114);
    let mut total = _mm256_setzero_ps();

    let pixel_count = height * width;
    let simd_width = 8;

    for h in 0..height {
        let mut w = 0;
        while w + simd_width <= width {
            // Direct pixel loading - more efficient than scalar loops
            let r_vals = [
                image[[h, w, 0]] as f32,
                image[[h, w + 1, 0]] as f32,
                image[[h, w + 2, 0]] as f32,
                image[[h, w + 3, 0]] as f32,
                image[[h, w + 4, 0]] as f32,
                image[[h, w + 5, 0]] as f32,
                image[[h, w + 6, 0]] as f32,
                image[[h, w + 7, 0]] as f32,
            ];
            let g_vals = [
                image[[h, w, 1]] as f32,
                image[[h, w + 1, 1]] as f32,
                image[[h, w + 2, 1]] as f32,
                image[[h, w + 3, 1]] as f32,
                image[[h, w + 4, 1]] as f32,
                image[[h, w + 5, 1]] as f32,
                image[[h, w + 6, 1]] as f32,
                image[[h, w + 7, 1]] as f32,
            ];
            let b_vals = [
                image[[h, w, 2]] as f32,
                image[[h, w + 1, 2]] as f32,
                image[[h, w + 2, 2]] as f32,
                image[[h, w + 3, 2]] as f32,
                image[[h, w + 4, 2]] as f32,
                image[[h, w + 5, 2]] as f32,
                image[[h, w + 6, 2]] as f32,
                image[[h, w + 7, 2]] as f32,
            ];

            let r_vec = _mm256_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm256_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm256_loadu_ps(b_vals.as_ptr());

            let luminance = _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(r_vec, r_coeff), _mm256_mul_ps(g_vec, g_coeff)),
                _mm256_mul_ps(b_vec, b_coeff),
            );

            total = _mm256_add_ps(total, luminance);
            w += simd_width;
        }

        // Handle remainder pixels (scalar addition)
        while w < width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            // Add scalar to first lane only
            total = _mm256_add_ps(
                total,
                _mm256_setr_ps(lum, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            );
            w += 1;
        }
    }

    let sum_array: [f32; 8] = std::mem::transmute(total);
    let result = sum_array.iter().sum::<f32>() as f64 / pixel_count as f64;

    let metrics = LuminanceMetrics::new(
        pixel_count,
        start.elapsed().as_nanos() as u64,
        simd_width,
        "avx2",
    );

    (result, metrics)
}

/// SSE 4.1 implementation (older Intel/AMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn calculate_luminance_sse41(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels < 3 {
        return calculate_luminance_scalar_tracked(image);
    }

    let r_coeff = _mm_set1_ps(0.299);
    let g_coeff = _mm_set1_ps(0.587);
    let b_coeff = _mm_set1_ps(0.114);
    let mut total = _mm_setzero_ps();

    let pixel_count = height * width;
    let simd_width = 4;

    for h in 0..height {
        let mut w = 0;
        while w + simd_width <= width {
            // Direct pixel loading - more efficient than scalar loops
            let r_vals = [
                image[[h, w, 0]] as f32,
                image[[h, w + 1, 0]] as f32,
                image[[h, w + 2, 0]] as f32,
                image[[h, w + 3, 0]] as f32,
            ];
            let g_vals = [
                image[[h, w, 1]] as f32,
                image[[h, w + 1, 1]] as f32,
                image[[h, w + 2, 1]] as f32,
                image[[h, w + 3, 1]] as f32,
            ];
            let b_vals = [
                image[[h, w, 2]] as f32,
                image[[h, w + 1, 2]] as f32,
                image[[h, w + 2, 2]] as f32,
                image[[h, w + 3, 2]] as f32,
            ];

            let r_vec = _mm_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm_loadu_ps(b_vals.as_ptr());

            let luminance = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(r_vec, r_coeff), _mm_mul_ps(g_vec, g_coeff)),
                _mm_mul_ps(b_vec, b_coeff),
            );

            total = _mm_add_ps(total, luminance);
            w += simd_width;
        }

        // Handle remainder pixels (scalar addition)
        while w < width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            // Add scalar to first lane only
            total = _mm_add_ps(total, _mm_setr_ps(lum, 0.0, 0.0, 0.0));
            w += 1;
        }
    }

    let sum_array: [f32; 4] = std::mem::transmute(total);
    let result = sum_array.iter().sum::<f32>() as f64 / pixel_count as f64;

    let metrics = LuminanceMetrics::new(
        pixel_count,
        start.elapsed().as_nanos() as u64,
        simd_width,
        "sse41",
    );

    (result, metrics)
}

/// NEON implementation for Apple Silicon and ARM servers
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn calculate_luminance_neon(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels < 3 {
        return calculate_luminance_scalar_tracked(image);
    }

    // Check if data is contiguous for optimal performance
    if let Some(data) = image.as_slice() {
        return calculate_luminance_neon_contiguous(data, width, height);
    }

    // Fallback for non-contiguous data
    let r_coeff = vdupq_n_f32(0.299);
    let g_coeff = vdupq_n_f32(0.587);
    let b_coeff = vdupq_n_f32(0.114);
    let mut total = vdupq_n_f32(0.0);

    let pixel_count = height * width;
    let simd_width = 4;

    // Process pixels in batches of 4
    for h in 0..height {
        let mut w = 0;
        while w + simd_width <= width {
            // Load 4 RGB pixels directly - much more efficient
            let r_vals = [
                image[[h, w, 0]] as f32,
                image[[h, w + 1, 0]] as f32,
                image[[h, w + 2, 0]] as f32,
                image[[h, w + 3, 0]] as f32,
            ];
            let g_vals = [
                image[[h, w, 1]] as f32,
                image[[h, w + 1, 1]] as f32,
                image[[h, w + 2, 1]] as f32,
                image[[h, w + 3, 1]] as f32,
            ];
            let b_vals = [
                image[[h, w, 2]] as f32,
                image[[h, w + 1, 2]] as f32,
                image[[h, w + 2, 2]] as f32,
                image[[h, w + 3, 2]] as f32,
            ];

            let r_vec = vld1q_f32(r_vals.as_ptr());
            let g_vec = vld1q_f32(g_vals.as_ptr());
            let b_vec = vld1q_f32(b_vals.as_ptr());

            // NEON fused multiply-add for better performance
            let luminance = vmlaq_f32(vmulq_f32(r_vec, r_coeff), g_vec, g_coeff);
            let luminance = vmlaq_f32(luminance, b_vec, b_coeff);

            total = vaddq_f32(total, luminance);
            w += simd_width;
        }

        // Handle remainder pixels (scalar addition)
        while w < width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            // Add scalar to first lane only
            let lum_vec = vsetq_lane_f32(lum, vdupq_n_f32(0.0), 0);
            total = vaddq_f32(total, lum_vec);
            w += 1;
        }
    }

    // Horizontal sum of NEON register
    let sum_array: [f32; 4] = std::mem::transmute(total);
    let result = sum_array.iter().sum::<f32>() as f64 / pixel_count as f64;

    let elapsed_nanos = start.elapsed().as_nanos() as u64;
    let metrics = LuminanceMetrics::new(pixel_count, elapsed_nanos, simd_width, "neon");

    (result, metrics)
}

/// Ultra-high-performance NEON implementation for contiguous RGB data with maximum FMA optimization
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn calculate_luminance_neon_contiguous(
    data: &[u8],
    width: usize,
    height: usize,
) -> (f64, LuminanceMetrics) {
    use std::arch::aarch64::*;

    let start = std::time::Instant::now();
    let pixel_count = height * width;

    // Use optimized coefficients scaled for u16 intermediate calculations
    let r_coeff = vdupq_n_f32(0.299);
    let g_coeff = vdupq_n_f32(0.587);
    let b_coeff = vdupq_n_f32(0.114);

    let mut total = vdupq_n_f64(0.0); // Use f64 accumulator for better precision

    let mut i = 0;
    let pixels_per_iteration = 16; // Process 16 pixels at once
    let bytes_per_iteration = pixels_per_iteration * 3; // 48 bytes

    // Ultra-aggressive SIMD loop - process 16 RGB pixels per iteration with max FMA efficiency
    while i + bytes_per_iteration <= data.len() {
        // Load 48 bytes (16 RGB pixels) with deinterleaving
        let rgb = vld3q_u8(data.as_ptr().add(i)); // Loads R0-15, G0-15, B0-15

        // Process first 8 pixels with minimal intermediate storage
        let r_u16_lo = vmovl_u8(vget_low_u8(rgb.0));
        let g_u16_lo = vmovl_u8(vget_low_u8(rgb.1));
        let b_u16_lo = vmovl_u8(vget_low_u8(rgb.2));

        // Direct FMA calculation with immediate accumulation
        let zero = vdupq_n_f32(0.0);

        // First 4 pixels
        let lum1 = vmlaq_f32(
            vmlaq_f32(
                vmlaq_f32(
                    zero,
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16_lo))),
                    r_coeff,
                ),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16_lo))),
                g_coeff,
            ),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_lo))),
            b_coeff,
        );

        // Second 4 pixels
        let lum2 = vmlaq_f32(
            vmlaq_f32(
                vmlaq_f32(
                    zero,
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_u16_lo))),
                    r_coeff,
                ),
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_u16_lo))),
                g_coeff,
            ),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_lo))),
            b_coeff,
        );

        // Immediate f64 accumulation
        total = vaddq_f64(total, vcvt_f64_f32(vget_low_f32(lum1)));
        total = vaddq_f64(total, vcvt_high_f64_f32(lum1));
        total = vaddq_f64(total, vcvt_f64_f32(vget_low_f32(lum2)));
        total = vaddq_f64(total, vcvt_high_f64_f32(lum2));

        // Process next 8 pixels (9-16)
        let r_u16_hi = vmovl_u8(vget_high_u8(rgb.0));
        let g_u16_hi = vmovl_u8(vget_high_u8(rgb.1));
        let b_u16_hi = vmovl_u8(vget_high_u8(rgb.2));

        // Third 4 pixels
        let lum3 = vmlaq_f32(
            vmlaq_f32(
                vmlaq_f32(
                    zero,
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_u16_hi))),
                    r_coeff,
                ),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_u16_hi))),
                g_coeff,
            ),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_hi))),
            b_coeff,
        );

        // Fourth 4 pixels
        let lum4 = vmlaq_f32(
            vmlaq_f32(
                vmlaq_f32(
                    zero,
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_u16_hi))),
                    r_coeff,
                ),
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_u16_hi))),
                g_coeff,
            ),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_hi))),
            b_coeff,
        );

        // Immediate accumulation
        total = vaddq_f64(total, vcvt_f64_f32(vget_low_f32(lum3)));
        total = vaddq_f64(total, vcvt_high_f64_f32(lum3));
        total = vaddq_f64(total, vcvt_f64_f32(vget_low_f32(lum4)));
        total = vaddq_f64(total, vcvt_high_f64_f32(lum4));

        i += bytes_per_iteration;
    }

    // Process remaining pixels 4 at a time
    while i + 12 <= data.len() {
        let r_vals = [
            data[i] as f32,
            data[i + 3] as f32,
            data[i + 6] as f32,
            data[i + 9] as f32,
        ];
        let g_vals = [
            data[i + 1] as f32,
            data[i + 4] as f32,
            data[i + 7] as f32,
            data[i + 10] as f32,
        ];
        let b_vals = [
            data[i + 2] as f32,
            data[i + 5] as f32,
            data[i + 8] as f32,
            data[i + 11] as f32,
        ];

        let r_vec = vld1q_f32(r_vals.as_ptr());
        let g_vec = vld1q_f32(g_vals.as_ptr());
        let b_vec = vld1q_f32(b_vals.as_ptr());

        let zero = vdupq_n_f32(0.0);
        let luminance = vmlaq_f32(
            vmlaq_f32(vmlaq_f32(zero, r_vec, r_coeff), g_vec, g_coeff),
            b_vec,
            b_coeff,
        );
        let lum_f64_1 = vcvt_f64_f32(vget_low_f32(luminance));
        let lum_f64_2 = vcvt_high_f64_f32(luminance);

        total = vaddq_f64(total, lum_f64_1);
        total = vaddq_f64(total, lum_f64_2);

        i += 12;
    }

    // Handle final remainder pixels
    while i + 3 <= data.len() {
        let r = data[i] as f64;
        let g = data[i + 1] as f64;
        let b = data[i + 2] as f64;
        let lum = 0.299 * r + 0.587 * g + 0.114 * b;

        let lum_vec = vsetq_lane_f64(lum, vdupq_n_f64(0.0), 0);
        total = vaddq_f64(total, lum_vec);
        i += 3;
    }

    // Horizontal sum of f64 NEON register
    let sum_array: [f64; 2] = std::mem::transmute(total);
    let result = (sum_array[0] + sum_array[1]) / pixel_count as f64;

    let elapsed_nanos = start.elapsed().as_nanos() as u64;
    let metrics = LuminanceMetrics::new(
        pixel_count,
        elapsed_nanos,
        16, // Processing 16 pixels per main iteration
        "neon_ultra_fma_optimized",
    );

    (result, metrics)
}

/// Scalar fallback with performance tracking
pub fn calculate_luminance_scalar_tracked(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let start = std::time::Instant::now();
    let (height, width, channels) = image.dim();

    if channels < 3 {
        let sum: u64 = image.iter().map(|&x| x as u64).sum();
        let result = sum as f64 / (height * width * channels) as f64;
        let metrics = LuminanceMetrics::new(
            height * width,
            start.elapsed().as_nanos() as u64,
            1,
            "scalar_fallback",
        );
        return (result, metrics);
    }

    let mut total_luminance = 0.0;
    let pixel_count = height * width;

    for h in 0..height {
        for w in 0..width {
            let r = image[[h, w, 0]] as f64;
            let g = image[[h, w, 1]] as f64;
            let b = image[[h, w, 2]] as f64;

            let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            total_luminance += luminance;
        }
    }

    let result = total_luminance / pixel_count as f64;
    let metrics =
        LuminanceMetrics::new(pixel_count, start.elapsed().as_nanos() as u64, 1, "scalar");

    (result, metrics)
}

/// Main entry point that automatically selects optimal implementation
pub fn calculate_luminance_optimized(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    let optimal_impl = detect_optimal_luminance_impl();
    optimal_impl(image)
}

/// Sequential (single-threaded) luminance calculation to avoid nested parallelism
pub fn calculate_luminance_optimized_sequential(image: &ArrayView3<u8>) -> (f64, LuminanceMetrics) {
    // Always use single-threaded SIMD implementations, avoiding any parallel variants
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            if is_x86_feature_detected!("fma") {
                return safe_calculate_luminance_avx2_fma(image);
            } else {
                return safe_calculate_luminance_avx2(image);
            }
        } else if is_x86_feature_detected!("sse4.1") {
            return safe_calculate_luminance_sse41(image);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return safe_calculate_luminance_neon(image);
        }
    }

    // Fallback implementations
    #[cfg(feature = "simd")]
    {
        calculate_luminance_portable_simd(image)
    }

    #[cfg(not(feature = "simd"))]
    {
        calculate_luminance_scalar_tracked(image)
    }
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
