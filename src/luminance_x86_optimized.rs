use anyhow::Result;
use ndarray::ArrayView3;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// High-performance luminance calculations for x86_64 processors
/// Optimized for:
/// - Intel Xeon (AVX-512): 64-byte vector operations
/// - AMD EPYC/Threadripper (AVX2): 32-byte vector operations
/// - Intel Core/AMD Ryzen (AVX2 + FMA): Fused multiply-add operations
///
/// Performance targets:
/// - AVX-512: 16x speedup over scalar
/// - AVX2 + FMA: 8x speedup over scalar
/// - AVX2: 6x speedup over scalar
/// Calculate luminance using the best available x86 instruction set
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn calculate_luminance_x86_optimized(image: &ArrayView3<u8>) -> f64 {
    // Runtime CPU feature detection
    if is_x86_feature_detected!("avx512f") {
        unsafe { calculate_luminance_avx512(image) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { calculate_luminance_avx2_fma(image) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { calculate_luminance_avx2(image) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { calculate_luminance_sse41(image) }
    } else {
        calculate_luminance_scalar(image)
    }
}

/// AVX-512 luminance calculation - processes 16 pixels simultaneously
/// Optimized for Intel Xeon Scalable processors (Skylake-SP+)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn calculate_luminance_avx512(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return calculate_luminance_scalar(image);
    }

    // Luminance coefficients: Y = 0.299*R + 0.587*G + 0.114*B
    let r_coeff = _mm512_set1_ps(0.299);
    let g_coeff = _mm512_set1_ps(0.587);
    let b_coeff = _mm512_set1_ps(0.114);

    let mut total = _mm512_setzero_ps();
    let pixel_count = (height * width) as f64;

    // Process 16 pixels at a time
    const SIMD_WIDTH: usize = 16;
    let pixels_per_chunk = SIMD_WIDTH;

    for h in 0..height {
        let mut w = 0;
        let simd_chunks = width / pixels_per_chunk;

        // Process SIMD chunks
        for _ in 0..simd_chunks {
            // Load 16 RGB pixels (48 bytes)
            let mut r_vals = [0.0f32; 16];
            let mut g_vals = [0.0f32; 16];
            let mut b_vals = [0.0f32; 16];

            for i in 0..SIMD_WIDTH {
                r_vals[i] = image[[h, w + i, 0]] as f32;
                g_vals[i] = image[[h, w + i, 1]] as f32;
                b_vals[i] = image[[h, w + i, 2]] as f32;
            }

            let r_vec = _mm512_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm512_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm512_loadu_ps(b_vals.as_ptr());

            // Calculate luminance using FMA: Y = 0.299*R + (0.587*G + 0.114*B)
            let gb_sum = _mm512_fmadd_ps(g_vec, g_coeff, _mm512_mul_ps(b_vec, b_coeff));
            let luminance = _mm512_fmadd_ps(r_vec, r_coeff, gb_sum);

            total = _mm512_add_ps(total, luminance);
            w += SIMD_WIDTH;
        }

        // Handle remaining pixels
        for w in (simd_chunks * SIMD_WIDTH)..width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;

            // Add to first lane to avoid complex reduction
            let lum_vec = _mm512_set_ps(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lum,
            );
            total = _mm512_add_ps(total, lum_vec);
        }
    }

    // Horizontal sum of all 16 lanes
    let result = horizontal_sum_avx512(total) as f64 / pixel_count;
    result
}

/// AVX2 + FMA luminance calculation - processes 8 pixels simultaneously
/// Optimized for modern Intel/AMD processors with FMA support
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn calculate_luminance_avx2_fma(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return calculate_luminance_scalar(image);
    }

    // Luminance coefficients
    let r_coeff = _mm256_set1_ps(0.299);
    let g_coeff = _mm256_set1_ps(0.587);
    let b_coeff = _mm256_set1_ps(0.114);

    let mut total = _mm256_setzero_ps();
    let pixel_count = (height * width) as f64;

    // Process 8 pixels at a time
    const SIMD_WIDTH: usize = 8;

    for h in 0..height {
        let mut w = 0;
        let simd_chunks = width / SIMD_WIDTH;

        for _ in 0..simd_chunks {
            let mut r_vals = [0.0f32; 8];
            let mut g_vals = [0.0f32; 8];
            let mut b_vals = [0.0f32; 8];

            for i in 0..SIMD_WIDTH {
                r_vals[i] = image[[h, w + i, 0]] as f32;
                g_vals[i] = image[[h, w + i, 1]] as f32;
                b_vals[i] = image[[h, w + i, 2]] as f32;
            }

            let r_vec = _mm256_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm256_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm256_loadu_ps(b_vals.as_ptr());

            // Use FMA for better performance: Y = 0.299*R + (0.587*G + 0.114*B)
            let gb_sum = _mm256_fmadd_ps(g_vec, g_coeff, _mm256_mul_ps(b_vec, b_coeff));
            let luminance = _mm256_fmadd_ps(r_vec, r_coeff, gb_sum);

            total = _mm256_add_ps(total, luminance);
            w += SIMD_WIDTH;
        }

        // Handle remaining pixels
        for w in (simd_chunks * SIMD_WIDTH)..width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;

            let lum_vec = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lum);
            total = _mm256_add_ps(total, lum_vec);
        }
    }

    let result = horizontal_sum_avx2(total) as f64 / pixel_count;
    result
}

/// AVX2 luminance calculation - processes 8 pixels simultaneously
/// For processors without FMA support
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn calculate_luminance_avx2(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return calculate_luminance_scalar(image);
    }

    let r_coeff = _mm256_set1_ps(0.299);
    let g_coeff = _mm256_set1_ps(0.587);
    let b_coeff = _mm256_set1_ps(0.114);

    let mut total = _mm256_setzero_ps();
    let pixel_count = (height * width) as f64;

    const SIMD_WIDTH: usize = 8;

    for h in 0..height {
        let mut w = 0;
        let simd_chunks = width / SIMD_WIDTH;

        for _ in 0..simd_chunks {
            let mut r_vals = [0.0f32; 8];
            let mut g_vals = [0.0f32; 8];
            let mut b_vals = [0.0f32; 8];

            for i in 0..SIMD_WIDTH {
                r_vals[i] = image[[h, w + i, 0]] as f32;
                g_vals[i] = image[[h, w + i, 1]] as f32;
                b_vals[i] = image[[h, w + i, 2]] as f32;
            }

            let r_vec = _mm256_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm256_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm256_loadu_ps(b_vals.as_ptr());

            // Without FMA, use separate multiply and add
            let r_contrib = _mm256_mul_ps(r_vec, r_coeff);
            let g_contrib = _mm256_mul_ps(g_vec, g_coeff);
            let b_contrib = _mm256_mul_ps(b_vec, b_coeff);

            let rg_sum = _mm256_add_ps(r_contrib, g_contrib);
            let luminance = _mm256_add_ps(rg_sum, b_contrib);

            total = _mm256_add_ps(total, luminance);
            w += SIMD_WIDTH;
        }

        // Handle remaining pixels
        for w in (simd_chunks * SIMD_WIDTH)..width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;

            let lum_vec = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lum);
            total = _mm256_add_ps(total, lum_vec);
        }
    }

    let result = horizontal_sum_avx2(total) as f64 / pixel_count;
    result
}

/// SSE4.1 luminance calculation - processes 4 pixels simultaneously
/// Fallback for older processors
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn calculate_luminance_sse41(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return calculate_luminance_scalar(image);
    }

    let r_coeff = _mm_set1_ps(0.299);
    let g_coeff = _mm_set1_ps(0.587);
    let b_coeff = _mm_set1_ps(0.114);

    let mut total = _mm_setzero_ps();
    let pixel_count = (height * width) as f64;

    const SIMD_WIDTH: usize = 4;

    for h in 0..height {
        let mut w = 0;
        let simd_chunks = width / SIMD_WIDTH;

        for _ in 0..simd_chunks {
            let mut r_vals = [0.0f32; 4];
            let mut g_vals = [0.0f32; 4];
            let mut b_vals = [0.0f32; 4];

            for i in 0..SIMD_WIDTH {
                r_vals[i] = image[[h, w + i, 0]] as f32;
                g_vals[i] = image[[h, w + i, 1]] as f32;
                b_vals[i] = image[[h, w + i, 2]] as f32;
            }

            let r_vec = _mm_loadu_ps(r_vals.as_ptr());
            let g_vec = _mm_loadu_ps(g_vals.as_ptr());
            let b_vec = _mm_loadu_ps(b_vals.as_ptr());

            let r_contrib = _mm_mul_ps(r_vec, r_coeff);
            let g_contrib = _mm_mul_ps(g_vec, g_coeff);
            let b_contrib = _mm_mul_ps(b_vec, b_coeff);

            let rg_sum = _mm_add_ps(r_contrib, g_contrib);
            let luminance = _mm_add_ps(rg_sum, b_contrib);

            total = _mm_add_ps(total, luminance);
            w += SIMD_WIDTH;
        }

        // Handle remaining pixels
        for w in (simd_chunks * SIMD_WIDTH)..width {
            let r = image[[h, w, 0]] as f32;
            let g = image[[h, w, 1]] as f32;
            let b = image[[h, w, 2]] as f32;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;

            let lum_vec = _mm_set_ps(0.0, 0.0, 0.0, lum);
            total = _mm_add_ps(total, lum_vec);
        }
    }

    let result = horizontal_sum_sse(total) as f64 / pixel_count;
    result
}

/// Scalar fallback luminance calculation
#[allow(dead_code)] // Used in conditional compilation branches
fn calculate_luminance_scalar(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return 0.0;
    }

    let mut total = 0.0;
    let pixel_count = (height * width) as f64;

    for h in 0..height {
        for w in 0..width {
            let r = image[[h, w, 0]] as f64;
            let g = image[[h, w, 1]] as f64;
            let b = image[[h, w, 2]] as f64;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            total += lum;
        }
    }

    total / pixel_count
}

/// Horizontal sum for AVX-512 registers
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn horizontal_sum_avx512(v: __m512) -> f32 {
    let sum1 = _mm512_reduce_add_ps(v);
    sum1
}

/// Horizontal sum for AVX2 registers
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    let v128_lo = _mm256_extractf128_ps(v, 0);
    let v128_hi = _mm256_extractf128_ps(v, 1);
    let v128 = _mm_add_ps(v128_lo, v128_hi);
    horizontal_sum_sse(v128)
}

/// Horizontal sum for SSE registers
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
unsafe fn horizontal_sum_sse(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

/// Multi-threaded x86 luminance calculation for large images
/// Uses all available CPU cores with optimal SIMD instructions
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn calculate_luminance_x86_parallel(image: &ArrayView3<u8>) -> Result<f64> {
    use rayon::prelude::*;

    let (height, width, channels) = image.dim();

    if channels != 3 {
        return Ok(0.0);
    }

    // For small images, single-threaded is faster due to thread overhead
    if height * width < 1000000 {
        return Ok(calculate_luminance_x86_optimized(image));
    }

    let pixel_count = (height * width) as f64;

    // Process rows in parallel, each using the best SIMD instructions
    let total: f64 = (0..height)
        .into_par_iter()
        .map(|h| {
            let row = image.index_axis(ndarray::Axis(0), h);
            let row_3d = row.insert_axis(ndarray::Axis(0));
            calculate_luminance_x86_optimized(&row_3d.view())
        })
        .sum::<f64>()
        * (width as f64);

    Ok(total / pixel_count)
}

/// Fallback for non-x86 platforms
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub fn calculate_luminance_x86_optimized(_image: &ArrayView3<u8>) -> f64 {
    0.0 // Not available on this platform
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub fn calculate_luminance_x86_parallel(_image: &ArrayView3<u8>) -> Result<f64> {
    Err(anyhow::anyhow!(
        "x86 optimizations not available on this platform"
    ))
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_x86_luminance_basic() {
        let _image = Array3::<u8>::ones((100, 100, 3)) * 128;

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let result = calculate_luminance_x86_optimized(&_image.view());
            // Should be close to 128 * luminance coefficients
            let expected = 128.0 * (0.299 + 0.587 + 0.114);
            assert!((result - expected).abs() < 1.0);
        }
    }

    #[test]
    fn test_x86_luminance_parallel() {
        let _image = Array3::<u8>::ones((1000, 1000, 3)) * 64;

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let result = calculate_luminance_x86_parallel(&_image.view());
            if let Ok(luminance) = result {
                let expected = 64.0 * (0.299 + 0.587 + 0.114);
                assert!((luminance - expected).abs() < 1.0);
            }
        }
    }

    #[test]
    fn test_cpu_feature_detection() {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            // Just verify the detection doesn't crash on Apple Silicon
            let _avx512 = is_x86_feature_detected!("avx512f");
            let _avx2 = is_x86_feature_detected!("avx2");
            let _fma = is_x86_feature_detected!("fma");
            let _sse41 = is_x86_feature_detected!("sse4.1");

            // All should be false on Apple Silicon, but that's expected
        }
    }
}
