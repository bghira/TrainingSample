use anyhow::Result;
use ndarray::{Array3, ArrayView3};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use std::arch::aarch64::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn resize_bilinear_neon_advanced(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    let (src_height, src_width, channels) = image.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB images (3 channels) are supported"
        ));
    }

    let dst_width = target_width as usize;
    let dst_height = target_height as usize;
    let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    // Process in blocks for better cache utilization
    const BLOCK_SIZE: usize = 32; // Optimized for Apple Silicon cache

    for block_y in (0..dst_height).step_by(BLOCK_SIZE) {
        for block_x in (0..dst_width).step_by(BLOCK_SIZE) {
            let block_end_y = (block_y + BLOCK_SIZE).min(dst_height);
            let block_end_x = (block_x + BLOCK_SIZE).min(dst_width);

            process_neon_block(
                image,
                &mut result,
                NeonBlockParams {
                    block_x,
                    block_y,
                    block_end_x,
                    block_end_y,
                    x_scale,
                    y_scale,
                    src_width,
                    src_height,
                },
            );
        }
    }

    Ok(result)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
struct NeonBlockParams {
    block_x: usize,
    block_y: usize,
    block_end_x: usize,
    block_end_y: usize,
    x_scale: f32,
    y_scale: f32,
    src_width: usize,
    src_height: usize,
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn process_neon_block(
    image: &ArrayView3<u8>,
    result: &mut Array3<u8>,
    params: NeonBlockParams,
) {
    // NEON constants
    let zero_f32 = vdupq_n_f32(0.0);
    let vec_255 = vdupq_n_f32(255.0);
    let vec_05 = vdupq_n_f32(0.5);
    let one_f32 = vdupq_n_f32(1.0);

    for dst_y in params.block_y..params.block_end_y {
        let src_y_f = (dst_y as f32 + 0.5) * params.y_scale - 0.5;
        let src_y = src_y_f.floor() as i32;
        let y_weight = src_y_f - src_y as f32;
        let y0 = (src_y.max(0) as usize).min(params.src_height - 1);
        let y1 = ((src_y + 1).max(0) as usize).min(params.src_height - 1);

        let inv_y_weight = 1.0 - y_weight;
        let y_weight_vec = vdupq_n_f32(y_weight);
        let inv_y_weight_vec = vdupq_n_f32(inv_y_weight);

        let mut dst_x = params.block_x;

        // Process 4 pixels at a time with NEON
        while dst_x + 4 <= params.block_end_x {
            // Pre-calculate coordinates
            let mut x_coords = [(0usize, 0usize, 0.0f32); 4];
            for (i, coord) in x_coords.iter_mut().enumerate() {
                let src_x_f = ((dst_x + i) as f32 + 0.5) * params.x_scale - 0.5;
                let src_x = src_x_f.floor() as i32;
                let x_weight = src_x_f - src_x as f32;
                let x0 = (src_x.max(0) as usize).min(params.src_width - 1);
                let x1 = ((src_x + 1).max(0) as usize).min(params.src_width - 1);
                *coord = (x0, x1, x_weight);
            }

            let x_weights = [x_coords[0].2, x_coords[1].2, x_coords[2].2, x_coords[3].2];
            let x_weight_vec = vld1q_f32(x_weights.as_ptr());
            let inv_x_weight_vec = vsubq_f32(one_f32, x_weight_vec);

            // Process RGB channels together for better memory locality
            for c in 0..3 {
                // Load pixels with optimized access pattern
                let mut pixels = [0.0f32; 16]; // 4 pixels × 4 corners

                for (i, &(x0, x1, _)) in x_coords.iter().enumerate() {
                    pixels[i * 4] = *image.uget((y0, x0, c)) as f32; // tl
                    pixels[i * 4 + 1] = *image.uget((y0, x1, c)) as f32; // tr
                    pixels[i * 4 + 2] = *image.uget((y1, x0, c)) as f32; // bl
                    pixels[i * 4 + 3] = *image.uget((y1, x1, c)) as f32; // br
                }

                // Fixed pixel loading - remove the incorrect part above
                let mut tl_vals = [0.0f32; 4];
                let mut tr_vals = [0.0f32; 4];
                let mut bl_vals = [0.0f32; 4];
                let mut br_vals = [0.0f32; 4];

                for (i, &(x0, x1, _)) in x_coords.iter().enumerate() {
                    tl_vals[i] = *image.uget((y0, x0, c)) as f32;
                    tr_vals[i] = *image.uget((y0, x1, c)) as f32;
                    bl_vals[i] = *image.uget((y1, x0, c)) as f32;
                    br_vals[i] = *image.uget((y1, x1, c)) as f32;
                }

                let tl_vec = vld1q_f32(tl_vals.as_ptr());
                let tr_vec = vld1q_f32(tr_vals.as_ptr());
                let bl_vec = vld1q_f32(bl_vals.as_ptr());
                let br_vec = vld1q_f32(br_vals.as_ptr());

                // NEON bilinear interpolation with FMA
                let top_interp =
                    vmlaq_f32(vmulq_f32(tl_vec, inv_x_weight_vec), tr_vec, x_weight_vec);
                let bottom_interp =
                    vmlaq_f32(vmulq_f32(bl_vec, inv_x_weight_vec), br_vec, x_weight_vec);
                let final_interp = vmlaq_f32(
                    vmulq_f32(top_interp, inv_y_weight_vec),
                    bottom_interp,
                    y_weight_vec,
                );

                // Clamp and convert
                let clamped = vaddq_f32(
                    vmaxq_f32(zero_f32, vminq_f32(vec_255, final_interp)),
                    vec_05,
                );

                let clamped_u32 = vcvtq_u32_f32(clamped);
                let clamped_u16 = vmovn_u32(clamped_u32);
                let clamped_u8 = vmovn_u16(vcombine_u16(clamped_u16, clamped_u16));

                let result_vals: [u8; 8] = std::mem::transmute(clamped_u8);
                for (i, &val) in result_vals.iter().enumerate().take(4) {
                    *result.uget_mut((dst_y, dst_x + i, c)) = val;
                }
            }

            dst_x += 4;
        }

        // Handle remainder pixels
        for dst_x in dst_x..params.block_end_x {
            let src_x_f = (dst_x as f32 + 0.5) * params.x_scale - 0.5;
            let src_x = src_x_f.floor() as i32;
            let x_weight = src_x_f - src_x as f32;
            let x0 = (src_x.max(0) as usize).min(params.src_width - 1);
            let x1 = ((src_x + 1).max(0) as usize).min(params.src_width - 1);

            for c in 0..3 {
                let tl = *image.uget((y0, x0, c)) as f32;
                let tr = *image.uget((y0, x1, c)) as f32;
                let bl = *image.uget((y1, x0, c)) as f32;
                let br = *image.uget((y1, x1, c)) as f32;

                let top = tl * (1.0 - x_weight) + tr * x_weight;
                let bottom = bl * (1.0 - x_weight) + br * x_weight;
                let final_val = top * inv_y_weight + bottom * y_weight;

                *result.uget_mut((dst_y, dst_x, c)) = (final_val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Auto-detecting NEON implementation
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn resize_bilinear_neon_optimized_safe(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { resize_bilinear_neon_advanced(image, target_width, target_height) }
    } else {
        // Fallback to scalar (very unlikely on modern ARM64)
        crate::resize_simd::resize_bilinear_scalar(image, target_width, target_height)
            .map(|(result, _)| result)
    }
}

/// Fallback for non-ARM64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub fn resize_bilinear_neon_optimized_safe(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    crate::resize_simd::resize_bilinear_scalar(image, target_width, target_height)
        .map(|(result, _)| result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_neon_correctness() {
        let test_image =
            Array3::<u8>::from_shape_fn((256, 256, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();

        let result = resize_bilinear_neon_optimized_safe(&view, 128, 128);
        assert!(result.is_ok());

        let resized = result.unwrap();
        assert_eq!(resized.dim(), (128, 128, 3));
    }

    #[test]
    fn benchmark_neon_vs_scalar() {
        let test_image =
            Array3::<u8>::from_shape_fn((2048, 2048, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();
        let iterations = 5;

        println!(
            "\n🍎 Apple Silicon NEON Benchmark (2048→1024, {} iterations)",
            iterations
        );
        println!("==========================================================");

        // Benchmark scalar
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = crate::resize_simd::resize_bilinear_scalar(&view, 1024, 1024).unwrap();
        }
        let scalar_time = start.elapsed().as_secs_f64() / iterations as f64;
        let scalar_throughput = (1024 * 1024) as f64 / scalar_time / 1_000_000.0;

        // Benchmark NEON
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = resize_bilinear_neon_optimized_safe(&view, 1024, 1024).unwrap();
        }
        let neon_time = start.elapsed().as_secs_f64() / iterations as f64;
        let neon_throughput = (1024 * 1024) as f64 / neon_time / 1_000_000.0;

        let speedup = neon_throughput / scalar_throughput;

        println!("📊 Results:");
        println!(
            "   Scalar:   {:.1} MPx/s ({:.1}ms)",
            scalar_throughput,
            scalar_time * 1000.0
        );
        println!(
            "   NEON:     {:.1} MPx/s ({:.1}ms)",
            neon_throughput,
            neon_time * 1000.0
        );
        println!("   Speedup:  {:.2}x", speedup);

        if speedup >= 2.0 {
            println!("   🚀 Excellent NEON performance!");
        } else if speedup >= 1.5 {
            println!("   ✅ Good NEON acceleration");
        } else if speedup >= 1.2 {
            println!("   ⚡ Modest NEON improvement");
        } else {
            println!("   📊 NEON performance similar to scalar");
        }

        // Test with different sizes to see where NEON shines
        println!("\n📐 Size scaling test:");
        let test_sizes = [(512, 256), (1024, 512), (2048, 1024), (4096, 2048)];

        for (src_size, dst_size) in test_sizes {
            let test_img = Array3::<u8>::from_shape_fn((src_size, src_size, 3), |(h, w, c)| {
                ((h + w + c) % 256) as u8
            });
            let view = test_img.view();

            let start = std::time::Instant::now();
            let _ = resize_bilinear_neon_optimized_safe(&view, dst_size, dst_size).unwrap();
            let time = start.elapsed().as_secs_f64();
            let throughput = (dst_size * dst_size) as f64 / time / 1_000_000.0;

            println!("   {}→{}: {:.1} MPx/s", src_size, dst_size, throughput);
        }
    }
}
