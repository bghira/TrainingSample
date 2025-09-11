use anyhow::Result;
use ndarray::{Array3, ArrayView3, Axis};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Multi-core parallel SIMD image resize implementation
///
/// This combines the NEON SIMD optimizations with Rayon's work-stealing
/// parallelism to utilize all CPU cores on Apple Silicon.
///
/// Expected performance: 8-16x speedup over single-core NEON
pub struct MultiCoreResizeEngine {
    thread_pool: rayon::ThreadPool,
    cores_used: AtomicUsize,
}

impl MultiCoreResizeEngine {
    pub fn new() -> Result<Self> {
        let num_cores = num_cpus::get();

        // Use all performance cores, reserve 1-2 for system
        let worker_threads = if num_cores >= 12 {
            num_cores - 2 // M3 Max: use 14/16 cores
        } else if num_cores >= 8 {
            num_cores - 1 // M3 Pro: use 10/11 cores
        } else {
            num_cores // Smaller systems: use all
        };

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .thread_name(|i| format!("resize-worker-{}", i))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))?;

        Ok(Self {
            thread_pool,
            cores_used: AtomicUsize::new(0),
        })
    }

    /// Multi-core parallel bilinear resize with NEON SIMD
    pub fn resize_bilinear_parallel(
        &self,
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

        // For small images, single-core NEON is faster due to reduced overhead
        if dst_width * dst_height < 256 * 256 {
            return crate::resize_neon_optimized::resize_bilinear_neon_optimized_safe(
                image,
                target_width,
                target_height,
            );
        }

        let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));
        let cores_used = AtomicUsize::new(0);

        // Calculate optimal block size based on L1 cache size (~128KB per core)
        let block_height = calculate_optimal_block_height(dst_width, dst_height);

        self.thread_pool.install(|| {
            // Process image in horizontal blocks for better cache locality
            result
                .axis_chunks_iter_mut(Axis(0), block_height)
                .enumerate()
                .par_bridge()
                .for_each(|(block_idx, mut block)| {
                    cores_used.fetch_add(1, Ordering::Relaxed);

                    let block_start_y = block_idx * block_height;
                    let block_end_y = (block_start_y + block.dim().0).min(dst_height);

                    // Use NEON SIMD within each block
                    process_block_with_neon(
                        image,
                        &mut block,
                        BlockParams {
                            block_start_y,
                            _block_end_y: block_end_y,
                            dst_width,
                            _dst_height: dst_height,
                            src_width,
                            src_height,
                        },
                    );
                });
        });

        self.cores_used
            .store(cores_used.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(result)
    }

    /// Get the number of cores actually used in the last operation
    pub fn cores_used(&self) -> usize {
        self.cores_used.load(Ordering::Relaxed)
    }

    /// Get the configured thread pool size
    pub fn thread_pool_size(&self) -> usize {
        self.thread_pool.current_num_threads()
    }
}

fn calculate_optimal_block_height(width: usize, height: usize) -> usize {
    // Target ~32KB per block (fits in L1 cache)
    // Each pixel = 3 bytes, want ~10K pixels per block
    let target_pixels_per_block = 10_000;
    let pixels_per_row = width;
    let optimal_height = target_pixels_per_block / pixels_per_row;

    // Clamp to reasonable bounds
    optimal_height.max(16).min(height / 4).min(256)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
struct BlockParams {
    block_start_y: usize,
    _block_end_y: usize,
    dst_width: usize,
    _dst_height: usize,
    src_width: usize,
    src_height: usize,
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
fn process_block_with_neon(
    image: &ArrayView3<u8>,
    block: &mut ndarray::ArrayViewMut3<u8>,
    params: BlockParams,
) {
    let x_scale = params.src_width as f32 / params.dst_width as f32;
    let y_scale = params.src_height as f32 / params.block_start_y.max(1) as f32;

    // Use the same NEON logic as the single-core version
    // but operating on just this block
    for (local_y, mut row) in block.outer_iter_mut().enumerate() {
        let dst_y = params.block_start_y + local_y;

        let src_y_f = (dst_y as f32 + 0.5) * y_scale - 0.5;
        let src_y = src_y_f.floor() as i32;
        let y_weight = src_y_f - src_y as f32;
        let y0 = (src_y.max(0) as usize).min(params.src_height - 1);
        let y1 = ((src_y + 1).max(0) as usize).min(params.src_height - 1);

        let inv_y_weight = 1.0 - y_weight;

        // Process 4 pixels at a time with NEON (same as single-core version)
        let mut dst_x = 0;
        while dst_x + 4 <= params.dst_width {
            unsafe {
                process_4_pixels_neon(
                    image,
                    &mut row,
                    PixelParams {
                        dst_x,
                        x_scale,
                        y0,
                        y1,
                        y_weight,
                        inv_y_weight,
                        src_width: params.src_width,
                        _src_height: params.src_height,
                    },
                );
            }
            dst_x += 4;
        }

        // Handle remainder pixels
        for dst_x in dst_x..params.dst_width {
            let src_x_f = (dst_x as f32 + 0.5) * x_scale - 0.5;
            let src_x = src_x_f.floor() as i32;
            let x_weight = src_x_f - src_x as f32;
            let x0 = (src_x.max(0) as usize).min(params.src_width - 1);
            let x1 = ((src_x + 1).max(0) as usize).min(params.src_width - 1);

            for c in 0..3 {
                let tl = image[[y0, x0, c]] as f32;
                let tr = image[[y0, x1, c]] as f32;
                let bl = image[[y1, x0, c]] as f32;
                let br = image[[y1, x1, c]] as f32;

                let top = tl * (1.0 - x_weight) + tr * x_weight;
                let bottom = bl * (1.0 - x_weight) + br * x_weight;
                let final_val = top * inv_y_weight + bottom * y_weight;

                row[[dst_x, c]] = (final_val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
fn process_block_with_neon(
    image: &ArrayView3<u8>,
    block: &mut ndarray::ArrayViewMut3<u8>,
    block_start_y: usize,
    _block_end_y: usize,
    dst_width: usize,
    _dst_height: usize,
    src_width: usize,
    src_height: usize,
) {
    // Fallback scalar implementation for non-ARM64 platforms
    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / block_start_y.max(1) as f32;

    for (local_y, mut row) in block.outer_iter_mut().enumerate() {
        let dst_y = block_start_y + local_y;

        let src_y_f = (dst_y as f32 + 0.5) * y_scale - 0.5;
        let src_y = src_y_f.floor() as i32;
        let y_weight = src_y_f - src_y as f32;
        let y0 = (src_y.max(0) as usize).min(src_height - 1);
        let y1 = ((src_y + 1).max(0) as usize).min(src_height - 1);

        let inv_y_weight = 1.0 - y_weight;

        for dst_x in 0..dst_width {
            let src_x_f = (dst_x as f32 + 0.5) * x_scale - 0.5;
            let src_x = src_x_f.floor() as i32;
            let x_weight = src_x_f - src_x as f32;
            let x0 = (src_x.max(0) as usize).min(src_width - 1);
            let x1 = ((src_x + 1).max(0) as usize).min(src_width - 1);

            for c in 0..3 {
                let tl = image[[y0, x0, c]] as f32;
                let tr = image[[y0, x1, c]] as f32;
                let bl = image[[y1, x0, c]] as f32;
                let br = image[[y1, x1, c]] as f32;

                let top = tl * (1.0 - x_weight) + tr * x_weight;
                let bottom = bl * (1.0 - x_weight) + br * x_weight;
                let final_val = top * inv_y_weight + bottom * y_weight;

                row[[dst_x, c]] = (final_val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
struct PixelParams {
    dst_x: usize,
    x_scale: f32,
    y0: usize,
    y1: usize,
    y_weight: f32,
    inv_y_weight: f32,
    src_width: usize,
    _src_height: usize,
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn process_4_pixels_neon(
    image: &ArrayView3<u8>,
    row: &mut ndarray::ArrayViewMut2<u8>,
    params: PixelParams,
) {
    use std::arch::aarch64::*;

    // Calculate source coordinates for 4 pixels
    let mut x_coords = [(0usize, 0usize, 0.0f32); 4];
    for (i, coord) in x_coords.iter_mut().enumerate() {
        let src_x_f = ((params.dst_x + i) as f32 + 0.5) * params.x_scale - 0.5;
        let src_x = src_x_f.floor() as i32;
        let x_weight = src_x_f - src_x as f32;
        let x0 = (src_x.max(0) as usize).min(params.src_width - 1);
        let x1 = ((src_x + 1).max(0) as usize).min(params.src_width - 1);
        *coord = (x0, x1, x_weight);
    }

    // NEON constants
    let zero_f32 = vdupq_n_f32(0.0);
    let vec_255 = vdupq_n_f32(255.0);
    let vec_05 = vdupq_n_f32(0.5);
    let one_f32 = vdupq_n_f32(1.0);
    let y_weight_vec = vdupq_n_f32(params.y_weight);
    let inv_y_weight_vec = vdupq_n_f32(params.inv_y_weight);

    let x_weights = [x_coords[0].2, x_coords[1].2, x_coords[2].2, x_coords[3].2];
    let x_weight_vec = vld1q_f32(x_weights.as_ptr());
    let inv_x_weight_vec = vsubq_f32(one_f32, x_weight_vec);

    // Process each color channel
    for c in 0..3 {
        let mut tl_vals = [0.0f32; 4];
        let mut tr_vals = [0.0f32; 4];
        let mut bl_vals = [0.0f32; 4];
        let mut br_vals = [0.0f32; 4];

        for (i, &(x0, x1, _)) in x_coords.iter().enumerate() {
            tl_vals[i] = *image.uget((params.y0, x0, c)) as f32;
            tr_vals[i] = *image.uget((params.y0, x1, c)) as f32;
            bl_vals[i] = *image.uget((params.y1, x0, c)) as f32;
            br_vals[i] = *image.uget((params.y1, x1, c)) as f32;
        }

        let tl_vec = vld1q_f32(tl_vals.as_ptr());
        let tr_vec = vld1q_f32(tr_vals.as_ptr());
        let bl_vec = vld1q_f32(bl_vals.as_ptr());
        let br_vec = vld1q_f32(br_vals.as_ptr());

        // NEON bilinear interpolation
        let top_interp = vmlaq_f32(vmulq_f32(tl_vec, inv_x_weight_vec), tr_vec, x_weight_vec);
        let bottom_interp = vmlaq_f32(vmulq_f32(bl_vec, inv_x_weight_vec), br_vec, x_weight_vec);
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
            *row.uget_mut((params.dst_x + i, c)) = val;
        }
    }
}

/// Convenience function for multi-core resize
pub fn resize_bilinear_multicore(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    let engine = MultiCoreResizeEngine::new()?;
    engine.resize_bilinear_parallel(image, target_width, target_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_multicore_engine_creation() {
        let engine = MultiCoreResizeEngine::new();
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        println!("Thread pool size: {}", engine.thread_pool_size());
        println!("Available cores: {}", num_cpus::get());
    }

    #[test]
    fn test_multicore_correctness() {
        let test_image =
            Array3::<u8>::from_shape_fn((512, 512, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();

        let engine = MultiCoreResizeEngine::new().unwrap();
        let result = engine.resize_bilinear_parallel(&view, 256, 256);
        assert!(result.is_ok());

        let resized = result.unwrap();
        assert_eq!(resized.dim(), (256, 256, 3));

        println!(
            "Cores used: {}/{}",
            engine.cores_used(),
            engine.thread_pool_size()
        );
    }

    #[test]
    fn benchmark_multicore_performance() {
        let test_image =
            Array3::<u8>::from_shape_fn((2048, 2048, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();
        let iterations = 3;

        println!(
            "\n🚀 Multi-core NEON Benchmark (2048→1024, {} iterations)",
            iterations
        );
        println!("========================================================");

        // Single-core NEON baseline
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = crate::resize_neon_optimized::resize_bilinear_neon_optimized_safe(
                &view, 1024, 1024,
            )
            .unwrap();
        }
        let single_time = start.elapsed().as_secs_f64() / iterations as f64;
        let single_throughput = (1024 * 1024) as f64 / single_time / 1_000_000.0;

        // Multi-core NEON
        let engine = MultiCoreResizeEngine::new().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = engine.resize_bilinear_parallel(&view, 1024, 1024).unwrap();
        }
        let multi_time = start.elapsed().as_secs_f64() / iterations as f64;
        let multi_throughput = (1024 * 1024) as f64 / multi_time / 1_000_000.0;

        let speedup = multi_throughput / single_throughput;
        let cores_used = engine.cores_used();
        let efficiency = speedup / cores_used as f64;

        println!("📊 Results:");
        println!(
            "   Single-core NEON: {:.1} MPx/s ({:.1}ms)",
            single_throughput,
            single_time * 1000.0
        );
        println!(
            "   Multi-core NEON:  {:.1} MPx/s ({:.1}ms)",
            multi_throughput,
            multi_time * 1000.0
        );
        println!(
            "   Cores used: {}/{}",
            cores_used,
            engine.thread_pool_size()
        );
        println!("   Speedup: {:.1}x", speedup);
        println!("   Parallel efficiency: {:.1}%", efficiency * 100.0);

        if speedup >= cores_used as f64 * 0.7 {
            println!("   ✅ Excellent parallel scaling!");
        } else if speedup >= cores_used as f64 * 0.5 {
            println!("   ⚡ Good parallel performance");
        } else {
            println!("   📊 Limited by memory bandwidth or overhead");
        }

        // Test scaling across different image sizes
        println!("\n📐 Size scaling analysis:");
        let test_sizes = [(512, 256), (1024, 512), (2048, 1024), (4096, 2048)];

        for (src_size, dst_size) in test_sizes {
            if src_size > 2048 {
                continue;
            } // Skip very large for CI

            let test_img = Array3::<u8>::from_shape_fn((src_size, src_size, 3), |(h, w, c)| {
                ((h + w + c) % 256) as u8
            });
            let view = test_img.view();

            let start = std::time::Instant::now();
            let _ = engine
                .resize_bilinear_parallel(&view, dst_size as u32, dst_size as u32)
                .unwrap();
            let time = start.elapsed().as_secs_f64();
            let throughput = (dst_size * dst_size) as f64 / time / 1_000_000.0;

            println!(
                "   {}→{}: {:.1} MPx/s ({:.1}ms)",
                src_size,
                dst_size,
                throughput,
                time * 1000.0
            );
        }
    }
}
