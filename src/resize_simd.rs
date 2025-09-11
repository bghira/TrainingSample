use anyhow::Result;
use ndarray::{Array3, Array4, ArrayView3, ArrayView4, Axis};
use rayon::prelude::*;

#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_imports)]
use std::arch::x86_64::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[allow(unused_imports)]
use std::arch::aarch64::*;

/// Performance tracking for SIMD resize implementations
#[derive(Debug, Clone)]
pub struct ResizeMetrics {
    pub pixels_processed: usize,
    pub output_pixels: usize,
    pub elapsed_nanos: u64,
    pub simd_width: usize,
    pub implementation: &'static str,
    pub throughput_mpixels_per_sec: f64,
}

impl ResizeMetrics {
    pub fn new(
        pixels_processed: usize,
        output_pixels: usize,
        elapsed_nanos: u64,
        simd_width: usize,
        implementation: &'static str,
    ) -> Self {
        let throughput_mpixels_per_sec =
            (output_pixels as f64) / (elapsed_nanos as f64 / 1_000_000_000.0) / 1_000_000.0;

        Self {
            pixels_processed,
            output_pixels,
            elapsed_nanos,
            simd_width,
            implementation,
            throughput_mpixels_per_sec,
        }
    }
}

/// Interpolation filter types
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    Bilinear,
    Lanczos3,
    Lanczos4,
}

/// Precomputed filter weights for a resize operation
struct FilterWeights {
    weights: Vec<f32>,
    indices: Vec<usize>,
    support: usize,
}

impl FilterWeights {
    fn new(src_size: usize, dst_size: usize, filter: FilterType) -> Self {
        let scale = src_size as f32 / dst_size as f32;
        let filter_scale = if scale > 1.0 { scale } else { 1.0 };

        let (filter_fn, support): (fn(f32) -> f32, f32) = match filter {
            FilterType::Bilinear => (bilinear_filter as fn(f32) -> f32, 1.0),
            FilterType::Lanczos3 => (lanczos3_filter as fn(f32) -> f32, 3.0),
            FilterType::Lanczos4 => (lanczos4_filter as fn(f32) -> f32, 4.0),
        };

        let filter_support = support * filter_scale;
        let mut weights = Vec::new();
        let mut indices = Vec::new();

        for dst_idx in 0..dst_size {
            let center = (dst_idx as f32 + 0.5) * scale - 0.5;
            let left = (center - filter_support).ceil() as i32;
            let right = (center + filter_support).floor() as i32;

            let mut weight_sum = 0.0;
            let mut local_weights = Vec::new();
            let mut local_indices = Vec::new();

            for src_idx in left..=right {
                if src_idx >= 0 && (src_idx as usize) < src_size {
                    let distance = (src_idx as f32 - center) / filter_scale;
                    let weight = filter_fn(distance);
                    if weight.abs() > 1e-6 {
                        local_weights.push(weight);
                        local_indices.push(src_idx as usize);
                        weight_sum += weight;
                    }
                }
            }

            // Normalize weights
            if weight_sum > 0.0 {
                for w in &mut local_weights {
                    *w /= weight_sum;
                }
            }

            weights.extend(local_weights);
            indices.extend(local_indices);
        }

        Self {
            weights,
            indices,
            support: filter_support.ceil() as usize * 2 + 1,
        }
    }
}

fn bilinear_filter(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

fn lanczos3_filter(x: f32) -> f32 {
    let x = x.abs();
    if x < 3.0 && x != 0.0 {
        let pi_x = std::f32::consts::PI * x;
        let pi_x_3 = pi_x / 3.0;
        3.0 * pi_x.sin() * pi_x_3.sin() / (pi_x * pi_x)
    } else if x == 0.0 {
        1.0
    } else {
        0.0
    }
}

fn lanczos4_filter(x: f32) -> f32 {
    let x = x.abs();
    if x < 4.0 && x != 0.0 {
        let pi_x = std::f32::consts::PI * x;
        let pi_x_4 = pi_x / 4.0;
        4.0 * pi_x.sin() * pi_x_4.sin() / (pi_x * pi_x)
    } else if x == 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Optimized SIMD bilinear interpolation with minimal overhead
#[cfg(feature = "simd")]
pub fn resize_bilinear_simd_fast(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<(Array3<u8>, ResizeMetrics)> {
    let start = std::time::Instant::now();
    let (src_height, src_width, channels) = image.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB images (3 channels) are supported"
        ));
    }

    if target_width == 0 || target_height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be greater than zero"
        ));
    }

    let dst_width = target_width as usize;
    let dst_height = target_height as usize;
    let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    // SIMD constants
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let vec_255 = f32x8::splat(255.0);
    let vec_05 = f32x8::splat(0.5);

    // Process output rows in parallel
    result
        .axis_iter_mut(Axis(0))
        .enumerate()
        .par_bridge()
        .for_each(|(dst_y, mut row)| {
            let src_y_f = (dst_y as f32 + 0.5) * y_scale - 0.5;
            let src_y = src_y_f.floor() as i32;
            let y_weight = src_y_f - src_y as f32;
            let y0 = (src_y.max(0) as usize).min(src_height - 1);
            let y1 = ((src_y + 1).max(0) as usize).min(src_height - 1);

            let inv_y_weight = 1.0 - y_weight;

            // Process 8 output pixels at a time with SIMD
            let mut dst_x = 0;
            while dst_x + 8 <= dst_width {
                // Pre-calculate all source coordinates and weights for this chunk
                let mut x0_coords = [0usize; 8];
                let mut x1_coords = [0usize; 8];
                let mut x_weights = [0.0f32; 8];

                for i in 0..8 {
                    let src_x_f = ((dst_x + i) as f32 + 0.5) * x_scale - 0.5;
                    let src_x = src_x_f.floor() as i32;
                    x_weights[i] = src_x_f - src_x as f32;
                    x0_coords[i] = (src_x.max(0) as usize).min(src_width - 1);
                    x1_coords[i] = ((src_x + 1).max(0) as usize).min(src_width - 1);
                }

                let x_weight_vec = f32x8::from(x_weights);
                let inv_x_weight_vec = one - x_weight_vec;

                // Process each color channel
                for c in 0..3 {
                    // Load source pixels efficiently
                    let mut tl_vals = [0.0f32; 8]; // top-left
                    let mut tr_vals = [0.0f32; 8]; // top-right
                    let mut bl_vals = [0.0f32; 8]; // bottom-left
                    let mut br_vals = [0.0f32; 8]; // bottom-right

                    // Optimized pixel loading
                    for i in 0..8 {
                        let x0 = x0_coords[i];
                        let x1 = x1_coords[i];

                        tl_vals[i] = image[[y0, x0, c]] as f32;
                        tr_vals[i] = image[[y0, x1, c]] as f32;
                        bl_vals[i] = image[[y1, x0, c]] as f32;
                        br_vals[i] = image[[y1, x1, c]] as f32;
                    }

                    let tl_vec = f32x8::from(tl_vals);
                    let tr_vec = f32x8::from(tr_vals);
                    let bl_vec = f32x8::from(bl_vals);
                    let br_vec = f32x8::from(br_vals);

                    // Optimized bilinear interpolation using SIMD
                    let top_interp = tl_vec * inv_x_weight_vec + tr_vec * x_weight_vec;
                    let bottom_interp = bl_vec * inv_x_weight_vec + br_vec * x_weight_vec;
                    let final_interp = top_interp * f32x8::splat(inv_y_weight)
                        + bottom_interp * f32x8::splat(y_weight);

                    // Clamp and convert to u8
                    let clamped = final_interp.fast_max(zero).fast_min(vec_255) + vec_05;
                    let result_vals: [f32; 8] = clamped.into();

                    // Store results efficiently
                    for i in 0..8 {
                        row[[dst_x + i, c]] = result_vals[i] as u8;
                    }
                }

                dst_x += 8;
            }

            // Handle remainder pixels (scalar fallback)
            for dst_x in dst_x..dst_width {
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
        });

    let metrics = ResizeMetrics::new(
        src_width * src_height,
        dst_width * dst_height,
        start.elapsed().as_nanos() as u64,
        8,
        "bilinear_simd_fast",
    );

    Ok((result, metrics))
}

/// SIMD-optimized bilinear interpolation (original implementation for comparison)
#[cfg(feature = "simd")]
pub fn resize_bilinear_simd(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<(Array3<u8>, ResizeMetrics)> {
    let start = std::time::Instant::now();
    let (src_height, src_width, channels) = image.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB images (3 channels) are supported"
        ));
    }

    if target_width == 0 || target_height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be greater than zero"
        ));
    }

    let dst_width = target_width as usize;
    let dst_height = target_height as usize;
    let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    // SIMD constants
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let vec_255 = f32x8::splat(255.0);
    let vec_05 = f32x8::splat(0.5);

    // Process output rows in parallel
    result
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(dst_y, mut row)| {
            let src_y_f = (dst_y as f32 + 0.5) * y_scale - 0.5;
            let src_y = src_y_f.floor() as i32;
            let y_weight = src_y_f - src_y as f32;
            let y0 = (src_y.max(0) as usize).min(src_height - 1);
            let y1 = ((src_y + 1).max(0) as usize).min(src_height - 1);

            // Process 8 output pixels at a time with SIMD
            let simd_chunks = dst_width / 8;
            let _remainder = dst_width % 8;

            for chunk in 0..simd_chunks {
                let base_x = chunk * 8;

                // Calculate source coordinates for 8 pixels
                let mut src_x_coords = [0.0f32; 8];
                let mut x_weights = [0.0f32; 8];
                let mut x0_coords = [0usize; 8];
                let mut x1_coords = [0usize; 8];

                for i in 0..8 {
                    let dst_x = base_x + i;
                    let src_x_f = (dst_x as f32 + 0.5) * x_scale - 0.5;
                    let src_x = src_x_f.floor() as i32;
                    src_x_coords[i] = src_x_f;
                    x_weights[i] = src_x_f - src_x as f32;
                    x0_coords[i] = (src_x.max(0) as usize).min(src_width - 1);
                    x1_coords[i] = ((src_x + 1).max(0) as usize).min(src_width - 1);
                }

                let x_weight_vec = f32x8::from(x_weights);
                let inv_x_weight_vec = one - x_weight_vec;
                let y_weight_scalar = y_weight;
                let inv_y_weight_scalar = 1.0 - y_weight;

                // Process each color channel
                for c in 0..3 {
                    // Load source pixels
                    let mut tl_vals = [0.0f32; 8]; // top-left
                    let mut tr_vals = [0.0f32; 8]; // top-right
                    let mut bl_vals = [0.0f32; 8]; // bottom-left
                    let mut br_vals = [0.0f32; 8]; // bottom-right

                    for i in 0..8 {
                        tl_vals[i] = image[[y0, x0_coords[i], c]] as f32;
                        tr_vals[i] = image[[y0, x1_coords[i], c]] as f32;
                        bl_vals[i] = image[[y1, x0_coords[i], c]] as f32;
                        br_vals[i] = image[[y1, x1_coords[i], c]] as f32;
                    }

                    let tl_vec = f32x8::from(tl_vals);
                    let tr_vec = f32x8::from(tr_vals);
                    let bl_vec = f32x8::from(bl_vals);
                    let br_vec = f32x8::from(br_vals);

                    // Bilinear interpolation using SIMD
                    let top_interp = tl_vec * inv_x_weight_vec + tr_vec * x_weight_vec;
                    let bottom_interp = bl_vec * inv_x_weight_vec + br_vec * x_weight_vec;
                    let final_interp = top_interp * f32x8::splat(inv_y_weight_scalar)
                        + bottom_interp * f32x8::splat(y_weight_scalar);

                    // Clamp and convert to u8
                    let clamped = final_interp.fast_max(zero).fast_min(vec_255) + vec_05;
                    let result_vals: [f32; 8] = clamped.into();

                    for i in 0..8 {
                        row[[base_x + i, c]] = result_vals[i] as u8;
                    }
                }
            }

            // Handle remainder pixels (scalar fallback)
            for dst_x in (simd_chunks * 8)..dst_width {
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
                    let final_val = top * (1.0 - y_weight) + bottom * y_weight;

                    row[[dst_x, c]] = (final_val + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });

    let metrics = ResizeMetrics::new(
        src_width * src_height,
        dst_width * dst_height,
        start.elapsed().as_nanos() as u64,
        8,
        "bilinear_simd_f32x8",
    );

    Ok((result, metrics))
}

/// SIMD-optimized Lanczos3 interpolation
#[cfg(feature = "simd")]
pub fn resize_lanczos3_simd(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<(Array3<u8>, ResizeMetrics)> {
    let start = std::time::Instant::now();
    let (src_height, src_width, channels) = image.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB images (3 channels) are supported"
        ));
    }

    if target_width == 0 || target_height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be greater than zero"
        ));
    }

    let dst_width = target_width as usize;
    let dst_height = target_height as usize;

    // Precompute filter weights for horizontal and vertical passes
    let h_weights = FilterWeights::new(src_width, dst_width, FilterType::Lanczos3);
    let v_weights = FilterWeights::new(src_height, dst_height, FilterType::Lanczos3);

    // Temporary buffer for horizontal pass
    let mut temp = Array3::<f32>::zeros((src_height, dst_width, 3));

    // Horizontal pass with SIMD
    temp.outer_iter_mut().enumerate().for_each(|(y, mut row)| {
        for dst_x in 0..dst_width {
            let weights_start = dst_x * h_weights.support;
            let weights_end = weights_start + h_weights.support;

            if weights_end > h_weights.weights.len() {
                continue;
            }

            // SIMD processing for multiple weights at once
            let simd_chunks = h_weights.support / 8;

            for c in 0..3 {
                let mut sum = f32x8::splat(0.0);

                // Process 8 weights at a time
                for chunk in 0..simd_chunks {
                    let base_idx = weights_start + chunk * 8;
                    if base_idx + 8 <= h_weights.weights.len() {
                        let mut pixel_vals = [0.0f32; 8];
                        let mut weight_vals = [0.0f32; 8];

                        for i in 0..8 {
                            let weight_idx = base_idx + i;
                            let src_x = h_weights.indices[weight_idx];
                            pixel_vals[i] = image[[y, src_x, c]] as f32;
                            weight_vals[i] = h_weights.weights[weight_idx];
                        }

                        let pixels = f32x8::from(pixel_vals);
                        let weights = f32x8::from(weight_vals);
                        sum += pixels * weights;
                    }
                }

                // Handle remainder weights (scalar)
                let mut scalar_sum = sum.reduce_add();
                for i in (simd_chunks * 8)..h_weights.support {
                    let weight_idx = weights_start + i;
                    if weight_idx < h_weights.weights.len() {
                        let src_x = h_weights.indices[weight_idx];
                        scalar_sum += image[[y, src_x, c]] as f32 * h_weights.weights[weight_idx];
                    }
                }

                row[[dst_x, c]] = scalar_sum.clamp(0.0, 255.0);
            }
        }
    });

    // Vertical pass with SIMD
    let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));

    result
        .outer_iter_mut()
        .enumerate()
        .for_each(|(dst_y, mut row)| {
            let weights_start = dst_y * v_weights.support;
            let weights_end = weights_start + v_weights.support;

            if weights_end > v_weights.weights.len() {
                return;
            }

            for dst_x in 0..dst_width {
                // SIMD processing for vertical weights
                let simd_chunks = v_weights.support / 8;

                for c in 0..3 {
                    let mut sum = f32x8::splat(0.0);

                    // Process 8 weights at a time
                    for chunk in 0..simd_chunks {
                        let base_idx = weights_start + chunk * 8;
                        if base_idx + 8 <= v_weights.weights.len() {
                            let mut pixel_vals = [0.0f32; 8];
                            let mut weight_vals = [0.0f32; 8];

                            for i in 0..8 {
                                let weight_idx = base_idx + i;
                                let src_y = v_weights.indices[weight_idx];
                                pixel_vals[i] = temp[[src_y, dst_x, c]];
                                weight_vals[i] = v_weights.weights[weight_idx];
                            }

                            let pixels = f32x8::from(pixel_vals);
                            let weights = f32x8::from(weight_vals);
                            sum += pixels * weights;
                        }
                    }

                    // Handle remainder weights (scalar)
                    let mut scalar_sum = sum.reduce_add();
                    for i in (simd_chunks * 8)..v_weights.support {
                        let weight_idx = weights_start + i;
                        if weight_idx < v_weights.weights.len() {
                            let src_y = v_weights.indices[weight_idx];
                            scalar_sum += temp[[src_y, dst_x, c]] * v_weights.weights[weight_idx];
                        }
                    }

                    row[[dst_x, c]] = (scalar_sum + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });

    let metrics = ResizeMetrics::new(
        src_width * src_height,
        dst_width * dst_height,
        start.elapsed().as_nanos() as u64,
        8,
        "lanczos3_simd_f32x8",
    );

    Ok((result, metrics))
}

/// SIMD-optimized Lanczos4 interpolation
#[cfg(feature = "simd")]
pub fn resize_lanczos4_simd(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<(Array3<u8>, ResizeMetrics)> {
    let start = std::time::Instant::now();
    let (src_height, src_width, channels) = image.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB images (3 channels) are supported"
        ));
    }

    if target_width == 0 || target_height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be greater than zero"
        ));
    }

    let dst_width = target_width as usize;
    let dst_height = target_height as usize;

    // Precompute filter weights for horizontal and vertical passes
    let h_weights = FilterWeights::new(src_width, dst_width, FilterType::Lanczos4);
    let v_weights = FilterWeights::new(src_height, dst_height, FilterType::Lanczos4);

    // Temporary buffer for horizontal pass
    let mut temp = Array3::<f32>::zeros((src_height, dst_width, 3));

    // Horizontal pass with SIMD
    temp.outer_iter_mut().enumerate().for_each(|(y, mut row)| {
        for dst_x in 0..dst_width {
            let weights_start = dst_x * h_weights.support;
            let weights_end = weights_start + h_weights.support;

            if weights_end > h_weights.weights.len() {
                continue;
            }

            // SIMD processing for multiple weights at once
            let simd_chunks = h_weights.support / 8;
            let _remainder = h_weights.support % 8;

            for c in 0..3 {
                let mut sum = f32x8::splat(0.0);

                // Process 8 weights at a time
                for chunk in 0..simd_chunks {
                    let base_idx = weights_start + chunk * 8;
                    if base_idx + 8 <= h_weights.weights.len() {
                        let mut pixel_vals = [0.0f32; 8];
                        let mut weight_vals = [0.0f32; 8];

                        for i in 0..8 {
                            let weight_idx = base_idx + i;
                            let src_x = h_weights.indices[weight_idx];
                            pixel_vals[i] = image[[y, src_x, c]] as f32;
                            weight_vals[i] = h_weights.weights[weight_idx];
                        }

                        let pixels = f32x8::from(pixel_vals);
                        let weights = f32x8::from(weight_vals);
                        sum += pixels * weights;
                    }
                }

                // Handle remainder weights (scalar)
                let mut scalar_sum = sum.reduce_add();
                for i in (simd_chunks * 8)..h_weights.support {
                    let weight_idx = weights_start + i;
                    if weight_idx < h_weights.weights.len() {
                        let src_x = h_weights.indices[weight_idx];
                        scalar_sum += image[[y, src_x, c]] as f32 * h_weights.weights[weight_idx];
                    }
                }

                row[[dst_x, c]] = scalar_sum.clamp(0.0, 255.0);
            }
        }
    });

    // Vertical pass with SIMD
    let mut result = Array3::<u8>::zeros((dst_height, dst_width, 3));

    result
        .outer_iter_mut()
        .enumerate()
        .for_each(|(dst_y, mut row)| {
            let weights_start = dst_y * v_weights.support;
            let weights_end = weights_start + v_weights.support;

            if weights_end > v_weights.weights.len() {
                return;
            }

            for dst_x in 0..dst_width {
                // SIMD processing for vertical weights
                let simd_chunks = v_weights.support / 8;

                for c in 0..3 {
                    let mut sum = f32x8::splat(0.0);

                    // Process 8 weights at a time
                    for chunk in 0..simd_chunks {
                        let base_idx = weights_start + chunk * 8;
                        if base_idx + 8 <= v_weights.weights.len() {
                            let mut pixel_vals = [0.0f32; 8];
                            let mut weight_vals = [0.0f32; 8];

                            for i in 0..8 {
                                let weight_idx = base_idx + i;
                                let src_y = v_weights.indices[weight_idx];
                                pixel_vals[i] = temp[[src_y, dst_x, c]];
                                weight_vals[i] = v_weights.weights[weight_idx];
                            }

                            let pixels = f32x8::from(pixel_vals);
                            let weights = f32x8::from(weight_vals);
                            sum += pixels * weights;
                        }
                    }

                    // Handle remainder weights (scalar)
                    let mut scalar_sum = sum.reduce_add();
                    for i in (simd_chunks * 8)..v_weights.support {
                        let weight_idx = weights_start + i;
                        if weight_idx < v_weights.weights.len() {
                            let src_y = v_weights.indices[weight_idx];
                            scalar_sum += temp[[src_y, dst_x, c]] * v_weights.weights[weight_idx];
                        }
                    }

                    row[[dst_x, c]] = (scalar_sum + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });

    let metrics = ResizeMetrics::new(
        src_width * src_height,
        dst_width * dst_height,
        start.elapsed().as_nanos() as u64,
        8,
        "lanczos4_simd_f32x8",
    );

    Ok((result, metrics))
}

/// Fallback scalar implementations
pub fn resize_bilinear_scalar(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<(Array3<u8>, ResizeMetrics)> {
    let start = std::time::Instant::now();
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

    for dst_y in 0..dst_height {
        let src_y_f = (dst_y as f32 + 0.5) * y_scale - 0.5;
        let src_y = src_y_f.floor() as i32;
        let y_weight = src_y_f - src_y as f32;
        let y0 = (src_y.max(0) as usize).min(src_height - 1);
        let y1 = ((src_y + 1).max(0) as usize).min(src_height - 1);

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
                let final_val = top * (1.0 - y_weight) + bottom * y_weight;

                result[[dst_y, dst_x, c]] = (final_val + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    let metrics = ResizeMetrics::new(
        src_width * src_height,
        dst_width * dst_height,
        start.elapsed().as_nanos() as u64,
        1,
        "bilinear_scalar",
    );

    Ok((result, metrics))
}

/// Main resize function with automatic SIMD selection
pub fn resize_image_optimized(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> Result<(Array3<u8>, ResizeMetrics)> {
    #[cfg(feature = "simd")]
    {
        match filter {
            FilterType::Bilinear => resize_bilinear_simd_fast(image, target_width, target_height),
            FilterType::Lanczos3 => resize_lanczos3_simd(image, target_width, target_height),
            FilterType::Lanczos4 => resize_lanczos4_simd(image, target_width, target_height),
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        match filter {
            FilterType::Bilinear => resize_bilinear_scalar(image, target_width, target_height),
            FilterType::Lanczos3 | FilterType::Lanczos4 => {
                // Fallback to bilinear for Lanczos in scalar mode
                resize_bilinear_scalar(image, target_width, target_height)
            }
        }
    }
}

/// Video resize with SIMD optimization
pub fn resize_video_optimized(
    video: &ArrayView4<u8>,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> Result<(Array4<u8>, Vec<ResizeMetrics>)> {
    let (num_frames, _height, _width, channels) = video.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB videos (3 channels) are supported"
        ));
    }

    // Process frames in parallel
    let results: Result<Vec<_>> = (0..num_frames)
        .into_par_iter()
        .map(|frame_idx| {
            let frame = video.index_axis(Axis(0), frame_idx);
            resize_image_optimized(&frame, target_width, target_height, filter)
        })
        .collect();

    let results = results?;
    let (resized_frames, metrics): (Vec<_>, Vec<_>) = results.into_iter().unzip();

    // Stack frames back into 4D array
    let frame_shape = resized_frames[0].dim();
    let mut result = Array4::<u8>::zeros((num_frames, frame_shape.0, frame_shape.1, frame_shape.2));

    for (idx, frame) in resized_frames.into_iter().enumerate() {
        result.index_axis_mut(Axis(0), idx).assign(&frame);
    }

    Ok((result, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_bilinear_correctness() {
        let test_image =
            Array3::<u8>::from_shape_fn((100, 100, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();

        let (scalar_result, _) = resize_bilinear_scalar(&view, 50, 50).unwrap();

        #[cfg(feature = "simd")]
        {
            let (simd_result, _) = resize_bilinear_simd(&view, 50, 50).unwrap();

            // Check that results are very similar (within interpolation tolerance)
            let mut max_diff = 0;
            for (&scalar_val, &simd_val) in scalar_result.iter().zip(simd_result.iter()) {
                let diff = (scalar_val as i32 - simd_val as i32).abs();
                max_diff = max_diff.max(diff);
            }

            assert!(
                max_diff <= 1,
                "SIMD and scalar results should be nearly identical, max diff: {}",
                max_diff
            );
        }
    }

    #[test]
    fn test_lanczos4_structure() {
        let test_image =
            Array3::<u8>::from_shape_fn((64, 64, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();

        #[cfg(feature = "simd")]
        {
            let result = resize_lanczos4_simd(&view, 32, 32);
            assert!(result.is_ok(), "Lanczos4 SIMD should not panic");

            let (resized, metrics) = result.unwrap();
            assert_eq!(resized.dim(), (32, 32, 3));
            assert_eq!(metrics.implementation, "lanczos4_simd_f32x8");
        }
    }

    #[test]
    fn benchmark_resize_performance() {
        let test_image =
            Array3::<u8>::from_shape_fn((512, 512, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
        let view = test_image.view();
        let iterations = 20;

        println!(
            "\n🔬 Resize Performance Benchmark (512→256, {} iterations)",
            iterations
        );
        println!("============================================================");

        // Benchmark scalar bilinear
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = resize_bilinear_scalar(&view, 256, 256).unwrap();
        }
        let scalar_time = start.elapsed().as_secs_f64() / iterations as f64;
        let scalar_throughput = (256 * 256) as f64 / scalar_time / 1_000_000.0;

        #[cfg(feature = "simd")]
        {
            // Benchmark fast SIMD bilinear
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = resize_bilinear_simd_fast(&view, 256, 256).unwrap();
            }
            let simd_fast_time = start.elapsed().as_secs_f64() / iterations as f64;
            let simd_fast_throughput = (256 * 256) as f64 / simd_fast_time / 1_000_000.0;

            // Benchmark original SIMD bilinear
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = resize_bilinear_simd(&view, 256, 256).unwrap();
            }
            let simd_time = start.elapsed().as_secs_f64() / iterations as f64;
            let simd_throughput = (256 * 256) as f64 / simd_time / 1_000_000.0;

            let speedup_fast = simd_fast_throughput / scalar_throughput;
            let speedup_original = simd_throughput / scalar_throughput;

            println!("📊 Bilinear Results:");
            println!(
                "   Scalar:      {:.1} MPx/s ({:.3}ms)",
                scalar_throughput,
                scalar_time * 1000.0
            );
            println!(
                "   SIMD Fast:   {:.1} MPx/s ({:.3}ms) - {:.2}x speedup",
                simd_fast_throughput,
                simd_fast_time * 1000.0,
                speedup_fast
            );
            println!(
                "   SIMD Orig:   {:.1} MPx/s ({:.3}ms) - {:.2}x speedup",
                simd_throughput,
                simd_time * 1000.0,
                speedup_original
            );

            // Test with larger image for better SIMD utilization
            let large_image =
                Array3::<u8>::from_shape_fn((1024, 1024, 3), |(h, w, c)| ((h + w + c) % 256) as u8);
            let large_view = large_image.view();

            let start = std::time::Instant::now();
            for _ in 0..5 {
                let _ = resize_bilinear_simd_fast(&large_view, 512, 512).unwrap();
            }
            let large_time = start.elapsed().as_secs_f64() / 5.0;
            let large_throughput = (512 * 512) as f64 / large_time / 1_000_000.0;

            println!(
                "   Large (1024→512): {:.1} MPx/s ({:.3}ms)",
                large_throughput,
                large_time * 1000.0
            );

            // Test Lanczos4 performance
            let start = std::time::Instant::now();
            for _ in 0..5 {
                let _ = resize_lanczos4_simd(&view, 256, 256).unwrap();
            }
            let lanczos_time = start.elapsed().as_secs_f64() / 5.0;
            let lanczos_throughput = (256 * 256) as f64 / lanczos_time / 1_000_000.0;

            println!(
                "   Lanczos4: {:.1} MPx/s ({:.3}ms)",
                lanczos_throughput,
                lanczos_time * 1000.0
            );

            if speedup_fast >= 1.2 {
                println!("   ✅ Fast SIMD provides meaningful speedup!");
            } else {
                println!("   ⚡ SIMD performance similar to scalar - may be memory bound");
            }

            // The test should pass if at least one SIMD version performs reasonably
            assert!(
                speedup_fast >= 0.8 || speedup_original >= 0.8,
                "SIMD should not be significantly slower than scalar"
            );
        }
    }
}
