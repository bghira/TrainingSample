// Core modules - always available
mod core;
mod cropping;
mod loading;
mod luminance;
mod resize;

// SIMD optimizations - only when feature is enabled
#[cfg(feature = "simd")]
mod luminance_simd;

#[cfg(feature = "simd")]
mod luminance_x86_optimized;

#[cfg(feature = "simd")]
pub mod resize_simd;

#[cfg(feature = "simd")]
mod resize_neon_optimized;

#[cfg(feature = "simd")]
mod resize_multicore;

#[cfg(feature = "simd")]
mod resize_x86_optimized;

#[cfg(feature = "simd")]
pub mod resize_optimized;

#[cfg(feature = "simd")]
pub mod resize_simd_advanced;

#[cfg(feature = "metal")]
mod resize_metal;

#[cfg(feature = "simd")]
mod format_conversion_simd;

// Python bindings - only when feature is enabled
#[cfg(feature = "python-bindings")]
mod python_bindings;

#[cfg(test)]
mod tests;

// Re-export core functionality for native Rust usage
pub use crate::core::*;

// Python module definition - only when python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn trainingsample(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::python_bindings::*;

    m.add_function(wrap_pyfunction!(load_image_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch_crop_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_center_crop_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_random_crop_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_resize_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_calculate_luminance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_resize_videos, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sft_pipeline, m)?)?;

    // High-performance SIMD optimizations (cross-platform)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_simd,
        m
    )?)?;

    // High-performance x86 optimizations (available when compiled for x86_64)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_bilinear_x86_optimized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_x86_optimized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::calculate_luminance_x86_optimized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::get_x86_cpu_features,
        m
    )?)?;

    // High-performance ARM NEON optimizations (available when compiled for aarch64)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_neon_optimized,
        m
    )?)?;

    // Ultra-optimized implementations for competitive performance
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_blocked_optimized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_fused_kernel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_adaptive_optimized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_avx512_ultra_wide,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_neon_ultra_wide,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos3_portable_wide,
        m
    )?)?;

    Ok(())
}
