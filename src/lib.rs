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
mod resize_simd;

#[cfg(feature = "simd")]
mod resize_neon_optimized;

#[cfg(feature = "simd")]
mod resize_multicore;

#[cfg(feature = "simd")]
mod resize_x86_optimized;

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

    // High-performance x86 optimizations (available when compiled for x86_64)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_bilinear_x86_optimized,
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

    Ok(())
}
