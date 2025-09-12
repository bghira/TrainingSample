// Core modules - always available
mod core;
mod cropping;
mod loading;
mod luminance;

// SIMD optimizations - only when feature is enabled  
#[cfg(feature = "simd")]
mod luminance_simd;

#[cfg(feature = "simd")]
mod format_conversion_simd;


// OpenCV integration for performance parity
mod opencv_ops;

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

    // TSR CROPPING OPERATIONS (BENCHMARK WINNERS)
    m.add_function(wrap_pyfunction!(load_image_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch_crop_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_center_crop_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_random_crop_images, m)?)?;
    
    // TSR LUMINANCE OPERATIONS (BENCHMARK WINNERS)
    m.add_function(wrap_pyfunction!(batch_calculate_luminance, m)?)?;
    
    // TSR FORMAT CONVERSION OPERATIONS (BENCHMARK WINNERS)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::rgb_to_rgba_optimized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::rgba_to_rgb_optimized,
        m
    )?)?;
    
    // OPENCV RESIZE OPERATIONS (BENCHMARK WINNERS)  
    m.add_function(wrap_pyfunction!(batch_resize_images, m)?)?;
    m.add_function(wrap_pyfunction!(batch_resize_videos, m)?)?;
    
    // HYBRID PIPELINE (TSR + OPENCV)
    m.add_function(wrap_pyfunction!(batch_sft_pipeline, m)?)?;
    
    // HIGH-PERFORMANCE OPENCV RESIZE (BENCHMARK WINNER)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_bilinear_opencv,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos4_opencv,
        m
    )?)?;

    Ok(())
}
