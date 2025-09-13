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

// OpenCV-compatible API layer
mod cv_compat;

// High-performance batch operations for CV APIs
mod cv_batch_ops;

// TRULY optimized batch operations with SIMD
mod cv_batch_ops_optimized;

// TRUE batch operations (actually batched, not just parallel loops)
mod true_batch_ops;

// Python bindings - only when feature is enabled
#[cfg(feature = "python-bindings")]
mod python_bindings;

#[cfg(test)]
mod tests;

// Re-export core functionality for native Rust usage
pub use crate::core::*;

// Re-export OpenCV-compatible APIs
pub use crate::cv_compat::*;

// Re-export batch processing APIs
pub use crate::cv_batch_ops::*;

// Re-export optimized batch processing APIs
pub use crate::cv_batch_ops_optimized::*;

// Re-export TRUE batch processing APIs
pub use crate::true_batch_ops::*;

// Python module definition - only when python-bindings feature is enabled
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn trainingsample(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::python_bindings::*;

    // ZERO-COPY UNSAFE OPERATIONS (MAXIMUM PERFORMANCE)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::batch_crop_images_zero_copy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::batch_center_crop_images_zero_copy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::batch_calculate_luminance_zero_copy,
        m
    )?)?;
    #[cfg(feature = "opencv")]
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::batch_resize_images_zero_copy,
        m
    )?)?;

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

    // HIGH-PERFORMANCE OPENCV RESIZE (BENCHMARK WINNER)
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_bilinear_opencv,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::resize_lanczos4_opencv,
        m
    )?)?;

    // OPENCV-COMPATIBLE API
    m.add_function(wrap_pyfunction!(crate::python_bindings::imdecode_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::cvt_color_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::canny_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::resize_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::python_bindings::fourcc_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::get_opencv_data_path_py,
        m
    )?)?;
    m.add_class::<crate::python_bindings::PyVideoCapture>()?;
    m.add_class::<crate::python_bindings::PyVideoWriter>()?;
    m.add_class::<crate::python_bindings::PyCascadeClassifier>()?;

    // ZERO-COPY ITERATOR API (ultimate performance)
    #[cfg(feature = "opencv")]
    m.add_class::<crate::python_bindings::ResizeIterator>()?;
    #[cfg(feature = "opencv")]
    m.add_function(wrap_pyfunction!(
        crate::python_bindings::batch_resize_images_iterator,
        m
    )?)?;

    // HIGH-PERFORMANCE BATCH PROCESSING API
    m.add_class::<crate::python_bindings::PyBatchProcessor>()?;

    // TRUE BATCH PROCESSING API (actually batched, not just parallel loops)
    m.add_class::<crate::python_bindings::PyTrueBatchProcessor>()?;

    // INTERPOLATION CONSTANTS (OpenCV-compatible)
    m.add("INTER_NEAREST", 0)?;
    m.add("INTER_LINEAR", 1)?;
    m.add("INTER_CUBIC", 2)?;
    m.add("INTER_LANCZOS4", 4)?;

    Ok(())
}
