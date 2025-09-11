#![allow(clippy::useless_conversion)]

#[cfg(feature = "python-bindings")]
use numpy::{PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyBytes;

#[cfg(feature = "python-bindings")]
use crate::core::*;

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn load_image_batch(py: Python, image_paths: Vec<String>) -> PyResult<Vec<PyObject>> {
    use rayon::prelude::*;

    let results: Vec<_> = image_paths
        .par_iter()
        .map(|path| load_image_from_path(path))
        .collect();

    let mut py_results = Vec::new();
    for result in results {
        match result {
            Ok(image_data) => {
                let py_bytes = PyBytes::new_bound(py, &image_data);
                py_results.push(py_bytes.into_any().unbind());
            }
            Err(_) => {
                py_results.push(py.None());
            }
        }
    }
    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_crop_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    crop_boxes: Vec<(usize, usize, usize, usize)>, // (x, y, width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    // Process sequentially for better performance with simple operations
    let mut py_results = Vec::with_capacity(images.len());

    for (image, &(x, y, width, height)) in images.iter().zip(crop_boxes.iter()) {
        let img_view = image.as_array();
        match crop_image_array(&img_view, x, y, width, height) {
            Ok(cropped) => {
                let py_array = PyArray3::from_array_bound(py, &cropped);
                py_results.push(py_array);
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Cropping failed: {}",
                    e
                )));
            }
        }
    }
    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_center_crop_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(usize, usize)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    // For simple operations like cropping, avoid parallel overhead
    // Process sequentially for better performance with small batches
    let mut py_results = Vec::with_capacity(images.len());

    for (image, &(width, height)) in images.iter().zip(target_sizes.iter()) {
        let img_view = image.as_array();
        match center_crop_image_array(&img_view, width, height) {
            Ok(cropped) => {
                let py_array = PyArray3::from_array_bound(py, &cropped);
                py_results.push(py_array);
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Center cropping failed: {}",
                    e
                )));
            }
        }
    }
    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_random_crop_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(usize, usize)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    // Process sequentially for better performance with simple operations
    let mut py_results = Vec::with_capacity(images.len());

    for (image, &(width, height)) in images.iter().zip(target_sizes.iter()) {
        let img_view = image.as_array();
        match random_crop_image_array(&img_view, width, height) {
            Ok(cropped) => {
                let py_array = PyArray3::from_array_bound(py, &cropped);
                py_results.push(py_array);
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Random cropping failed: {}",
                    e
                )));
            }
        }
    }
    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_resize_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(u32, u32)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    use rayon::prelude::*;

    // For compute-intensive operations like resizing, parallel processing provides
    // significant performance benefits that outweigh memory overhead
    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    let results: Vec<_> = image_views
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| resize_image_array(img, width, height))
        .collect();

    let mut py_results = Vec::new();
    for result in results {
        match result {
            Ok(resized) => {
                let py_array = PyArray3::from_array_bound(py, &resized);
                py_results.push(py_array);
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Resizing failed: {}",
                    e
                )));
            }
        }
    }
    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_calculate_luminance(images: Vec<PyReadonlyArray3<u8>>) -> PyResult<Vec<f64>> {
    use crate::luminance::calculate_luminance_array_sequential;

    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    // Use parallel batch processing with sequential individual processing
    // to avoid nested parallelism that causes performance degradation
    use rayon::prelude::*;
    let results: Vec<_> = image_views
        .par_iter()
        .map(calculate_luminance_array_sequential)
        .collect();

    Ok(results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_resize_videos<'py>(
    py: Python<'py>,
    videos: Vec<PyReadonlyArray4<u8>>,
    target_sizes: Vec<(u32, u32)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray4<u8>>>> {
    use rayon::prelude::*;

    // Video resizing is extremely compute-intensive, so parallel processing is beneficial
    let video_views: Vec<_> = videos.iter().map(|arr| arr.as_array()).collect();

    let results: Vec<_> = video_views
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(video, &(width, height))| resize_video_array(video, width, height))
        .collect();

    let mut py_results = Vec::new();
    for result in results {
        match result {
            Ok(resized) => {
                let py_array = PyArray4::from_array_bound(py, &resized);
                py_results.push(py_array);
            }
            Err(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Video resizing failed",
                ));
            }
        }
    }
    Ok(py_results)
}

// End-to-end batch SFT pipeline - TSR's main strength
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_sft_pipeline<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    crop_size: (usize, usize), // (width, height) for center crop
    resize_size: (u32, u32),   // (width, height) for final resize
) -> PyResult<(Vec<Bound<'py, PyArray3<u8>>>, Vec<f64>)> {
    use crate::luminance::calculate_luminance_array_sequential;
    use rayon::prelude::*;

    // Process entire pipeline in parallel without Python boundary crossings
    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    let results: Result<Vec<_>, anyhow::Error> = image_views
        .par_iter()
        .map(|img| {
            // Step 1: Center crop in Rust
            let cropped = center_crop_image_array(img, crop_size.0, crop_size.1)?;

            // Step 2: Resize in Rust
            let resized = resize_image_array(&cropped.view(), resize_size.0, resize_size.1)?;

            // Step 3: Calculate luminance in Rust
            let luminance = calculate_luminance_array_sequential(&resized.view());

            Ok((resized, luminance))
        })
        .collect();

    let processed_results = results.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("SFT pipeline failed: {}", e))
    })?;

    // Convert results back to Python only once at the end
    let mut py_images = Vec::with_capacity(processed_results.len());
    let mut py_luminances = Vec::with_capacity(processed_results.len());

    for (resized, luminance) in processed_results {
        let py_array = PyArray3::from_array_bound(py, &resized);
        py_images.push(py_array);
        py_luminances.push(luminance);
    }

    Ok((py_images, py_luminances))
}

// High-performance x86 optimization Python bindings
#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "x86_64"))]
#[pyfunction]
pub fn resize_bilinear_x86_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::resize_x86_optimized::resize_bilinear_x86_optimized;

    let image_array = image.as_array();

    match resize_bilinear_x86_optimized(&image_array, target_width, target_height) {
        Ok(resized) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x86 optimized resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "x86_64"))]
#[pyfunction]
pub fn resize_lanczos3_x86_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::resize_x86_optimized::resize_lanczos3_x86_optimized;

    let image_array = image.as_array();

    match resize_lanczos3_x86_optimized(&image_array, target_width, target_height) {
        Ok(resized) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x86 optimized Lanczos3 resize failed: {}",
            e
        ))),
    }
}

// High-performance ARM NEON optimization Python bindings
#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "aarch64"))]
#[pyfunction]
pub fn resize_lanczos3_neon_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::resize_neon_optimized::resize_lanczos3_neon_optimized_safe;

    let image_array = image.as_array();

    match resize_lanczos3_neon_optimized_safe(&image_array, target_width, target_height) {
        Ok(resized) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "NEON optimized Lanczos3 resize failed: {}",
            e
        ))),
    }
}

// Generic SIMD Lanczos3 function (works on all platforms)
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn resize_lanczos3_simd<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::resize_simd::{resize_image_optimized, FilterType};

    let image_array = image.as_array();

    match resize_image_optimized(
        &image_array,
        target_width,
        target_height,
        FilterType::Lanczos3,
    ) {
        Ok((resized, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SIMD Lanczos3 resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "x86_64"))]
#[pyfunction]
pub fn calculate_luminance_x86_optimized(image: PyReadonlyArray3<u8>) -> PyResult<f64> {
    use crate::luminance_x86_optimized::calculate_luminance_x86_optimized;

    let image_array = image.as_array();
    let luminance = calculate_luminance_x86_optimized(&image_array)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(luminance)
}

#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "x86_64"))]
#[pyfunction]
pub fn get_x86_cpu_features() -> PyResult<std::collections::HashMap<String, bool>> {
    use crate::resize_x86_optimized::detect_cpu_features;

    let features = detect_cpu_features();
    let mut feature_map = std::collections::HashMap::new();

    feature_map.insert("avx512f".to_string(), features.has_avx512f);
    feature_map.insert("avx512bw".to_string(), features.has_avx512bw);
    feature_map.insert("avx512dq".to_string(), features.has_avx512dq);
    feature_map.insert("avx2".to_string(), features.has_avx2);
    feature_map.insert("fma".to_string(), features.has_fma);
    feature_map.insert("sse41".to_string(), features.has_sse41);
    feature_map.insert("is_amd_zen".to_string(), features.is_amd_zen);

    Ok(feature_map)
}

// Fallback functions for non-x86 platforms
#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "x86_64"))
))]
#[pyfunction]
pub fn resize_bilinear_x86_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "x86 optimizations are not available on this platform",
    ))
}

#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "x86_64"))
))]
#[pyfunction]
pub fn resize_lanczos3_x86_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "x86 optimizations are not available on this platform",
    ))
}

#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "x86_64"))
))]
#[pyfunction]
pub fn calculate_luminance_x86_optimized(_image: PyReadonlyArray3<u8>) -> PyResult<f64> {
    Ok(0.0) // Return 0.0 as fallback
}

// Fallback functions for non-ARM64 platforms
#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "aarch64"))
))]
#[pyfunction]
pub fn resize_lanczos3_neon_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "NEON optimizations are not available on this platform",
    ))
}

#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "x86_64"))
))]
#[pyfunction]
pub fn get_x86_cpu_features() -> PyResult<std::collections::HashMap<String, bool>> {
    let mut feature_map = std::collections::HashMap::new();
    feature_map.insert("avx512f".to_string(), false);
    feature_map.insert("avx512bw".to_string(), false);
    feature_map.insert("avx512dq".to_string(), false);
    feature_map.insert("avx2".to_string(), false);
    feature_map.insert("fma".to_string(), false);
    feature_map.insert("sse41".to_string(), false);
    feature_map.insert("is_amd_zen".to_string(), false);
    Ok(feature_map)
}

// New optimized implementations for competitive performance

#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_blocked_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    match crate::resize_optimized::resize_lanczos3_blocked_optimized(
        &img_view,
        target_width,
        target_height,
    ) {
        Ok((result, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &result);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Blocked resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_fused_kernel<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    match crate::resize_optimized::resize_lanczos3_fused_kernel(
        &img_view,
        target_width,
        target_height,
    ) {
        Ok((result, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &result);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Fused kernel resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_adaptive_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    match crate::resize_optimized::resize_lanczos3_adaptive_optimized(
        &img_view,
        target_width,
        target_height,
    ) {
        Ok((result, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &result);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Adaptive resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "x86_64"))]
#[pyfunction]
pub fn resize_lanczos3_avx512_ultra_wide<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    match crate::resize_simd_advanced::resize_lanczos3_avx512_ultra_wide(
        &img_view,
        target_width,
        target_height,
    ) {
        Ok((result, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &result);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "AVX-512 resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd", target_arch = "aarch64"))]
#[pyfunction]
pub fn resize_lanczos3_neon_ultra_wide<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    match crate::resize_simd_advanced::resize_lanczos3_neon_ultra_wide(
        &img_view,
        target_width,
        target_height,
    ) {
        Ok((result, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &result);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "NEON ultra-wide resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_portable_wide<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    match crate::resize_simd_advanced::resize_lanczos3_portable_wide(
        &img_view,
        target_width,
        target_height,
    ) {
        Ok((result, _metrics)) => {
            let py_array = PyArray3::from_array_bound(py, &result);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Portable wide resize failed: {}",
            e
        ))),
    }
}

// Fallback functions for non-SIMD builds
#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_blocked_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD optimizations are not available in this build",
    ))
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_fused_kernel<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD optimizations are not available in this build",
    ))
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_adaptive_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD optimizations are not available in this build",
    ))
}

#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "x86_64"))
))]
#[pyfunction]
pub fn resize_lanczos3_avx512_ultra_wide<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "AVX-512 optimizations are not available on this platform",
    ))
}

#[cfg(all(
    feature = "python-bindings",
    not(all(feature = "simd", target_arch = "aarch64"))
))]
#[pyfunction]
pub fn resize_lanczos3_neon_ultra_wide<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "NEON optimizations are not available on this platform",
    ))
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_portable_wide<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD optimizations are not available in this build",
    ))
}

/// Ultra-optimized Lanczos3 resize with zero-copy operations and extreme vectorization
/// Targets OpenCV performance parity through aggressive memory access optimization
#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_ultra_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let array = image.as_array();
    let start = std::time::Instant::now();
    
    match crate::resize_optimized::resize_lanczos3_ultra_optimized(&array, target_width, target_height) {
        Ok((result, metrics)) => {
            let elapsed = start.elapsed();
            eprintln!(
                "🚀 Ultra-optimized resize: {}x{} → {}x{} in {:.1}ms ({:.1} MPx/s, {})",
                array.dim().1, array.dim().0,
                target_width, target_height,
                elapsed.as_secs_f64() * 1000.0,
                metrics.throughput_mpixels_per_sec,
                metrics.implementation
            );
            
            // Convert result to numpy array
            let result_array = PyArray3::from_array_bound(py, &result.view());
            Ok(result_array)
        },
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Ultra-optimized resize failed: {}", e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_ultra_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Ultra-optimized resize requires SIMD features",
    ))
}

/// 🚀 Metal GPU-accelerated bilinear resize - should beat Pillow by >>3.3x!
#[cfg(all(feature = "python-bindings", feature = "metal"))]
#[pyfunction]
pub fn resize_bilinear_metal_gpu<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    let start = std::time::Instant::now();
    
    match crate::resize_metal::MetalResizeEngine::new() {
        Ok(engine) => {
            match engine.resize_bilinear_gpu(&img_view, target_width, target_height) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let pixels = (img_view.dim().0 * img_view.dim().1) as f64;
                    let throughput = pixels / elapsed.as_secs_f64() / 1_000_000.0;
                    
                    eprintln!(
                        "🚀 Metal GPU Bilinear: {:.1} MPx/s ({}×{} → {}×{}) in {:.2}ms", 
                        throughput,
                        img_view.dim().1, img_view.dim().0,
                        target_width, target_height,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    
                    Ok(PyArray3::from_array_bound(py, &result))
                }
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Metal GPU resize failed: {}", e
                ))),
            }
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to initialize Metal GPU: {}", e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "metal")))]
#[pyfunction]
pub fn resize_bilinear_metal_gpu<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Metal GPU acceleration not available (compile with --features metal)",
    ))
}

/// 🚀 Metal GPU-accelerated Lanczos4 resize - ultimate quality + speed!
#[cfg(all(feature = "python-bindings", feature = "metal"))]
#[pyfunction] 
pub fn resize_lanczos4_metal_gpu<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    let start = std::time::Instant::now();
    
    match crate::resize_metal::MetalResizeEngine::new() {
        Ok(engine) => {
            match engine.resize_lanczos4_gpu(&img_view, target_width, target_height) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let pixels = (img_view.dim().0 * img_view.dim().1) as f64;
                    let throughput = pixels / elapsed.as_secs_f64() / 1_000_000.0;
                    
                    eprintln!(
                        "🚀 Metal GPU Lanczos4: {:.1} MPx/s ({}×{} → {}×{}) in {:.2}ms",
                        throughput,
                        img_view.dim().1, img_view.dim().0,
                        target_width, target_height,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    
                    Ok(PyArray3::from_array_bound(py, &result))
                }
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Metal GPU Lanczos4 resize failed: {}", e
                ))),
            }
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to initialize Metal GPU: {}", e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "metal")))]
#[pyfunction]
pub fn resize_lanczos4_metal_gpu<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Metal GPU acceleration not available (compile with --features metal)",
    ))
}

/// 🚀 LANCIR-inspired cache-optimized resize - targeting OpenCV performance!
#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_cache_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    let start = std::time::Instant::now();
    
    let resizer = crate::resize_cache_optimized::CacheOptimizedResizer::new();
    match resizer.resize_lanczos3_cache_optimized(&img_view, target_width, target_height) {
        Ok((result, metrics)) => {
            let elapsed = start.elapsed();
            
            eprintln!(
                "🔄 Cache-Optimized Lanczos3: {:.1} MPx/s ({}×{} → {}×{}) in {:.2}ms [batch_size: {}]",
                metrics.throughput_mpixels_per_sec,
                img_view.dim().1, img_view.dim().0,
                target_width, target_height,
                elapsed.as_secs_f64() * 1000.0,
                metrics.batch_size.unwrap_or(0)
            );
            
            Ok(PyArray3::from_array_bound(py, &result))
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Cache-optimized resize failed: {}", e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_cache_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "Cache-optimized resize requires SIMD features",
    ))
}

/// 🚀 Intel IPP-inspired resize - TARGET: 4x speedup with weight table caching!
#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn resize_lanczos3_ipp_inspired<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_view = image.as_array();
    let start = std::time::Instant::now();
    
    let resizer = crate::resize_ipp_inspired::IPPInspiredResizer::new();
    match resizer.resize_lanczos3_ipp_inspired(&img_view, target_width, target_height) {
        Ok((result, metrics)) => {
            let elapsed = start.elapsed();
            
            eprintln!(
                "🚀 IPP-Inspired Lanczos3: {:.1} MPx/s ({}×{} → {}×{}) in {:.2}ms [cache hits: {}, misses: {}]",
                metrics.throughput_mpixels_per_sec,
                img_view.dim().1, img_view.dim().0,
                target_width, target_height,
                elapsed.as_secs_f64() * 1000.0,
                metrics.weight_table_hits,
                metrics.weight_table_misses
            );
            
            Ok(PyArray3::from_array_bound(py, &result))
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "IPP-inspired resize failed: {}", e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn resize_lanczos3_ipp_inspired<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "IPP-inspired resize requires SIMD features",
    ))
}
