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
    use rayon::prelude::*;

    // Convert to Vec of views for parallel processing
    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    let results: Vec<_> = image_views
        .par_iter()
        .zip(crop_boxes.par_iter())
        .map(|(img, &(x, y, width, height))| crop_image_array(img, x, y, width, height))
        .collect();

    let mut py_results = Vec::new();
    for result in results {
        match result {
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
    use rayon::prelude::*;

    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    let results: Vec<_> = image_views
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| center_crop_image_array(img, width, height))
        .collect();

    let mut py_results = Vec::new();
    for result in results {
        match result {
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
    use rayon::prelude::*;

    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    let results: Vec<_> = image_views
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| random_crop_image_array(img, width, height))
        .collect();

    let mut py_results = Vec::new();
    for result in results {
        match result {
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
pub fn calculate_luminance_x86_optimized(_image: PyReadonlyArray3<u8>) -> PyResult<f64> {
    Ok(0.0) // Return 0.0 as fallback
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
