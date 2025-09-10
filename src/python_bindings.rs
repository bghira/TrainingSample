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
    use rayon::prelude::*;

    let image_views: Vec<_> = images.iter().map(|arr| arr.as_array()).collect();

    let results: Vec<_> = image_views
        .par_iter()
        .map(calculate_luminance_array)
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
