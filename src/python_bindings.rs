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
use crate::cropping::batch_center_crop_arrays;
#[cfg(feature = "python-bindings")]
use crate::opencv_ops::OpenCVBatchProcessor;

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

// TSR CROPPING OPERATIONS (BENCHMARK WINNERS)

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_crop_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    crop_boxes: Vec<(usize, usize, usize, usize)>, // (x, y, width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
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
    let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();
    
    match batch_center_crop_arrays(&image_views, &target_sizes) {
        Ok(cropped_images) => {
            let py_results: Vec<_> = cropped_images
                .into_iter()
                .map(|cropped| PyArray3::from_array_bound(py, &cropped))
                .collect();
            Ok(py_results)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch center cropping failed: {}",
            e
        ))),
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_random_crop_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(usize, usize)>, // (width, height)  
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    let mut py_results = Vec::with_capacity(images.len());

    for (image, &(target_width, target_height)) in images.iter().zip(target_sizes.iter()) {
        let img_view = image.as_array();
        match random_crop_image_array(&img_view, target_width, target_height) {
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

// TSR LUMINANCE OPERATIONS (BENCHMARK WINNERS)

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_calculate_luminance(
    images: Vec<PyReadonlyArray3<u8>>,
) -> PyResult<Vec<f64>> {
    let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();
    let luminances: Vec<f64> = image_views
        .iter()
        .map(|img| crate::luminance::calculate_luminance_array(img))
        .collect();
    Ok(luminances)
}

// TSR FORMAT CONVERSION OPERATIONS (BENCHMARK WINNERS)

#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn rgb_to_rgba_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    alpha: u8,
) -> PyResult<(Bound<'py, PyArray3<u8>>, f64)> {
    use crate::format_conversion_simd::rgb_to_rgba_optimized;
    
    let image_array = image.as_array();
    let (rgba_data, metrics) = rgb_to_rgba_optimized(&image_array, alpha);
    
    let (height, width, _) = image_array.dim();
    let rgba_array = ndarray::Array3::from_shape_vec((height, width, 4), rgba_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;
    
    let py_array = PyArray3::from_array_bound(py, &rgba_array);
    Ok((py_array, metrics.throughput_mpixels_per_sec))
}

#[cfg(all(feature = "python-bindings", feature = "simd"))]
#[pyfunction]
pub fn rgba_to_rgb_optimized<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, f64)> {
    use crate::format_conversion_simd::rgba_to_rgb_optimized;
    
    let image_array = image.as_array();
    let (height, width, channels) = image_array.dim();
    
    if channels != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Expected RGBA image with 4 channels"
        ));
    }
    
    let rgba_data = image_array.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Image data is not contiguous"
        )
    })?;
    
    let (rgb_data, metrics) = rgba_to_rgb_optimized(rgba_data, width, height);
    let rgb_array = ndarray::Array3::from_shape_vec((height, width, 3), rgb_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;
    
    let py_array = PyArray3::from_array_bound(py, &rgb_array);
    Ok((py_array, metrics.throughput_mpixels_per_sec))
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn rgb_to_rgba_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _alpha: u8,
) -> PyResult<(Bound<'py, PyArray3<u8>>, f64)> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD format conversion not available - compile with simd feature"
    ))
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn rgba_to_rgb_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, f64)> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD format conversion not available - compile with simd feature"
    ))
}

// OPENCV RESIZE OPERATIONS (BENCHMARK WINNERS)

#[cfg(all(feature = "python-bindings", feature = "opencv"))]
#[pyfunction]
pub fn batch_resize_images<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(u32, u32)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    let processor = OpenCVBatchProcessor::new();
    
    let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();
    
    match processor.batch_resize_images(&image_views, &target_sizes) {
        Ok(resized_images) => {
            let py_results: Vec<_> = resized_images
                .into_iter()
                .map(|resized| PyArray3::from_array_bound(py, &resized))
                .collect();
            Ok(py_results)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch resizing failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "opencv")))]
#[pyfunction]
pub fn batch_resize_images<'py>(
    _py: Python<'py>,
    _images: Vec<PyReadonlyArray3<u8>>,
    _target_sizes: Vec<(u32, u32)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    Err(pyo3::exceptions::PyValueError::new_err(
        "Batch resizing failed: OpenCV feature not enabled. Rebuild with --features opencv"
    ))
}

#[cfg(all(feature = "python-bindings", feature = "opencv"))]
#[pyfunction]
pub fn batch_resize_videos<'py>(
    py: Python<'py>,
    videos: Vec<PyReadonlyArray4<u8>>,
    target_sizes: Vec<(u32, u32)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray4<u8>>>> {
    let processor = OpenCVBatchProcessor::new();
    
    let video_views: Vec<_> = videos.iter().map(|vid| vid.as_array()).collect();
    
    match processor.batch_resize_videos(&video_views, &target_sizes) {
        Ok(resized_videos) => {
            let py_results: Vec<_> = resized_videos
                .into_iter()
                .map(|resized| PyArray4::from_array_bound(py, &resized))
                .collect();
            Ok(py_results)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch video resizing failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", not(feature = "opencv")))]
#[pyfunction]
pub fn batch_resize_videos<'py>(
    _py: Python<'py>,
    _videos: Vec<PyReadonlyArray4<u8>>,
    _target_sizes: Vec<(u32, u32)>, // (width, height)
) -> PyResult<Vec<Bound<'py, PyArray4<u8>>>> {
    Err(pyo3::exceptions::PyValueError::new_err(
        "Batch video resizing failed: OpenCV feature not enabled. Rebuild with --features opencv"
    ))
}

// HYBRID PIPELINE (TSR + OPENCV)

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_sft_pipeline<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    crop_boxes: Vec<(usize, usize, usize, usize)>, // (x, y, width, height) for TSR cropping
    target_sizes: Vec<(u32, u32)>, // (width, height) for OpenCV resizing
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    // Step 1: TSR cropping (benchmark winner for mixed-shape batching)
    let mut cropped_images = Vec::with_capacity(images.len());
    for (image, &(x, y, width, height)) in images.iter().zip(crop_boxes.iter()) {
        let img_view = image.as_array();
        match crop_image_array(&img_view, x, y, width, height) {
            Ok(cropped) => cropped_images.push(cropped),
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "SFT cropping failed: {}",
                    e
                )));
            }
        }
    }
    
    // Step 2: OpenCV resizing (benchmark winner for resize performance)
    let processor = OpenCVBatchProcessor::new();
    let cropped_views: Vec<_> = cropped_images.iter().map(|img| img.view()).collect();
    
    match processor.batch_resize_images(&cropped_views, &target_sizes) {
        Ok(resized_images) => {
            let py_results: Vec<_> = resized_images
                .into_iter()
                .map(|resized| PyArray3::from_array_bound(py, &resized))
                .collect();
            Ok(py_results)
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SFT resizing failed: {}",
            e
        ))),
    }
}

// HIGH-PERFORMANCE OPENCV RESIZE (BENCHMARK WINNER - REPLACES METAL)

#[cfg(all(feature = "python-bindings", feature = "opencv"))]
#[pyfunction]
pub fn resize_bilinear_opencv<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::opencv_ops::resize_bilinear_opencv;
    
    let image_array = image.as_array();
    
    match resize_bilinear_opencv(&image_array, target_width, target_height) {
        Ok(resized) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "OpenCV resize failed: {}",
            e
        ))),
    }
}

#[cfg(all(feature = "python-bindings", feature = "opencv"))]
#[pyfunction]
pub fn resize_lanczos4_opencv<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::opencv_ops::resize_lanczos4_opencv;
    
    let image_array = image.as_array();
    
    match resize_lanczos4_opencv(&image_array, target_width, target_height) {
        Ok(resized) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "OpenCV Lanczos4 resize failed: {}",
            e
        ))),
    }
}

// PLACEHOLDER FUNCTIONS FOR NON-OPENCV PLATFORMS

#[cfg(all(feature = "python-bindings", not(feature = "opencv")))]
#[pyfunction]
pub fn resize_bilinear_opencv<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "OpenCV acceleration not available - compile with opencv feature"
    ))
}

#[cfg(all(feature = "python-bindings", not(feature = "opencv")))]
#[pyfunction]  
pub fn resize_lanczos4_opencv<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "OpenCV acceleration not available - compile with opencv feature"
    ))
}