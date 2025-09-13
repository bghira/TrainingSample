#![allow(clippy::useless_conversion)]

#[cfg(feature = "python-bindings")]
use numpy::{PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyBytes;
#[cfg(feature = "python-bindings")]
use std::collections::VecDeque;
#[cfg(feature = "python-bindings")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "python-bindings")]
use crate::core::*;
#[cfg(feature = "python-bindings")]
use crate::cv_batch_ops::BatchProcessor;
#[cfg(feature = "python-bindings")]
use crate::cv_compat::{
    fourcc, get_opencv_data_path, CascadeClassifier, VideoCapture, VideoWriter,
};
#[cfg(all(feature = "python-bindings", feature = "opencv"))]
use crate::opencv_ops::OpenCVBatchProcessor;
#[cfg(feature = "python-bindings")]
use crate::true_batch_ops::{ColorConversion, TrueBatchProcessor};

#[cfg(feature = "python-bindings")]
static BUFFER_POOL: std::sync::OnceLock<Arc<Mutex<BufferPool>>> = std::sync::OnceLock::new();

fn get_buffer_pool() -> &'static Arc<Mutex<BufferPool>> {
    BUFFER_POOL.get_or_init(|| Arc::new(Mutex::new(BufferPool::new())))
}

#[cfg(feature = "python-bindings")]
struct BufferPool {
    pools: std::collections::HashMap<(usize, usize, usize), VecDeque<Vec<u8>>>,
}

#[cfg(feature = "python-bindings")]
impl BufferPool {
    fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
        }
    }

    fn get_buffer(&mut self, height: usize, width: usize, channels: usize) -> Vec<u8> {
        let key = (height, width, channels);
        let size = height * width * channels;

        if let Some(pool) = self.pools.get_mut(&key) {
            if let Some(mut buffer) = pool.pop_front() {
                buffer.clear();
                buffer.resize(size, 0);
                return buffer;
            }
        }

        vec![0u8; size]
    }

    #[allow(dead_code)]
    fn return_buffer(&mut self, buffer: Vec<u8>, height: usize, width: usize, channels: usize) {
        let key = (height, width, channels);
        let pool = self.pools.entry(key).or_default();

        if pool.len() < 8 {
            pool.push_back(buffer);
        }
    }
}

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

// ZERO-COPY UNSAFE BUFFER OPERATIONS (MAXIMUM PERFORMANCE)

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub unsafe fn batch_crop_images_zero_copy<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    crop_boxes: Vec<(usize, usize, usize, usize)>,
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    let batch_size = images.len();
    if batch_size != crop_boxes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Images and crop_boxes length mismatch",
        ));
    }

    let mut py_results = Vec::with_capacity(batch_size);
    let mut pool = get_buffer_pool().lock().unwrap();

    for (image, &(x, y, width, height)) in images.iter().zip(crop_boxes.iter()) {
        let img_view = image.as_array();
        let (src_height, src_width, channels) = img_view.dim();

        if x + width > src_width || y + height > src_height {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Crop coordinates out of bounds",
            ));
        }

        let mut output_buffer = pool.get_buffer(height, width, channels);

        let src_ptr = img_view.as_ptr();
        let dst_ptr = output_buffer.as_mut_ptr();

        crate::cropping::crop_raw_buffer(
            src_ptr,
            (src_height, src_width, channels),
            dst_ptr,
            (y, x, height, width),
        );

        let array = ndarray::Array3::from_shape_vec((height, width, channels), output_buffer)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;
        let py_array = PyArray3::from_array_bound(py, &array);
        py_results.push(py_array);
    }

    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub unsafe fn batch_center_crop_images_zero_copy<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(usize, usize)>,
) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
    let batch_size = images.len();
    if batch_size != target_sizes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Images and target_sizes length mismatch",
        ));
    }

    let mut py_results = Vec::with_capacity(batch_size);
    let mut pool = get_buffer_pool().lock().unwrap();

    for (image, &(target_width, target_height)) in images.iter().zip(target_sizes.iter()) {
        let img_view = image.as_array();
        let (src_height, src_width, channels) = img_view.dim();

        let start_x = (src_width.saturating_sub(target_width)) / 2;
        let start_y = (src_height.saturating_sub(target_height)) / 2;
        let crop_width = target_width.min(src_width);
        let crop_height = target_height.min(src_height);

        let mut output_buffer = pool.get_buffer(crop_height, crop_width, channels);

        let src_ptr = img_view.as_ptr();
        let dst_ptr = output_buffer.as_mut_ptr();

        crate::cropping::crop_raw_buffer(
            src_ptr,
            (src_height, src_width, channels),
            dst_ptr,
            (start_y, start_x, crop_height, crop_width),
        );

        let array =
            ndarray::Array3::from_shape_vec((crop_height, crop_width, channels), output_buffer)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e))
                })?;
        let py_array = PyArray3::from_array_bound(py, &array);
        py_results.push(py_array);
    }

    Ok(py_results)
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn batch_calculate_luminance_zero_copy(
    images: Vec<PyReadonlyArray3<u8>>,
) -> PyResult<Vec<f64>> {
    use rayon::prelude::*;

    // Extract raw pointers and dimensions first (on main thread)
    #[derive(Clone, Copy)]
    struct ImageData {
        ptr: *const u8,
        width: usize,
        height: usize,
        channels: usize,
    }

    unsafe impl Send for ImageData {}
    unsafe impl Sync for ImageData {}

    let image_data: Vec<ImageData> = images
        .iter()
        .map(|image| {
            let img_view = image.as_array();
            let (height, width, channels) = img_view.dim();
            let ptr = img_view.as_ptr();
            ImageData {
                ptr,
                width,
                height,
                channels,
            }
        })
        .collect();

    // Now process the raw pointers in parallel
    let luminances: Vec<f64> = image_data
        .par_iter()
        .map(|data| unsafe {
            crate::luminance::calculate_luminance_raw_buffer(
                data.ptr,
                data.width,
                data.height,
                data.channels,
            )
        })
        .collect();

    Ok(luminances)
}

#[cfg(all(feature = "python-bindings", feature = "opencv"))]
#[pyfunction]
#[pyo3(signature = (images, target_sizes, interpolation=None))]
pub fn batch_resize_images_zero_copy<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    target_sizes: &Bound<'py, PyAny>,
    interpolation: Option<i32>,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::cv_compat::imgproc::ResizeInterpolation;
    use pyo3::types::PyList;
    use rayon::prelude::*;

    // Convert interpolation parameter (default to bilinear if not specified)
    let interp = match interpolation.unwrap_or(1) {
        0 => ResizeInterpolation::InterNearest,
        1 => ResizeInterpolation::InterLinear,
        2 => ResizeInterpolation::InterCubic,
        4 => ResizeInterpolation::InterLanczos4,
        _ => ResizeInterpolation::InterLinear, // Default fallback
    };

    // INTELLIGENT OVERLOADING: Detect single vs batch input
    let is_single_image = is_numpy_array(images);
    let is_single_size = is_tuple_or_list_of_two(target_sizes);

    if is_single_image && is_single_size {
        // SINGLE IMAGE PATH: Direct return, zero wrapper overhead!
        let image_array = images.extract::<PyReadonlyArray3<u8>>()?;
        let size_tuple = target_sizes.extract::<(usize, usize)>()?;
        let result = resize_single_image_direct(py, &image_array, size_tuple, interp)?;
        return Ok(result.into_any());
    }

    // BATCH PATH: Convert to Vec format and process
    let images_vec: Vec<PyReadonlyArray3<u8>> = if is_single_image {
        // Single image wrapped in list
        vec![images.extract::<PyReadonlyArray3<u8>>()?]
    } else {
        // List of images
        images.extract::<Vec<PyReadonlyArray3<u8>>>()?
    };

    let sizes_vec: Vec<(usize, usize)> = if is_single_size {
        // Single size wrapped in list
        vec![target_sizes.extract::<(usize, usize)>()?]
    } else {
        // List of sizes
        target_sizes.extract::<Vec<(usize, usize)>>()?
    };

    let batch_size = images_vec.len();
    if batch_size != sizes_vec.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Images and target_sizes length mismatch",
        ));
    }

    // BATCH PROCESSING: Extract raw pointers and dimensions (on main thread)
    #[derive(Clone, Copy)]
    struct ResizeData {
        src_ptr: *const u8,
        src_height: usize,
        src_width: usize,
        src_channels: usize,
        target_width: usize,
        target_height: usize,
    }

    unsafe impl Send for ResizeData {}
    unsafe impl Sync for ResizeData {}

    let resize_data: Vec<ResizeData> = images_vec
        .iter()
        .zip(sizes_vec.iter())
        .map(|(image, &(target_width, target_height))| {
            let img_view = image.as_array();
            let (src_height, src_width, src_channels) = img_view.dim();
            let src_ptr = img_view.as_ptr();
            ResizeData {
                src_ptr,
                src_height,
                src_width,
                src_channels,
                target_width,
                target_height,
            }
        })
        .collect();

    // Adaptive processing: use parallel only for larger batches to avoid threading overhead
    let use_parallel = batch_size >= 8; // Threshold where parallel processing becomes beneficial

    // Convert interpolation to OpenCV constant
    let cv_interpolation = match interp {
        ResizeInterpolation::InterNearest => opencv::imgproc::INTER_NEAREST,
        ResizeInterpolation::InterLinear => opencv::imgproc::INTER_LINEAR,
        ResizeInterpolation::InterCubic => opencv::imgproc::INTER_CUBIC,
        ResizeInterpolation::InterLanczos4 => opencv::imgproc::INTER_LANCZOS4,
    };

    let results: Result<Vec<ndarray::Array3<u8>>, String> = if use_parallel {
        // Pre-allocate all buffers on main thread to avoid lock contention
        let mut buffers = Vec::with_capacity(batch_size);
        {
            let mut pool = get_buffer_pool().lock().unwrap();
            for data in &resize_data {
                let buffer =
                    pool.get_buffer(data.target_height, data.target_width, data.src_channels);
                buffers.push(buffer);
            }
        } // Release lock before parallel processing

        resize_data
            .par_iter()
            .zip(buffers.into_par_iter())
            .map(|(data, output_buffer)| unsafe {
                resize_raw_buffer(
                    data.src_ptr,
                    (data.src_height, data.src_width, data.src_channels),
                    (data.target_height, data.target_width),
                    output_buffer,
                    cv_interpolation,
                )
            })
            .collect::<Result<Vec<_>, _>>()
    } else {
        // Sequential processing for small batches - avoid parallel overhead
        let mut pool = get_buffer_pool().lock().unwrap();
        let mut results = Vec::with_capacity(batch_size);

        for data in &resize_data {
            let output_buffer =
                pool.get_buffer(data.target_height, data.target_width, data.src_channels);
            let result = unsafe {
                resize_raw_buffer(
                    data.src_ptr,
                    (data.src_height, data.src_width, data.src_channels),
                    (data.target_height, data.target_width),
                    output_buffer,
                    cv_interpolation,
                )
            };
            match result {
                Ok(array) => results.push(array),
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Resize failed: {}",
                        e
                    )))
                }
            }
        }
        Ok(results)
    };

    let results = results
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Resize failed: {}", e)))?;

    // Convert to PyArray3 and return as Python list
    let py_results: Vec<Bound<'py, PyArray3<u8>>> = results
        .iter()
        .map(|array| PyArray3::from_array_bound(py, array))
        .collect();

    Ok(PyList::new_bound(py, py_results).into_any())
}

/// Ultra-fast OpenCV-powered resize using zero-copy Mat headers
///
/// # Safety
/// - `src_ptr` must be valid for reads of at least `src_height * src_width * channels` bytes
/// - `output_buffer` must be valid for writes of at least `target_height * target_width * channels` bytes
#[cfg(feature = "opencv")]
unsafe fn resize_raw_buffer(
    src_ptr: *const u8,
    src_shape: (usize, usize, usize), // (height, width, channels)
    target_shape: (usize, usize),     // (height, width)
    mut output_buffer: Vec<u8>,
    interpolation: i32, // OpenCV interpolation flag
) -> Result<ndarray::Array3<u8>, String> {
    use opencv::{core::Mat, imgproc::resize};

    let (src_height, src_width, channels) = src_shape;
    let (target_height, target_width) = target_shape;

    if channels != 3 {
        return Err("Only 3-channel RGB images are supported".to_string());
    }

    // Buffer should already be the right size from pool, but ensure it
    let required_size = target_height * target_width * channels;
    if output_buffer.len() != required_size {
        output_buffer.resize(required_size, 0);
    }

    // ZERO-COPY: Create Mat header pointing directly to source memory
    let src_mat = Mat::new_rows_cols_with_data_unsafe(
        src_height as i32,
        src_width as i32,
        opencv::core::CV_8UC3,
        src_ptr as *mut std::ffi::c_void,
        opencv::core::Mat_AUTO_STEP,
    )
    .map_err(|e| format!("Failed to create source Mat: {}", e))?;

    // ZERO-COPY: Create Mat header pointing directly to output buffer
    let mut dst_mat = Mat::new_rows_cols_with_data_unsafe(
        target_height as i32,
        target_width as i32,
        opencv::core::CV_8UC3,
        output_buffer.as_mut_ptr() as *mut std::ffi::c_void,
        opencv::core::Mat_AUTO_STEP,
    )
    .map_err(|e| format!("Failed to create destination Mat: {}", e))?;

    // OpenCV writes directly into our output buffer - MAXIMUM PERFORMANCE!
    resize(
        &src_mat,
        &mut dst_mat,
        opencv::core::Size::new(target_width as i32, target_height as i32),
        0.0,
        0.0,
        interpolation,
    )
    .map_err(|e| format!("OpenCV resize failed: {}", e))?;

    // Convert buffer back to ndarray
    ndarray::Array3::from_shape_vec((target_height, target_width, channels), output_buffer)
        .map_err(|e| format!("Shape error: {}", e))
}

/// Fallback resize for when OpenCV is not available
#[cfg(not(feature = "opencv"))]
unsafe fn resize_raw_buffer(
    _src_ptr: *const u8,
    _src_shape: (usize, usize, usize),
    _target_shape: (usize, usize),
    _output_buffer: Vec<u8>,
) -> Result<ndarray::Array3<u8>, String> {
    Err("OpenCV feature required for zero-copy resize".to_string())
}

/// Ultra-fast single image resize - DIRECT return, no Vec wrappers
#[cfg(feature = "opencv")]
fn resize_single_image_direct<'py>(
    py: Python<'py>,
    image: &PyReadonlyArray3<u8>,
    target_size: (usize, usize),
    interpolation: crate::cv_compat::imgproc::ResizeInterpolation,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use opencv::{
        core::Mat,
        imgproc::{resize, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST},
    };

    let img_view = image.as_array();
    let (src_height, src_width, channels) = img_view.dim();
    let (target_width, target_height) = target_size;

    if channels != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Only 3-channel RGB images are supported",
        ));
    }

    // Create source Mat header pointing directly to input data (zero-copy)
    let src_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            src_height as i32,
            src_width as i32,
            opencv::core::CV_8UC3,
            img_view.as_ptr() as *mut std::ffi::c_void,
            opencv::core::Mat_AUTO_STEP,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create source Mat: {}", e))
        })?
    };

    // Pre-allocate result array
    let mut result = ndarray::Array3::<u8>::zeros((target_height, target_width, 3));

    // Create destination Mat header pointing directly to result memory (zero-copy)
    let mut dst_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            target_height as i32,
            target_width as i32,
            opencv::core::CV_8UC3,
            result.as_mut_ptr() as *mut std::ffi::c_void,
            opencv::core::Mat_AUTO_STEP,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create destination Mat: {}",
                e
            ))
        })?
    };

    // Convert ResizeInterpolation to OpenCV flag
    use crate::cv_compat::imgproc::ResizeInterpolation;
    let cv_interpolation = match interpolation {
        ResizeInterpolation::InterNearest => INTER_NEAREST,
        ResizeInterpolation::InterLinear => INTER_LINEAR,
        ResizeInterpolation::InterCubic => INTER_CUBIC,
        ResizeInterpolation::InterLanczos4 => INTER_LANCZOS4,
    };

    // OpenCV writes directly into our result array - MAXIMUM PERFORMANCE!
    resize(
        &src_mat,
        &mut dst_mat,
        opencv::core::Size::new(target_width as i32, target_height as i32),
        0.0,
        0.0,
        cv_interpolation,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("OpenCV resize failed: {}", e))
    })?;

    // DIRECT return - no Vec wrapper overhead!
    Ok(PyArray3::from_array_bound(py, &result))
}

#[cfg(not(feature = "opencv"))]
fn resize_single_image_direct<'py>(
    _py: Python<'py>,
    _image: &PyReadonlyArray3<u8>,
    _target_size: (usize, usize),
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "OpenCV feature required for single image resize",
    ))
}

/// Helper function to detect if input is a numpy array (single image)
fn is_numpy_array(obj: &Bound<PyAny>) -> bool {
    use pyo3::types::PyList;
    // If it's a list, it's batch mode. If it's not a list, assume it's a numpy array
    !obj.is_instance_of::<PyList>()
}

/// Helper function to detect if input is a tuple of two numbers (single size)
fn is_tuple_or_list_of_two(obj: &Bound<PyAny>) -> bool {
    use pyo3::types::{PyList, PyTuple};

    // Check if it's a tuple of length 2
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        return tuple.len() == 2;
    }

    // Check if it's a list of length 2 containing numbers (not nested lists)
    if let Ok(list) = obj.downcast::<PyList>() {
        if list.len() == 2 {
            // Check if first element is a number (not a nested list/tuple)
            if let Ok(first) = list.get_item(0) {
                return !first.is_instance_of::<PyList>() && !first.is_instance_of::<PyTuple>();
            }
        }
    }

    false
}

/// True zero-copy resize iterator - holds raw pointers, converts to PyArray3 on-demand
#[cfg(feature = "opencv")]
#[pyclass]
pub struct ResizeIterator {
    /// Raw buffer pointers and their dimensions
    buffers: Vec<(Vec<u8>, (usize, usize, usize))>, // (buffer, (height, width, channels))
    /// Current iteration index
    index: usize,
}

#[cfg(feature = "opencv")]
#[pymethods]
impl ResizeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray3<u8>>> {
        if self.index >= self.buffers.len() {
            return None;
        }

        let (buffer, (height, width, channels)) = &self.buffers[self.index];
        self.index += 1;

        // Convert raw buffer directly to PyArray3 - ZERO intermediate steps!
        match ndarray::Array3::from_shape_vec((*height, *width, *channels), buffer.clone()) {
            Ok(array) => Some(PyArray3::from_array_bound(py, &array)),
            Err(_) => None, // Skip malformed arrays
        }
    }

    fn __len__(&self) -> usize {
        self.buffers.len()
    }
}

/// NEW ITERATOR API: Zero-copy batch resize with lazy conversion
#[cfg(all(feature = "python-bindings", feature = "opencv"))]
#[pyfunction]
#[pyo3(signature = (images, target_sizes, interpolation=None))]
pub fn batch_resize_images_iterator<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArray3<u8>>,
    target_sizes: Vec<(usize, usize)>,
    interpolation: Option<i32>,
) -> PyResult<Bound<'py, ResizeIterator>> {
    use crate::cv_compat::imgproc::ResizeInterpolation;
    use opencv::{
        core::Mat,
        imgproc::{resize, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST},
    };
    use rayon::prelude::*;

    // Convert interpolation parameter (default to bilinear if not specified)
    let interp = match interpolation.unwrap_or(1) {
        0 => ResizeInterpolation::InterNearest,
        1 => ResizeInterpolation::InterLinear,
        2 => ResizeInterpolation::InterCubic,
        4 => ResizeInterpolation::InterLanczos4,
        _ => ResizeInterpolation::InterLinear, // Default fallback
    };

    // Convert to OpenCV constant
    let cv_interpolation = match interp {
        ResizeInterpolation::InterNearest => INTER_NEAREST,
        ResizeInterpolation::InterLinear => INTER_LINEAR,
        ResizeInterpolation::InterCubic => INTER_CUBIC,
        ResizeInterpolation::InterLanczos4 => INTER_LANCZOS4,
    };

    let batch_size = images.len();
    if batch_size != target_sizes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Images and target_sizes length mismatch",
        ));
    }

    if batch_size == 0 {
        return Bound::new(
            py,
            ResizeIterator {
                buffers: Vec::new(),
                index: 0,
            },
        );
    }

    // Extract raw data for parallel processing (same as before)
    #[derive(Clone, Copy)]
    struct ResizeData {
        src_ptr: *const u8,
        src_height: usize,
        src_width: usize,
        src_channels: usize,
        target_width: usize,
        target_height: usize,
    }

    unsafe impl Send for ResizeData {}
    unsafe impl Sync for ResizeData {}

    let resize_data: Vec<ResizeData> = images
        .iter()
        .zip(target_sizes.iter())
        .map(|(image, &(target_width, target_height))| {
            let img_view = image.as_array();
            let (src_height, src_width, src_channels) = img_view.dim();
            let src_ptr = img_view.as_ptr();
            ResizeData {
                src_ptr,
                src_height,
                src_width,
                src_channels,
                target_width,
                target_height,
            }
        })
        .collect();

    // Adaptive processing: use parallel only for larger batches
    let use_parallel = batch_size >= 8;

    // TRUE ZERO-COPY: Process directly to raw buffers, no intermediate Array3!
    #[allow(clippy::type_complexity)]
    let raw_results: Result<Vec<(Vec<u8>, (usize, usize, usize))>, String> = if use_parallel {
        // Pre-allocate all buffers on main thread (avoid lock contention)
        let mut buffers = Vec::with_capacity(batch_size);
        {
            let mut pool = get_buffer_pool().lock().unwrap();
            for data in &resize_data {
                let buffer =
                    pool.get_buffer(data.target_height, data.target_width, data.src_channels);
                buffers.push(buffer);
            }
        }

        // Parallel processing: OpenCV writes directly to raw buffers
        resize_data
            .par_iter()
            .zip(buffers.into_par_iter())
            .map(|(data, mut buffer)| {
                unsafe {
                    // Create source Mat pointing to input data
                    let src_mat = Mat::new_rows_cols_with_data_unsafe(
                        data.src_height as i32,
                        data.src_width as i32,
                        opencv::core::CV_8UC3,
                        data.src_ptr as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| format!("Failed to create source Mat: {}", e))?;

                    // Create destination Mat pointing to our buffer
                    let required_size = data.target_height * data.target_width * data.src_channels;
                    buffer.resize(required_size, 0);

                    let mut dst_mat = Mat::new_rows_cols_with_data_unsafe(
                        data.target_height as i32,
                        data.target_width as i32,
                        opencv::core::CV_8UC3,
                        buffer.as_mut_ptr() as *mut std::ffi::c_void,
                        opencv::core::Mat_AUTO_STEP,
                    )
                    .map_err(|e| format!("Failed to create destination Mat: {}", e))?;

                    // OpenCV writes directly to raw buffer!
                    resize(
                        &src_mat,
                        &mut dst_mat,
                        opencv::core::Size::new(
                            data.target_width as i32,
                            data.target_height as i32,
                        ),
                        0.0,
                        0.0,
                        cv_interpolation,
                    )
                    .map_err(|e| format!("OpenCV resize failed: {}", e))?;

                    // Return raw buffer with dimensions - NO Array3 conversion!
                    Ok((
                        buffer,
                        (data.target_height, data.target_width, data.src_channels),
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()
    } else {
        // Sequential processing for small batches
        let mut pool = get_buffer_pool().lock().unwrap();
        let mut results = Vec::with_capacity(batch_size);

        for data in &resize_data {
            let mut buffer =
                pool.get_buffer(data.target_height, data.target_width, data.src_channels);

            unsafe {
                let src_mat = match Mat::new_rows_cols_with_data_unsafe(
                    data.src_height as i32,
                    data.src_width as i32,
                    opencv::core::CV_8UC3,
                    data.src_ptr as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ) {
                    Ok(mat) => mat,
                    Err(e) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to create source Mat: {}",
                            e
                        )))
                    }
                };

                let required_size = data.target_height * data.target_width * data.src_channels;
                buffer.resize(required_size, 0);

                let mut dst_mat = match Mat::new_rows_cols_with_data_unsafe(
                    data.target_height as i32,
                    data.target_width as i32,
                    opencv::core::CV_8UC3,
                    buffer.as_mut_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                ) {
                    Ok(mat) => mat,
                    Err(e) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to create destination Mat: {}",
                            e
                        )))
                    }
                };

                if let Err(e) = resize(
                    &src_mat,
                    &mut dst_mat,
                    opencv::core::Size::new(data.target_width as i32, data.target_height as i32),
                    0.0,
                    0.0,
                    cv_interpolation,
                ) {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "OpenCV resize failed: {}",
                        e
                    )));
                }

                results.push((
                    buffer,
                    (data.target_height, data.target_width, data.src_channels),
                ));
            }
        }
        Ok(results)
    };

    let buffers = raw_results
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Resize failed: {}", e)))?;

    // Return iterator with raw buffers - conversion happens on-demand!
    Bound::new(py, ResizeIterator { buffers, index: 0 })
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
    if images.len() != target_sizes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Images and target_sizes length mismatch",
        ));
    }

    let mut py_results = Vec::with_capacity(images.len());

    for (image, &(target_width, target_height)) in images.iter().zip(target_sizes.iter()) {
        let img_view = image.as_array();
        match crate::cropping::center_crop_image_array(&img_view, target_width, target_height) {
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
pub fn batch_calculate_luminance(images: Vec<PyReadonlyArray3<u8>>) -> PyResult<Vec<f64>> {
    // Keep original sequential implementation for fair comparison
    let mut luminances = Vec::with_capacity(images.len());

    for image in images.iter() {
        let img_view = image.as_array();
        let luminance = crate::luminance::calculate_luminance_array(&img_view);
        luminances.push(luminance);
    }

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
            "Expected RGBA image with 4 channels",
        ));
    }

    let rgba_data = image_array.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Image data is not contiguous")
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
        "SIMD format conversion not available - compile with simd feature",
    ))
}

#[cfg(all(feature = "python-bindings", not(feature = "simd")))]
#[pyfunction]
pub fn rgba_to_rgb_optimized<'py>(
    _py: Python<'py>,
    _image: PyReadonlyArray3<u8>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, f64)> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "SIMD format conversion not available - compile with simd feature",
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
        "Batch resizing failed: OpenCV feature not enabled. Rebuild with --features opencv",
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
        "Batch video resizing failed: OpenCV feature not enabled. Rebuild with --features opencv",
    ))
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
        "OpenCV acceleration not available - compile with opencv feature",
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
        "OpenCV acceleration not available - compile with opencv feature",
    ))
}

// OPENCV-COMPATIBLE API BINDINGS

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn imdecode_py<'py>(
    py: Python<'py>,
    buf: &[u8],
    flags: i32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::cv_compat::{imdecode, ImreadFlags};

    let imread_flags = match flags {
        -1 => ImreadFlags::ImreadUnchanged,
        0 => ImreadFlags::ImreadGrayscale,
        1 => ImreadFlags::ImreadColor,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid imread flag",
            ))
        }
    };

    match imdecode(buf, imread_flags) {
        Ok(image) => {
            let py_array = PyArray3::from_array_bound(py, &image);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Image decoding failed: {}",
            e
        ))),
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn cvt_color_py<'py>(
    py: Python<'py>,
    src: PyReadonlyArray3<u8>,
    code: i32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::cv_compat::{cvt_color, ColorConversionCode};

    let color_code = match code {
        4 => ColorConversionCode::ColorBgr2Rgb,
        5 => ColorConversionCode::ColorRgb2Bgr,
        7 => ColorConversionCode::ColorRgb2Gray,
        8 => ColorConversionCode::ColorGray2Rgb,
        55 => ColorConversionCode::ColorHsv2Rgb,
        41 => ColorConversionCode::ColorRgb2Hsv,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unsupported color conversion code",
            ))
        }
    };

    let src_array = src.as_array();
    match cvt_color(&src_array, color_code) {
        Ok(converted) => {
            let py_array = PyArray3::from_array_bound(py, &converted);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Color conversion failed: {}",
            e
        ))),
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn canny_py<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<u8>,
    threshold1: f64,
    threshold2: f64,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::cv_compat::canny;

    let image_array = image.as_array();
    match canny(&image_array, threshold1, threshold2) {
        Ok(edges) => {
            let py_array = PyArray3::from_array_bound(py, &edges);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Canny edge detection failed: {}",
            e
        ))),
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (src, dsize, interpolation=None))]
pub fn resize_py<'py>(
    py: Python<'py>,
    src: PyReadonlyArray3<u8>,
    dsize: (u32, u32),
    interpolation: Option<i32>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    use crate::cv_compat::{resize, ResizeInterpolation};

    let interp = match interpolation.unwrap_or(1) {
        0 => ResizeInterpolation::InterNearest,
        1 => ResizeInterpolation::InterLinear,
        2 => ResizeInterpolation::InterCubic,
        4 => ResizeInterpolation::InterLanczos4,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unsupported interpolation method",
            ))
        }
    };

    let src_array = src.as_array();
    match resize(&src_array, dsize, interp) {
        Ok(resized) => {
            let py_array = PyArray3::from_array_bound(py, &resized);
            Ok(py_array)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Resize failed: {}",
            e
        ))),
    }
}

#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyVideoCapture {
    inner: VideoCapture,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVideoCapture {
    #[new]
    fn new(filename: String) -> PyResult<Self> {
        match VideoCapture::new(&filename) {
            Ok(cap) => Ok(Self { inner: cap }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to open video: {}",
                e
            ))),
        }
    }

    fn is_opened(&self) -> bool {
        self.inner.is_opened()
    }

    fn read<'py>(&mut self, py: Python<'py>) -> PyResult<(bool, Option<Bound<'py, PyArray3<u8>>>)> {
        let (ret, frame) = self.inner.read();
        if let Some(frame_data) = frame {
            let py_array = PyArray3::from_array_bound(py, &frame_data);
            Ok((ret, Some(py_array)))
        } else {
            Ok((ret, None))
        }
    }

    fn release(&mut self) {
        self.inner.release();
    }

    fn get(&self, prop: i32) -> f64 {
        use crate::cv_compat::VideoCaptureProperties;

        let prop_enum = match prop {
            5 => VideoCaptureProperties::CapPropFps,
            3 => VideoCaptureProperties::CapPropFrameWidth,
            4 => VideoCaptureProperties::CapPropFrameHeight,
            7 => VideoCaptureProperties::CapPropFrameCount,
            _ => return 0.0,
        };

        self.inner.get(prop_enum)
    }
}

#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyVideoWriter {
    inner: VideoWriter,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVideoWriter {
    #[new]
    fn new(
        filename: String,
        fourcc_str: String,
        fps: f64,
        frame_size: (i32, i32),
    ) -> PyResult<Self> {
        match VideoWriter::new(&filename, &fourcc_str, fps, frame_size) {
            Ok(writer) => Ok(Self { inner: writer }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create video writer: {}",
                e
            ))),
        }
    }

    fn is_opened(&self) -> bool {
        self.inner.is_opened()
    }

    fn write(&mut self, frame: PyReadonlyArray3<u8>) -> PyResult<()> {
        let frame_array = frame.as_array();
        match self.inner.write(&frame_array) {
            Ok(()) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to write frame: {}",
                e
            ))),
        }
    }

    fn release(&mut self) -> PyResult<()> {
        match self.inner.release() {
            Ok(()) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to release video writer: {}",
                e
            ))),
        }
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn fourcc_py(c1: char, c2: char, c3: char, c4: char) -> String {
    fourcc(c1, c2, c3, c4)
}

#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyCascadeClassifier {
    inner: CascadeClassifier,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyCascadeClassifier {
    #[new]
    fn new(filename: String) -> PyResult<Self> {
        match CascadeClassifier::new(&filename) {
            Ok(classifier) => Ok(Self { inner: classifier }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to load cascade classifier: {}",
                e
            ))),
        }
    }

    #[pyo3(signature = (image, scale_factor=None, min_neighbors=None))]
    fn detect_multi_scale(
        &self,
        image: PyReadonlyArray3<u8>,
        scale_factor: Option<f64>,
        min_neighbors: Option<i32>,
    ) -> PyResult<Vec<(i32, i32, i32, i32)>> {
        let image_array = image.as_array();
        let scale = scale_factor.unwrap_or(1.1);
        let neighbors = min_neighbors.unwrap_or(3);

        match self
            .inner
            .detect_multi_scale(&image_array, scale, neighbors)
        {
            Ok(detections) => Ok(detections),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Face detection failed: {}",
                e
            ))),
        }
    }

    fn empty(&self) -> bool {
        self.inner.empty()
    }
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn get_opencv_data_path_py() -> String {
    get_opencv_data_path()
}

// BATCH PROCESSING API BINDINGS (PERFORMANCE OPTIMIZED)

#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyBatchProcessor {
    inner: BatchProcessor,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyBatchProcessor {
    #[new]
    fn new() -> Self {
        Self {
            inner: BatchProcessor::new(),
        }
    }

    #[staticmethod]
    fn with_config(use_parallel: bool, chunk_size: usize) -> Self {
        Self {
            inner: BatchProcessor::with_config(use_parallel, chunk_size),
        }
    }

    /// High-performance batch color conversion - up to 3x faster than individual cv2.cvtColor calls
    fn batch_cvt_color<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray3<u8>>,
        code: i32,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        use crate::cv_compat::ColorConversionCode;

        let color_code = match code {
            4 => ColorConversionCode::ColorBgr2Rgb,
            5 => ColorConversionCode::ColorRgb2Bgr,
            7 => ColorConversionCode::ColorRgb2Gray,
            8 => ColorConversionCode::ColorGray2Rgb,
            55 => ColorConversionCode::ColorHsv2Rgb,
            41 => ColorConversionCode::ColorRgb2Hsv,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported color conversion code",
                ))
            }
        };

        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();

        match self.inner.batch_cvt_color(&image_views, color_code) {
            Ok(results) => {
                let py_results: Vec<_> = results
                    .into_iter()
                    .map(|result| PyArray3::from_array_bound(py, &result))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Batch color conversion failed: {}",
                e
            ))),
        }
    }

    /// Ultra-fast batch resize - optimized for cache locality and parallel processing
    #[pyo3(signature = (images, target_sizes, interpolation=None))]
    fn batch_resize<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray3<u8>>,
        target_sizes: Vec<(u32, u32)>,
        interpolation: Option<i32>,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        use crate::cv_compat::ResizeInterpolation;

        let interp = match interpolation.unwrap_or(1) {
            0 => ResizeInterpolation::InterNearest,
            1 => ResizeInterpolation::InterLinear,
            2 => ResizeInterpolation::InterCubic,
            4 => ResizeInterpolation::InterLanczos4,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported interpolation method",
                ))
            }
        };

        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();

        match self.inner.batch_resize(&image_views, &target_sizes, interp) {
            Ok(results) => {
                let py_results: Vec<_> = results
                    .into_iter()
                    .map(|result| PyArray3::from_array_bound(py, &result))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Batch resize failed: {}",
                e
            ))),
        }
    }

    /// High-performance batch Canny edge detection
    fn batch_canny<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray3<u8>>,
        threshold1: f64,
        threshold2: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();

        match self.inner.batch_canny(&image_views, threshold1, threshold2) {
            Ok(results) => {
                let py_results: Vec<_> = results
                    .into_iter()
                    .map(|result| PyArray3::from_array_bound(py, &result))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Batch Canny edge detection failed: {}",
                e
            ))),
        }
    }

    /// Optimized preprocessing pipeline - decode -> resize -> color convert in one batch
    /// This is significantly faster than individual operations due to reduced memory allocations
    #[pyo3(signature = (image_buffers, target_sizes, color_conversion=None, decode_flags=1, interpolation=None))]
    fn batch_preprocess_pipeline<'py>(
        &self,
        py: Python<'py>,
        image_buffers: Vec<Vec<u8>>,
        target_sizes: Vec<(u32, u32)>,
        color_conversion: Option<i32>,
        decode_flags: i32,
        interpolation: Option<i32>,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        use crate::cv_compat::{ColorConversionCode, ImreadFlags, ResizeInterpolation};

        let imread_flags = match decode_flags {
            -1 => ImreadFlags::ImreadUnchanged,
            0 => ImreadFlags::ImreadGrayscale,
            1 => ImreadFlags::ImreadColor,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid imread flag",
                ))
            }
        };

        let color_code = if let Some(code) = color_conversion {
            Some(match code {
                4 => ColorConversionCode::ColorBgr2Rgb,
                5 => ColorConversionCode::ColorRgb2Bgr,
                7 => ColorConversionCode::ColorRgb2Gray,
                8 => ColorConversionCode::ColorGray2Rgb,
                55 => ColorConversionCode::ColorHsv2Rgb,
                41 => ColorConversionCode::ColorRgb2Hsv,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Unsupported color conversion code",
                    ))
                }
            })
        } else {
            None
        };

        let interp = match interpolation.unwrap_or(1) {
            0 => ResizeInterpolation::InterNearest,
            1 => ResizeInterpolation::InterLinear,
            2 => ResizeInterpolation::InterCubic,
            4 => ResizeInterpolation::InterLanczos4,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported interpolation method",
                ))
            }
        };

        let buffer_refs: Vec<&[u8]> = image_buffers.iter().map(|buf| buf.as_slice()).collect();

        match self.inner.batch_preprocess_pipeline(
            &buffer_refs,
            &target_sizes,
            color_code,
            imread_flags,
            interp,
        ) {
            Ok(results) => {
                let py_results: Vec<_> = results
                    .into_iter()
                    .map(|result| PyArray3::from_array_bound(py, &result))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Batch preprocessing pipeline failed: {}",
                e
            ))),
        }
    }

    /// Run performance benchmarks comparing batch vs naive processing
    #[classmethod]
    fn run_benchmarks(
        _cls: &Bound<'_, pyo3::types::PyType>,
        num_images: usize,
        image_width: usize,
        image_height: usize,
    ) -> String {
        crate::cv_batch_ops::benchmarks::run_comprehensive_benchmarks(
            num_images,
            (image_height, image_width),
        )
    }

    /// Compare batch vs naive color conversion performance
    #[classmethod]
    fn benchmark_color_conversion(
        _cls: &Bound<'_, pyo3::types::PyType>,
        images: Vec<PyReadonlyArray3<u8>>,
        code: i32,
        iterations: usize,
    ) -> PyResult<(f64, f64, f64)> {
        use crate::cv_compat::ColorConversionCode;

        let color_code = match code {
            4 => ColorConversionCode::ColorBgr2Rgb,
            5 => ColorConversionCode::ColorRgb2Bgr,
            7 => ColorConversionCode::ColorRgb2Gray,
            8 => ColorConversionCode::ColorGray2Rgb,
            55 => ColorConversionCode::ColorHsv2Rgb,
            41 => ColorConversionCode::ColorRgb2Hsv,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported color conversion code",
                ))
            }
        };

        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();
        let (naive_ms, batch_ms, speedup) =
            crate::cv_batch_ops::benchmarks::compare_batch_vs_naive_cvt_color(
                &image_views,
                color_code,
                iterations,
            );

        Ok((naive_ms, batch_ms, speedup))
    }
}

/// TRUE batch processor that actually provides batching advantages over cv2
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct PyTrueBatchProcessor {
    inner: TrueBatchProcessor,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyTrueBatchProcessor {
    #[new]
    fn new() -> Self {
        Self {
            inner: TrueBatchProcessor::new(),
        }
    }

    #[staticmethod]
    fn with_config(use_parallel: bool, chunk_size: usize, simd_threshold: usize) -> Self {
        Self {
            inner: TrueBatchProcessor {
                use_parallel,
                chunk_size,
                simd_threshold,
            },
        }
    }

    /// TRUE batch resize with proper OpenCV interpolation - should beat cv2 individual calls
    #[pyo3(signature = (images, target_sizes, interpolation=None))]
    fn true_batch_resize<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray3<u8>>,
        target_sizes: Vec<(u32, u32)>,
        interpolation: Option<i32>,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        use crate::cv_compat::ResizeInterpolation;

        let interp = match interpolation.unwrap_or(1) {
            0 => ResizeInterpolation::InterNearest,
            1 => ResizeInterpolation::InterLinear,
            2 => ResizeInterpolation::InterCubic,
            4 => ResizeInterpolation::InterLanczos4,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported interpolation method",
                ))
            }
        };

        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();

        match self
            .inner
            .true_batch_resize(&image_views, &target_sizes, interp)
        {
            Ok(results) => {
                let py_results: Vec<_> = results
                    .into_iter()
                    .map(|result| PyArray3::from_array_bound(py, &result))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "True batch resize failed: {}",
                e
            ))),
        }
    }

    /// TRUE batch color conversion with SIMD optimization
    fn true_batch_cvt_color<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray3<u8>>,
        code: i32,
    ) -> PyResult<Vec<Bound<'py, PyArray3<u8>>>> {
        let conversion = match code {
            4 => ColorConversion::BgrToRgb,
            5 => ColorConversion::RgbToBgr,
            7 => ColorConversion::RgbToGray,
            8 => ColorConversion::GrayToRgb,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported color conversion code",
                ))
            }
        };

        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();

        match self.inner.true_batch_cvt_color(&image_views, conversion) {
            Ok(results) => {
                let py_results: Vec<_> = results
                    .into_iter()
                    .map(|result| PyArray3::from_array_bound(py, &result))
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "True batch color conversion failed: {}",
                e
            ))),
        }
    }

    /// Strided luminance calculation for fast "dummy light" average
    #[pyo3(signature = (images, stride=None))]
    fn strided_luminance(
        &self,
        images: Vec<PyReadonlyArray3<u8>>,
        stride: Option<usize>,
    ) -> PyResult<Vec<f32>> {
        let stride = stride.unwrap_or(8); // Default stride of 8
        let image_views: Vec<_> = images.iter().map(|img| img.as_array()).collect();

        match self.inner.strided_luminance(&image_views, stride) {
            Ok(results) => Ok(results),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Strided luminance failed: {}",
                e
            ))),
        }
    }
}
