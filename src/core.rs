// Re-export all core functionality
pub use crate::cropping::*;
pub use crate::loading::*;
pub use crate::luminance::*;

// Re-export SIMD optimizations when available (benchmark winners only)
#[cfg(feature = "simd")]
pub use crate::luminance_simd::*;

#[cfg(feature = "simd")]
pub use crate::format_conversion_simd::*;

// OpenCV integration for performance parity
pub use crate::opencv_ops::*;

// Core batch processing functions for native Rust usage
use anyhow::Result;
use ndarray::Array3;
use rayon::prelude::*;
use std::path::Path;

pub fn batch_load_images<P: AsRef<Path> + Send + Sync>(image_paths: &[P]) -> Vec<Result<Vec<u8>>> {
    image_paths
        .par_iter()
        .map(|path| load_image_from_path(path.as_ref().to_str().unwrap()))
        .collect()
}

pub fn batch_crop_image_arrays(
    images: &[Array3<u8>],
    crop_boxes: &[(usize, usize, usize, usize)], // (x, y, width, height)
) -> Vec<Result<Array3<u8>>> {
    images
        .par_iter()
        .zip(crop_boxes.par_iter())
        .map(|(img, &(x, y, width, height))| crop_image_array(&img.view(), x, y, width, height))
        .collect()
}

pub fn batch_center_crop_image_arrays(
    images: &[Array3<u8>],
    target_sizes: &[(usize, usize)], // (width, height)
) -> Vec<Result<Array3<u8>>> {
    images
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| center_crop_image_array(&img.view(), width, height))
        .collect()
}

// Resize wrapper functions for backwards compatibility with tests
#[cfg(feature = "opencv")]
pub fn resize_image_array(
    image: &ndarray::ArrayView3<u8>,
    target_width: usize,
    target_height: usize,
) -> Result<Array3<u8>> {
    use crate::opencv_ops::resize_bilinear_opencv;
    resize_bilinear_opencv(image, target_width as u32, target_height as u32)
}

#[cfg(not(feature = "opencv"))]
pub fn resize_image_array(
    _image: &ndarray::ArrayView3<u8>,
    _target_width: usize,
    _target_height: usize,
) -> Result<Array3<u8>> {
    anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
}

#[cfg(feature = "opencv")]
pub fn resize_video_array(
    video: &ndarray::ArrayView4<u8>,
    target_width: usize,
    target_height: usize,
) -> Result<ndarray::Array4<u8>> {
    use crate::opencv_ops::OpenCVBatchProcessor;
    let processor = OpenCVBatchProcessor::new();
    let target_sizes = vec![(target_width as u32, target_height as u32)];
    let results = processor.batch_resize_videos(&[*video], &target_sizes)?;
    Ok(results.into_iter().next().unwrap())
}

#[cfg(not(feature = "opencv"))]
pub fn resize_video_array(
    _video: &ndarray::ArrayView4<u8>,
    _target_width: usize,
    _target_height: usize,
) -> Result<ndarray::Array4<u8>> {
    anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
}

#[cfg(feature = "opencv")]
pub fn batch_resize_image_arrays(
    images: &[Array3<u8>],
    target_sizes: &[(u32, u32)], // (width, height)
) -> Vec<Result<Array3<u8>>> {
    use crate::opencv_ops::OpenCVBatchProcessor;
    let processor = OpenCVBatchProcessor::new();
    let image_views: Vec<_> = images.iter().map(|img| img.view()).collect();
    let opencv_sizes: Vec<_> = target_sizes.iter().map(|&(w, h)| (w, h)).collect();

    match processor.batch_resize_images(&image_views, &opencv_sizes) {
        Ok(results) => results.into_iter().map(Ok).collect(),
        Err(e) => {
            let error_msg = e.to_string();
            (0..images.len())
                .map(|_| Err(anyhow::anyhow!("{}", error_msg)))
                .collect()
        }
    }
}

#[cfg(not(feature = "opencv"))]
pub fn batch_resize_image_arrays(
    images: &[Array3<u8>],
    _target_sizes: &[(u32, u32)],
) -> Vec<Result<Array3<u8>>> {
    (0..images.len())
        .map(|_| {
            Err(anyhow::anyhow!(
                "OpenCV feature not enabled. Rebuild with --features opencv"
            ))
        })
        .collect()
}

#[cfg(feature = "opencv")]
pub fn batch_resize_video_arrays(
    videos: &[ndarray::Array4<u8>],
    target_sizes: &[(u32, u32)], // (width, height)
) -> Vec<Result<ndarray::Array4<u8>>> {
    use crate::opencv_ops::OpenCVBatchProcessor;
    let processor = OpenCVBatchProcessor::new();
    let video_views: Vec<_> = videos.iter().map(|vid| vid.view()).collect();
    let opencv_sizes: Vec<_> = target_sizes.iter().map(|&(w, h)| (w, h)).collect();

    match processor.batch_resize_videos(&video_views, &opencv_sizes) {
        Ok(results) => results.into_iter().map(Ok).collect(),
        Err(e) => {
            let error_msg = e.to_string();
            (0..videos.len())
                .map(|_| Err(anyhow::anyhow!("{}", error_msg)))
                .collect()
        }
    }
}

#[cfg(not(feature = "opencv"))]
pub fn batch_resize_video_arrays(
    videos: &[ndarray::Array4<u8>],
    _target_sizes: &[(u32, u32)],
) -> Vec<Result<ndarray::Array4<u8>>> {
    (0..videos.len())
        .map(|_| {
            Err(anyhow::anyhow!(
                "OpenCV feature not enabled. Rebuild with --features opencv"
            ))
        })
        .collect()
}

pub fn batch_random_crop_image_arrays(
    images: &[Array3<u8>],
    target_sizes: &[(usize, usize)], // (width, height)
) -> Vec<Result<Array3<u8>>> {
    images
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| random_crop_image_array(&img.view(), width, height))
        .collect()
}

// Note: resize operations now handled by OpenCV in opencv_ops.rs for optimal performance

pub fn batch_calculate_luminance_arrays(images: &[Array3<u8>]) -> Vec<f64> {
    use crate::luminance::calculate_luminance_array_sequential;

    // Use parallel batch processing with sequential individual processing
    // to avoid nested parallelism that causes performance degradation
    images
        .par_iter()
        .map(|img| calculate_luminance_array_sequential(&img.view()))
        .collect()
}

// Note: resize operations (images and videos) now handled by OpenCV in opencv_ops.rs for optimal performance
