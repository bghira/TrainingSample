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
