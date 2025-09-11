// Re-export all core functionality
pub use crate::cropping::*;
pub use crate::loading::*;
pub use crate::luminance::*;
pub use crate::resize::*;

// Re-export SIMD optimizations when available
#[cfg(feature = "simd")]
pub use crate::resize_simd::*;

#[cfg(feature = "simd")]
pub use crate::resize_multicore::*;

#[cfg(feature = "metal")]
pub use crate::resize_metal::*;

#[cfg(feature = "simd")]
pub use crate::format_conversion_simd::*;

// Core batch processing functions for native Rust usage
use anyhow::Result;
use ndarray::{Array3, Array4};
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

pub fn batch_resize_image_arrays(
    images: &[Array3<u8>],
    target_sizes: &[(u32, u32)], // (width, height)
) -> Vec<Result<Array3<u8>>> {
    images
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| resize_image_array(&img.view(), width, height))
        .collect()
}

pub fn batch_calculate_luminance_arrays(images: &[Array3<u8>]) -> Vec<f64> {
    images
        .par_iter()
        .map(|img| calculate_luminance_array(&img.view()))
        .collect()
}

pub fn batch_resize_video_arrays(
    videos: &[Array4<u8>],
    target_sizes: &[(u32, u32)], // (width, height)
) -> Vec<Result<Array4<u8>>> {
    videos
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(video, &(width, height))| resize_video_array(&video.view(), width, height))
        .collect()
}

// SIMD-optimized batch processing functions
#[cfg(feature = "simd")]
pub fn batch_resize_image_arrays_simd(
    images: &[Array3<u8>],
    target_sizes: &[(u32, u32)], // (width, height)
    filter: FilterType,
) -> Vec<Result<(Array3<u8>, ResizeMetrics)>> {
    images
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(img, &(width, height))| resize_image_optimized(&img.view(), width, height, filter))
        .collect()
}

#[cfg(feature = "simd")]
pub fn batch_resize_video_arrays_simd(
    videos: &[Array4<u8>],
    target_sizes: &[(u32, u32)], // (width, height)
    filter: FilterType,
) -> Vec<Result<(Array4<u8>, Vec<ResizeMetrics>)>> {
    videos
        .par_iter()
        .zip(target_sizes.par_iter())
        .map(|(video, &(width, height))| {
            resize_video_optimized(&video.view(), width, height, filter)
        })
        .collect()
}
