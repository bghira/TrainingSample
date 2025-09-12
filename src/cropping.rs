use anyhow::Result;
use ndarray::{s, Array3, ArrayView3};

pub fn crop_image_array(
    image: &ArrayView3<u8>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<Array3<u8>> {
    let (img_height, img_width, _channels) = image.dim();

    if width == 0 || height == 0 {
        return Err(anyhow::anyhow!("Crop dimensions must be greater than zero"));
    }

    if x + width > img_width || y + height > img_height {
        return Err(anyhow::anyhow!("Crop bounds exceed image dimensions"));
    }

    // Use assign to avoid allocation where possible, but still return owned array
    let slice = image.slice(s![y..y + height, x..x + width, ..]);
    // Only allocate when we need to return an owned array
    Ok(slice.to_owned())
}

pub fn center_crop_image_array(
    image: &ArrayView3<u8>,
    target_width: usize,
    target_height: usize,
) -> Result<Array3<u8>> {
    let (img_height, img_width, _) = image.dim();

    let x = if img_width > target_width {
        (img_width - target_width) / 2
    } else {
        0
    };

    let y = if img_height > target_height {
        (img_height - target_height) / 2
    } else {
        0
    };

    let actual_width = target_width.min(img_width);
    let actual_height = target_height.min(img_height);

    crop_image_array(image, x, y, actual_width, actual_height)
}

pub fn random_crop_image_array(
    image: &ArrayView3<u8>,
    target_width: usize,
    target_height: usize,
) -> Result<Array3<u8>> {
    let (img_height, img_width, _) = image.dim();

    let max_x = img_width.saturating_sub(target_width);

    let max_y = img_height.saturating_sub(target_height);

    let x = if max_x > 0 {
        fastrand::usize(0..=max_x)
    } else {
        0
    };

    let y = if max_y > 0 {
        fastrand::usize(0..=max_y)
    } else {
        0
    };

    let actual_width = target_width.min(img_width);
    let actual_height = target_height.min(img_height);

    crop_image_array(image, x, y, actual_width, actual_height)
}

/// Ultra-fast batch center cropping optimized for minimal overhead
pub fn batch_center_crop_arrays(
    images: &[ArrayView3<u8>],
    target_sizes: &[(usize, usize)], // (width, height)
) -> Result<Vec<Array3<u8>>> {
    if images.len() != target_sizes.len() {
        return Err(anyhow::anyhow!(
            "Number of images and target sizes must match"
        ));
    }

    let mut results = Vec::with_capacity(images.len());

    // Process each image with minimal overhead
    for (image, &(target_width, target_height)) in images.iter().zip(target_sizes.iter()) {
        let (img_height, img_width, _) = image.dim();

        // Calculate crop coordinates (same logic as center_crop_image_array)
        let x = if img_width > target_width {
            (img_width - target_width) / 2
        } else {
            0
        };

        let y = if img_height > target_height {
            (img_height - target_height) / 2
        } else {
            0
        };

        let actual_width = target_width.min(img_width);
        let actual_height = target_height.min(img_height);

        // Validate bounds
        if x + actual_width > img_width || y + actual_height > img_height {
            return Err(anyhow::anyhow!("Crop bounds exceed image dimensions"));
        }

        // Direct slice and copy - optimized for speed
        let cropped = image
            .slice(s![y..y + actual_height, x..x + actual_width, ..])
            .to_owned();
        results.push(cropped);
    }

    Ok(results)
}

/// Ultra-fast zero-copy cropping using raw pointers (bypasses PyO3 overhead)
///
/// # Safety
/// - `src_ptr` must be valid for reads of at least `src_shape.0 * src_shape.1 * src_shape.2` bytes
/// - `dst_ptr` must be valid for writes of at least `crop_coords.2 * crop_coords.3 * src_shape.2` bytes
/// - Crop coordinates must be within bounds of the source image
/// - Pointers must not overlap in a way that violates memory safety
pub unsafe fn crop_raw_buffer(
    src_ptr: *const u8,
    src_shape: (usize, usize, usize), // (height, width, channels)
    dst_ptr: *mut u8,
    crop_coords: (usize, usize, usize, usize), // (start_y, start_x, height, width)
) {
    let (_src_height, src_width, channels) = src_shape;
    let (start_y, start_x, crop_height, crop_width) = crop_coords;

    // Calculate strides
    let src_row_stride = src_width * channels;
    let dst_row_stride = crop_width * channels;
    let pixel_size = channels;

    // Copy row by row for cache efficiency
    for y in 0..crop_height {
        let src_row_start = src_ptr.add((start_y + y) * src_row_stride + start_x * pixel_size);
        let dst_row_start = dst_ptr.add(y * dst_row_stride);

        // Copy entire row at once (faster than pixel-by-pixel)
        std::ptr::copy_nonoverlapping(src_row_start, dst_row_start, crop_width * pixel_size);
    }
}

/// SIMD-optimized batch cropping with raw memory access
///
/// # Safety
/// - All pointers in `src_ptrs` and `dst_ptrs` must be valid for their respective operations
/// - Source pointers must be valid for reads according to their corresponding shapes
/// - Destination pointers must be valid for writes according to crop dimensions
/// - All crop coordinates must be within bounds of their respective source images
/// - Arrays must have matching lengths: `src_ptrs.len() == src_shapes.len() == dst_ptrs.len() == crop_coords.len()`
pub unsafe fn batch_crop_raw_buffers(
    src_ptrs: &[*const u8],
    src_shapes: &[(usize, usize, usize)],
    dst_ptrs: &[*mut u8],
    crop_coords: &[(usize, usize, usize, usize)],
) {
    // Process each image with zero-copy semantics
    for (((src_ptr, src_shape), dst_ptr), coords) in src_ptrs
        .iter()
        .zip(src_shapes.iter())
        .zip(dst_ptrs.iter())
        .zip(crop_coords.iter())
    {
        crop_raw_buffer(*src_ptr, *src_shape, *dst_ptr, *coords);
    }
}
