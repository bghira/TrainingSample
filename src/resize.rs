use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array3, Array4, ArrayView3, ArrayView4, Axis};
use rayon::prelude::*;

pub fn resize_image_array(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    let (height, width, channels) = image.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB images (3 channels) are supported"
        ));
    }

    if target_width == 0 || target_height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be greater than zero"
        ));
    }

    // Use pre-allocated buffer to avoid repeated allocations
    let mut img_data = Vec::with_capacity(width * height * 3);

    // Fill buffer in row-major order for better cache locality
    for y in 0..height {
        for x in 0..width {
            img_data.push(image[[y, x, 0]]);
            img_data.push(image[[y, x, 1]]);
            img_data.push(image[[y, x, 2]]);
        }
    }

    // Create ImageBuffer directly from Vec to avoid double allocation
    let img_buffer =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width as u32, height as u32, img_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    // Resize using image crate
    let resized = DynamicImage::ImageRgb8(img_buffer)
        .resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();

    // Convert back to ndarray efficiently using from_shape_vec
    let target_h = target_height as usize;
    let target_w = target_width as usize;
    let resized_data = resized.into_raw();

    let result = Array3::from_shape_vec((target_h, target_w, 3), resized_data)
        .map_err(|e| anyhow::anyhow!("Failed to reshape result array: {}", e))?;

    Ok(result)
}

pub fn resize_video_array(
    video: &ArrayView4<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array4<u8>> {
    let (num_frames, _height, _width, channels) = video.dim();

    if channels != 3 {
        return Err(anyhow::anyhow!(
            "Only RGB videos (3 channels) are supported"
        ));
    }

    if target_width == 0 || target_height == 0 {
        return Err(anyhow::anyhow!(
            "Target dimensions must be greater than zero"
        ));
    }

    // Process frames in parallel
    let resized_frames: Result<Vec<_>> = (0..num_frames)
        .into_par_iter()
        .map(|frame_idx| {
            let frame = video.index_axis(Axis(0), frame_idx);
            resize_image_array(&frame, target_width, target_height)
        })
        .collect();

    let resized_frames = resized_frames?;

    // Stack frames back into 4D array
    let frame_shape = resized_frames[0].dim();
    let mut result = Array4::<u8>::zeros((num_frames, frame_shape.0, frame_shape.1, frame_shape.2));

    for (idx, frame) in resized_frames.into_iter().enumerate() {
        result.index_axis_mut(Axis(0), idx).assign(&frame);
    }

    Ok(result)
}
