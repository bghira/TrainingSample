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

    // Convert ndarray to image::RgbImage
    let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
        let r = image[[y as usize, x as usize, 0]];
        let g = image[[y as usize, x as usize, 1]];
        let b = image[[y as usize, x as usize, 2]];
        *pixel = Rgb([r, g, b]);
    }

    // Resize using image crate with exact dimensions (force resize, no aspect ratio preservation)
    let resized = DynamicImage::ImageRgb8(img_buffer)
        .resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();

    // Convert back to ndarray
    let (w, h) = resized.dimensions();
    let mut result = Array3::<u8>::zeros((h as usize, w as usize, 3));

    for (x, y, pixel) in resized.enumerate_pixels() {
        result[[y as usize, x as usize, 0]] = pixel[0];
        result[[y as usize, x as usize, 1]] = pixel[1];
        result[[y as usize, x as usize, 2]] = pixel[2];
    }

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
