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

    // Create direct slice without unnecessary allocations - slice is already contiguous
    let cropped = image.slice(s![y..y + height, x..x + width, ..]).to_owned();
    Ok(cropped)
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
