use anyhow::Result;
use image::ImageReader;
use std::fs::File;
use std::io::BufReader;

pub fn load_image_from_path(path: &str) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut buf_reader = BufReader::new(file);
    let mut buffer = Vec::new();
    std::io::copy(&mut buf_reader, &mut buffer)?;
    Ok(buffer)
}

pub fn load_and_decode_image(path: &str) -> Result<image::DynamicImage> {
    let img = ImageReader::open(path)?.decode()?;
    Ok(img)
}
