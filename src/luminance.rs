use ndarray::ArrayView3;

pub fn calculate_luminance_array(image: &ArrayView3<u8>) -> f64 {
    let (height, width, channels) = image.dim();

    if channels < 3 {
        // Grayscale or single channel - just average the values
        let sum: u64 = image.iter().map(|&x| x as u64).sum();
        return sum as f64 / (height * width * channels) as f64;
    }

    let mut total_luminance = 0.0;
    let pixel_count = height * width;

    for h in 0..height {
        for w in 0..width {
            let r = image[[h, w, 0]] as f64;
            let g = image[[h, w, 1]] as f64;
            let b = image[[h, w, 2]] as f64;

            // Standard RGB to luminance conversion
            let luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            total_luminance += luminance;
        }
    }

    total_luminance / pixel_count as f64
}
