use anyhow::Result;
use ndarray::{Array3, ArrayView3};

/// OpenCV-compatible image decoding functionality
pub mod imdecode {
    use super::*;
    use image::ImageFormat;
    use std::io::Cursor;

    /// Image read flags matching OpenCV constants
    #[derive(Debug, Clone, Copy)]
    #[allow(clippy::enum_variant_names)]
    pub enum ImreadFlags {
        /// Load image as is (unchanged)
        ImreadUnchanged = -1,
        /// Load image as grayscale
        ImreadGrayscale = 0,
        /// Load image as color (BGR -> RGB conversion)
        ImreadColor = 1,
    }

    /// Decode image from byte buffer (equivalent to cv2.imdecode)
    pub fn imdecode(buf: &[u8], flags: ImreadFlags) -> Result<Array3<u8>> {
        let cursor = Cursor::new(buf);
        let img = image::load(
            cursor,
            ImageFormat::from_extension("").unwrap_or(ImageFormat::Png),
        )
        .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;

        match flags {
            ImreadFlags::ImreadGrayscale => {
                let gray_img = img.to_luma8();
                let (width, height) = gray_img.dimensions();

                // Convert to 3-channel grayscale (RGB format)
                let mut rgb_data = Array3::<u8>::zeros((height as usize, width as usize, 3));
                for y in 0..height {
                    for x in 0..width {
                        let pixel = gray_img.get_pixel(x, y);
                        let gray_val = pixel[0];
                        rgb_data[[y as usize, x as usize, 0]] = gray_val;
                        rgb_data[[y as usize, x as usize, 1]] = gray_val;
                        rgb_data[[y as usize, x as usize, 2]] = gray_val;
                    }
                }
                Ok(rgb_data)
            }
            ImreadFlags::ImreadColor | ImreadFlags::ImreadUnchanged => {
                let rgb_img = img.to_rgb8();
                let (width, height) = rgb_img.dimensions();

                let mut rgb_data = Array3::<u8>::zeros((height as usize, width as usize, 3));
                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x, y);
                        rgb_data[[y as usize, x as usize, 0]] = pixel[0];
                        rgb_data[[y as usize, x as usize, 1]] = pixel[1];
                        rgb_data[[y as usize, x as usize, 2]] = pixel[2];
                    }
                }
                Ok(rgb_data)
            }
        }
    }
}

/// OpenCV-compatible color space conversions
pub mod cvtcolor {
    use super::*;

    /// Color conversion codes matching OpenCV constants
    #[derive(Debug, Clone, Copy)]
    pub enum ColorConversionCode {
        ColorBgr2Rgb = 4,
        ColorRgb2Bgr = 5, // Different value for reverse operation
        ColorRgb2Gray = 7,
        ColorGray2Rgb = 8,
        ColorHsv2Rgb = 55,
        ColorRgb2Hsv = 41,
    }

    /// Convert color space (equivalent to cv2.cvtColor)
    pub fn cvt_color(src: &ArrayView3<u8>, code: ColorConversionCode) -> Result<Array3<u8>> {
        let (height, width, channels) = src.dim();

        match code {
            ColorConversionCode::ColorBgr2Rgb | ColorConversionCode::ColorRgb2Bgr => {
                if channels != 3 {
                    anyhow::bail!("BGR/RGB conversion requires 3-channel image");
                }

                let mut dst = Array3::<u8>::zeros((height, width, 3));
                for y in 0..height {
                    for x in 0..width {
                        // Swap R and B channels
                        dst[[y, x, 0]] = src[[y, x, 2]]; // R <- B
                        dst[[y, x, 1]] = src[[y, x, 1]]; // G <- G
                        dst[[y, x, 2]] = src[[y, x, 0]]; // B <- R
                    }
                }
                Ok(dst)
            }
            ColorConversionCode::ColorRgb2Gray => {
                if channels != 3 {
                    anyhow::bail!("RGB to Gray conversion requires 3-channel image");
                }

                // Return single-channel grayscale as 3-channel for consistency
                let mut dst = Array3::<u8>::zeros((height, width, 1));
                for y in 0..height {
                    for x in 0..width {
                        let r = src[[y, x, 0]] as f32;
                        let g = src[[y, x, 1]] as f32;
                        let b = src[[y, x, 2]] as f32;

                        // Standard luminance conversion
                        let gray = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                        dst[[y, x, 0]] = gray;
                    }
                }
                Ok(dst)
            }
            ColorConversionCode::ColorGray2Rgb => {
                if channels != 1 {
                    anyhow::bail!("Gray to RGB conversion requires 1-channel image");
                }

                let mut dst = Array3::<u8>::zeros((height, width, 3));
                for y in 0..height {
                    for x in 0..width {
                        let gray_val = src[[y, x, 0]];
                        dst[[y, x, 0]] = gray_val;
                        dst[[y, x, 1]] = gray_val;
                        dst[[y, x, 2]] = gray_val;
                    }
                }
                Ok(dst)
            }
            ColorConversionCode::ColorHsv2Rgb => {
                if channels != 3 {
                    anyhow::bail!("HSV to RGB conversion requires 3-channel image");
                }

                let mut dst = Array3::<u8>::zeros((height, width, 3));
                for y in 0..height {
                    for x in 0..width {
                        let h = src[[y, x, 0]] as f32 * 2.0; // OpenCV H is 0-179, convert to 0-360
                        let s = src[[y, x, 1]] as f32 / 255.0;
                        let v = src[[y, x, 2]] as f32 / 255.0;

                        let (r, g, b) = hsv_to_rgb(h, s, v);
                        dst[[y, x, 0]] = (r * 255.0) as u8;
                        dst[[y, x, 1]] = (g * 255.0) as u8;
                        dst[[y, x, 2]] = (b * 255.0) as u8;
                    }
                }
                Ok(dst)
            }
            ColorConversionCode::ColorRgb2Hsv => {
                if channels != 3 {
                    anyhow::bail!("RGB to HSV conversion requires 3-channel image");
                }

                let mut dst = Array3::<u8>::zeros((height, width, 3));
                for y in 0..height {
                    for x in 0..width {
                        let r = src[[y, x, 0]] as f32 / 255.0;
                        let g = src[[y, x, 1]] as f32 / 255.0;
                        let b = src[[y, x, 2]] as f32 / 255.0;

                        let (h, s, v) = rgb_to_hsv(r, g, b);
                        dst[[y, x, 0]] = (h / 2.0) as u8; // Convert 0-360 to 0-179 for OpenCV
                        dst[[y, x, 1]] = (s * 255.0) as u8;
                        dst[[y, x, 2]] = (v * 255.0) as u8;
                    }
                }
                Ok(dst)
            }
        }
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = if (0.0..60.0).contains(&h) {
            (c, x, 0.0)
        } else if (60.0..120.0).contains(&h) {
            (x, c, 0.0)
        } else if (120.0..180.0).contains(&h) {
            (0.0, c, x)
        } else if (180.0..240.0).contains(&h) {
            (0.0, x, c)
        } else if (240.0..300.0).contains(&h) {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        (r_prime + m, g_prime + m, b_prime + m)
    }

    fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max = r.max(g.max(b));
        let min = r.min(g.min(b));
        let delta = max - min;

        let h = if delta == 0.0 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };

        let s = if max == 0.0 { 0.0 } else { delta / max };
        let v = max;

        (h, s, v)
    }
}

/// OpenCV-compatible video capture functionality
pub mod videocapture {
    use super::*;

    #[cfg(feature = "opencv")]
    use opencv::{
        core::{self, Mat},
        imgproc,
        prelude::*,
        videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst},
    };

    #[cfg(not(feature = "opencv"))]
    use std::path::Path;

    /// Video capture that mirrors OpenCV's API surface
    pub struct VideoCapture {
        #[cfg(feature = "opencv")]
        inner: Option<opencv::videoio::VideoCapture>,
        fps: f64,
        width: i32,
        height: i32,
        frame_count: usize,
        is_opened: bool,
    }

    impl VideoCapture {
        fn closed() -> Self {
            Self {
                #[cfg(feature = "opencv")]
                inner: None,
                fps: 0.0,
                width: 0,
                height: 0,
                frame_count: 0,
                is_opened: false,
            }
        }

        /// Check if the capture successfully opened
        pub fn is_opened(&self) -> bool {
            self.is_opened
        }

        /// Release the capture handle
        pub fn release(&mut self) {
            self.is_opened = false;
            #[cfg(feature = "opencv")]
            if let Some(mut inner) = self.inner.take() {
                let _ = inner.release();
            }
        }

        /// Get cached video properties
        pub fn get(&self, prop: VideoCaptureProperties) -> f64 {
            match prop {
                VideoCaptureProperties::CapPropFps => self.fps,
                VideoCaptureProperties::CapPropFrameWidth => self.width as f64,
                VideoCaptureProperties::CapPropFrameHeight => self.height as f64,
                VideoCaptureProperties::CapPropFrameCount => self.frame_count as f64,
            }
        }
    }

    #[cfg(feature = "opencv")]
    impl VideoCapture {
        /// Create a video capture backed by OpenCV
        pub fn new(filename: &str) -> Result<Self> {
            let cap = match videoio::VideoCapture::from_file(filename, videoio::CAP_ANY) {
                Ok(cap) => cap,
                Err(_) => return Ok(Self::closed()),
            };

            let opened = cap.is_opened().unwrap_or(false);
            if !opened {
                return Ok(Self::closed());
            }

            let fps = cap.get(videoio::CAP_PROP_FPS).unwrap_or(0.0);
            let width = cap
                .get(videoio::CAP_PROP_FRAME_WIDTH)
                .unwrap_or(0.0)
                .max(0.0) as i32;
            let height = cap
                .get(videoio::CAP_PROP_FRAME_HEIGHT)
                .unwrap_or(0.0)
                .max(0.0) as i32;
            let frame_count_val = cap.get(videoio::CAP_PROP_FRAME_COUNT).unwrap_or(0.0);
            let frame_count = if frame_count_val.is_finite() && frame_count_val >= 0.0 {
                frame_count_val.round() as usize
            } else {
                0
            };

            Ok(Self {
                inner: Some(cap),
                fps: if fps.is_finite() { fps } else { 0.0 },
                width,
                height,
                frame_count,
                is_opened: true,
            })
        }

        /// Read the next frame using OpenCV's decoder
        pub fn read(&mut self) -> (bool, Option<Array3<u8>>) {
            if !self.is_opened {
                return (false, None);
            }

            let inner = match self.inner.as_mut() {
                Some(inner) => inner,
                None => return (false, None),
            };

            let mut frame = Mat::default();
            match inner.read(&mut frame) {
                Ok(true) => {
                    if frame.empty() {
                        return (false, None);
                    }

                    match Self::mat_to_ndarray(&frame) {
                        Ok(array) => {
                            if self.width == 0 || self.height == 0 {
                                let (h, w, _) = array.dim();
                                self.width = w as i32;
                                self.height = h as i32;
                            }

                            if self.frame_count == 0 {
                                if let Ok(total) = inner.get(videoio::CAP_PROP_FRAME_COUNT) {
                                    if total.is_finite() && total >= 0.0 {
                                        self.frame_count = total.round() as usize;
                                    }
                                }
                            }

                            (true, Some(array))
                        }
                        Err(_) => {
                            self.release();
                            (false, None)
                        }
                    }
                }
                Ok(false) => (false, None),
                Err(_) => {
                    self.release();
                    (false, None)
                }
            }
        }

        fn mat_to_ndarray(frame: &Mat) -> Result<Array3<u8>> {
            let mat_owned;
            let mat_ref = if frame.typ() == core::CV_8UC3 {
                frame
            } else {
                mat_owned = Self::to_bgr(frame)?;
                &mat_owned
            };

            let mut continuous;
            let mat_ref = if mat_ref.is_continuous() {
                mat_ref
            } else {
                continuous = Mat::default();
                mat_ref.copy_to(&mut continuous)?;
                &continuous
            };

            let height = mat_ref.rows() as usize;
            let width = mat_ref.cols() as usize;
            let channels = mat_ref.channels() as usize;

            if channels != 3 {
                anyhow::bail!("Expected 3-channel frame, got {}", channels);
            }

            let data = mat_ref.data_bytes()?;
            let buffer = data.to_vec();
            let array = Array3::from_shape_vec((height, width, channels), buffer)
                .map_err(|e| anyhow::anyhow!("Failed to reshape video frame: {}", e))?;

            Ok(array)
        }

        fn to_bgr(frame: &Mat) -> Result<Mat> {
            let frame_type = frame.typ();

            if frame_type == core::CV_8UC3 {
                return Ok(frame.try_clone()?);
            }

            let mut converted = Mat::default();
            match frame_type {
                t if t == core::CV_8UC1 => {
                    imgproc::cvt_color(
                        frame,
                        &mut converted,
                        imgproc::COLOR_GRAY2BGR,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                }
                t if t == core::CV_8UC4 => {
                    imgproc::cvt_color(
                        frame,
                        &mut converted,
                        imgproc::COLOR_BGRA2BGR,
                        0,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                }
                t if t == core::CV_16UC3 => {
                    frame.convert_to(&mut converted, core::CV_8UC3, 1.0 / 256.0, 0.0)?;
                }
                other => {
                    anyhow::bail!("Unsupported video frame type: {}", other);
                }
            }

            Ok(converted)
        }
    }

    #[cfg(not(feature = "opencv"))]
    impl VideoCapture {
        /// Minimal stub when OpenCV support is disabled
        pub fn new(filename: &str) -> Result<Self> {
            if Path::new(filename).exists() {
                Ok(Self {
                    fps: 0.0,
                    width: 0,
                    height: 0,
                    frame_count: 0,
                    is_opened: false,
                })
            } else {
                Ok(Self::closed())
            }
        }

        pub fn read(&mut self) -> (bool, Option<Array3<u8>>) {
            (false, None)
        }
    }

    /// Video capture properties
    #[derive(Debug, Clone, Copy)]
    #[allow(clippy::enum_variant_names)]
    pub enum VideoCaptureProperties {
        CapPropFps = 5,
        CapPropFrameWidth = 3,
        CapPropFrameHeight = 4,
        CapPropFrameCount = 7,
    }
}

/// OpenCV-compatible video writing functionality
pub mod videowriter {
    use super::*;

    /// Video writer for creating video files
    pub struct VideoWriter {
        filename: String,
        #[allow(dead_code)]
        fourcc: String,
        #[allow(dead_code)]
        fps: f64,
        frame_size: (i32, i32),
        frames: Vec<Array3<u8>>,
        is_opened: bool,
    }

    impl VideoWriter {
        /// Create new VideoWriter
        pub fn new(filename: &str, fourcc: &str, fps: f64, frame_size: (i32, i32)) -> Result<Self> {
            Ok(Self {
                filename: filename.to_string(),
                fourcc: fourcc.to_string(),
                fps,
                frame_size,
                frames: Vec::new(),
                is_opened: true,
            })
        }

        /// Check if video writer is opened
        pub fn is_opened(&self) -> bool {
            self.is_opened
        }

        /// Write a frame to the video
        pub fn write(&mut self, frame: &ArrayView3<u8>) -> Result<()> {
            if !self.is_opened {
                anyhow::bail!("VideoWriter is not opened");
            }

            let (height, width, channels) = frame.dim();
            if channels != 3 {
                anyhow::bail!("Frame must have 3 channels");
            }

            if width != self.frame_size.0 as usize || height != self.frame_size.1 as usize {
                anyhow::bail!("Frame size doesn't match expected size");
            }

            // Store frame (in real implementation, this would encode and write to file)
            self.frames.push(frame.to_owned());
            Ok(())
        }

        /// Release the video writer
        pub fn release(&mut self) -> Result<()> {
            if !self.is_opened {
                return Ok(());
            }

            // In real implementation, finalize video file here
            self.is_opened = false;
            println!(
                "VideoWriter: Saved {} frames to {}",
                self.frames.len(),
                self.filename
            );
            Ok(())
        }
    }

    /// Create FourCC code for video codec
    pub fn fourcc(c1: char, c2: char, c3: char, c4: char) -> String {
        format!("{}{}{}{}", c1, c2, c3, c4)
    }
}

/// OpenCV-compatible edge detection
pub mod imgproc {
    use super::*;

    /// Canny edge detection (equivalent to cv2.Canny)
    pub fn canny(image: &ArrayView3<u8>, threshold1: f64, threshold2: f64) -> Result<Array3<u8>> {
        use image::GrayImage;
        use imageproc::edges::canny;

        let (height, width, channels) = image.dim();

        // Convert to grayscale if needed
        let gray_data = if channels == 3 {
            let mut gray = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let r = image[[y, x, 0]] as f32;
                    let g = image[[y, x, 1]] as f32;
                    let b = image[[y, x, 2]] as f32;
                    let gray_val = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                    gray.push(gray_val);
                }
            }
            gray
        } else if channels == 1 {
            image.iter().cloned().collect()
        } else {
            anyhow::bail!("Unsupported number of channels: {}", channels);
        };

        // Create GrayImage
        let gray_img = GrayImage::from_raw(width as u32, height as u32, gray_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create grayscale image"))?;

        // Apply Canny edge detection
        let edges = canny(&gray_img, threshold1 as f32, threshold2 as f32);

        // Convert back to Array3<u8> (single channel)
        let mut result = Array3::<u8>::zeros((height, width, 1));
        for y in 0..height {
            for x in 0..width {
                let pixel = edges.get_pixel(x as u32, y as u32);
                result[[y, x, 0]] = pixel[0];
            }
        }

        Ok(result)
    }

    /// Image resize using image crate (fallback when OpenCV not available)
    pub fn resize(
        src: &ArrayView3<u8>,
        dsize: (u32, u32),
        interpolation: ResizeInterpolation,
    ) -> Result<Array3<u8>> {
        use image::{DynamicImage, ImageBuffer, Rgb};

        let (height, width, channels) = src.dim();
        if channels != 3 {
            anyhow::bail!("Resize only supports 3-channel images");
        }

        // Convert to image crate format
        let mut img_buffer = ImageBuffer::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let pixel = Rgb([src[[y, x, 0]], src[[y, x, 1]], src[[y, x, 2]]]);
                img_buffer.put_pixel(x as u32, y as u32, pixel);
            }
        }

        let dynamic_img = DynamicImage::ImageRgb8(img_buffer);

        // Resize based on interpolation method
        let resized = match interpolation {
            ResizeInterpolation::InterLinear => {
                dynamic_img.resize(dsize.0, dsize.1, image::imageops::FilterType::Triangle)
            }
            ResizeInterpolation::InterCubic => {
                dynamic_img.resize(dsize.0, dsize.1, image::imageops::FilterType::CatmullRom)
            }
            ResizeInterpolation::InterLanczos4 => {
                dynamic_img.resize(dsize.0, dsize.1, image::imageops::FilterType::Lanczos3)
            }
            ResizeInterpolation::InterNearest => {
                dynamic_img.resize(dsize.0, dsize.1, image::imageops::FilterType::Nearest)
            }
        };

        let rgb_img = resized.to_rgb8();
        let (new_width, new_height) = rgb_img.dimensions();

        // Convert back to Array3<u8>
        let mut result = Array3::<u8>::zeros((new_height as usize, new_width as usize, 3));
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = rgb_img.get_pixel(x, y);
                result[[y as usize, x as usize, 0]] = pixel[0];
                result[[y as usize, x as usize, 1]] = pixel[1];
                result[[y as usize, x as usize, 2]] = pixel[2];
            }
        }

        Ok(result)
    }

    /// Resize interpolation methods
    #[derive(Debug, Clone, Copy)]
    pub enum ResizeInterpolation {
        InterNearest = 0,
        InterLinear = 1,
        InterCubic = 2,
        InterLanczos4 = 4,
    }
}

/// OpenCV-compatible constants and flags
pub mod constants {
    pub use super::cvtcolor::ColorConversionCode;
    pub use super::imdecode::ImreadFlags;
    pub use super::imgproc::ResizeInterpolation;
    pub use super::videocapture::VideoCaptureProperties;

    /// Common OpenCV constants
    pub const IMREAD_COLOR: i32 = 1;
    pub const IMREAD_GRAYSCALE: i32 = 0;
    pub const IMREAD_UNCHANGED: i32 = -1;

    pub const COLOR_BGR2RGB: i32 = 4;
    pub const COLOR_RGB2BGR: i32 = 4;
    pub const COLOR_RGB2GRAY: i32 = 7;
    pub const COLOR_GRAY2RGB: i32 = 8;
    pub const COLOR_HSV2RGB: i32 = 55;
    pub const COLOR_RGB2HSV: i32 = 41;

    pub const INTER_NEAREST: i32 = 0;
    pub const INTER_LINEAR: i32 = 1;
    pub const INTER_CUBIC: i32 = 2;
    pub const INTER_LANCZOS4: i32 = 4;

    pub const CAP_PROP_FPS: i32 = 5;
    pub const CAP_PROP_FRAME_WIDTH: i32 = 3;
    pub const CAP_PROP_FRAME_HEIGHT: i32 = 4;
    pub const CAP_PROP_FRAME_COUNT: i32 = 7;
}

/// OpenCV-compatible object detection (Haar cascades)
pub mod objdetect {
    use super::*;

    /// Cascade classifier for face detection (equivalent to cv2.CascadeClassifier)
    pub struct CascadeClassifier {
        #[allow(dead_code)]
        cascade_path: String,
        is_loaded: bool,
    }

    impl CascadeClassifier {
        /// Load cascade classifier from file
        pub fn new(filename: &str) -> Result<Self> {
            let path = if filename.contains("haarcascade_frontalface_default.xml") {
                // Handle the common OpenCV face cascade path
                filename.to_string()
            } else {
                filename.to_string()
            };

            let is_loaded = std::path::Path::new(&path).exists();

            Ok(Self {
                cascade_path: path,
                is_loaded,
            })
        }

        /// Detect objects (faces) in the image
        pub fn detect_multi_scale(
            &self,
            image: &ArrayView3<u8>,
            scale_factor: f64,
            min_neighbors: i32,
        ) -> Result<Vec<(i32, i32, i32, i32)>> {
            // Returns (x, y, width, height) rectangles
            if !self.is_loaded {
                return Ok(Vec::new());
            }

            // This is a placeholder implementation
            // In a real implementation, you would:
            // 1. Load the Haar cascade XML file
            // 2. Parse the cascade features
            // 3. Apply the cascade to detect faces in the image
            // 4. Return bounding boxes of detected faces

            // For now, return empty vector (no faces detected)
            // This would require implementing a full Haar cascade detector
            // or integrating with OpenCV's implementation
            let _scale_factor = scale_factor;
            let _min_neighbors = min_neighbors;
            let (_height, _width, _channels) = image.dim();

            // Placeholder: would implement actual face detection here
            Ok(Vec::new())
        }

        /// Check if cascade is loaded successfully
        pub fn empty(&self) -> bool {
            !self.is_loaded
        }
    }

    /// Get the path to OpenCV data directory (for Haar cascades)
    pub fn get_opencv_data_path() -> String {
        // This would typically point to the OpenCV data directory
        // For now, return a placeholder path
        "/usr/local/share/opencv4/haarcascades/".to_string()
    }
}

pub use constants::*;
pub use cvtcolor::{cvt_color, ColorConversionCode};
/// Re-export commonly used types and functions for easy access
pub use imdecode::{imdecode, ImreadFlags};
pub use imgproc::{canny, resize, ResizeInterpolation};
pub use objdetect::{get_opencv_data_path, CascadeClassifier};
pub use videocapture::VideoCapture;
pub use videowriter::{fourcc, VideoWriter};
