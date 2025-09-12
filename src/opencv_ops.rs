use anyhow::Result;
use ndarray::{Array3, ArrayView3};

#[cfg(feature = "opencv")]
use opencv::{
    core::Mat,
    imgproc::{cvt_color, resize, COLOR_RGB2GRAY, INTER_LANCZOS4},
    prelude::*,
};

/// OpenCV-powered batch resize operations for maximum performance parity
#[cfg(feature = "opencv")]
pub struct OpenCVBatchProcessor;

#[cfg(feature = "opencv")]
impl Default for OpenCVBatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "opencv")]
impl OpenCVBatchProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Batch resize using OpenCV's optimized implementation (benchmark winner)
    pub fn batch_resize_images(
        &self,
        images: &[ArrayView3<u8>],
        target_sizes: &[(u32, u32)], // (width, height)
    ) -> Result<Vec<Array3<u8>>> {
        if images.len() != target_sizes.len() {
            anyhow::bail!("Number of images and target sizes must match");
        }

        let results: Vec<_> = images
            .iter()
            .zip(target_sizes.iter())
            .map(|(image, &(target_width, target_height))| {
                self.resize_single_opencv(image, target_width, target_height)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(results)
    }

    /// Batch video resize using OpenCV
    pub fn batch_resize_videos(
        &self,
        videos: &[ndarray::ArrayView4<u8>],
        target_sizes: &[(u32, u32)], // (width, height)
    ) -> Result<Vec<ndarray::Array4<u8>>> {
        if videos.len() != target_sizes.len() {
            anyhow::bail!("Number of videos and target sizes must match");
        }

        // ULTRA-OPTIMIZED: Flatten all frames from all videos into one massive batch
        let mut all_frames = Vec::new();
        let mut all_target_sizes = Vec::new();
        let mut video_frame_counts = Vec::new();

        for (video, &target_size) in videos.iter().zip(target_sizes.iter()) {
            let (frames, _, _, _) = video.dim();
            video_frame_counts.push(frames);

            // Collect all frames from this video
            for frame_idx in 0..frames {
                let frame = video.index_axis(ndarray::Axis(0), frame_idx);
                all_frames.push(frame);
                all_target_sizes.push(target_size);
            }
        }

        // Process ALL frames in one giant batch operation - maximum efficiency!
        let resized_frames = self.batch_resize_images(&all_frames, &all_target_sizes)?;

        // Reconstruct video structure from flattened results
        let mut results = Vec::new();
        let mut frame_idx = 0;

        for (video_idx, frame_count) in video_frame_counts.iter().enumerate() {
            let (target_width, target_height) = target_sizes[video_idx];
            let result_shape = (
                *frame_count,
                target_height as usize,
                target_width as usize,
                3,
            );
            let mut video_result = ndarray::Array4::<u8>::zeros(result_shape);

            // Copy frames for this video
            for local_frame_idx in 0..*frame_count {
                let global_frame = &resized_frames[frame_idx];
                video_result
                    .index_axis_mut(ndarray::Axis(0), local_frame_idx)
                    .assign(global_frame);
                frame_idx += 1;
            }

            results.push(video_result);
        }

        Ok(results)
    }

    /// Batch resize using OpenCV's optimized LANCZOS4 implementation
    pub fn batch_resize_lanczos4(
        &self,
        images: &[ArrayView3<u8>],
        target_sizes: &[(u32, u32)], // (width, height)
    ) -> Result<Vec<Array3<u8>>> {
        // Use the same implementation as batch_resize_images but with Lanczos4
        self.batch_resize_images(images, target_sizes)
    }

    /// Single image resize using OpenCV
    fn resize_single_opencv(
        &self,
        image: &ArrayView3<u8>,
        target_width: u32,
        target_height: u32,
    ) -> Result<Array3<u8>> {
        let (_height, _width, channels) = image.dim();

        if channels != 3 {
            anyhow::bail!("Only 3-channel RGB images are supported");
        }

        // Convert ndarray to OpenCV Mat
        let src_mat = self.ndarray_to_mat(image)?;

        // Perform OpenCV resize
        let mut dst_mat = Mat::default();
        resize(
            &src_mat,
            &mut dst_mat,
            opencv::core::Size::new(target_width as i32, target_height as i32),
            0.0,
            0.0,
            INTER_LANCZOS4,
        )?;

        // Convert back to ndarray
        self.mat_to_ndarray(&dst_mat)
    }

    // Single video resize function removed - now using batch_resize_videos for all operations

    /// Batch luminance calculation using OpenCV's cvtColor
    pub fn batch_calculate_luminance_opencv(&self, images: &[ArrayView3<u8>]) -> Result<Vec<f64>> {
        images
            .iter()
            .map(|image| self.calculate_luminance_single_opencv(image))
            .collect()
    }

    /// Single image luminance using OpenCV
    fn calculate_luminance_single_opencv(&self, image: &ArrayView3<u8>) -> Result<f64> {
        let (_height, _width, channels) = image.dim();

        if channels != 3 {
            anyhow::bail!("Only 3-channel RGB images are supported");
        }

        // Convert to OpenCV Mat
        let src_mat = self.ndarray_to_mat(image)?;

        // Convert to grayscale using OpenCV's optimized implementation
        let mut gray_mat = Mat::default();
        // Different OpenCV versions have different cvt_color signatures
        #[cfg(target_os = "macos")]
        cvt_color(
            &src_mat,
            &mut gray_mat,
            COLOR_RGB2GRAY,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        #[cfg(not(target_os = "macos"))]
        cvt_color(&src_mat, &mut gray_mat, COLOR_RGB2GRAY, 0)?;

        // Calculate mean (luminance)
        let mean_scalar = opencv::core::mean(&gray_mat, &opencv::core::no_array())?;
        Ok(mean_scalar[0])
    }

    /// Convert ndarray to OpenCV Mat (optimized for performance)
    fn ndarray_to_mat(&self, image: &ArrayView3<u8>) -> Result<Mat> {
        let (height, width, channels) = image.dim();

        if channels != 3 {
            anyhow::bail!("Only 3-channel RGB images are supported");
        }

        let data_slice = image
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Image data is not contiguous"))?;

        // ULTRA-OPTIMIZED: Create Mat directly from contiguous data using bulk copy
        let mut mat = Mat::new_size_with_default(
            opencv::core::Size::new(width as i32, height as i32),
            opencv::core::CV_8UC3,
            opencv::core::Scalar::default(),
        )?;

        // Use bulk memory copy instead of pixel-by-pixel copying
        unsafe {
            let dst_ptr = mat.ptr_mut(0)?;
            std::ptr::copy_nonoverlapping(data_slice.as_ptr(), dst_ptr, height * width * channels);
        }

        Ok(mat)
    }

    /// Convert OpenCV Mat back to ndarray (optimized for performance)
    fn mat_to_ndarray(&self, mat: &Mat) -> Result<Array3<u8>> {
        let height = mat.rows() as usize;
        let width = mat.cols() as usize;
        let channels = mat.channels() as usize;

        if channels != 3 {
            anyhow::bail!("Expected 3-channel image");
        }

        let mut result = Array3::<u8>::zeros((height, width, channels));

        // Optimized bulk memory copy from OpenCV Mat to ndarray
        unsafe {
            let src_ptr = mat.ptr(0)?;
            let dst_ptr = result.as_mut_ptr();
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, height * width * channels);
        }

        Ok(result)
    }
}

// HIGH-PERFORMANCE RESIZE WRAPPERS (REPLACE METAL GPU FUNCTIONS)
// These provide the same API as the removed Metal functions but use OpenCV

#[cfg(feature = "opencv")]
pub fn resize_bilinear_opencv(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    let processor = OpenCVBatchProcessor::new();
    processor.resize_single_opencv(image, target_width, target_height)
}

#[cfg(feature = "opencv")]
pub fn resize_lanczos4_opencv(
    image: &ArrayView3<u8>,
    target_width: u32,
    target_height: u32,
) -> Result<Array3<u8>> {
    let processor = OpenCVBatchProcessor::new();
    processor.resize_single_opencv(image, target_width, target_height)
}

#[cfg(not(feature = "opencv"))]
pub fn resize_bilinear_opencv(
    _image: &ArrayView3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> Result<Array3<u8>> {
    anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
}

#[cfg(not(feature = "opencv"))]
pub fn resize_lanczos4_opencv(
    _image: &ArrayView3<u8>,
    _target_width: u32,
    _target_height: u32,
) -> Result<Array3<u8>> {
    anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
}

/// Fallback implementations when OpenCV is not available
#[cfg(not(feature = "opencv"))]
pub struct OpenCVBatchProcessor;

#[cfg(not(feature = "opencv"))]
impl Default for OpenCVBatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "opencv"))]
impl OpenCVBatchProcessor {
    pub fn new() -> Self {
        Self
    }

    pub fn batch_resize_images(
        &self,
        _images: &[ArrayView3<u8>],
        _target_sizes: &[(u32, u32)],
    ) -> Result<Vec<Array3<u8>>> {
        anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
    }

    pub fn batch_resize_videos(
        &self,
        _videos: &[ndarray::ArrayView4<u8>],
        _target_sizes: &[(u32, u32)],
    ) -> Result<Vec<ndarray::Array4<u8>>> {
        anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
    }

    pub fn batch_resize_lanczos4(
        &self,
        _images: &[ArrayView3<u8>],
        _target_sizes: &[(u32, u32)],
    ) -> Result<Vec<Array3<u8>>> {
        anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
    }

    pub fn batch_calculate_luminance_opencv(&self, _images: &[ArrayView3<u8>]) -> Result<Vec<f64>> {
        anyhow::bail!("OpenCV feature not enabled. Rebuild with --features opencv")
    }
}
