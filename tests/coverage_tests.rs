use image::{DynamicImage, ImageFormat, RgbImage};
use ndarray::{Array3, Array4};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use trainingsample::{
    batch_load_images, fourcc, load_and_decode_image, load_image_from_path, BatchProcessor,
    CascadeClassifier, ColorConversion, ColorConversionCode, ImreadFlags, OptimizedBatchProcessor,
    ResizeInterpolation, TrueBatchProcessor, VideoCapture, VideoCaptureProperties, VideoOperation,
    VideoWriter,
};

#[cfg(feature = "simd")]
use trainingsample::{
    rgb_to_rgba_optimized, rgb_to_rgba_scalar, rgba_to_rgb_optimized, rgba_to_rgb_scalar,
    FormatConversionMetrics,
};

fn sample_rgb(height: usize, width: usize) -> Array3<u8> {
    Array3::from_shape_fn((height, width, 3), |(y, x, channel)| {
        ((y * 37 + x * 19 + channel * 71) % 256) as u8
    })
}

fn encode_png(image: &Array3<u8>) -> Vec<u8> {
    let (height, width, channels) = image.dim();
    assert_eq!(channels, 3);

    let rgb = RgbImage::from_raw(width as u32, height as u32, image.iter().copied().collect())
        .expect("test image dimensions should match its data");
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(rgb)
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("test image should encode");
    encoded.into_inner()
}

fn temporary_path(extension: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "trainingsample-coverage-{}-{nonce}.{extension}",
        std::process::id()
    ))
}

struct TemporaryImage(PathBuf);

impl TemporaryImage {
    fn create(contents: &[u8]) -> Self {
        let path = temporary_path("png");
        std::fs::write(&path, contents).expect("temporary image should be writable");
        Self(path)
    }
}

impl Drop for TemporaryImage {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
#[cfg(feature = "simd")]
fn format_conversion_implementations_preserve_pixels() {
    let image = sample_rgb(2, 8);

    let (scalar_rgba, scalar_metrics) = rgb_to_rgba_scalar(&image.view(), 173);
    let (optimized_rgba, optimized_metrics) = rgb_to_rgba_optimized(&image.view(), 173);
    assert_eq!(optimized_rgba, scalar_rgba);
    assert!(scalar_rgba.chunks_exact(4).all(|pixel| pixel[3] == 173));
    assert_eq!(scalar_metrics.pixels_processed, 16);
    assert_eq!(scalar_metrics.simd_width, 1);
    assert_eq!(scalar_metrics.implementation, "scalar_rgb_to_rgba");
    assert_eq!(optimized_metrics.pixels_processed, 16);
    assert_eq!(optimized_metrics.simd_width, 8);
    assert_eq!(optimized_metrics.implementation, "simd_rgb_to_rgba");

    let (scalar_rgb, _) = rgba_to_rgb_scalar(&scalar_rgba, 8, 2);
    let (optimized_rgb, optimized_reverse_metrics) = rgba_to_rgb_optimized(&scalar_rgba, 8, 2);
    assert_eq!(optimized_rgb, scalar_rgb);
    assert_eq!(scalar_rgb, image.iter().copied().collect::<Vec<_>>());
    assert_eq!(optimized_reverse_metrics.pixels_processed, 16);

    let metrics = FormatConversionMetrics::new(1_000_000, 1_000_000_000, 4, "test");
    assert_eq!(metrics.throughput_mpixels_per_sec, 1.0);
    assert_eq!(metrics.simd_width, 4);
}

#[test]
fn loading_functions_return_raw_and_decoded_images() {
    let image = sample_rgb(3, 5);
    let encoded = encode_png(&image);
    let temporary = TemporaryImage::create(&encoded);
    let path = temporary
        .0
        .to_str()
        .expect("temporary test path should be valid UTF-8");

    assert_eq!(load_image_from_path(path).unwrap(), encoded);
    assert_eq!(
        load_and_decode_image(path).unwrap().to_rgb8().dimensions(),
        (5, 3)
    );

    let missing = temporary.0.with_extension("missing");
    let results = batch_load_images(&[temporary.0.as_path(), missing.as_path()]);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].as_ref().unwrap(), &encoded);
    assert!(results[1].is_err());
}

#[test]
fn batch_processor_covers_sequential_pipelines_and_validation() {
    let processor = BatchProcessor::with_config(false, 2);
    let image = sample_rgb(8, 8);
    let second = sample_rgb(6, 10);
    let views = [image.view(), second.view()];
    let encoded = encode_png(&image);

    let decoded = processor.batch_imdecode(&[encoded.as_slice()], ImreadFlags::ImreadColor);
    assert_eq!(decoded[0].as_ref().unwrap().dim(), image.dim());
    assert!(processor.batch_imdecode(&[b"not an image"], ImreadFlags::ImreadColor)[0].is_err());

    assert!(processor
        .batch_cvt_color(&[], ColorConversionCode::ColorBgr2Rgb)
        .unwrap()
        .is_empty());
    let swapped = processor
        .batch_cvt_color(&views, ColorConversionCode::ColorBgr2Rgb)
        .unwrap();
    assert_eq!(swapped[0][[0, 0, 0]], image[[0, 0, 2]]);
    assert_eq!(swapped[0][[0, 0, 2]], image[[0, 0, 0]]);

    assert!(processor
        .batch_resize(&views, &[(4, 4)], ResizeInterpolation::InterLinear)
        .is_err());
    assert!(processor
        .batch_resize(&[], &[], ResizeInterpolation::InterLinear)
        .unwrap()
        .is_empty());
    let resized = processor
        .batch_resize(&views, &[(4, 4), (5, 3)], ResizeInterpolation::InterNearest)
        .unwrap();
    assert_eq!(resized[0].dim(), (4, 4, 3));
    assert_eq!(resized[1].dim(), (3, 5, 3));

    assert!(processor.batch_canny(&[], 25.0, 50.0).unwrap().is_empty());
    let edges = processor.batch_canny(&views, 25.0, 50.0).unwrap();
    assert_eq!(edges[0].dim(), (8, 8, 1));
    assert_eq!(edges[1].dim(), (6, 10, 1));

    assert!(processor
        .batch_preprocess_pipeline(
            &[encoded.as_slice()],
            &[],
            None,
            ImreadFlags::ImreadColor,
            ResizeInterpolation::InterLinear,
        )
        .is_err());
    let preprocessed = processor
        .batch_preprocess_pipeline(
            &[encoded.as_slice()],
            &[(5, 5)],
            Some(ColorConversionCode::ColorRgb2Gray),
            ImreadFlags::ImreadColor,
            ResizeInterpolation::InterCubic,
        )
        .unwrap();
    assert_eq!(preprocessed[0].dim(), (5, 5, 1));

    let detections = processor
        .batch_face_detection_pipeline(&views, 1.1, 3)
        .unwrap();
    assert_eq!(detections, vec![Vec::new(), Vec::new()]);
}

#[test]
fn batch_processor_covers_parallel_and_video_paths() {
    let processor = BatchProcessor::with_config(true, 1);
    let images = [sample_rgb(5, 6), sample_rgb(5, 6), sample_rgb(5, 6)];
    let views: Vec<_> = images.iter().map(|image| image.view()).collect();
    let buffers: Vec<_> = images.iter().map(encode_png).collect();
    let buffer_views: Vec<_> = buffers.iter().map(Vec::as_slice).collect();

    assert_eq!(
        processor
            .batch_imdecode(&buffer_views, ImreadFlags::ImreadColor)
            .len(),
        3
    );
    assert_eq!(
        processor
            .batch_cvt_color(&views, ColorConversionCode::ColorRgb2Gray)
            .unwrap()
            .len(),
        3
    );
    let resized = processor
        .batch_resize(
            &views,
            &[(12, 10), (6, 5), (12, 10)],
            ResizeInterpolation::InterLanczos4,
        )
        .unwrap();
    assert_eq!(resized[0].dim(), (10, 12, 3));
    assert_eq!(resized[1].dim(), (5, 6, 3));
    assert_eq!(processor.batch_canny(&views, 10.0, 20.0).unwrap().len(), 3);

    let video = Array4::from_shape_fn((2, 5, 6, 3), |(frame, y, x, channel)| {
        ((frame * 43 + y * 17 + x * 11 + channel * 67) % 256) as u8
    });
    let processed = processor
        .batch_video_frame_processing(
            &[video.view()],
            (12, 10),
            &[
                VideoOperation::Resize(ResizeInterpolation::InterLinear),
                VideoOperation::ColorConvert(ColorConversionCode::ColorBgr2Rgb),
                VideoOperation::EdgeDetection(10.0, 20.0),
            ],
        )
        .unwrap();
    assert_eq!(processed[0].dim(), (2, 10, 12, 1));

    let empty_video = Array4::<u8>::zeros((0, 5, 6, 3));
    let empty_processed = processor
        .batch_video_frame_processing(&[empty_video.view()], (4, 3), &[])
        .unwrap();
    assert_eq!(empty_processed[0].dim(), empty_video.dim());
}

#[test]
fn optimized_batch_processor_covers_conversion_and_resize_paths() {
    let sequential = OptimizedBatchProcessor {
        use_parallel: false,
        chunk_size: 8,
    };
    let image = sample_rgb(2, 4);
    let views = [image.view()];

    assert!(sequential
        .batch_cvt_color_optimized(&[], ColorConversionCode::ColorBgr2Rgb)
        .unwrap()
        .is_empty());
    let swapped = sequential
        .batch_cvt_color_optimized(&views, ColorConversionCode::ColorBgr2Rgb)
        .unwrap();
    assert_eq!(swapped[0][[1, 3, 0]], image[[1, 3, 2]]);
    assert_eq!(swapped[0][[1, 3, 2]], image[[1, 3, 0]]);

    let grayscale = sequential
        .batch_cvt_color_optimized(&views, ColorConversionCode::ColorRgb2Gray)
        .unwrap();
    assert_eq!(grayscale[0].dim(), (2, 4, 1));

    let single_channel = Array3::from_shape_vec((1, 2, 1), vec![25, 200]).unwrap();
    let expanded = sequential
        .batch_cvt_color_optimized(&[single_channel.view()], ColorConversionCode::ColorGray2Rgb)
        .unwrap();
    assert_eq!(expanded[0].dim(), (1, 2, 3));
    assert_eq!(expanded[0][[0, 1, 0]], 200);
    assert!(sequential
        .batch_cvt_color_optimized(&[single_channel.view()], ColorConversionCode::ColorRgb2Gray,)
        .is_err());

    assert!(sequential
        .batch_resize_optimized(&views, &[], ResizeInterpolation::InterLinear)
        .is_err());
    let resized = sequential
        .batch_resize_optimized(&views, &[(2, 1)], ResizeInterpolation::InterNearest)
        .unwrap();
    assert_eq!(resized[0].dim(), (1, 2, 3));

    let parallel = OptimizedBatchProcessor {
        use_parallel: true,
        chunk_size: 1,
    };
    let parallel_images = [sample_rgb(4, 4), sample_rgb(4, 4), sample_rgb(4, 4)];
    let parallel_views: Vec<_> = parallel_images.iter().map(|image| image.view()).collect();
    assert_eq!(
        parallel
            .batch_cvt_color_optimized(&parallel_views, ColorConversionCode::ColorRgb2Gray)
            .unwrap()
            .len(),
        3
    );
    let parallel_resized = parallel
        .batch_resize_optimized(
            &parallel_views,
            &[(2, 2), (3, 3), (2, 2)],
            ResizeInterpolation::InterCubic,
        )
        .unwrap();
    assert_eq!(parallel_resized[0].dim(), (2, 2, 3));
    assert_eq!(parallel_resized[1].dim(), (3, 3, 3));
}

#[test]
fn true_batch_processor_covers_resize_color_and_luminance_paths() {
    let sequential = TrueBatchProcessor {
        use_parallel: false,
        chunk_size: 8,
        simd_threshold: 4,
    };
    let image = sample_rgb(4, 5);
    let views = [image.view()];

    assert!(sequential
        .true_batch_resize(&[], &[], ResizeInterpolation::InterLinear)
        .unwrap()
        .is_empty());
    #[cfg(feature = "opencv")]
    assert!(sequential
        .true_batch_resize(&views, &[], ResizeInterpolation::InterLinear)
        .is_err());

    for interpolation in [
        ResizeInterpolation::InterNearest,
        ResizeInterpolation::InterLinear,
        ResizeInterpolation::InterCubic,
        ResizeInterpolation::InterLanczos4,
    ] {
        let resized = sequential
            .true_batch_resize(&views, &[(3, 2)], interpolation)
            .unwrap();
        assert_eq!(resized[0].dim(), (2, 3, 3));
    }

    assert!(sequential
        .true_batch_cvt_color(&[], ColorConversion::BgrToRgb)
        .unwrap()
        .is_empty());
    let swapped = sequential
        .true_batch_cvt_color(&views, ColorConversion::BgrToRgb)
        .unwrap();
    assert_eq!(swapped[0][[0, 0, 0]], image[[0, 0, 2]]);
    assert_eq!(swapped[0][[0, 0, 2]], image[[0, 0, 0]]);
    assert_eq!(
        sequential
            .true_batch_cvt_color(&views, ColorConversion::RgbToBgr)
            .unwrap()[0]
            .dim(),
        image.dim()
    );
    assert_eq!(
        sequential
            .true_batch_cvt_color(&views, ColorConversion::RgbToGray)
            .unwrap()[0]
            .dim(),
        (4, 5, 1)
    );
    assert!(sequential
        .true_batch_cvt_color(&views, ColorConversion::GrayToRgb)
        .is_err());

    let luminance = sequential.strided_luminance(&views, 0).unwrap();
    assert_eq!(luminance.len(), 1);
    assert!((0.0..=255.0).contains(&luminance[0]));
    let empty_image = Array3::<u8>::zeros((0, 0, 3));
    assert_eq!(
        sequential
            .strided_luminance(&[empty_image.view()], 3)
            .unwrap(),
        vec![0.0]
    );
    let invalid_image = Array3::<u8>::zeros((2, 2, 1));
    assert!(sequential
        .strided_luminance(&[invalid_image.view()], 1)
        .is_err());

    let parallel = TrueBatchProcessor {
        use_parallel: true,
        chunk_size: 1,
        simd_threshold: 1,
    };
    let parallel_images = [sample_rgb(4, 5), sample_rgb(4, 5)];
    let parallel_views: Vec<_> = parallel_images.iter().map(|image| image.view()).collect();
    assert_eq!(
        parallel
            .strided_luminance(&parallel_views, 2)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn compatibility_objects_cover_success_and_error_states() {
    assert_eq!(fourcc('M', 'J', 'P', 'G'), "MJPG");

    let missing_video = temporary_path("avi");
    let mut capture = VideoCapture::new(missing_video.to_str().unwrap()).unwrap();
    assert!(!capture.is_opened());
    assert_eq!(capture.get(VideoCaptureProperties::CapPropFps), 0.0);
    assert_eq!(capture.get(VideoCaptureProperties::CapPropFrameWidth), 0.0);
    assert_eq!(capture.get(VideoCaptureProperties::CapPropFrameHeight), 0.0);
    assert_eq!(capture.get(VideoCaptureProperties::CapPropFrameCount), 0.0);
    assert_eq!(capture.read(), (false, None));
    capture.release();

    let image = sample_rgb(2, 3);
    let mut writer = VideoWriter::new("unused.avi", "MJPG", 24.0, (3, 2)).unwrap();
    assert!(writer.is_opened());
    writer.write(&image.view()).unwrap();
    assert!(writer
        .write(&Array3::<u8>::zeros((2, 3, 1)).view())
        .is_err());
    assert!(writer
        .write(&Array3::<u8>::zeros((3, 2, 3)).view())
        .is_err());
    writer.release().unwrap();
    writer.release().unwrap();
    assert!(!writer.is_opened());
    assert!(writer.write(&image.view()).is_err());

    let missing_cascade = temporary_path("xml");
    let cascade = CascadeClassifier::new(missing_cascade.to_str().unwrap()).unwrap();
    assert!(cascade.empty());
    assert!(cascade
        .detect_multi_scale(&image.view(), 1.1, 3)
        .unwrap()
        .is_empty());
}
