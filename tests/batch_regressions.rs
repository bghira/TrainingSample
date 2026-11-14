use ndarray::{s, Array3};
use trainingsample::{
    BatchProcessor, ColorConversion, ColorConversionCode, ImreadFlags, OptimizedBatchProcessor,
    ResizeInterpolation, TrueBatchProcessor,
};

fn image() -> Array3<u8> {
    Array3::from_shape_fn((4, 17, 3), |(y, x, c)| (y * 41 + x * 7 + c * 53) as u8)
}

#[test]
fn optimized_channel_swap_preserves_pixel_boundaries() {
    let image = image();
    for parallel in [false, true] {
        let processor = OptimizedBatchProcessor {
            use_parallel: parallel,
            chunk_size: 1,
        };
        let results = processor
            .batch_cvt_color_optimized(&[image.view(); 3], ColorConversionCode::ColorRgb2Bgr)
            .unwrap();
        for result in results {
            assert_eq!(result, image.slice(s![.., .., ..;-1]));
        }
    }
}

#[test]
fn batch_conversions_respect_strides() {
    let image = image();
    for view in [
        image.slice(s![..;2, ..;2, ..]),
        image.slice(s![..;-1, ..;-1, ..;-1]),
        image.view().permuted_axes([1, 0, 2]),
    ] {
        let contiguous = view.as_standard_layout().into_owned();
        let optimized = OptimizedBatchProcessor::new();
        for code in [
            ColorConversionCode::ColorRgb2Bgr,
            ColorConversionCode::ColorRgb2Gray,
        ] {
            assert_eq!(
                optimized.batch_cvt_color_optimized(&[view], code).unwrap(),
                optimized
                    .batch_cvt_color_optimized(&[contiguous.view()], code)
                    .unwrap(),
            );
        }
        let processor = TrueBatchProcessor::new();
        for code in [ColorConversion::RgbToBgr, ColorConversion::RgbToGray] {
            assert_eq!(
                processor.true_batch_cvt_color(&[view], code).unwrap(),
                processor
                    .true_batch_cvt_color(&[contiguous.view()], code)
                    .unwrap(),
            );
        }
    }
}

#[test]
fn sampled_luminance_respects_strides() {
    let image = image();
    let view = image.slice(s![..;2, ..;2, ..]);
    let contiguous = view.as_standard_layout().into_owned();
    let processor = TrueBatchProcessor::new();
    assert_eq!(
        processor.strided_luminance(&[view], 1).unwrap(),
        processor
            .strided_luminance(&[contiguous.view()], 1)
            .unwrap()
    );
}

#[test]
fn true_resize_respects_strides() {
    let image = image();
    let view = image.slice(s![..;2, ..;2, ..]);
    let result = TrueBatchProcessor::new()
        .true_batch_resize(&[view], &[(9, 2)], ResizeInterpolation::InterNearest)
        .unwrap();
    assert_eq!(result[0], view);
}

#[test]
fn true_gray_to_rgb_is_supported() {
    let gray = Array3::from_shape_fn((2, 3, 1), |(y, x, _)| (y * 10 + x) as u8);
    let result = TrueBatchProcessor::new()
        .true_batch_cvt_color(&[gray.view()], ColorConversion::GrayToRgb)
        .unwrap();
    assert_eq!(
        result[0],
        Array3::from_shape_fn((2, 3, 3), |(y, x, _)| gray[[y, x, 0]])
    );
}

#[test]
fn empty_pipeline_and_zero_chunk_size_do_not_panic() {
    for parallel in [false, true] {
        let processor = BatchProcessor::with_config(parallel, 0);
        assert!(processor
            .batch_preprocess_pipeline(
                &[],
                &[],
                None,
                ImreadFlags::ImreadColor,
                ResizeInterpolation::InterNearest
            )
            .unwrap()
            .is_empty());
        assert!(processor.batch_imdecode(&[b"invalid"], ImreadFlags::ImreadColor)[0].is_err());
        let image = image();
        assert_eq!(
            processor
                .batch_cvt_color(&[image.view()], ColorConversionCode::ColorRgb2Bgr)
                .unwrap()[0],
            image.slice(s![.., .., ..;-1])
        );
    }
}

#[test]
fn resize_uses_exact_requested_dimensions() {
    let image = image();
    for interpolation in [
        ResizeInterpolation::InterNearest,
        ResizeInterpolation::InterLinear,
        ResizeInterpolation::InterCubic,
        ResizeInterpolation::InterLanczos4,
    ] {
        let result = trainingsample::resize(&image.view(), (3, 8), interpolation).unwrap();
        assert_eq!(result.dim(), (8, 3, 3));
    }
}

#[test]
fn true_resize_rejects_empty_batch_size_mismatch() {
    assert!(TrueBatchProcessor::new()
        .true_batch_resize(&[], &[(2, 2)], ResizeInterpolation::InterNearest)
        .is_err());
}

#[test]
fn parallel_resize_preserves_input_order_and_propagates_errors() {
    let images: Vec<_> = (0..8)
        .map(|i| Array3::from_elem((3, 4, 3), i * 30))
        .collect();
    let views: Vec<_> = images.iter().map(|image| image.view()).collect();
    let sizes: Vec<_> = (0..8).map(|i| (2 + i % 3, 2 + i % 2)).collect();
    let processor = BatchProcessor::with_config(true, 1);
    let optimized = OptimizedBatchProcessor {
        use_parallel: true,
        chunk_size: 1,
    };
    let interpolation = ResizeInterpolation::InterNearest;
    for results in [
        processor
            .batch_resize(&views, &sizes, interpolation)
            .unwrap(),
        optimized
            .batch_resize_optimized(&views, &sizes, interpolation)
            .unwrap(),
    ] {
        for (i, (result, &(width, height))) in results.iter().zip(&sizes).enumerate() {
            assert_eq!(result.dim(), (height as usize, width as usize, 3));
            assert!(result.iter().all(|&pixel| pixel == i as u8 * 30));
        }
    }
    let mut invalid_sizes = sizes;
    invalid_sizes[3] = (0, 1);
    assert!(processor
        .batch_resize(&views, &invalid_sizes, interpolation)
        .is_err());
    assert!(optimized
        .batch_resize_optimized(&views, &invalid_sizes, interpolation)
        .is_err());
}

#[cfg(feature = "simd")]
#[test]
fn rgba_conversion_handles_vector_tails_and_strided_inputs() {
    for width in [1, 7, 8, 9, 15, 16, 17] {
        let image =
            Array3::from_shape_fn((4, width, 3), |(y, x, c)| (y * 31 + x * 7 + c * 53) as u8);
        for view in [image.view(), image.slice(s![..;-1, ..;-1, ..;-1])] {
            let (actual, _) = trainingsample::rgb_to_rgba_optimized(&view, 173);
            let (expected, _) = trainingsample::rgb_to_rgba_scalar(&view, 173);
            assert_eq!(actual, expected, "width {width}");
        }
    }
}
