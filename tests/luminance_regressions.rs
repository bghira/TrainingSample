use ndarray::Array3;

#[cfg(feature = "simd")]
#[test]
fn large_images_do_not_accumulate_rounding_error() {
    use ndarray::s;
    for channels in [3, 4] {
        // Each channel cycles evenly through all 256 values in every column.
        let image = Array3::from_shape_fn((1024, 1024, channels), |(y, x, c)| {
            ((y * 37 + x * 19 + c * 71) % 256) as u8
        });
        for view in [
            image.view(),
            image.slice(s![.., ..;2, ..]),
            image.slice(s![..;-1, .., ..]),
        ] {
            for actual in [
                trainingsample::calculate_luminance_array(&view),
                trainingsample::calculate_luminance_optimized(&view).0,
                trainingsample::calculate_luminance_portable_simd(&view).0,
            ] {
                assert!((actual - 127.5).abs() < 1e-9, "{actual} != 127.5");
            }
        }
    }
}

#[test]
fn luminance_handles_channel_counts_and_simd_tails() {
    for channels in 1..=5 {
        for width in [1, 15, 16, 17, 31, 32, 33] {
            let image =
                Array3::from_shape_fn((3, width, channels), |(_, _, c)| [19, 83, 201, 0, 255][c]);
            let expected = match channels {
                1 => 19.0,
                2 => 51.0,
                _ => 0.299 * 19.0 + 0.587 * 83.0 + 0.114 * 201.0,
            };
            let actual = trainingsample::calculate_luminance_array(&image.view());
            assert!(
                (actual - expected).abs() < 1e-9,
                "{channels} channels, width {width}: {actual}"
            );
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn optimized_luminance_ignores_alpha() {
    for alpha in [0, 255] {
        let image = Array3::from_shape_fn((17, 19, 4), |(_, _, c)| [255, 0, 0, alpha][c]);
        let expected = 0.299 * 255.0;
        for actual in [
            trainingsample::calculate_luminance_optimized(&image.view()).0,
            trainingsample::calculate_luminance_optimized_sequential(&image.view()).0,
            trainingsample::detect_optimal_luminance_impl()(&image.view()).0,
        ] {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
    }
}

#[test]
fn empty_luminance_is_zero_for_every_entry_point() {
    for shape in [(0, 3, 3), (3, 0, 4), (2, 3, 0), (0, 0, 1)] {
        let image = Array3::<u8>::zeros(shape);
        assert_eq!(
            trainingsample::calculate_luminance_scalar(&image.view()),
            0.0
        );
        assert_eq!(
            trainingsample::calculate_luminance_with_metrics(&image.view()).0,
            0.0
        );
        #[cfg(feature = "simd")]
        {
            assert_eq!(
                trainingsample::calculate_luminance_portable_simd(&image.view()).0,
                0.0
            );
            assert_eq!(
                trainingsample::calculate_luminance_scalar_tracked(&image.view()).0,
                0.0
            );
        }
    }
}
