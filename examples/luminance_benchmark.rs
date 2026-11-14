//! Run with `cargo run --release --example luminance_benchmark`.
use ndarray::{s, Array3, ArrayView3};
use std::{hint::black_box, time::Instant};
use trainingsample::calculate_luminance_array;

fn measure(name: &str, image: &ArrayView3<u8>, operation: impl Fn(&ArrayView3<u8>) -> f64) {
    for _ in 0..10 {
        black_box(operation(black_box(image)));
    }
    let mut samples = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        for _ in 0..20 {
            black_box(operation(black_box(image)));
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{name},{:.3},{:.3},{:.3},{:.9}",
        samples[5],
        samples[0],
        samples[10],
        operation(image)
    );
}

fn main() {
    eprintln!(
        "target={}-{}, simd={}, image=1024x1024 uint8, warmups=10, rounds=11, calls_per_round=20",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cfg!(feature = "simd")
    );
    println!("case,median_us,min_us,max_us,luminance");
    for channels in [3, 4] {
        let image = Array3::from_shape_fn((1024, 1024, channels), |(y, x, c)| {
            ((y * 37 + x * 19 + c * 71) % 256) as u8
        });
        for (layout, view) in [
            ("contiguous", image.view()),
            ("strided", image.slice(s![.., ..;2, ..])),
        ] {
            let prefix = format!("{channels}ch_{layout}");
            measure(
                &format!("{prefix}_default"),
                &view,
                calculate_luminance_array,
            );
            #[cfg(feature = "simd")]
            {
                measure(&format!("{prefix}_optimized"), &view, |v| {
                    trainingsample::calculate_luminance_optimized(v).0
                });
                measure(&format!("{prefix}_portable"), &view, |v| {
                    trainingsample::calculate_luminance_portable_simd(v).0
                });
            }
        }
    }
}
