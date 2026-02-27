//! Benchmark configuration and utilities for continuous benchmarking

use criterion::Criterion;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;
use std::hint::black_box;

/// Generate synthetic data for benchmarking
pub fn generate_benchmark_data(
    n_samples: usize,
    n_features: usize,
) -> (Array2<Float>, Array1<Float>) {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::Rng;
    use scirs2_core::random::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);

    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random::<Float>() * 10.0);
    let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as Float);

    (x, y)
}

/// Generate classification data with multiple classes
pub fn generate_multiclass_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<Float>, Array1<i32>) {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::Rng;
    use scirs2_core::random::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);

    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random::<Float>() * 10.0);
    let y = Array1::from_shape_fn(n_samples, |i| (i % n_classes) as i32);

    (x, y)
}

/// Benchmark group configuration
pub struct BenchmarkConfig {
    pub small_size: usize,
    pub medium_size: usize,
    pub large_size: usize,
    pub n_features: usize,
    pub measurement_time: u64,
    pub sample_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            small_size: 100,
            medium_size: 1000,
            large_size: 10000,
            n_features: 10,
            measurement_time: 10, // seconds
            sample_size: 100,
        }
    }
}

/// Configure criterion for consistent benchmarking
pub fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(std::time::Duration::from_secs(10))
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(3))
        .with_plots()
}

/// Macro to generate benchmark groups
#[macro_export]
macro_rules! benchmark_algorithm {
    ($name:expr, $algorithm:ty, $fit_method:expr, $predict_method:expr) => {
        pub fn $name(c: &mut Criterion) {
            let config = BenchmarkConfig::default();
            let mut group = c.benchmark_group(stringify!($algorithm));

            for size in &[config.small_size, config.medium_size, config.large_size] {
                let (x, y) = generate_benchmark_data(*size, config.n_features);

                group.bench_with_input(BenchmarkId::new("fit", size), &(&x, &y), |b, (x, y)| {
                    b.iter(|| {
                        let model = $fit_method(black_box(x), black_box(y));
                        black_box(model)
                    });
                });

                // Benchmark prediction if model supports it
                let model = $fit_method(&x, &y).unwrap();
                group.bench_with_input(BenchmarkId::new("predict", size), &x, |b, x| {
                    b.iter(|| {
                        let predictions = $predict_method(&model, black_box(x));
                        black_box(predictions)
                    });
                });
            }

            group.finish();
        }
    };
}

/// Memory usage tracking utility
pub fn track_memory_usage<F: FnOnce()>(name: &str, f: F) {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let before = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(0);

        f();

        let after = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(0);

        println!("{} memory usage: {} KB", name, after - before);
    }

    #[cfg(not(target_os = "linux"))]
    {
        f();
        println!("{} memory tracking not available on this platform", name);
    }
}

/// Utility to benchmark throughput
pub fn benchmark_throughput<F: Fn() -> R, R>(name: &str, iterations: usize, f: F) {
    use std::time::Instant;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f());
    }
    let duration = start.elapsed();

    let throughput = iterations as f64 / duration.as_secs_f64();
    println!("{}: {:.2} ops/sec", name, throughput);
}
