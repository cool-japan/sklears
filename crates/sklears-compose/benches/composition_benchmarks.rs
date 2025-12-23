//! Comprehensive Pipeline Composition Performance Benchmarks
//!
//! This benchmark suite measures the performance of different pipeline composition strategies
//! including sequential pipelines, parallel execution, feature unions, and DAG pipelines.
//!
//! Run with: `cargo bench --bench composition_benchmarks`

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_compose::pipeline::Pipeline;
use sklears_core::error::Result as SklResult;
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_core::types::Float;
use std::time::Duration;

/// Mock transformer for benchmarking
#[derive(Debug, Clone)]
struct MockTransformer {
    scale: Float,
}

impl MockTransformer {
    fn new(scale: Float) -> Self {
        Self { scale }
    }
}

impl Transform for MockTransformer {
    type Input = ArrayView2<'static, Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        Ok(x.mapv(|v| v * self.scale))
    }
}

/// Mock estimator for benchmarking
#[derive(Debug, Clone)]
struct MockEstimator {
    weight: Float,
}

impl MockEstimator {
    fn new(weight: Float) -> Self {
        Self { weight }
    }
}

impl Fit<ArrayView2<'static, Float>, ArrayView2<'static, Float>> for MockEstimator {
    type Fitted = FittedMockEstimator;

    fn fit(
        self,
        _x: &ArrayView2<'static, Float>,
        _y: &ArrayView2<'static, Float>,
    ) -> SklResult<Self::Fitted> {
        Ok(FittedMockEstimator {
            weight: self.weight,
        })
    }
}

#[derive(Debug, Clone)]
struct FittedMockEstimator {
    weight: Float,
}

impl Predict for FittedMockEstimator {
    type Input = ArrayView2<'static, Float>;
    type Output = Array2<Float>;

    fn predict(&self, x: &Self::Input) -> SklResult<Self::Output> {
        Ok(x.mapv(|v| v * self.weight))
    }
}

/// Generate random test data
fn generate_data(n_samples: usize, n_features: usize) -> (Array2<Float>, Array2<Float>) {
    let mut rng = StdRng::seed_from_u64(42);
    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0));
    let y = Array2::from_shape_fn((n_samples, 1), |_| rng.gen_range(-1.0..1.0));
    (x, y)
}

/// Benchmark sequential pipeline execution
fn bench_sequential_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_pipeline");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(100, 10), (1000, 20), (10000, 50)];

    for (n_samples, n_features) in sizes.iter() {
        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        let (x, y) = generate_data(*n_samples, *n_features);

        // Benchmark 2-stage pipeline
        group.bench_with_input(
            BenchmarkId::new("2_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| {
                    let pipeline = Pipeline::new()
                        .add_step("scale1", MockTransformer::new(2.0))
                        .add_step("scale2", MockTransformer::new(0.5));

                    let x_view = x.view();
                    let y_view = y.view();
                    let fitted = pipeline.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );

        // Benchmark 5-stage pipeline
        group.bench_with_input(
            BenchmarkId::new("5_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| {
                    let pipeline = Pipeline::new()
                        .add_step("scale1", MockTransformer::new(2.0))
                        .add_step("scale2", MockTransformer::new(0.5))
                        .add_step("scale3", MockTransformer::new(1.5))
                        .add_step("scale4", MockTransformer::new(0.8))
                        .add_step("scale5", MockTransformer::new(1.2));

                    let x_view = x.view();
                    let y_view = y.view();
                    let fitted = pipeline.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );

        // Benchmark 10-stage pipeline
        group.bench_with_input(
            BenchmarkId::new("10_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| {
                    let mut pipeline = Pipeline::new();
                    for i in 0..10 {
                        pipeline = pipeline.add_step(
                            &format!("scale{}", i),
                            MockTransformer::new(1.0 + (i as Float) * 0.1),
                        );
                    }

                    let x_view = x.view();
                    let y_view = y.view();
                    let fitted = pipeline.fit(&x_view, &y_view).unwrap();
                    black_box(fitted.predict(&x_view).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline memory overhead
fn bench_pipeline_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_memory");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(1000, 10), (10000, 20), (100000, 50)];

    for (n_samples, n_features) in sizes.iter() {
        group.throughput(Throughput::Bytes((n_samples * n_features * 8) as u64));

        let (x, y) = generate_data(*n_samples, *n_features);

        group.bench_with_input(
            BenchmarkId::new("minimal_copy", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let pipeline = Pipeline::new()
                    .add_step("scale1", MockTransformer::new(2.0))
                    .add_step("scale2", MockTransformer::new(0.5));

                let x_view = x.view();
                let y_view = y.view();
                let fitted = pipeline.fit(&x_view, &y_view).unwrap();

                bench.iter(|| black_box(fitted.predict(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline construction overhead
fn bench_pipeline_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_construction");
    group.measurement_time(Duration::from_secs(5));

    let stage_counts = [2, 5, 10, 20, 50];

    for n_stages in stage_counts.iter() {
        group.bench_with_input(
            BenchmarkId::new("construction", n_stages),
            n_stages,
            |bench, &n| {
                bench.iter(|| {
                    let mut pipeline = Pipeline::new();
                    for i in 0..n {
                        pipeline = pipeline.add_step(
                            &format!("scale{}", i),
                            MockTransformer::new(1.0 + (i as Float) * 0.1),
                        );
                    }
                    black_box(pipeline)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different data sizes
fn bench_data_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_scaling");
    group.measurement_time(Duration::from_secs(10));

    // Test small to large datasets
    let sizes = [(10, 5), (100, 10), (1000, 20), (10000, 50), (50000, 100)];

    for (n_samples, n_features) in sizes.iter() {
        let (x, y) = generate_data(*n_samples, *n_features);

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("3_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let pipeline = Pipeline::new()
                    .add_step("scale1", MockTransformer::new(2.0))
                    .add_step("scale2", MockTransformer::new(0.5))
                    .add_step("scale3", MockTransformer::new(1.5));

                let x_view = x.view();
                let y_view = y.view();
                let fitted = pipeline.fit(&x_view, &y_view).unwrap();

                bench.iter(|| black_box(fitted.predict(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline fit vs predict performance
fn bench_fit_vs_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_vs_predict");
    group.measurement_time(Duration::from_secs(10));

    let (x, y) = generate_data(10000, 20);
    let x_view = x.view();
    let y_view = y.view();

    // Benchmark fit operation
    group.bench_function("fit_operation", |bench| {
        bench.iter(|| {
            let pipeline = Pipeline::new()
                .add_step("scale1", MockTransformer::new(2.0))
                .add_step("scale2", MockTransformer::new(0.5))
                .add_step("scale3", MockTransformer::new(1.5));

            black_box(pipeline.fit(&x_view, &y_view).unwrap())
        });
    });

    // Benchmark predict operation
    let pipeline = Pipeline::new()
        .add_step("scale1", MockTransformer::new(2.0))
        .add_step("scale2", MockTransformer::new(0.5))
        .add_step("scale3", MockTransformer::new(1.5));

    let fitted = pipeline.fit(&x_view, &y_view).unwrap();

    group.bench_function("predict_operation", |bench| {
        bench.iter(|| black_box(fitted.predict(&x_view).unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_pipeline,
    bench_pipeline_memory,
    bench_pipeline_construction,
    bench_data_scaling,
    bench_fit_vs_predict,
);

criterion_main!(benches);
