//! Comprehensive Pipeline Composition Performance Benchmarks
//!
//! This benchmark suite measures the performance of different pipeline composition strategies
//! including sequential pipelines, parallel execution, feature unions, and DAG pipelines.
//!
//! Run with: `cargo bench --bench composition_benchmarks`

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_compose::{Pipeline, PipelineStep};
use sklears_core::error::Result as SklResult;
use sklears_core::traits::Fit;
use sklears_core::types::Float;
use std::hint::black_box;
use std::time::Duration;

/// Mock transformer implementing PipelineStep
#[derive(Debug, Clone)]
struct MockTransformer {
    scale: Float,
}

impl MockTransformer {
    fn new(scale: Float) -> Self {
        Self { scale }
    }
}

impl PipelineStep for MockTransformer {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        Ok(x.mapv(|v| v * self.scale))
    }

    fn fit(
        &mut self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<()> {
        Ok(())
    }

    fn clone_step(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

/// Build a Pipeline from (name, scale) pairs
fn build_pipeline(steps: &[(&str, Float)]) -> Pipeline {
    let mut pipeline = Pipeline::new();
    for (name, scale) in steps {
        pipeline.add_step(name.to_string(), Box::new(MockTransformer::new(*scale)));
    }
    pipeline
}

/// Generate random test data
fn generate_data(n_samples: usize, n_features: usize) -> Array2<Float> {
    let mut rng = StdRng::seed_from_u64(42);
    Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.random_range(-1.0_f64..1.0_f64)
    })
}

/// Benchmark sequential pipeline execution
fn bench_sequential_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_pipeline");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(100usize, 10usize), (1000, 20), (10000, 50)];

    for (n_samples, n_features) in sizes.iter() {
        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();
        let y_opt: Option<&ArrayView1<'_, Float>> = None;

        // Benchmark 2-stage pipeline - pre-fit, then measure transform
        let fitted_2 = build_pipeline(&[("scale1", 2.0), ("scale2", 0.5)])
            .fit(&x_view, &y_opt)
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("2_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted_2.transform(&x_view).unwrap()));
            },
        );

        // Benchmark 5-stage pipeline
        let fitted_5 = build_pipeline(&[
            ("scale1", 2.0),
            ("scale2", 0.5),
            ("scale3", 1.5),
            ("scale4", 0.8),
            ("scale5", 1.2),
        ])
        .fit(&x_view, &y_opt)
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("5_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted_5.transform(&x_view).unwrap()));
            },
        );

        // Benchmark 10-stage pipeline
        let steps_10: Vec<(String, Float)> = (0..10)
            .map(|i| (format!("scale{}", i), 1.0 + i as Float * 0.1))
            .collect();
        let step_refs: Vec<(&str, Float)> =
            steps_10.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        let fitted_10 = build_pipeline(&step_refs).fit(&x_view, &y_opt).unwrap();

        group.bench_with_input(
            BenchmarkId::new("10_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted_10.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline memory overhead
fn bench_pipeline_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_memory");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(1000usize, 10usize), (10000, 20), (100000, 50)];

    for (n_samples, n_features) in sizes.iter() {
        group.throughput(Throughput::Bytes((n_samples * n_features * 8) as u64));

        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();
        let y_opt: Option<&ArrayView1<'_, Float>> = None;

        let fitted = build_pipeline(&[("scale1", 2.0), ("scale2", 0.5)])
            .fit(&x_view, &y_opt)
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("minimal_copy", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline construction overhead
fn bench_pipeline_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_construction");
    group.measurement_time(Duration::from_secs(5));

    let stage_counts = [2usize, 5, 10, 20, 50];

    for n_stages in stage_counts.iter() {
        group.bench_with_input(
            BenchmarkId::new("construction", n_stages),
            n_stages,
            |bench, &n| {
                bench.iter(|| {
                    let mut pipeline = Pipeline::new();
                    for i in 0..n {
                        pipeline.add_step(
                            format!("scale{}", i),
                            Box::new(MockTransformer::new(1.0 + i as Float * 0.1)),
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

    let sizes = [
        (10usize, 5usize),
        (100, 10),
        (1000, 20),
        (10000, 50),
        (50000, 100),
    ];

    for (n_samples, n_features) in sizes.iter() {
        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();
        let y_opt: Option<&ArrayView1<'_, Float>> = None;

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        let fitted = build_pipeline(&[("scale1", 2.0), ("scale2", 0.5), ("scale3", 1.5)])
            .fit(&x_view, &y_opt)
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("3_stage", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark pipeline fit vs transform performance
fn bench_fit_vs_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_vs_transform");
    group.measurement_time(Duration::from_secs(10));

    let x = generate_data(10000, 20);
    let x_view = x.view();
    let y_opt: Option<&ArrayView1<'_, Float>> = None;

    // Benchmark fit operation
    group.bench_function("fit_operation", |bench| {
        bench.iter(|| {
            let pipeline = build_pipeline(&[("scale1", 2.0), ("scale2", 0.5), ("scale3", 1.5)]);
            black_box(pipeline.fit(&x_view, &y_opt).unwrap())
        });
    });

    // Benchmark transform operation (pre-fitted)
    let fitted = build_pipeline(&[("scale1", 2.0), ("scale2", 0.5), ("scale3", 1.5)])
        .fit(&x_view, &y_opt)
        .unwrap();

    group.bench_function("transform_operation", |bench| {
        bench.iter(|| black_box(fitted.transform(&x_view).unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_pipeline,
    bench_pipeline_memory,
    bench_pipeline_construction,
    bench_data_scaling,
    bench_fit_vs_transform,
);

criterion_main!(benches);
