//! Feature Transformation Performance Benchmarks
//!
//! This benchmark suite measures the performance of feature transformation operations
//! including FeatureUnion and various preprocessing steps.
//!
//! Run with: `cargo bench --bench feature_transformation_benchmarks`

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_compose::{FeatureUnion, PipelineStep};
use sklears_core::error::Result as SklResult;
use sklears_core::traits::{Fit, Untrained};
use sklears_core::types::Float;
use std::hint::black_box;
use std::time::Duration;

/// Mock feature transformer for benchmarking that implements PipelineStep
#[derive(Debug, Clone)]
struct MockFeatureTransformer {
    operation: TransformOperation,
}

#[derive(Debug, Clone)]
enum TransformOperation {
    Scale(Float),
    Square,
    Log,
    Polynomial(usize),
}

impl MockFeatureTransformer {
    fn new(operation: TransformOperation) -> Self {
        Self { operation }
    }
}

impl PipelineStep for MockFeatureTransformer {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let result = match &self.operation {
            TransformOperation::Scale(factor) => x.mapv(|v| v * factor),
            TransformOperation::Square => x.mapv(|v| v * v),
            TransformOperation::Log => x.mapv(|v: f64| (v.abs() + 1.0).ln()),
            TransformOperation::Polynomial(degree) => {
                let mut result = x.to_owned();
                for _ in 1..*degree {
                    result *= x;
                }
                result
            }
        };
        Ok(result)
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

/// Build a FeatureUnion from a list of (name, transformer) pairs
fn build_union(transformers: Vec<(&str, MockFeatureTransformer)>) -> FeatureUnion<Untrained> {
    let mut union = FeatureUnion::new();
    for (name, t) in transformers {
        union = union.transformer(name, Box::new(t));
    }
    union
}

/// Generate random test data
fn generate_data(n_samples: usize, n_features: usize) -> Array2<Float> {
    let mut rng = StdRng::seed_from_u64(42);
    Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.random_range(-1.0_f64..1.0_f64)
    })
}

/// Benchmark FeatureUnion with different numbers of transformers
fn bench_feature_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_union");
    group.measurement_time(Duration::from_secs(10));

    let (n_samples, n_features) = (10000, 20);
    let x = generate_data(n_samples, n_features);
    let x_view = x.view();
    let y_opt: Option<&ArrayView1<'_, Float>> = None;

    let transformer_counts = [2usize, 5, 10, 20];

    for n_transformers in transformer_counts.iter() {
        group.throughput(Throughput::Elements(*n_transformers as u64));

        // Pre-fit the union before the benchmark loop
        let transformers: Vec<(&str, MockFeatureTransformer)> = (0..*n_transformers)
            .map(|i| {
                let name = Box::leak(format!("trans_{}", i).into_boxed_str()) as &str;
                (
                    name,
                    MockFeatureTransformer::new(TransformOperation::Scale(1.0 + i as Float * 0.1)),
                )
            })
            .collect();
        let fitted = build_union(transformers).fit(&x_view, &y_opt).unwrap();

        group.bench_with_input(
            BenchmarkId::new("parallel_transforms", n_transformers),
            n_transformers,
            |bench, _| {
                bench.iter(|| black_box(fitted.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark different transformation operations
fn bench_transformation_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformation_types");
    group.measurement_time(Duration::from_secs(10));

    let (n_samples, n_features) = (10000, 20);
    let x = generate_data(n_samples, n_features);
    let x_view = x.view();

    // Benchmark scaling
    group.bench_function("scale_operation", |bench| {
        let transformer = MockFeatureTransformer::new(TransformOperation::Scale(2.5));
        bench.iter(|| black_box(transformer.transform(&x_view).unwrap()));
    });

    // Benchmark squaring
    group.bench_function("square_operation", |bench| {
        let transformer = MockFeatureTransformer::new(TransformOperation::Square);
        bench.iter(|| black_box(transformer.transform(&x_view).unwrap()));
    });

    // Benchmark log transform
    group.bench_function("log_operation", |bench| {
        let transformer = MockFeatureTransformer::new(TransformOperation::Log);
        bench.iter(|| black_box(transformer.transform(&x_view).unwrap()));
    });

    // Benchmark polynomial features
    for degree in [2usize, 3, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("polynomial", degree),
            degree,
            |bench, &d| {
                let transformer = MockFeatureTransformer::new(TransformOperation::Polynomial(d));
                bench.iter(|| black_box(transformer.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark feature union with different data sizes
fn bench_feature_union_data_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_union_data_scaling");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(100usize, 10usize), (1000, 20), (10000, 50), (50000, 100)];

    for (n_samples, n_features) in sizes.iter() {
        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();
        let y_opt: Option<&ArrayView1<'_, Float>> = None;

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        let fitted = build_union(vec![
            (
                "scale1",
                MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
            ),
            (
                "scale2",
                MockFeatureTransformer::new(TransformOperation::Scale(0.5)),
            ),
            (
                "square",
                MockFeatureTransformer::new(TransformOperation::Square),
            ),
            ("log", MockFeatureTransformer::new(TransformOperation::Log)),
            (
                "poly",
                MockFeatureTransformer::new(TransformOperation::Polynomial(2)),
            ),
        ])
        .fit(&x_view, &y_opt)
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("5_transformers", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark weighted vs unweighted feature union
fn bench_feature_union_weighting(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_union_weighting");
    group.measurement_time(Duration::from_secs(10));

    let (n_samples, n_features) = (10000, 20);
    let x = generate_data(n_samples, n_features);
    let x_view = x.view();
    let y_opt: Option<&ArrayView1<'_, Float>> = None;

    // Benchmark unweighted
    let fitted_unweighted = build_union(vec![
        (
            "scale1",
            MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
        ),
        (
            "scale2",
            MockFeatureTransformer::new(TransformOperation::Scale(0.5)),
        ),
        (
            "square",
            MockFeatureTransformer::new(TransformOperation::Square),
        ),
        ("log", MockFeatureTransformer::new(TransformOperation::Log)),
        (
            "poly",
            MockFeatureTransformer::new(TransformOperation::Polynomial(2)),
        ),
    ])
    .fit(&x_view, &y_opt)
    .unwrap();

    group.bench_function("unweighted", |bench| {
        bench.iter(|| black_box(fitted_unweighted.transform(&x_view).unwrap()));
    });

    // Benchmark with transformer weights
    use std::collections::HashMap;
    let mut weights = HashMap::new();
    weights.insert("scale1".to_string(), 1.0_f64);
    weights.insert("scale2".to_string(), 0.8_f64);
    weights.insert("square".to_string(), 0.6_f64);
    weights.insert("log".to_string(), 0.4_f64);
    weights.insert("poly".to_string(), 0.2_f64);

    let mut union_weighted = build_union(vec![
        (
            "scale1",
            MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
        ),
        (
            "scale2",
            MockFeatureTransformer::new(TransformOperation::Scale(0.5)),
        ),
        (
            "square",
            MockFeatureTransformer::new(TransformOperation::Square),
        ),
        ("log", MockFeatureTransformer::new(TransformOperation::Log)),
        (
            "poly",
            MockFeatureTransformer::new(TransformOperation::Polynomial(2)),
        ),
    ]);
    union_weighted = union_weighted.transformer_weights(weights);
    let fitted_weighted = union_weighted.fit(&x_view, &y_opt).unwrap();

    group.bench_function("weighted", |bench| {
        bench.iter(|| black_box(fitted_weighted.transform(&x_view).unwrap()));
    });

    group.finish();
}

/// Benchmark feature dimensionality expansion
fn bench_feature_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_dimensionality");
    group.measurement_time(Duration::from_secs(10));

    let n_samples = 10000;
    let feature_counts = [5usize, 10, 20, 50, 100];

    for n_features in feature_counts.iter() {
        let x = generate_data(n_samples, *n_features);
        let x_view = x.view();
        let y_opt: Option<&ArrayView1<'_, Float>> = None;

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        let fitted = build_union(vec![
            (
                "scale",
                MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
            ),
            (
                "square",
                MockFeatureTransformer::new(TransformOperation::Square),
            ),
            ("log", MockFeatureTransformer::new(TransformOperation::Log)),
        ])
        .fit(&x_view, &y_opt)
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("3_transformers", n_features),
            n_features,
            |bench, _| {
                bench.iter(|| black_box(fitted.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency of transformations
fn bench_transformation_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformation_memory");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(1000usize, 10usize), (10000, 20), (100000, 50)];

    for (n_samples, n_features) in sizes.iter() {
        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();
        let y_opt: Option<&ArrayView1<'_, Float>> = None;

        group.throughput(Throughput::Bytes((n_samples * n_features * 8) as u64));

        group.bench_with_input(
            BenchmarkId::new("single_transform", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let transformer = MockFeatureTransformer::new(TransformOperation::Scale(2.0));
                bench.iter(|| black_box(transformer.transform(&x_view).unwrap()));
            },
        );

        let fitted_union = build_union(vec![
            (
                "t1",
                MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
            ),
            (
                "t2",
                MockFeatureTransformer::new(TransformOperation::Square),
            ),
            ("t3", MockFeatureTransformer::new(TransformOperation::Log)),
        ])
        .fit(&x_view, &y_opt)
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("union_transform", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                bench.iter(|| black_box(fitted_union.transform(&x_view).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_feature_union,
    bench_transformation_types,
    bench_feature_union_data_scaling,
    bench_feature_union_weighting,
    bench_feature_dimensionality,
    bench_transformation_memory,
);

criterion_main!(benches);
