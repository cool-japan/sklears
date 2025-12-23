//! Feature Transformation Performance Benchmarks
//!
//! This benchmark suite measures the performance of feature transformation operations
//! including FeatureUnion, ColumnTransformer, and various preprocessing steps.
//!
//! Run with: `cargo bench --bench feature_transformation_benchmarks`

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_compose::feature_union::FeatureUnion;
use sklears_core::error::Result as SklResult;
use sklears_core::traits::Transform;
use sklears_core::types::Float;
use std::time::Duration;

/// Mock feature transformer for benchmarking
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

impl Transform for MockFeatureTransformer {
    type Input = ArrayView2<'static, Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        match &self.operation {
            TransformOperation::Scale(factor) => Ok(x.mapv(|v| v * factor)),
            TransformOperation::Square => Ok(x.mapv(|v| v * v)),
            TransformOperation::Log => Ok(x.mapv(|v| (v.abs() + 1.0).ln())),
            TransformOperation::Polynomial(degree) => {
                let mut result = x.to_owned();
                for _ in 1..*degree {
                    result = &result * x;
                }
                Ok(result)
            }
        }
    }
}

/// Generate random test data
fn generate_data(n_samples: usize, n_features: usize) -> Array2<Float> {
    let mut rng = StdRng::seed_from_u64(42);
    Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0))
}

/// Benchmark FeatureUnion with different numbers of transformers
fn bench_feature_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_union");
    group.measurement_time(Duration::from_secs(10));

    let (n_samples, n_features) = (10000, 20);
    let x = generate_data(n_samples, n_features);
    let x_view = x.view();

    let transformer_counts = [2, 5, 10, 20];

    for n_transformers in transformer_counts.iter() {
        group.throughput(Throughput::Elements(*n_transformers as u64));

        group.bench_with_input(
            BenchmarkId::new("parallel_transforms", n_transformers),
            n_transformers,
            |bench, &n| {
                let mut transformers = Vec::new();
                for i in 0..n {
                    transformers.push((
                        format!("trans_{}", i),
                        MockFeatureTransformer::new(TransformOperation::Scale(
                            1.0 + i as Float * 0.1,
                        )),
                    ));
                }

                bench.iter(|| {
                    let union = FeatureUnion::new(transformers.clone(), None).unwrap();
                    black_box(union.transform(&x_view).unwrap())
                });
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
    for degree in [2, 3, 4].iter() {
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

    let sizes = [(100, 10), (1000, 20), (10000, 50), (50000, 100)];

    for (n_samples, n_features) in sizes.iter() {
        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("5_transformers", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let transformers = vec![
                    (
                        "scale1".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
                    ),
                    (
                        "scale2".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Scale(0.5)),
                    ),
                    (
                        "square".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Square),
                    ),
                    (
                        "log".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Log),
                    ),
                    (
                        "poly".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Polynomial(2)),
                    ),
                ];

                bench.iter(|| {
                    let union = FeatureUnion::new(transformers.clone(), None).unwrap();
                    black_box(union.transform(&x_view).unwrap())
                });
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

    let transformers = vec![
        (
            "scale1".to_string(),
            MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
        ),
        (
            "scale2".to_string(),
            MockFeatureTransformer::new(TransformOperation::Scale(0.5)),
        ),
        (
            "square".to_string(),
            MockFeatureTransformer::new(TransformOperation::Square),
        ),
        (
            "log".to_string(),
            MockFeatureTransformer::new(TransformOperation::Log),
        ),
        (
            "poly".to_string(),
            MockFeatureTransformer::new(TransformOperation::Polynomial(2)),
        ),
    ];

    // Benchmark unweighted
    group.bench_function("unweighted", |bench| {
        let union = FeatureUnion::new(transformers.clone(), None).unwrap();
        bench.iter(|| black_box(union.transform(&x_view).unwrap()));
    });

    // Benchmark weighted
    group.bench_function("weighted", |bench| {
        let weights = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let union = FeatureUnion::new(transformers.clone(), Some(weights)).unwrap();
        bench.iter(|| black_box(union.transform(&x_view).unwrap()));
    });

    group.finish();
}

/// Benchmark feature dimensionality expansion
fn bench_feature_dimensionality(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_dimensionality");
    group.measurement_time(Duration::from_secs(10));

    let n_samples = 10000;
    let feature_counts = [5, 10, 20, 50, 100];

    for n_features in feature_counts.iter() {
        let x = generate_data(n_samples, *n_features);
        let x_view = x.view();

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("3_transformers", n_features),
            n_features,
            |bench, _| {
                let transformers = vec![
                    (
                        "scale".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
                    ),
                    (
                        "square".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Square),
                    ),
                    (
                        "log".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Log),
                    ),
                ];

                bench.iter(|| {
                    let union = FeatureUnion::new(transformers.clone(), None).unwrap();
                    black_box(union.transform(&x_view).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency of transformations
fn bench_transformation_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformation_memory");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [(1000, 10), (10000, 20), (100000, 50)];

    for (n_samples, n_features) in sizes.iter() {
        let x = generate_data(*n_samples, *n_features);
        let x_view = x.view();

        group.throughput(Throughput::Bytes((n_samples * n_features * 8) as u64));

        group.bench_with_input(
            BenchmarkId::new("single_transform", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let transformer = MockFeatureTransformer::new(TransformOperation::Scale(2.0));
                bench.iter(|| black_box(transformer.transform(&x_view).unwrap()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("union_transform", format!("{}x{}", n_samples, n_features)),
            &(*n_samples, *n_features),
            |bench, _| {
                let transformers = vec![
                    (
                        "t1".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Scale(2.0)),
                    ),
                    (
                        "t2".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Square),
                    ),
                    (
                        "t3".to_string(),
                        MockFeatureTransformer::new(TransformOperation::Log),
                    ),
                ];

                bench.iter(|| {
                    let union = FeatureUnion::new(transformers.clone(), None).unwrap();
                    black_box(union.transform(&x_view).unwrap())
                });
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
