//! Comprehensive Performance Benchmarks for sklears
//!
//! This file contains criterion-based benchmarks to track performance
//! across different components of the sklears library.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array1, Array2, Array};
use sklears_core::traits::{Transform};
use sklears_utils::data_generation::{make_classification, make_regression, make_blobs};
use sklears_preprocessing::scaling::{StandardScaler, MinMaxScaler, RobustScaler};
use sklears_preprocessing::encoding::{OneHotEncoder, LabelEncoder};
use sklears_preprocessing::imputation::SimpleImputer;
use sklears_metrics::classification::{accuracy_score, precision_score};
use sklears_metrics::regression::{mean_squared_error, r2_score};
use sklears_model_selection::train_test_split::train_test_split;

fn benchmark_data_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_generation");
    
    for &n_samples in &[100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("make_classification", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    black_box(make_classification(
                        n_samples, 10, 3, None, None, 0.1, 1.0, Some(42)
                    ))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("make_regression", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    black_box(make_regression(
                        n_samples, 10, Some(8), 0.1, 0.0, Some(42)
                    ))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("make_blobs", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    black_box(make_blobs(
                        n_samples, 5, Some(3), 1.0, (-5.0, 5.0), Some(42)
                    ))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_preprocessing_scalers(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing_scalers");
    
    for &n_samples in &[100, 500, 1000, 5000] {
        let (X, _) = make_regression(n_samples, 10, Some(8), 0.1, 0.0, Some(42)).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("StandardScaler_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let scaler = StandardScaler::new();
                    black_box(scaler.fit(&X))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("MinMaxScaler_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let scaler = MinMaxScaler::new();
                    black_box(scaler.fit(&X))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("RobustScaler_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let scaler = RobustScaler::new();
                    black_box(scaler.fit(&X))
                })
            },
        );
        
        // Transform benchmarks
        let fitted_standard = StandardScaler::new().fit(&X).unwrap();
        let fitted_minmax = MinMaxScaler::new().fit(&X).unwrap();
        let fitted_robust = RobustScaler::new().fit(&X).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("StandardScaler_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_standard.transform(&X))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("MinMaxScaler_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_minmax.transform(&X))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("RobustScaler_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_robust.transform(&X))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_preprocessing_encoders(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing_encoders");
    
    for &n_samples in &[100, 500, 1000, 2000] {
        // Create categorical data
        let categorical_data: Vec<Vec<i32>> = (0..n_samples)
            .map(|i| vec![(i % 5) as i32, (i % 3) as i32])
            .collect();
        
        let label_data: Vec<i32> = (0..n_samples)
            .map(|i| (i % 10) as i32)
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("OneHotEncoder_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut encoder = OneHotEncoder::new();
                    black_box(encoder.fit(&categorical_data))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("LabelEncoder_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut encoder = LabelEncoder::new();
                    black_box(encoder.fit(&label_data))
                })
            },
        );
        
        // Transform benchmarks
        let mut fitted_onehot = OneHotEncoder::new();
        fitted_onehot.fit(&categorical_data).unwrap();
        
        let mut fitted_label = LabelEncoder::new();
        fitted_label.fit(&label_data).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("OneHotEncoder_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_onehot.transform(&categorical_data))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("LabelEncoder_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_label.transform(&label_data))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_imputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("imputation");
    
    for &n_samples in &[100, 500, 1000] {
        // Create data with missing values
        let mut data = Array2::from_shape_fn((n_samples, 5), |(i, j)| {
            if (i + j) % 7 == 0 {
                f64::NAN
            } else {
                (i as f64 + j as f64) * 0.5
            }
        });
        
        group.bench_with_input(
            BenchmarkId::new("SimpleImputer_mean_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let imputer = SimpleImputer::new().strategy("mean");
                    black_box(imputer.fit(&data))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("SimpleImputer_median_fit", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let imputer = SimpleImputer::new().strategy("median");
                    black_box(imputer.fit(&data))
                })
            },
        );
        
        // Transform benchmarks
        let fitted_mean = SimpleImputer::new().strategy("mean").fit(&data).unwrap();
        let fitted_median = SimpleImputer::new().strategy("median").fit(&data).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("SimpleImputer_mean_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_mean.transform(&data))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("SimpleImputer_median_transform", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(fitted_median.transform(&data))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");
    
    for &n_samples in &[100, 500, 1000, 5000] {
        let y_true: Array1<i32> = Array1::from_vec(
            (0..n_samples).map(|i| (i % 3) as i32).collect()
        );
        let y_pred: Array1<i32> = Array1::from_vec(
            (0..n_samples).map(|i| ((i + 1) % 3) as i32).collect()
        );
        
        let y_true_reg: Array1<f64> = Array1::from_vec(
            (0..n_samples).map(|i| i as f64 * 0.1).collect()
        );
        let y_pred_reg: Array1<f64> = Array1::from_vec(
            (0..n_samples).map(|i| i as f64 * 0.1 + 0.05).collect()
        );
        
        group.bench_with_input(
            BenchmarkId::new("accuracy_score", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(accuracy_score(&y_true, &y_pred))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("precision_score", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(precision_score(&y_true, &y_pred))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mean_squared_error", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(mean_squared_error(&y_true_reg, &y_pred_reg))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("r2_score", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(r2_score(&y_true_reg, &y_pred_reg))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_train_test_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_test_split");
    
    for &n_samples in &[100, 500, 1000, 5000] {
        let (X, y) = make_classification(n_samples, 10, 3, None, None, 0.1, 1.0, Some(42)).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("train_test_split", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    black_box(train_test_split(&X, &y, 0.3, Some(42)))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_complete_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_pipeline");
    
    for &n_samples in &[100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("classification_pipeline", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    // Generate data
                    let (X, y) = black_box(make_classification(
                        n_samples, 10, 3, None, None, 0.1, 1.0, Some(42)
                    ).unwrap());
                    
                    // Split data
                    let (X_train, X_test, y_train, y_test) = black_box(
                        train_test_split(&X, &y, 0.3, Some(42)).unwrap()
                    );
                    
                    // Scale features
                    let scaler = StandardScaler::new();
                    let fitted_scaler = black_box(scaler.fit(&X_train).unwrap());
                    let X_train_scaled = black_box(fitted_scaler.transform(&X_train).unwrap());
                    let X_test_scaled = black_box(fitted_scaler.transform(&X_test).unwrap());
                    
                    // Simple prediction (mock classifier)
                    let predictions: Array1<i32> = Array1::from_elem(y_test.len(), 0);
                    
                    // Evaluate
                    let accuracy = black_box(accuracy_score(&y_test, &predictions));
                    
                    accuracy
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("regression_pipeline", n_samples),
            &n_samples,
            |b, &n_samples| {
                b.iter(|| {
                    // Generate data
                    let (X, y) = black_box(make_regression(
                        n_samples, 10, Some(8), 0.1, 0.0, Some(42)
                    ).unwrap());
                    
                    // Split data
                    let (X_train, X_test, y_train, y_test) = black_box(
                        train_test_split(&X, &y, 0.3, Some(42)).unwrap()
                    );
                    
                    // Scale features
                    let scaler = StandardScaler::new();
                    let fitted_scaler = black_box(scaler.fit(&X_train).unwrap());
                    let X_train_scaled = black_box(fitted_scaler.transform(&X_train).unwrap());
                    let X_test_scaled = black_box(fitted_scaler.transform(&X_test).unwrap());
                    
                    // Simple prediction (mock regressor - predict mean)
                    let mean_target = y_train.mean().unwrap();
                    let predictions: Array1<f64> = Array1::from_elem(y_test.len(), mean_target);
                    
                    // Evaluate
                    let mse = black_box(mean_squared_error(&y_test, &predictions));
                    let r2 = black_box(r2_score(&y_test, &predictions));
                    
                    (mse, r2)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_data_generation,
    benchmark_preprocessing_scalers,
    benchmark_preprocessing_encoders,
    benchmark_imputation,
    benchmark_metrics,
    benchmark_train_test_split,
    benchmark_complete_pipeline
);

criterion_main!(benches);