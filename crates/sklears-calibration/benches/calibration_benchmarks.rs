//! Calibration Performance Benchmarks
//!
//! This module contains benchmarks comparing our calibration implementations
//! against reference performance expectations and theoretical bounds.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_calibration::{
    bayesian::{BayesianModelAveragingCalibrator, VariationalInferenceCalibrator},
    beta::BetaCalibrator,
    binary::SigmoidCalibrator,
    histogram::HistogramBinningCalibrator,
    isotonic::IsotonicCalibrator,
    kde::KDECalibrator,
    local::LocalKNNCalibrator,
    multi_modal::{
        EnsembleCombination, FusionStrategy, HeterogeneousEnsembleCalibrator, MultiModalCalibrator,
    },
    neural_calibration::NeuralCalibrationLayer,
    temperature::TemperatureScalingCalibrator,
    CalibrationEstimator,
};
use sklears_core::types::Float;

/// Generate synthetic calibration data
fn generate_calibration_data(n_samples: usize, noise_level: Float) -> (Array1<Float>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut probabilities = Array1::zeros(n_samples);
    let mut targets = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Generate base probability
        let base_prob = rng.gen::<Float>();

        // Add calibration bias (makes data miscalibrated)
        let biased_prob = (base_prob + noise_level * rng.gen::<Float>()).clamp(0.01, 0.99);
        probabilities[i] = biased_prob;

        // Generate target based on true probability
        targets[i] = if rng.gen::<Float>() < base_prob { 1 } else { 0 };
    }

    (probabilities, targets)
}

/// Generate multi-class calibration data
fn generate_multiclass_data(n_samples: usize, n_classes: usize) -> (Array2<Float>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut probabilities = Array2::zeros((n_samples, n_classes));
    let mut targets = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Generate raw logits
        let mut logits = vec![0.0; n_classes];
        for j in 0..n_classes {
            logits[j] = rng.gen_range(-2.0..2.0);
        }

        // Softmax to get probabilities
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: Float = logits.iter().map(|&x| (x - max_logit).exp()).sum();

        for j in 0..n_classes {
            probabilities[[i, j]] = (logits[j] - max_logit).exp() / exp_sum;
        }

        // Generate target
        targets[i] = rng.gen_range(0..n_classes) as i32;
    }

    (probabilities, targets)
}

/// Benchmark basic calibration methods
fn bench_basic_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_calibration");

    for &n_samples in &[100, 1000, 10000] {
        let (probabilities, targets) = generate_calibration_data(n_samples, 0.1);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Sigmoid calibration
        group.bench_with_input(
            BenchmarkId::new("sigmoid", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let calibrator = SigmoidCalibrator::new()
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        // Isotonic calibration
        group.bench_with_input(
            BenchmarkId::new("isotonic", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let calibrator = IsotonicCalibrator::new()
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        // Temperature scaling
        group.bench_with_input(
            BenchmarkId::new("temperature", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let calibrator = TemperatureScalingCalibrator::new()
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        // Histogram binning
        group.bench_with_input(
            BenchmarkId::new("histogram", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let calibrator = HistogramBinningCalibrator::new(10)
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark advanced calibration methods
fn bench_advanced_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_calibration");

    for &n_samples in &[100, 1000] {
        // Smaller sizes for expensive methods
        let (probabilities, targets) = generate_calibration_data(n_samples, 0.1);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Beta calibration
        group.bench_with_input(BenchmarkId::new("beta", n_samples), &n_samples, |b, _| {
            b.iter(|| {
                let calibrator = BetaCalibrator::new()
                    .fit(black_box(&probabilities), black_box(&targets))
                    .unwrap();
                calibrator.predict_proba(black_box(&probabilities)).unwrap()
            })
        });

        // KDE calibration (smaller datasets only)
        if n_samples <= 1000 {
            group.bench_with_input(BenchmarkId::new("kde", n_samples), &n_samples, |b, _| {
                b.iter(|| {
                    let calibrator = KDECalibrator::new()
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            });
        }

        // Local KNN calibration
        group.bench_with_input(
            BenchmarkId::new("local_knn", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut calibrator = LocalKNNCalibrator::new(5);
                    calibrator
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark neural calibration methods
fn bench_neural_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_calibration");
    group.sample_size(10); // Fewer samples for neural methods

    for &n_samples in &[100, 500] {
        let (probabilities, targets) = generate_calibration_data(n_samples, 0.1);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Neural calibration layer
        group.bench_with_input(
            BenchmarkId::new("neural_layer", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut calibrator =
                        NeuralCalibrationLayer::new(1, 1).with_learning_params(0.01, 10); // Fewer epochs for benchmarking
                    calibrator
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark multi-modal calibration methods
fn bench_multimodal_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("multimodal_calibration");

    for &n_samples in &[100, 500] {
        let (probabilities, targets) = generate_calibration_data(n_samples * 2, 0.1); // Double size for multi-modal

        group.throughput(Throughput::Elements(n_samples as u64));

        // Multi-modal calibration
        group.bench_with_input(
            BenchmarkId::new("multi_modal", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut calibrator =
                        MultiModalCalibrator::new(2, FusionStrategy::WeightedAverage);
                    calibrator.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));
                    calibrator.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));
                    calibrator
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        // Heterogeneous ensemble
        group.bench_with_input(
            BenchmarkId::new("heterogeneous_ensemble", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut calibrator = HeterogeneousEnsembleCalibrator::new(
                        EnsembleCombination::PerformanceWeighted,
                    );
                    calibrator.add_calibrator(Box::new(SigmoidCalibrator::new()));
                    calibrator.add_calibrator(Box::new(IsotonicCalibrator::new()));
                    calibrator
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Bayesian calibration methods
fn bench_bayesian_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("bayesian_calibration");
    group.sample_size(10); // Fewer samples for expensive Bayesian methods

    for &n_samples in &[100, 200] {
        let (probabilities, targets) = generate_calibration_data(n_samples, 0.1);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Bayesian model averaging
        group.bench_with_input(
            BenchmarkId::new("bayesian_model_averaging", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let models: Vec<Box<dyn CalibrationEstimator>> = vec![
                        Box::new(SigmoidCalibrator::new()),
                        Box::new(IsotonicCalibrator::new()),
                    ];
                    let mut calibrator = BayesianModelAveragingCalibrator::new(models);
                    calibrator
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        // Variational inference (small datasets only)
        if n_samples <= 100 {
            group.bench_with_input(
                BenchmarkId::new("variational_inference", n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let mut calibrator = VariationalInferenceCalibrator::new(); // Use default params
                        calibrator
                            .fit(black_box(&probabilities), black_box(&targets))
                            .unwrap();
                        calibrator.predict_proba(black_box(&probabilities)).unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark prediction-only performance (after training)
fn bench_prediction_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction_performance");

    // Pre-train calibrators
    let (train_probs, train_targets) = generate_calibration_data(1000, 0.1);
    let sigmoid_calibrator = SigmoidCalibrator::new()
        .fit(&train_probs, &train_targets)
        .unwrap();
    let isotonic_calibrator = IsotonicCalibrator::new()
        .fit(&train_probs, &train_targets)
        .unwrap();
    let temp_calibrator = TemperatureScalingCalibrator::new()
        .fit(&train_probs, &train_targets)
        .unwrap();

    for &n_samples in &[100, 1000, 10000, 100000] {
        let (test_probs, _) = generate_calibration_data(n_samples, 0.1);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Sigmoid prediction
        group.bench_with_input(
            BenchmarkId::new("sigmoid_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    sigmoid_calibrator
                        .predict_proba(black_box(&test_probs))
                        .unwrap()
                })
            },
        );

        // Isotonic prediction
        group.bench_with_input(
            BenchmarkId::new("isotonic_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    isotonic_calibrator
                        .predict_proba(black_box(&test_probs))
                        .unwrap()
                })
            },
        );

        // Temperature prediction
        group.bench_with_input(
            BenchmarkId::new("temperature_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    temp_calibrator
                        .predict_proba(black_box(&test_probs))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage and scalability
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    // Test with increasingly large datasets
    for &n_samples in &[1000, 5000, 10000, 50000] {
        let (probabilities, targets) = generate_calibration_data(n_samples, 0.1);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Only test fast methods for large datasets
        group.bench_with_input(
            BenchmarkId::new("sigmoid_large", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let calibrator = SigmoidCalibrator::new()
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("temperature_large", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let calibrator = TemperatureScalingCalibrator::new()
                        .fit(black_box(&probabilities), black_box(&targets))
                        .unwrap();
                    calibrator.predict_proba(black_box(&probabilities)).unwrap()
                })
            },
        );

        // Only test isotonic for smaller datasets due to O(n log n) complexity
        if n_samples <= 10000 {
            group.bench_with_input(
                BenchmarkId::new("isotonic_large", n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let calibrator = IsotonicCalibrator::new()
                            .fit(black_box(&probabilities), black_box(&targets))
                            .unwrap();
                        calibrator.predict_proba(black_box(&probabilities)).unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_calibration_methods,
    bench_advanced_calibration_methods,
    bench_neural_calibration_methods,
    bench_multimodal_calibration_methods,
    bench_bayesian_calibration_methods,
    bench_prediction_performance,
    bench_scalability,
);
criterion_main!(benches);
