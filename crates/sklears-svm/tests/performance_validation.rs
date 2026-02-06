use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::prelude::*;
use sklears_core::types::Float;
use sklears_svm::SVC;
use std::time::Instant;

/// Generate synthetic linearly separable dataset (truly separable with no overlap)
fn generate_linearly_separable_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<Float>, Array1<Float>) {
    let mut rng = seeded_rng(seed);
    let uniform = Uniform::new(-1.0, 1.0).unwrap(); // Reduced noise

    let samples_per_class = n_samples / 2;
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    // Class -1: centered at (-2, -2, ...) with range [-3, -1] - no overlap
    for i in 0..samples_per_class {
        for j in 0..n_features {
            x[[i, j]] = -2.0 + uniform.sample(&mut rng); // Range: [-3, -1]
        }
        y[i] = -1.0;
    }

    // Class +1: centered at (2, 2, ...) with range [1, 3] - no overlap
    for i in samples_per_class..n_samples {
        for j in 0..n_features {
            x[[i, j]] = 2.0 + uniform.sample(&mut rng); // Range: [1, 3]
        }
        y[i] = 1.0;
    }

    (x, y)
}

#[test]
fn test_svc_performance_6_samples() {
    let (x, y) = generate_linearly_separable_data(6, 2, 42);

    let model = SVC::new();

    let start = Instant::now();
    let trained_model = model.fit(&x, &y).unwrap();
    let duration = start.elapsed();

    println!("6 samples - Training time: {:?}", duration);
    println!(
        "6 samples - Support vectors: {}",
        trained_model.support_vectors().nrows()
    );

    assert!(duration.as_secs() < 2, "Should complete in under 2 seconds");

    // Test prediction
    let predictions = trained_model.predict(&x).unwrap();
    println!("6 samples - Predictions: {:?}", predictions);
}

#[test]
fn test_svc_performance_20_samples() {
    let (x, y) = generate_linearly_separable_data(20, 2, 42);

    let model = SVC::new();

    let start = Instant::now();
    let trained_model = model.fit(&x, &y).unwrap();
    let duration = start.elapsed();

    println!("20 samples - Training time: {:?}", duration);
    println!(
        "20 samples - Support vectors: {}",
        trained_model.support_vectors().nrows()
    );

    assert!(duration.as_secs() < 3, "Should complete in under 3 seconds");

    // Test prediction accuracy
    let predictions = trained_model.predict(&x).unwrap();
    let accuracy = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| (*pred - *true_label).abs() < 0.1)
        .count() as Float
        / y.len() as Float;

    println!("20 samples - Training accuracy: {:.2}%", accuracy * 100.0);
    assert!(
        accuracy > 0.8,
        "Should achieve > 80% accuracy on training data"
    );
}

#[test]
fn test_svc_performance_50_samples() {
    let (x, y) = generate_linearly_separable_data(50, 2, 42);

    let model = SVC::new();

    let start = Instant::now();
    let trained_model = model.fit(&x, &y).unwrap();
    let duration = start.elapsed();

    println!("50 samples - Training time: {:?}", duration);
    println!(
        "50 samples - Support vectors: {}",
        trained_model.support_vectors().nrows()
    );

    assert!(duration.as_secs() < 5, "Should complete in under 5 seconds");

    // Test prediction accuracy
    let predictions = trained_model.predict(&x).unwrap();
    let accuracy = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| (*pred - *true_label).abs() < 0.1)
        .count() as Float
        / y.len() as Float;

    println!("50 samples - Training accuracy: {:.2}%", accuracy * 100.0);
    assert!(
        accuracy > 0.9,
        "Should achieve > 90% accuracy on linearly separable training data (got {:.2}%)",
        accuracy * 100.0
    );
}

#[test]
fn test_svc_performance_100_samples() {
    let (x, y) = generate_linearly_separable_data(100, 2, 42);

    let model = SVC::new();

    let start = Instant::now();
    let trained_model = model.fit(&x, &y).unwrap();
    let duration = start.elapsed();

    println!("100 samples - Training time: {:?}", duration);
    println!(
        "100 samples - Support vectors: {}",
        trained_model.support_vectors().nrows()
    );

    assert!(
        duration.as_secs() < 10,
        "Should complete in under 10 seconds"
    );

    // Test prediction accuracy
    let predictions = trained_model.predict(&x).unwrap();
    let accuracy = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| (*pred - *true_label).abs() < 0.1)
        .count() as Float
        / y.len() as Float;

    println!("100 samples - Training accuracy: {:.2}%", accuracy * 100.0);
    assert!(
        accuracy > 0.9,
        "Should achieve > 90% accuracy on linearly separable training data (got {:.2}%)",
        accuracy * 100.0
    );
}

#[test]
fn test_svc_convergence_stats() {
    let (x, y) = generate_linearly_separable_data(30, 2, 42);

    let model = SVC::new().c(1.0).max_iter(1000).tol(1e-3);

    let start = Instant::now();
    let trained_model = model.fit(&x, &y).unwrap();
    let duration = start.elapsed();

    println!("\nConvergence Statistics:");
    println!("  Training time: {:?}", duration);
    println!(
        "  Support vectors: {}",
        trained_model.support_vectors().nrows()
    );
    println!("  Total samples: {}", y.len());
    println!(
        "  Support vector ratio: {:.1}%",
        (trained_model.support_vectors().nrows() as Float / y.len() as Float) * 100.0
    );

    // Verify the model converged properly
    assert!(
        trained_model.support_vectors().nrows() > 0,
        "Should have at least one support vector"
    );
    assert!(
        trained_model.support_vectors().nrows() <= y.len(),
        "Support vectors should not exceed sample count"
    );
}
