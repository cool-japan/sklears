/// Integration test to reproduce and verify SVM accuracy on large datasets
/// This test was created to diagnose an issue where SVM accuracy degrades
/// on datasets with 50-100 samples but works correctly on smaller datasets.
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::traits::{Fit, Predict};
use sklears_svm::svc::SVC;

/// Generate a linearly separable dataset for binary classification
fn generate_linearly_separable_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = seeded_rng(seed);

    // Generate linearly separable data
    // Class 0: samples clustered around (-2, -2, ...)
    // Class 1: samples clustered around (2, 2, ...)
    // With small noise, these should be perfectly separable

    let samples_per_class = n_samples / 2;
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    let dist = Uniform::new(-1.0, 1.0).unwrap();

    // Class 0 (label 0.0) - centered at negative values
    for i in 0..samples_per_class {
        for j in 0..n_features {
            x[[i, j]] = -2.0 + dist.sample(&mut rng);
        }
        y[i] = 0.0;
    }

    // Class 1 (label 1.0) - centered at positive values
    for i in samples_per_class..n_samples {
        for j in 0..n_features {
            x[[i, j]] = 2.0 + dist.sample(&mut rng);
        }
        y[i] = 1.0;
    }

    // Split into train (70%) and test (30%)
    let train_size = (n_samples as f64 * 0.7) as usize;

    let x_train = x.slice(s![0..train_size, ..]).to_owned();
    let y_train = y.slice(s![0..train_size]).to_owned();
    let x_test = x.slice(s![train_size.., ..]).to_owned();
    let y_test = y.slice(s![train_size..]).to_owned();

    (x_train, y_train, x_test, y_test)
}

/// Compute classification accuracy
fn compute_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mut correct = 0;
    for i in 0..y_true.len() {
        if (y_true[i] - y_pred[i]).abs() < 0.5 {
            correct += 1;
        }
    }
    correct as f64 / y_true.len() as f64
}

#[test]
fn test_svm_small_dataset_accuracy() {
    // Small datasets (6-30 samples) should work correctly
    let (x_train, y_train, x_test, y_test) = generate_linearly_separable_data(20, 2, 42);

    let svc = SVC::new()
        .linear()
        .c(1.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x_train, &y_train)
        .expect("Failed to fit SVC");

    let y_pred_train = svc.predict(&x_train).expect("Failed to predict train");
    let y_pred_test = svc.predict(&x_test).expect("Failed to predict test");

    let train_accuracy = compute_accuracy(&y_train, &y_pred_train);
    let test_accuracy = compute_accuracy(&y_test, &y_pred_test);

    println!("Small dataset (20 samples):");
    println!("  Train accuracy: {:.2}%", train_accuracy * 100.0);
    println!("  Test accuracy: {:.2}%", test_accuracy * 100.0);
    println!("  Support vectors: {}", svc.support_vectors().nrows());

    // For linearly separable data, we should get >90% accuracy
    assert!(
        train_accuracy >= 0.90,
        "Train accuracy too low: {:.2}%",
        train_accuracy * 100.0
    );
    assert!(
        test_accuracy >= 0.80,
        "Test accuracy too low: {:.2}%",
        test_accuracy * 100.0
    );
}

#[test]
fn test_svm_medium_dataset_accuracy() {
    // Medium datasets (50 samples) were showing degraded accuracy
    let (x_train, y_train, x_test, y_test) = generate_linearly_separable_data(50, 2, 42);

    let svc = SVC::new()
        .linear()
        .c(1.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x_train, &y_train)
        .expect("Failed to fit SVC");

    let y_pred_train = svc.predict(&x_train).expect("Failed to predict train");
    let y_pred_test = svc.predict(&x_test).expect("Failed to predict test");

    let train_accuracy = compute_accuracy(&y_train, &y_pred_train);
    let test_accuracy = compute_accuracy(&y_test, &y_pred_test);

    println!("Medium dataset (50 samples):");
    println!("  Train accuracy: {:.2}%", train_accuracy * 100.0);
    println!("  Test accuracy: {:.2}%", test_accuracy * 100.0);
    println!("  Support vectors: {}", svc.support_vectors().nrows());

    // Should maintain high accuracy on linearly separable data
    assert!(
        train_accuracy >= 0.90,
        "Train accuracy too low: {:.2}%",
        train_accuracy * 100.0
    );
    assert!(
        test_accuracy >= 0.80,
        "Test accuracy too low: {:.2}%",
        test_accuracy * 100.0
    );
}

#[test]
fn test_svm_large_dataset_accuracy() {
    // Large datasets (100 samples) showed ~50% accuracy (random guessing)
    let (x_train, y_train, x_test, y_test) = generate_linearly_separable_data(100, 2, 42);

    let svc = SVC::new()
        .linear()
        .c(1.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x_train, &y_train)
        .expect("Failed to fit SVC");

    let y_pred_train = svc.predict(&x_train).expect("Failed to predict train");
    let y_pred_test = svc.predict(&x_test).expect("Failed to predict test");

    let train_accuracy = compute_accuracy(&y_train, &y_pred_train);
    let test_accuracy = compute_accuracy(&y_test, &y_pred_test);

    println!("Large dataset (100 samples):");
    println!("  Train accuracy: {:.2}%", train_accuracy * 100.0);
    println!("  Test accuracy: {:.2}%", test_accuracy * 100.0);
    println!("  Support vectors: {}", svc.support_vectors().nrows());
    println!("  Support indices: {:?}", svc.support_indices());

    // This should NOT be ~50% (random guessing)
    // For linearly separable data, we expect >90% accuracy
    assert!(
        train_accuracy >= 0.90,
        "Train accuracy too low: {:.2}%. This indicates the SVM bug is present!",
        train_accuracy * 100.0
    );
    assert!(
        test_accuracy >= 0.80,
        "Test accuracy too low: {:.2}%. This indicates the SVM bug is present!",
        test_accuracy * 100.0
    );
}

#[test]
#[ignore = "Slow test: runs multiple dataset sizes for comprehensive validation"]
fn test_svm_accuracy_across_dataset_sizes() {
    // Comprehensive test across multiple dataset sizes
    for &n_samples in &[10, 20, 30, 50, 100, 150, 200] {
        let (x_train, y_train, x_test, y_test) = generate_linearly_separable_data(n_samples, 2, 42);

        let svc = SVC::new()
            .linear()
            .c(1.0)
            .tol(1e-3)
            .max_iter(1000)
            .fit(&x_train, &y_train)
            .expect("Failed to fit SVC");

        let y_pred_train = svc.predict(&x_train).expect("Failed to predict train");
        let y_pred_test = svc.predict(&x_test).expect("Failed to predict test");

        let train_accuracy = compute_accuracy(&y_train, &y_pred_train);
        let test_accuracy = compute_accuracy(&y_test, &y_pred_test);

        println!("Dataset size: {} samples", n_samples);
        println!(
            "  Train: {} samples, Test: {} samples",
            x_train.nrows(),
            x_test.nrows()
        );
        println!("  Train accuracy: {:.2}%", train_accuracy * 100.0);
        println!("  Test accuracy: {:.2}%", test_accuracy * 100.0);
        println!("  Support vectors: {}", svc.support_vectors().nrows());

        // All dataset sizes should maintain high accuracy
        assert!(
            train_accuracy >= 0.90,
            "Train accuracy too low for {} samples: {:.2}%",
            n_samples,
            train_accuracy * 100.0
        );
    }
}
