/// Test SVM accuracy with RBF kernel on various dataset sizes
/// This tests if there's an indexing bug that only appears with certain kernels
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::traits::{Fit, Predict};
use sklears_svm::svc::SVC;

/// Generate non-linearly separable dataset (XOR-like pattern)
fn generate_xor_dataset(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = seeded_rng(seed);
    let dist = Uniform::new(-0.5, 0.5).unwrap();

    let samples_per_quadrant = n_samples / 4;
    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    let mut idx = 0;

    // Quadrant 1: (1, 1) -> class 0
    for _ in 0..samples_per_quadrant {
        x[[idx, 0]] = 1.0 + dist.sample(&mut rng);
        x[[idx, 1]] = 1.0 + dist.sample(&mut rng);
        y[idx] = 0.0;
        idx += 1;
    }

    // Quadrant 2: (-1, 1) -> class 1
    for _ in 0..samples_per_quadrant {
        x[[idx, 0]] = -1.0 + dist.sample(&mut rng);
        x[[idx, 1]] = 1.0 + dist.sample(&mut rng);
        y[idx] = 1.0;
        idx += 1;
    }

    // Quadrant 3: (-1, -1) -> class 0
    for _ in 0..samples_per_quadrant {
        x[[idx, 0]] = -1.0 + dist.sample(&mut rng);
        x[[idx, 1]] = -1.0 + dist.sample(&mut rng);
        y[idx] = 0.0;
        idx += 1;
    }

    // Quadrant 4: (1, -1) -> class 1
    for _ in 0..(n_samples - idx) {
        x[[idx, 0]] = 1.0 + dist.sample(&mut rng);
        x[[idx, 1]] = -1.0 + dist.sample(&mut rng);
        y[idx] = 1.0;
        idx += 1;
    }

    (x, y)
}

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
fn test_rbf_kernel_small_dataset() {
    let (x, y) = generate_xor_dataset(20, 42);

    let svc = SVC::new()
        .rbf(Some(1.0))
        .c(10.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x, &y)
        .expect("Failed to fit SVC");

    let y_pred = svc.predict(&x).expect("Failed to predict");
    let accuracy = compute_accuracy(&y, &y_pred);

    println!("RBF Small dataset (20 samples):");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Support vectors: {}", svc.support_vectors().nrows());
    println!("  Support indices: {:?}", svc.support_indices());

    // XOR pattern should be learnable with RBF kernel
    assert!(
        accuracy >= 0.70,
        "Accuracy too low: {:.2}%",
        accuracy * 100.0
    );
}

#[test]
fn test_rbf_kernel_medium_dataset() {
    let (x, y) = generate_xor_dataset(60, 42);

    let svc = SVC::new()
        .rbf(Some(1.0))
        .c(10.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x, &y)
        .expect("Failed to fit SVC");

    let y_pred = svc.predict(&x).expect("Failed to predict");
    let accuracy = compute_accuracy(&y, &y_pred);

    println!("RBF Medium dataset (60 samples):");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Support vectors: {}", svc.support_vectors().nrows());
    println!("  Support indices: {:?}", svc.support_indices());

    assert!(
        accuracy >= 0.70,
        "Accuracy too low: {:.2}%",
        accuracy * 100.0
    );
}

#[test]
fn test_rbf_kernel_large_dataset() {
    let (x, y) = generate_xor_dataset(100, 42);

    let svc = SVC::new()
        .rbf(Some(1.0))
        .c(10.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x, &y)
        .expect("Failed to fit SVC");

    let y_pred = svc.predict(&x).expect("Failed to predict");
    let accuracy = compute_accuracy(&y, &y_pred);

    println!("RBF Large dataset (100 samples):");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Support vectors: {}", svc.support_vectors().nrows());
    println!("  Support indices: {:?}", svc.support_indices());

    // This is where the bug might show up
    assert!(
        accuracy >= 0.70,
        "Accuracy too low: {:.2}%. BUG DETECTED!",
        accuracy * 100.0
    );
}

#[test]
#[ignore = "Diagnostic test to check decision function values"]
fn test_decision_function_debug() {
    // Create a small dataset where we can manually verify the decision function
    let (x, y) = generate_xor_dataset(12, 42);

    let svc = SVC::new()
        .rbf(Some(1.0))
        .c(10.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x, &y)
        .expect("Failed to fit SVC");

    println!("\nDebug info:");
    println!("Training samples: {}", x.nrows());
    println!("Support vectors: {}", svc.support_vectors().nrows());
    println!("Support indices: {:?}", svc.support_indices());
    println!("Dual coefficients: {:?}", svc.dual_coef());
    println!("Intercept: {:.4}", svc.intercept());

    let decision_vals = svc
        .decision_function(&x)
        .expect("Failed to compute decision");
    let y_pred = svc.predict(&x).expect("Failed to predict");

    println!("\nDecision values and predictions:");
    for i in 0..x.nrows() {
        println!(
            "  Sample {}: decision={:.4}, predicted={:.0}, actual={:.0}",
            i, decision_vals[i], y_pred[i], y[i]
        );
    }

    let accuracy = compute_accuracy(&y, &y_pred);
    println!("\nOverall accuracy: {:.2}%", accuracy * 100.0);
}
