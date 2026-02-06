/// Diagnostic test to understand the SVM accuracy bug
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::traits::{Fit, Predict};
use sklears_svm::svc::SVC;

fn generate_linearly_separable_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = seeded_rng(seed);
    let uniform = Uniform::new(-10.0, 10.0).unwrap();

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    let half = n_samples / 2;

    // Cluster 1: centered around (-5, -5, ...)
    for _ in 0..half {
        for _ in 0..n_features {
            x_data.push(uniform.sample(&mut rng) - 5.0);
        }
        y_data.push(-1.0);
    }

    // Cluster 2: centered around (5, 5, ...)
    for _ in 0..(n_samples - half) {
        for _ in 0..n_features {
            x_data.push(uniform.sample(&mut rng) + 5.0);
        }
        y_data.push(1.0);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

#[test]
fn diagnose_svm_bug_50_samples() {
    println!("\n=== Diagnosing SVM accuracy bug on 50-sample dataset ===\n");

    let (x, y) = generate_linearly_separable_data(50, 2, 42);

    println!("Dataset info:");
    println!("  Samples: {}", x.nrows());
    println!("  Features: {}", x.ncols());
    println!(
        "  Class -1 samples: {}",
        y.iter().filter(|&&label| label == -1.0).count()
    );
    println!(
        "  Class +1 samples: {}",
        y.iter().filter(|&&label| label == 1.0).count()
    );

    // Check data separability (print some samples from each class)
    println!("\nSample data points:");
    println!("  Class -1 samples (first 5):");
    let mut count = 0;
    for i in 0..x.nrows() {
        if y[i] == -1.0 && count < 5 {
            println!("    Sample {}: x=[{:.2}, {:.2}]", i, x[[i, 0]], x[[i, 1]]);
            count += 1;
        }
    }
    println!("  Class +1 samples (first 5):");
    count = 0;
    for i in 0..x.nrows() {
        if y[i] == 1.0 && count < 5 {
            println!("    Sample {}: x=[{:.2}, {:.2}]", i, x[[i, 0]], x[[i, 1]]);
            count += 1;
        }
    }

    // Train SVM
    let svc = SVC::new()
        .c(1.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x, &y)
        .expect("Failed to fit SVC");

    println!("\nModel info:");
    println!("  Support vectors: {}", svc.support_vectors().nrows());
    println!(
        "  Support indices (first 10): {:?}",
        &svc.support_indices()[..10.min(svc.support_indices().len())]
    );
    println!("  Intercept: {:.4}", svc.intercept());
    println!("  Classes: {:?}", svc.classes());

    // Check dual coefficients (first 10)
    println!("\nDual coefficients (first 10 support vectors):");
    for i in 0..10.min(svc.dual_coef().len()) {
        let sv_idx = svc.support_indices()[i];
        println!(
            "  SV {}: original_idx={}, dual_coef={:.4}, label={:.1}",
            i,
            sv_idx,
            svc.dual_coef()[i],
            y[sv_idx]
        );
    }

    // Get predictions and decision values
    let decision_vals = svc
        .decision_function(&x)
        .expect("Failed to compute decision");
    let predictions = svc.predict(&x).expect("Failed to predict");

    // Check ALL decision values (first 20 samples)
    println!("\nDecision values and predictions (first 20 samples):");
    for i in 0..20.min(x.nrows()) {
        println!(
            "  Sample {}: x=[{:.2}, {:.2}], decision={:.4}, predicted={:.1}, actual={:.1}, {}",
            i,
            x[[i, 0]],
            x[[i, 1]],
            decision_vals[i],
            predictions[i],
            y[i],
            if (predictions[i] - y[i]).abs() < 0.1 {
                "✓"
            } else {
                "✗"
            }
        );
    }

    // Check decision value statistics
    let min_decision = decision_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_decision = decision_vals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_decision = decision_vals.iter().sum::<f64>() / decision_vals.len() as f64;
    println!("\nDecision value statistics:");
    println!("  Min: {:.4}", min_decision);
    println!("  Max: {:.4}", max_decision);
    println!("  Mean: {:.4}", mean_decision);
    println!(
        "  Samples with negative decision (should be class -1): {}",
        decision_vals.iter().filter(|&&d| d < 0.0).count()
    );

    // Overall accuracy
    let accuracy = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| (*pred - *true_label).abs() < 0.1)
        .count() as f64
        / y.len() as f64;

    println!("\nOverall accuracy: {:.2}%", accuracy * 100.0);

    // Count misclassifications by class
    let mut false_pos = 0; // Predicted +1, actual -1
    let mut false_neg = 0; // Predicted -1, actual +1
    let mut true_pos = 0;
    let mut true_neg = 0;

    for (pred, actual) in predictions.iter().zip(y.iter()) {
        if *pred > 0.0 && *actual > 0.0 {
            true_pos += 1;
        } else if *pred < 0.0 && *actual < 0.0 {
            true_neg += 1;
        } else if *pred > 0.0 && *actual < 0.0 {
            false_pos += 1;
        } else if *pred < 0.0 && *actual > 0.0 {
            false_neg += 1;
        }
    }

    println!("\nConfusion matrix:");
    println!("  True positives:  {}", true_pos);
    println!("  True negatives:  {}", true_neg);
    println!("  False positives: {}", false_pos);
    println!("  False negatives: {}", false_neg);

    // This test always runs to completion to show diagnostic info
}
