/// Manual computation test to find the decision function bug
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::seeded_rng;
use scirs2_core::Distribution;
use sklears_core::traits::Fit;
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

    for _ in 0..half {
        for _ in 0..n_features {
            x_data.push(uniform.sample(&mut rng) - 5.0);
        }
        y_data.push(-1.0);
    }

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
fn manual_decision_function_computation() {
    println!("\n=== Manual Decision Function Computation ===\n");

    let (x, y) = generate_linearly_separable_data(50, 2, 42);

    // Train SVM with linear kernel to match manual computation
    let svc = SVC::new()
        .linear() // Use linear kernel for manual verification
        .c(1.0)
        .tol(1e-3)
        .max_iter(1000)
        .fit(&x, &y)
        .expect("Failed to fit SVC");

    println!("Model trained successfully");
    println!("Support vectors: {}", svc.support_vectors().nrows());
    println!("Intercept: {:.6}", svc.intercept());

    // Get the model's decision values
    let model_decision = svc
        .decision_function(&x)
        .expect("Failed to compute decision");

    // Manually compute decision function for the first sample
    let test_idx = 0;
    println!("\nManually computing decision for sample {}:", test_idx);
    println!(
        "  Sample x: [{:.4}, {:.4}]",
        x[[test_idx, 0]],
        x[[test_idx, 1]]
    );
    println!("  Actual label: {:.1}", y[test_idx]);

    let mut manual_score = 0.0;

    println!("\n  Contributions from each support vector:");
    for (sv_idx, &orig_idx) in svc.support_indices().iter().enumerate() {
        let sv = svc.support_vectors().row(sv_idx);
        let dual_coef = svc.dual_coef()[sv_idx];

        // Compute linear kernel manually: K(x, sv) = x · sv
        let mut dot_product = 0.0;
        for feat_idx in 0..x.ncols() {
            dot_product += x[[test_idx, feat_idx]] * sv[feat_idx];
        }

        let contribution = dual_coef * dot_product;
        manual_score += contribution;

        if sv_idx < 5 {
            // Show first 5 support vectors
            println!("    SV {}: orig_idx={}, sv=[{:.4}, {:.4}], dual_coef={:.4}, K(x,sv)={:.4}, contribution={:.4}",
                     sv_idx, orig_idx, sv[0], sv[1], dual_coef, dot_product, contribution);
        }
    }

    let manual_decision = manual_score + svc.intercept();

    println!("\n  Manual computation:");
    println!("    Sum of (dual_coef * kernel): {:.6}", manual_score);
    println!("    Intercept: {:.6}", svc.intercept());
    println!("    Total decision value: {:.6}", manual_decision);

    println!(
        "\n  Model's decision_function output: {:.6}",
        model_decision[test_idx]
    );

    println!(
        "\n  Difference: {:.6}",
        (manual_decision - model_decision[test_idx]).abs()
    );

    // They should match within floating point precision
    assert!(
        (manual_decision - model_decision[test_idx]).abs() < 1e-6,
        "Manual computation doesn't match model output! Manual: {:.6}, Model: {:.6}",
        manual_decision,
        model_decision[test_idx]
    );

    // Now check several more samples
    println!("\n=== Checking multiple samples ===");
    for test_idx in [0, 10, 20, 30, 40] {
        let mut manual_score = 0.0;
        for (sv_idx, _) in svc.support_indices().iter().enumerate() {
            let sv = svc.support_vectors().row(sv_idx);
            let dual_coef = svc.dual_coef()[sv_idx];

            let mut dot_product = 0.0;
            for feat_idx in 0..x.ncols() {
                dot_product += x[[test_idx, feat_idx]] * sv[feat_idx];
            }

            manual_score += dual_coef * dot_product;
        }

        let manual_decision = manual_score + svc.intercept();
        let model_decision_val = model_decision[test_idx];

        println!(
            "Sample {}: manual={:.4}, model={:.4}, diff={:.6}, actual_label={:.1}",
            test_idx,
            manual_decision,
            model_decision_val,
            (manual_decision - model_decision_val).abs(),
            y[test_idx]
        );

        assert!(
            (manual_decision - model_decision_val).abs() < 1e-6,
            "Mismatch at sample {}",
            test_idx
        );
    }
}
