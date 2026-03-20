//! Example: Showcase of Linear Models with Different Solvers
//!
//! This example demonstrates:
//! - Lasso regression with coordinate descent
//! - ElasticNet regression
//! - Logistic regression with L-BFGS, SAG, and SAGA solvers

#![allow(unexpected_cfgs)]

use scirs2_core::ndarray::Array2;
use sklears::linear::LinearRegression;
#[cfg(feature = "logistic-regression")]
use sklears::linear::LogisticRegression;
use sklears::metrics::regression::r2_score;
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("=== Linear Models Showcase ===\n");

    // 1. Lasso Regression Example
    println!("1. Lasso Regression (L1 Penalty)");
    println!("--------------------------------");

    // Generate synthetic data with sparse coefficients
    let n_samples = 50;
    let n_features = 10;
    let mut x_reg = Array2::zeros((n_samples, n_features));

    // Create data where only first 3 features are relevant
    for i in 0..n_samples {
        x_reg[[i, 0]] = i as f64 / 10.0;
        x_reg[[i, 1]] = (i as f64 / 5.0).sin();
        x_reg[[i, 2]] = (i as f64 / 7.0).cos();
        // Add noise to other features
        for j in 3..n_features {
            x_reg[[i, j]] = ((i * j) % 17) as f64 / 20.0 - 0.425;
        }
    }

    // Target: linear combination of first 3 features
    let y_reg = &x_reg.column(0) * 3.0 + &x_reg.column(1) * 2.0 - &x_reg.column(2) * 1.5;

    // Fit Lasso model
    let lasso = LinearRegression::lasso(0.1).fit(&x_reg, &y_reg)?;

    println!("Lasso coefficients:");
    for (i, &coef) in lasso.coef().iter().enumerate() {
        if coef.abs() > 1e-6 {
            println!("  Feature {}: {:.3}", i, coef);
        }
    }

    let y_pred = lasso.predict(&x_reg)?;
    let r2 = r2_score(&y_reg, &y_pred)?;
    println!("R² score: {:.4}\n", r2);

    // 2. ElasticNet Regression Example
    println!("2. ElasticNet Regression (L1 + L2 Penalty)");
    println!("------------------------------------------");

    let elastic_net = LinearRegression::elastic_net(0.1, 0.5).fit(&x_reg, &y_reg)?;

    println!("ElasticNet coefficients (α=0.1, l1_ratio=0.5):");
    for (i, &coef) in elastic_net.coef().iter().enumerate() {
        if coef.abs() > 1e-6 {
            println!("  Feature {}: {:.3}", i, coef);
        }
    }

    let y_pred = elastic_net.predict(&x_reg)?;
    let r2 = r2_score(&y_reg, &y_pred)?;
    println!("R² score: {:.4}\n", r2);

    #[cfg(feature = "logistic-regression")]
    {
        // 3. Logistic Regression with Different Solvers
        println!("3. Logistic Regression Classification");
        println!("------------------------------------");

        // Create binary classification data
        let x_clf = array![
            [2.0, 3.0, 1.0],
            [1.5, 2.5, 1.2],
            [3.0, 4.0, 0.8],
            [2.5, 3.5, 1.1],
            [-1.0, -2.0, -0.5],
            [-1.5, -2.5, -0.7],
            [-2.0, -3.0, -0.3],
            [-2.5, -3.5, -0.6],
        ];
        let y_clf = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        // Test different solvers
        let solvers = vec![
            ("L-BFGS", Solver::Lbfgs),
            ("SAG", Solver::Sag),
            ("SAGA", Solver::Saga),
        ];

        for (name, solver) in solvers {
            println!("\nLogistic Regression with {} solver:", name);

            let model = LogisticRegression::new()
                .solver(solver)
                .penalty(Penalty::L2(0.1))
                .max_iter(200)
                .random_state(42)
                .fit(&x_clf, &y_clf)?;

            let accuracy = model.score(&x_clf, &y_clf)?;
            println!("  Training accuracy: {:.2}%", accuracy * 100.0);

            let coef = model.coef();
            println!(
                "  Coefficients: [{:.3}, {:.3}, {:.3}]",
                coef[0], coef[1], coef[2]
            );

            if let Some(intercept) = model.intercept() {
                println!("  Intercept: {:.3}", intercept);
            }

            // Test prediction probabilities
            let test_point = array![[1.0, 1.5, 0.5]];
            let proba = model.predict_proba(&test_point)?;
            println!("  P(class=0) for [1.0, 1.5, 0.5]: {:.3}", proba[[0, 0]]);
            println!("  P(class=1) for [1.0, 1.5, 0.5]: {:.3}", proba[[0, 1]]);
        }
    }

    #[cfg(not(feature = "logistic-regression"))]
    {
        println!("3. Logistic Regression Classification");
        println!("------------------------------------");
        println!("Logistic regression requires the 'logistic-regression' feature.");
        println!("Enable it with: cargo run --example linear_models_showcase --features logistic-regression\n");
    }

    #[cfg(feature = "logistic-regression")]
    {
        // 4. Logistic Regression with L1 Penalty (using SAGA)
        println!("\n4. Sparse Logistic Regression (L1 Penalty with SAGA)");
        println!("----------------------------------------------------");

        // Add some irrelevant features
        let mut x_sparse = Array2::zeros((8, 6));
        x_sparse.slice_mut(ndarray::s![.., 0..3]).assign(&x_clf);
        // Irrelevant features
        for i in 0..8 {
            for j in 3..6 {
                x_sparse[[i, j]] = ((i + j) % 5) as f64 / 5.0 - 0.4;
            }
        }

        let sparse_model = LogisticRegression::new()
            .solver(Solver::Saga)
            .penalty(Penalty::L1(0.5))
            .max_iter(200)
            .random_state(42)
            .fit(&x_sparse, &y_clf)?;

        println!("Sparse Logistic Regression coefficients:");
        for (i, &coef) in sparse_model.coef().iter().enumerate() {
            if coef.abs() > 1e-6 {
                println!("  Feature {}: {:.3}", i, coef);
            } else {
                println!("  Feature {}: 0.000 (zeroed by L1)", i);
            }
        }

        let accuracy = sparse_model.score(&x_sparse, &y_clf)?;
        println!("Training accuracy: {:.2}%", accuracy * 100.0);
    }

    #[cfg(not(feature = "logistic-regression"))]
    {
        println!("\n4. Sparse Logistic Regression (L1 Penalty with SAGA)");
        println!("----------------------------------------------------");
        println!("Sparse logistic regression requires the 'logistic-regression' feature.");
        println!("Enable it with: cargo run --example linear_models_showcase --features logistic-regression");
    }

    Ok(())
}
