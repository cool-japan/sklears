//! Linear SVM demonstration using coordinate descent
//!
//! This example shows how to use LinearSVC and LinearSVR for classification and regression
//! tasks without requiring BLAS operations.

use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};
use sklears_svm::{LinearSVC, LinearSVR};

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Linear SVM Demonstration");
    println!("========================");

    // Classification example
    println!(
        "
1. Linear Support Vector Classification"
    );
    println!("---------------------------------------");

    // Simple binary classification dataset
    let X_class_var = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [1.0, 1.0],
        [4.0, 4.0],
        [5.0, 5.0]
    ];
    let y_class = array![0, 1, 1, 0, 1, 0, 1, 1];

    let svc_model = LinearSVC::new()
        .with_c(1.0)
        .with_loss("squared_hinge")
        .with_max_iter(1000)
        .with_random_state(42);

    println!(
        "Training LinearSVC with {} samples...",
        X_class_var.len_of(scirs2_core::ndarray::Axis(0))
    );
    let trained_svc = svc_model.fit(&X_class_var, &y_class)?;

    let predictions = trained_svc.predict(&X_class_var)?;
    println!("Predictions: {:?}", predictions);
    println!("Actual:      {:?}", y_class);

    // Calculate accuracy
    let correct = predictions
        .iter()
        .zip(y_class.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count();
    let accuracy = correct as f64 / y_class.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Show decision function
    let decision_scores = trained_svc.decision_function(&X_class_var)?;
    println!("Decision scores shape: {:?}", decision_scores.dim());

    println!("Model coefficients: {:?}", trained_svc.coef());
    println!("Model intercept: {:?}", trained_svc.intercept());

    // Multi-class classification example
    println!(
        "
2. Multi-class Linear SVC"
    );
    println!("-------------------------");

    let X_multi_var = array![
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 1.0],
        [3.0, 3.0],
        [3.5, 3.5],
        [4.0, 3.0],
        [6.0, 6.0],
        [6.5, 6.5],
        [7.0, 6.0]
    ];
    let y_multi = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let multi_svc = LinearSVC::new()
        .with_c(1.0)
        .with_max_iter(1000)
        .with_random_state(42);

    println!("Training multi-class LinearSVC...");
    let trained_multi_svc = multi_svc.fit(&X_multi_var, &y_multi)?;

    let multi_predictions = trained_multi_svc.predict(&X_multi_var)?;
    println!("Multi-class predictions: {:?}", multi_predictions);
    println!("Multi-class actual:      {:?}", y_multi);

    let multi_decision_scores = trained_multi_svc.decision_function(&X_multi_var)?;
    println!(
        "Multi-class decision scores shape: {:?}",
        multi_decision_scores.dim()
    );

    // Regression example
    println!(
        "
3. Linear Support Vector Regression"
    );
    println!("-----------------------------------");

    // Simple regression dataset
    let X_reg_var = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0]
    ];
    let y_reg = array![2.5, 4.0, 5.5, 7.0, 8.5, 10.0]; // Linear relationship

    let svr_model = LinearSVR::new()
        .with_c(1.0)
        .with_epsilon(0.1)
        .with_loss("epsilon_insensitive")
        .with_max_iter(1000)
        .with_random_state(42);

    println!(
        "Training LinearSVR with {} samples...",
        X_reg_var.len_of(scirs2_core::ndarray::Axis(0))
    );
    let trained_svr = svr_model.fit(&X_reg_var, &y_reg)?;

    let reg_predictions = trained_svr.predict(&X_reg_var)?;
    println!("Regression predictions: {:?}", reg_predictions);
    println!("Regression actual:      {:?}", y_reg);

    // Calculate Mean Squared Error
    let mse = reg_predictions
        .iter()
        .zip(y_reg.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / y_reg.len() as f64;
    println!("Mean Squared Error: {:.4}", mse);

    println!("SVR coefficients: {:?}", trained_svr.coef());
    println!("SVR intercept: {:.4}", trained_svr.intercept());

    // Comparison with different loss functions
    println!(
        "
4. Comparing Loss Functions"
    );
    println!("---------------------------");

    // Test with hinge loss
    let svc_hinge = LinearSVC::new()
        .with_c(1.0)
        .with_loss("hinge")
        .with_max_iter(1000)
        .with_random_state(42);

    let trained_hinge = svc_hinge.fit(&X_class_var, &y_class)?;
    let predictions_hinge = trained_hinge.predict(&X_class_var)?;

    let accuracy_hinge = predictions_hinge
        .iter()
        .zip(y_class.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count() as f64
        / y_class.len() as f64;

    println!("Hinge loss accuracy: {:.2}%", accuracy_hinge * 100.0);

    // Test with squared epsilon insensitive loss for regression
    let svr_squared = LinearSVR::new()
        .with_c(1.0)
        .with_epsilon(0.1)
        .with_loss("squared_epsilon_insensitive")
        .with_max_iter(1000)
        .with_random_state(42);

    let trained_squared = svr_squared.fit(&X_reg_var, &y_reg)?;
    let predictions_squared = trained_squared.predict(&X_reg_var)?;

    let mse_squared = predictions_squared
        .iter()
        .zip(y_reg.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / y_reg.len() as f64;

    println!("Squared epsilon loss MSE: {:.4}", mse_squared);

    println!(
        "
Linear SVM demonstration completed successfully!"
    );
    Ok(())
}
