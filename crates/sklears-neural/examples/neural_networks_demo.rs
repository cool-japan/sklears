//! Neural Networks demonstration for sklears-neural crate.
//!
//! This example shows how to use MLPClassifier and MLPRegressor
//! for various machine learning tasks.

use scirs2_core::ndarray::{array, Array2, Axis};
use sklears_core::traits::{Fit, Predict};
use sklears_neural::{
    activation::Activation, mlp_classifier::MLPClassifier, mlp_regressor::MLPRegressor,
    solvers::Solver, utils::WeightInit,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Neural Networks Demo ===\n");

    // Classification example: XOR problem
    println!("🧠 MLPClassifier: XOR Problem");
    println!("-------------------------------");

    let x_xor = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];
    let y_xor = vec![0, 1, 1, 0];

    println!("Training data:");
    for (i, row) in x_xor.axis_iter(Axis(0)).enumerate() {
        println!(
            "  Input: [{:.1}, {:.1}] -> Output: {}",
            row[0], row[1], y_xor[i]
        );
    }

    let mlp_classifier = MLPClassifier::new()
        .hidden_layer_sizes(&[10, 8])
        .activation(Activation::Tanh)
        .solver(Solver::Adam)
        .learning_rate_init(0.01)
        .max_iter(500)
        .random_state(42)
        .verbose(true);

    println!("\nTraining MLPClassifier...");
    let trained_classifier = mlp_classifier.fit(&x_xor, &y_xor)?;

    let predictions = trained_classifier.predict(&x_xor)?;
    let probabilities = trained_classifier.predict_proba(&x_xor)?;

    println!("\nResults:");
    println!("Final loss: {:.6}", trained_classifier.loss());
    println!("Iterations: {}", trained_classifier.n_iter());
    println!("Classes: {:?}", trained_classifier.classes());

    println!("\nPredictions vs Truth:");
    for i in 0..x_xor.nrows() {
        println!(
            "  [{:.1}, {:.1}] -> Predicted: {}, True: {}, Prob: [{:.3}, {:.3}]",
            x_xor[[i, 0]],
            x_xor[[i, 1]],
            predictions[i],
            y_xor[i],
            probabilities[[i, 0]],
            probabilities[[i, 1]]
        );
    }

    // Regression example: Linear function approximation
    println!("\n🧠 MLPRegressor: Linear Function Approximation");
    println!("------------------------------------------------");

    let x_reg = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [0.0, 1.0],
        [1.5, 2.5],
        [3.5, 4.5],
    ];

    // Target function: y = 2*x1 + 3*x2 + 1
    let mut y_reg = Array2::zeros((x_reg.nrows(), 1));
    for i in 0..x_reg.nrows() {
        y_reg[[i, 0]] = 2.0 * x_reg[[i, 0]] + 3.0 * x_reg[[i, 1]] + 1.0;
    }

    println!("Target function: y = 2*x1 + 3*x2 + 1");
    println!("Training data (first 5 samples):");
    for i in 0..5.min(x_reg.nrows()) {
        println!(
            "  Input: [{:.1}, {:.1}] -> Output: {:.1}",
            x_reg[[i, 0]],
            x_reg[[i, 1]],
            y_reg[[i, 0]]
        );
    }

    let mlp_regressor = MLPRegressor::new()
        .hidden_layer_sizes(&[20, 10])
        .activation(Activation::Relu)
        .solver(Solver::Adam)
        .learning_rate_init(0.001)
        .alpha(0.0001) // L2 regularization
        .max_iter(300)
        .early_stopping(true)
        .random_state(123)
        .verbose(true);

    println!("\nTraining MLPRegressor...");
    let trained_regressor = mlp_regressor.fit(&x_reg, &y_reg)?;

    let reg_predictions = trained_regressor.predict(&x_reg)?;

    println!("\nResults:");
    println!("Final loss: {:.6}", trained_regressor.loss());
    println!("Iterations: {}", trained_regressor.n_iter());

    // Calculate mean absolute error
    let mut total_error = 0.0;
    for i in 0..y_reg.nrows() {
        total_error += (reg_predictions[[i, 0]] - y_reg[[i, 0]]).abs();
    }
    let mae = total_error / y_reg.nrows() as f64;
    println!("Mean Absolute Error: {:.4}", mae);

    println!("\nPredictions vs Truth (first 5 samples):");
    for i in 0..5.min(x_reg.nrows()) {
        println!(
            "  [{:.1}, {:.1}] -> Predicted: {:.2}, True: {:.1}, Error: {:.3}",
            x_reg[[i, 0]],
            x_reg[[i, 1]],
            reg_predictions[[i, 0]],
            y_reg[[i, 0]],
            (reg_predictions[[i, 0]] - y_reg[[i, 0]]).abs()
        );
    }

    // Multi-class classification example
    println!("\n🧠 MLPClassifier: Multi-class Classification");
    println!("---------------------------------------------");

    let x_multi = array![
        // Class 0: bottom-left cluster
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.1],
        [0.1, 0.0],
        // Class 1: top-right cluster
        [1.0, 1.0],
        [1.1, 1.0],
        [1.0, 1.1],
        [0.9, 0.9],
        // Class 2: top-left cluster
        [0.0, 1.0],
        [0.1, 1.1],
        [0.0, 0.9],
        [0.1, 0.9],
    ];
    let y_multi = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    println!("3-class classification problem with clustered data");
    println!("Classes: 0 (bottom-left), 1 (top-right), 2 (top-left)");

    let mlp_multi = MLPClassifier::new()
        .hidden_layer_sizes(&[8, 4])
        .activation(Activation::Relu)
        .solver(Solver::Adam)
        .learning_rate_init(0.01)
        .max_iter(200)
        .weight_init(WeightInit::He)
        .random_state(42);

    println!("\nTraining multi-class MLPClassifier...");
    let trained_multi = mlp_multi.fit(&x_multi, &y_multi)?;

    let multi_predictions = trained_multi.predict(&x_multi)?;
    let multi_probabilities = trained_multi.predict_proba(&x_multi)?;

    println!("\nResults:");
    println!("Final loss: {:.6}", trained_multi.loss());
    println!("Iterations: {}", trained_multi.n_iter());
    println!("Classes: {:?}", trained_multi.classes());

    // Calculate accuracy
    let correct = multi_predictions
        .iter()
        .zip(y_multi.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();
    let accuracy = correct as f64 / y_multi.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    println!("\nSample predictions (first 6):");
    for i in 0..6 {
        println!(
            "  [{:.1}, {:.1}] -> Predicted: {}, True: {}, Confidence: {:.3}",
            x_multi[[i, 0]],
            x_multi[[i, 1]],
            multi_predictions[i],
            y_multi[i],
            multi_probabilities[[i, multi_predictions[i]]]
        );
    }

    // Demonstrate different activation functions
    println!("\n🧠 Comparing Activation Functions");
    println!("----------------------------------");

    let activations = vec![
        (Activation::Relu, "ReLU"),
        (Activation::Tanh, "Tanh"),
        (Activation::Logistic, "Logistic"),
    ];

    for (activation, name) in activations {
        let mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[8])
            .activation(activation)
            .max_iter(100)
            .random_state(42);

        let trained = mlp.fit(&x_xor, &y_xor)?;
        let preds = trained.predict(&x_xor)?;

        let correct = preds
            .iter()
            .zip(y_xor.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();
        let acc = correct as f64 / y_xor.len() as f64;

        println!(
            "  {} activation: {:.2}% accuracy, {:.6} final loss, {} iterations",
            name,
            acc * 100.0,
            trained.loss(),
            trained.n_iter()
        );
    }

    println!("\n✅ Neural Networks demo completed!");

    Ok(())
}
