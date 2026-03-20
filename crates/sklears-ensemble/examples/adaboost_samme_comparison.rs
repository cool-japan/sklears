//! Comparison between SAMME and SAMME.R AdaBoost algorithms
//!
//! This example demonstrates the differences between the two AdaBoost variants:
//! - SAMME: Uses hard predictions and traditional weight updates
//! - SAMME.R: Uses probability estimates and real-valued updates

use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};
use sklears_ensemble::{AdaBoostAlgorithm, AdaBoostClassifier};

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AdaBoost SAMME vs SAMME.R Comparison");
    println!("=====================================");

    // Create a simple binary classification dataset
    let X = array![
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 3.0],
        [3.0, 4.0],
        [5.0, 4.0],
        [4.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [6.0, 6.0],
        [7.0, 7.0],
    ];

    let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

    println!(
        "Dataset size: {} samples, {} features",
        X.nrows(),
        X.ncols()
    );
    println!("Class distribution: {:?}", y);
    println!();

    // Test SAMME algorithm
    println!("1. Training AdaBoost with SAMME algorithm");
    println!("-----------------------------------------");

    let adaboost_samme = AdaBoostClassifier::new()
        .algorithm(AdaBoostAlgorithm::SAMME)
        .n_estimators(10)
        .learning_rate(1.0)
        .random_state(42)
        .fit(&X, &y)?;

    let predictions_samme = adaboost_samme.predict(&X)?;
    println!("SAMME predictions: {:?}", predictions_samme);

    // Calculate accuracy
    let correct_samme = predictions_samme
        .iter()
        .zip(y.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count();
    let accuracy_samme = correct_samme as f64 / y.len() as f64;
    println!("SAMME accuracy: {:.2}%", accuracy_samme * 100.0);

    // Show model details
    println!("SAMME estimators: {}", adaboost_samme.estimators().len());
    println!(
        "SAMME estimator weights: {:?}",
        adaboost_samme.estimator_weights()
    );
    println!(
        "SAMME estimator errors: {:?}",
        adaboost_samme.estimator_errors()
    );
    println!();

    // Test SAMME.R algorithm
    println!("2. Training AdaBoost with SAMME.R algorithm");
    println!("-------------------------------------------");

    let adaboost_sammer = AdaBoostClassifier::new()
        .algorithm(AdaBoostAlgorithm::SAMMER)
        .n_estimators(10)
        .learning_rate(0.5) // Often SAMME.R works better with smaller learning rates
        .random_state(42)
        .fit(&X, &y)?;

    let predictions_sammer = adaboost_sammer.predict(&X)?;
    println!("SAMME.R predictions: {:?}", predictions_sammer);

    // Calculate accuracy
    let correct_sammer = predictions_sammer
        .iter()
        .zip(y.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count();
    let accuracy_sammer = correct_sammer as f64 / y.len() as f64;
    println!("SAMME.R accuracy: {:.2}%", accuracy_sammer * 100.0);

    // Show model details
    println!("SAMME.R estimators: {}", adaboost_sammer.estimators().len());
    println!(
        "SAMME.R estimator weights: {:?}",
        adaboost_sammer.estimator_weights()
    );
    println!(
        "SAMME.R estimator errors: {:?}",
        adaboost_sammer.estimator_errors()
    );
    println!();

    // Compare probability predictions
    println!("3. Probability Predictions Comparison");
    println!("-------------------------------------");

    let probas_samme = adaboost_samme.predict_proba(&X)?;
    let probas_sammer = adaboost_sammer.predict_proba(&X)?;

    println!("Sample probability predictions (first 5 samples):");
    for i in 0..5.min(X.nrows()) {
        println!("Sample {}: True={:.0}", i, y[i]);
        println!(
            "  SAMME probabilities:  [{:.3}, {:.3}]",
            probas_samme[[i, 0]],
            probas_samme[[i, 1]]
        );
        println!(
            "  SAMME.R probabilities: [{:.3}, {:.3}]",
            probas_sammer[[i, 0]],
            probas_sammer[[i, 1]]
        );
        println!();
    }

    // Feature importance comparison
    println!("4. Feature Importance Comparison");
    println!("--------------------------------");

    let importances_samme = adaboost_samme.feature_importances()?;
    let importances_sammer = adaboost_sammer.feature_importances()?;

    println!("SAMME feature importances: {:?}", importances_samme);
    println!("SAMME.R feature importances: {:?}", importances_sammer);
    println!();

    // Multi-class example
    println!("5. Multi-class Classification Example");
    println!("------------------------------------");

    let X_multi = array![
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 1.0],
        [3.0, 3.0],
        [3.5, 3.5],
        [4.0, 3.0],
        [6.0, 6.0],
        [6.5, 6.5],
        [7.0, 6.0],
        [1.0, 7.0],
        [1.5, 7.5],
        [2.0, 7.0],
    ];

    let y_multi = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

    // SAMME with multi-class
    let multi_samme = AdaBoostClassifier::new()
        .algorithm(AdaBoostAlgorithm::SAMME)
        .n_estimators(15)
        .learning_rate(1.0)
        .random_state(42)
        .fit(&X_multi, &y_multi)?;

    let multi_pred_samme = multi_samme.predict(&X_multi)?;
    let multi_acc_samme = multi_pred_samme
        .iter()
        .zip(y_multi.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count() as f64
        / y_multi.len() as f64;

    println!(
        "Multi-class SAMME accuracy: {:.2}%",
        multi_acc_samme * 100.0
    );

    // SAMME.R with multi-class
    let multi_sammer = AdaBoostClassifier::new()
        .algorithm(AdaBoostAlgorithm::SAMMER)
        .n_estimators(15)
        .learning_rate(0.5)
        .random_state(42)
        .fit(&X_multi, &y_multi)?;

    let multi_pred_sammer = multi_sammer.predict(&X_multi)?;
    let multi_acc_sammer = multi_pred_sammer
        .iter()
        .zip(y_multi.iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count() as f64
        / y_multi.len() as f64;

    println!(
        "Multi-class SAMME.R accuracy: {:.2}%",
        multi_acc_sammer * 100.0
    );
    println!();

    println!("Algorithm Characteristics:");
    println!("---------------------------");
    println!("SAMME:");
    println!("  - Uses discrete class predictions");
    println!("  - Traditional AdaBoost weight updates");
    println!("  - Can work with any weak learner");
    println!("  - Weight calculation: log((1-err)/err) + log(K-1)");
    println!();
    println!("SAMME.R:");
    println!("  - Uses class probability estimates");
    println!("  - Real-valued probability-based updates");
    println!("  - Requires probability-capable weak learners");
    println!("  - Often converges faster than SAMME");
    println!("  - More stable weight updates");
    println!();

    println!("AdaBoost comparison completed successfully!");
    Ok(())
}
