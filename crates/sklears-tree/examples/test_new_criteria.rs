// Test for new criteria implementations
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict};
use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing new criteria implementations...");

    // Test data for classification
    let x_class = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
    )?;
    let y_class = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

    // Test Twoing criterion
    println!("Testing Twoing criterion...");
    let tree_twoing = DecisionTreeClassifier::new()
        .criterion(SplitCriterion::Twoing)
        .fit(&x_class, &y_class)?;
    let pred_twoing = tree_twoing.predict(&x_class)?;
    println!("Twoing predictions: {:?}", pred_twoing);

    // Test LogLoss criterion
    println!("Testing LogLoss criterion...");
    let tree_logloss = DecisionTreeClassifier::new()
        .criterion(SplitCriterion::LogLoss)
        .fit(&x_class, &y_class)?;
    let pred_logloss = tree_logloss.predict(&x_class)?;
    println!("LogLoss predictions: {:?}", pred_logloss);

    // Test data for regression
    let x_reg = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
    )?;
    let y_reg = Array1::from_vec(vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);

    // Test MAE criterion
    println!("Testing MAE criterion...");
    let tree_mae = DecisionTreeRegressor::new()
        .criterion(SplitCriterion::MAE)
        .fit(&x_reg, &y_reg)?;
    let pred_mae = tree_mae.predict(&x_reg)?;
    println!("MAE predictions: {:?}", pred_mae);

    println!("All tests completed successfully!");
    Ok(())
}
