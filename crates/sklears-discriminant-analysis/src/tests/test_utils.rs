//! Common test utilities and shared imports for discriminant analysis tests

pub use super::super::*;
pub use scirs2_core::ndarray::{array, Array1, Array2, Axis};
pub use sklears_core::{
    traits::{Fit, Predict, PredictProba, Transform},
    types::Float,
};

/// Helper function to create simple 2D test data
pub fn create_simple_2d_data() -> (Array2<Float>, Array1<i32>) {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];
    (x, y)
}

/// Helper function to create 3D test data
pub fn create_simple_3d_data() -> (Array2<Float>, Array1<i32>) {
    let x = array![
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ];
    let y = array![0, 0, 1, 1];
    (x, y)
}

/// Helper function to create test data with outliers
pub fn create_data_with_outliers() -> (Array2<Float>, Array1<i32>) {
    let x = array![
        [1.0, 1.0],
        [1.1, 1.1],
        [1.0, 1.1],
        [1.1, 1.0],   // Class 0 - clean data
        [10.0, 10.0], // Outlier in class 0
        [3.0, 3.0],
        [3.1, 3.1],
        [3.0, 3.1],
        [3.1, 3.0],   // Class 1 - clean data
        [20.0, 20.0]  // Outlier in class 1
    ];
    let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    (x, y)
}

/// Helper function to check that probabilities sum to 1
pub fn assert_probabilities_sum_to_one(probas: &Array2<Float>) {
    for row in probas.axis_iter(Axis(0)) {
        let sum: Float = row.sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {}",
            sum
        );
    }
}
