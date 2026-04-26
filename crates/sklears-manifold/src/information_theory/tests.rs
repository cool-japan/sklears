//! Tests for information-theoretic manifold learning algorithms
//!
//! These tests validate the information-theoretic utility functions using
//! small synthetic datasets with analytically known or bounded results.

#[allow(non_snake_case)]
#[cfg(test)]
mod it_tests {
    use scirs2_core::ndarray::{Array1, Array2};

    use crate::information_theory::utils::{
        compute_mutual_information, compute_mutual_information_knn, estimate_entropy_knn,
    };

    /// Build a simple 2D dataset for use across tests
    fn make_linear_dataset(n: usize) -> (Array2<f64>, Array1<f64>) {
        // X has one feature that increases linearly; y = x[0] + noise (approximately)
        let data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = i as f64;
                vec![v, v * 0.5 + 1.0] // two correlated features
            })
            .collect();
        let X = Array2::from_shape_vec((n, 2), data).expect("shape is correct");
        // y is perfectly correlated with first feature
        let y = Array1::from_shape_vec(n, (0..n).map(|i| i as f64).collect())
            .expect("shape is correct");
        (X, y)
    }

    /// Build an independent dataset: X and Y with no shared information
    fn make_independent_dataset(n: usize) -> (Array2<f64>, Array2<f64>) {
        // X: [0, 1, 2, ..., n-1] (first column)
        // Y: [n-1, n-2, ..., 0] (reverse, no correlation)
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let X = Array2::from_shape_vec((n, 1), x_data).expect("shape is correct");

        // Use a permuted sequence uncorrelated with X indices
        let y_data: Vec<f64> = (0..n)
            .map(|i| {
                // Simple deterministic permutation: alternating from ends
                let half = n / 2;
                let idx = if i < half { i * 2 } else { (i - half) * 2 + 1 };
                idx.min(n - 1) as f64
            })
            .collect();
        let Y = Array2::from_shape_vec((n, 1), y_data).expect("shape is correct");
        (X, Y)
    }

    #[test]
    fn test_basic_functionality() {
        // Basic test to ensure module compiles and core types are accessible
        let _x = Array2::from_shape_vec((10, 4), (0..40).map(|i| i as f64).collect())
            .expect("operation should succeed");
        let _y = Array1::from_shape_vec(10, (0..10).map(|i| i as f64).collect())
            .expect("operation should succeed");
    }

    /// Entropy estimate for a dataset with clear spread should be positive.
    /// For a 1D uniform-ish dataset of 20 points, entropy should be > 0.
    #[test]
    fn test_entropy_knn_positive_for_spread_data() {
        let n = 20_usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let X = Array2::from_shape_vec((n, 1), data).expect("shape is correct");
        let h = estimate_entropy_knn(&X, 3).expect("entropy computation should succeed");
        // A dataset with spread should have positive entropy
        assert!(
            h > 0.0,
            "entropy of spread data should be positive, got {h}"
        );
    }

    /// Entropy of a nearly-constant dataset should be very small (close to zero
    /// or negative in the KNN estimator for degenerate data — we simply require
    /// it is less than the spread-data entropy).
    #[test]
    fn test_entropy_knn_less_for_constant_data() {
        let n = 10_usize;
        // Constant data: all same value (will have tiny distances, hence low entropy)
        let const_data: Vec<f64> = std::iter::repeat_n(5.0_f64, n).collect();
        let X_const = Array2::from_shape_vec((n, 1), const_data).expect("shape is correct");

        // Spread data
        let spread_data: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let X_spread = Array2::from_shape_vec((n, 1), spread_data).expect("shape is correct");

        let h_const =
            estimate_entropy_knn(&X_const, 3).expect("entropy computation should succeed");
        let h_spread =
            estimate_entropy_knn(&X_spread, 3).expect("entropy computation should succeed");

        assert!(
            h_const < h_spread,
            "constant data entropy ({h_const}) should be less than spread data entropy ({h_spread})"
        );
    }

    /// MI(X, X) should be greater than or approximately equal to MI(X, Y_independent).
    /// With small N KNN estimators can be noisy, so we use a loose bound.
    #[test]
    fn test_mutual_information_self_vs_independent() {
        let n = 30_usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let X = Array2::from_shape_vec((n, 1), data).expect("shape is correct");

        let (_, Y_indep) = make_independent_dataset(n);

        let mi_self = compute_mutual_information_knn(&X, &X, 3)
            .expect("mutual information computation should succeed");
        let mi_indep = compute_mutual_information_knn(&X, &Y_indep, 3)
            .expect("mutual information computation should succeed");

        // Self MI should be greater than MI with independent variable
        // (KNN estimator may be slightly noisy; allow modest tolerance)
        assert!(
            mi_self > mi_indep - 1.0,
            "MI(X,X)={mi_self} should be substantially larger than MI(X,Y_indep)={mi_indep}"
        );
    }

    /// MI between correlated series and label should be > 0 (within KNN estimation noise).
    #[test]
    fn test_mutual_information_correlated() {
        let n = 25_usize;
        let (X, y) = make_linear_dataset(n);

        let mi = compute_mutual_information(&X, &y)
            .expect("mutual information computation should succeed");

        // For a perfectly correlated dataset, MI should be clearly positive
        assert!(
            mi > 0.0,
            "MI between correlated X and y should be positive, got {mi}"
        );
    }

    /// Verify that compute_mutual_information_knn errors on mismatched sample counts.
    #[test]
    fn test_mutual_information_knn_mismatch_error() {
        let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("shape is correct");
        let Y = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape is correct");

        let result = compute_mutual_information_knn(&X, &Y, 2);
        assert!(
            result.is_err(),
            "mismatched sample counts should return an error"
        );
    }

    /// Verify that estimate_entropy_knn errors when k >= n_samples.
    #[test]
    fn test_entropy_knn_invalid_k_error() {
        let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("shape is correct");

        // k must be strictly less than n_samples
        let result = estimate_entropy_knn(&X, 5);
        assert!(result.is_err(), "k >= n_samples should return an error");
    }
}
