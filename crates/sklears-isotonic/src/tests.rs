//! Test suite for basic isotonic regression functionality

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use crate::*;
    use approx::assert_abs_diff_eq;
    use proptest::{
        prelude::{any, prop},
        proptest,
    };
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};

    #[test]
    fn test_isotonic_regression_increasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let iso = IsotonicRegression::new();
        let fitted = iso.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_isotonic_regression_decreasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0];

        let iso = IsotonicRegression::new().increasing(false);
        let fitted = iso.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_isotonic_regression_function() {
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];
        let fitted = isotonic_regression(&y, true);

        // Check that result is increasing
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_weighted_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];
        let weights = array![1.0, 1.0, 10.0, 1.0, 1.0]; // Heavy weight on the third point

        let iso = IsotonicRegression::new();
        let fitted = iso.fit_weighted(&x, &y, Some(&weights)).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // The heavily weighted point should have influence
        assert!(predictions[2] <= 2.1); // Should be close to 2.0 due to heavy weight
    }

    #[test]
    fn test_weighted_isotonic_regression_function() {
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];
        let weights = array![1.0, 1.0, 5.0, 1.0, 1.0];
        let fitted = isotonic_regression_weighted(&y, &weights, true);

        // Check that result is increasing
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_invalid_weights() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let weights = array![1.0, -1.0, 1.0]; // Negative weight

        let iso = IsotonicRegression::new();
        let result = iso.fit_weighted(&x, &y, Some(&weights));
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_length_mismatch() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let weights = array![1.0, 1.0]; // Wrong length

        let iso = IsotonicRegression::new();
        let result = iso.fit_weighted(&x, &y, Some(&weights));
        assert!(result.is_err());
    }

    #[test]
    fn test_l1_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 100.0, 2.0, 4.0, 5.0]; // Contains outlier

        let iso = IsotonicRegression::new().loss(LossFunction::AbsoluteLoss);
        let fitted = iso.fit_weighted(&x, &y, None).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // L1 should be more robust to the outlier at position 1
        assert!(predictions[1] < 50.0); // Should not be pulled too much by outlier
    }

    #[test]
    fn test_huber_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 100.0, 2.0, 4.0, 5.0]; // Contains outlier

        let iso = IsotonicRegression::new().loss(LossFunction::HuberLoss { delta: 1.0 });
        let fitted = iso.fit_weighted(&x, &y, None).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Huber should be more robust to the outlier at position 1
        assert!(predictions[1] < 70.0); // Should not be pulled too much by outlier
    }

    #[test]
    fn test_quantile_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 100.0, 2.0, 4.0, 5.0]; // Contains outlier

        // Test median regression (quantile = 0.5)
        let iso = IsotonicRegression::new().loss(LossFunction::QuantileLoss { quantile: 0.5 });
        let fitted = iso.fit_weighted(&x, &y, None).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }

        // Quantile regression should be robust to the outlier
        assert!(predictions[1] < 50.0); // Should not be pulled too much by outlier
    }

    #[test]
    fn test_bounds() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![-1.0, 3.0, 2.0, 4.0, 10.0];

        let iso = IsotonicRegression::new().bounds(0.0, 8.0);
        let fitted = iso.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are increasing and within bounds
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
            assert!(predictions[i] >= 0.0);
            assert!(predictions[i] <= 8.0);
        }
    }

    #[test]
    fn test_weighted_median() {
        // Test simple case
        let values = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)];
        let median = crate::pav::weighted_median(&values);
        assert_abs_diff_eq!(median, 2.0, epsilon = 1e-10);

        // Test weighted case
        let values = vec![(1.0, 1.0), (2.0, 10.0), (3.0, 1.0)];
        let median = crate::pav::weighted_median(&values);
        assert_abs_diff_eq!(median, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_input() {
        let x = array![];
        let y = array![];

        let iso = IsotonicRegression::new();
        let result = iso.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_point() {
        let x = array![1.0];
        let y = array![2.0];

        let iso = IsotonicRegression::new();
        let fitted = iso.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions[0], 2.0);
    }

    #[test]
    fn test_already_monotonic() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0]; // Already increasing

        let iso = IsotonicRegression::new();
        let fitted = iso.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Should be very close to the original values
        for i in 0..predictions.len() {
            assert_abs_diff_eq!(predictions[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_constraint_validation() {
        let increasing = array![1.0, 2.0, 3.0, 4.0];
        let not_increasing = array![1.0, 3.0, 2.0, 4.0];

        assert!(validate_monotonicity(
            &increasing,
            MonotonicityConstraint::Increasing,
            1e-10
        ));
        assert!(!validate_monotonicity(
            &not_increasing,
            MonotonicityConstraint::Increasing,
            1e-10
        ));

        let decreasing = array![4.0, 3.0, 2.0, 1.0];
        let not_decreasing = array![4.0, 2.0, 3.0, 1.0];

        assert!(validate_monotonicity(
            &decreasing,
            MonotonicityConstraint::Decreasing,
            1e-10
        ));
        assert!(!validate_monotonicity(
            &not_decreasing,
            MonotonicityConstraint::Decreasing,
            1e-10
        ));
    }

    #[test]
    fn test_global_constraint_compatibility() {
        let values = array![1.0, 3.0, 2.0, 4.0, 3.5];

        // Test Global constraint with increasing=true
        let result = apply_global_constraint(
            &values,
            MonotonicityConstraint::Global { increasing: true },
            None,
        )
        .unwrap();
        for i in 0..result.len() - 1 {
            assert!(result[i] <= result[i + 1]);
        }

        // Test Global constraint with increasing=false
        let result = apply_global_constraint(
            &values,
            MonotonicityConstraint::Global { increasing: false },
            None,
        )
        .unwrap();
        for i in 0..result.len() - 1 {
            assert!(result[i] >= result[i + 1]);
        }
    }

    #[test]
    fn test_utility_functions() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // 2*x

        let corr = pearson_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);

        let spearman_corr = spearman_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(spearman_corr, 1.0, epsilon = 1e-10);

        let max_idx = argmax(&y).unwrap();
        assert_eq!(max_idx, 4); // Last element

        let mse = mean_squared_error(&x, &y).unwrap();
        assert!(mse > 0.0);
    }

    // Property-based test for monotonicity preservation
    proptest! {
        #[test]
        fn prop_isotonic_preserves_monotonicity(
            y in prop::collection::vec(any::<f64>(), 3..20)
        ) {
            let y_array: scirs2_core::ndarray::Array1<f64> = y.into();

            // Test increasing constraint
            if let Ok(result) = pool_adjacent_violators_l2(&y_array, None, true) {
                for i in 0..result.len() - 1 {
                    assert!(result[i] <= result[i + 1] + 1e-10);
                }
            }

            // Test decreasing constraint
            if let Ok(result) = pool_adjacent_violators_l2(&y_array, None, false) {
                for i in 0..result.len() - 1 {
                    assert!(result[i] >= result[i + 1] - 1e-10);
                }
            }
        }
    }
}
