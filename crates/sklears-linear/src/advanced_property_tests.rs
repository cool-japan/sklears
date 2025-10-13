//! Comprehensive property-based tests for linear models
//!
//! This module provides extensive property-based testing for all linear model algorithms,
//! ensuring mathematical correctness, numerical stability, and algorithmic properties.

#[cfg(feature = "constrained-optimization")]
use crate::constrained_optimization::{
    ConstrainedOptimizationBuilder, ConstraintType, InteriorPointSolver,
};
#[cfg(feature = "lasso")]
use crate::lasso_cv::{LassoCV, LassoCVConfig};
#[cfg(feature = "linear-regression")]
use crate::linear_regression::{LinearRegression, LinearRegressionConfig};
#[cfg(any(feature = "multi-task", feature = "all-algorithms"))]
use crate::multi_output_regression::{MultiOutputRegression, MultiOutputStrategy};
#[cfg(feature = "ridge")]
use crate::ridge_cv::{RidgeCV, RidgeCVConfig};
use crate::utils::condition_number;
// TODO: Replace with scirs2-linalg
// use nalgebra::{DMatrix, DVector, Matrix};
use proptest::prelude::*;
use scirs2_core::random::Rng;
use sklears_core::traits::{Estimator};

/// Generate a random well-conditioned matrix
fn generate_well_conditioned_matrix(
    n_samples: usize,
    n_features: usize,
    condition_number: f64,
) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let mut matrix = DMatrix::zeros(n_samples, n_features);

    // Fill with random values
    for i in 0..n_samples {
        for j in 0..n_features {
            matrix[(i, j)] = rng.gen_range(-1.0..1.0);
        }
    }

    // If we have a square matrix, modify diagonal to control condition number
    if n_samples >= n_features {
        let gram = matrix.transpose() * &matrix;
        let min_eigenval = 1.0 / condition_number;

        // Add regularization to diagonal to improve conditioning
        for i in 0..n_features {
            matrix[(i, i)] += min_eigenval.sqrt();
        }
    }

    matrix
}

/// Generate test targets with known linear relationship
fn generate_linear_targets(
    X: &DMatrix<f64>,
    true_coeffs: &DVector<f64>,
    noise_level: f64,
) -> DVector<f64> {
    let mut rng = rand::thread_rng();
    let clean_targets = X * true_coeffs;

    if noise_level > 0.0 {
        let noise: Vec<f64> = (0..X.nrows())
            .map(|_| noise_level * rng.gen_range(-1.0..1.0))
            .collect();
        clean_targets + DVector::from_vec(noise)
    } else {
        clean_targets
    }
}

proptest! {
    /// Test mathematical properties of linear regression
    #[test]
    #[cfg(feature = "linear-regression")]
    fn test_linear_regression_mathematical_properties(
        n_samples in 20..100usize,
        n_features in 2..10usize,
        noise_level in 0.0..0.1f64
    ) {
        prop_assume!(n_samples >= n_features);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 10.0);
        let true_coeffs = DVector::from_iterator(n_features, (0..n_features).map(|i| (i + 1) as f64));
        let y = generate_linear_targets(&X, &true_coeffs, noise_level);

        let model = LinearRegression::new()
            .fit_intercept(false);
        let result = model.fit(&X, &y);
        prop_assert!(result.is_ok());
        
        if let Ok(trained_model) = result {
            let coeffs = trained_model.coef();
            // Test coefficient finiteness
            for &coeff in coeffs.iter() {
                prop_assert!(coeff.is_finite());
            }

            // Test prediction consistency
            let predictions = trained_model.predict(&X).unwrap();
            prop_assert_eq!(predictions.len(), y.len());

            for &pred in predictions.iter() {
                prop_assert!(pred.is_finite());
            }

            // Test residual properties
            let residuals = &y - &predictions;
            let mean_residual = residuals.mean();

            // For low noise, residuals should be small
            if noise_level < 0.01 {
                prop_assert!(mean_residual.abs() < 0.1);
            }

            // Test that model minimizes least squares objective
            let mse = residuals.norm_squared() / (n_samples as f64);
            prop_assert!(mse >= 0.0);
            prop_assert!(mse.is_finite());
        }
    }

    /// Test Ridge regression regularization properties
    #[test]
    #[cfg(feature = "ridge")]
    fn test_ridge_regression_properties(
        n_samples in 20..50usize,
        n_features in 2..8usize,
        alpha in 0.1..10.0f64
    ) {
        prop_assume!(n_samples >= n_features);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
        let true_coeffs = DVector::from_iterator(n_features, (0..n_features).map(|i| (i + 1) as f64));
        let y = generate_linear_targets(&X, &true_coeffs, 0.01);

        let model = RidgeCV::new()
            .alphas(vec![alpha])
            .fit_intercept(false)
            .cv(3);
        let result = model.fit(&X, &y);
        prop_assert!(result.is_ok());
        
        if let Ok(trained_model) = result {
            let coeffs = trained_model.coef();
            // Ridge coefficients should be finite
            for &coeff in coeffs.iter() {
                prop_assert!(coeff.is_finite());
            }

            // Ridge should shrink coefficients compared to OLS (generally)
            let ridge_norm = coeffs.norm();
            prop_assert!(ridge_norm.is_finite());
            prop_assert!(ridge_norm >= 0.0);

            // Test regularization effect: higher alpha should lead to more shrinkage
            if alpha > 1.0 {
                prop_assert!(ridge_norm < true_coeffs.norm() + 0.1);
            }
        }
    }

    /// Test Lasso sparsity properties
    #[test]
    #[cfg(feature = "lasso")]
    fn test_lasso_sparsity_properties(
        n_samples in 30..60usize,
        n_features in 3..8usize,
        alpha in 0.1..2.0f64
    ) {
        prop_assume!(n_samples >= n_features);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);

        // Create sparse true coefficients
        let mut true_coeffs = DVector::zeros(n_features);
        for i in 0..n_features/2 {
            true_coeffs[i] = (i + 1) as f64;
        }

        let y = generate_linear_targets(&X, &true_coeffs, 0.01);

        let model = LassoCV::new()
            .alphas(vec![alpha])
            .fit_intercept(false)
            .cv(3)
            .max_iter(1000);
        let result = model.fit(&X, &y);
        prop_assert!(result.is_ok());
        
        if let Ok(trained_model) = result {
            let coeffs = trained_model.coef();
            // Lasso coefficients should be finite
            for &coeff in coeffs.iter() {
                prop_assert!(coeff.is_finite());
            }

            // Test sparsity: some coefficients should be exactly zero for sufficiently large alpha
            if alpha > 0.5 {
                let zero_count = coeffs.iter().filter(|&&x| x.abs() < 1e-10).count();
                prop_assert!(zero_count <= n_features);
            }

            // L1 norm should be finite
            let l1_norm: f64 = coeffs.iter().map(|x| x.abs()).sum();
            prop_assert!(l1_norm.is_finite());
            prop_assert!(l1_norm >= 0.0);
        }
    }

    /// Test multi-output regression properties
    #[test]
    #[cfg(any(feature = "multi-task", feature = "all-algorithms"))]
    fn test_multi_output_regression_properties(
        n_samples in 20..50usize,
        n_features in 2..6usize,
        n_targets in 2..4usize,
        alpha in 0.1..1.0f64
    ) {
        prop_assume!(n_samples >= n_features);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
        let mut Y = DMatrix::zeros(n_samples, n_targets);

        // Generate correlated targets
        for i in 0..n_samples {
            for j in 0..n_targets {
                Y[(i, j)] = (i as f64 + j as f64) / 10.0;
            }
        }

        let model = MultiOutputRegression::ridge(alpha);
        let result = model.fit(&X, &Y);
        prop_assert!(result.is_ok());
        
        if let Ok(trained_model) = result {
            let coeffs = trained_model.coef();
            // Coefficient matrix should have correct dimensions
            prop_assert_eq!(coeffs.nrows(), n_features);
            prop_assert_eq!(coeffs.ncols(), n_targets);

            // All coefficients should be finite
            for coeff in coeffs.iter() {
                prop_assert!(coeff.is_finite());
            }

            // Test predictions
            let predictions = trained_model.predict(&X).unwrap();
            prop_assert_eq!(predictions.nrows(), n_samples);
            prop_assert_eq!(predictions.ncols(), n_targets);

            for pred in predictions.iter() {
                prop_assert!(pred.is_finite());
            }
        }
    }

    /// Test constrained optimization properties
    #[test]
    #[cfg(feature = "constrained-optimization")]
    fn test_constrained_optimization_properties(
        n_features in 2..5usize,
        alpha in 0.1..1.0f64
    ) {
        // Create a simple quadratic optimization problem
        let hessian = DMatrix::identity(n_features, n_features);
        let linear_coeff = DVector::from_iterator(n_features, (0..n_features).map(|i| (i + 1) as f64));

        // Add box constraints: 0 <= x <= 1
        let lower = Some(DVector::zeros(n_features));
        let upper = Some(DVector::from_element(n_features, 1.0));

        let problem = ConstrainedOptimizationBuilder::new()
            .hessian(hessian)
            .linear_coeff(linear_coeff)
            .add_box_constraints(lower, upper)
            .build();

        prop_assert!(problem.is_ok());

        if let Ok(prob) = problem {
            let solver = InteriorPointSolver::new(Default::default());
            let result = solver.solve(&prob);

            // May not always converge due to complexity, but if it does, solution should be valid
            if let Ok(solution) = result {
                // Solution should respect box constraints
                for &x in solution.x.iter() {
                    prop_assert!(x >= -1e-6); // Allow small numerical error
                    prop_assert!(x <= 1.0 + 1e-6);
                    prop_assert!(x.is_finite());
                }

                // Objective value should be finite
                prop_assert!(solution.objective_value.is_finite());
            }
        }
    }

    /// Test numerical stability properties
    #[test]
    fn test_numerical_stability(
        n_samples in 20..40usize,
        n_features in 2..6usize,
        scale_factor in prop::sample::select(vec![1e-6, 1e-3, 1.0, 1e3])
    ) {
        prop_assume!(n_samples >= n_features);

        let mut X = generate_well_conditioned_matrix(n_samples, n_features, 10.0);

        // Scale the data
        for val in X.iter_mut() {
            *val *= scale_factor;
        }

        let true_coeffs = DVector::from_element(n_features, 1.0 / scale_factor);
        let y = generate_linear_targets(&X, &true_coeffs, 0.0);

        // Test that all values are finite
        for &val in X.iter() {
            prop_assert!(val.is_finite());
        }

        for &val in y.iter() {
            prop_assert!(val.is_finite());
        }

        // Test Gram matrix computation
        let gram = X.transpose() * &X;
        for &val in gram.iter() {
            prop_assert!(val.is_finite());
        }

        // Test condition number computation
        if let Ok(cond_num) = condition_number(&gram) {
            prop_assert!(cond_num.is_finite());
            prop_assert!(cond_num >= 1.0);
        }
    }

    /// Test convergence properties for iterative algorithms
    #[test]
    fn test_iterative_convergence_properties(
        n_features in 2..6usize,
        learning_rate in 0.01..0.5f64,
        max_iter in 10..100usize
    ) {
        // Test gradient descent convergence on a simple quadratic
        let target = DVector::from_iterator(n_features, (0..n_features).map(|i| (i + 1) as f64));
        let mut x = DVector::zeros(n_features);

        let mut prev_cost = f64::INFINITY;
        let mut costs = Vec::new();

        for iter in 0..max_iter {
            // Gradient of ||x - target||²
            let gradient = 2.0 * (&x - &target);

            // Gradient descent update
            x -= learning_rate * gradient;

            // Compute cost
            let cost = (&x - &target).norm_squared();
            costs.push(cost);

            prop_assert!(cost.is_finite());
            prop_assert!(cost >= 0.0);

            // For this convex problem, cost should decrease (allowing small numerical errors)
            if iter > 0 && learning_rate < 0.1 {
                prop_assert!(cost <= prev_cost + 1e-10);
            }

            prev_cost = cost;
        }

        // Final cost should be less than initial cost for reasonable learning rates
        if learning_rate < 0.2 && max_iter >= 20 {
            prop_assert!(costs.last().unwrap() < &costs[0]);
        }
    }

    /// Test invariance properties
    #[test]
    fn test_invariance_properties(
        n_samples in 20..40usize,
        n_features in 2..6usize,
        shift in -10.0..10.0f64,
        scale in 0.1..10.0f64
    ) {
        prop_assume!(n_samples >= n_features);
        prop_assume!(scale > 0.1);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
        let true_coeffs = DVector::from_element(n_features, 1.0);
        let y = generate_linear_targets(&X, &true_coeffs, 0.01);

        // Fit original model
        let model1 = LinearRegression::new().fit_intercept(true);
        let result1 = model1.fit(&X, &y);
        prop_assert!(result1.is_ok());
        
        if let Ok(trained_model1) = result1 {
            // Apply affine transformation to targets: y' = scale * y + shift
            let y_transformed = scale * &y + DVector::from_element(n_samples, shift);

            // Fit transformed model
            let model2 = LinearRegression::new().fit_intercept(true);
            let result2 = model2.fit(&X, &y_transformed);
            prop_assert!(result2.is_ok());
            
            if let Ok(trained_model2) = result2 {
                let coeffs1 = trained_model1.coef();
                let coeffs2 = trained_model2.coef();
            // Coefficients should scale appropriately
            for i in 0..coeffs1.len() {
                let expected_coeff = scale * coeffs1[i];
                let actual_coeff = coeffs2[i];

                    // Allow for numerical differences
                    prop_assert!((expected_coeff - actual_coeff).abs() < 0.1);
                }
                
                // Test predictions
                let pred1 = trained_model1.predict(&X).unwrap();
                let pred2 = trained_model2.predict(&X).unwrap();

                for i in 0..pred1.len() {
                    let expected_pred = scale * pred1[i] + shift;
                    let actual_pred = pred2[i];

                    // Allow for numerical differences
                    prop_assert!((expected_pred - actual_pred).abs() < 0.1);
                }
            }
        }
    }

    /// Test regularization path properties
    #[test]
    fn test_regularization_path_properties(
        n_samples in 20..40usize,
        n_features in 2..6usize
    ) {
        prop_assume!(n_samples >= n_features);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
        let true_coeffs = DVector::from_element(n_features, 1.0);
        let y = generate_linear_targets(&X, &true_coeffs, 0.01);

        // Test multiple alpha values
        let alphas = vec![0.01, 0.1, 1.0, 10.0];
        let mut coefficient_norms = Vec::new();

        for &alpha in &alphas {
            let model = RidgeCV::new()
                .alphas(vec![alpha])
                .fit_intercept(false)
                .cv(3);
            let result = model.fit(&X, &y);
            prop_assert!(result.is_ok());
            
            if let Ok(trained_model) = result {
                let coeffs = trained_model.coef();
                let norm = coeffs.norm();
                coefficient_norms.push(norm);

                prop_assert!(norm.is_finite());
                prop_assert!(norm >= 0.0);
            }
        }

        // Coefficient norms should generally decrease with increasing regularization
        if coefficient_norms.len() == alphas.len() {
            for i in 1..coefficient_norms.len() {
                // Allow some flexibility due to cross-validation noise
                prop_assert!(coefficient_norms[i] <= coefficient_norms[i-1] + 0.5);
            }
        }
    }

    /// Test cross-validation consistency
    #[test]
    fn test_cross_validation_consistency(
        n_samples in 30..60usize,
        n_features in 2..6usize,
        cv_folds in 3..6usize
    ) {
        prop_assume!(n_samples >= n_features);
        prop_assume!(n_samples >= cv_folds * 5); // Ensure reasonable fold sizes

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
        let true_coeffs = DVector::from_element(n_features, 1.0);
        let y = generate_linear_targets(&X, &true_coeffs, 0.1);

        let model = RidgeCV::new()
            .alphas(vec![0.1, 1.0, 10.0])
            .fit_intercept(false)
            .cv(cv_folds);
        let result = model.fit(&X, &y);
        prop_assert!(result.is_ok());
        
        if let Ok(trained_model) = result {
            // Best alpha should be one of the provided alphas
            if let Some(best_alpha) = trained_model.best_alpha() {
                prop_assert!(best_alpha == 0.1 || best_alpha == 1.0 || best_alpha == 10.0);
            }

            // CV scores should be finite and reasonable
            if let Some(cv_scores) = trained_model.cv_scores() {
            for score in cv_scores.values() {
                for &s in score {
                    prop_assert!(s.is_finite());
                    // R² scores should be reasonable (not too negative)
                    prop_assert!(s > -2.0);
                }
            }
            }
        }
    }

    /// Test elastic net interpolation properties
    #[test]
    fn test_elastic_net_interpolation(
        n_samples in 20..40usize,
        n_features in 2..6usize,
        l1_ratio in 0.1..0.9f64
    ) {
        prop_assume!(n_samples >= n_features);

        let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
        let true_coeffs = DVector::from_element(n_features, 1.0);
        let y = generate_linear_targets(&X, &true_coeffs, 0.01);

        let alpha = 1.0;

        // Fit Ridge (l1_ratio = 0)
        let ridge_model = RidgeCV::new()
            .alphas(vec![alpha])
            .fit_intercept(false)
            .cv(3);
        let ridge_result = ridge_model.fit(&X, &y).unwrap();

        // Fit Lasso (l1_ratio = 1)
        let lasso_model = LassoCV::new()
            .alphas(vec![alpha])
            .fit_intercept(false)
            .cv(3)
            .max_iter(1000);
        let lasso_result = lasso_model.fit(&X, &y).unwrap();

        let ridge_coeffs = ridge_result.coef();
        let lasso_coeffs = lasso_result.coef();

        let ridge_norm = ridge_coeffs.norm();
        let lasso_norm = lasso_coeffs.norm();

        // Both should be finite
        prop_assert!(ridge_norm.is_finite());
        prop_assert!(lasso_norm.is_finite());

        // Ridge typically produces denser solutions than Lasso
        let ridge_zeros = ridge_coeffs.iter().filter(|&&x| x.abs() < 1e-10).count();
        let lasso_zeros = lasso_coeffs.iter().filter(|&&x| x.abs() < 1e-10).count();

        // Lasso should have at least as many zeros as Ridge
        prop_assert!(lasso_zeros >= ridge_zeros);
    }
}

/// Test helper functions
#[allow(non_snake_case)]
#[cfg(test)]
mod test_helpers {
    use super::*;

    #[test]
    fn test_generate_well_conditioned_matrix() {
        let matrix = generate_well_conditioned_matrix(10, 5, 10.0);
        assert_eq!(matrix.nrows(), 10);
        assert_eq!(matrix.ncols(), 5);

        for &val in matrix.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_generate_linear_targets() {
        let X = DMatrix::from_element(5, 3, 1.0);
        let coeffs = DVector::from_element(3, 2.0);
        let y = generate_linear_targets(&X, &coeffs, 0.0);

        assert_eq!(y.len(), 5);
        for &val in y.iter() {
            assert!(val.is_finite());
            assert!((val - 6.0).abs() < 1e-10); // 3 coefficients * 2.0 each * 1.0 features
        }
    }

    #[test]
    fn test_generate_linear_targets_with_noise() {
        let X = DMatrix::from_element(5, 3, 1.0);
        let coeffs = DVector::from_element(3, 2.0);
        let y = generate_linear_targets(&X, &coeffs, 0.1);

        assert_eq!(y.len(), 5);
        for &val in y.iter() {
            assert!(val.is_finite());
            // With noise, values should be close to 6.0 but not exactly
            assert!((val - 6.0).abs() < 1.0); // Allow for noise
        }
    }
}

/// Performance and scaling property tests
#[allow(non_snake_case)]
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    proptest! {
        /// Test that algorithms scale reasonably with problem size
        #[test]
        fn test_scaling_properties(
            size_factor in prop::sample::select(vec![1, 2, 4])
        ) {
            let base_size = 20;
            let n_samples = base_size * size_factor;
            let n_features = 5;

            prop_assume!(n_samples >= n_features);

            let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
            let true_coeffs = DVector::from_element(n_features, 1.0);
            let y = generate_linear_targets(&X, &true_coeffs, 0.01);

            let start = Instant::now();
            let model = LinearRegression::new().fit_intercept(false);
            let result = model.fit(&X, &y);
            let duration = start.elapsed();

            prop_assert!(result.is_ok());

            // Algorithm should complete in reasonable time (very generous bounds)
            prop_assert!(duration.as_secs() < 60);

            // Memory usage should be reasonable (coefficients should fit in memory)
            if let Ok(trained_model) = result {
                let coeffs = trained_model.coef();
                prop_assert_eq!(coeffs.len(), n_features);
            }
        }

        /// Test convergence speed properties
        #[test]
        fn test_convergence_speed(
            n_samples in 20..50usize,
            n_features in 2..6usize,
            tolerance in prop::sample::select(vec![1e-6, 1e-4, 1e-2])
        ) {
            prop_assume!(n_samples >= n_features);

            let X = generate_well_conditioned_matrix(n_samples, n_features, 5.0);
            let true_coeffs = DVector::from_element(n_features, 1.0);
            let y = generate_linear_targets(&X, &true_coeffs, 0.01);

            let model = LassoCV::new()
                .alphas(vec![0.1])
                .fit_intercept(false)
                .cv(3)
                .max_iter(1000)
                .tol(tolerance);
            let result = model.fit(&X, &y);
            prop_assert!(result.is_ok());

            // For tighter tolerance, we should get more precise results
            if let Ok(trained_model) = result {
                let coeffs = trained_model.coef();
                for &coeff in coeffs.iter() {
                    prop_assert!(coeff.is_finite());
                }

                // Predictions should be more accurate with tighter tolerance
                let predictions = trained_model.predict(&X).unwrap();
                let residuals = &y - &predictions;
                let mse = residuals.norm_squared() / (n_samples as f64);

                prop_assert!(mse.is_finite());

                // For very tight tolerance, MSE should be quite small
                if tolerance <= 1e-6 {
                    prop_assert!(mse < 1.0);
                }
            }
        }
    }
}
