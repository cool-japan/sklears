//! Sparse Linear Regression implementations
//!
//! This module provides memory-efficient linear regression models for sparse data,
//! offering significant performance improvements for high-dimensional sparse datasets.

#[cfg(feature = "sparse")]
use crate::sparse::{SparseCoordinateDescentSolver, SparseMatrix, SparseMatrixCSR};
use crate::{LinearRegression, LinearRegressionConfig};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearnContext, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};
use std::marker::PhantomData;

/// Configuration for Sparse Linear Regression
#[derive(Debug, Clone)]
pub struct SparseLinearRegressionConfig {
    /// Base linear regression configuration
    pub base_config: LinearRegressionConfig,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Sparsity threshold for determining zero coefficients
    pub sparsity_threshold: Float,
    /// Whether to automatically convert dense matrices to sparse
    pub auto_sparse_conversion: bool,
    /// Minimum sparsity ratio to use sparse algorithms
    pub min_sparsity_ratio: f64,
}

impl Default for SparseLinearRegressionConfig {
    fn default() -> Self {
        Self {
            base_config: LinearRegressionConfig::default(),
            fit_intercept: true,
            sparsity_threshold: 1e-10,
            auto_sparse_conversion: true,
            min_sparsity_ratio: 0.3, // Use sparse if <30% non-zero
        }
    }
}

impl Validate for SparseLinearRegressionConfig {
    fn validate(&self) -> Result<()> {
        // Note: LinearRegressionConfig doesn't implement Validate trait yet
        // TODO: Add validation for base_config once Validate trait is implemented

        // Validate sparsity threshold
        ValidationRules::new("sparsity_threshold")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.sparsity_threshold)?;

        // Validate min_sparsity_ratio
        ValidationRules::new("min_sparsity_ratio")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 })
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.min_sparsity_ratio)?;

        Ok(())
    }
}

impl ConfigValidation for SparseLinearRegressionConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        if self.min_sparsity_ratio > 0.5 {
            log::warn!(
                "High min_sparsity_ratio ({:.2}) may underutilize sparse optimizations",
                self.min_sparsity_ratio
            );
        }

        if self.sparsity_threshold > 1e-6 {
            log::warn!(
                "Large sparsity_threshold ({:.2e}) may ignore small but significant coefficients",
                self.sparsity_threshold
            );
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if !self.fit_intercept && self.auto_sparse_conversion {
            warnings.push(
                "Sparse conversion without intercept fitting may affect performance".to_string(),
            );
        }

        warnings
    }
}

/// Sparse Linear Regression model that can handle both sparse and dense inputs
#[derive(Debug, Clone)]
pub struct SparseLinearRegression<State = Untrained> {
    config: SparseLinearRegressionConfig,
    state: PhantomData<State>,
    // Trained state fields
    coefficients_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    is_sparse_fitted_: Option<bool>,
    n_features_: Option<usize>,
}

impl SparseLinearRegression<Untrained> {
    /// Create a new sparse linear regression model
    pub fn new() -> Self {
        Self {
            config: SparseLinearRegressionConfig::default(),
            state: PhantomData,
            coefficients_: None,
            intercept_: None,
            is_sparse_fitted_: None,
            n_features_: None,
        }
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: Float) -> Self {
        self.config.sparsity_threshold = threshold;
        self
    }

    /// Set automatic sparse conversion
    pub fn auto_sparse_conversion(mut self, auto_convert: bool) -> Self {
        self.config.auto_sparse_conversion = auto_convert;
        self
    }

    /// Set minimum sparsity ratio for using sparse algorithms
    pub fn min_sparsity_ratio(mut self, ratio: f64) -> Self {
        self.config.min_sparsity_ratio = ratio;
        self
    }
}

impl Default for SparseLinearRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SparseLinearRegression<Untrained> {
    type Config = SparseLinearRegressionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Fit implementation for dense matrices
impl Fit<Array2<Float>, Array1<Float>> for SparseLinearRegression<Untrained> {
    type Fitted = SparseLinearRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate configuration
        self.config.validate_config().fit_context(
            "SparseLinearRegression",
            n_samples,
            n_features,
        )?;

        // Validate data
        use sklears_core::validation::ml;
        ml::validate_supervised_data(x, y).fit_context(
            "SparseLinearRegression",
            n_samples,
            n_features,
        )?;

        #[cfg(feature = "sparse")]
        {
            // Check if we should use sparse algorithms
            let should_use_sparse = if self.config.auto_sparse_conversion {
                crate::sparse::utils::should_use_sparse(
                    x,
                    &crate::sparse::SparseConfig {
                        sparsity_threshold: self.config.sparsity_threshold,
                        min_sparsity_ratio: self.config.min_sparsity_ratio,
                        max_dense_memory_ratio: 0.5,
                    },
                )
            } else {
                false
            };

            if should_use_sparse {
                // Convert to sparse and fit using sparse algorithms
                let x_sparse = SparseMatrixCSR::from_dense(x, self.config.sparsity_threshold);
                self.fit_sparse(&x_sparse, y)
            } else {
                // Use dense algorithms
                self.fit_dense(x, y)
            }
        }

        #[cfg(not(feature = "sparse"))]
        {
            // Sparse feature not available, use dense algorithms
            self.fit_dense(x, y)
        }
    }
}

/// Fit implementation for sparse matrices
#[cfg(feature = "sparse")]
impl Fit<SparseMatrixCSR<Float>, Array1<Float>> for SparseLinearRegression<Untrained> {
    type Fitted = SparseLinearRegression<Trained>;

    fn fit(self, x: &SparseMatrixCSR<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        self.fit_sparse(x, y)
    }
}

impl SparseLinearRegression<Untrained> {
    /// Fit using dense algorithms (fallback)
    fn fit_dense(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<SparseLinearRegression<Trained>> {
        // Use the standard dense linear regression
        let dense_model = LinearRegression::new()
            .fit_intercept(self.config.fit_intercept)
            .fit(x, y)?;

        Ok(SparseLinearRegression {
            config: self.config,
            state: PhantomData,
            coefficients_: Some(dense_model.coef().clone()),
            intercept_: dense_model.intercept(),
            is_sparse_fitted_: Some(false),
            n_features_: Some(x.ncols()),
        })
    }

    /// Fit using sparse algorithms
    #[cfg(feature = "sparse")]
    fn fit_sparse(
        self,
        x: &SparseMatrixCSR<Float>,
        y: &Array1<Float>,
    ) -> Result<SparseLinearRegression<Trained>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} samples but y has {} samples",
                n_samples,
                y.len()
            )));
        }

        // For linear regression, we use least squares which can be solved
        // using normal equations: X^T X β = X^T y
        // For sparse matrices, we solve this using coordinate descent

        // Use sparse coordinate descent solver without regularization
        let alpha = 1e-12; // Minimal regularization for numerical stability
        let solver = SparseCoordinateDescentSolver {
            alpha,
            l1_ratio: 1.0, // Not used for linear regression, but required
            max_iter: 1000,
            tol: 1e-6,
            cyclic: true,
            sparse_config: crate::sparse::SparseConfig::default(),
        };

        // Solve with very small alpha (essentially no regularization)
        let (coefficients, intercept) =
            solver.solve_sparse_lasso(x, y, alpha, self.config.fit_intercept)?;

        Ok(SparseLinearRegression {
            config: self.config,
            state: PhantomData,
            coefficients_: Some(coefficients),
            intercept_: Some(intercept),
            is_sparse_fitted_: Some(true),
            n_features_: Some(n_features),
        })
    }
}

impl SparseLinearRegression<Trained> {
    /// Helper method to get coefficients safely
    fn get_coefficients(&self) -> Result<&Array1<Float>> {
        self.coefficients_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "coefficients".to_string(),
            })
    }

    /// Helper method to get n_features safely
    fn get_n_features(&self) -> Result<usize> {
        self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_features".to_string(),
        })
    }

    /// Get the fitted coefficients
    pub fn coefficients(&self) -> Result<&Array1<Float>> {
        self.get_coefficients()
    }

    /// Get the fitted intercept (if fit_intercept=true)
    pub fn intercept(&self) -> Option<&Float> {
        self.intercept_.as_ref()
    }

    /// Check if the model was fitted using sparse algorithms
    pub fn is_sparse_fitted(&self) -> bool {
        self.is_sparse_fitted_.unwrap_or(false)
    }

    /// Get number of features
    pub fn n_features(&self) -> Result<usize> {
        self.get_n_features()
    }

    /// Get sparsity ratio of the fitted coefficients
    pub fn coefficient_sparsity(&self) -> Result<f64> {
        let coeffs = self.get_coefficients()?;
        let nnz = coeffs
            .iter()
            .filter(|&&c| c.abs() > self.config.sparsity_threshold)
            .count();
        if !coeffs.is_empty() {
            Ok(1.0 - (nnz as f64 / coeffs.len() as f64))
        } else {
            Ok(0.0)
        }
    }
}

/// Prediction for dense inputs
impl Predict<Array2<Float>, Array1<Float>> for SparseLinearRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.get_n_features()?;
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let coeffs = self.get_coefficients()?;
        let mut predictions = x.dot(coeffs);

        // Add intercept if fitted
        if let Some(intercept) = self.intercept_ {
            predictions.mapv_inplace(|p| p + intercept);
        }

        Ok(predictions)
    }
}

/// Prediction for sparse inputs
#[cfg(feature = "sparse")]
impl Predict<SparseMatrixCSR<Float>, Array1<Float>> for SparseLinearRegression<Trained> {
    fn predict(&self, x: &SparseMatrixCSR<Float>) -> Result<Array1<Float>> {
        let n_features = self.get_n_features()?;
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let coeffs = self.get_coefficients()?;
        let mut predictions = x.matvec(coeffs)?;

        // Add intercept if fitted
        if let Some(intercept) = self.intercept_ {
            predictions.mapv_inplace(|p| p + intercept);
        }

        Ok(predictions)
    }
}

#[cfg(all(test, feature = "sparse"))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_linear_regression_dense_input() {
        // Simple linear relationship: y = 2x1 + 3x2 + 1
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],];
        let y = array![3.0, 4.0, 6.0, 8.0]; // 2*1 + 3*0 + 1, 2*0 + 3*1 + 1, etc.

        let model = SparseLinearRegression::new()
            .fit_intercept(true)
            .fit(&x, &y)
            .unwrap();

        let coeffs = model.coefficients().unwrap();
        let intercept = model.intercept().unwrap();

        // Should recover the true parameters
        assert_abs_diff_eq!(coeffs[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(coeffs[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(*intercept, 1.0, epsilon = 1e-10);

        // Test prediction
        let x_test = array![[1.5, 0.5]];
        let y_pred = model.predict(&x_test).unwrap();
        let expected = 2.0 * 1.5 + 3.0 * 0.5 + 1.0; // 5.5
        assert_abs_diff_eq!(y_pred[0], expected, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "Waiting for SciRS2-sparse v0.1.0-rc.2 to be published to crates.io"]
    fn test_sparse_linear_regression_sparse_input() {
        // Create sparse design matrix
        let triplets = vec![
            (0, 0, 1.0),
            (0, 1, 0.0),
            (1, 0, 0.0),
            (1, 1, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (3, 0, 2.0),
            (3, 1, 1.0),
        ];
        let x_sparse = SparseMatrixCSR::from_triplets(4, 2, &triplets).unwrap();
        let y = array![3.0, 4.0, 6.0, 8.0];

        let model = SparseLinearRegression::new()
            .fit_intercept(true)
            .fit(&x_sparse, &y)
            .unwrap();

        assert!(model.is_sparse_fitted());

        let coeffs = model.coefficients().unwrap();
        assert_eq!(coeffs.len(), 2);

        // Test sparse prediction
        let x_test_sparse =
            SparseMatrixCSR::from_triplets(1, 2, &[(0, 0, 1.5), (0, 1, 0.5)]).unwrap();
        let y_pred = model.predict(&x_test_sparse).unwrap();
        assert_eq!(y_pred.len(), 1);
    }

    #[test]
    #[ignore = "Waiting for SciRS2-sparse v0.1.0-rc.2 to be published to crates.io"]
    fn test_sparse_linear_regression_auto_conversion() {
        // Create a very sparse matrix
        let mut x = Array2::zeros((10, 5));
        x[[0, 0]] = 1.0;
        x[[1, 1]] = 1.0;
        x[[5, 2]] = 1.0;
        // Only 3 non-zero elements out of 50

        let y = Array1::ones(10);

        let model = SparseLinearRegression::new()
            .auto_sparse_conversion(true)
            .min_sparsity_ratio(0.8) // Should trigger sparse conversion
            .fit(&x, &y)
            .unwrap();

        // Should use sparse algorithms for very sparse data
        assert!(model.is_sparse_fitted());
    }

    #[test]
    fn test_sparse_linear_regression_no_auto_conversion() {
        // Use a well-conditioned matrix instead of all zeros
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ];
        let y = array![1.0, 2.0, 3.0, 3.0, 5.0];

        let model = SparseLinearRegression::new()
            .auto_sparse_conversion(false)
            .fit(&x, &y)
            .unwrap();

        // Should use dense algorithms when auto conversion is disabled
        assert!(!model.is_sparse_fitted());
    }

    #[test]
    fn test_sparse_linear_regression_validation() {
        let valid_config = SparseLinearRegressionConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_config_neg = SparseLinearRegressionConfig {
            sparsity_threshold: -1.0,
            ..SparseLinearRegressionConfig::default()
        };
        assert!(invalid_config_neg.validate().is_err());

        let invalid_config_ratio = SparseLinearRegressionConfig {
            min_sparsity_ratio: 1.5,
            ..SparseLinearRegressionConfig::default()
        };
        assert!(invalid_config_ratio.validate().is_err());
    }

    #[test]
    fn test_coefficient_sparsity() {
        // Use a better conditioned problem
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.0],
            [1.5, 1.5, 0.7],
            [0.5, 2.5, 0.3],
        ];
        let y = array![1.0, 2.0, 1.8, 1.3];

        let model = SparseLinearRegression::new()
            .sparsity_threshold(1e-3)
            .fit(&x, &y)
            .unwrap();

        let sparsity = model.coefficient_sparsity().unwrap();
        assert!((0.0..=1.0).contains(&sparsity));
    }
}

// Provide disabled functionality when sparse feature is not enabled
#[cfg(not(feature = "sparse"))]
pub mod disabled {
    use super::*;

    pub type SparseLinearRegression = ();
    pub type SparseLinearRegressionConfig = ();

    pub fn sparse_linear_regression_disabled_error() -> SklearsError {
        crate::sparse::sparse_feature_disabled_error()
    }
}

#[cfg(not(feature = "sparse"))]
pub use disabled::*;
