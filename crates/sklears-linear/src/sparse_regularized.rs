//! Sparse regularized linear models (Lasso, Ridge, ElasticNet)
//!
//! This module provides memory-efficient implementations of regularized linear models
//! optimized for sparse data using coordinate descent algorithms.

#[cfg(feature = "sparse")]
use crate::sparse::{SparseConfig, SparseCoordinateDescentSolver, SparseMatrix, SparseMatrixCSR};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearnContext, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
    validation::{ConfigValidation, Validate, ValidationRule, ValidationRules},
};
use std::marker::PhantomData;

/// Configuration for Sparse Lasso regression
#[derive(Debug, Clone)]
pub struct SparseLassoConfig {
    /// Regularization strength (alpha parameter)
    pub alpha: Float,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to use cyclic coordinate selection
    pub cyclic: bool,
    /// Sparsity threshold for determining zero coefficients
    pub sparsity_threshold: Float,
    /// Whether to automatically convert dense matrices to sparse
    pub auto_sparse_conversion: bool,
    /// Minimum sparsity ratio to use sparse algorithms
    pub min_sparsity_ratio: f64,
}

impl Default for SparseLassoConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            cyclic: true,
            sparsity_threshold: 1e-10,
            auto_sparse_conversion: true,
            min_sparsity_ratio: 0.3,
        }
    }
}

impl Validate for SparseLassoConfig {
    fn validate(&self) -> Result<()> {
        ValidationRules::new("alpha")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.alpha)?;

        ValidationRules::new("max_iter")
            .add_rule(ValidationRule::Positive)
            .validate_numeric(&self.max_iter)?;

        ValidationRules::new("tol")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.tol)?;

        ValidationRules::new("sparsity_threshold")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.sparsity_threshold)?;

        ValidationRules::new("min_sparsity_ratio")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 })
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.min_sparsity_ratio)?;

        Ok(())
    }
}

impl ConfigValidation for SparseLassoConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        if self.alpha < 1e-6 {
            log::warn!(
                "Very small alpha ({:.2e}) may lead to overfitting",
                self.alpha
            );
        }

        if self.alpha > 10.0 {
            log::warn!("Large alpha ({:.2}) may lead to underfitting", self.alpha);
        }

        if self.max_iter < 100 {
            log::warn!("Small max_iter ({}) may prevent convergence", self.max_iter);
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.tol > 1e-2 {
            warnings.push("Large tolerance may lead to inaccurate solutions".to_string());
        }

        if !self.cyclic {
            warnings.push("Random coordinate selection may be slower to converge".to_string());
        }

        warnings
    }
}

/// Sparse Lasso regression model
#[derive(Debug, Clone)]
pub struct SparseLasso<State = Untrained> {
    config: SparseLassoConfig,
    state: PhantomData<State>,
    // Trained state fields
    coefficients_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    is_sparse_fitted_: Option<bool>,
    n_features_: Option<usize>,
    n_iter_: Option<usize>,
}

impl SparseLasso<Untrained> {
    /// Create a new sparse Lasso model
    pub fn new(alpha: Float) -> Self {
        Self {
            config: SparseLassoConfig {
                alpha,
                ..Default::default()
            },
            state: PhantomData,
            coefficients_: None,
            intercept_: None,
            is_sparse_fitted_: None,
            n_features_: None,
            n_iter_: None,
        }
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set coordinate selection method
    pub fn cyclic(mut self, cyclic: bool) -> Self {
        self.config.cyclic = cyclic;
        self
    }

    /// Set automatic sparse conversion
    pub fn auto_sparse_conversion(mut self, auto_convert: bool) -> Self {
        self.config.auto_sparse_conversion = auto_convert;
        self
    }
}

impl Default for SparseLasso<Untrained> {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Estimator for SparseLasso<Untrained> {
    type Config = SparseLassoConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Fit implementation for dense matrices
impl Fit<Array2<Float>, Array1<Float>> for SparseLasso<Untrained> {
    type Fitted = SparseLasso<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate configuration
        self.config
            .validate_config()
            .fit_context("SparseLasso", n_samples, n_features)?;

        // Validate data
        use sklears_core::validation::ml;
        ml::validate_supervised_data(x, y).fit_context("SparseLasso", n_samples, n_features)?;

        #[cfg(feature = "sparse")]
        {
            // Check if we should use sparse algorithms
            let should_use_sparse = if self.config.auto_sparse_conversion {
                crate::sparse::utils::should_use_sparse(
                    x,
                    &SparseConfig {
                        sparsity_threshold: self.config.sparsity_threshold,
                        min_sparsity_ratio: self.config.min_sparsity_ratio,
                        max_dense_memory_ratio: 0.5,
                    },
                )
            } else {
                false
            };

            if should_use_sparse {
                let x_sparse = SparseMatrixCSR::from_dense(x, self.config.sparsity_threshold);
                self.fit_sparse(&x_sparse, y)
            } else {
                self.fit_dense(x, y)
            }
        }

        #[cfg(not(feature = "sparse"))]
        {
            self.fit_dense(x, y)
        }
    }
}

/// Fit implementation for sparse matrices
#[cfg(feature = "sparse")]
impl Fit<SparseMatrixCSR<Float>, Array1<Float>> for SparseLasso<Untrained> {
    type Fitted = SparseLasso<Trained>;

    fn fit(self, x: &SparseMatrixCSR<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        self.fit_sparse(x, y)
    }
}

impl SparseLasso<Untrained> {
    /// Fit using dense coordinate descent
    fn fit_dense(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<SparseLasso<Trained>> {
        // Use the regular coordinate descent solver from the coordinate_descent module
        let solver = crate::coordinate_descent::CoordinateDescentSolver {
            max_iter: self.config.max_iter,
            tol: self.config.tol,
            cyclic: self.config.cyclic,
            early_stopping_config: None,
        };

        let (coefficients, intercept) = solver
            .solve_lasso(x, y, self.config.alpha, self.config.fit_intercept)
            .map_err(|e| SklearsError::Other(e.to_string()))?;

        Ok(SparseLasso {
            config: self.config,
            state: PhantomData,
            coefficients_: Some(coefficients),
            intercept_: intercept,
            is_sparse_fitted_: Some(false),
            n_features_: Some(x.ncols()),
            n_iter_: None, // Dense solver doesn't return iteration count
        })
    }

    /// Fit using sparse coordinate descent
    #[cfg(feature = "sparse")]
    fn fit_sparse(
        self,
        x: &SparseMatrixCSR<Float>,
        y: &Array1<Float>,
    ) -> Result<SparseLasso<Trained>> {
        let solver = SparseCoordinateDescentSolver {
            alpha: self.config.alpha,
            l1_ratio: 1.0, // Pure L1 regularization for Lasso
            max_iter: self.config.max_iter,
            tol: self.config.tol,
            cyclic: self.config.cyclic,
            sparse_config: SparseConfig {
                sparsity_threshold: self.config.sparsity_threshold,
                min_sparsity_ratio: self.config.min_sparsity_ratio,
                max_dense_memory_ratio: 0.5,
            },
        };

        let (coefficients, intercept) =
            solver.solve_sparse_lasso(x, y, self.config.alpha, self.config.fit_intercept)?;

        Ok(SparseLasso {
            config: self.config,
            state: PhantomData,
            coefficients_: Some(coefficients),
            intercept_: Some(intercept),
            is_sparse_fitted_: Some(true),
            n_features_: Some(x.ncols()),
            n_iter_: None, // Sparse solver doesn't return iteration count
        })
    }
}

impl SparseLasso<Trained> {
    /// Get the fitted coefficients
    pub fn coefficients(&self) -> &Array1<Float> {
        self.coefficients_.as_ref().expect("Model is trained")
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
    pub fn n_features(&self) -> usize {
        self.n_features_.expect("Model is trained")
    }

    /// Get number of non-zero coefficients
    pub fn n_nonzero_coefficients(&self) -> usize {
        let coeffs = self.coefficients();
        coeffs
            .iter()
            .filter(|&&c| c.abs() > self.config.sparsity_threshold)
            .count()
    }

    /// Get sparsity ratio of the fitted coefficients
    pub fn coefficient_sparsity(&self) -> f64 {
        let coeffs = self.coefficients();
        let nnz = self.n_nonzero_coefficients();
        if !coeffs.is_empty() {
            1.0 - (nnz as f64 / coeffs.len() as f64)
        } else {
            0.0
        }
    }
}

/// Prediction for dense inputs
impl Predict<Array2<Float>, Array1<Float>> for SparseLasso<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let coeffs = self.coefficients_.as_ref().expect("Model is trained");
        let mut predictions = x.dot(coeffs);

        if let Some(intercept) = self.intercept_ {
            predictions.mapv_inplace(|p| p + intercept);
        }

        Ok(predictions)
    }
}

/// Prediction for sparse inputs
#[cfg(feature = "sparse")]
impl Predict<SparseMatrixCSR<Float>, Array1<Float>> for SparseLasso<Trained> {
    fn predict(&self, x: &SparseMatrixCSR<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let coeffs = self.coefficients_.as_ref().expect("Model is trained");
        let mut predictions = x.matvec(coeffs)?;

        if let Some(intercept) = self.intercept_ {
            predictions.mapv_inplace(|p| p + intercept);
        }

        Ok(predictions)
    }
}

/// Configuration for Sparse ElasticNet regression
#[derive(Debug, Clone)]
pub struct SparseElasticNetConfig {
    /// Overall regularization strength
    pub alpha: Float,
    /// ElasticNet mixing parameter (0 <= l1_ratio <= 1)
    /// l1_ratio=1 is pure Lasso, l1_ratio=0 is pure Ridge
    pub l1_ratio: Float,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to use cyclic coordinate selection
    pub cyclic: bool,
    /// Sparsity threshold for determining zero coefficients
    pub sparsity_threshold: Float,
    /// Whether to automatically convert dense matrices to sparse
    pub auto_sparse_conversion: bool,
    /// Minimum sparsity ratio to use sparse algorithms
    pub min_sparsity_ratio: f64,
}

impl Default for SparseElasticNetConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            cyclic: true,
            sparsity_threshold: 1e-10,
            auto_sparse_conversion: true,
            min_sparsity_ratio: 0.3,
        }
    }
}

impl Validate for SparseElasticNetConfig {
    fn validate(&self) -> Result<()> {
        ValidationRules::new("alpha")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.alpha)?;

        ValidationRules::new("l1_ratio")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 })
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.l1_ratio)?;

        ValidationRules::new("max_iter")
            .add_rule(ValidationRule::Positive)
            .validate_numeric(&self.max_iter)?;

        ValidationRules::new("tol")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.tol)?;

        ValidationRules::new("sparsity_threshold")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.sparsity_threshold)?;

        ValidationRules::new("min_sparsity_ratio")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 })
            .add_rule(ValidationRule::Finite)
            .validate_numeric(&self.min_sparsity_ratio)?;

        Ok(())
    }
}

impl ConfigValidation for SparseElasticNetConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        if self.l1_ratio == 0.0 {
            log::warn!("l1_ratio=0 is pure Ridge regression, consider using Ridge model instead");
        }

        if self.l1_ratio == 1.0 {
            log::warn!("l1_ratio=1 is pure Lasso regression, consider using Lasso model instead");
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.alpha < 1e-6 {
            warnings.push("Very small alpha may lead to overfitting".to_string());
        }

        if self.alpha > 10.0 {
            warnings.push("Large alpha may lead to underfitting".to_string());
        }

        warnings
    }
}

/// Sparse ElasticNet regression model
#[derive(Debug, Clone)]
pub struct SparseElasticNet<State = Untrained> {
    config: SparseElasticNetConfig,
    state: PhantomData<State>,
    // Trained state fields
    coefficients_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    is_sparse_fitted_: Option<bool>,
    n_features_: Option<usize>,
}

impl SparseElasticNet<Untrained> {
    /// Create a new sparse ElasticNet model
    pub fn new(alpha: Float, l1_ratio: Float) -> Self {
        Self {
            config: SparseElasticNetConfig {
                alpha,
                l1_ratio,
                ..Default::default()
            },
            state: PhantomData,
            coefficients_: None,
            intercept_: None,
            is_sparse_fitted_: None,
            n_features_: None,
        }
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set L1 ratio
    pub fn l1_ratio(mut self, l1_ratio: Float) -> Self {
        self.config.l1_ratio = l1_ratio;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }
}

impl Default for SparseElasticNet<Untrained> {
    fn default() -> Self {
        Self::new(1.0, 0.5)
    }
}

impl Estimator for SparseElasticNet<Untrained> {
    type Config = SparseElasticNetConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Fit implementation for dense matrices
impl Fit<Array2<Float>, Array1<Float>> for SparseElasticNet<Untrained> {
    type Fitted = SparseElasticNet<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate configuration
        self.config
            .validate_config()
            .fit_context("SparseElasticNet", n_samples, n_features)?;

        // Validate data
        use sklears_core::validation::ml;
        ml::validate_supervised_data(x, y).fit_context(
            "SparseElasticNet",
            n_samples,
            n_features,
        )?;

        #[cfg(feature = "sparse")]
        {
            let should_use_sparse = if self.config.auto_sparse_conversion {
                crate::sparse::utils::should_use_sparse(
                    x,
                    &SparseConfig {
                        sparsity_threshold: self.config.sparsity_threshold,
                        min_sparsity_ratio: self.config.min_sparsity_ratio,
                        max_dense_memory_ratio: 0.5,
                    },
                )
            } else {
                false
            };

            if should_use_sparse {
                let x_sparse = SparseMatrixCSR::from_dense(x, self.config.sparsity_threshold);
                self.fit_sparse(&x_sparse, y)
            } else {
                self.fit_dense(x, y)
            }
        }

        #[cfg(not(feature = "sparse"))]
        {
            self.fit_dense(x, y)
        }
    }
}

/// Fit implementation for sparse matrices
#[cfg(feature = "sparse")]
impl Fit<SparseMatrixCSR<Float>, Array1<Float>> for SparseElasticNet<Untrained> {
    type Fitted = SparseElasticNet<Trained>;

    fn fit(self, x: &SparseMatrixCSR<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        self.fit_sparse(x, y)
    }
}

impl SparseElasticNet<Untrained> {
    /// Fit using dense coordinate descent
    fn fit_dense(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<SparseElasticNet<Trained>> {
        let solver = crate::coordinate_descent::CoordinateDescentSolver {
            max_iter: self.config.max_iter,
            tol: self.config.tol,
            cyclic: self.config.cyclic,
            early_stopping_config: None,
        };

        let (coefficients, intercept) = solver
            .solve_elastic_net(
                x,
                y,
                self.config.alpha,
                self.config.l1_ratio,
                self.config.fit_intercept,
            )
            .map_err(|e| SklearsError::Other(e.to_string()))?;

        Ok(SparseElasticNet {
            config: self.config,
            state: PhantomData,
            coefficients_: Some(coefficients),
            intercept_: intercept,
            is_sparse_fitted_: Some(false),
            n_features_: Some(x.ncols()),
        })
    }

    /// Fit using sparse coordinate descent
    #[cfg(feature = "sparse")]
    fn fit_sparse(
        self,
        x: &SparseMatrixCSR<Float>,
        y: &Array1<Float>,
    ) -> Result<SparseElasticNet<Trained>> {
        let solver = SparseCoordinateDescentSolver {
            alpha: self.config.alpha,
            l1_ratio: self.config.l1_ratio,
            max_iter: self.config.max_iter,
            tol: self.config.tol,
            cyclic: self.config.cyclic,
            sparse_config: SparseConfig {
                sparsity_threshold: self.config.sparsity_threshold,
                min_sparsity_ratio: self.config.min_sparsity_ratio,
                max_dense_memory_ratio: 0.5,
            },
        };

        let (coefficients, intercept) = solver.solve_sparse_elastic_net(
            x,
            y,
            self.config.alpha,
            self.config.l1_ratio,
            self.config.fit_intercept,
        )?;

        Ok(SparseElasticNet {
            config: self.config,
            state: PhantomData,
            coefficients_: Some(coefficients),
            intercept_: Some(intercept),
            is_sparse_fitted_: Some(true),
            n_features_: Some(x.ncols()),
        })
    }
}

impl SparseElasticNet<Trained> {
    /// Get the fitted coefficients
    pub fn coefficients(&self) -> &Array1<Float> {
        self.coefficients_.as_ref().expect("Model is trained")
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
    pub fn n_features(&self) -> usize {
        self.n_features_.expect("Model is trained")
    }

    /// Get number of non-zero coefficients
    pub fn n_nonzero_coefficients(&self) -> usize {
        let coeffs = self.coefficients();
        coeffs
            .iter()
            .filter(|&&c| c.abs() > self.config.sparsity_threshold)
            .count()
    }

    /// Get sparsity ratio of the fitted coefficients
    pub fn coefficient_sparsity(&self) -> f64 {
        let coeffs = self.coefficients();
        let nnz = self.n_nonzero_coefficients();
        if !coeffs.is_empty() {
            1.0 - (nnz as f64 / coeffs.len() as f64)
        } else {
            0.0
        }
    }
}

/// Prediction for dense inputs
impl Predict<Array2<Float>, Array1<Float>> for SparseElasticNet<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let coeffs = self.coefficients_.as_ref().expect("Model is trained");
        let mut predictions = x.dot(coeffs);

        if let Some(intercept) = self.intercept_ {
            predictions.mapv_inplace(|p| p + intercept);
        }

        Ok(predictions)
    }
}

/// Prediction for sparse inputs
#[cfg(feature = "sparse")]
impl Predict<SparseMatrixCSR<Float>, Array1<Float>> for SparseElasticNet<Trained> {
    fn predict(&self, x: &SparseMatrixCSR<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let coeffs = self.coefficients_.as_ref().expect("Model is trained");
        let mut predictions = x.matvec(coeffs)?;

        if let Some(intercept) = self.intercept_ {
            predictions.mapv_inplace(|p| p + intercept);
        }

        Ok(predictions)
    }
}

#[cfg(all(test, feature = "sparse"))]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_lasso_dense_input() {
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.0, 1.0],];
        let y = array![1.0, 2.0, 3.0, 1.5];

        let model = SparseLasso::new(0.1)
            .fit_intercept(true)
            .fit(&x, &y)
            .unwrap();

        let coeffs = model.coefficients();
        assert_eq!(coeffs.len(), 2);

        // Test prediction
        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.len(), 4);

        // Check sparsity
        let sparsity = model.coefficient_sparsity();
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }

    #[test]
    #[ignore = "Waiting for SciRS2-sparse v0.1.0-rc.2 to be published to crates.io"]
    fn test_sparse_lasso_sparse_input() {
        let triplets = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 0, 2.0),
            (1, 1, 1.0),
            (2, 0, 3.0),
            (2, 1, 3.0),
        ];
        let x_sparse = SparseMatrixCSR::from_triplets(3, 2, &triplets).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = SparseLasso::new(0.1).fit(&x_sparse, &y).unwrap();

        assert!(model.is_sparse_fitted());
        assert_eq!(model.n_features(), 2);
        assert!(model.n_nonzero_coefficients() <= 2);

        // Test sparse prediction
        let y_pred = model.predict(&x_sparse).unwrap();
        assert_eq!(y_pred.len(), 3);
    }

    #[test]
    fn test_sparse_elastic_net_dense_input() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = SparseElasticNet::new(0.1, 0.5)
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        let coeffs = model.coefficients();
        assert_eq!(coeffs.len(), 2);

        // ElasticNet should find a solution
        assert!(coeffs.iter().any(|&c| c.abs() > 1e-6));

        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.len(), 4);
    }

    #[test]
    fn test_sparse_elastic_net_validation() {
        let valid_config = SparseElasticNetConfig::default();
        assert!(valid_config.validate().is_ok());

        let mut invalid_config = SparseElasticNetConfig::default();
        invalid_config.l1_ratio = 1.5;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = SparseElasticNetConfig::default();
        invalid_config.alpha = -1.0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_lasso_sparsity_inducing() {
        // Create data where some features are irrelevant
        let x = array![
            [1.0, 0.0, 0.5], // First feature is relevant
            [2.0, 0.0, 0.3], // Second feature is always zero (irrelevant)
            [3.0, 0.0, 0.1], // Third feature has small coefficients
            [1.5, 0.0, 0.4],
        ];
        let y = array![1.0, 2.0, 3.0, 1.5]; // Mainly depends on first feature

        let model = SparseLasso::new(0.5) // High regularization
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        let coeffs = model.coefficients();

        // Should have induced sparsity (some coefficients should be zero or very small)
        let nnz = model.n_nonzero_coefficients();
        assert!(nnz < coeffs.len(), "Lasso should induce sparsity");

        // The second feature should be zero or very small (it was always zero in data)
        assert!(
            coeffs[1].abs() < 1e-3,
            "Irrelevant feature should be zeroed out"
        );
    }

    #[test]
    fn test_elastic_net_l1_ratio_extremes() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];

        // Test pure Lasso (l1_ratio = 1.0)
        let lasso_model = SparseElasticNet::new(0.1, 1.0).fit(&x, &y).unwrap();

        // Test pure Ridge (l1_ratio = 0.0)
        let ridge_model = SparseElasticNet::new(0.1, 0.0).fit(&x, &y).unwrap();

        // Both should converge to reasonable solutions
        assert!(lasso_model.coefficients()[0].abs() > 1e-6);
        assert!(ridge_model.coefficients()[0].abs() > 1e-6);
    }
}

// Provide disabled functionality when sparse feature is not enabled
#[cfg(not(feature = "sparse"))]
pub mod disabled {
    use super::*;

    pub type SparseLasso = ();
    pub type SparseElasticNet = ();
    pub type SparseLassoConfig = ();
    pub type SparseElasticNetConfig = ();

    pub fn sparse_regularized_disabled_error() -> SklearsError {
        crate::sparse::sparse_feature_disabled_error()
    }
}

#[cfg(not(feature = "sparse"))]
pub use disabled::*;
