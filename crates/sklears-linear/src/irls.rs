//! Iteratively Reweighted Least Squares (IRLS) for robust regression
//!
//! IRLS is a method for fitting robust regression models by iteratively reweighting
//! observations based on their residuals. It's particularly useful for M-estimators
//! and can handle various loss functions and weight functions to downweight outliers.

use sklears_core::error::SklearsError;
use std::cmp::Ordering;

/// Weight functions for IRLS
#[derive(Debug, Clone, PartialEq)]
pub enum WeightFunction {
    /// Huber weight function: w(r) = min(1, c/|r|)
    Huber { c: f64 },
    /// Bisquare (Tukey) weight function: w(r) = (1 - (r/c)^2)^2 if |r| <= c, 0 otherwise
    Bisquare { c: f64 },
    /// Andrews wave weight function: w(r) = sin(πr/c) / (πr/c) if |r| <= c, 0 otherwise
    Andrews { c: f64 },
    /// Cauchy weight function: w(r) = 1 / (1 + (r/c)^2)
    Cauchy { c: f64 },
    /// Fair weight function: w(r) = 1 / (1 + |r|/c)
    Fair { c: f64 },
    /// Logistic weight function: w(r) = tanh(c*r) / (c*r)
    Logistic { c: f64 },
}

/// Scale estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleEstimator {
    /// Median Absolute Deviation (MAD)
    MAD,
    /// Standard deviation
    StandardDeviation,
    /// Interquartile range scaled to approximate standard deviation
    IQR,
    /// Custom fixed scale
    Fixed(f64),
}

/// IRLS configuration
#[derive(Debug, Clone)]
pub struct IRLSConfig {
    /// Weight function for robust estimation
    pub weight_function: WeightFunction,
    /// Scale estimation method
    pub scale_estimator: ScaleEstimator,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Initial scale estimate (if None, estimated from data)
    pub initial_scale: Option<f64>,
    /// Minimum weight threshold (weights below this are set to 0)
    pub min_weight: f64,
    /// Whether to update scale at each iteration
    pub update_scale: bool,
    /// Regularization parameter (L2)
    pub alpha: f64,
}

impl Default for IRLSConfig {
    fn default() -> Self {
        Self {
            weight_function: WeightFunction::Huber { c: 1.345 },
            scale_estimator: ScaleEstimator::MAD,
            max_iter: 100,
            tol: 1e-6,
            fit_intercept: true,
            initial_scale: None,
            min_weight: 1e-8,
            update_scale: true,
            alpha: 0.0,
        }
    }
}

/// IRLS result
#[derive(Debug, Clone)]
pub struct IRLSResult {
    /// Fitted coefficients
    pub coefficients: Vec<f64>,
    /// Fitted intercept (if fit_intercept=true)
    pub intercept: Option<f64>,
    /// Final weights for each observation
    pub weights: Vec<f64>,
    /// Final scale estimate
    pub scale: f64,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Convergence history (coefficient changes)
    pub convergence_history: Vec<f64>,
    /// Configuration used
    pub config: IRLSConfig,
}

/// Iteratively Reweighted Least Squares estimator
pub struct IRLSEstimator {
    config: IRLSConfig,
    is_fitted: bool,
    result: Option<IRLSResult>,
}

impl IRLSEstimator {
    /// Create a new IRLS estimator with default configuration
    pub fn new() -> Self {
        Self {
            config: IRLSConfig::default(),
            is_fitted: false,
            result: None,
        }
    }

    /// Create an IRLS estimator with custom configuration
    pub fn with_config(config: IRLSConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            result: None,
        }
    }

    /// Set the weight function
    pub fn with_weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.config.weight_function = weight_function;
        self
    }

    /// Set the scale estimator
    pub fn with_scale_estimator(mut self, scale_estimator: ScaleEstimator) -> Self {
        self.config.scale_estimator = scale_estimator;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to fit intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization parameter
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Fit the IRLS model
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), SklearsError> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit IRLS on empty dataset".to_string(),
            ));
        }

        let n_samples = x.len();
        let n_features = x[0].len();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("target.len() == {}", n_samples),
                actual: format!("target.len() == {}", y.len()),
            });
        }

        // Validate input data
        for (i, row) in x.iter().enumerate() {
            if row.len() != n_features {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("row[{}].len() == {}", i, n_features),
                    actual: format!("row[{}].len() == {}", i, row.len()),
                });
            }
        }

        // Prepare design matrix
        let x_matrix = if self.config.fit_intercept {
            self.add_intercept_column(x)
        } else {
            x.to_vec()
        };

        let _effective_n_features = x_matrix[0].len();

        // Initialize coefficients with ordinary least squares
        let mut coefficients = self.ordinary_least_squares(&x_matrix, y)?;
        let mut weights = vec![1.0; n_samples];
        let mut convergence_history = Vec::new();

        // Initial residuals and scale estimate
        let mut residuals = self.compute_residuals(&x_matrix, y, &coefficients);
        let mut scale = self.estimate_scale(&residuals)?;

        let mut converged = false;
        let mut n_iter = 0;

        // IRLS iterations
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;

            // Update weights based on residuals
            self.update_weights(&residuals, scale, &mut weights);

            // Weighted least squares
            let new_coefficients = self.weighted_least_squares(&x_matrix, y, &weights)?;

            // Check convergence
            let coefficient_change =
                self.compute_coefficient_change(&coefficients, &new_coefficients);
            convergence_history.push(coefficient_change);

            if coefficient_change < self.config.tol {
                converged = true;
                coefficients = new_coefficients;
                break;
            }

            coefficients = new_coefficients;

            // Update residuals
            residuals = self.compute_residuals(&x_matrix, y, &coefficients);

            // Update scale if requested
            if self.config.update_scale {
                scale = self.estimate_scale(&residuals)?;
            }
        }

        // Split coefficients and intercept
        let (final_coefficients, intercept) = if self.config.fit_intercept {
            let intercept = coefficients[0];
            let coefs = coefficients[1..].to_vec();
            (coefs, Some(intercept))
        } else {
            (coefficients, None)
        };

        self.result = Some(IRLSResult {
            coefficients: final_coefficients,
            intercept,
            weights,
            scale,
            n_iter,
            converged,
            convergence_history,
            config: self.config.clone(),
        });

        self.is_fitted = true;
        Ok(())
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let result = self.result.as_ref().unwrap();

        if x.is_empty() {
            return Ok(Vec::new());
        }

        let n_features = x[0].len();
        if n_features != result.coefficients.len() {
            return Err(SklearsError::FeatureMismatch {
                expected: result.coefficients.len(),
                actual: n_features,
            });
        }

        let mut predictions = Vec::new();

        for row in x {
            let mut pred = 0.0;
            for (i, &coef) in result.coefficients.iter().enumerate() {
                pred += coef * row[i];
            }

            if let Some(intercept) = result.intercept {
                pred += intercept;
            }

            predictions.push(pred);
        }

        Ok(predictions)
    }

    /// Get the fitting result
    pub fn get_result(&self) -> Option<&IRLSResult> {
        self.result.as_ref()
    }

    /// Get fitted coefficients
    pub fn get_coefficients(&self) -> Option<&Vec<f64>> {
        self.result.as_ref().map(|r| &r.coefficients)
    }

    /// Get fitted intercept
    pub fn get_intercept(&self) -> Option<f64> {
        self.result.as_ref().and_then(|r| r.intercept)
    }

    /// Get final weights
    pub fn get_weights(&self) -> Option<&Vec<f64>> {
        self.result.as_ref().map(|r| &r.weights)
    }

    /// Add intercept column to design matrix
    fn add_intercept_column(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let mut new_row = vec![1.0];
                new_row.extend(row);
                new_row
            })
            .collect()
    }

    /// Compute ordinary least squares solution
    fn ordinary_least_squares(&self, x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, SklearsError> {
        let n_samples = x.len();
        let n_features = x[0].len();

        // Compute X^T X
        let mut xtx = vec![vec![0.0; n_features]; n_features];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_features {
            for j in 0..n_features {
                #[allow(clippy::needless_range_loop)]
                for k in 0..n_samples {
                    xtx[i][j] += x[k][i] * x[k][j];
                }

                // Add regularization to diagonal
                if i == j {
                    xtx[i][j] += self.config.alpha;
                }
            }
        }

        // Compute X^T y
        let mut xty = vec![0.0; n_features];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_features {
            for j in 0..n_samples {
                xty[i] += x[j][i] * y[j];
            }
        }

        // Solve system
        self.solve_linear_system(&xtx, &xty)
    }

    /// Weighted least squares solution
    fn weighted_least_squares(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        weights: &[f64],
    ) -> Result<Vec<f64>, SklearsError> {
        let n_samples = x.len();
        let n_features = x[0].len();

        // Compute X^T W X
        let mut xtwx = vec![vec![0.0; n_features]; n_features];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    xtwx[i][j] += weights[k] * x[k][i] * x[k][j];
                }

                // Add regularization to diagonal
                if i == j {
                    xtwx[i][j] += self.config.alpha;
                }
            }
        }

        // Compute X^T W y
        let mut xtwy = vec![0.0; n_features];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_features {
            for j in 0..n_samples {
                xtwy[i] += weights[j] * x[j][i] * y[j];
            }
        }

        // Solve system
        self.solve_linear_system(&xtwx, &xtwy)
    }

    /// Solve linear system using Gaussian elimination with partial pivoting
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SklearsError> {
        match Self::gaussian_elimination(a, b) {
            Ok(solution) => Ok(solution),
            Err(original_error) => {
                // Apply a small ridge regularization and retry to handle rank-deficient systems
                let mut regularized = a.to_vec();
                if regularized.is_empty() || regularized[0].is_empty() {
                    return Err(original_error);
                }

                let ridge = if self.config.alpha > 0.0 {
                    self.config.alpha
                } else {
                    1e-6
                };

                #[allow(clippy::needless_range_loop)]
                for i in 0..regularized.len() {
                    regularized[i][i] += ridge;
                }

                match Self::gaussian_elimination(&regularized, b) {
                    Ok(solution) => Ok(solution),
                    Err(_) => Err(original_error),
                }
            }
        }
    }

    fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SklearsError> {
        let n = a.len();
        if n == 0 || b.len() != n {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions do not align for Gaussian elimination".to_string(),
            ));
        }

        let mut aug_matrix = vec![vec![0.0; n + 1]; n];

        // Create augmented matrix
        for i in 0..n {
            if a[i].len() != n {
                return Err(SklearsError::InvalidInput(
                    "Matrix must be square for Gaussian elimination".to_string(),
                ));
            }

            for j in 0..n {
                aug_matrix[i][j] = a[i][j];
            }
            aug_matrix[i][n] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                aug_matrix.swap(i, max_row);
            }

            // Check for singularity
            if aug_matrix[i][i].abs() < 1e-12 {
                return Err(SklearsError::InvalidInput(
                    "Matrix is singular or nearly singular. Add regularization or check for multicollinearity".to_string(),
                ));
            }

            // Eliminate
            for k in i + 1..n {
                let factor = aug_matrix[k][i] / aug_matrix[i][i];
                for j in i..n + 1 {
                    aug_matrix[k][j] -= factor * aug_matrix[i][j];
                }
            }
        }

        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            solution[i] = aug_matrix[i][n];
            for j in i + 1..n {
                solution[i] -= aug_matrix[i][j] * solution[j];
            }
            solution[i] /= aug_matrix[i][i];
        }

        Ok(solution)
    }

    /// Compute residuals
    fn compute_residuals(&self, x: &[Vec<f64>], y: &[f64], coefficients: &[f64]) -> Vec<f64> {
        let mut residuals = Vec::new();

        for (i, row) in x.iter().enumerate() {
            let mut pred = 0.0;
            for (j, &coef) in coefficients.iter().enumerate() {
                pred += coef * row[j];
            }
            residuals.push(y[i] - pred);
        }

        residuals
    }

    /// Estimate scale parameter
    fn estimate_scale(&self, residuals: &[f64]) -> Result<f64, SklearsError> {
        if residuals.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot estimate scale from empty residuals".to_string(),
            ));
        }

        let scale = match &self.config.scale_estimator {
            ScaleEstimator::MAD => {
                let mut abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
                abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let median = abs_residuals[abs_residuals.len() / 2];
                median * 1.4826 // MAD to standard deviation conversion
            }

            ScaleEstimator::StandardDeviation => {
                let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
                let variance = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
                    / (residuals.len() - 1) as f64;
                variance.sqrt()
            }

            ScaleEstimator::IQR => {
                let mut sorted_residuals: Vec<f64> = residuals.to_vec();
                sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let n = sorted_residuals.len();
                let q1 = sorted_residuals[n / 4];
                let q3 = sorted_residuals[3 * n / 4];
                (q3 - q1) / 1.349 // IQR to standard deviation conversion
            }

            ScaleEstimator::Fixed(scale) => *scale,
        };

        if scale <= 0.0 {
            Ok(1e-6) // Minimum scale to avoid division by zero
        } else {
            Ok(scale)
        }
    }

    /// Update weights based on residuals
    fn update_weights(&self, residuals: &[f64], scale: f64, weights: &mut [f64]) {
        for (i, &residual) in residuals.iter().enumerate() {
            let standardized_residual = residual / scale;
            weights[i] = self
                .compute_weight(standardized_residual)
                .max(self.config.min_weight);
        }
    }

    /// Compute weight for a given standardized residual
    fn compute_weight(&self, r: f64) -> f64 {
        match &self.config.weight_function {
            WeightFunction::Huber { c } => {
                if r.abs() <= *c {
                    1.0
                } else {
                    c / r.abs()
                }
            }

            WeightFunction::Bisquare { c } => {
                if r.abs() <= *c {
                    let ratio = r / c;
                    (1.0 - ratio.powi(2)).powi(2)
                } else {
                    0.0
                }
            }

            WeightFunction::Andrews { c } => {
                if r.abs() <= *c {
                    let ratio = std::f64::consts::PI * r / c;
                    if ratio.abs() < 1e-10 {
                        1.0
                    } else {
                        ratio.sin() / ratio
                    }
                } else {
                    0.0
                }
            }

            WeightFunction::Cauchy { c } => 1.0 / (1.0 + (r / c).powi(2)),

            WeightFunction::Fair { c } => 1.0 / (1.0 + r.abs() / c),

            WeightFunction::Logistic { c } => {
                let cr = c * r;
                if cr.abs() < 1e-10 {
                    1.0
                } else {
                    cr.tanh() / cr
                }
            }
        }
    }

    /// Compute coefficient change between iterations
    fn compute_coefficient_change(&self, old_coefs: &[f64], new_coefs: &[f64]) -> f64 {
        old_coefs
            .iter()
            .zip(new_coefs.iter())
            .map(|(&old, &new)| (old - new).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for IRLSEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        // Linear relationship with some outliers
        let x = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
            vec![9.0],
            vec![10.0],
        ];

        // y = 2*x + 1 with outliers
        let mut y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];
        y[8] = 50.0; // outlier
        y[9] = 5.0; // outlier

        (x, y)
    }

    #[test]
    fn test_irls_basic() {
        let mut irls = IRLSEstimator::new();
        let (x, y) = create_sample_data();

        let result = irls.fit(&x, &y);
        assert!(result.is_ok());

        let coefficients = irls.get_coefficients().unwrap();
        assert_eq!(coefficients.len(), 1);

        // Should be robust to outliers and close to true slope of 2
        assert!((coefficients[0] - 2.0).abs() < 0.5);

        let intercept = irls.get_intercept().unwrap();
        // Should be close to true intercept of 1
        assert!((intercept - 1.0).abs() < 1.0);
    }

    #[test]
    fn test_irls_huber() {
        let mut irls =
            IRLSEstimator::new().with_weight_function(WeightFunction::Huber { c: 1.345 });

        let (x, y) = create_sample_data();
        let result = irls.fit(&x, &y);

        assert!(result.is_ok());

        let weights = irls.get_weights().unwrap();
        assert_eq!(weights.len(), y.len());

        // Outliers should have lower weights
        assert!(weights[8] < weights[0]); // outlier has lower weight
        assert!(weights[9] < weights[0]); // outlier has lower weight
    }

    #[test]
    fn test_irls_bisquare() {
        let mut irls =
            IRLSEstimator::new().with_weight_function(WeightFunction::Bisquare { c: 4.685 });

        let (x, y) = create_sample_data();
        let result = irls.fit(&x, &y);

        assert!(result.is_ok());
        assert!(irls.is_fitted);
    }

    #[test]
    fn test_irls_prediction() {
        let mut irls = IRLSEstimator::new();
        let (x, y) = create_sample_data();

        irls.fit(&x, &y).unwrap();

        let x_test = vec![vec![5.5], vec![7.5]];
        let predictions = irls.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 2);

        // Predictions should be reasonable
        assert!(predictions[0] > 10.0 && predictions[0] < 15.0);
        assert!(predictions[1] > 14.0 && predictions[1] < 18.0);
    }

    #[test]
    fn test_irls_no_intercept() {
        let mut irls = IRLSEstimator::new().with_fit_intercept(false);

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0]; // y = 2*x

        let result = irls.fit(&x, &y);
        assert!(result.is_ok());

        assert_eq!(irls.get_intercept(), None);

        let coefficients = irls.get_coefficients().unwrap();
        assert!((coefficients[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_irls_multivariate() {
        let mut irls = IRLSEstimator::new();

        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
        ];
        let y = vec![8.0, 13.0, 18.0, 23.0, 28.0]; // y = 1*x1 + 3*x2 + 1

        let result = irls.fit(&x, &y);
        assert!(result.is_ok());

        let coefficients = irls.get_coefficients().unwrap();
        assert_eq!(coefficients.len(), 2);
    }

    #[test]
    fn test_irls_convergence() {
        let mut irls = IRLSEstimator::new().with_max_iter(5).with_tolerance(1e-3);

        let (x, y) = create_sample_data();
        irls.fit(&x, &y).unwrap();

        let result = irls.get_result().unwrap();
        assert!(result.n_iter <= 5);
        assert!(!result.convergence_history.is_empty());
    }

    #[test]
    fn test_irls_different_scale_estimators() {
        let (x, y) = create_sample_data();

        let scale_estimators = vec![
            ScaleEstimator::MAD,
            ScaleEstimator::StandardDeviation,
            ScaleEstimator::IQR,
            ScaleEstimator::Fixed(1.0),
        ];

        for scale_estimator in scale_estimators {
            let mut irls = IRLSEstimator::new().with_scale_estimator(scale_estimator);

            let result = irls.fit(&x, &y);
            assert!(
                result.is_ok(),
                "Failed with scale estimator: {:?}",
                irls.config.scale_estimator
            );
        }
    }

    #[test]
    fn test_irls_weight_functions() {
        let (x, y) = create_sample_data();

        let weight_functions = vec![
            WeightFunction::Huber { c: 1.345 },
            WeightFunction::Bisquare { c: 4.685 },
            WeightFunction::Andrews { c: 1.339 },
            WeightFunction::Cauchy { c: 2.385 },
            WeightFunction::Fair { c: 1.4 },
            WeightFunction::Logistic { c: 1.2 },
        ];

        for weight_function in weight_functions {
            let mut irls = IRLSEstimator::new().with_weight_function(weight_function.clone());

            let result = irls.fit(&x, &y);
            assert!(
                result.is_ok(),
                "Failed with weight function: {:?}",
                weight_function
            );
        }
    }

    #[test]
    fn test_irls_empty_data_error() {
        let mut irls = IRLSEstimator::new();
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let result = irls.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_irls_dimension_mismatch_error() {
        let mut irls = IRLSEstimator::new();
        let (x, _) = create_sample_data();
        let wrong_y = vec![1.0, 2.0]; // Wrong length

        let result = irls.fit(&x, &wrong_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_irls_predict_before_fit_error() {
        let irls = IRLSEstimator::new();
        let x = vec![vec![1.0]];

        let result = irls.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_irls_regularization() {
        let mut irls = IRLSEstimator::new().with_alpha(0.1); // Add L2 regularization

        let (x, y) = create_sample_data();
        let result = irls.fit(&x, &y);

        assert!(result.is_ok());

        // With regularization, coefficients should be slightly smaller
        let coefficients = irls.get_coefficients().unwrap();
        assert!(coefficients[0] < 2.1); // Regularized coefficient
    }
}
