//! Multi-task Elastic Net model implementation
//!
//! Multi-task Elastic Net jointly solves multiple regression problems with shared sparsity patterns
//! and additional L2 regularization.
//! It minimizes: (1/2n) ||Y - XW||²_F + α * ρ * ||W||_{2,1} + α * (1-ρ)/2 * ||W||²_F
//! where ||W||_{2,1} = Σ_i ||w_i||_2 encourages row sparsity and ||W||²_F is the Frobenius norm.

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

/// Configuration for Multi-task Elastic Net
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNetConfig {
    /// Regularization parameter
    pub alpha: Float,
    /// The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
    /// l1_ratio = 0 corresponds to L2 penalty (Ridge)
    /// l1_ratio = 1 corresponds to L1 penalty (Lasso)
    pub l1_ratio: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to use warm start
    pub warm_start: bool,
    /// Whether to use precomputed Gram matrix
    pub precompute: bool,
}

impl Default for MultiTaskElasticNetConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            precompute: false,
        }
    }
}

/// Multi-task Elastic Net model
#[derive(Debug, Clone)]
pub struct MultiTaskElasticNet<State = Untrained> {
    config: MultiTaskElasticNetConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    n_tasks_: Option<usize>,
    n_iter_: Option<usize>,
}

impl MultiTaskElasticNet<Untrained> {
    /// Create a new Multi-task Elastic Net model
    pub fn new() -> Self {
        Self {
            config: MultiTaskElasticNetConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_tasks_: None,
            n_iter_: None,
        }
    }

    /// Create with specific alpha and l1_ratio
    pub fn with_params(alpha: Float, l1_ratio: Float) -> Self {
        Self::new().alpha(alpha).l1_ratio(l1_ratio)
    }

    /// Set alpha parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set l1_ratio parameter
    pub fn l1_ratio(mut self, l1_ratio: Float) -> Self {
        if !(0.0..=1.0).contains(&l1_ratio) {
            panic!("l1_ratio must be between 0 and 1");
        }
        self.config.l1_ratio = l1_ratio;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Builder method to create from config
    pub fn builder() -> MultiTaskElasticNetBuilder {
        MultiTaskElasticNetBuilder::default()
    }
}

/// Builder for MultiTaskElasticNet
#[derive(Debug, Default)]
pub struct MultiTaskElasticNetBuilder {
    config: MultiTaskElasticNetConfig,
}

impl MultiTaskElasticNetBuilder {
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn l1_ratio(mut self, l1_ratio: Float) -> Self {
        if !(0.0..=1.0).contains(&l1_ratio) {
            panic!("l1_ratio must be between 0 and 1");
        }
        self.config.l1_ratio = l1_ratio;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn build(self) -> MultiTaskElasticNet<Untrained> {
        MultiTaskElasticNet {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_tasks_: None,
            n_iter_: None,
        }
    }
}

impl Default for MultiTaskElasticNet<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for MultiTaskElasticNet<State> {
    type Float = Float;
    type Config = MultiTaskElasticNetConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array2<Float>> for MultiTaskElasticNet<Untrained> {
    type Fitted = MultiTaskElasticNet<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_tasks = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] == Y.shape[0]"),
                actual: format!("X.shape[0]={}, Y.shape[0]={}", n_samples, y.nrows()),
            });
        }

        if n_samples < n_features {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be >= number of features".to_string(),
            ));
        }

        // Regularization parameters
        let l1_reg = self.config.alpha * self.config.l1_ratio;
        let l2_reg = self.config.alpha * (1.0 - self.config.l1_ratio);

        // Initialize coefficients
        let mut coef: Array2<Float> = Array2::zeros((n_features, n_tasks));
        let intercept = if self.config.fit_intercept {
            y.mean_axis(Axis(0)).unwrap()
        } else {
            Array1::zeros(n_tasks)
        };

        // Center data if fitting intercept
        let (x_centered, y_centered) = if self.config.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let x_c = x - &x_mean.insert_axis(Axis(0));
            let y_c = y - &intercept.clone().insert_axis(Axis(0));
            (x_c, y_c)
        } else {
            (x.clone(), y.clone())
        };

        // Coordinate descent for multi-task elastic net
        let mut n_iter = 0;
        let n_samples_f = n_samples as Float;

        for iter in 0..self.config.max_iter {
            let old_coef = coef.clone();

            // Update each feature
            for j in 0..n_features {
                // Compute residuals without j-th feature
                let mut residuals = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let x_col = x_centered.column(k);
                        let coef_row = coef.row(k);
                        for t in 0..n_tasks {
                            let mut res_col = residuals.column_mut(t);
                            res_col.scaled_add(-coef_row[t], &x_col);
                        }
                    }
                }

                // Compute gradient for j-th feature across all tasks
                let x_j = x_centered.column(j);
                let mut gradient: Array1<Float> = Array1::zeros(n_tasks);
                for t in 0..n_tasks {
                    gradient[t] = x_j.dot(&residuals.column(t)) / n_samples_f;
                }

                // Compute L2 norm of gradient
                let gradient_norm = gradient.dot(&gradient).sqrt();

                // Apply block soft thresholding with L2 regularization
                if gradient_norm > l1_reg {
                    let scale = (gradient_norm - l1_reg) / gradient_norm;
                    let x_j_norm_sq = x_j.dot(&x_j) / n_samples_f + l2_reg;

                    for t in 0..n_tasks {
                        coef[[j, t]] = scale * gradient[t] / x_j_norm_sq;
                    }
                } else {
                    // Set entire row to zero
                    for t in 0..n_tasks {
                        coef[[j, t]] = 0.0;
                    }
                }
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(Float::abs).sum();
            if coef_change < self.config.tol {
                n_iter = iter + 1;
                break;
            }
            n_iter = iter + 1;
        }

        if n_iter == self.config.max_iter {
            eprintln!(
                "Warning: MultiTaskElasticNet did not converge. Consider increasing max_iter."
            );
        }

        Ok(MultiTaskElasticNet {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array2<Float>> for MultiTaskElasticNet<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Validate input
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let n_features = self.n_features_.unwrap();
        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();

        // Compute predictions: Y = X @ W + intercept
        let mut predictions = x.dot(coef);

        // Add intercept
        for i in 0..predictions.nrows() {
            let mut row = predictions.row_mut(i);
            row += intercept;
        }

        Ok(predictions)
    }
}

impl MultiTaskElasticNet<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> &Array1<Float> {
        self.intercept_.as_ref().unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }

    /// Get the number of tasks
    pub fn n_tasks(&self) -> usize {
        self.n_tasks_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_task_elastic_net_simple() {
        // Simple test with known solution
        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];

        // Two tasks with simple linear relationship
        let y = array![
            [1.0, 2.0], // 1*1 + 0*0 = 1, 1*2 + 0*0 = 2
            [0.0, 0.0], // 0*1 + 1*0 = 0, 0*2 + 1*0 = 0
            [1.0, 2.0], // 1*1 + 1*0 = 1, 1*2 + 1*0 = 2
            [2.0, 4.0], // 2*1 + 0*0 = 2, 2*2 + 0*0 = 4
            [0.0, 0.0], // 0*1 + 2*0 = 0, 0*2 + 2*0 = 0
            [2.0, 4.0], // 2*1 + 2*0 = 2, 2*2 + 2*0 = 4
        ];

        let model = MultiTaskElasticNet::new()
            .alpha(0.01)
            .l1_ratio(0.5)
            .fit_intercept(false);

        let fitted = model.fit(&x, &y).unwrap();
        let coef = fitted.coef();

        // First feature should have coefficients close to [1.0, 2.0]
        assert_abs_diff_eq!(coef[[0, 0]], 1.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[[0, 1]], 2.0, epsilon = 0.1);

        // Second feature should be close to zero
        assert_abs_diff_eq!(coef[[1, 0]], 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[[1, 1]], 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_multi_task_elastic_net_l1_ratio_extremes() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]];

        // Test with l1_ratio = 0 (pure Ridge)
        let model_ridge = MultiTaskElasticNet::new()
            .alpha(0.1)
            .l1_ratio(0.0)
            .fit_intercept(false);

        let fitted_ridge = model_ridge.fit(&x, &y).unwrap();
        let coef_ridge = fitted_ridge.coef();

        // Test with l1_ratio = 1 (pure Lasso)
        let model_lasso = MultiTaskElasticNet::new()
            .alpha(0.1)
            .l1_ratio(1.0)
            .fit_intercept(false);

        let fitted_lasso = model_lasso.fit(&x, &y).unwrap();
        let coef_lasso = fitted_lasso.coef();

        // Ridge and Lasso should produce different coefficients
        // With the same alpha, pure Lasso (l1_ratio=1) should produce sparser solutions
        assert!((coef_ridge[[0, 0]] - coef_lasso[[0, 0]]).abs() > 1e-4);
        assert!((coef_ridge[[0, 1]] - coef_lasso[[0, 1]]).abs() > 1e-4);
    }

    #[test]
    fn test_multi_task_elastic_net_with_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![
            [3.0, 5.0],
            [5.0, 7.0],
            [7.0, 9.0],
            [9.0, 11.0],
            [11.0, 13.0]
        ];

        let model = MultiTaskElasticNet::new()
            .alpha(0.05)
            .l1_ratio(0.7)
            .fit_intercept(true);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[5, 2]);

        // Check intercepts
        let intercept = fitted.intercept();
        assert_eq!(intercept.len(), 2);

        // Intercepts should be positive for this data
        assert!(intercept[0] > 0.0);
        assert!(intercept[1] > 0.0);
    }

    #[test]
    fn test_multi_task_elastic_net_sparsity() {
        // Create a simple dataset where sparsity can be verified
        let n_samples = 20;
        let n_features = 5;
        let n_tasks = 2;

        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array2::zeros((n_samples, n_tasks));

        // Create orthogonal features for better sparsity
        for i in 0..n_samples {
            // First feature: strong signal
            x[[i, 0]] = (i as Float) / (n_samples as Float);
            // Second feature: weak signal
            x[[i, 1]] = if i % 2 == 0 { 1.0 } else { -1.0 };
            // Rest: noise
            for j in 2..n_features {
                x[[i, j]] = 0.1 * ((i * j) as Float * 0.1).sin();
            }

            // Y depends mainly on first feature
            for t in 0..n_tasks {
                y[[i, t]] = 3.0 * x[[i, 0]] * (t + 1) as Float;
            }
        }

        let model = MultiTaskElasticNet::new()
            .alpha(0.1)
            .l1_ratio(0.95) // Very high L1 ratio for sparsity
            .fit_intercept(false);

        let fitted = model.fit(&x, &y).unwrap();
        let coef = fitted.coef();

        // Count non-zero features
        let mut non_zero_features = 0;
        let mut feature_norms = Vec::new();

        for j in 0..n_features {
            let mut norm = 0.0;
            for t in 0..n_tasks {
                norm += coef[[j, t]].powi(2);
            }
            norm = norm.sqrt();
            feature_norms.push(norm);

            if norm > 1e-6 {
                non_zero_features += 1;
            }
        }

        // Should have sparse solution
        assert!(
            non_zero_features <= 3,
            "Too many non-zero features: {}",
            non_zero_features
        );

        // First feature should have largest coefficient
        let max_norm_idx = feature_norms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert_eq!(
            max_norm_idx, 0,
            "First feature should have largest coefficient"
        );
    }

    #[test]
    fn test_builder_pattern() {
        let model = MultiTaskElasticNet::builder()
            .alpha(0.3)
            .l1_ratio(0.6)
            .max_iter(800)
            .tol(1e-5)
            .fit_intercept(false)
            .build();

        assert_eq!(model.config.alpha, 0.3);
        assert_eq!(model.config.l1_ratio, 0.6);
        assert_eq!(model.config.max_iter, 800);
        assert_eq!(model.config.tol, 1e-5);
        assert!(!model.config.fit_intercept);
    }

    #[test]
    #[should_panic(expected = "l1_ratio must be between 0 and 1")]
    fn test_invalid_l1_ratio() {
        MultiTaskElasticNet::new().l1_ratio(1.5);
    }
}
