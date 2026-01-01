//! Robust Kernel Ridge Regression Implementation
//!
//! This module implements robust variants of kernel ridge regression that are resistant
//! to outliers and noise in the data. Multiple robust loss functions are supported,
//! and the optimization is performed using iteratively reweighted least squares (IRLS).

use crate::{
    FastfoodTransform, Nystroem, RBFSampler, StructuredRandomFeatures, Trained, Untrained,
};
use scirs2_linalg::compat::ArrayLinalgExt;
// Removed SVD import - using ArrayLinalgExt for both solve and svd methods

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::{Estimator, Fit, Float, Predict};
use std::marker::PhantomData;

use super::core_types::*;

/// Robust kernel ridge regression
///
/// This implements robust variants of kernel ridge regression that are resistant
/// to outliers and noise in the data. Multiple robust loss functions are supported.
///
/// # Parameters
///
/// * `approximation_method` - Method for kernel approximation
/// * `alpha` - Regularization strength
/// * `robust_loss` - Robust loss function to use
/// * `solver` - Method for solving the optimization problem
/// * `max_iter` - Maximum number of iterations for robust optimization
/// * `tolerance` - Convergence tolerance
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::kernel_ridge_regression::{
#[derive(Debug, Clone)]
pub struct RobustKernelRidgeRegression<State = Untrained> {
    pub approximation_method: ApproximationMethod,
    pub alpha: Float,
    pub robust_loss: RobustLoss,
    pub solver: Solver,
    pub max_iter: usize,
    pub tolerance: Float,
    pub random_state: Option<u64>,

    // Fitted parameters
    weights_: Option<Array1<Float>>,
    feature_transformer_: Option<FeatureTransformer>,
    sample_weights_: Option<Array1<Float>>, // Adaptive weights for robustness

    _state: PhantomData<State>,
}

/// Robust loss functions for kernel ridge regression
#[derive(Debug, Clone)]
pub enum RobustLoss {
    /// Huber loss - quadratic for small residuals, linear for large ones
    Huber { delta: Float },
    /// Epsilon-insensitive loss (used in SVR)
    EpsilonInsensitive { epsilon: Float },
    /// Quantile loss for quantile regression
    Quantile { tau: Float },
    /// Tukey's biweight loss
    Tukey { c: Float },
    /// Cauchy loss
    Cauchy { sigma: Float },
    /// Logistic loss
    Logistic { scale: Float },
    /// Fair loss
    Fair { c: Float },
    /// Welsch loss
    Welsch { c: Float },
    /// Custom robust loss function
    Custom {
        loss_fn: fn(Float) -> Float,
        weight_fn: fn(Float) -> Float,
    },
}

impl Default for RobustLoss {
    fn default() -> Self {
        Self::Huber { delta: 1.0 }
    }
}

impl RobustLoss {
    /// Compute the loss for a given residual
    pub fn loss(&self, residual: Float) -> Float {
        let abs_r = residual.abs();
        match self {
            RobustLoss::Huber { delta } => {
                if abs_r <= *delta {
                    0.5 * residual * residual
                } else {
                    delta * (abs_r - 0.5 * delta)
                }
            }
            RobustLoss::EpsilonInsensitive { epsilon } => (abs_r - epsilon).max(0.0),
            RobustLoss::Quantile { tau } => {
                if residual >= 0.0 {
                    tau * residual
                } else {
                    (tau - 1.0) * residual
                }
            }
            RobustLoss::Tukey { c } => {
                if abs_r <= *c {
                    let r_norm = residual / c;
                    (c * c / 6.0) * (1.0 - (1.0 - r_norm * r_norm).powi(3))
                } else {
                    c * c / 6.0
                }
            }
            RobustLoss::Cauchy { sigma } => {
                (sigma * sigma / 2.0) * ((1.0 + (residual / sigma).powi(2)).ln())
            }
            RobustLoss::Logistic { scale } => scale * (1.0 + (-abs_r / scale).exp()).ln(),
            RobustLoss::Fair { c } => c * (abs_r / c - (1.0 + abs_r / c).ln()),
            RobustLoss::Welsch { c } => (c * c / 2.0) * (1.0 - (-((residual / c).powi(2))).exp()),
            RobustLoss::Custom { loss_fn, .. } => loss_fn(residual),
        }
    }

    /// Compute the weight for iteratively reweighted least squares
    pub fn weight(&self, residual: Float) -> Float {
        let abs_r = residual.abs();
        if abs_r < 1e-10 {
            return 1.0; // Avoid division by zero
        }

        match self {
            RobustLoss::Huber { delta } => {
                if abs_r <= *delta {
                    1.0
                } else {
                    delta / abs_r
                }
            }
            RobustLoss::EpsilonInsensitive { epsilon } => {
                if abs_r <= *epsilon {
                    0.0
                } else {
                    1.0
                }
            }
            RobustLoss::Quantile { tau } => {
                // For quantile regression, weights are constant
                if residual >= 0.0 {
                    *tau
                } else {
                    1.0 - tau
                }
            }
            RobustLoss::Tukey { c } => {
                if abs_r <= *c {
                    let r_norm = residual / c;
                    (1.0 - r_norm * r_norm).powi(2)
                } else {
                    0.0
                }
            }
            RobustLoss::Cauchy { sigma } => 1.0 / (1.0 + (residual / sigma).powi(2)),
            RobustLoss::Logistic { scale } => {
                let exp_term = (-abs_r / scale).exp();
                exp_term / (1.0 + exp_term)
            }
            RobustLoss::Fair { c } => 1.0 / (1.0 + abs_r / c),
            RobustLoss::Welsch { c } => (-((residual / c).powi(2))).exp(),
            RobustLoss::Custom { weight_fn, .. } => weight_fn(residual),
        }
    }
}

impl RobustKernelRidgeRegression<Untrained> {
    /// Create a new robust kernel ridge regression model
    pub fn new(approximation_method: ApproximationMethod) -> Self {
        Self {
            approximation_method,
            alpha: 1.0,
            robust_loss: RobustLoss::default(),
            solver: Solver::Direct,
            max_iter: 100,
            tolerance: 1e-6,
            random_state: None,
            weights_: None,
            feature_transformer_: None,
            sample_weights_: None,
            _state: PhantomData,
        }
    }

    /// Set regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set robust loss function
    pub fn robust_loss(mut self, robust_loss: RobustLoss) -> Self {
        self.robust_loss = robust_loss;
        self
    }

    /// Set solver method
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set maximum iterations for robust optimization
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for RobustKernelRidgeRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for RobustKernelRidgeRegression<Untrained> {
    type Fitted = RobustKernelRidgeRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match".to_string(),
            ));
        }

        // Fit the feature transformer
        let feature_transformer = self.fit_feature_transformer(x)?;
        let x_transformed = feature_transformer.transform(x)?;

        // Solve robust regression using iteratively reweighted least squares (IRLS)
        let (weights, sample_weights) = self.solve_robust_regression(&x_transformed, y)?;

        Ok(RobustKernelRidgeRegression {
            approximation_method: self.approximation_method,
            alpha: self.alpha,
            robust_loss: self.robust_loss,
            solver: self.solver,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            random_state: self.random_state,
            weights_: Some(weights),
            feature_transformer_: Some(feature_transformer),
            sample_weights_: Some(sample_weights),
            _state: PhantomData,
        })
    }
}

impl RobustKernelRidgeRegression<Untrained> {
    /// Fit the feature transformer based on the approximation method
    fn fit_feature_transformer(&self, x: &Array2<Float>) -> Result<FeatureTransformer> {
        match &self.approximation_method {
            ApproximationMethod::Nystroem {
                kernel,
                n_components,
                sampling_strategy,
            } => {
                let mut nystroem = Nystroem::new(kernel.clone(), *n_components)
                    .sampling_strategy(sampling_strategy.clone());
                if let Some(seed) = self.random_state {
                    nystroem = nystroem.random_state(seed);
                }
                let fitted = nystroem.fit(x, &())?;
                Ok(FeatureTransformer::Nystroem(fitted))
            }
            ApproximationMethod::RandomFourierFeatures {
                n_components,
                gamma,
            } => {
                let mut rff = RBFSampler::new(*n_components).gamma(*gamma);
                if let Some(seed) = self.random_state {
                    rff = rff.random_state(seed);
                }
                let fitted = rff.fit(x, &())?;
                Ok(FeatureTransformer::RBFSampler(fitted))
            }
            ApproximationMethod::StructuredRandomFeatures {
                n_components,
                gamma,
            } => {
                let mut srf = StructuredRandomFeatures::new(*n_components).gamma(*gamma);
                if let Some(seed) = self.random_state {
                    srf = srf.random_state(seed);
                }
                let fitted = srf.fit(x, &())?;
                Ok(FeatureTransformer::StructuredRFF(fitted))
            }
            ApproximationMethod::Fastfood {
                n_components,
                gamma,
            } => {
                let mut fastfood = FastfoodTransform::new(*n_components).gamma(*gamma);
                if let Some(seed) = self.random_state {
                    fastfood = fastfood.random_state(seed);
                }
                let fitted = fastfood.fit(x, &())?;
                Ok(FeatureTransformer::Fastfood(fitted))
            }
        }
    }

    /// Solve robust regression using iteratively reweighted least squares
    fn solve_robust_regression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize with ordinary least squares solution
        let x_f64 = Array2::from_shape_fn(x.dim(), |(i, j)| x[[i, j]]);
        let y_f64 = Array1::from_vec(y.iter().copied().collect());

        let xtx = x_f64.t().dot(&x_f64);
        let regularized_xtx = xtx + Array2::<f64>::eye(n_features) * self.alpha;
        let xty = x_f64.t().dot(&y_f64);
        let mut weights_f64 =
            regularized_xtx
                .solve(&xty)
                .map_err(|e| SklearsError::InvalidParameter {
                    name: "regularization".to_string(),
                    reason: format!("Initial linear system solving failed: {:?}", e),
                })?;

        let mut sample_weights = Array1::ones(n_samples);
        let mut prev_weights = weights_f64.clone();

        // Iteratively reweighted least squares
        for _iter in 0..self.max_iter {
            // Compute residuals
            let predictions = x_f64.dot(&weights_f64);
            let residuals = &y_f64 - &predictions;

            // Update sample weights based on residuals
            for (i, &residual) in residuals.iter().enumerate() {
                sample_weights[i] = self.robust_loss.weight(residual as Float);
            }

            // Solve weighted least squares
            let mut weighted_xtx = Array2::zeros((n_features, n_features));
            let mut weighted_xty = Array1::zeros(n_features);

            for i in 0..n_samples {
                let weight = sample_weights[i];
                let x_row = x_f64.row(i);

                // X^T W X
                for j in 0..n_features {
                    for k in 0..n_features {
                        weighted_xtx[[j, k]] += weight * x_row[j] * x_row[k];
                    }
                }

                // X^T W y
                for j in 0..n_features {
                    weighted_xty[j] += weight * x_row[j] * y_f64[i];
                }
            }

            // Add regularization
            weighted_xtx += &(Array2::eye(n_features) * self.alpha);

            // Solve the weighted system
            weights_f64 = match self.solver {
                Solver::Direct => weighted_xtx.solve(&weighted_xty).map_err(|e| {
                    SklearsError::InvalidParameter {
                        name: "weighted_system".to_string(),
                        reason: format!("Weighted linear system solving failed: {:?}", e),
                    }
                })?,
                Solver::SVD => {
                    let (u, s, vt) =
                        weighted_xtx
                            .svd(true)
                            .map_err(|e| SklearsError::InvalidParameter {
                                name: "svd".to_string(),
                                reason: format!("SVD decomposition failed: {:?}", e),
                            })?;
                    let ut_b = u.t().dot(&weighted_xty);
                    let s_inv = s.mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 });
                    let y_svd = ut_b * s_inv;
                    vt.t().dot(&y_svd)
                }
                Solver::ConjugateGradient { max_iter, tol } => {
                    self.solve_cg_weighted(&weighted_xtx, &weighted_xty, max_iter, tol)?
                }
            };

            // Check convergence
            let weight_change = (&weights_f64 - &prev_weights).mapv(|x| x.abs()).sum();
            if weight_change < self.tolerance {
                break;
            }

            prev_weights = weights_f64.clone();
        }

        // Convert back to Float
        let weights = Array1::from_vec(weights_f64.iter().map(|&val| val as Float).collect());
        let sample_weights_float =
            Array1::from_vec(sample_weights.iter().map(|&val| val as Float).collect());

        Ok((weights, sample_weights_float))
    }

    /// Solve using conjugate gradient method
    fn solve_cg_weighted(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<Array1<f64>> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        let mut r = b - &a.dot(&x);
        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for _iter in 0..max_iter {
            let ap = a.dot(&p);
            let alpha = rsold / p.dot(&ap);

            x = x + &p * alpha;
            r = r - &ap * alpha;

            let rsnew = r.dot(&r);

            if rsnew.sqrt() < tol {
                break;
            }

            let beta = rsnew / rsold;
            p = &r + &p * beta;
            rsold = rsnew;
        }

        Ok(x)
    }
}

impl Predict<Array2<Float>, Array1<Float>> for RobustKernelRidgeRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let feature_transformer =
            self.feature_transformer_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let x_transformed = feature_transformer.transform(x)?;
        let predictions = x_transformed.dot(weights);

        Ok(predictions)
    }
}

impl RobustKernelRidgeRegression<Trained> {
    /// Get the fitted weights
    pub fn weights(&self) -> Option<&Array1<Float>> {
        self.weights_.as_ref()
    }

    /// Get the sample weights from robust fitting
    pub fn sample_weights(&self) -> Option<&Array1<Float>> {
        self.sample_weights_.as_ref()
    }

    /// Compute robust residuals and their weights
    pub fn robust_residuals(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let predictions = self.predict(x)?;
        let residuals = y - &predictions;

        let mut weights = Array1::zeros(residuals.len());
        for (i, &residual) in residuals.iter().enumerate() {
            weights[i] = self.robust_loss.weight(residual);
        }

        Ok((residuals, weights))
    }

    /// Get outlier scores (lower weight means more likely to be outlier)
    pub fn outlier_scores(&self) -> Option<Array1<Float>> {
        self.sample_weights_.as_ref().map(|weights| {
            // Convert weights to outlier scores (1 - weight)
            weights.mapv(|w| 1.0 - w)
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_robust_kernel_ridge_regression() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [10.0, 10.0]]; // Last point is outlier
        let y = array![1.0, 2.0, 3.0, 100.0]; // Last target is outlier

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 20,
            gamma: 0.1,
        };

        let robust_krr = RobustKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .robust_loss(RobustLoss::Huber { delta: 1.0 });

        let fitted = robust_krr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);

        // Check that predictions are reasonable
        for pred in predictions.iter() {
            assert!(pred.is_finite());
        }

        // Check that outlier has lower weight
        let sample_weights = fitted.sample_weights().unwrap();
        assert!(sample_weights[3] < sample_weights[0]); // Outlier should have lower weight
    }

    #[test]
    fn test_different_robust_losses() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let loss_functions = vec![
            RobustLoss::Huber { delta: 1.0 },
            RobustLoss::EpsilonInsensitive { epsilon: 0.1 },
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Tukey { c: 4.685 },
            RobustLoss::Cauchy { sigma: 1.0 },
        ];

        for loss in loss_functions {
            let robust_krr = RobustKernelRidgeRegression::new(approximation.clone())
                .alpha(0.1)
                .robust_loss(loss);

            let fitted = robust_krr.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 3);
        }
    }

    #[test]
    fn test_robust_loss_functions() {
        let huber = RobustLoss::Huber { delta: 1.0 };

        // Test loss computation
        assert_eq!(huber.loss(0.5), 0.125); // Quadratic region
        assert_eq!(huber.loss(2.0), 1.5); // Linear region

        // Test weight computation
        assert_eq!(huber.weight(0.5), 1.0); // Quadratic region
        assert_eq!(huber.weight(2.0), 0.5); // Linear region
    }

    #[test]
    fn test_robust_outlier_detection() {
        let x = array![[1.0], [2.0], [3.0], [100.0]]; // Last point is outlier
        let y = array![1.0, 2.0, 3.0, 100.0]; // Last target is outlier

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let robust_krr = RobustKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .robust_loss(RobustLoss::Huber { delta: 1.0 });

        let fitted = robust_krr.fit(&x, &y).unwrap();
        let outlier_scores = fitted.outlier_scores().unwrap();

        // Outlier should have higher score
        assert!(outlier_scores[3] > outlier_scores[0]);
        assert!(outlier_scores[3] > outlier_scores[1]);
        assert!(outlier_scores[3] > outlier_scores[2]);
    }

    #[test]
    fn test_robust_convergence() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let robust_krr = RobustKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .robust_loss(RobustLoss::Huber { delta: 1.0 })
            .max_iter(5) // Small number of iterations
            .tolerance(1e-3);

        let fitted = robust_krr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 3);
        // Should converge even with few iterations for this simple case
        for pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_robust_reproducibility() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let robust_krr1 = RobustKernelRidgeRegression::new(approximation.clone())
            .alpha(0.1)
            .robust_loss(RobustLoss::Huber { delta: 1.0 })
            .random_state(42);
        let fitted1 = robust_krr1.fit(&x, &y).unwrap();
        let pred1 = fitted1.predict(&x).unwrap();

        let robust_krr2 = RobustKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .robust_loss(RobustLoss::Huber { delta: 1.0 })
            .random_state(42);
        let fitted2 = robust_krr2.fit(&x, &y).unwrap();
        let pred2 = fitted2.predict(&x).unwrap();

        assert_eq!(pred1.len(), pred2.len());
        for i in 0..pred1.len() {
            assert!((pred1[i] - pred2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_robust_loss_edge_cases() {
        let losses = vec![
            RobustLoss::Huber { delta: 1.0 },
            RobustLoss::EpsilonInsensitive { epsilon: 0.1 },
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Tukey { c: 4.685 },
            RobustLoss::Cauchy { sigma: 1.0 },
            RobustLoss::Logistic { scale: 1.0 },
            RobustLoss::Fair { c: 1.0 },
            RobustLoss::Welsch { c: 1.0 },
        ];

        for loss in losses {
            // Test with zero residual
            let loss_zero = loss.loss(0.0);
            let weight_zero = loss.weight(0.0);

            assert!(loss_zero >= 0.0);
            assert!(weight_zero >= 0.0);
            assert!(weight_zero <= 1.5); // Most weights should be <= 1.0, allowing some tolerance

            // Test with non-zero residual
            let loss_nonzero = loss.loss(1.0);
            let weight_nonzero = loss.weight(1.0);

            assert!(loss_nonzero >= 0.0);
            assert!(weight_nonzero >= 0.0);
        }
    }

    #[test]
    fn test_custom_robust_loss() {
        let custom_loss = RobustLoss::Custom {
            loss_fn: |r| r * r, // Simple quadratic loss
            weight_fn: |_| 1.0, // Constant weight
        };

        assert_eq!(custom_loss.loss(2.0), 4.0);
        assert_eq!(custom_loss.weight(5.0), 1.0);
    }
}
