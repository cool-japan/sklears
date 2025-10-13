//! Adaptive Lasso Covariance Estimator
//!
//! Implements sparse covariance estimation using adaptive lasso regularization.
//! The adaptive lasso uses adaptive weights for different precision matrix elements,
//! allowing for better feature selection and sparse structure recovery.

use crate::utils::{matrix_determinant, matrix_inverse};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Adaptive Lasso Covariance Estimator
///
/// Estimates sparse covariance matrices using adaptive lasso regularization.
/// The adaptive lasso assigns different penalty weights to different elements
/// of the precision matrix, typically based on an initial estimate.
///
/// # Parameters
///
/// * `alpha` - Regularization strength
/// * `gamma` - Power parameter for adaptive weights (typically 1 or 2)
/// * `initial_estimator` - Initial estimator for computing adaptive weights
/// * `max_iter` - Maximum number of iterations for optimization
/// * `tol` - Convergence tolerance
/// * `store_precision` - Whether to store the precision matrix
///
/// # Examples
///
/// ```
/// use sklears_covariance::AdaptiveLassoCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 0.5], [0.5, 2.0], [2.0, 1.0]];
///
/// let estimator = AdaptiveLassoCovariance::new()
///     .alpha(0.1)
///     .gamma(1.0);
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let precision = fitted.get_precision().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveLassoCovariance<S = Untrained> {
    state: S,
    alpha: f64,
    gamma: f64,
    max_iter: usize,
    tol: f64,
    store_precision: bool,
    assume_centered: bool,
}

/// Trained state for AdaptiveLassoCovariance
#[derive(Debug, Clone)]
pub struct AdaptiveLassoCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The adaptive weights used in optimization
    pub weights: Array2<f64>,
    /// Number of iterations used in optimization
    pub n_iter: usize,
    /// Final objective value
    pub objective: f64,
}

impl AdaptiveLassoCovariance<Untrained> {
    /// Create a new AdaptiveLassoCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 0.1,
            gamma: 1.0,
            max_iter: 100,
            tol: 1e-4,
            store_precision: true,
            assume_centered: false,
        }
    }

    /// Set the regularization strength
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Set the power parameter for adaptive weights
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma.max(0.0);
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(0.0);
        self
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }
}

impl Default for AdaptiveLassoCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AdaptiveLassoCovariance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AdaptiveLassoCovariance<Untrained> {
    type Fitted = AdaptiveLassoCovariance<AdaptiveLassoCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute empirical covariance as initial estimate
        let mut empirical_cov = compute_empirical_covariance(&x, self.assume_centered)?;

        // Add small regularization to diagonal for numerical stability
        for i in 0..n_features {
            empirical_cov[[i, i]] += 1e-6;
        }

        // Compute initial precision matrix for adaptive weights
        let initial_precision = matrix_inverse(&empirical_cov)?;

        // Compute adaptive weights
        let weights = compute_adaptive_weights(&initial_precision, self.gamma);

        // Perform adaptive lasso optimization
        let (precision, n_iter, objective) = adaptive_lasso_optimization(
            &empirical_cov,
            &weights,
            self.alpha,
            self.max_iter,
            self.tol,
        )?;

        // Compute covariance from precision
        let covariance = matrix_inverse(&precision)?;

        let stored_precision = if self.store_precision {
            Some(precision)
        } else {
            None
        };

        Ok(AdaptiveLassoCovariance {
            state: AdaptiveLassoCovarianceTrained {
                covariance,
                precision: stored_precision,
                weights,
                n_iter,
                objective,
            },
            alpha: self.alpha,
            gamma: self.gamma,
            max_iter: self.max_iter,
            tol: self.tol,
            store_precision: self.store_precision,
            assume_centered: self.assume_centered,
        })
    }
}

impl AdaptiveLassoCovariance<AdaptiveLassoCovarianceTrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the adaptive weights used in optimization
    pub fn get_weights(&self) -> &Array2<f64> {
        &self.state.weights
    }

    /// Get the number of iterations used in optimization
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the final objective value
    pub fn get_objective(&self) -> f64 {
        self.state.objective
    }

    /// Get the regularization strength
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the power parameter for adaptive weights
    pub fn get_gamma(&self) -> f64 {
        self.gamma
    }
}

/// Compute empirical covariance matrix
fn compute_empirical_covariance(
    x: &ArrayView2<Float>,
    assume_centered: bool,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();

    // Compute mean
    let mean = if assume_centered {
        Array1::zeros(n_features)
    } else {
        x.mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::InvalidInput("Failed to compute mean".to_string()))?
    };

    // Compute covariance
    let mut cov = Array2::zeros((n_features, n_features));

    for sample in x.axis_iter(Axis(0)) {
        let centered = &sample - &mean;
        for i in 0..n_features {
            for j in 0..n_features {
                cov[[i, j]] += centered[i] * centered[j];
            }
        }
    }

    cov /= (n_samples - 1) as f64;

    Ok(cov)
}

/// Compute adaptive weights based on initial estimate
fn compute_adaptive_weights(initial_precision: &Array2<f64>, gamma: f64) -> Array2<f64> {
    let (n_features, _) = initial_precision.dim();
    let mut weights = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            if i != j {
                // Adaptive weight is 1 / |beta|^gamma where beta is the initial estimate
                let abs_value = initial_precision[[i, j]].abs();
                weights[[i, j]] = if abs_value > 1e-8 {
                    1.0 / abs_value.powf(gamma)
                } else {
                    1e8 // Large weight for near-zero entries
                };
            } else {
                weights[[i, j]] = 0.0; // No penalty on diagonal
            }
        }
    }

    weights
}

/// Perform adaptive lasso optimization using coordinate descent
fn adaptive_lasso_optimization(
    empirical_cov: &Array2<f64>,
    weights: &Array2<f64>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> SklResult<(Array2<f64>, usize, f64)> {
    let n_features = empirical_cov.nrows();

    // Initialize precision matrix as empirical precision (regularized)
    let mut precision = matrix_inverse(empirical_cov)?;

    // Add small regularization to diagonal to ensure positive definiteness
    for i in 0..n_features {
        precision[[i, i]] += alpha * 0.01;
    }

    let mut objective_old = f64::INFINITY;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let mut precision_new = precision.clone();

        // Coordinate descent: update each off-diagonal element
        for i in 0..n_features {
            for j in 0..n_features {
                if i != j {
                    // Soft thresholding for adaptive lasso
                    let grad = empirical_cov[[i, j]] - precision[[i, j]];
                    let penalty = alpha * weights[[i, j]];

                    precision_new[[i, j]] = soft_threshold(grad, penalty);
                    precision_new[[j, i]] = precision_new[[i, j]]; // Maintain symmetry
                }
            }
        }

        // Compute objective function (negative log-likelihood + penalty)
        let log_det = matrix_determinant(&precision_new).ln();
        let trace_term = (empirical_cov * &precision_new).diag().sum();

        let penalty_term: f64 = (0..n_features)
            .flat_map(|i| (0..n_features).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| alpha * weights[[i, j]] * precision_new[[i, j]].abs())
            .sum();

        let objective = -log_det + trace_term + penalty_term;

        // Check for convergence
        if (objective - objective_old).abs() < tol {
            break;
        }

        precision = precision_new;
        objective_old = objective;
    }

    Ok((precision, n_iter, objective_old))
}

/// Soft thresholding function
fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_lasso_basic() {
        let x = array![
            [1.0, 0.5, 0.1],
            [0.5, 2.0, 0.2],
            [2.0, 1.0, 0.3],
            [1.5, 1.5, 0.4]
        ];

        let estimator = AdaptiveLassoCovariance::new()
            .alpha(0.1)
            .gamma(1.0)
            .max_iter(50);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_precision().is_some());
        assert_eq!(fitted.get_weights().dim(), (3, 3));
        assert_eq!(fitted.get_alpha(), 0.1);
        assert_eq!(fitted.get_gamma(), 1.0);
        assert!(fitted.get_n_iter() > 0);
    }

    #[test]
    fn test_adaptive_weights() {
        let precision = array![[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]];

        let weights = compute_adaptive_weights(&precision, 1.0);

        // Check that diagonal weights are zero
        assert_eq!(weights[[0, 0]], 0.0);
        assert_eq!(weights[[1, 1]], 0.0);
        assert_eq!(weights[[2, 2]], 0.0);

        // Check that off-diagonal weights are positive
        assert!(weights[[0, 1]] > 0.0);
        assert!(weights[[1, 0]] > 0.0);

        // Check symmetry
        assert_eq!(weights[[0, 1]], weights[[1, 0]]);
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.5, 1.0), 0.0);
    }
}
