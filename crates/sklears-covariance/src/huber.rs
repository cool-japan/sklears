//! Huber Robust Covariance Estimator
//!
//! M-estimator for robust covariance estimation using Huber's function.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for HuberCovariance estimator
#[derive(Debug, Clone)]
pub struct HuberCovarianceConfig {
    /// The threshold for Huber's function (controls robustness)
    pub delta: f64,
    /// Maximum number of iterations for the iterative algorithm
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
}

/// Huber Robust Covariance Estimator
///
/// A robust M-estimator for covariance that uses Huber's loss function
/// to downweight outliers. This estimator is more robust to outliers
/// than the empirical covariance but less robust than MinCovDet.
///
/// # Parameters
///
/// * `delta` - The threshold for Huber's function (controls robustness)
/// * `max_iter` - Maximum number of iterations for the iterative algorithm
/// * `tol` - Convergence tolerance
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::HuberCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [100.0, 100.0]]; // Last point is outlier
///
/// let estimator = HuberCovariance::new().delta(1.345);
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// ```
#[derive(Debug, Clone)]
pub struct HuberCovariance<S = Untrained> {
    state: S,
    config: HuberCovarianceConfig,
}

/// Trained state for HuberCovariance
#[derive(Debug, Clone)]
pub struct HuberCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Weights for each sample
    pub weights: Array1<f64>,
}

impl HuberCovariance<Untrained> {
    /// Create a new HuberCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: HuberCovarianceConfig {
                delta: 1.345, // Standard choice for 95% efficiency at Gaussian
                max_iter: 100,
                tol: 1e-6,
                store_precision: true,
                assume_centered: false,
            },
        }
    }

    /// Set the delta parameter for Huber's function
    pub fn delta(mut self, delta: f64) -> Self {
        self.config.delta = delta.max(0.1); // Prevent too small delta
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.config.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.config.assume_centered = assume_centered;
        self
    }
}

impl Default for HuberCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HuberCovariance<Untrained> {
    type Config = HuberCovarianceConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for HuberCovariance<Untrained> {
    type Fitted = HuberCovariance<HuberCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Start with empirical estimates
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x, &())?;

        let mut location = emp_cov.get_location().clone();
        let mut covariance = emp_cov.get_covariance().clone();
        let mut weights = Array1::ones(n_samples);

        // Iterative reweighting algorithm
        let mut n_iter = 0;
        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;
            let old_location = location.clone();
            let old_covariance = covariance.clone();

            // Compute Mahalanobis distances
            let precision = matrix_inverse(&covariance)?;
            let mut distances = Array1::zeros(n_samples);

            for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
                let centered = &sample - &location;
                let temp = precision.dot(&centered);
                distances[i] = centered.dot(&temp).sqrt();
            }

            // Update weights using Huber function
            for (i, &dist) in distances.iter().enumerate() {
                weights[i] = self.huber_weight(dist);
            }

            // Update location (weighted mean)
            if !self.config.assume_centered {
                let weight_sum = weights.sum();
                location.fill(0.0);
                for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
                    location += &(sample.to_owned() * weights[i]);
                }
                location /= weight_sum;
            }

            // Update covariance (weighted)
            covariance.fill(0.0);
            let mut weight_sum = 0.0;
            for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
                let centered = &sample - &location;
                let weight = weights[i];
                weight_sum += weight;

                for j in 0..n_features {
                    for k in 0..n_features {
                        covariance[[j, k]] += weight * centered[j] * centered[k];
                    }
                }
            }
            covariance /= weight_sum;

            // Check convergence
            let location_diff = (&location - &old_location).mapv(|x| x.abs()).sum();
            let cov_diff = (&covariance - &old_covariance).mapv(|x| x.abs()).sum();

            if location_diff < self.config.tol && cov_diff < self.config.tol {
                break;
            }
        }

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(HuberCovariance {
            state: HuberCovarianceTrained {
                covariance,
                precision,
                location,
                n_iter,
                weights,
            },
            config: self.config,
        })
    }
}

impl HuberCovariance<Untrained> {
    /// Compute Huber weight function
    fn huber_weight(&self, distance: f64) -> f64 {
        if distance <= self.config.delta {
            1.0
        } else {
            self.config.delta / distance
        }
    }
}

impl HuberCovariance<HuberCovarianceTrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the location (mean)
    pub fn get_location(&self) -> &Array1<f64> {
        &self.state.location
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the weights for each sample
    pub fn get_weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Compute Mahalanobis distance
    pub fn mahalanobis_distance(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let x = *x;
        let precision = self.state.precision.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Precision matrix not computed".to_string())
        })?;

        let mut distances = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let centered = &sample - &self.state.location;
            let temp = precision.dot(&centered);
            distances[i] = centered.dot(&temp).sqrt();
        }

        Ok(distances)
    }
}
