//! Ridge Regularized Covariance Estimator
//!
//! L2 regularized covariance estimation.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for RidgeCovariance estimator
#[derive(Debug, Clone)]
pub struct RidgeCovarianceConfig {
    /// Regularization parameter (positive scalar)
    pub alpha: f64,
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
}

/// Ridge Regularized Covariance Estimator
///
/// Covariance estimator with L2 (Ridge) regularization. This adds a small
/// positive value to the diagonal of the empirical covariance matrix,
/// improving numerical stability and conditioning.
///
/// The estimator computes:
/// C_ridge = C_empirical + alpha * I
///
/// where alpha is the regularization parameter.
///
/// # Parameters
///
/// * `alpha` - Regularization parameter (positive scalar)
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::RidgeCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::{array, ArrayView2};
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let estimator = RidgeCovariance::new().alpha(0.1);
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// ```
#[derive(Debug, Clone)]
pub struct RidgeCovariance<S = Untrained> {
    state: S,
    config: RidgeCovarianceConfig,
}

/// Trained state for RidgeCovariance
#[derive(Debug, Clone)]
pub struct RidgeCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The regularization parameter used
    pub alpha: f64,
}

impl RidgeCovariance<Untrained> {
    /// Create a new RidgeCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: RidgeCovarianceConfig {
                alpha: 0.01,
                store_precision: true,
                assume_centered: false,
            },
        }
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha.max(0.0);
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

impl Default for RidgeCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RidgeCovariance<Untrained> {
    type Config = RidgeCovarianceConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RidgeCovariance<Untrained> {
    type Fitted = RidgeCovariance<RidgeCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute empirical covariance
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x, &())?;

        let mut covariance = emp_cov.get_covariance().clone();
        let location = emp_cov.get_location().clone();

        // Add Ridge regularization to diagonal
        for i in 0..n_features {
            covariance[[i, i]] += self.config.alpha;
        }

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(RidgeCovariance {
            state: RidgeCovarianceTrained {
                covariance,
                precision,
                location,
                alpha: self.config.alpha,
            },
            config: self.config,
        })
    }
}

impl RidgeCovariance<RidgeCovarianceTrained> {
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

    /// Get the regularization parameter used
    pub fn get_alpha(&self) -> f64 {
        self.state.alpha
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
