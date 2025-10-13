//! Shrunk Covariance Estimator

use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Shrunk Covariance Estimator
///
/// Covariance estimator with shrinkage towards the identity matrix.
/// This can help when the sample size is small relative to the number of features.
///
/// # Parameters
///
/// * `shrinkage` - Shrinkage parameter (0 = no shrinkage, 1 = full shrinkage to identity)
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::ShrunkCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let estimator = ShrunkCovariance::new().shrinkage(0.1);
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// ```
#[derive(Debug, Clone)]
pub struct ShrunkCovariance<S = Untrained> {
    state: S,
    shrinkage: f64,
    store_precision: bool,
    assume_centered: bool,
}

/// Trained state for ShrunkCovariance
#[derive(Debug, Clone)]
pub struct ShrunkCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The shrinkage parameter used
    pub shrinkage: f64,
}

impl ShrunkCovariance<Untrained> {
    /// Create a new ShrunkCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            shrinkage: 0.1,
            store_precision: true,
            assume_centered: false,
        }
    }

    /// Set the shrinkage parameter
    pub fn shrinkage(mut self, shrinkage: f64) -> Self {
        self.shrinkage = shrinkage.clamp(0.0, 1.0);
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

impl Default for ShrunkCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ShrunkCovariance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ShrunkCovariance<Untrained> {
    type Fitted = ShrunkCovariance<ShrunkCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute mean if not assumed centered
        let mean = if self.assume_centered {
            Array1::zeros(n_features)
        } else {
            x.mean_axis(Axis(0)).unwrap()
        };

        // Center the data
        let mut x_centered = x.to_owned();
        if !self.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= &mean;
            }
        }

        // Compute empirical covariance
        let emp_cov = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Apply shrinkage towards identity
        let trace = emp_cov.diag().sum();
        let mu = trace / n_features as f64;
        let mut identity = Array2::eye(n_features);
        identity *= mu;

        let covariance = (1.0 - self.shrinkage) * &emp_cov + self.shrinkage * &identity;

        // Compute precision matrix if requested
        let precision = if self.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(ShrunkCovariance {
            state: ShrunkCovarianceTrained {
                covariance,
                precision,
                location: mean,
                shrinkage: self.shrinkage,
            },
            shrinkage: self.shrinkage,
            store_precision: self.store_precision,
            assume_centered: self.assume_centered,
        })
    }
}

impl ShrunkCovariance<ShrunkCovarianceTrained> {
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

    /// Get the shrinkage parameter used
    pub fn get_shrinkage(&self) -> f64 {
        self.state.shrinkage
    }
}
