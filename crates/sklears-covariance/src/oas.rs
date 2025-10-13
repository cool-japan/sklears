//! Oracle Approximating Shrinkage (OAS) Estimator
//!
//! Covariance estimator that uses the Oracle Approximating Shrinkage approach
//! for optimal shrinkage estimation.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for OAS estimator
#[derive(Debug, Clone)]
pub struct OASConfig {
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
}

/// Oracle Approximating Shrinkage (OAS) Estimator
///
/// Covariance estimator that uses the Oracle Approximating Shrinkage approach
/// for optimal shrinkage estimation. OAS provides a more robust shrinkage
/// parameter estimation than Ledoit-Wolf in certain scenarios.
///
/// # Parameters
///
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::OAS;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let oas = OAS::new();
/// let fitted = oas.fit(&X.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// let shrinkage = fitted.get_shrinkage();
/// ```
#[derive(Debug, Clone)]
pub struct OAS<S = Untrained> {
    state: S,
    config: OASConfig,
}

/// Trained state for OAS
#[derive(Debug, Clone)]
pub struct OASTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The shrinkage parameter used
    pub shrinkage: f64,
}

impl OAS<Untrained> {
    /// Create a new OAS instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: OASConfig {
                store_precision: true,
                assume_centered: false,
            },
        }
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

impl Default for OAS<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OAS<Untrained> {
    type Config = OASConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for OAS<Untrained> {
    type Fitted = OAS<OASTrained>;

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

        let covariance_emp = emp_cov.get_covariance().clone();
        let location = emp_cov.get_location().clone();

        // Compute OAS shrinkage parameter
        let shrinkage = self.compute_oas_shrinkage(n_samples, n_features, &covariance_emp)?;

        // Apply shrinkage
        let trace = covariance_emp.diag().sum();
        let mu = trace / n_features as f64;
        let mut identity = Array2::eye(n_features);
        identity *= mu;

        let covariance = (1.0 - shrinkage) * &covariance_emp + shrinkage * &identity;

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(OAS {
            state: OASTrained {
                covariance,
                precision,
                location,
                shrinkage,
            },
            config: self.config,
        })
    }
}

impl OAS<Untrained> {
    fn compute_oas_shrinkage(
        &self,
        n_samples: usize,
        n_features: usize,
        covariance_emp: &Array2<f64>,
    ) -> SklResult<f64> {
        let n = n_samples as f64;
        let p = n_features as f64;

        // Compute trace of empirical covariance
        let trace = covariance_emp.diag().sum();
        let mu = trace / p;

        // Compute trace of squared empirical covariance
        let trace_s2 = covariance_emp.mapv(|x| x * x).sum();

        // OAS shrinkage formula
        let numerator = (n - 2.0) / n * trace_s2 + trace * trace;
        let denominator = (n + 1.0) * (trace_s2 - (trace * trace) / p);

        let shrinkage = if denominator > 0.0 {
            numerator / denominator
        } else {
            1.0
        };

        Ok(shrinkage.clamp(0.0, 1.0))
    }
}

impl OAS<OASTrained> {
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

    /// Get the shrinkage parameter
    pub fn get_shrinkage(&self) -> f64 {
        self.state.shrinkage
    }
}
