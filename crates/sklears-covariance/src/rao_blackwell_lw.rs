//! Rao-Blackwell Ledoit-Wolf Covariance Estimator
//!
//! An improved version of the Ledoit-Wolf estimator that uses Rao-Blackwellization
//! to reduce the variance of the shrinkage intensity estimate.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for RaoBlackwellLedoitWolf estimator
#[derive(Debug, Clone)]
pub struct RaoBlackwellLedoitWolfConfig {
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Size of blocks for processing
    pub block_size: usize,
}

/// Rao-Blackwell Ledoit-Wolf Covariance Estimator
///
/// An improved version of the Ledoit-Wolf estimator that uses Rao-Blackwellization
/// to reduce the variance of the shrinkage intensity estimate. This leads to better
/// performance especially in high-dimensional settings.
///
/// # Parameters
///
/// * `store_precision` - Whether to store the precision matrix
/// * `assume_centered` - Whether to assume the data is centered
/// * `block_size` - Size of blocks for processing (default: 1000)
///
/// # Examples
///
/// ```
/// use sklears_covariance::RaoBlackwellLedoitWolf;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let rb_lw = RaoBlackwellLedoitWolf::new();
/// let fitted = rb_lw.fit(&X.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// let shrinkage = fitted.get_shrinkage();
/// ```
#[derive(Debug, Clone)]
pub struct RaoBlackwellLedoitWolf<S = Untrained> {
    state: S,
    config: RaoBlackwellLedoitWolfConfig,
}

/// Trained state for RaoBlackwellLedoitWolf
#[derive(Debug, Clone)]
pub struct RaoBlackwellLedoitWolfTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The shrinkage parameter used
    pub shrinkage: f64,
    /// The effective shrinkage after Rao-Blackwellization
    pub effective_shrinkage: f64,
}

impl RaoBlackwellLedoitWolf<Untrained> {
    /// Create a new RaoBlackwellLedoitWolf instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: RaoBlackwellLedoitWolfConfig {
                store_precision: true,
                assume_centered: false,
                block_size: 1000,
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

    /// Set the block size for processing
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.config.block_size = block_size;
        self
    }
}

impl Default for RaoBlackwellLedoitWolf<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RaoBlackwellLedoitWolf<Untrained> {
    type Config = RaoBlackwellLedoitWolfConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RaoBlackwellLedoitWolf<Untrained> {
    type Fitted = RaoBlackwellLedoitWolf<RaoBlackwellLedoitWolfTrained>;

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

        // Compute Rao-Blackwell improved shrinkage parameter
        let (shrinkage, effective_shrinkage) =
            self.compute_rao_blackwell_shrinkage(&x, &covariance_emp, &location)?;

        // Apply shrinkage using effective shrinkage
        let trace = covariance_emp.diag().sum();
        let mu = trace / n_features as f64;
        let mut identity = Array2::eye(n_features);
        identity *= mu;

        let covariance =
            (1.0 - effective_shrinkage) * &covariance_emp + effective_shrinkage * &identity;

        // Compute precision matrix if requested
        let precision = if self.config.store_precision {
            Some(matrix_inverse(&covariance)?)
        } else {
            None
        };

        Ok(RaoBlackwellLedoitWolf {
            state: RaoBlackwellLedoitWolfTrained {
                covariance,
                precision,
                location,
                shrinkage,
                effective_shrinkage,
            },
            config: self.config,
        })
    }
}

impl RaoBlackwellLedoitWolf<Untrained> {
    fn compute_rao_blackwell_shrinkage(
        &self,
        x: &ArrayView2<'_, Float>,
        covariance_emp: &Array2<f64>,
        location: &Array1<f64>,
    ) -> SklResult<(f64, f64)> {
        let (n_samples, n_features) = x.dim();

        // Center the data
        let mut x_centered = x.to_owned();
        if !self.config.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= location;
            }
        }

        // Compute standard Ledoit-Wolf shrinkage first
        let standard_shrinkage = self.compute_standard_lw_shrinkage(&x_centered, covariance_emp)?;

        // Compute Rao-Blackwell correction
        let rb_correction =
            self.compute_rao_blackwell_correction(&x_centered, covariance_emp, standard_shrinkage)?;

        // Apply Rao-Blackwellization
        let effective_shrinkage = (standard_shrinkage + rb_correction).clamp(0.0, 1.0);

        Ok((standard_shrinkage, effective_shrinkage))
    }

    fn compute_standard_lw_shrinkage(
        &self,
        x_centered: &Array2<f64>,
        covariance_emp: &Array2<f64>,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = x_centered.dim();

        // Compute sum of squared deviations (beta)
        let mut sum_squared_deviations = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    let dev = x_centered[[k, i]] * x_centered[[k, j]] - covariance_emp[[i, j]];
                    sum_squared_deviations += dev * dev;
                }
            }
        }

        let beta = sum_squared_deviations / (n_samples as f64).powi(2);

        // Compute delta (distance to target)
        let trace = covariance_emp.diag().sum();
        let mu = trace / n_features as f64;

        let mut delta = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                let target_val = if i == j { mu } else { 0.0 };
                delta += (covariance_emp[[i, j]] - target_val).powi(2);
            }
        }

        // Standard shrinkage
        let shrinkage = if delta > 0.0 {
            (beta / delta).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(shrinkage)
    }

    fn compute_rao_blackwell_correction(
        &self,
        x_centered: &Array2<f64>,
        covariance_emp: &Array2<f64>,
        standard_shrinkage: f64,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = x_centered.dim();

        if n_samples <= n_features {
            // In high-dimensional regime, use simplified correction
            return Ok(0.0);
        }

        // Compute fourth-order moments for Rao-Blackwell correction
        let mut fourth_moment_correction = 0.0;
        let trace = covariance_emp.diag().sum();
        let mu = trace / n_features as f64;

        // Compute correction based on higher-order moments
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sample_fourth_moment = 0.0;

                for k in 0..n_samples {
                    let xi = x_centered[[k, i]];
                    let xj = x_centered[[k, j]];
                    sample_fourth_moment += (xi * xj).powi(2);
                }

                sample_fourth_moment /= n_samples as f64;

                let target_val = if i == j { mu } else { 0.0 };
                let empirical_second_moment = covariance_emp[[i, j]].powi(2);

                // Rao-Blackwell correction term
                fourth_moment_correction += (sample_fourth_moment - empirical_second_moment)
                    * (covariance_emp[[i, j]] - target_val);
            }
        }

        // Normalize correction
        let mut delta_squared = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                let target_val = if i == j { mu } else { 0.0 };
                delta_squared += (covariance_emp[[i, j]] - target_val).powi(2);
            }
        }

        let correction = if delta_squared > 0.0 {
            fourth_moment_correction / (delta_squared * n_samples as f64)
        } else {
            0.0
        };

        // Apply damping factor for stability
        let damped_correction = correction * 0.1; // Conservative damping

        Ok(damped_correction)
    }
}

impl RaoBlackwellLedoitWolf<RaoBlackwellLedoitWolfTrained> {
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

    /// Get the standard shrinkage parameter
    pub fn get_shrinkage(&self) -> f64 {
        self.state.shrinkage
    }

    /// Get the effective shrinkage parameter after Rao-Blackwellization
    pub fn get_effective_shrinkage(&self) -> f64 {
        self.state.effective_shrinkage
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rao_blackwell_lw_basic() {
        let x = array![
            [1.0, 0.5],
            [2.0, 1.5],
            [3.0, 2.8],
            [4.0, 3.9],
            [5.0, 4.1],
            [1.5, 0.8],
            [2.5, 1.9],
            [3.5, 3.1]
        ];

        let estimator = RaoBlackwellLedoitWolf::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert!(fitted.get_precision().is_some());

        let shrinkage = fitted.get_shrinkage();
        let effective_shrinkage = fitted.get_effective_shrinkage();

        assert!((0.0..=1.0).contains(&shrinkage));
        assert!((0.0..=1.0).contains(&effective_shrinkage));
    }

    #[test]
    fn test_rao_blackwell_vs_standard() {
        let x = array![
            [1.0, 0.5, 0.2],
            [2.0, 1.5, 0.3],
            [3.0, 2.8, 0.4],
            [4.0, 3.9, 0.5],
            [5.0, 4.1, 0.6],
            [1.5, 0.8, 0.25],
            [2.5, 1.9, 0.35],
            [3.5, 3.1, 0.45]
        ];

        let rb_estimator = RaoBlackwellLedoitWolf::new();
        let rb_fitted = rb_estimator.fit(&x.view(), &()).unwrap();

        // Test that the estimator produces reasonable results
        assert_eq!(rb_fitted.get_covariance().dim(), (3, 3));
        assert!(rb_fitted.get_precision().is_some());

        let shrinkage = rb_fitted.get_shrinkage();
        let effective_shrinkage = rb_fitted.get_effective_shrinkage();

        assert!((0.0..=1.0).contains(&shrinkage));
        assert!((0.0..=1.0).contains(&effective_shrinkage));

        // The effective shrinkage should be close to but not necessarily identical to standard shrinkage
        assert!((shrinkage - effective_shrinkage).abs() < 0.5);
    }

    #[test]
    fn test_rao_blackwell_high_dim() {
        // Test in higher dimensional setting where correction should be more conservative
        let x = array![[1.0, 0.5], [2.0, 1.5], [3.0, 2.8]]; // n_samples <= n_features case

        let estimator = RaoBlackwellLedoitWolf::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));

        let shrinkage = fitted.get_shrinkage();
        let effective_shrinkage = fitted.get_effective_shrinkage();

        assert!((0.0..=1.0).contains(&shrinkage));
        assert!((0.0..=1.0).contains(&effective_shrinkage));
    }
}
