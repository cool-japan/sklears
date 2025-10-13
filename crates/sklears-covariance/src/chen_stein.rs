//! Chen-Stein Shrinkage Covariance Estimator
//!
//! Alternative shrinkage estimator based on Chen-Stein theory.

use crate::empirical::EmpiricalCovariance;
use crate::utils::matrix_inverse;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for ChenSteinCovariance estimator
#[derive(Debug, Clone)]
pub struct ChenSteinCovarianceConfig {
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Size of blocks for processing
    pub block_size: usize,
}

/// Chen-Stein Shrinkage Covariance Estimator
///
/// Implements the Chen-Stein shrinkage approach for covariance estimation.
/// This method provides an alternative to Ledoit-Wolf shrinkage with
/// different theoretical foundations based on Stein's unbiased risk estimation.
///
/// The estimator computes:
/// C_shrunk = (1 - ρ) * C_empirical + ρ * target
///
/// where ρ is the optimal shrinkage parameter estimated using Chen-Stein theory,
/// and the target is typically the scaled identity matrix.
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
/// use sklears_covariance::ChenSteinCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let cs = ChenSteinCovariance::new();
/// let fitted = cs.fit(&X.view(), &()).unwrap();
/// let covariance = fitted.get_covariance();
/// let shrinkage = fitted.get_shrinkage();
/// ```
#[derive(Debug, Clone)]
pub struct ChenSteinCovariance<S = Untrained> {
    state: S,
    config: ChenSteinCovarianceConfig,
}

/// Trained state for ChenSteinCovariance
#[derive(Debug, Clone)]
pub struct ChenSteinCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The shrinkage parameter used
    pub shrinkage: f64,
}

impl ChenSteinCovariance<Untrained> {
    /// Create a new ChenSteinCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: ChenSteinCovarianceConfig {
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

impl Default for ChenSteinCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ChenSteinCovariance<Untrained> {
    type Config = ChenSteinCovarianceConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ChenSteinCovariance<Untrained> {
    type Fitted = ChenSteinCovariance<ChenSteinCovarianceTrained>;

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

        // Compute Chen-Stein shrinkage parameter
        let shrinkage = self.compute_chen_stein_shrinkage(&x, &covariance_emp, &location)?;

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

        Ok(ChenSteinCovariance {
            state: ChenSteinCovarianceTrained {
                covariance,
                precision,
                location,
                shrinkage,
            },
            config: self.config,
        })
    }
}

impl ChenSteinCovariance<Untrained> {
    /// Compute Chen-Stein shrinkage parameter
    fn compute_chen_stein_shrinkage(
        &self,
        x: &ArrayView2<'_, Float>,
        covariance_emp: &Array2<f64>,
        location: &Array1<f64>,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = x.dim();
        let n = n_samples as f64;
        let p = n_features as f64;

        // Center the data
        let mut x_centered = x.to_owned();
        if !self.config.assume_centered {
            for mut row in x_centered.axis_iter_mut(Axis(0)) {
                row -= location;
            }
        }

        // Compute target shrinkage matrix (scaled identity)
        let trace = covariance_emp.diag().sum();
        let mu = trace / p;

        // Compute the loss functions for Chen-Stein estimation
        let mut sum_squared_deviations = 0.0;
        let mut sum_target_deviations = 0.0;

        for i in 0..n_features {
            for j in 0..n_features {
                let target_val = if i == j { mu } else { 0.0 };

                // Compute sample-based estimate of variance
                let mut sample_var = 0.0;
                for k in 0..n_samples {
                    let sample_cov_ij = x_centered[[k, i]] * x_centered[[k, j]];
                    let deviation = sample_cov_ij - covariance_emp[[i, j]];
                    sample_var += deviation * deviation;
                }
                sample_var /= n;

                sum_squared_deviations += sample_var;
                sum_target_deviations += (covariance_emp[[i, j]] - target_val).powi(2);
            }
        }

        // Chen-Stein shrinkage formula
        // This is a simplified version - the full Chen-Stein theory involves more complex calculations
        let numerator = sum_squared_deviations / n;
        let denominator = sum_target_deviations;

        let shrinkage = if denominator > 1e-10 {
            (numerator / denominator).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Apply additional correction for finite sample bias
        let bias_correction = (n - 1.0) / n;
        let corrected_shrinkage = shrinkage * bias_correction;

        Ok(corrected_shrinkage.clamp(0.0, 1.0))
    }
}

impl ChenSteinCovariance<ChenSteinCovarianceTrained> {
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

    /// Compare with Ledoit-Wolf shrinkage parameter
    /// This method can be used to understand the differences between the two approaches
    pub fn compare_with_ledoit_wolf(&self, x: &ArrayView2<'_, Float>) -> SklResult<(f64, f64)> {
        // This would require implementing a simple Ledoit-Wolf calculation for comparison
        // For now, we return the Chen-Stein shrinkage and a placeholder
        let chen_stein_shrinkage = self.get_shrinkage();
        let placeholder_lw_shrinkage = 0.0; // Would need to compute actual LW shrinkage

        Ok((chen_stein_shrinkage, placeholder_lw_shrinkage))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_chen_stein_covariance_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let covariance = fitted.get_covariance();
        let precision = fitted.get_precision().unwrap();
        let shrinkage = fitted.get_shrinkage();

        // Check dimensions
        assert_eq!(covariance.dim(), (3, 3));
        assert_eq!(precision.dim(), (3, 3));

        // Check shrinkage parameter is valid
        assert!(shrinkage >= 0.0);
        assert!(shrinkage <= 1.0);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((covariance[[i, j]] - covariance[[j, i]]).abs() < 1e-10);
                assert!((precision[[i, j]] - precision[[j, i]]).abs() < 1e-10);
            }
        }

        // Check positive definiteness (diagonal elements should be positive)
        for i in 0..3 {
            assert!(covariance[[i, i]] > 0.0);
            assert!(precision[[i, i]] > 0.0);
        }
    }

    #[test]
    fn test_chen_stein_covariance_small_data() {
        // Test with small dataset where shrinkage should be high
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let covariance = fitted.get_covariance();
        let shrinkage = fitted.get_shrinkage();

        assert_eq!(covariance.dim(), (2, 2));
        // With small sample size, shrinkage should typically be positive
        assert!(shrinkage >= 0.0);
        assert!(shrinkage <= 1.0);
    }

    #[test]
    fn test_chen_stein_covariance_parameters() {
        let estimator = ChenSteinCovariance::new()
            .store_precision(false)
            .assume_centered(true)
            .block_size(500);

        assert!(!estimator.config.store_precision);
        assert!(estimator.config.assume_centered);
        assert_eq!(estimator.config.block_size, 500);
    }

    #[test]
    fn test_chen_stein_covariance_no_precision() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let estimator = ChenSteinCovariance::new().store_precision(false);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert!(fitted.get_precision().is_none());
        assert_eq!(fitted.get_covariance().dim(), (2, 2));
    }

    #[test]
    fn test_chen_stein_covariance_centered_data() {
        let x = array![[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]];

        let estimator = ChenSteinCovariance::new().assume_centered(true);
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let location = fitted.get_location();
        let covariance = fitted.get_covariance();

        // Location should be close to zero when data is assumed centered
        for &val in location.iter() {
            assert!(val.abs() < 1e-10);
        }

        assert_eq!(covariance.dim(), (2, 2));
    }

    #[test]
    fn test_chen_stein_covariance_mahalanobis() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let x_test = array![[2.0, 3.0], [6.0, 7.0]];

        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x_train.view(), &()).unwrap();

        let distances = fitted.mahalanobis_distance(&x_test.view()).unwrap();

        assert_eq!(distances.len(), 2);
        for &dist in distances.iter() {
            assert!(dist >= 0.0);
        }
    }

    #[test]
    fn test_chen_stein_identity_matrix() {
        // Test with identity-like data
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];

        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let covariance = fitted.get_covariance();
        let shrinkage = fitted.get_shrinkage();

        assert_eq!(covariance.dim(), (3, 3));
        assert!(shrinkage >= 0.0);
        assert!(shrinkage <= 1.0);

        // For identity-like data, shrinkage might be low
        // but we just check it's in valid range
    }

    #[test]
    fn test_chen_stein_covariance_highly_correlated() {
        // Test with highly correlated data
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]];

        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let covariance = fitted.get_covariance();
        let shrinkage = fitted.get_shrinkage();

        assert_eq!(covariance.dim(), (2, 2));
        assert!(shrinkage >= 0.0);
        assert!(shrinkage <= 1.0);

        // High correlation should show up in off-diagonal elements
        assert!(covariance[[0, 1]].abs() > 0.0);
        assert!(covariance[[1, 0]].abs() > 0.0);
    }

    #[test]
    fn test_chen_stein_compare_with_ledoit_wolf() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let (cs_shrinkage, _lw_shrinkage) = fitted.compare_with_ledoit_wolf(&x.view()).unwrap();

        // Just check that Chen-Stein shrinkage is valid
        assert!(cs_shrinkage >= 0.0);
        assert!(cs_shrinkage <= 1.0);
        assert_eq!(cs_shrinkage, fitted.get_shrinkage());
    }

    #[test]
    fn test_chen_stein_covariance_edge_cases() {
        // Test with insufficient samples
        let x = array![[1.0, 2.0]];
        let estimator = ChenSteinCovariance::new();
        let result = estimator.fit(&x.view(), &());
        assert!(result.is_err());

        // Test with valid minimal data
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let estimator = ChenSteinCovariance::new();
        let fitted = estimator.fit(&x.view(), &()).unwrap();
        assert_eq!(fitted.get_covariance().dim(), (2, 2));
    }

    #[test]
    fn test_chen_stein_shrinkage_bounds() {
        // Test various scenarios to ensure shrinkage is always bounded
        let test_cases = vec![
            array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        ];

        for x in test_cases {
            let estimator = ChenSteinCovariance::new();
            let fitted = estimator.fit(&x.view(), &()).unwrap();
            let shrinkage = fitted.get_shrinkage();

            assert!(
                shrinkage >= 0.0,
                "Shrinkage should be non-negative: {}",
                shrinkage
            );
            assert!(
                shrinkage <= 1.0,
                "Shrinkage should be at most 1: {}",
                shrinkage
            );
        }
    }
}
