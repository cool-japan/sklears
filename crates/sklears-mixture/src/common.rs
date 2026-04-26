//! Common utilities for mixture models
//!
//! This module provides shared functionality used across different mixture model implementations.
//! All implementations follow SciRS2 Policy for numerical computing and random number generation.

// IMPORTANT: SciRS2 Policy compliance - use scirs2-core instead of direct dependencies
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, RandNormal};
use scirs2_linalg::cholesky;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::f64::consts::PI;

/// Perform forward substitution to solve `L z = b` where `L` is lower triangular.
///
/// Returns `z` such that `L z = b`.
fn forward_substitute(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let d = b.len();
    let mut z = Array1::<f64>::zeros(d);
    for i in 0..d {
        let s: f64 = (0..i).map(|k| l[[i, k]] * z[k]).sum();
        z[i] = (b[i] - s) / l[[i, i]];
    }
    z
}

/// Sample from multivariate normal distribution using full Cholesky decomposition.
///
/// Computes `x = μ + L z` where `Σ = L Lᵀ` and `z ~ N(0, I)`.
/// Uses SciRS2 random number generation following the SciRS2 Policy.
pub fn sample_multivariate_normal(
    mean: &ArrayView1<f64>,
    cov: &ArrayView2<f64>,
) -> SklResult<Array1<f64>> {
    let n_features = mean.len();
    let mut rng = thread_rng();

    // Generate i.i.d. standard normal samples z ~ N(0, I)
    let mut z = Array1::<f64>::zeros(n_features);
    for i in 0..n_features {
        let normal = RandNormal::new(0.0, 1.0)
            .map_err(|e| SklearsError::InvalidInput(format!("Normal distribution error: {}", e)))?;
        z[i] = rng.sample(normal);
    }

    // Compute Cholesky factor L such that Σ = L Lᵀ
    let l = cholesky(cov, None)
        .map_err(|e| SklearsError::InvalidInput(format!("Cholesky decomposition failed: {}", e)))?;

    // x = μ + L z  (matrix-vector product using full lower triangular L)
    let mut sample = mean.to_owned();
    for i in 0..n_features {
        let li_dot_z: f64 = (0..=i).map(|k| l[[i, k]] * z[k]).sum();
        sample[i] += li_dot_z;
    }

    Ok(sample)
}

/// Compute log probability density function for multivariate Gaussian with full covariance.
///
/// Uses the Cholesky decomposition `Σ = L Lᵀ` to compute:
/// ```text
/// log p(x) = -0.5 * (d·ln(2π) + ln|det(Σ)| + (x-μ)ᵀ Σ⁻¹ (x-μ))
/// ```
/// where `ln|det(Σ)| = 2 Σᵢ ln(L[i,i])` and the quadratic form is
/// `‖z‖²` with `z` the solution of `L z = (x-μ)`.
pub fn gaussian_log_pdf(
    x: &ArrayView1<f64>,
    mean: &ArrayView1<f64>,
    cov: &ArrayView2<f64>,
) -> SklResult<f64> {
    let n_features = x.len();
    let diff = x - mean;

    // Cholesky decomposition: Σ = L Lᵀ
    let l = cholesky(cov, None)
        .map_err(|e| SklearsError::InvalidInput(format!("Cholesky decomposition failed: {}", e)))?;

    // ln|det(Σ)| = 2 * Σᵢ ln(L[i,i])
    let log_det: f64 = 2.0 * (0..n_features).map(|i| l[[i, i]].ln()).sum::<f64>();

    // Solve L z = diff by forward substitution; ‖z‖² = diffᵀ Σ⁻¹ diff
    let z = forward_substitute(&l, &diff);
    let quad_form = z.dot(&z);

    let log_2pi = (2.0 * PI).ln();
    Ok(-0.5 * (n_features as f64 * log_2pi + log_det + quad_form))
}

/// Compute log probability density function for multivariate Gaussian with diagonal covariance
pub fn gaussian_log_pdf_diagonal(
    x: &ArrayView1<f64>,
    mean: &ArrayView1<f64>,
    diag_cov: &ArrayView1<f64>,
) -> SklResult<f64> {
    let n_features = x.len();
    let diff = x - mean;

    let log_det = diag_cov.mapv(|v| v.ln()).sum();
    let inv_quad_form = (&diff * &diff / diag_cov).sum();

    let log_norm = -0.5 * (n_features as f64 * (2.0 * PI).ln() + log_det);
    let log_exp = -0.5 * inv_quad_form;

    Ok(log_norm + log_exp)
}

/// Compute log probability density function for multivariate Gaussian with spherical covariance
pub fn gaussian_log_pdf_spherical(
    x: &ArrayView1<f64>,
    mean: &ArrayView1<f64>,
    variance: f64,
) -> SklResult<f64> {
    let n_features = x.len();
    let diff = x - mean;

    let squared_dist = diff.dot(&diff);

    let log_norm = -0.5 * (n_features as f64 * (2.0 * PI * variance).ln());
    let log_exp = -0.5 * squared_dist / variance;

    Ok(log_norm + log_exp)
}

/// Covariance type for Gaussian mixture models
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceType {
    /// Full covariance matrix for each component
    Full,
    /// Diagonal covariance matrix for each component
    Diagonal,
    /// Single full covariance matrix tied to all components
    Tied,
    /// Single scalar variance for each component (spherical)
    Spherical,
}

/// Covariance matrices for different types
#[derive(Debug, Clone)]
pub enum CovarianceMatrices {
    /// Full covariance matrices
    Full(Vec<Array2<f64>>),
    /// Diagonal covariances
    Diagonal(Array2<f64>),
    /// Tied covariance
    Tied(Array2<f64>),
    /// Spherical variances
    Spherical(Array1<f64>),
}

impl CovarianceType {
    /// Parse covariance type from string
    pub fn parse_type(s: &str) -> Result<Self, SklearsError> {
        match s.to_lowercase().as_str() {
            "full" => Ok(CovarianceType::Full),
            "diag" | "diagonal" => Ok(CovarianceType::Diagonal),
            "tied" => Ok(CovarianceType::Tied),
            "spherical" => Ok(CovarianceType::Spherical),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown covariance type: {}. Must be one of: 'full', 'diagonal', 'tied', 'spherical'",
                s
            ))),
        }
    }
}

/// Initialization method for Gaussian mixture models
#[derive(Debug, Clone, PartialEq)]
pub enum InitMethod {
    /// K-means++ initialization
    KMeansPlus,
    /// Random initialization
    Random,
    /// User-provided initialization
    Params,
}

/// Model selection criteria results
#[derive(Debug, Clone)]
pub struct ModelSelection {
    /// aic
    pub aic: f64,
    /// bic
    pub bic: f64,
    /// log_likelihood
    pub log_likelihood: f64,
    /// n_parameters
    pub n_parameters: usize,
}

impl ModelSelection {
    /// Calculate Bayesian Information Criterion (BIC)
    pub fn bic(log_likelihood: f64, n_params: usize, n_samples: usize) -> f64 {
        -2.0 * log_likelihood + (n_params as f64) * (n_samples as f64).ln()
    }

    /// Calculate Akaike Information Criterion (AIC)
    pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
        -2.0 * log_likelihood + 2.0 * (n_params as f64)
    }

    /// Calculate number of parameters for given configuration
    pub fn n_parameters(
        n_components: usize,
        n_features: usize,
        covariance_type: &CovarianceType,
    ) -> usize {
        let weight_params = n_components - 1; // n_components - 1 for weights (sum to 1)
        let mean_params = n_components * n_features;
        let covariance_params = match covariance_type {
            CovarianceType::Full => n_components * n_features * (n_features + 1) / 2,
            CovarianceType::Diagonal => n_components * n_features,
            CovarianceType::Tied => n_features * (n_features + 1) / 2,
            CovarianceType::Spherical => n_components,
        };
        weight_params + mean_params + covariance_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// 2-D standard normal: μ=[0,0], Σ=I.  Expected: -0.5·2·ln(2π) ≈ -1.8379
    #[test]
    fn test_gaussian_log_pdf_standard_normal() {
        let x = array![0.0_f64, 0.0];
        let mean = array![0.0_f64, 0.0];
        let cov = Array2::<f64>::eye(2);
        let log_p = gaussian_log_pdf(&x.view(), &mean.view(), &cov.view())
            .expect("gaussian_log_pdf should succeed for identity covariance");
        let expected = -0.5 * 2.0 * std::f64::consts::TAU.ln();
        assert!(
            (log_p - expected).abs() < 1e-10,
            "standard-normal log pdf: got {log_p}, expected {expected}"
        );
    }

    /// Non-trivial test: Σ=[[2,0.5],[0.5,1]], x=[1,1], μ=[0,0].
    /// Expected ≈ -2.6891135318 (verified against scipy.stats.multivariate_normal).
    #[test]
    fn test_gaussian_log_pdf_full_covariance() {
        let x = array![1.0_f64, 1.0];
        let mean = array![0.0_f64, 0.0];
        let cov = array![[2.0_f64, 0.5], [0.5, 1.0]];
        let log_p = gaussian_log_pdf(&x.view(), &mean.view(), &cov.view())
            .expect("gaussian_log_pdf should succeed for positive-definite covariance");
        let expected = -2.689_113_531_805_628_f64;
        assert!(
            (log_p - expected).abs() < 1e-10,
            "full-covariance log pdf: got {log_p}, expected {expected}"
        );
    }

    /// Verify that sample_multivariate_normal returns a vector of the correct dimension.
    #[test]
    fn test_sample_multivariate_normal_dimension() {
        let mean = array![1.0_f64, 2.0, 3.0];
        let cov = Array2::<f64>::eye(3);
        let sample = sample_multivariate_normal(&mean.view(), &cov.view())
            .expect("sample_multivariate_normal should succeed for identity covariance");
        assert_eq!(
            sample.len(),
            3,
            "sample dimension should match mean dimension"
        );
    }
}
