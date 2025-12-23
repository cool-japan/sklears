//! Common utilities for mixture models
//!
//! This module provides shared functionality used across different mixture model implementations.
//! All implementations follow SciRS2 Policy for numerical computing and random number generation.

// IMPORTANT: SciRS2 Policy compliance - use scirs2-core instead of direct dependencies
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, RandNormal};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::f64::consts::PI;

/// Sample from multivariate normal distribution
/// Uses SciRS2 random number generation following the SciRS2 Policy
pub fn sample_multivariate_normal(
    mean: &ArrayView1<f64>,
    cov: &ArrayView2<f64>,
) -> SklResult<Array1<f64>> {
    let n_features = mean.len();
    let mut sample = Array1::zeros(n_features);
    let mut rng = thread_rng();

    // Generate independent standard normal samples
    for i in 0..n_features {
        let normal = RandNormal::new(0.0, 1.0)
            .map_err(|e| SklearsError::InvalidInput(format!("Normal distribution error: {}", e)))?;
        sample[i] = rng.sample(normal);
    }

    // For simplicity, using diagonal covariance approximation
    // In a full implementation, this would use Cholesky decomposition
    for i in 0..n_features {
        let variance = cov[[i, i]].abs();
        sample[i] = mean[i] + sample[i] * variance.sqrt();
    }

    Ok(sample)
}

/// Compute log probability density function for multivariate Gaussian with full covariance
pub fn gaussian_log_pdf(
    x: &ArrayView1<f64>,
    mean: &ArrayView1<f64>,
    _cov: &ArrayView2<f64>,
) -> SklResult<f64> {
    let n_features = x.len();
    let diff = x - mean;

    // For now, use simple implementation - should be replaced with proper matrix operations
    let det: f64 = 1.0; // Placeholder - should compute determinant of cov
    let inv_quad_form = diff.dot(&diff); // Placeholder - should compute diff^T * inv(cov) * diff

    let log_norm = -0.5 * (n_features as f64 * (2.0 * PI).ln() + det.ln());
    let log_exp = -0.5 * inv_quad_form;

    Ok(log_norm + log_exp)
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
