//! Posterior distribution parameters and MCMC sampling structures

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Posterior distribution parameters
#[derive(Debug, Clone)]
pub struct PosteriorParameters {
    /// Posterior means
    pub mu: Array2<Float>,
    /// Posterior covariance matrices
    pub sigma: Vec<Array2<Float>>,
    /// Posterior degrees of freedom
    pub nu: Array1<Float>,
    /// Posterior scale matrices
    pub psi: Vec<Array2<Float>>,
    /// Posterior precision parameters
    pub kappa: Array1<Float>,
    /// Hierarchical posterior parameters (optional)
    pub hierarchical: Option<HierarchicalPosterior>,
    /// MCMC samples (optional)
    pub mcmc_samples: Option<MCMCSamples>,
}

/// MCMC samples for Bayesian inference
#[derive(Debug, Clone)]
pub struct MCMCSamples {
    /// Sampled means for each class
    pub mu_samples: Vec<Array2<Float>>, // [n_samples, n_classes, n_features]
    /// Sampled covariance matrices for each class
    pub sigma_samples: Vec<Vec<Array2<Float>>>, // [n_samples][n_classes]
    /// Sampled precision matrices for each class
    pub precision_samples: Vec<Vec<Array2<Float>>>, // [n_samples][n_classes]
    /// Log posterior values for each sample
    pub log_posterior: Array1<Float>,
    /// Acceptance rates for Metropolis-Hastings steps
    pub acceptance_rates: HashMap<String, Float>,
    /// Effective sample sizes
    pub effective_sample_sizes: HashMap<String, Float>,
    /// Number of samples
    pub n_samples: usize,
    /// Burn-in period used
    pub burn_in: usize,
    /// Thinning interval used
    pub thin: usize,
}

/// Hierarchical posterior parameters for multi-level models
#[derive(Debug, Clone)]
pub struct HierarchicalPosterior {
    /// Global-level posterior parameters
    pub global_mu: Array1<Float>,
    /// global_kappa
    pub global_kappa: Float,
    /// global_nu
    pub global_nu: Float,
    /// global_psi
    pub global_psi: Array2<Float>,
    /// Group-level posterior parameters
    pub group_mu: Array2<Float>,
    /// group_sigma
    pub group_sigma: Vec<Array2<Float>>,
    /// group_precision
    pub group_precision: Array1<Float>,
    /// Hierarchical variance components
    pub between_group_variance: Array2<Float>,
    /// within_group_variance
    pub within_group_variance: Vec<Array2<Float>>,
    /// Level indicators
    pub group_assignments: Array1<usize>,
    /// n_groups
    pub n_groups: usize,
}
