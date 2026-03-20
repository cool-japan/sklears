//! Types and configuration for Bayesian Discriminant Analysis

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;

/// Prior type for Bayesian inference
#[derive(Debug, Clone)]
pub enum PriorType {
    /// Jeffreys (non-informative) prior
    Jeffreys,
    /// Normal-inverse-Wishart conjugate prior
    NormalInverseWishart {
        mu_0: Array1<Float>,

        kappa_0: Float,

        nu_0: Float,

        psi_0: Array2<Float>,
    },
    /// Empirical Bayes (data-driven prior)
    EmpiricalBayes,
    /// Hierarchical prior with multiple levels
    Hierarchical {
        /// Level 1 (global) hyperpriors
        global_mu_prior: Array1<Float>,
        global_kappa_prior: Float,
        global_nu_prior: Float,
        global_psi_prior: Array2<Float>,
        /// Level 2 (group) hyperpriors
        group_precision_shape: Float,
        group_precision_rate: Float,
        /// Number of hierarchical levels
        n_levels: usize,
    },
}

/// Inference method for Bayesian analysis
#[derive(Debug, Clone)]
pub enum InferenceMethod {
    /// Variational Bayes
    VariationalBayes,
    /// Monte Carlo Markov Chain
    MCMC {
        n_samples: usize,

        burn_in: usize,

        thin: usize,
    },
    /// Laplace approximation
    LaplaceApproximation,
    /// Hierarchical Variational Bayes
    HierarchicalVariationalBayes {
        max_levels: usize,
        level_convergence_tol: Float,
    },
}

/// Configuration for Bayesian Discriminant Analysis
#[derive(Debug, Clone)]
pub struct BayesianDiscriminantAnalysisConfig {
    /// Prior type
    pub prior: PriorType,
    /// Inference method
    pub inference: InferenceMethod,
    /// Number of components for dimensionality reduction
    pub n_components: Option<usize>,
    /// Regularization parameter
    pub reg_param: Float,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for BayesianDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            prior: PriorType::Jeffreys,
            inference: InferenceMethod::VariationalBayes,
            n_components: None,
            reg_param: 1e-6,
            tol: 1e-6,
            max_iter: 1000,
            random_state: None,
        }
    }
}
