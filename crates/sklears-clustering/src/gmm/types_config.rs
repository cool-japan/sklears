//! Core types, enums, and configuration structures for GMM
//!
//! This module defines the fundamental types used throughout the GMM implementation,
//! including covariance types, initialization methods, model selection criteria,
//! and configuration structures for both classical and Bayesian GMM.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::types::Float;

// Import from scirs2 for type conversions
use scirs2_cluster::gmm::{CovarianceType as ScirsCovType, GMMInit};

/// Type of covariance parameters to use
#[derive(Debug, Clone, Copy)]
pub enum CovarianceType {
    Full,
    Diagonal,
    Tied,
    Spherical,
}

impl From<CovarianceType> for ScirsCovType {
    fn from(cov_type: CovarianceType) -> Self {
        match cov_type {
            CovarianceType::Full => ScirsCovType::Full,
            CovarianceType::Diagonal => ScirsCovType::Diagonal,
            CovarianceType::Tied => ScirsCovType::Tied,
            CovarianceType::Spherical => ScirsCovType::Spherical,
        }
    }
}

/// Weight initialization method
#[derive(Debug, Clone, Copy)]
pub enum WeightInit {
    /// Initialize using K-means clustering
    KMeans,
    /// Random initialization
    Random,
}

impl From<WeightInit> for GMMInit {
    fn from(init: WeightInit) -> Self {
        match init {
            WeightInit::KMeans => GMMInit::KMeans,
            WeightInit::Random => GMMInit::Random,
        }
    }
}

/// Model selection criteria for determining optimal number of components
#[derive(Debug, Clone, Copy)]
pub enum ModelSelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Integrated Completed Likelihood
    ICL,
}

/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// Best number of components found
    pub best_n_components: usize,
    /// Criterion values for each number of components tested
    pub criterion_values: Vec<Float>,
    /// Log-likelihood values for each number of components tested
    pub log_likelihoods: Vec<Float>,
    /// Model selection criterion used
    pub criterion: ModelSelectionCriterion,
}

/// Configuration for Gaussian Mixture Model
#[derive(Debug, Clone)]
pub struct GaussianMixtureConfig {
    /// Number of mixture components
    pub n_components: usize,
    /// Type of covariance parameters
    pub covariance_type: CovarianceType,
    /// Convergence tolerance
    pub tol: Float,
    /// Regularization added to diagonal of covariance
    pub reg_covar: Float,
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Number of initializations to perform
    pub n_init: usize,
    /// Method used to initialize weights
    pub init_params: WeightInit,
    /// Weight concentration prior (Dirichlet parameter)
    pub weight_concentration_prior_type: String,
    /// Weight concentration prior value
    pub weight_concentration_prior: Option<Float>,
    /// Mean precision prior
    pub mean_precision_prior: Option<Float>,
    /// Mean prior
    pub mean_prior: Option<Array1<Float>>,
    /// Degrees of freedom prior
    pub degrees_of_freedom_prior: Option<Float>,
    /// Covariance prior
    pub covariance_prior: Option<Array2<Float>>,
    /// Random seed
    pub random_state: Option<u64>,
    /// Warm start
    pub warm_start: bool,
    /// Verbosity level
    pub verbose: usize,
    /// Interval between verbose outputs
    pub verbose_interval: usize,
}

impl Default for GaussianMixtureConfig {
    fn default() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            init_params: WeightInit::KMeans,
            weight_concentration_prior_type: "dirichlet_process".to_string(),
            weight_concentration_prior: None,
            mean_precision_prior: None,
            mean_prior: None,
            degrees_of_freedom_prior: None,
            covariance_prior: None,
            random_state: None,
            warm_start: false,
            verbose: 0,
            verbose_interval: 10,
        }
    }
}

/// Configuration for Bayesian Gaussian Mixture Model
#[derive(Debug, Clone)]
pub struct BayesianGaussianMixtureConfig {
    /// Number of components (upper bound)
    pub n_components: usize,
    /// Covariance type
    pub covariance_type: CovarianceType,
    /// Convergence tolerance
    pub tol: Float,
    /// Regularization term
    pub reg_covar: Float,
    /// Maximum iterations
    pub max_iter: usize,
    /// Number of initializations
    pub n_init: usize,
    /// Initialization method
    pub init_params: WeightInit,
    /// Random seed
    pub random_state: Option<u64>,
    /// Weight concentration prior
    pub weight_concentration_prior: Float,
    /// Weight concentration prior type
    pub weight_concentration_prior_type: String,
    /// Mean precision prior
    pub mean_precision_prior: Float,
    /// Mean prior
    pub mean_prior: Option<Array1<Float>>,
    /// Degrees of freedom prior
    pub degrees_of_freedom_prior: Option<Float>,
    /// Covariance prior
    pub covariance_prior: Option<Array2<Float>>,
}

impl Default for BayesianGaussianMixtureConfig {
    fn default() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            init_params: WeightInit::KMeans,
            random_state: None,
            weight_concentration_prior: 1e-3,
            weight_concentration_prior_type: "dirichlet_process".to_string(),
            mean_precision_prior: 1e-2,
            mean_prior: None,
            degrees_of_freedom_prior: None,
            covariance_prior: None,
        }
    }
}

impl GaussianMixtureConfig {
    /// Create a new configuration with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Default::default()
        }
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, reg_covar: Float) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the initialization method
    pub fn initialization(mut self, init_params: WeightInit) -> Self {
        self.init_params = init_params;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, level: usize) -> Self {
        self.verbose = level;
        self
    }
}

impl BayesianGaussianMixtureConfig {
    /// Create a new Bayesian configuration with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Default::default()
        }
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the weight concentration prior
    pub fn weight_concentration_prior(mut self, prior: Float) -> Self {
        self.weight_concentration_prior = prior;
        self
    }

    /// Set the mean precision prior
    pub fn mean_precision_prior(mut self, prior: Float) -> Self {
        self.mean_precision_prior = prior;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}
