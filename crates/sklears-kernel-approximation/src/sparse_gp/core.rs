//! Core types, enums, and structures for sparse Gaussian Process implementation
//!
//! This module provides the foundational data structures and type definitions
//! for sparse Gaussian Process approximations with SIMD acceleration.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::fmt;

/// Available sparse approximation methods for Gaussian Processes
#[derive(Debug, Clone)]
pub enum SparseApproximation {
    /// Subset of Regressors (SoR) - Uses subset of training points as inducing points
    SubsetOfRegressors,

    /// Fully Independent Conditional (FIC) - Assumes independence given inducing points
    FullyIndependentConditional,

    /// Partially Independent Conditional (PIC) - Block-diagonal conditional independence
    PartiallyIndependentConditional {
        /// Block size for PIC approximation
        block_size: usize,
    },

    /// Variational Free Energy (VFE) - Variational sparse approximation
    VariationalFreeEnergy {
        /// Use whitened representation
        whitened: bool,
        /// Use natural gradients for optimization
        natural_gradients: bool,
    },
}

/// Strategies for selecting inducing points in sparse GP approximations
#[derive(Debug, Clone)]
pub enum InducingPointStrategy {
    /// Random selection from training data
    Random,

    /// K-means clustering to find representative points
    KMeans,

    /// Uniform grid over input space
    UniformGrid {
        /// Grid size for each dimension
        grid_size: Vec<usize>,
    },

    /// Greedy selection based on maximum posterior variance
    GreedyVariance,

    /// User-specified inducing points
    UserSpecified(Array2<f64>),
}

/// Scalable inference methods for large-scale sparse GP prediction
#[derive(Debug, Clone)]
pub enum ScalableInferenceMethod {
    /// Direct matrix inversion (for small problems)
    Direct,

    /// Preconditioned Conjugate Gradient solver
    PreconditionedCG {
        /// Maximum number of iterations
        max_iter: usize,
        /// Convergence tolerance
        tol: f64,
        /// Preconditioner type
        preconditioner: PreconditionerType,
    },

    /// Lanczos eigendecomposition method
    Lanczos {
        /// Number of Lanczos vectors to compute
        num_vectors: usize,
        /// Tolerance for convergence
        tol: f64,
    },
}

/// Preconditioner types for iterative solvers
#[derive(Debug, Clone)]
pub enum PreconditionerType {
    /// No preconditioning
    None,

    /// Diagonal preconditioning M = diag(A)^(-1)
    Diagonal,

    /// Incomplete Cholesky factorization
    IncompleteCholesky {
        /// Fill factor for sparsity control
        fill_factor: f64,
    },

    /// Symmetric Successive Over-Relaxation (SSOR)
    SSOR {
        /// Relaxation parameter
        omega: f64,
    },
}

/// Interpolation methods for structured kernel approximations
#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
}

/// Core sparse Gaussian Process structure with configuration parameters
#[derive(Debug, Clone)]
pub struct SparseGaussianProcess<K> {
    /// Number of inducing points
    pub num_inducing: usize,

    /// Kernel function
    pub kernel: K,

    /// Sparse approximation method
    pub approximation: SparseApproximation,

    /// Strategy for selecting inducing points
    pub inducing_strategy: InducingPointStrategy,

    /// Observation noise variance
    pub noise_variance: f64,

    /// Maximum optimization iterations
    pub max_iter: usize,

    /// Convergence tolerance
    pub tol: f64,
}

/// Fitted sparse Gaussian Process with learned parameters
#[derive(Debug, Clone)]
pub struct FittedSparseGP<K> {
    /// Inducing point locations
    pub inducing_points: Array2<f64>,

    /// Kernel function with learned parameters
    pub kernel: K,

    /// Sparse approximation method used
    pub approximation: SparseApproximation,

    /// Precomputed alpha coefficients
    pub alpha: Array1<f64>,

    /// Inverse of K_mm (inducing point kernel matrix)
    pub k_mm_inv: Array2<f64>,

    /// Noise variance
    pub noise_variance: f64,

    /// Variational parameters (if using VFE)
    pub variational_params: Option<VariationalParams>,
}

/// Variational parameters for Variational Free Energy approximation
#[derive(Debug, Clone)]
pub struct VariationalParams {
    /// Variational mean parameter
    pub mean: Array1<f64>,

    /// Cholesky factor of variational covariance
    pub cov_factor: Array2<f64>,

    /// Evidence Lower BOund (ELBO) value
    pub elbo: f64,

    /// KL divergence term
    pub kl_divergence: f64,

    /// Log likelihood term
    pub log_likelihood: f64,
}

/// Structured Kernel Interpolation (KISS-GP) for fast structured GP inference
#[derive(Debug, Clone)]
pub struct StructuredKernelInterpolation<K> {
    /// Grid size for each dimension
    pub grid_size: Vec<usize>,

    /// Kernel function
    pub kernel: K,

    /// Noise variance
    pub noise_variance: f64,

    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Fitted structured kernel interpolation
#[derive(Debug, Clone)]
pub struct FittedSKI<K> {
    /// Grid points
    pub grid_points: Array2<f64>,

    /// Interpolation weights
    pub weights: Array2<f64>,

    /// Kernel function
    pub kernel: K,

    /// Precomputed alpha
    pub alpha: Array1<f64>,
}

/// Configuration for sparse GP optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of iterations
    pub max_iter: usize,

    /// Convergence tolerance
    pub tolerance: f64,

    /// Learning rate for gradient-based methods
    pub learning_rate: f64,

    /// Whether to use natural gradients
    pub natural_gradients: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            learning_rate: 0.01,
            natural_gradients: false,
        }
    }
}

/// Error types specific to sparse GP operations
#[derive(Debug)]
pub enum SparseGPError {
    /// Invalid inducing point configuration
    InvalidInducingPoints(String),

    /// Numerical instability in computation
    NumericalInstability(String),

    /// Convergence failure
    ConvergenceFailure(String),

    /// Invalid approximation parameters
    InvalidApproximation(String),
}

impl fmt::Display for SparseGPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseGPError::InvalidInducingPoints(msg) => {
                write!(f, "Invalid inducing points: {}", msg)
            }
            SparseGPError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            SparseGPError::ConvergenceFailure(msg) => {
                write!(f, "Convergence failure: {}", msg)
            }
            SparseGPError::InvalidApproximation(msg) => {
                write!(f, "Invalid approximation: {}", msg)
            }
        }
    }
}

impl std::error::Error for SparseGPError {}

/// Convert SparseGPError to SklearsError
impl From<SparseGPError> for SklearsError {
    fn from(err: SparseGPError) -> Self {
        match err {
            SparseGPError::InvalidInducingPoints(msg) => SklearsError::InvalidInput(msg),
            SparseGPError::NumericalInstability(msg) => SklearsError::NumericalError(msg),
            SparseGPError::ConvergenceFailure(msg) => SklearsError::NumericalError(msg),
            SparseGPError::InvalidApproximation(msg) => SklearsError::InvalidInput(msg),
        }
    }
}

/// Builder-style methods for sparse GP configuration
impl<K> SparseGaussianProcess<K> {
    /// Set the sparse approximation method
    pub fn approximation(mut self, approximation: SparseApproximation) -> Self {
        self.approximation = approximation;
        self
    }

    /// Set the inducing point selection strategy
    pub fn inducing_strategy(mut self, strategy: InducingPointStrategy) -> Self {
        self.inducing_strategy = strategy;
        self
    }

    /// Set the observation noise variance
    pub fn noise_variance(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Set optimization parameters
    pub fn optimization_params(mut self, max_iter: usize, tol: f64) -> Self {
        self.max_iter = max_iter;
        self.tol = tol;
        self
    }
}

/// Builder-style methods for SKI configuration
impl<K> StructuredKernelInterpolation<K> {
    /// Set noise variance
    pub fn noise_variance(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Set interpolation method
    pub fn interpolation(mut self, interpolation: InterpolationMethod) -> Self {
        self.interpolation = interpolation;
        self
    }
}

/// Helper functions for sparse GP operations
pub mod utils {
    use super::*;

    /// Validate inducing point configuration
    pub fn validate_inducing_points(
        num_inducing: usize,
        n_features: usize,
        strategy: &InducingPointStrategy,
    ) -> Result<()> {
        match strategy {
            InducingPointStrategy::UniformGrid { grid_size } => {
                if grid_size.len() != n_features {
                    return Err(SklearsError::InvalidInput(
                        "Grid size must match number of features".to_string(),
                    ));
                }

                let total_points: usize = grid_size.iter().product();
                if total_points != num_inducing {
                    return Err(SklearsError::InvalidInput(format!(
                        "Grid size product {} must equal num_inducing {}",
                        total_points, num_inducing
                    )));
                }
            }
            InducingPointStrategy::UserSpecified(points) => {
                if points.nrows() != num_inducing {
                    return Err(SklearsError::InvalidInput(
                        "User-specified points must match num_inducing".to_string(),
                    ));
                }
                if points.ncols() != n_features {
                    return Err(SklearsError::InvalidInput(
                        "User-specified points must match number of features".to_string(),
                    ));
                }
            }
            _ => {} // Other strategies are validated during execution
        }

        Ok(())
    }

    /// Check for numerical stability in matrices
    pub fn check_matrix_stability(matrix: &Array2<f64>, name: &str) -> Result<()> {
        let has_nan = matrix.iter().any(|&x| x.is_nan());
        let has_inf = matrix.iter().any(|&x| x.is_infinite());

        if has_nan || has_inf {
            return Err(SklearsError::NumericalError(format!(
                "Matrix {} contains NaN or infinite values",
                name
            )));
        }

        Ok(())
    }

    /// Compute matrix condition number estimate
    pub fn estimate_condition_number(matrix: &Array2<f64>) -> f64 {
        // Simple condition number estimate using diagonal dominance
        let diag_sum: f64 = matrix.diag().iter().map(|x| x.abs()).sum();
        let off_diag_sum: f64 = matrix.iter().map(|x| x.abs()).sum::<f64>() - diag_sum;

        if diag_sum > 0.0 {
            (diag_sum + off_diag_sum) / diag_sum
        } else {
            f64::INFINITY
        }
    }
}
