//! Core Types for Kernel Ridge Regression
//!
//! This module contains the fundamental types, enums, and utilities shared
//! across all kernel ridge regression implementations.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, traits::Transform};
pub use sklears_core::{traits::Trained, types::Float};

pub use crate::{
    fastfood::FastfoodTransform,
    nystroem::{Kernel, Nystroem, SamplingStrategy},
    rbf_sampler::RBFSampler,
    structured_random_features::StructuredRandomFeatures,
};

/// Kernel approximation method for ridge regression
#[derive(Debug, Clone)]
pub enum ApproximationMethod {
    /// Nyström method
    Nystroem {
        kernel: Kernel,

        n_components: usize,

        sampling_strategy: SamplingStrategy,
    },
    /// Random Fourier Features (RBF sampler)
    RandomFourierFeatures { n_components: usize, gamma: Float },
    /// Structured Random Features
    StructuredRandomFeatures { n_components: usize, gamma: Float },
    /// Fastfood Transform
    Fastfood { n_components: usize, gamma: Float },
}

impl ApproximationMethod {
    /// Get the number of components for this approximation method
    pub fn n_components(&self) -> usize {
        match self {
            ApproximationMethod::Nystroem { n_components, .. } => *n_components,
            ApproximationMethod::RandomFourierFeatures { n_components, .. } => *n_components,
            ApproximationMethod::StructuredRandomFeatures { n_components, .. } => *n_components,
            ApproximationMethod::Fastfood { n_components, .. } => *n_components,
        }
    }

    /// Get the gamma parameter if applicable
    pub fn gamma(&self) -> Option<Float> {
        match self {
            ApproximationMethod::Nystroem { .. } => None,
            ApproximationMethod::RandomFourierFeatures { gamma, .. } => Some(*gamma),
            ApproximationMethod::StructuredRandomFeatures { gamma, .. } => Some(*gamma),
            ApproximationMethod::Fastfood { gamma, .. } => Some(*gamma),
        }
    }

    /// Get the kernel if applicable (for Nyström method)
    pub fn kernel(&self) -> Option<&Kernel> {
        match self {
            ApproximationMethod::Nystroem { kernel, .. } => Some(kernel),
            _ => None,
        }
    }

    /// Get the sampling strategy if applicable (for Nyström method)
    pub fn sampling_strategy(&self) -> Option<&SamplingStrategy> {
        match self {
            ApproximationMethod::Nystroem {
                sampling_strategy, ..
            } => Some(sampling_strategy),
            _ => None,
        }
    }
}

/// Solver for the linear system
#[derive(Debug, Clone, Default)]
pub enum Solver {
    /// Direct solver using Cholesky decomposition
    #[default]
    Direct,
    /// SVD-based solver (more stable but slower)
    SVD,
    /// Conjugate Gradient (iterative, memory efficient)
    ConjugateGradient { max_iter: usize, tol: Float },
}

impl Solver {
    /// Get solver name as string
    pub fn name(&self) -> &'static str {
        match self {
            Solver::Direct => "direct",
            Solver::SVD => "svd",
            Solver::ConjugateGradient { .. } => "conjugate_gradient",
        }
    }

    /// Check if solver is iterative
    pub fn is_iterative(&self) -> bool {
        matches!(self, Solver::ConjugateGradient { .. })
    }

    /// Get maximum iterations for iterative solvers
    pub fn max_iter(&self) -> Option<usize> {
        match self {
            Solver::ConjugateGradient { max_iter, .. } => Some(*max_iter),
            _ => None,
        }
    }

    /// Get tolerance for iterative solvers
    pub fn tolerance(&self) -> Option<Float> {
        match self {
            Solver::ConjugateGradient { tol, .. } => Some(*tol),
            _ => None,
        }
    }
}

/// Wrapper for different feature transformers
#[derive(Debug, Clone)]
pub enum FeatureTransformer {
    /// Nystroem
    Nystroem(Nystroem<Trained>),
    /// RBFSampler
    RBFSampler(RBFSampler<Trained>),
    /// StructuredRFF
    StructuredRFF(StructuredRandomFeatures<Trained>),
    /// Fastfood
    Fastfood(FastfoodTransform<Trained>),
}

impl FeatureTransformer {
    /// Transform features using the appropriate transformer
    pub fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        match self {
            FeatureTransformer::Nystroem(transformer) => transformer.transform(x),
            FeatureTransformer::RBFSampler(transformer) => transformer.transform(x),
            FeatureTransformer::StructuredRFF(transformer) => transformer.transform(x),
            FeatureTransformer::Fastfood(transformer) => transformer.transform(x),
        }
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        match self {
            FeatureTransformer::Nystroem(transformer) => {
                // For Nyström, the number of output features equals n_components
                transformer.n_components
            }
            FeatureTransformer::RBFSampler(transformer) => {
                // For RBF sampler, output features = 2 * n_components
                // (one dimension for cos and one for sin)
                2 * transformer.n_components
            }
            FeatureTransformer::StructuredRFF(transformer) => {
                // For structured RFF, output features = n_components
                transformer.n_components
            }
            FeatureTransformer::Fastfood(transformer) => {
                // For Fastfood, output features = n_components
                transformer.n_components
            }
        }
    }

    /// Get transformer type name
    pub fn transformer_type(&self) -> &'static str {
        match self {
            FeatureTransformer::Nystroem(_) => "nystroem",
            FeatureTransformer::RBFSampler(_) => "rbf_sampler",
            FeatureTransformer::StructuredRFF(_) => "structured_rff",
            FeatureTransformer::Fastfood(_) => "fastfood",
        }
    }
}

/// Configuration for ridge regression solving
#[derive(Debug, Clone)]
pub struct RidgeConfig {
    /// Regularization parameter
    pub alpha: Float,
    /// Solver method
    pub solver: Solver,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to normalize features
    pub normalize: bool,
}

impl RidgeConfig {
    /// Create new ridge configuration
    pub fn new(alpha: Float) -> Self {
        Self {
            alpha,
            solver: Solver::default(),
            fit_intercept: true,
            normalize: false,
        }
    }

    /// Set solver method
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set fit intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for RidgeConfig {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Common validation functions for kernel ridge regression
pub mod validation {
    use super::*;
    use sklears_core::error::SklearsError;

    /// Validate input dimensions for fitting
    pub fn validate_fit_input(x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("X cannot be empty".to_string()));
        }

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "X must have at least one feature".to_string(),
            ));
        }

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                n_samples,
                y.len()
            )));
        }

        // Check for NaN or infinite values
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "X contains NaN or infinite values".to_string(),
            ));
        }

        if y.iter().any(|&val| !val.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "y contains NaN or infinite values".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate input dimensions for multi-task fitting
    pub fn validate_multitask_fit_input(x: &Array2<Float>, y: &Array2<Float>) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let (n_targets, n_tasks) = y.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("X cannot be empty".to_string()));
        }

        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "X must have at least one feature".to_string(),
            ));
        }

        if n_tasks == 0 {
            return Err(SklearsError::InvalidInput(
                "y must have at least one task".to_string(),
            ));
        }

        if n_targets != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                n_samples, n_targets
            )));
        }

        // Check for NaN or infinite values
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "X contains NaN or infinite values".to_string(),
            ));
        }

        if y.iter().any(|&val| !val.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "y contains NaN or infinite values".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate input dimensions for prediction
    pub fn validate_predict_input(x: &Array2<Float>, n_features_expected: usize) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("X cannot be empty".to_string()));
        }

        if n_features != n_features_expected {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but {} features were expected",
                n_features, n_features_expected
            )));
        }

        // Check for NaN or infinite values
        if x.iter().any(|&val| !val.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "X contains NaN or infinite values".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate regularization parameter
    pub fn validate_alpha(alpha: Float) -> Result<()> {
        if !alpha.is_finite() {
            return Err(SklearsError::InvalidInput(
                "alpha must be finite".to_string(),
            ));
        }

        if alpha <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate approximation method parameters
    pub fn validate_approximation_method(method: &ApproximationMethod) -> Result<()> {
        let n_components = method.n_components();

        if n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }

        if let Some(gamma) = method.gamma() {
            if !gamma.is_finite() || gamma <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "gamma must be positive and finite".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Common utility functions for kernel ridge regression
pub mod utils {
    use super::*;

    /// Add regularization to a matrix (in-place)
    pub fn add_regularization(matrix: &mut Array2<Float>, alpha: Float) {
        let n = matrix.nrows().min(matrix.ncols());
        for i in 0..n {
            matrix[[i, i]] += alpha;
        }
    }

    /// Compute mean of an array
    pub fn compute_mean(array: &Array1<Float>) -> Float {
        array.mean().unwrap_or(0.0)
    }

    /// Center data (subtract mean)
    pub fn center_data(data: &mut Array2<Float>) -> Array1<Float> {
        let means = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        for mut row in data.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
            row -= &means;
        }
        means
    }

    /// Center targets (subtract mean)
    pub fn center_targets(targets: &mut Array1<Float>) -> Float {
        let mean = compute_mean(targets);
        *targets -= mean;
        mean
    }

    /// Generate a random seed if none provided
    pub fn get_or_generate_seed(seed: Option<u64>) -> u64 {
        seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        })
    }
}

/// Builder pattern for approximation methods
pub mod builders {
    use super::*;

    /// Builder for Nyström approximation method
    pub struct NystroemBuilder {
        kernel: Kernel,
        n_components: usize,
        sampling_strategy: SamplingStrategy,
    }

    impl NystroemBuilder {
        pub fn new(kernel: Kernel) -> Self {
            Self {
                kernel,
                n_components: 100,
                sampling_strategy: SamplingStrategy::Random,
            }
        }

        pub fn n_components(mut self, n_components: usize) -> Self {
            self.n_components = n_components;
            self
        }

        pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
            self.sampling_strategy = strategy;
            self
        }

        pub fn build(self) -> ApproximationMethod {
            ApproximationMethod::Nystroem {
                kernel: self.kernel,
                n_components: self.n_components,
                sampling_strategy: self.sampling_strategy,
            }
        }
    }

    /// Builder for Random Fourier Features
    pub struct RFFBuilder {
        n_components: usize,
        gamma: Float,
    }

    impl RFFBuilder {
        pub fn new(gamma: Float) -> Self {
            Self {
                n_components: 100,
                gamma,
            }
        }

        pub fn n_components(mut self, n_components: usize) -> Self {
            self.n_components = n_components;
            self
        }

        pub fn build(self) -> ApproximationMethod {
            ApproximationMethod::RandomFourierFeatures {
                n_components: self.n_components,
                gamma: self.gamma,
            }
        }
    }

    /// Builder for Structured Random Features
    pub struct StructuredRFFBuilder {
        n_components: usize,
        gamma: Float,
    }

    impl StructuredRFFBuilder {
        pub fn new(gamma: Float) -> Self {
            Self {
                n_components: 100,
                gamma,
            }
        }

        pub fn n_components(mut self, n_components: usize) -> Self {
            self.n_components = n_components;
            self
        }

        pub fn build(self) -> ApproximationMethod {
            ApproximationMethod::StructuredRandomFeatures {
                n_components: self.n_components,
                gamma: self.gamma,
            }
        }
    }

    /// Builder for Fastfood Transform
    pub struct FastfoodBuilder {
        n_components: usize,
        gamma: Float,
    }

    impl FastfoodBuilder {
        pub fn new(gamma: Float) -> Self {
            Self {
                n_components: 100,
                gamma,
            }
        }

        pub fn n_components(mut self, n_components: usize) -> Self {
            self.n_components = n_components;
            self
        }

        pub fn build(self) -> ApproximationMethod {
            ApproximationMethod::Fastfood {
                n_components: self.n_components,
                gamma: self.gamma,
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximation_method_accessors() {
        let method = ApproximationMethod::RandomFourierFeatures {
            n_components: 200,
            gamma: 0.5,
        };

        assert_eq!(method.n_components(), 200);
        assert_eq!(method.gamma(), Some(0.5));
        assert!(method.kernel().is_none());
        assert!(method.sampling_strategy().is_none());
    }

    #[test]
    fn test_solver_properties() {
        let solver = Solver::ConjugateGradient {
            max_iter: 1000,
            tol: 1e-6,
        };

        assert_eq!(solver.name(), "conjugate_gradient");
        assert!(solver.is_iterative());
        assert_eq!(solver.max_iter(), Some(1000));
        assert_eq!(solver.tolerance(), Some(1e-6));
    }

    #[test]
    fn test_ridge_config_builder() {
        let config = RidgeConfig::new(0.1)
            .with_solver(Solver::SVD)
            .with_fit_intercept(false)
            .with_normalize(true);

        assert_eq!(config.alpha, 0.1);
        assert!(matches!(config.solver, Solver::SVD));
        assert!(!config.fit_intercept);
        assert!(config.normalize);
    }

    #[test]
    fn test_validation_functions() {
        use scirs2_core::ndarray::{array, Array1, Array2};

        let x: Array2<Float> = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<Float> = array![1.0, 2.0];

        assert!(validation::validate_fit_input(&x, &y).is_ok());

        let y_wrong: Array1<Float> = array![1.0]; // Wrong size
        assert!(validation::validate_fit_input(&x, &y_wrong).is_err());

        assert!(validation::validate_alpha(1.0).is_ok());
        assert!(validation::validate_alpha(-1.0).is_err());
    }
}
