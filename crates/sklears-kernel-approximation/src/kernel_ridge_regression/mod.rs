//! Kernel Ridge Regression Module
//!
//! This module provides comprehensive kernel ridge regression implementations with
//! various approximation methods, regularization strategies, and robust variants.
//!
//! ## Overview
//!
//! Kernel ridge regression extends linear ridge regression to nonlinear problems
//! by using kernel methods. This module includes:
//!
//! - **Basic Kernel Ridge Regression**: Standard implementation with multiple solvers
//! - **Multi-Task Learning**: Joint learning across multiple related regression tasks
//! - **Robust Variants**: Resistance to outliers using robust loss functions
//! - **Kernel Approximation**: Efficient large-scale methods using feature approximations
//!
//! ## Kernel Approximation Methods
//!
//! All implementations support multiple kernel approximation methods for scalability:
//!
//! - **Nyström Method**: Landmark-based kernel approximation
//! - **Random Fourier Features**: Approximate shift-invariant kernels
//! - **Structured Random Features**: Computational improvements to RFF
//! - **Fastfood Transform**: Fast Walsh-Hadamard based approximation
//!
//! ## Architecture
//!
//! The module is organized into focused submodules:
//!
//! - [`core_types`] - Shared types, enums, and utility functions
//! - [`basic_regression`] - Standard kernel ridge regression
//! - [`multitask_regression`] - Multi-task learning capabilities
//! - [`robust_regression`] - Robust variants for outlier resistance
//!
//! ## Examples
//!
//! ### Basic Usage
//!
//! ```rust,ignore
//! use sklears_kernel_approximation::kernel_ridge_regression::{
//!     KernelRidgeRegression, ApproximationMethod
//! };
//! use sklears_core::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! // Prepare data
//! let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
//! let y = array![1.0, 4.0, 9.0, 16.0];
//!
//! // Create approximation method
//! let approximation = ApproximationMethod::RandomFourierFeatures {
//!     n_components: 100,
//!     gamma: 1.0,
//! };
//!
//! // Create and fit model
//! let krr = KernelRidgeRegression::new(approximation)
//!     .alpha(0.1)
//!     .random_state(42);
//!
//! let fitted_model = krr.fit(&X, &y)?;
//! let predictions = fitted_model.predict(&X)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Multi-Task Learning
//!
//! ```rust,ignore
//! use sklears_kernel_approximation::kernel_ridge_regression::{
//!     MultiTaskKernelRidgeRegression, ApproximationMethod, TaskRegularization
//! };
//! use sklears_core::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
//! let Y = array![[1.0, 2.0], [4.0, 5.0], [9.0, 10.0]]; // Two related tasks
//!
//! let approximation = ApproximationMethod::RandomFourierFeatures {
//!     n_components: 50,
//!     gamma: 1.0,
//! };
//!
//! let mtkrr = MultiTaskKernelRidgeRegression::new(approximation)
//!     .alpha(0.1)
//!     .task_regularization(TaskRegularization::L2 { beta: 0.01 });
//!
//! let fitted_model = mtkrr.fit(&X, &Y)?;
//! let predictions = fitted_model.predict(&X)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Robust Regression
//!
//! ```rust,ignore
//! use sklears_kernel_approximation::kernel_ridge_regression::{
//!     RobustKernelRidgeRegression, ApproximationMethod, RobustLoss
//! };
//! use sklears_core::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [10.0, 10.0]]; // Last point is outlier
//! let y = array![1.0, 2.0, 3.0, 100.0]; // Last target is outlier
//!
//! let approximation = ApproximationMethod::RandomFourierFeatures {
//!     n_components: 50,
//!     gamma: 1.0,
//! };
//!
//! let robust_krr = RobustKernelRidgeRegression::new(approximation)
//!     .alpha(0.1)
//!     .robust_loss(RobustLoss::Huber { delta: 1.0 });
//!
//! let fitted_model = robust_krr.fit(&X, &y)?;
//! let predictions = fitted_model.predict(&X)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Module declarations
pub mod basic_regression;
pub mod core_types;
pub mod multitask_regression;
pub mod robust_regression;

// Re-export core types and utilities
pub use crate::Untrained;
pub use core_types::{
    ApproximationMethod, FastfoodTransform, FeatureTransformer, Float, Kernel, Nystroem,
    RBFSampler, SamplingStrategy, Solver, StructuredRandomFeatures, Trained,
};

// Re-export basic kernel ridge regression
pub use basic_regression::{IntoUntrained, KernelRidgeRegression, OnlineKernelRidgeRegression};

// Re-export multi-task kernel ridge regression
pub use multitask_regression::{MultiTaskKernelRidgeRegression, TaskRegularization};

// Re-export robust kernel ridge regression
pub use robust_regression::{RobustKernelRidgeRegression, RobustLoss};

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and traits
/// for kernel ridge regression.
pub mod prelude {
    pub use super::basic_regression::{KernelRidgeRegression, OnlineKernelRidgeRegression};
    pub use super::core_types::{ApproximationMethod, Kernel, SamplingStrategy, Solver};
    pub use super::multitask_regression::{MultiTaskKernelRidgeRegression, TaskRegularization};
    pub use super::robust_regression::{RobustKernelRidgeRegression, RobustLoss};
    pub use sklears_core::prelude::{Estimator, Fit, Predict};
}

/// Utility functions for kernel ridge regression
pub mod utils {
    use super::core_types::*;
    use super::multitask_regression::TaskRegularization;
    use super::robust_regression::RobustLoss;
    use super::Untrained;
    use scirs2_core::ndarray::{Array1, Array2};
    use sklears_core::error::Result;
    use sklears_core::prelude::Float;

    /// Create a basic kernel ridge regression model with Random Fourier Features
    pub fn create_rff_model(
        n_components: usize,
        gamma: Float,
        alpha: Float,
    ) -> KernelRidgeRegression<Untrained> {
        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components,
            gamma,
        };
        KernelRidgeRegression::new(approximation).alpha(alpha)
    }

    /// Create a basic kernel ridge regression model with Nyström approximation
    pub fn create_nystroem_model(
        kernel: Kernel,
        n_components: usize,
        alpha: Float,
    ) -> KernelRidgeRegression<Untrained> {
        let approximation = ApproximationMethod::Nystroem {
            kernel,
            n_components,
            sampling_strategy: SamplingStrategy::Random,
        };
        KernelRidgeRegression::new(approximation).alpha(alpha)
    }

    /// Create a multi-task model with L2 task regularization
    pub fn create_multitask_l2_model(
        approximation: ApproximationMethod,
        alpha: Float,
        beta: Float,
    ) -> MultiTaskKernelRidgeRegression<Untrained> {
        MultiTaskKernelRidgeRegression::new(approximation)
            .alpha(alpha)
            .task_regularization(TaskRegularization::L2 { beta })
    }

    /// Create a robust model with Huber loss
    pub fn create_robust_huber_model(
        approximation: ApproximationMethod,
        alpha: Float,
        delta: Float,
    ) -> RobustKernelRidgeRegression<Untrained> {
        RobustKernelRidgeRegression::new(approximation)
            .alpha(alpha)
            .robust_loss(RobustLoss::Huber { delta })
    }

    /// Validate input dimensions for kernel ridge regression
    pub fn validate_input_dimensions(x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(),
                y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays must not be empty".to_string(),
            ));
        }

        if x.ncols() == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input must have at least one feature".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate input dimensions for multi-task kernel ridge regression
    pub fn validate_multitask_input_dimensions(x: &Array2<Float>, y: &Array2<Float>) -> Result<()> {
        if x.nrows() != y.nrows() {
            return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and Y ({}) must match",
                x.nrows(),
                y.nrows()
            )));
        }

        if x.nrows() == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays must not be empty".to_string(),
            ));
        }

        if x.ncols() == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input must have at least one feature".to_string(),
            ));
        }

        if y.ncols() == 0 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Must have at least one task".to_string(),
            ));
        }

        Ok(())
    }

    use super::basic_regression::KernelRidgeRegression;
    use super::multitask_regression::MultiTaskKernelRidgeRegression;
    use super::robust_regression::RobustKernelRidgeRegression;
}
