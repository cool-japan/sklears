//! Kernel-based imputation methods
//!
//! This module provides imputation strategies using kernel methods including
//! kernel ridge regression, support vector regression, Gaussian processes,
//! and reproducing kernel Hilbert space methods.

pub mod gaussian_process;
pub mod kernel_ridge;
pub mod reproducing_kernel;
pub mod svr;

// Re-export all the main types for convenience
pub use gaussian_process::{
    GPPredictionResult, GaussianProcessImputer, GaussianProcessImputerTrained,
};
pub use kernel_ridge::{KernelRidgeImputer, KernelRidgeImputerTrained};
pub use reproducing_kernel::{ReproducingKernelImputer, ReproducingKernelImputerTrained};
pub use svr::{SVRImputer, SVRImputerTrained};
