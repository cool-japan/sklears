#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Isotonic regression
//!
//! This module is part of sklears, providing scikit-learn compatible
//! machine learning algorithms in Rust.
//!
//! This crate provides comprehensive isotonic regression functionality including:
//! - Basic isotonic regression with Pool Adjacent Violators Algorithm
//! - Robust loss functions (L1, L2, Huber, Quantile)
//! - Weighted isotonic regression
//! - Constraint handling and validation
//!
//! Additional advanced features are being progressively enabled as they pass compilation and testing.

// #![warn(missing_docs)]

// Core isotonic regression functionality (stable)
pub mod constraints;
pub mod core;
pub mod pav;
pub mod robust;
pub mod utils;

// Testing advanced modules one by one
pub mod algorithms;
pub mod fluent_api;
pub mod serialization;

// Advanced regularized isotonic regression
pub mod regularized;

// Advanced optimization algorithms
pub mod optimization;

// Convex optimization methods (semidefinite programming, cone programming, ADMM, etc.)
pub mod convex_optimization;

// Differential equations module (isotonic differential equations, boundary value problems, etc.)
pub mod differential_equations;

// Engineering applications module (stress-strain, fatigue, reliability, control, signal processing)
pub mod engineering_applications;

// Environmental science module (dose-response, threshold estimation, climate, pollution, ecosystem)
pub mod environmental_science;

// Machine learning integration module (neural networks, deep learning, ensemble methods, transfer learning)
pub mod ml_integration;

// Advanced Bayesian methods module (nonparametric Bayesian, GP constraints, variational inference, MCMC)
pub mod advanced_bayesian;

// Advanced graph methods module (spectral, random walk, network-constrained, GNN integration)
pub mod graph_methods;

// Middleware for constraint pipelines
pub mod middleware;

// Real-world case studies and examples
pub mod case_studies;

// Unsafe optimizations for performance-critical paths
pub mod unsafe_optimizations;

// Advanced benchmarking suite
pub mod advanced_benchmarks;

// Compatibility layer (for backward compatibility with existing modules)
mod isotonic;

#[allow(non_snake_case)]
#[cfg(test)]
pub mod tests;

// Re-export core functionality
pub use constraints::*;
pub use core::*;
pub use pav::*;
pub use robust::{huber_weighted_mean, loss_functions, robust_statistics}; // Exclude weighted_quantile to avoid ambiguity
pub use utils::*;

// Re-export advanced functionality
pub use algorithms::*;
pub use fluent_api::*;
pub use serialization::*;

// Re-export regularized functionality
pub use regularized::*;

// Re-export optimization functionality (excluding conflicting names)
pub use optimization::{
    create_partial_order,
    interpolate_multidimensional,
    isotonic_regression_active_set,

    isotonic_regression_dual_decomposition,
    isotonic_regression_interior_point,

    isotonic_regression_projected_gradient,

    isotonic_regression_qp,
    non_separable_isotonic_regression,

    parallel_dual_decomposition,

    separable_isotonic_regression,
    simd_armijo_line_search,
    simd_constraint_violations,
    simd_dot_product,
    simd_gradient_computation,
    simd_hessian_approximation,
    simd_isotonic_projection,

    simd_newton_step,
    // SIMD operations
    simd_qp_matrix_vector_multiply,
    simd_vector_norm,
    sparse_isotonic_regression,

    ActiveSetIsotonicRegressor,
    BenchmarkResults,
    // Dual decomposition
    DualDecompositionIsotonicRegressor,
    // Interior point
    InteriorPointIsotonicRegressor,
    NonSeparableMultiDimensionalIsotonicRegression,
    // Configuration and benchmarking
    OptimizationAlgorithm,
    OptimizationConfig,
    // Projected gradient
    ProjectedGradientIsotonicRegressor,
    // Quadratic programming
    QuadraticProgrammingIsotonicRegressor,
    // Multidimensional
    SeparableMultiDimensionalIsotonicRegression,
    // Sparse
    SparseIsotonicRegression,
};

// Re-export differential equations functionality
pub use differential_equations::*;

// Re-export engineering applications functionality
pub use engineering_applications::*;

// Re-export environmental science functionality
pub use environmental_science::*;

// Re-export ML integration functionality
pub use ml_integration::*;

// Re-export advanced Bayesian functionality
pub use advanced_bayesian::*;

// Re-export graph methods functionality
pub use graph_methods::*;

// Re-export middleware functionality
pub use middleware::*;

// Re-export case studies
pub use case_studies::*;

// Re-export unsafe optimizations
pub use unsafe_optimizations::*;

// Re-export advanced benchmarks
pub use advanced_benchmarks::*;
