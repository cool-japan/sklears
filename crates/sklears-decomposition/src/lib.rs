#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
//! Matrix and tensor decomposition algorithms for dimensionality reduction
//!
//! This module provides various decomposition techniques including:
//! - PCA: Principal Component Analysis with SVD (including Randomized SVD)
//! - Incremental PCA: Memory-efficient PCA for large datasets
//! - Kernel PCA: Non-linear dimensionality reduction using kernel methods
//! - ICA: Independent Component Analysis (including constrained ICA)
//! - NMF: Non-negative Matrix Factorization
//! - Factor Analysis: Statistical model for latent variables
//! - Dictionary Learning: Sparse coding and dictionary learning
//! - Tensor Decomposition: CP (CANDECOMP/PARAFAC) and Tucker decomposition
//! - Matrix Completion: Filling missing values using low-rank matrix completion
//! - CCA: Canonical Correlation Analysis for finding linear relationships between two datasets
//! - PLS: Partial Least Squares for regression and dimensionality reduction
//! - Time Series: SSA, seasonal decomposition, and trend extraction
//! - Signal Processing: EMD, spectral decomposition, and adaptive methods
//! - Image & Computer Vision: 2D-PCA, image denoising, face recognition, texture analysis
//! - Manifold Learning: LLE, Isomap, Laplacian Eigenmaps, t-SNE, UMAP
//! - Component Selection: Cross-validation, bootstrap, information criteria, parallel analysis
//! - Quality Metrics: Goodness-of-fit statistics, reconstruction quality, interpretability measures
//! - Robust Methods: Robust PCA with L1 loss, M-estimators, outlier-resistant methods
//! - Hardware Acceleration: SIMD optimizations, parallel processing, and mixed-precision arithmetic
//! - Distributed Processing: Large-scale distributed decomposition across multiple nodes/workers
//! - Scikit-learn Compatibility: Drop-in replacements for scikit-learn transformers with full API compatibility
//! - Advanced Format Support: HDF5, sparse matrices, memory-mapped files, and compressed storage
//! - Cache Optimization: Memory-aligned data structures, tiled algorithms, and performance analysis
//! - Comprehensive Validation: Input validation, parameter checking, and result quality assessment
//! - Modular Architecture: Pluggable algorithms, preprocessing pipelines, and extensible framework
//! - Constrained Decomposition: Orthogonality, non-negativity, sparsity, and smoothness constraints
//! - Type-Safe Decomposition: Zero-cost abstractions with compile-time dimension and rank checking

// Import the s! macro from scirs2_core for array slicing
#[allow(unused_imports)]
pub use scirs2_core::s;

pub mod cache_optimization;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod cca;
pub mod component_selection;
pub mod constrained_decomposition;
pub mod dictionary_learning;
pub mod distributed;
pub mod error_diagnostics;
pub mod factor_analysis;
pub mod fluent_api;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod format_support;
pub mod hardware_acceleration;
pub mod ica;
pub mod image_cv;
pub mod integration;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod incremental_pca;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod kernel_pca;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod manifold;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod matrix_completion;
pub mod memory_efficiency;
pub mod modular_framework;
pub mod nmf;
pub mod online_nmf;
pub mod pca;
pub mod performance;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod pls;
pub mod quality_metrics;
pub mod robust_methods;
pub mod signal_processing;
mod simd_signal;
pub mod sklearn_compat;
pub mod streaming;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod tensor_decomposition;
pub mod time_series;
pub mod type_safe;
pub mod validation;
pub mod visualization;

#[allow(non_snake_case)]
#[cfg(test)]
// TODO: Migrate to scirs2-linalg (uses nalgebra/ICA types)
// pub mod property_tests;
pub use cache_optimization::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use cca::*;
pub use component_selection::*;
pub use constrained_decomposition::*;
pub use dictionary_learning::*;
pub use distributed::*;
pub use error_diagnostics::*;
pub use factor_analysis::*;
pub use fluent_api::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use format_support::*;
pub use hardware_acceleration::{
    AccelerationConfig, AlignedMemoryOps, MixedPrecisionOps, ParallelDecomposition, SimdMatrixOps,
};
#[cfg(feature = "gpu")]
pub use hardware_acceleration::{GpuAcceleration, GpuDecomposition};
pub use ica::*;
pub use image_cv::*;
pub use integration::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use incremental_pca::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use kernel_pca::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use manifold::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use matrix_completion::*;
pub use memory_efficiency::*;
// Re-export modular_framework excluding DecompositionAlgorithm enum and DecompositionPipeline struct to avoid conflicts
pub use modular_framework::{
    AlgorithmCapabilities, AlgorithmCapability, AlgorithmMetadata, AlgorithmRegistry,
    ComputationalComplexity, DecompositionAlgorithm as DecompositionAlgorithmTrait,
    DecompositionComponents, DecompositionParams, DecompositionWorkflowBuilder, MatrixProperty,
    ParamValue, PostprocessingStep, PreprocessingStep, StandardizationStep, VarimaxRotationStep,
};
pub use nmf::*;
pub use online_nmf::*;
pub use pca::*;
pub use performance::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use pls::*;
pub use quality_metrics::*;
pub use robust_methods::{
    BreakdownPointAnalysis, BreakdownResult, LossFunction, MEstimatorDecomposition,
    MEstimatorResult, RobustConfig, RobustPCAResult,
};
pub use signal_processing::*;
// Re-export sklearn_compat with ParameterValue aliased to avoid conflict with validation
pub use sklearn_compat::{
    CrossValidation, GridSearchCV, ParameterValue as SklearnParameterValue, SklearnPCA,
    SklearnPipeline, SklearnTransformer,
};
pub use streaming::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use tensor_decomposition::*;
pub use time_series::*;
// Re-export type_safe with DecompositionPipeline aliased to avoid conflict
pub use type_safe::{
    CenteringOperation, ComponentAccess, ComponentIndex, DecompositionOperation,
    DecompositionPipeline as TypeSafeDecompositionPipeline, DecompositionState, Dimensions, Fitted,
    Rank, ScalingOperation, TypeSafeDecomposition, TypeSafeMatrix, TypeSafePCA, Untrained,
};
pub use validation::*;
pub use visualization::*;
