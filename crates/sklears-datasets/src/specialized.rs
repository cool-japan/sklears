//! Specialized data generators
//!
//! This module provides a comprehensive collection of specialized generators for various
//! machine learning and data analysis tasks. The generators are organized into focused
//! submodules for better maintainability and easier navigation.
//!
//! # Module Organization
//!
//! - [`classification_regression`] - Classification and regression dataset generators
//! - [`timeseries`] - Time series generators with various patterns
//! - [`graphs`] - Graph and network topology generators
//! - [`distributions`] - Statistical distribution generators
//! - [`missing_data`] - Missing data patterns and outlier generators
//! - [`domain_specific`] - Domain-specific generators (bioinformatics, NLP, CV, etc.)
//! - [`manifolds_spatial`] - Manifold and spatial pattern generators
//!
//! # Usage
//!
//! All functions are re-exported at this module level for backward compatibility:
//!
//! ```rust
//! use sklears_datasets::specialized::make_multilabel_classification;
//!
//! let (X, y) = make_multilabel_classification(100, 10, 5, 2, Some(42))?;
//! ```
//!
//! You can also import from specific submodules:
//!
//! ```rust
//! use sklears_datasets::specialized::classification_regression::make_multilabel_classification;
//! ```
//!
//! # Architectural Benefits
//!
//! This modular organization provides several advantages:
//!
//! - **Focused Responsibility**: Each submodule handles a specific domain of data generation
//! - **Improved Maintainability**: Smaller, focused modules are easier to understand and modify
//! - **Better Testability**: Domain-specific modules can be tested independently
//! - **Enhanced Documentation**: Each submodule provides detailed documentation for its domain
//! - **SciRS2 Compliance**: All modules use SciRS2 for consistent random generation and array operations
//! - **Performance**: Modular compilation allows for better optimization and faster builds

// Submodule declarations
pub mod classification_regression;
pub mod timeseries;
pub mod graphs;
pub mod distributions;
pub mod missing_data;
pub mod domain_specific;
pub mod manifolds_spatial;

// Re-export all public functions for backward compatibility

// Classification and regression generators
pub use classification_regression::{
    make_multilabel_classification,
    make_sparse_uncorrelated,
    make_polynomial_regression,
};

// Time series generators
pub use timeseries::{
    make_nonstationary_timeseries,
    make_stationary_arma,
};

// Graph generators
pub use graphs::{
    make_erdos_renyi_graph,
    make_barabasi_albert_graph,
    make_watts_strogatz_graph,
    make_stochastic_block_graph,
    make_random_tree,
};

// Distribution generators
pub use distributions::{
    make_gaussian_mixture,
    make_distribution_mixture,
    make_multivariate_mixture,
    make_heavy_tailed_distribution,
};

// Missing data and outlier generators
pub use missing_data::{
    make_missing_completely_at_random,
    make_missing_at_random,
    make_missing_not_at_random,
    make_outliers,
    make_imbalanced_classification,
    make_anomalies,
};

// Domain-specific generators
pub use domain_specific::{
    make_gene_expression_dataset,
    make_dna_sequence_dataset,
    make_document_clustering_dataset,
    make_synthetic_image_classification,
    make_privacy_preserving_dataset,
    make_multi_agent_environment,
    make_ab_testing_simulation,
};

// Manifold and spatial generators
pub use manifolds_spatial::{
    ManifoldGenerator,
    make_custom_manifold,
    make_n_sphere,
    make_n_torus,
    make_spatial_point_pattern,
    make_geostatistical_data,
};