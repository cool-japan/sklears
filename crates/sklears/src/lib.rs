#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_doc_comments)]
#![allow(unused_parens)]
#![allow(unused_comparisons)]
//! Machine learning library for Rust
//!
//! This library provides comprehensive machine learning implementations including
//! linear models, tree-based methods, neural networks, clustering algorithms,
//! preprocessing utilities, evaluation metrics, and more. It follows scikit-learn
//! API patterns and is built on top of SciRS2 for scientific computing.

// Core traits and utilities - always available
pub use sklears_core::*;
pub use sklears_utils::*;

// Feature-gated algorithm modules
#[cfg(feature = "linear")]
pub use sklears_linear as linear;

#[cfg(feature = "clustering")]
pub use sklears_clustering as clustering;

#[cfg(feature = "ensemble")]
pub use sklears_ensemble as ensemble;

#[cfg(feature = "svm")]
pub use sklears_svm as svm;

#[cfg(feature = "tree")]
pub use sklears_tree as tree;

#[cfg(feature = "neighbors")]
pub use sklears_neighbors as neighbors;

#[cfg(feature = "decomposition")]
pub use sklears_decomposition as decomposition;

#[cfg(feature = "model-selection")]
pub use sklears_model_selection as model_selection;

#[cfg(feature = "metrics")]
pub use sklears_metrics as metrics;

#[cfg(feature = "neural")]
pub use sklears_neural as neural;

#[cfg(feature = "datasets")]
pub use sklears_datasets as datasets;

#[cfg(feature = "feature-selection")]
pub use sklears_feature_selection as feature_selection;

#[cfg(feature = "naive-bayes")]
pub use sklears_naive_bayes as naive_bayes;

#[cfg(feature = "gaussian-process")]
pub use sklears_gaussian_process as gaussian_process;

#[cfg(feature = "discriminant-analysis")]
pub use sklears_discriminant_analysis as discriminant_analysis;

#[cfg(feature = "manifold")]
pub use sklears_manifold as manifold;

#[cfg(feature = "semi-supervised")]
pub use sklears_semi_supervised as semi_supervised;

#[cfg(feature = "feature-extraction")]
pub use sklears_feature_extraction as feature_extraction;

#[cfg(feature = "covariance")]
pub use sklears_covariance as covariance;

#[cfg(feature = "cross-decomposition")]
pub use sklears_cross_decomposition as cross_decomposition;

#[cfg(feature = "isotonic")]
pub use sklears_isotonic as isotonic;

#[cfg(feature = "kernel-approximation")]
pub use sklears_kernel_approximation as kernel_approximation;

#[cfg(feature = "dummy")]
pub use sklears_dummy as dummy;

#[cfg(feature = "calibration")]
pub use sklears_calibration as calibration;

#[cfg(feature = "multiclass")]
pub use sklears_multiclass as multiclass;

#[cfg(feature = "multioutput")]
pub use sklears_multioutput as multioutput;

#[cfg(feature = "compose")]
pub use sklears_compose as compose;

#[cfg(feature = "impute")]
pub use sklears_impute as impute;

#[cfg(feature = "inspection")]
pub use sklears_inspection as inspection;

#[cfg(feature = "mixture")]
pub use sklears_mixture as mixture;
