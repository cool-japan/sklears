#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![recursion_limit = "1048576"]
#![allow(deprecated)]
#![allow(ambiguous_glob_reexports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrow)]
//! Support Vector Machines for classification and regression
//!
//! This module provides Support Vector Machine implementations including:
//! - SVC: Support Vector Classification
//! - SVR: Support Vector Regression
//! - LinearSVC: Linear Support Vector Classification (coordinate descent)
//! - LinearSVR: Linear Support Vector Regression (coordinate descent)
//! - SGDClassifier: Stochastic Gradient Descent SVM for large-scale learning
//! - NuSVC: Nu Support Vector Classification with automatic parameter selection
//! - NuSVR: Nu Support Vector Regression
//! - LSSVM: Least Squares Support Vector Machine for efficient training
//! - RobustSVM: Robust SVM with Huber and other robust loss functions
//! - OutlierResistantSVM: Outlier-resistant SVM with automatic outlier detection and handling
//! - FuzzySVM: Fuzzy SVM for handling noisy and uncertain data
//! - RankingSVM: Ranking SVM for learning-to-rank and structured output problems
//! - OrdinalRegressionSVM: Ordinal regression SVM for ordered categorical targets
//! - BinaryRelevanceSVM: Multi-label SVM using binary relevance strategy
//! - ClassifierChainsSVM: Multi-label SVM using classifier chains
//! - LabelPowersetSVM: Multi-label SVM using label powerset transformation
//! - StructuredSVM: Structured SVM for sequence labeling and structured prediction
//! - MetricLearningSVM: Metric learning SVM for learning optimal distance metrics
//! - TransductiveSVM: Transductive SVM for semi-supervised learning with unlabeled data
//! - SelfTrainingSVM: Self-training SVM for iterative semi-supervised learning
//! - CoTrainingSVM: Co-training SVM using multiple views for semi-supervised learning
//! - KernelPCA: Kernel Principal Component Analysis for dimensionality reduction
//! - OnlineSVM: Online learning for streaming data
//! - OutOfCoreSVM: Out-of-core training for datasets larger than memory
//! - DistributedSVM: Distributed training across multiple processes/machines
//! - AdaptiveSVM: Adaptive regularization with automatic parameter selection
//! - ADMMSVM: Alternating Direction Method of Multipliers for distributed optimization
//! - NewtonSVM: Newton methods for fast second-order optimization
//! - GridSearchCV: Grid search for hyperparameter optimization
//! - RandomSearchCV: Random search for hyperparameter optimization
//! - BayesianOptimizationCV: Bayesian optimization for efficient hyperparameter tuning
//! - Various kernel functions (Linear, RBF, Polynomial, Graph kernels, etc.)
//! - SMO algorithm for training

// Re-export common types for all modules to use
pub use sklears_core::prelude::*;

pub mod adaptive_regularization;
pub mod calibration;
pub mod chunked_processing;
pub mod compressed_kernels;
pub mod computer_vision_kernels;
pub mod crammer_singer;
pub mod decomposition;
pub mod distributed_svm;
pub mod dual_coordinate_ascent;
pub mod errors;
pub mod fuzzy_svm;
pub mod gpu_kernels;
pub mod graph_semi_supervised;
pub mod group_lasso_svm;
pub mod hyperparameter_optimization;
pub mod kernel_pca;
pub mod kernels;
pub mod linear_svc;
pub mod linear_svr;
pub mod ls_svm;
pub mod memory_mapped_kernels;
pub mod metric_learning_svm;
pub mod multi_label_svm;
pub mod multiclass;
pub mod nusvc;
pub mod nusvr;
pub mod online_svm;
pub mod ordinal_regression_svm;
pub mod out_of_core_svm;
pub mod outlier_resistant_svm;
pub mod parallel_smo;
pub mod primal_dual_methods;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod property_tests;
pub mod ranking_svm;
pub mod regularization_path;
pub mod robust_svm;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
//pub mod semi_supervised;
pub mod sgd_svm;
pub mod simd_kernels;
pub mod smo;
pub mod sparse_svm;
pub mod structured_svm;
pub mod svc;
pub mod svr;
pub mod text_classification;
pub mod thread_safe_cache;
pub mod time_series;
pub mod topic_model_integration;
pub mod visualization;

#[allow(non_snake_case)]
#[cfg(test)]
mod elastic_net_tests;

pub use adaptive_regularization::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use advanced_optimization::*;
pub use calibration::*;
pub use chunked_processing::*;
pub use compressed_kernels::*;
pub use computer_vision_kernels::*;
pub use crammer_singer::*;
pub use decomposition::*;
pub use distributed_svm::*;
pub use dual_coordinate_ascent::*;
pub use errors::{ErrorSeverity, SVMError, SVMResult};
pub use fuzzy_svm::*;
pub use gpu_kernels::*;
pub use graph_semi_supervised::*;
pub use group_lasso_svm::*;
pub use hyperparameter_optimization::{
    BayesianOptimizationCV, EvolutionaryOptimizationCV, GridSearchCV, OptimizationConfig,
    OptimizationResult, ParameterSet, ParameterSpec, RandomSearchCV, ScoringMetric, SearchSpace,
};
pub use kernel_pca::*;
pub use kernels::*;
pub use linear_svc::*;
pub use linear_svr::*;
pub use ls_svm::*;
pub use memory_mapped_kernels::*;
pub use metric_learning_svm::*;
pub use multi_label_svm::*;
pub use multiclass::*;
pub use nusvc::*;
pub use nusvr::*;
pub use online_svm::*;
pub use ordinal_regression_svm::*;
pub use out_of_core_svm::*;
pub use outlier_resistant_svm::*;
pub use parallel_smo::*;
pub use primal_dual_methods::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use property_tests::*;
pub use ranking_svm::*;
pub use regularization_path::*;
pub use robust_svm::*;
// TODO: Migrate to scirs2-linalg (uses nalgebra types)
// pub use semi_supervised::*;
pub use sgd_svm::*;
pub use simd_kernels::*;
pub use smo::*;
pub use sparse_svm::*;
pub use structured_svm::*;
pub use svc::*;
pub use svr::*;
pub use text_classification::*;
pub use thread_safe_cache::*;
pub use time_series::*;
pub use topic_model_integration::*;
pub use visualization::*;
