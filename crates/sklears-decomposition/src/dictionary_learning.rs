//! Dictionary Learning implementation
//!
//! This module provides comprehensive dictionary learning algorithms including sparse coding,
//! matrix factorization, and online learning methods. Dictionary learning aims at finding
//! sparse representations of input data as linear combinations of basic elements (atoms).
//! All algorithms have been refactored into focused modules for better maintainability
//! and comply with SciRS2 Policy.

// Core dictionary learning types and configuration
mod dictionary_learning_core;
pub use dictionary_learning_core::{
    DictionaryLearning, DictionaryLearningConfig, DictionaryTransformAlgorithm,
    TrainedDictionaryLearning,
};

// Orthogonal Matching Pursuit (OMP) algorithms
mod omp_algorithms;
pub use omp_algorithms::{OMPConfig, OMPEncoder, OMPResult};

// Least Angle Regression (LARS) algorithms
mod lars_algorithms;
pub use lars_algorithms::{LARSConfig, LARSDirection, LARSEncoder, LARSResult, LARSStepSize};

// Coordinate Descent (CD) algorithms
mod coordinate_descent;
pub use coordinate_descent::{CDConfig, CDEncoder, CDResult, SoftThresholding};

// K-SVD algorithms
mod ksvd_algorithms;
pub use ksvd_algorithms::{KSVDConfig, KSVDEncoder, KSVDResult};

// Online dictionary learning
mod online_dictionary_learning;
pub use online_dictionary_learning::{
    AdaptiveDictionaryLearning, OnlineDictLearningAlgorithm, OnlineDictionaryLearning,
    OnlineGradientDescent, OnlineKSvd, TrainedOnlineDictionaryLearning,
};

// Mini-batch dictionary learning
mod minibatch_dictionary_learning;
pub use minibatch_dictionary_learning::{
    MiniBatchConfig, MiniBatchDictionaryLearning, MiniBatchResult,
    TrainedMiniBatchDictionaryLearning,
};

// Sparse encoding utilities
mod sparse_encoding;
pub use sparse_encoding::{
    EncodingAlgorithm, SparseCoder, SparseEncoder, SparseEncodingConfig, SparseEncodingResult,
};

// Dictionary update algorithms
mod dictionary_update;
pub use dictionary_update::{
    AtomUpdater, DictionaryUpdateResult, DictionaryUpdater, UpdateAlgorithm, UpdateConfig,
};

// Linear algebra utilities for dictionary learning
mod dictionary_linalg;
pub use dictionary_linalg::{
    CholeskyDecomposition, LeastSquaresSolver, LinearSystemSolver, MatrixFactorization,
    QRDecomposition, SVDDecomposition,
};

// Convergence monitoring and optimization
mod convergence_monitoring;
pub use convergence_monitoring::{
    ConvergenceConfig, ConvergenceMonitor, ConvergenceResult, LearningCurve, OptimizationMetrics,
};

// Dictionary initialization strategies
mod dictionary_initialization;
pub use dictionary_initialization::{
    DataDrivenInitializer, DictionaryInitializer, InitializationConfig, InitializationStrategy,
    OrthogonalInitializer, RandomInitializer,
};

// Evaluation and metrics for dictionary learning
mod dictionary_metrics;
pub use dictionary_metrics::{
    CoherenceMetrics, DictionaryMetrics, DictionaryQuality, ReconstructionError, SparsityMetrics,
};
