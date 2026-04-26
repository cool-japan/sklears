//! Feature engineering and automated preprocessing utilities
//!
//! This module provides comprehensive feature engineering implementations including
//! automatic feature transformation, feature selection, statistical testing, feature type detection,
//! data preprocessing pipelines, outlier detection, missing value handling, feature interaction analysis,
//! dimensionality reduction, polynomial feature generation, discretization techniques, normalization methods,
//! imbalanced data handling, automated ML pipelines, feature importance analysis, and
//! high-performance feature engineering pipelines. All algorithms have been refactored
//! into focused modules for better maintainability and comply with SciRS2 Policy.

// Core feature engineering types and base structures
#[allow(dead_code)]
mod feature_engineering_core;

// Feature transformation methods and scaling
#[allow(dead_code)]
mod feature_transformations;
pub use feature_transformations::TransformMethod;

// Feature type detection and analysis
#[allow(dead_code)]
mod feature_type_detection;
pub use feature_type_detection::FeatureType;

// Feature selection methods and statistical testing
#[allow(dead_code)]
mod feature_selection;
pub use feature_selection::{FeatureSelectionMethod, FeatureSelectionResults};

// Statistical tests and hypothesis testing
#[allow(dead_code)]
mod statistical_testing;
pub use statistical_testing::StatisticalTest;

// Automated feature transformation and optimization
#[allow(dead_code)]
mod automated_transformation;
pub use automated_transformation::{AutoFeatureTransformer, AutoTransformConfig};

// Feature interaction detection and analysis
#[allow(dead_code)]
mod feature_interactions;
pub use feature_interactions::{FeatureInteractionDetector, InteractionMethod, InteractionResults};

// Automated preprocessing pipelines and workflows
#[allow(dead_code)]
mod automated_preprocessing;
pub use automated_preprocessing::{AutoPipelineConfig, AutomatedPreprocessingPipeline};

// Outlier detection and anomaly handling
#[allow(dead_code)]
mod outlier_detection;
pub use outlier_detection::OutlierDetectionMethod;

// Missing value handling and imputation
#[allow(dead_code)]
mod missing_value_handling;
pub use missing_value_handling::MissingValueStrategy;

// Imbalanced data handling and resampling
#[allow(dead_code)]
mod imbalance_handling;
pub use imbalance_handling::ImbalanceHandlingMethod;

// Data validation and quality assessment
#[allow(dead_code)]
mod data_validation;
pub use data_validation::DataValidationConfig;

// Discretization and binning techniques
#[allow(dead_code)]
mod discretization;

// Polynomial features and feature engineering
#[allow(dead_code)]
mod polynomial_features;

// Performance optimization and computational efficiency
#[allow(dead_code)]
mod performance_optimization;

// Utilities and helper functions
#[allow(dead_code)]
mod feature_engineering_utilities;
