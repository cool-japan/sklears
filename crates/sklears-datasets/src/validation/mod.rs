//! Statistical property validation framework for dataset quality assurance
//!
//! This module provides comprehensive validation functions to ensure generated datasets
//! meet statistical expectations and quality standards, along with detailed statistical
//! summary generation capabilities.

// Module declarations
pub mod advanced;
pub mod basic;
pub mod distributions;
pub mod summary;
pub mod types;

// Re-export all public types and functions to maintain API compatibility
pub use types::{
    AnomalyDetectionResult, DataDriftReport, DatasetQualityMetrics, FeatureStatistics,
    StatisticalSummary, SummaryConfig, TargetStatistics, ValidationConfig, ValidationReport,
    ValidationResult,
};

pub use basic::{
    validate_basic_statistics, validate_correlation_structure, validate_distribution_properties,
    validate_normality, validate_outliers,
};

pub use distributions::{
    chi_square_goodness_of_fit_test, kolmogorov_smirnov_test, validate_exponential_distribution,
    validate_normal_distribution, validate_uniform_distribution, DistributionType,
};

pub use advanced::{
    calculate_dataset_quality_metrics, detect_anomalies, detect_data_drift, validate_dataset,
};

pub use summary::generate_statistical_summary;
