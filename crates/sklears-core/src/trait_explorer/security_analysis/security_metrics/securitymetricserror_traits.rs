//! # `SecurityMetricsError` - Trait Implementations
//!
//! This module contains trait implementations for `SecurityMetricsError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_7::SecurityMetricsError;

impl std::fmt::Display for SecurityMetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityMetricsError::CollectionError(msg) => {
                write!(f, "Collection error: {}", msg)
            }
            SecurityMetricsError::AnalysisError(msg) => {
                write!(f, "Analysis error: {}", msg)
            }
            SecurityMetricsError::StorageError(msg) => {
                write!(f, "Storage error: {}", msg)
            }
            SecurityMetricsError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            SecurityMetricsError::DataQualityError(msg) => {
                write!(f, "Data quality error: {}", msg)
            }
            SecurityMetricsError::VisualizationError(msg) => {
                write!(f, "Visualization error: {}", msg)
            }
            SecurityMetricsError::AlertingError(msg) => {
                write!(f, "Alerting error: {}", msg)
            }
            SecurityMetricsError::BenchmarkingError(msg) => {
                write!(f, "Benchmarking error: {}", msg)
            }
            SecurityMetricsError::CorrelationError(msg) => {
                write!(f, "Correlation error: {}", msg)
            }
            SecurityMetricsError::ForecastingError(msg) => {
                write!(f, "Forecasting error: {}", msg)
            }
        }
    }
}

impl std::error::Error for SecurityMetricsError {}

