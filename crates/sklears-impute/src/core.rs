//! Core types and traits for imputation operations
//!
//! This module provides fundamental types, error handling, and traits
//! that are used throughout the imputation framework.

use std::fmt;

/// Result type for imputation operations
pub type ImputationResult<T> = Result<T, ImputationError>;

/// Error types for imputation operations
#[derive(Debug, Clone)]
pub enum ImputationError {
    /// Invalid parameter provided
    InvalidParameter(String),
    /// Insufficient data to perform imputation
    InsufficientData(String),
    /// Convergence failure in iterative methods
    ConvergenceFailure(String),
    /// Matrix operation error (e.g., singular matrix)
    MatrixError(String),
    /// Dimension mismatch between arrays
    DimensionMismatch(String),
    /// Numerical computation error
    NumericalError(String),
    /// Data validation error
    ValidationError(String),
    /// I/O operation error
    IOError(String),
    /// Memory allocation error
    MemoryError(String),
    /// Feature not implemented
    NotImplemented(String),
    /// General processing error
    ProcessingError(String),
    /// Invalid configuration error
    InvalidConfiguration(String),
}

impl fmt::Display for ImputationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImputationError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            ImputationError::InsufficientData(msg) => {
                write!(f, "Insufficient data: {}", msg)
            }
            ImputationError::ConvergenceFailure(msg) => {
                write!(f, "Convergence failure: {}", msg)
            }
            ImputationError::MatrixError(msg) => {
                write!(f, "Matrix error: {}", msg)
            }
            ImputationError::DimensionMismatch(msg) => {
                write!(f, "Dimension mismatch: {}", msg)
            }
            ImputationError::NumericalError(msg) => {
                write!(f, "Numerical error: {}", msg)
            }
            ImputationError::ValidationError(msg) => {
                write!(f, "Validation error: {}", msg)
            }
            ImputationError::IOError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            ImputationError::MemoryError(msg) => {
                write!(f, "Memory error: {}", msg)
            }
            ImputationError::NotImplemented(msg) => {
                write!(f, "Not implemented: {}", msg)
            }
            ImputationError::ProcessingError(msg) => {
                write!(f, "Processing error: {}", msg)
            }
            ImputationError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
        }
    }
}

impl std::error::Error for ImputationError {}

impl From<std::io::Error> for ImputationError {
    fn from(err: std::io::Error) -> Self {
        ImputationError::IOError(err.to_string())
    }
}

impl From<sklears_core::error::SklearsError> for ImputationError {
    fn from(err: sklears_core::error::SklearsError) -> Self {
        ImputationError::ProcessingError(err.to_string())
    }
}

impl From<ImputationError> for sklears_core::error::SklearsError {
    fn from(err: ImputationError) -> Self {
        match err {
            ImputationError::InvalidParameter(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::InsufficientData(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::ConvergenceFailure(msg) => {
                sklears_core::error::SklearsError::FitError(msg)
            }
            ImputationError::MatrixError(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::DimensionMismatch(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::NumericalError(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::ValidationError(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::IOError(msg) => sklears_core::error::SklearsError::InvalidInput(msg),
            ImputationError::MemoryError(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::NotImplemented(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::ProcessingError(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
            ImputationError::InvalidConfiguration(msg) => {
                sklears_core::error::SklearsError::InvalidInput(msg)
            }
        }
    }
}

/// Core trait for imputation methods
pub trait Imputer {
    /// Fit the imputer to the data and return the imputed data
    fn fit_transform(
        &self,
        X: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<scirs2_core::ndarray::Array2<f64>>;
}

/// Trait for imputers that can be trained separately
pub trait TrainableImputer {
    /// The trained state type
    type Trained;

    /// Fit the imputer to training data
    fn fit(&self, X: &scirs2_core::ndarray::ArrayView2<f64>) -> ImputationResult<Self::Trained>;
}

/// Trait for trained imputers that can transform data
pub trait TransformableImputer {
    /// Transform data using the trained imputer
    fn transform(
        &self,
        X: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<scirs2_core::ndarray::Array2<f64>>;
}

/// Configuration trait for imputation methods
pub trait ImputerConfig {
    /// Validate the configuration
    fn validate(&self) -> ImputationResult<()>;

    /// Get default configuration
    fn default_config() -> Self;
}

/// Trait for imputation quality assessment
pub trait QualityAssessment {
    /// Assess the quality of imputation
    fn assess_quality(
        &self,
        original: &scirs2_core::ndarray::ArrayView2<f64>,
        imputed: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<f64>;
}

/// Trait for handling missing value patterns
pub trait MissingPatternHandler {
    /// Analyze missing value patterns
    fn analyze_patterns(
        &self,
        X: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<std::collections::HashMap<String, f64>>;

    /// Identify missing value mechanism (MCAR, MAR, MNAR)
    fn identify_mechanism(
        &self,
        X: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<String>;
}

/// Trait for statistical validation of imputations
pub trait StatisticalValidator {
    /// Validate distributional properties
    fn validate_distribution(
        &self,
        original: &scirs2_core::ndarray::ArrayView2<f64>,
        imputed: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<bool>;

    /// Test for bias in imputation
    fn test_bias(
        &self,
        original: &scirs2_core::ndarray::ArrayView2<f64>,
        imputed: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<f64>;
}

/// Metadata about the imputation process
#[derive(Debug, Clone)]
pub struct ImputationMetadata {
    /// Method used for imputation
    pub method: String,
    /// Parameters used
    pub parameters: std::collections::HashMap<String, String>,
    /// Number of values imputed
    pub n_imputed: usize,
    /// Convergence information (if applicable)
    pub convergence_info: Option<ConvergenceInfo>,
    /// Quality metrics
    pub quality_metrics: Option<std::collections::HashMap<String, f64>>,
    /// Processing time in milliseconds
    pub processing_time_ms: Option<u64>,
}

/// Information about convergence for iterative methods
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Final convergence criterion value
    pub final_criterion: f64,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Convergence history
    pub history: Vec<f64>,
}

impl ImputationMetadata {
    /// Create new metadata
    pub fn new(method: String) -> Self {
        Self {
            method,
            parameters: std::collections::HashMap::new(),
            n_imputed: 0,
            convergence_info: None,
            quality_metrics: None,
            processing_time_ms: None,
        }
    }

    /// Add parameter information
    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Set number of imputed values
    pub fn with_n_imputed(mut self, n_imputed: usize) -> Self {
        self.n_imputed = n_imputed;
        self
    }

    /// Set convergence information
    pub fn with_convergence(mut self, convergence: ConvergenceInfo) -> Self {
        self.convergence_info = Some(convergence);
        self
    }

    /// Set quality metrics
    pub fn with_quality_metrics(mut self, metrics: std::collections::HashMap<String, f64>) -> Self {
        self.quality_metrics = Some(metrics);
        self
    }

    /// Set processing time
    pub fn with_processing_time(mut self, time_ms: u64) -> Self {
        self.processing_time_ms = Some(time_ms);
        self
    }
}

/// Result of an imputation operation with metadata
#[derive(Debug, Clone)]
pub struct ImputationOutputWithMetadata {
    /// The imputed data
    pub data: scirs2_core::ndarray::Array2<f64>,
    /// Metadata about the imputation process
    pub metadata: ImputationMetadata,
}

impl ImputationOutputWithMetadata {
    /// Create new output with metadata
    pub fn new(data: scirs2_core::ndarray::Array2<f64>, metadata: ImputationMetadata) -> Self {
        Self { data, metadata }
    }
}

/// Utility functions for common operations
pub mod utils {
    use super::*;

    /// Count missing values in an array
    pub fn count_missing(X: &scirs2_core::ndarray::ArrayView2<f64>) -> usize {
        X.iter().filter(|&&x| x.is_nan()).count()
    }

    /// Get missing value positions
    pub fn get_missing_positions(X: &scirs2_core::ndarray::ArrayView2<f64>) -> Vec<(usize, usize)> {
        X.indexed_iter()
            .filter_map(|((i, j), &val)| if val.is_nan() { Some((i, j)) } else { None })
            .collect()
    }

    /// Compute missing value rate per feature
    pub fn missing_rates_per_feature(X: &scirs2_core::ndarray::ArrayView2<f64>) -> Vec<f64> {
        let (n_rows, n_cols) = X.dim();
        let mut rates = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            let missing_count = X.column(j).iter().filter(|&&x| x.is_nan()).count();
            rates.push(missing_count as f64 / n_rows as f64);
        }

        rates
    }

    /// Compute missing value rate per sample
    pub fn missing_rates_per_sample(X: &scirs2_core::ndarray::ArrayView2<f64>) -> Vec<f64> {
        let (n_rows, n_cols) = X.dim();
        let mut rates = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            let missing_count = X.row(i).iter().filter(|&&x| x.is_nan()).count();
            rates.push(missing_count as f64 / n_cols as f64);
        }

        rates
    }

    /// Validate input data for imputation
    pub fn validate_input(X: &scirs2_core::ndarray::ArrayView2<f64>) -> ImputationResult<()> {
        let (n_rows, n_cols) = X.dim();

        if n_rows == 0 {
            return Err(ImputationError::ValidationError(
                "Input array has zero rows".to_string(),
            ));
        }

        if n_cols == 0 {
            return Err(ImputationError::ValidationError(
                "Input array has zero columns".to_string(),
            ));
        }

        // Check if all values are missing
        let all_missing = X.iter().all(|&x| x.is_nan());
        if all_missing {
            return Err(ImputationError::InsufficientData(
                "All values in the input array are missing".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if arrays have compatible dimensions
    pub fn check_dimensions_compatible(
        X1: &scirs2_core::ndarray::ArrayView2<f64>,
        X2: &scirs2_core::ndarray::ArrayView2<f64>,
    ) -> ImputationResult<()> {
        if X1.dim() != X2.dim() {
            return Err(ImputationError::DimensionMismatch(format!(
                "Array dimensions don't match: {:?} vs {:?}",
                X1.dim(),
                X2.dim()
            )));
        }
        Ok(())
    }
}
