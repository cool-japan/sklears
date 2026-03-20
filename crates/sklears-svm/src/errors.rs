//! Comprehensive Error Types for SVM Operations
//!
//! This module provides a unified error handling system for all SVM operations,
//! including training, prediction, kernel computation, and various specialized algorithms.
//! All errors provide detailed context, suggestions for resolution, and appropriate
//! error codes for programmatic handling.

use thiserror::Error;

/// Main SVM Error type that encompasses all possible SVM operation failures
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SVMError {
    // === Data and Input Validation Errors ===
    /// Invalid input data dimensions or format
    #[error("Invalid input data: {message}")]
    InvalidInput {
        message: String,
        suggestion: Option<String>,
    },

    /// Dimension mismatch between arrays
    #[error("Dimension mismatch in {context}: expected {expected_str}, got {actual_str}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: String,
        expected_str: String,
        actual_str: String,
    },

    /// Empty dataset provided
    #[error("Empty dataset: cannot train on zero samples")]
    EmptyDataset { context: String },

    /// Invalid labels (e.g., non-finite, wrong format)
    #[error("Invalid labels: {reason}")]
    InvalidLabels { reason: String, suggestion: String },

    /// Insufficient data for training
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData {
        required: usize,
        actual: usize,
        context: String,
    },

    // === Hyperparameter and Configuration Errors ===
    /// Invalid hyperparameters
    #[error("Invalid hyperparameter '{parameter}': {reason}")]
    InvalidHyperparameter {
        parameter: String,
        value: String,
        reason: String,
        valid_range: Option<String>,
    },

    /// Invalid kernel configuration
    #[error("Invalid kernel configuration: {message}")]
    InvalidKernel {
        kernel_type: String,
        message: String,
        suggestion: String,
    },

    /// Invalid solver configuration
    #[error("Invalid solver configuration for {solver}: {reason}")]
    InvalidSolver {
        solver: String,
        reason: String,
        compatible_solvers: Vec<String>,
    },

    // === Training and Optimization Errors ===
    /// Training failed to converge
    #[error("Training failed to converge after {iterations} iterations")]
    ConvergenceFailure {
        iterations: usize,
        final_objective: Option<f64>,
        tolerance: f64,
        suggestions: Vec<String>,
    },

    /// Numerical instability during training
    #[error("Numerical instability detected: {issue}")]
    NumericalInstability {
        issue: String,
        context: String,
        suggestions: Vec<String>,
    },

    /// Optimization algorithm failure
    #[error("Optimization algorithm '{algorithm}' failed: {reason}")]
    OptimizationFailure {
        algorithm: String,
        reason: String,
        iteration: Option<usize>,
        objective_value: Option<f64>,
    },

    /// Infeasible optimization problem
    #[error("Optimization problem is infeasible: {reason}")]
    InfeasibleProblem {
        reason: String,
        suggestions: Vec<String>,
    },

    // === Kernel Computation Errors ===
    /// Kernel matrix computation failed
    #[error("Kernel matrix computation failed: {reason}")]
    KernelComputationError {
        kernel_type: String,
        reason: String,
        sample_indices: Option<Vec<usize>>,
    },

    /// Kernel matrix is not positive semidefinite
    #[error("Kernel matrix is not positive semidefinite")]
    NonPositiveSemidefiniteKernel {
        eigenvalue_info: Option<String>,
        suggestions: Vec<String>,
    },

    /// Kernel cache errors
    #[error("Kernel cache error: {operation} failed")]
    KernelCacheError {
        operation: String,
        reason: String,
        memory_usage: Option<usize>,
    },

    // === Memory and Resource Errors ===
    /// Out of memory during computation
    #[error("Out of memory during {operation}")]
    OutOfMemory {
        operation: String,
        requested_bytes: Option<usize>,
        available_bytes: Option<usize>,
        suggestions: Vec<String>,
    },

    /// Insufficient computational resources
    #[error("Insufficient computational resources: {resource}")]
    InsufficientResources {
        resource: String,
        required: String,
        available: String,
    },

    /// Memory allocation failed
    #[error("Memory allocation failed for {purpose}: {reason}")]
    AllocationError {
        purpose: String,
        reason: String,
        size_bytes: Option<usize>,
    },

    // === Prediction and Model State Errors ===
    /// Model not trained
    #[error("Model not trained: call fit() before predict()")]
    ModelNotTrained {
        operation: String,
        suggestions: Vec<String>,
    },

    /// Prediction failed
    #[error("Prediction failed: {reason}")]
    PredictionError {
        reason: String,
        sample_count: Option<usize>,
        context: String,
    },

    /// Model state inconsistency
    #[error("Model state inconsistency detected: {issue}")]
    ModelStateInconsistency { issue: String, context: String },

    // === Parallel Processing Errors ===
    /// Parallel processing error
    #[error("Parallel processing error in {operation}: {reason}")]
    ParallelProcessingError {
        operation: String,
        reason: String,
        thread_count: Option<usize>,
    },

    /// Thread synchronization error
    #[error("Thread synchronization error: {details}")]
    SynchronizationError { details: String, operation: String },

    // === GPU and Hardware Acceleration Errors ===
    /// GPU computation error
    #[error("GPU computation error: {reason}")]
    GpuError {
        reason: String,
        device_info: Option<String>,
        fallback_available: bool,
    },

    /// SIMD operation error
    #[error("SIMD operation error: {operation} failed")]
    SimdError {
        operation: String,
        reason: String,
        fallback_used: bool,
    },

    // === I/O and Serialization Errors ===
    /// Model serialization/deserialization error
    #[error("Model serialization error: {operation}")]
    SerializationError {
        operation: String,
        reason: String,
        format: String,
    },

    /// File I/O error during model operations
    #[error("File I/O error: {operation}")]
    IoError {
        operation: String,
        path: Option<String>,
        reason: String,
    },

    // === Cross-validation and Model Selection Errors ===
    /// Cross-validation error
    #[error("Cross-validation error: {reason}")]
    CrossValidationError {
        fold: Option<usize>,
        reason: String,
        total_folds: Option<usize>,
    },

    /// Hyperparameter optimization error
    #[error("Hyperparameter optimization failed: {method}")]
    HyperparameterOptimizationError {
        method: String,
        reason: String,
        iteration: Option<usize>,
        best_score: Option<f64>,
    },

    // === Multi-class and Specialized SVM Errors ===
    /// Multi-class strategy error
    #[error("Multi-class strategy '{strategy}' error: {reason}")]
    MultiClassError {
        strategy: String,
        reason: String,
        class_count: Option<usize>,
    },

    /// Multi-label SVM error
    #[error("Multi-label SVM error: {reason}")]
    MultiLabelError {
        reason: String,
        label_indices: Option<Vec<usize>>,
    },

    /// Structured SVM error
    #[error("Structured SVM error: {reason}")]
    StructuredSVMError {
        reason: String,
        sequence_info: Option<String>,
    },

    // === External Integration Errors ===
    /// Topic modeling integration error
    #[error("Topic modeling error: {reason}")]
    TopicModelingError {
        model_type: String,
        reason: String,
        topic_count: Option<usize>,
    },

    /// Text processing error
    #[error("Text processing error: {operation}")]
    TextProcessingError {
        operation: String,
        reason: String,
        document_index: Option<usize>,
    },

    /// Computer vision kernel error
    #[error("Computer vision kernel error: {kernel_type}")]
    ComputerVisionError {
        kernel_type: String,
        reason: String,
        image_dimensions: Option<(usize, usize)>,
    },

    // === Generic and Unknown Errors ===
    /// Internal error (should not normally occur)
    #[error("Internal error: {message}")]
    InternalError {
        message: String,
        location: String,
        debug_info: Option<String>,
    },

    /// Unknown or unexpected error
    #[error("Unknown error occurred: {message}")]
    Unknown { message: String, context: String },
}

/// Result type for SVM operations
pub type SVMResult<T> = Result<T, SVMError>;

/// Error severity levels for logging and handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue or has fallback
    Low,
    /// Medium severity - operation failed but recovery possible
    Medium,
    /// High severity - operation failed, no recovery possible
    High,
    /// Critical severity - system state may be compromised
    Critical,
}

/// Error context for providing additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub timestamp: std::time::SystemTime,
    pub severity: ErrorSeverity,
    pub metadata: std::collections::HashMap<String, String>,
}

impl SVMError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Data validation errors are usually medium severity
            SVMError::InvalidInput { .. } => ErrorSeverity::Medium,
            SVMError::DimensionMismatch { .. } => ErrorSeverity::Medium,
            SVMError::EmptyDataset { .. } => ErrorSeverity::High,
            SVMError::InvalidLabels { .. } => ErrorSeverity::Medium,
            SVMError::InsufficientData { .. } => ErrorSeverity::High,

            // Configuration errors are medium severity
            SVMError::InvalidHyperparameter { .. } => ErrorSeverity::Medium,
            SVMError::InvalidKernel { .. } => ErrorSeverity::Medium,
            SVMError::InvalidSolver { .. } => ErrorSeverity::Medium,

            // Training errors vary in severity
            SVMError::ConvergenceFailure { .. } => ErrorSeverity::Medium,
            SVMError::NumericalInstability { .. } => ErrorSeverity::High,
            SVMError::OptimizationFailure { .. } => ErrorSeverity::High,
            SVMError::InfeasibleProblem { .. } => ErrorSeverity::High,

            // Kernel errors are usually high severity
            SVMError::KernelComputationError { .. } => ErrorSeverity::High,
            SVMError::NonPositiveSemidefiniteKernel { .. } => ErrorSeverity::High,
            SVMError::KernelCacheError { .. } => ErrorSeverity::Medium,

            // Memory errors are critical
            SVMError::OutOfMemory { .. } => ErrorSeverity::Critical,
            SVMError::InsufficientResources { .. } => ErrorSeverity::Critical,
            SVMError::AllocationError { .. } => ErrorSeverity::Critical,

            // Prediction errors vary
            SVMError::ModelNotTrained { .. } => ErrorSeverity::High,
            SVMError::PredictionError { .. } => ErrorSeverity::Medium,
            SVMError::ModelStateInconsistency { .. } => ErrorSeverity::Critical,

            // Parallel processing errors are medium
            SVMError::ParallelProcessingError { .. } => ErrorSeverity::Medium,
            SVMError::SynchronizationError { .. } => ErrorSeverity::High,

            // Hardware errors have fallbacks so are low-medium
            SVMError::GpuError {
                fallback_available: true,
                ..
            } => ErrorSeverity::Low,
            SVMError::GpuError {
                fallback_available: false,
                ..
            } => ErrorSeverity::High,
            SVMError::SimdError {
                fallback_used: true,
                ..
            } => ErrorSeverity::Low,
            SVMError::SimdError {
                fallback_used: false,
                ..
            } => ErrorSeverity::Medium,

            // I/O errors are medium-high
            SVMError::SerializationError { .. } => ErrorSeverity::Medium,
            SVMError::IoError { .. } => ErrorSeverity::Medium,

            // Model selection errors are medium
            SVMError::CrossValidationError { .. } => ErrorSeverity::Medium,
            SVMError::HyperparameterOptimizationError { .. } => ErrorSeverity::Medium,

            // Specialized SVM errors are medium-high
            SVMError::MultiClassError { .. } => ErrorSeverity::Medium,
            SVMError::MultiLabelError { .. } => ErrorSeverity::Medium,
            SVMError::StructuredSVMError { .. } => ErrorSeverity::Medium,

            // Integration errors are low-medium
            SVMError::TopicModelingError { .. } => ErrorSeverity::Medium,
            SVMError::TextProcessingError { .. } => ErrorSeverity::Medium,
            SVMError::ComputerVisionError { .. } => ErrorSeverity::Medium,

            // Internal and unknown errors are critical
            SVMError::InternalError { .. } => ErrorSeverity::Critical,
            SVMError::Unknown { .. } => ErrorSeverity::Critical,
        }
    }

    /// Get error code for programmatic handling
    pub fn error_code(&self) -> u32 {
        match self {
            SVMError::InvalidInput { .. } => 1001,
            SVMError::DimensionMismatch { .. } => 1002,
            SVMError::EmptyDataset { .. } => 1003,
            SVMError::InvalidLabels { .. } => 1004,
            SVMError::InsufficientData { .. } => 1005,

            SVMError::InvalidHyperparameter { .. } => 2001,
            SVMError::InvalidKernel { .. } => 2002,
            SVMError::InvalidSolver { .. } => 2003,

            SVMError::ConvergenceFailure { .. } => 3001,
            SVMError::NumericalInstability { .. } => 3002,
            SVMError::OptimizationFailure { .. } => 3003,
            SVMError::InfeasibleProblem { .. } => 3004,

            SVMError::KernelComputationError { .. } => 4001,
            SVMError::NonPositiveSemidefiniteKernel { .. } => 4002,
            SVMError::KernelCacheError { .. } => 4003,

            SVMError::OutOfMemory { .. } => 5001,
            SVMError::InsufficientResources { .. } => 5002,
            SVMError::AllocationError { .. } => 5003,

            SVMError::ModelNotTrained { .. } => 6001,
            SVMError::PredictionError { .. } => 6002,
            SVMError::ModelStateInconsistency { .. } => 6003,

            SVMError::ParallelProcessingError { .. } => 7001,
            SVMError::SynchronizationError { .. } => 7002,

            SVMError::GpuError { .. } => 8001,
            SVMError::SimdError { .. } => 8002,

            SVMError::SerializationError { .. } => 9001,
            SVMError::IoError { .. } => 9002,

            SVMError::CrossValidationError { .. } => 10001,
            SVMError::HyperparameterOptimizationError { .. } => 10002,

            SVMError::MultiClassError { .. } => 11001,
            SVMError::MultiLabelError { .. } => 11002,
            SVMError::StructuredSVMError { .. } => 11003,

            SVMError::TopicModelingError { .. } => 12001,
            SVMError::TextProcessingError { .. } => 12002,
            SVMError::ComputerVisionError { .. } => 12003,

            SVMError::InternalError { .. } => 99001,
            SVMError::Unknown { .. } => 99999,
        }
    }

    /// Get suggestions for resolving this error
    pub fn suggestions(&self) -> Vec<String> {
        match self {
            SVMError::ConvergenceFailure { suggestions, .. } => suggestions.clone(),
            SVMError::NumericalInstability { suggestions, .. } => suggestions.clone(),
            SVMError::InfeasibleProblem { suggestions, .. } => suggestions.clone(),
            SVMError::NonPositiveSemidefiniteKernel { suggestions, .. } => suggestions.clone(),
            SVMError::OutOfMemory { suggestions, .. } => suggestions.clone(),
            SVMError::ModelNotTrained { suggestions, .. } => suggestions.clone(),

            SVMError::InvalidInput {
                suggestion: Some(s),
                ..
            } => vec![s.clone()],
            SVMError::InvalidKernel { suggestion, .. } => vec![suggestion.clone()],
            SVMError::InvalidLabels { suggestion, .. } => vec![suggestion.clone()],

            // Default suggestions for common errors
            SVMError::DimensionMismatch { .. } => vec![
                "Check that input arrays have compatible dimensions".to_string(),
                "Ensure training and test data have same number of features".to_string(),
            ],

            SVMError::EmptyDataset { .. } => vec![
                "Provide at least one training sample".to_string(),
                "Check data loading and preprocessing steps".to_string(),
            ],

            SVMError::InvalidHyperparameter {
                valid_range: Some(range),
                ..
            } => vec![
                format!("Use values in range: {}", range),
                "Check hyperparameter documentation for valid ranges".to_string(),
            ],

            _ => vec!["Check documentation for this error type".to_string()],
        }
    }

    /// Create a detailed error report
    pub fn detailed_report(&self) -> String {
        let mut report = format!("SVM Error [Code: {}]\n", self.error_code());
        report.push_str(&format!("Severity: {:?}\n", self.severity()));
        report.push_str(&format!("Message: {}\n", self));

        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            report.push_str("\nSuggestions:\n");
            for (i, suggestion) in suggestions.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, suggestion));
            }
        }

        report
    }
}

/// Convenience functions for creating common errors
impl SVMError {
    /// Create an invalid input error with suggestions
    pub fn invalid_input(message: impl Into<String>) -> Self {
        SVMError::InvalidInput {
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create an invalid input error with a suggestion
    pub fn invalid_input_with_suggestion(
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        SVMError::InvalidInput {
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: impl Into<String>,
    ) -> Self {
        let expected_str = format!("{:?}", expected);
        let actual_str = format!("{:?}", actual);
        SVMError::DimensionMismatch {
            expected,
            actual,
            context: context.into(),
            expected_str,
            actual_str,
        }
    }

    /// Create a convergence failure error
    pub fn convergence_failure(
        iterations: usize,
        tolerance: f64,
        suggestions: Vec<String>,
    ) -> Self {
        SVMError::ConvergenceFailure {
            iterations,
            final_objective: None,
            tolerance,
            suggestions,
        }
    }

    /// Create a model not trained error
    pub fn model_not_trained(operation: impl Into<String>) -> Self {
        SVMError::ModelNotTrained {
            operation: operation.into(),
            suggestions: vec![
                "Call fit() method to train the model first".to_string(),
                "Ensure training completed successfully".to_string(),
            ],
        }
    }

    /// Create an out of memory error with suggestions
    pub fn out_of_memory(operation: impl Into<String>) -> Self {
        SVMError::OutOfMemory {
            operation: operation.into(),
            requested_bytes: None,
            available_bytes: None,
            suggestions: vec![
                "Reduce dataset size or use chunked processing".to_string(),
                "Increase system memory or use out-of-core algorithms".to_string(),
                "Consider using LinearSVC for large datasets".to_string(),
            ],
        }
    }
}

/// Integration with other error types
impl From<std::io::Error> for SVMError {
    fn from(err: std::io::Error) -> Self {
        SVMError::IoError {
            operation: "file operation".to_string(),
            path: None,
            reason: err.to_string(),
        }
    }
}

/// Conversion functions for backward compatibility with existing error types
impl From<crate::gpu_kernels::GpuKernelError> for SVMError {
    fn from(err: crate::gpu_kernels::GpuKernelError) -> Self {
        use crate::gpu_kernels::GpuKernelError::*;
        match err {
            DeviceNotAvailable => SVMError::GpuError {
                reason: "GPU device not available".to_string(),
                device_info: None,
                fallback_available: true,
            },
            InsufficientMemory => SVMError::GpuError {
                reason: "Insufficient GPU memory".to_string(),
                device_info: None,
                fallback_available: true,
            },
            ComputationFailed(msg) => SVMError::GpuError {
                reason: format!("GPU computation failed: {}", msg),
                device_info: None,
                fallback_available: true,
            },
            ShaderCompilationFailed(msg) => SVMError::GpuError {
                reason: format!("Shader compilation failed: {}", msg),
                device_info: None,
                fallback_available: false,
            },
            BufferCreationFailed => SVMError::GpuError {
                reason: "Buffer creation failed".to_string(),
                device_info: None,
                fallback_available: true,
            },
            FeatureNotSupported(feature) => SVMError::GpuError {
                reason: format!("Feature not supported: {}", feature),
                device_info: None,
                fallback_available: true,
            },
            DimensionMismatch => SVMError::DimensionMismatch {
                expected: vec![],
                actual: vec![],
                context: "GPU kernel computation".to_string(),
                expected_str: "[]".to_string(),
                actual_str: "[]".to_string(),
            },
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = SVMError::invalid_input("test message");
        assert_eq!(error.error_code(), 1001);
        assert_eq!(error.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_error_suggestions() {
        let error = SVMError::model_not_trained("predict");
        let suggestions = error.suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("fit()"));
    }

    #[test]
    fn test_detailed_report() {
        let error = SVMError::convergence_failure(
            1000,
            1e-3,
            vec![
                "Increase max_iter".to_string(),
                "Decrease tolerance".to_string(),
            ],
        );

        let report = error.detailed_report();
        assert!(report.contains("Code: 3001"));
        assert!(report.contains("Severity:"));
        assert!(report.contains("Suggestions:"));
        assert!(report.contains("Increase max_iter"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let error = SVMError::dimension_mismatch(vec![100, 50], vec![100, 40], "training data");

        assert_eq!(error.error_code(), 1002);
        let suggestions = error.suggestions();
        assert!(suggestions
            .iter()
            .any(|s| s.contains("compatible dimensions")));
    }
}
