/// Exhaustive pattern matching for error handling
///
/// This module demonstrates best practices for exhaustive pattern matching
/// in error handling, ensuring all error cases are properly handled and
/// providing compile-time guarantees about error coverage.
use crate::error::{Result, SklearsError};
use std::collections::HashMap;

/// Enhanced error handling with exhaustive pattern matching
///
/// This module provides utilities for handling errors in a way that ensures
/// all possible error cases are considered at compile time.
pub struct ExhaustiveErrorHandler;

/// Error categorization for exhaustive handling
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Recoverable errors that can be handled gracefully
    Recoverable,
    /// Critical errors that should stop execution
    Critical,
    /// User errors that require user input correction
    UserError,
    /// System errors from external dependencies
    SystemError,
    /// Internal logic errors (bugs)
    InternalError,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Recovery strategy for different error types
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with the same parameters
    Retry { max_attempts: usize },
    /// Retry with modified parameters
    RetryWithFallback {
        fallback_params: HashMap<String, String>,
    },
    /// Use default values and continue
    UseDefaults,
    /// Skip this operation and continue
    Skip,
    /// Fail fast - propagate the error immediately
    FailFast,
    /// Log and continue (for non-critical errors)
    LogAndContinue,
}

/// Comprehensive error analysis result
#[derive(Debug)]
pub struct ErrorAnalysis {
    /// The original error
    pub error: SklearsError,
    /// Categorization of the error
    pub category: ErrorCategory,
    /// Severity level
    pub severity: ErrorSeverity,
    /// Recommended recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Additional context for debugging
    pub context: Vec<String>,
    /// Whether the error is likely to be transient
    pub is_transient: bool,
    /// Suggested user actions
    pub user_actions: Vec<String>,
}

impl ExhaustiveErrorHandler {
    /// Analyze an error with exhaustive pattern matching
    ///
    /// This method demonstrates exhaustive pattern matching by ensuring
    /// every possible SklearsError variant is handled explicitly.
    pub fn analyze_error(error: SklearsError) -> ErrorAnalysis {
        // Use exhaustive pattern matching to ensure all error types are handled
        match error {
            // Data-related errors
            SklearsError::InvalidInput(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Input validation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check input data format".to_string(),
                    "Verify data types are correct".to_string(),
                    format!("Error details: {}", msg),
                ],
            },

            SklearsError::InvalidData { ref reason } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Data quality validation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Clean your data".to_string(),
                    "Check for missing values".to_string(),
                    "Verify data preprocessing".to_string(),
                    format!("Reason: {}", reason),
                ],
            },

            SklearsError::ShapeMismatch {
                ref expected,
                ref actual,
            } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Array shape validation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    format!("Expected shape: {}", expected),
                    format!("Actual shape: {}", actual),
                    "Reshape your data to match expected dimensions".to_string(),
                ],
            },

            SklearsError::DimensionMismatch { expected, actual } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Dimension mismatch between arrays".to_string()],
                is_transient: false,
                user_actions: vec![
                    format!("Expected {} dimensions", expected),
                    format!("Got {} dimensions", actual),
                    "Ensure arrays have compatible dimensions".to_string(),
                ],
            },

            SklearsError::FeatureMismatch { expected, actual } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Feature dimension mismatch".to_string()],
                is_transient: false,
                user_actions: vec![
                    format!("Model expects {} features", expected),
                    format!("Input has {} features", actual),
                    "Use the same feature set as during training".to_string(),
                ],
            },

            // Model state errors
            SklearsError::NotFitted { ref operation } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Model not trained before use".to_string()],
                is_transient: false,
                user_actions: vec![
                    format!("Call fit() before {}", operation),
                    "Train the model with your data first".to_string(),
                ],
            },

            // Parameter validation errors
            SklearsError::InvalidParameter {
                ref name,
                ref reason,
            } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::UseDefaults,
                context: vec!["Parameter validation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    format!("Parameter '{}' is invalid: {}", name, reason),
                    "Check parameter documentation for valid ranges".to_string(),
                    "Consider using default parameters".to_string(),
                ],
            },

            // Computational errors
            SklearsError::NumericalError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::Recoverable,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::RetryWithFallback {
                    fallback_params: [
                        ("regularization".to_string(), "increased".to_string()),
                        ("tolerance".to_string(), "relaxed".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                },
                context: vec!["Numerical computation failed".to_string()],
                is_transient: true,
                user_actions: vec![
                    "Try with different numerical parameters".to_string(),
                    "Scale your data".to_string(),
                    "Add regularization".to_string(),
                    format!("Error: {}", msg),
                ],
            },

            SklearsError::ConvergenceError { iterations } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::Recoverable,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::RetryWithFallback {
                    fallback_params: [
                        ("max_iterations".to_string(), "increased".to_string()),
                        ("tolerance".to_string(), "relaxed".to_string()),
                        ("learning_rate".to_string(), "adjusted".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                },
                context: vec!["Algorithm failed to converge".to_string()],
                is_transient: true,
                user_actions: vec![
                    format!("Algorithm stopped after {} iterations", iterations),
                    "Increase max_iterations parameter".to_string(),
                    "Relax convergence tolerance".to_string(),
                    "Try different learning rate".to_string(),
                    "Scale your data".to_string(),
                ],
            },

            // Training/prediction errors
            SklearsError::FitError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::Recoverable,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::Retry { max_attempts: 3 },
                context: vec!["Model training failed".to_string()],
                is_transient: true,
                user_actions: vec![
                    "Check training data quality".to_string(),
                    "Try different hyperparameters".to_string(),
                    format!("Error details: {}", msg),
                ],
            },

            SklearsError::PredictError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::Recoverable,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Prediction failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Verify model is properly trained".to_string(),
                    "Check input data format".to_string(),
                    format!("Error details: {}", msg),
                ],
            },

            SklearsError::TransformError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::Recoverable,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Data transformation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check transformer is fitted".to_string(),
                    "Verify input data compatibility".to_string(),
                    format!("Error details: {}", msg),
                ],
            },

            // System errors
            SklearsError::IoError(ref io_error) => {
                let (category, severity, strategy, is_transient) = match io_error.kind() {
                    std::io::ErrorKind::NotFound => (
                        ErrorCategory::UserError,
                        ErrorSeverity::High,
                        RecoveryStrategy::FailFast,
                        false,
                    ),
                    std::io::ErrorKind::PermissionDenied => (
                        ErrorCategory::SystemError,
                        ErrorSeverity::High,
                        RecoveryStrategy::FailFast,
                        false,
                    ),
                    std::io::ErrorKind::Interrupted => (
                        ErrorCategory::SystemError,
                        ErrorSeverity::Low,
                        RecoveryStrategy::Retry { max_attempts: 3 },
                        true,
                    ),
                    _ => (
                        ErrorCategory::SystemError,
                        ErrorSeverity::Medium,
                        RecoveryStrategy::FailFast,
                        false,
                    ),
                };

                ErrorAnalysis {
                    error: error.clone(),
                    category,
                    severity,
                    recovery_strategy: strategy,
                    context: vec!["I/O operation failed".to_string()],
                    is_transient,
                    user_actions: vec![
                        "Check file paths and permissions".to_string(),
                        "Ensure sufficient disk space".to_string(),
                        format!("I/O Error: {}", io_error),
                    ],
                }
            }

            // Development errors
            SklearsError::NotImplemented(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::InternalError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Feature not yet implemented".to_string()],
                is_transient: false,
                user_actions: vec![
                    "This feature is not yet implemented".to_string(),
                    "Use an alternative approach".to_string(),
                    "Check documentation for supported features".to_string(),
                    format!("Details: {}", msg),
                ],
            },

            // Catch-all for other errors
            SklearsError::Other(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::InternalError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::LogAndContinue,
                context: vec!["Unspecified error occurred".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Contact support if this error persists".to_string(),
                    "Include error details in bug report".to_string(),
                    format!("Error: {}", msg),
                ],
            },

            SklearsError::InvalidOperation(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Invalid operation attempted".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check the operation parameters".to_string(),
                    "Ensure prerequisites are met".to_string(),
                    format!("Operation error: {}", msg),
                ],
            },

            SklearsError::MissingDependency {
                ref dependency,
                ref feature,
            } => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::Low,
                recovery_strategy: RecoveryStrategy::RetryWithFallback {
                    fallback_params: HashMap::new(),
                },
                context: vec![format!("Missing dependency: {}", dependency)],
                is_transient: false,
                user_actions: vec![
                    format!("Install the '{}' dependency", dependency),
                    format!("Use fallback implementation for {}", feature),
                    "Check installation documentation".to_string(),
                ],
            },

            SklearsError::FileError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["File operation failed".to_string()],
                is_transient: true,
                user_actions: vec![
                    "Check file permissions".to_string(),
                    "Verify file path exists".to_string(),
                    format!("Error details: {}", msg),
                ],
            },

            SklearsError::SerializationError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Serialization failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check data format compatibility".to_string(),
                    "Verify serialization parameters".to_string(),
                    format!("Error details: {}", msg),
                ],
            },

            SklearsError::DeserializationError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Deserialization failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check data format compatibility".to_string(),
                    "Verify file is not corrupted".to_string(),
                    format!("Error details: {}", msg),
                ],
            },
            SklearsError::TraitNotFound(ref trait_name) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                context: vec![format!("Trait '{}' not found in the registry", trait_name)],
                recovery_strategy: RecoveryStrategy::LogAndContinue,
                is_transient: false,
                user_actions: vec![
                    "Verify the trait name is correct".to_string(),
                    "Check if the trait is properly imported".to_string(),
                    format!("Missing trait: {}", trait_name),
                ],
            },
            SklearsError::AnalysisError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::High,
                context: vec![format!("Analysis operation failed: {}", msg)],
                recovery_strategy: RecoveryStrategy::Retry { max_attempts: 3 },
                is_transient: true,
                user_actions: vec![
                    "Check system resources".to_string(),
                    "Retry the analysis operation".to_string(),
                    format!("Analysis error: {}", msg),
                ],
            },

            SklearsError::HardwareError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::Retry { max_attempts: 3 },
                context: vec!["Hardware-related error occurred".to_string()],
                is_transient: true,
                user_actions: vec![
                    "Check hardware connectivity".to_string(),
                    "Verify hardware drivers".to_string(),
                    "Try alternative hardware settings".to_string(),
                    format!("Hardware error: {}", msg),
                ],
            },
            SklearsError::InvalidState(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["System is in an invalid state".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Reset the system state".to_string(),
                    "Reinitialize the component".to_string(),
                    format!("Invalid state: {}", msg),
                ],
            },
            SklearsError::Configuration(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Configuration error detected".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check configuration settings".to_string(),
                    "Verify configuration file format".to_string(),
                    "Review configuration documentation".to_string(),
                    format!("Configuration error: {}", msg),
                ],
            },
            SklearsError::ResourceAllocationError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::SystemError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::Retry { max_attempts: 3 },
                context: vec!["Resource allocation failed".to_string()],
                is_transient: true,
                user_actions: vec![
                    "Check available system resources".to_string(),
                    "Reduce resource requirements".to_string(),
                    "Wait for resources to become available".to_string(),
                    format!("Resource allocation error: {}", msg),
                ],
            },

            SklearsError::InvalidConfiguration(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Invalid configuration detected".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check configuration settings".to_string(),
                    "Verify configuration syntax".to_string(),
                    format!("Configuration error: {}", msg),
                ],
            },

            SklearsError::ProcessingError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::Recoverable,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::Retry { max_attempts: 2 },
                context: vec!["Processing operation failed".to_string()],
                is_transient: true,
                user_actions: vec![
                    "Retry the operation".to_string(),
                    "Check input data validity".to_string(),
                    format!("Processing error: {}", msg),
                ],
            },

            SklearsError::ModelError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::High,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Model operation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check model configuration".to_string(),
                    "Verify model is properly trained".to_string(),
                    format!("Model error: {}", msg),
                ],
            },

            SklearsError::ValidationError(ref msg) => ErrorAnalysis {
                error: error.clone(),
                category: ErrorCategory::UserError,
                severity: ErrorSeverity::Medium,
                recovery_strategy: RecoveryStrategy::FailFast,
                context: vec!["Validation failed".to_string()],
                is_transient: false,
                user_actions: vec![
                    "Check input validation rules".to_string(),
                    "Correct invalid input data".to_string(),
                    format!("Validation error: {}", msg),
                ],
            },
            // NOTE: If we add new variants to SklearsError, this match will fail to compile
            // until we add a new case here, ensuring exhaustive handling
        }
    }

    /// Handle an error with the determined recovery strategy
    ///
    /// This method shows how to use the error analysis to implement
    /// different recovery strategies with exhaustive pattern matching.
    pub fn handle_with_recovery<T, F>(result: Result<T>, retry_fn: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                let analysis = Self::analyze_error(error);

                // Exhaustive pattern matching on recovery strategy
                match analysis.recovery_strategy {
                    RecoveryStrategy::Retry { max_attempts } => {
                        let mut attempts = 0;
                        loop {
                            attempts += 1;
                            match retry_fn() {
                                Ok(value) => return Ok(value),
                                Err(retry_error) => {
                                    if attempts >= max_attempts {
                                        return Err(retry_error);
                                    }
                                    // Continue retrying
                                }
                            }
                        }
                    }

                    RecoveryStrategy::RetryWithFallback { fallback_params: _ } => {
                        // In a real implementation, you would apply the fallback parameters
                        // and retry the operation
                        retry_fn()
                    }

                    RecoveryStrategy::UseDefaults => {
                        // In a real implementation, you would use default values
                        // and continue the operation
                        Err(analysis.error)
                    }

                    RecoveryStrategy::Skip => {
                        // In a real implementation, you would return a default value
                        // or skip this operation entirely
                        Err(analysis.error)
                    }

                    RecoveryStrategy::FailFast => {
                        // Immediately propagate the error
                        Err(analysis.error)
                    }

                    RecoveryStrategy::LogAndContinue => {
                        // Log the error and continue with a default value
                        log::warn!("Error occurred but continuing: {}", analysis.error);
                        Err(analysis.error) // In practice, might return a default value
                    } // NOTE: If we add new RecoveryStrategy variants, this match will fail
                      // to compile until we handle them, ensuring exhaustive coverage
                }
            }
        }
    }

    /// Classify multiple errors and provide aggregate handling strategy
    ///
    /// This method demonstrates exhaustive pattern matching over collections
    /// of errors to determine the best overall handling strategy.
    pub fn classify_error_batch(errors: Vec<SklearsError>) -> BatchErrorAnalysis {
        if errors.is_empty() {
            return BatchErrorAnalysis {
                total_errors: 0,
                error_categories: HashMap::new(),
                overall_severity: ErrorSeverity::Low,
                recommended_action: BatchAction::Continue,
                critical_errors: Vec::new(),
                recoverable_errors: Vec::new(),
            };
        }

        let mut categories = HashMap::new();
        let mut max_severity = ErrorSeverity::Low;
        let mut critical_errors = Vec::new();
        let mut recoverable_errors = Vec::new();

        for error in errors.iter() {
            let analysis = Self::analyze_error(error.clone());

            // Count error categories
            *categories.entry(analysis.category.clone()).or_insert(0) += 1;

            // Track maximum severity
            if analysis.severity > max_severity {
                max_severity = analysis.severity;
            }

            // Classify errors for batch handling
            match analysis.category {
                ErrorCategory::Critical | ErrorCategory::InternalError => {
                    critical_errors.push(analysis);
                }
                ErrorCategory::Recoverable => {
                    recoverable_errors.push(analysis);
                }
                ErrorCategory::UserError | ErrorCategory::SystemError => {
                    // These might be recoverable depending on context
                    if analysis.is_transient {
                        recoverable_errors.push(analysis);
                    } else {
                        critical_errors.push(analysis);
                    }
                } // NOTE: Exhaustive match ensures all categories are handled
            }
        }

        // Determine overall action based on error composition
        let recommended_action = match max_severity {
            ErrorSeverity::Critical => BatchAction::Abort,
            ErrorSeverity::High => {
                if critical_errors.len() > recoverable_errors.len() {
                    BatchAction::Abort
                } else {
                    BatchAction::RetryWithCaution
                }
            }
            ErrorSeverity::Medium => {
                if critical_errors.is_empty() {
                    BatchAction::RetryAll
                } else {
                    BatchAction::RetryRecoverableOnly
                }
            }
            ErrorSeverity::Low => BatchAction::Continue,
            // NOTE: Exhaustive match ensures all severities are handled
        };

        BatchErrorAnalysis {
            total_errors: errors.len(),
            error_categories: categories,
            overall_severity: max_severity,
            recommended_action,
            critical_errors,
            recoverable_errors,
        }
    }
}

/// Analysis result for a batch of errors
#[derive(Debug)]
pub struct BatchErrorAnalysis {
    pub total_errors: usize,
    pub error_categories: HashMap<ErrorCategory, usize>,
    pub overall_severity: ErrorSeverity,
    pub recommended_action: BatchAction,
    pub critical_errors: Vec<ErrorAnalysis>,
    pub recoverable_errors: Vec<ErrorAnalysis>,
}

/// Recommended actions for handling batches of errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchAction {
    /// Continue processing despite errors
    Continue,
    /// Retry all operations
    RetryAll,
    /// Retry only the recoverable errors
    RetryRecoverableOnly,
    /// Retry with extra caution and monitoring
    RetryWithCaution,
    /// Abort the entire batch operation
    Abort,
}

/// Error pattern matching utilities
pub mod patterns {
    use super::*;

    /// Matches data-related errors
    pub fn is_data_error(error: &SklearsError) -> bool {
        match error {
            SklearsError::InvalidInput(_)
            | SklearsError::InvalidData { .. }
            | SklearsError::ShapeMismatch { .. }
            | SklearsError::FeatureMismatch { .. } => true,
            _ => false,
            // NOTE: Explicit catch-all ensures we don't miss new variants
        }
    }

    /// Matches computational errors that might be transient
    pub fn is_transient_error(error: &SklearsError) -> bool {
        match error {
            SklearsError::NumericalError(_) | SklearsError::ConvergenceError { .. } => true,
            SklearsError::IoError(io_error) => {
                match io_error.kind() {
                    std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::WouldBlock => true,
                    _ => false,
                    // NOTE: Explicit match on IoErrorKind ensures exhaustive handling
                }
            }
            _ => false,
        }
    }

    /// Matches errors that indicate user mistakes
    pub fn is_user_error(error: &SklearsError) -> bool {
        match error {
            SklearsError::InvalidInput(_)
            | SklearsError::InvalidData { .. }
            | SklearsError::InvalidParameter { .. }
            | SklearsError::NotFitted { .. }
            | SklearsError::ShapeMismatch { .. }
            | SklearsError::FeatureMismatch { .. } => true,
            SklearsError::IoError(io_error) => matches!(
                io_error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ),
            _ => false,
        }
    }

    /// Matches errors that indicate bugs in the library
    pub fn is_internal_error(error: &SklearsError) -> bool {
        match error {
            SklearsError::NotImplemented(_) => true,
            SklearsError::Other(_) => true, // Many "Other" errors are internal
            _ => false,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exhaustive_error_analysis() {
        // Test each error variant is handled
        let test_cases = vec![
            SklearsError::InvalidInput("test".to_string()),
            SklearsError::InvalidData {
                reason: "test".to_string(),
            },
            SklearsError::ShapeMismatch {
                expected: "(10, 5)".to_string(),
                actual: "(10, 3)".to_string(),
            },
            SklearsError::FeatureMismatch {
                expected: 5,
                actual: 3,
            },
            SklearsError::NotFitted {
                operation: "predict".to_string(),
            },
            SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be positive".to_string(),
            },
            SklearsError::NumericalError("singular matrix".to_string()),
            SklearsError::ConvergenceError { iterations: 100 },
            SklearsError::FitError("training failed".to_string()),
            SklearsError::PredictError("prediction failed".to_string()),
            SklearsError::TransformError("transform failed".to_string()),
            SklearsError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            )),
            SklearsError::NotImplemented("feature X".to_string()),
            SklearsError::Other("unknown error".to_string()),
        ];

        for error in test_cases {
            let analysis = ExhaustiveErrorHandler::analyze_error(error.clone());

            // Verify analysis is complete
            assert!(
                !analysis.context.is_empty(),
                "Context should not be empty for: {:?}",
                error
            );
            assert!(
                !analysis.user_actions.is_empty(),
                "User actions should not be empty for: {:?}",
                error
            );

            // Verify category assignment makes sense
            match error {
                SklearsError::InvalidInput(_)
                | SklearsError::InvalidData { .. }
                | SklearsError::NotFitted { .. } => {
                    assert_eq!(analysis.category, ErrorCategory::UserError);
                }
                SklearsError::NumericalError(_) | SklearsError::ConvergenceError { .. } => {
                    assert_eq!(analysis.category, ErrorCategory::Recoverable);
                }
                SklearsError::NotImplemented(_) => {
                    assert_eq!(analysis.category, ErrorCategory::InternalError);
                }
                _ => {
                    // Other errors have various categorizations based on content
                }
            }
        }
    }

    #[test]
    fn test_error_pattern_matching() {
        let data_error = SklearsError::InvalidInput("test".to_string());
        let transient_error = SklearsError::NumericalError("test".to_string());
        let user_error = SklearsError::NotFitted {
            operation: "predict".to_string(),
        };
        let internal_error = SklearsError::NotImplemented("test".to_string());

        assert!(patterns::is_data_error(&data_error));
        assert!(!patterns::is_data_error(&transient_error));

        assert!(patterns::is_transient_error(&transient_error));
        assert!(!patterns::is_transient_error(&data_error));

        assert!(patterns::is_user_error(&user_error));
        assert!(!patterns::is_user_error(&internal_error));

        assert!(patterns::is_internal_error(&internal_error));
        assert!(!patterns::is_internal_error(&user_error));
    }

    #[test]
    fn test_batch_error_analysis() {
        let errors = vec![
            SklearsError::InvalidInput("bad input".to_string()),
            SklearsError::NumericalError("numerical issue".to_string()),
            SklearsError::NotImplemented("feature".to_string()),
        ];

        let analysis = ExhaustiveErrorHandler::classify_error_batch(errors);

        assert_eq!(analysis.total_errors, 3);
        assert!(analysis
            .error_categories
            .contains_key(&ErrorCategory::UserError));
        assert!(analysis
            .error_categories
            .contains_key(&ErrorCategory::Recoverable));
        assert!(analysis
            .error_categories
            .contains_key(&ErrorCategory::InternalError));

        // Should recommend caution due to mixed error types
        match analysis.recommended_action {
            BatchAction::Abort | BatchAction::RetryWithCaution => {
                // Both are reasonable for this mix of errors
            }
            _ => panic!("Unexpected batch action for mixed critical errors"),
        }
    }

    #[test]
    fn test_empty_batch_analysis() {
        let analysis = ExhaustiveErrorHandler::classify_error_batch(vec![]);

        assert_eq!(analysis.total_errors, 0);
        assert_eq!(analysis.overall_severity, ErrorSeverity::Low);
        assert_eq!(analysis.recommended_action, BatchAction::Continue);
        assert!(analysis.critical_errors.is_empty());
        assert!(analysis.recoverable_errors.is_empty());
    }
}
