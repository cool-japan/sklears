//! Domain-specific error types for linear models
//!
//! This module provides comprehensive error types specific to linear model operations,
//! including detailed context, recovery suggestions, and error categorization.

use sklears_core::error::SklearsError;
use std::fmt;

/// Comprehensive error type for linear model operations
#[derive(Debug, Clone)]
pub enum LinearModelError {
    /// Data-related errors (input validation, preprocessing)
    DataError(DataError),
    /// Model configuration errors
    ConfigurationError(ConfigurationError),
    /// Numerical computation errors
    NumericalError(NumericalError),
    /// Optimization/convergence errors
    OptimizationError(OptimizationError),
    /// Model state errors (fitting, prediction)
    StateError(StateError),
    /// Feature-related errors
    FeatureError(FeatureError),
    /// Matrix operation errors
    MatrixError(MatrixError),
    /// Cross-validation errors
    CrossValidationError(CrossValidationError),
    /// Memory/resource errors
    ResourceError(ResourceError),
}

/// Data-related errors
#[derive(Debug, Clone)]
pub struct DataError {
    pub kind: DataErrorKind,
    pub context: String,
    pub suggestions: Vec<String>,
    pub error_location: Option<String>,
}

#[derive(Debug, Clone)]
pub enum DataErrorKind {
    /// Empty dataset
    EmptyData,
    /// Mismatched dimensions
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Invalid data values (NaN, infinity)
    InvalidValues { count: usize, total: usize },
    /// Missing target values
    MissingTargets,
    /// Insufficient data for operation
    InsufficientData { required: usize, available: usize },
    /// Data type incompatibility
    IncompatibleDataType { expected: String, actual: String },
    /// Data range issues (e.g., for Box-Cox transformation)
    DataRangeError { min_required: f64, actual_min: f64 },
}

/// Model configuration errors
#[derive(Debug, Clone)]
pub struct ConfigurationError {
    pub kind: ConfigurationErrorKind,
    pub parameter_name: String,
    pub provided_value: String,
    pub valid_range: Option<String>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ConfigurationErrorKind {
    /// Invalid parameter value
    InvalidParameter,
    /// Parameter out of valid range
    OutOfRange,
    /// Incompatible parameter combination
    IncompatibleParameters { conflicting_params: Vec<String> },
    /// Missing required parameter
    MissingParameter,
    /// Deprecated parameter usage
    DeprecatedParameter { replacement: Option<String> },
}

/// Numerical computation errors
#[derive(Debug, Clone)]
pub struct NumericalError {
    pub kind: NumericalErrorKind,
    pub operation: String,
    pub context: String,
    pub matrix_info: Option<MatrixInfo>,
    pub recovery_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum NumericalErrorKind {
    /// Matrix is singular or near-singular
    SingularMatrix { condition_number: Option<f64> },
    /// Numerical overflow
    Overflow,
    /// Numerical underflow
    Underflow,
    /// Loss of precision
    PrecisionLoss { digits_lost: usize },
    /// Ill-conditioned problem
    IllConditioned { condition_number: f64 },
    /// Failed to invert matrix
    MatrixInversionFailed,
    /// Eigenvalue computation failed
    EigenvalueFailed,
    /// Cholesky decomposition failed
    CholeskyFailed,
}

/// Matrix-related information
#[derive(Debug, Clone)]
pub struct MatrixInfo {
    pub dimensions: (usize, usize),
    pub rank: Option<usize>,
    pub condition_number: Option<f64>,
    pub determinant: Option<f64>,
    pub is_symmetric: Option<bool>,
    pub is_positive_definite: Option<bool>,
}

/// Optimization and convergence errors
#[derive(Debug, Clone)]
pub struct OptimizationError {
    pub kind: OptimizationErrorKind,
    pub algorithm: String,
    pub iteration: Option<usize>,
    pub max_iterations: Option<usize>,
    pub convergence_info: Option<ConvergenceInfo>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum OptimizationErrorKind {
    /// Failed to converge within max iterations
    ConvergenceFailed,
    /// Converged to local minimum (suspected)
    LocalMinimum,
    /// Objective function not decreasing
    NoProgress,
    /// Step size too small
    StepSizeTooSmall,
    /// Gradient computation failed
    GradientFailed,
    /// Hessian computation failed
    HessianFailed,
    /// Line search failed
    LineSearchFailed,
    /// Invalid optimization direction
    InvalidDirection,
    /// Invalid problem dimensions
    InvalidProblemDimensions,
    /// Model not fitted
    ModelNotFitted,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    pub final_objective: Option<f64>,
    pub final_gradient_norm: Option<f64>,
    pub final_step_size: Option<f64>,
    pub objective_history: Vec<f64>,
    pub gradient_norm_history: Vec<f64>,
}

/// Model state errors
#[derive(Debug, Clone)]
pub struct StateError {
    pub kind: StateErrorKind,
    pub current_state: String,
    pub required_state: String,
    pub operation: String,
}

#[derive(Debug, Clone)]
pub enum StateErrorKind {
    /// Model not fitted
    NotFitted,
    /// Model already fitted
    AlreadyFitted,
    /// Invalid state transition
    InvalidStateTransition,
    /// Operation not available in current state
    OperationNotAvailable,
}

/// Feature-related errors
#[derive(Debug, Clone)]
pub struct FeatureError {
    pub kind: FeatureErrorKind,
    pub feature_indices: Vec<usize>,
    pub context: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum FeatureErrorKind {
    /// Features with zero variance
    ZeroVariance,
    /// Highly correlated features
    Multicollinearity { correlation_threshold: f64 },
    /// Features outside expected range
    OutOfRange { min: f64, max: f64 },
    /// Missing features
    MissingFeatures,
    /// Too many features relative to samples
    CurseOfDimensionality { n_features: usize, n_samples: usize },
    /// Feature scaling issues
    ScalingError { method: String },
}

/// Matrix operation errors
#[derive(Debug, Clone)]
pub struct MatrixError {
    pub kind: MatrixErrorKind,
    pub operation: String,
    pub matrix_info: MatrixInfo,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum MatrixErrorKind {
    /// Dimension mismatch in operation
    DimensionMismatch,
    /// Matrix not square when required
    NotSquare,
    /// Matrix not symmetric when required
    NotSymmetric,
    /// Matrix not positive definite when required
    NotPositiveDefinite,
    /// Sparse matrix operation failed
    SparseOperationFailed,
    /// Memory allocation failed for matrix
    AllocationFailed { required_bytes: usize },
}

/// Cross-validation errors
#[derive(Debug, Clone)]
pub struct CrossValidationError {
    pub kind: CrossValidationErrorKind,
    pub fold_info: Option<FoldInfo>,
    pub context: String,
}

#[derive(Debug, Clone)]
pub enum CrossValidationErrorKind {
    /// Insufficient data for CV
    InsufficientData,
    /// Invalid fold configuration
    InvalidFolds,
    /// Fold contains no positive/negative samples
    ImbalancedFold,
    /// CV scoring failed
    ScoringFailed { metric: String },
    /// Early stopping criteria not met
    EarlyStoppingFailed,
}

/// Cross-validation fold information
#[derive(Debug, Clone)]
pub struct FoldInfo {
    pub current_fold: usize,
    pub total_folds: usize,
    pub train_size: usize,
    pub test_size: usize,
    pub class_distribution: Option<Vec<(String, usize)>>,
}

/// Resource-related errors
#[derive(Debug, Clone)]
pub struct ResourceError {
    pub kind: ResourceErrorKind,
    pub resource_info: ResourceInfo,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ResourceErrorKind {
    /// Insufficient memory
    InsufficientMemory,
    /// Operation would take too long
    TimeoutExceeded,
    /// File I/O error
    FileIoError { operation: String, path: String },
    /// Network/distributed computation error
    NetworkError,
}

/// Resource information
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    pub memory_required: Option<usize>,
    pub memory_available: Option<usize>,
    pub time_elapsed: Option<std::time::Duration>,
    pub time_limit: Option<std::time::Duration>,
}

impl LinearModelError {
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            LinearModelError::DataError(e) => match e.kind {
                DataErrorKind::EmptyData | DataErrorKind::MissingTargets => ErrorSeverity::Critical,
                DataErrorKind::InvalidValues { .. } => ErrorSeverity::High,
                _ => ErrorSeverity::Medium,
            },
            LinearModelError::NumericalError(e) => match e.kind {
                NumericalErrorKind::SingularMatrix { .. }
                | NumericalErrorKind::Overflow
                | NumericalErrorKind::Underflow => ErrorSeverity::High,
                _ => ErrorSeverity::Medium,
            },
            LinearModelError::OptimizationError(_) => ErrorSeverity::Medium,
            LinearModelError::ConfigurationError(_) => ErrorSeverity::Low,
            LinearModelError::StateError(_) => ErrorSeverity::Medium,
            LinearModelError::FeatureError(_) => ErrorSeverity::Medium,
            LinearModelError::MatrixError(_) => ErrorSeverity::High,
            LinearModelError::CrossValidationError(_) => ErrorSeverity::Medium,
            LinearModelError::ResourceError(e) => match e.kind {
                ResourceErrorKind::InsufficientMemory => ErrorSeverity::Critical,
                _ => ErrorSeverity::Medium,
            },
        }
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            LinearModelError::DataError(e) => e.user_message(),
            LinearModelError::ConfigurationError(e) => e.user_message(),
            LinearModelError::NumericalError(e) => e.user_message(),
            LinearModelError::OptimizationError(e) => e.user_message(),
            LinearModelError::StateError(e) => e.user_message(),
            LinearModelError::FeatureError(e) => e.user_message(),
            LinearModelError::MatrixError(e) => e.user_message(),
            LinearModelError::CrossValidationError(e) => e.user_message(),
            LinearModelError::ResourceError(e) => e.user_message(),
        }
    }

    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            LinearModelError::DataError(e) => e.suggestions.clone(),
            LinearModelError::ConfigurationError(e) => e.suggestions.clone(),
            LinearModelError::NumericalError(e) => e.recovery_suggestions.clone(),
            LinearModelError::OptimizationError(e) => e.suggestions.clone(),
            LinearModelError::StateError(_) => vec![
                "Check model state before calling this method".to_string(),
                "Call fit() before predict() or transform()".to_string(),
            ],
            LinearModelError::FeatureError(e) => e.suggestions.clone(),
            LinearModelError::MatrixError(e) => e.suggestions.clone(),
            LinearModelError::CrossValidationError(_) => vec![
                "Check data distribution across folds".to_string(),
                "Consider stratified cross-validation".to_string(),
            ],
            LinearModelError::ResourceError(e) => e.suggestions.clone(),
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            LinearModelError::DataError(e) => !matches!(
                e.kind,
                DataErrorKind::EmptyData | DataErrorKind::MissingTargets
            ),
            LinearModelError::NumericalError(e) => match e.kind {
                NumericalErrorKind::SingularMatrix { .. } => true,
                NumericalErrorKind::Overflow | NumericalErrorKind::Underflow => false,
                _ => true,
            },
            LinearModelError::ConfigurationError(_) => true,
            LinearModelError::OptimizationError(_) => true,
            LinearModelError::StateError(_) => true,
            LinearModelError::FeatureError(_) => true,
            LinearModelError::MatrixError(_) => true,
            LinearModelError::CrossValidationError(_) => true,
            LinearModelError::ResourceError(e) => {
                !matches!(e.kind, ResourceErrorKind::InsufficientMemory)
            }
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl DataError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            DataErrorKind::EmptyData =>
                "The provided dataset is empty. Please provide data with at least one sample.".to_string(),
            DataErrorKind::DimensionMismatch { expected, actual } =>
                format!("Data dimensions don't match. Expected {:?}, got {:?}. {}", expected, actual, self.context),
            DataErrorKind::InvalidValues { count, total } =>
                format!("Found {} invalid values (NaN/infinity) out of {} total values. {}", count, total, self.context),
            DataErrorKind::MissingTargets =>
                "Target values are missing or empty. Linear models require target values for training.".to_string(),
            DataErrorKind::InsufficientData { required, available } =>
                format!("Insufficient data for operation. Required: {}, Available: {}. {}", required, available, self.context),
            DataErrorKind::IncompatibleDataType { expected, actual } =>
                format!("Data type mismatch. Expected: {}, Actual: {}. {}", expected, actual, self.context),
            DataErrorKind::DataRangeError { min_required, actual_min } =>
                format!("Data values out of required range. Minimum required: {}, Actual minimum: {}. {}", min_required, actual_min, self.context),
        }
    }
}

impl ConfigurationError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            ConfigurationErrorKind::InvalidParameter => format!(
                "Invalid value '{}' for parameter '{}'. {}",
                self.provided_value,
                self.parameter_name,
                self.valid_range.as_deref().unwrap_or("")
            ),
            ConfigurationErrorKind::OutOfRange => format!(
                "Parameter '{}' value '{}' is out of valid range: {}",
                self.parameter_name,
                self.provided_value,
                self.valid_range.as_deref().unwrap_or("unknown")
            ),
            ConfigurationErrorKind::IncompatibleParameters { conflicting_params } => format!(
                "Parameter '{}' is incompatible with: {}. Current value: '{}'",
                self.parameter_name,
                conflicting_params.join(", "),
                self.provided_value
            ),
            ConfigurationErrorKind::MissingParameter => {
                format!("Required parameter '{}' is missing.", self.parameter_name)
            }
            ConfigurationErrorKind::DeprecatedParameter { replacement } => format!(
                "Parameter '{}' is deprecated. {}",
                self.parameter_name,
                replacement
                    .as_deref()
                    .map(|r| format!("Use '{}' instead.", r))
                    .unwrap_or("".to_string())
            ),
        }
    }
}

impl NumericalError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            NumericalErrorKind::SingularMatrix { condition_number } => {
                let cond_info = condition_number
                    .map(|c| format!(" (condition number: {:.2e})", c))
                    .unwrap_or_default();
                format!(
                    "Matrix is singular or nearly singular{} during {}. {}",
                    cond_info, self.operation, self.context
                )
            }
            NumericalErrorKind::Overflow => format!(
                "Numerical overflow occurred during {}. {}",
                self.operation, self.context
            ),
            NumericalErrorKind::Underflow => format!(
                "Numerical underflow occurred during {}. {}",
                self.operation, self.context
            ),
            NumericalErrorKind::PrecisionLoss { digits_lost } => format!(
                "Significant precision loss ({} digits) during {}. {}",
                digits_lost, self.operation, self.context
            ),
            NumericalErrorKind::IllConditioned { condition_number } => format!(
                "Ill-conditioned problem (condition number: {:.2e}) during {}. {}",
                condition_number, self.operation, self.context
            ),
            NumericalErrorKind::MatrixInversionFailed => format!(
                "Failed to invert matrix during {}. {}",
                self.operation, self.context
            ),
            NumericalErrorKind::EigenvalueFailed => format!(
                "Eigenvalue computation failed during {}. {}",
                self.operation, self.context
            ),
            NumericalErrorKind::CholeskyFailed => format!(
                "Cholesky decomposition failed during {}. Matrix may not be positive definite. {}",
                self.operation, self.context
            ),
        }
    }
}

impl OptimizationError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            OptimizationErrorKind::ConvergenceFailed => {
                let iter_info = match (self.iteration, self.max_iterations) {
                    (Some(iter), Some(max_iter)) => {
                        format!(" after {} iterations (max: {})", iter, max_iter)
                    }
                    (Some(iter), None) => format!(" after {} iterations", iter),
                    _ => String::new(),
                };
                format!("{} failed to converge{}.", self.algorithm, iter_info)
            }
            OptimizationErrorKind::LocalMinimum => format!(
                "{} may have converged to a local minimum rather than global minimum.",
                self.algorithm
            ),
            OptimizationErrorKind::NoProgress => format!(
                "{} is not making progress. Objective function is not decreasing.",
                self.algorithm
            ),
            OptimizationErrorKind::StepSizeTooSmall => format!(
                "{} step size became too small to make progress.",
                self.algorithm
            ),
            OptimizationErrorKind::GradientFailed => {
                format!("Gradient computation failed in {}.", self.algorithm)
            }
            OptimizationErrorKind::HessianFailed => {
                format!("Hessian computation failed in {}.", self.algorithm)
            }
            OptimizationErrorKind::LineSearchFailed => {
                format!("Line search failed in {}.", self.algorithm)
            }
            OptimizationErrorKind::InvalidDirection => format!(
                "Invalid optimization direction computed in {}.",
                self.algorithm
            ),
            OptimizationErrorKind::InvalidProblemDimensions => {
                format!("Invalid problem dimensions for {}.", self.algorithm)
            }
            OptimizationErrorKind::ModelNotFitted => {
                format!("Model must be fitted before use in {}.", self.algorithm)
            }
        }
    }
}

impl StateError {
    pub fn user_message(&self) -> String {
        format!(
            "Cannot perform '{}' operation. Model is in '{}' state but requires '{}' state.",
            self.operation, self.current_state, self.required_state
        )
    }
}

impl FeatureError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            FeatureErrorKind::ZeroVariance => format!(
                "Features with zero variance detected at indices: {:?}. {}",
                self.feature_indices, self.context
            ),
            FeatureErrorKind::Multicollinearity {
                correlation_threshold,
            } => format!(
                "High correlation (>{}) detected between features: {:?}. {}",
                correlation_threshold, self.feature_indices, self.context
            ),
            FeatureErrorKind::OutOfRange { min, max } => format!(
                "Features out of expected range [{}, {}] at indices: {:?}. {}",
                min, max, self.feature_indices, self.context
            ),
            FeatureErrorKind::MissingFeatures => format!(
                "Missing features at indices: {:?}. {}",
                self.feature_indices, self.context
            ),
            FeatureErrorKind::CurseOfDimensionality {
                n_features,
                n_samples,
            } => format!(
                "Too many features ({}) relative to samples ({}). This may lead to overfitting. {}",
                n_features, n_samples, self.context
            ),
            FeatureErrorKind::ScalingError { method } => format!(
                "Feature scaling failed using method '{}' for features: {:?}. {}",
                method, self.feature_indices, self.context
            ),
        }
    }
}

impl MatrixError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            MatrixErrorKind::DimensionMismatch => format!(
                "Matrix dimension mismatch during {}. Matrix is {}x{}.",
                self.operation, self.matrix_info.dimensions.0, self.matrix_info.dimensions.1
            ),
            MatrixErrorKind::NotSquare => format!(
                "Square matrix required for {} but got {}x{} matrix.",
                self.operation, self.matrix_info.dimensions.0, self.matrix_info.dimensions.1
            ),
            MatrixErrorKind::NotSymmetric => format!(
                "Symmetric matrix required for {} but matrix is not symmetric.",
                self.operation
            ),
            MatrixErrorKind::NotPositiveDefinite => format!(
                "Positive definite matrix required for {} but matrix is not positive definite.",
                self.operation
            ),
            MatrixErrorKind::SparseOperationFailed => {
                format!("Sparse matrix operation '{}' failed.", self.operation)
            }
            MatrixErrorKind::AllocationFailed { required_bytes } => format!(
                "Failed to allocate {} bytes for matrix operation '{}'.",
                required_bytes, self.operation
            ),
        }
    }
}

impl CrossValidationError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            CrossValidationErrorKind::InsufficientData => {
                "Insufficient data for cross-validation. Need more samples than number of folds."
                    .to_string()
            }
            CrossValidationErrorKind::InvalidFolds => {
                "Invalid cross-validation fold configuration.".to_string()
            }
            CrossValidationErrorKind::ImbalancedFold => {
                if let Some(ref fold_info) = self.fold_info {
                    format!(
                        "Fold {}/{} contains imbalanced classes or missing classes.",
                        fold_info.current_fold + 1,
                        fold_info.total_folds
                    )
                } else {
                    "Cross-validation fold contains imbalanced classes.".to_string()
                }
            }
            CrossValidationErrorKind::ScoringFailed { metric } => {
                format!("Cross-validation scoring failed for metric '{}'.", metric)
            }
            CrossValidationErrorKind::EarlyStoppingFailed => {
                "Early stopping criteria could not be applied during cross-validation.".to_string()
            }
        }
    }
}

impl ResourceError {
    pub fn user_message(&self) -> String {
        match &self.kind {
            ResourceErrorKind::InsufficientMemory => {
                if let (Some(required), Some(available)) = (
                    self.resource_info.memory_required,
                    self.resource_info.memory_available,
                ) {
                    format!(
                        "Insufficient memory. Required: {} bytes, Available: {} bytes.",
                        required, available
                    )
                } else {
                    "Insufficient memory for operation.".to_string()
                }
            }
            ResourceErrorKind::TimeoutExceeded => {
                if let (Some(elapsed), Some(limit)) = (
                    self.resource_info.time_elapsed,
                    self.resource_info.time_limit,
                ) {
                    format!(
                        "Operation timed out. Elapsed: {:?}, Limit: {:?}.",
                        elapsed, limit
                    )
                } else {
                    "Operation exceeded time limit.".to_string()
                }
            }
            ResourceErrorKind::FileIoError { operation, path } => format!(
                "File I/O error during '{}' operation on path: '{}'.",
                operation, path
            ),
            ResourceErrorKind::NetworkError => {
                "Network error occurred during distributed computation.".to_string()
            }
        }
    }
}

impl fmt::Display for LinearModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.user_message())
    }
}

impl std::error::Error for LinearModelError {}

impl From<LinearModelError> for SklearsError {
    fn from(err: LinearModelError) -> Self {
        match err {
            LinearModelError::DataError(_) => SklearsError::InvalidInput(err.to_string()),
            LinearModelError::ConfigurationError(_) => SklearsError::InvalidInput(err.to_string()),
            LinearModelError::NumericalError(_) => SklearsError::NumericalError(err.to_string()),
            LinearModelError::OptimizationError(_) => {
                SklearsError::ConvergenceError { iterations: 0 }
            }
            LinearModelError::StateError(_) => SklearsError::NotFitted {
                operation: err.to_string(),
            },
            LinearModelError::FeatureError(_) => SklearsError::InvalidInput(err.to_string()),
            LinearModelError::MatrixError(_) => SklearsError::NumericalError(err.to_string()),
            LinearModelError::CrossValidationError(_) => {
                SklearsError::InvalidInput(err.to_string())
            }
            LinearModelError::ResourceError(_) => SklearsError::Other(err.to_string()),
        }
    }
}

/// Builder for creating specific error types
pub struct ErrorBuilder;

impl ErrorBuilder {
    /// Create a data error
    pub fn data_error(kind: DataErrorKind, context: &str) -> LinearModelError {
        let suggestions = match &kind {
            DataErrorKind::EmptyData => vec![
                "Provide a dataset with at least one sample".to_string(),
                "Check data loading pipeline".to_string(),
            ],
            DataErrorKind::DimensionMismatch { .. } => vec![
                "Check input data dimensions".to_string(),
                "Ensure X and y have compatible shapes".to_string(),
            ],
            DataErrorKind::InvalidValues { .. } => vec![
                "Remove or impute NaN/infinity values".to_string(),
                "Check data preprocessing pipeline".to_string(),
            ],
            DataErrorKind::MissingTargets => {
                vec!["Provide target values (y) for supervised learning".to_string()]
            }
            DataErrorKind::InsufficientData { required, .. } => vec![
                format!("Collect at least {} samples", required),
                "Consider reducing model complexity".to_string(),
            ],
            DataErrorKind::IncompatibleDataType { expected, .. } => {
                vec![format!("Convert data to {} type", expected)]
            }
            DataErrorKind::DataRangeError { min_required, .. } => vec![
                format!("Ensure all values are >= {}", min_required),
                "Consider data transformation".to_string(),
            ],
        };

        LinearModelError::DataError(DataError {
            kind,
            context: context.to_string(),
            suggestions,
            error_location: None,
        })
    }

    /// Create a numerical error
    pub fn numerical_error(
        kind: NumericalErrorKind,
        operation: &str,
        context: &str,
    ) -> LinearModelError {
        let recovery_suggestions = match &kind {
            NumericalErrorKind::SingularMatrix { .. } => vec![
                "Add regularization (Ridge, Lasso)".to_string(),
                "Remove linearly dependent features".to_string(),
                "Use pseudo-inverse instead of inverse".to_string(),
            ],
            NumericalErrorKind::IllConditioned { .. } => vec![
                "Apply feature scaling/normalization".to_string(),
                "Add regularization".to_string(),
                "Use iterative refinement".to_string(),
            ],
            _ => vec![
                "Try different solver".to_string(),
                "Adjust numerical precision".to_string(),
            ],
        };

        LinearModelError::NumericalError(NumericalError {
            kind,
            operation: operation.to_string(),
            context: context.to_string(),
            matrix_info: None,
            recovery_suggestions,
        })
    }

    /// Create an optimization error
    pub fn optimization_error(
        kind: OptimizationErrorKind,
        algorithm: &str,
        iteration: Option<usize>,
        max_iterations: Option<usize>,
    ) -> LinearModelError {
        let suggestions = match &kind {
            OptimizationErrorKind::ConvergenceFailed => vec![
                "Increase max_iterations".to_string(),
                "Adjust convergence tolerance".to_string(),
                "Try different solver".to_string(),
                "Scale features".to_string(),
            ],
            OptimizationErrorKind::LocalMinimum => vec![
                "Use different initialization".to_string(),
                "Try global optimization method".to_string(),
                "Add regularization".to_string(),
            ],
            _ => vec![
                "Adjust learning rate".to_string(),
                "Try different optimization algorithm".to_string(),
            ],
        };

        LinearModelError::OptimizationError(OptimizationError {
            kind,
            algorithm: algorithm.to_string(),
            iteration,
            max_iterations,
            convergence_info: None,
            suggestions,
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_error_creation() {
        let error = ErrorBuilder::data_error(DataErrorKind::EmptyData, "Test context");

        assert!(matches!(error, LinearModelError::DataError(_)));
        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert!(!error.recovery_suggestions().is_empty());
    }

    #[test]
    fn test_numerical_error_creation() {
        let error = ErrorBuilder::numerical_error(
            NumericalErrorKind::SingularMatrix {
                condition_number: Some(1e-15),
            },
            "matrix inversion",
            "During normal equations solve",
        );

        assert!(matches!(error, LinearModelError::NumericalError(_)));
        assert_eq!(error.severity(), ErrorSeverity::High);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_optimization_error_creation() {
        let error = ErrorBuilder::optimization_error(
            OptimizationErrorKind::ConvergenceFailed,
            "L-BFGS",
            Some(100),
            Some(100),
        );

        assert!(matches!(error, LinearModelError::OptimizationError(_)));
        assert_eq!(error.severity(), ErrorSeverity::Medium);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_error_conversion_to_skl_error() {
        let linear_error = ErrorBuilder::data_error(DataErrorKind::EmptyData, "Test");

        let skl_error: SklearsError = linear_error.into();
        assert!(matches!(skl_error, SklearsError::InvalidInput(_)));
    }

    #[test]
    fn test_user_message_formatting() {
        let error = ErrorBuilder::data_error(
            DataErrorKind::DimensionMismatch {
                expected: vec![100, 10],
                actual: vec![100, 5],
            },
            "Training data validation",
        );

        let message = error.user_message();
        assert!(message.contains("dimensions don't match"));
        assert!(message.contains("[100, 10]"));
        assert!(message.contains("[100, 5]"));
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::High);
        assert!(ErrorSeverity::High > ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium > ErrorSeverity::Low);
    }
}
