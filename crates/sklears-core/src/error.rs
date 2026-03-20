use thiserror::Error;

/// Main error type for sklears
#[derive(Error, Debug)]
pub enum SklearsError {
    /// Error during model fitting
    #[error("Fit error: {0}")]
    FitError(String),

    /// Error during prediction
    #[error("Prediction error: {0}")]
    PredictError(String),

    /// Error during data transformation
    #[error("Transform error: {0}")]
    TransformError(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid data quality
    #[error("Invalid data: {reason}")]
    InvalidData { reason: String },

    /// Shape mismatch between arrays
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Dimension mismatch between arrays
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Model not fitted
    #[error("Model not fitted. Call fit() before {operation}")]
    NotFitted { operation: String },

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Convergence failure
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceError { iterations: usize },

    /// Feature dimension mismatch
    #[error("Feature dimension mismatch: model expects {expected} features, got {actual}")]
    FeatureMismatch { expected: usize, actual: usize },

    /// Missing dependency error
    #[error("Missing dependency '{dependency}' required for {feature}")]
    MissingDependency { dependency: String, feature: String },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// File operation error
    #[error("File error: {0}")]
    FileError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Invalid state error
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Trait not found
    #[error("Trait not found: {0}")]
    TraitNotFound(String),

    /// Analysis error
    #[error("Analysis error: {0}")]
    AnalysisError(String),

    /// Hardware error
    #[error("Hardware error: {0}")]
    HardwareError(String),

    /// Resource allocation error
    #[error("Resource allocation error: {0}")]
    ResourceAllocationError(String),

    /// Invalid configuration error
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Model error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

impl Clone for SklearsError {
    fn clone(&self) -> Self {
        match self {
            SklearsError::FitError(s) => SklearsError::FitError(s.clone()),
            SklearsError::PredictError(s) => SklearsError::PredictError(s.clone()),
            SklearsError::TransformError(s) => SklearsError::TransformError(s.clone()),
            SklearsError::InvalidInput(s) => SklearsError::InvalidInput(s.clone()),
            SklearsError::InvalidData { reason } => SklearsError::InvalidData {
                reason: reason.clone(),
            },
            SklearsError::ShapeMismatch { expected, actual } => SklearsError::ShapeMismatch {
                expected: expected.clone(),
                actual: actual.clone(),
            },
            SklearsError::InvalidParameter { name, reason } => SklearsError::InvalidParameter {
                name: name.clone(),
                reason: reason.clone(),
            },
            SklearsError::DimensionMismatch { expected, actual } => {
                SklearsError::DimensionMismatch {
                    expected: *expected,
                    actual: *actual,
                }
            }
            SklearsError::NotFitted { operation } => SklearsError::NotFitted {
                operation: operation.clone(),
            },
            SklearsError::NumericalError(s) => SklearsError::NumericalError(s.clone()),
            SklearsError::ConvergenceError { iterations } => SklearsError::ConvergenceError {
                iterations: *iterations,
            },
            SklearsError::FeatureMismatch { expected, actual } => SklearsError::FeatureMismatch {
                expected: *expected,
                actual: *actual,
            },
            SklearsError::IoError(io_err) => {
                // Since std::io::Error doesn't implement Clone, we create a new one with the same kind and message
                SklearsError::IoError(std::io::Error::new(io_err.kind(), format!("{io_err}")))
            }
            SklearsError::FileError(s) => SklearsError::FileError(s.clone()),
            SklearsError::SerializationError(s) => SklearsError::SerializationError(s.clone()),
            SklearsError::DeserializationError(s) => SklearsError::DeserializationError(s.clone()),
            SklearsError::NotImplemented(s) => SklearsError::NotImplemented(s.clone()),
            SklearsError::InvalidOperation(s) => SklearsError::InvalidOperation(s.clone()),
            SklearsError::InvalidState(s) => SklearsError::InvalidState(s.clone()),
            SklearsError::Configuration(s) => SklearsError::Configuration(s.clone()),
            SklearsError::MissingDependency {
                dependency,
                feature,
            } => SklearsError::MissingDependency {
                dependency: dependency.clone(),
                feature: feature.clone(),
            },
            SklearsError::TraitNotFound(s) => SklearsError::TraitNotFound(s.clone()),
            SklearsError::AnalysisError(s) => SklearsError::AnalysisError(s.clone()),
            SklearsError::HardwareError(s) => SklearsError::HardwareError(s.clone()),
            SklearsError::ResourceAllocationError(s) => {
                SklearsError::ResourceAllocationError(s.clone())
            }
            SklearsError::InvalidConfiguration(s) => SklearsError::InvalidConfiguration(s.clone()),
            SklearsError::ProcessingError(s) => SklearsError::ProcessingError(s.clone()),
            SklearsError::ModelError(s) => SklearsError::ModelError(s.clone()),
            SklearsError::ValidationError(s) => SklearsError::ValidationError(s.clone()),
            SklearsError::Other(s) => SklearsError::Other(s.clone()),
        }
    }
}

// Convert from String
impl From<String> for SklearsError {
    fn from(error: String) -> Self {
        SklearsError::Other(error)
    }
}

// Convert from &str
impl From<&str> for SklearsError {
    fn from(error: &str) -> Self {
        SklearsError::Other(error.to_string())
    }
}

// Convert from ndarray ShapeError
impl From<scirs2_core::ndarray::ShapeError> for SklearsError {
    fn from(error: scirs2_core::ndarray::ShapeError) -> Self {
        SklearsError::InvalidInput(format!("Array shape error: {error}"))
    }
}

// Convert from serde_json::Error
impl From<serde_json::Error> for SklearsError {
    fn from(error: serde_json::Error) -> Self {
        SklearsError::SerializationError(format!("JSON serialization error: {error}"))
    }
}

/// Result type alias for sklears operations
pub type Result<T> = std::result::Result<T, SklearsError>;

/// Enhanced error context trait for better error propagation
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context(self, msg: &str) -> Result<T>;

    /// Add context with a lazy-evaluated closure
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    /// Add operation context for debugging
    fn with_operation(self, operation: &str) -> Result<T>;

    /// Add location context for debugging  
    fn with_location(self, file: &str, line: u32) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: std::error::Error,
{
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| SklearsError::Other(format!("{msg}: {e}")))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| SklearsError::Other(format!("{}: {e}", f())))
    }

    fn with_operation(self, operation: &str) -> Result<T> {
        self.map_err(|e| SklearsError::Other(format!("Operation '{operation}' failed: {e}")))
    }

    fn with_location(self, file: &str, line: u32) -> Result<T> {
        self.map_err(|e| SklearsError::Other(format!("Error at {file}:{line}: {e}")))
    }
}

/// Macro for adding location context automatically
#[macro_export]
macro_rules! error_context {
    ($result:expr) => {
        $result.with_location(file!(), line!())
    };
    ($result:expr, $msg:expr) => {
        $result.context($msg).with_location(file!(), line!())
    };
}

/// Enhanced context propagation for sklearn-specific operations
pub trait SklearnContext<T> {
    /// Add context for fitting operations
    fn fit_context(self, estimator: &str, samples: usize, features: usize) -> Result<T>;

    /// Add context for prediction operations  
    fn predict_context(self, estimator: &str, samples: usize) -> Result<T>;

    /// Add context for transformation operations
    fn transform_context(self, transformer: &str, samples: usize, features: usize) -> Result<T>;

    /// Add context for validation operations
    fn validation_context(self, parameter: &str, value: &str) -> Result<T>;
}

impl<T, E> SklearnContext<T> for std::result::Result<T, E>
where
    E: std::error::Error,
{
    fn fit_context(self, estimator: &str, samples: usize, features: usize) -> Result<T> {
        self.with_context(|| {
            format!("Failed to fit {estimator} with {samples} samples and {features} features")
        })
    }

    fn predict_context(self, estimator: &str, samples: usize) -> Result<T> {
        self.with_context(|| format!("Failed to predict using {estimator} with {samples} samples"))
    }

    fn transform_context(self, transformer: &str, samples: usize, features: usize) -> Result<T> {
        self.with_context(|| {
            format!("Failed to transform using {transformer} with {samples} samples and {features} features")
        })
    }

    fn validation_context(self, parameter: &str, value: &str) -> Result<T> {
        self.with_context(|| {
            format!("Validation failed for parameter '{parameter}' with value '{value}'")
        })
    }
}

/// Convenience macro for validation
#[macro_export]
macro_rules! validate {
    ($condition:expr, $message:expr) => {
        if !($condition) {
            return Err($crate::error::SklearsError::InvalidInput($message.to_string()));
        }
    };
    ($condition:expr, $message:expr, $($arg:tt)*) => {
        if !($condition) {
            return Err($crate::error::SklearsError::InvalidInput(format!($message, $($arg)*)));
        }
    };
}

/// Chain multiple errors together for better debugging
#[derive(Debug)]
pub struct ErrorChain {
    errors: Vec<Box<dyn std::error::Error + Send + Sync>>,
    context: Vec<String>,
}

impl ErrorChain {
    /// Create a new error chain
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            context: Vec::new(),
        }
    }

    /// Add an error to the chain
    pub fn push_error<E>(mut self, error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.errors.push(Box::new(error));
        self
    }

    /// Add context to the chain
    pub fn push_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context.push(context.into());
        self
    }

    /// Convert to SklearsError
    pub fn into_error(self) -> SklearsError {
        let message = if self.context.is_empty() && self.errors.is_empty() {
            "Unknown error chain".to_string()
        } else {
            let context_str = self.context.join(" -> ");
            let error_str = self
                .errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");

            if context_str.is_empty() {
                error_str
            } else if error_str.is_empty() {
                context_str
            } else {
                format!("{context_str}: {error_str}")
            }
        };

        SklearsError::Other(message)
    }
}

impl Default for ErrorChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation utilities
pub mod validate {
    use super::*;
    use crate::types::{Array1, Array2, FloatBounds, Numeric};

    /// Check if X and y have compatible shapes
    pub fn check_consistent_length<T, U>(x: &Array2<T>, y: &Array1<U>) -> Result<()> {
        let n_samples_x = x.nrows();
        let n_samples_y = y.len();

        if n_samples_x != n_samples_y {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={n_samples_x}, y.shape[0]={n_samples_y}"),
            });
        }

        Ok(())
    }

    /// Check if array has the expected number of features
    pub fn check_n_features<T>(x: &Array2<T>, expected: usize) -> Result<()> {
        let actual = x.ncols();
        if actual != expected {
            return Err(SklearsError::FeatureMismatch { expected, actual });
        }
        Ok(())
    }

    /// Check if value is finite (generic over floating point types)
    pub fn check_finite<T: FloatBounds>(value: T, name: &str) -> Result<()> {
        if !value.is_finite() {
            return Err(SklearsError::InvalidParameter {
                name: name.to_string(),
                reason: "must be finite".to_string(),
            });
        }
        Ok(())
    }

    /// Check if value is positive (generic over numeric types)
    pub fn check_positive<T: Numeric + PartialOrd>(value: T, name: &str) -> Result<()> {
        if value <= T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: name.to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(())
    }

    /// Check if value is non-negative (generic over numeric types)
    pub fn check_non_negative<T: Numeric + PartialOrd>(value: T, name: &str) -> Result<()> {
        if value < T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: name.to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(())
    }

    /// Check if value is in a specific range
    pub fn check_in_range<T: Numeric + PartialOrd>(
        value: T,
        min: T,
        max: T,
        name: &str,
    ) -> Result<()> {
        if value < min || value > max {
            return Err(SklearsError::InvalidParameter {
                name: name.to_string(),
                reason: format!("must be in range [{min}, {max}]"),
            });
        }
        Ok(())
    }

    /// Check if arrays have compatible shapes for matrix multiplication
    pub fn check_matmul_compatible<T, U>(a: &Array2<T>, b: &Array2<U>) -> Result<()> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::ShapeMismatch {
                expected: "A.shape[1] == B.shape[0]".to_string(),
                actual: format!("A.shape[1]={}, B.shape[0]={}", a.ncols(), b.nrows()),
            });
        }
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));

        let with_context = result.context("Failed to read config file");
        assert!(with_context.is_err());
        assert!(with_context
            .unwrap_err()
            .to_string()
            .contains("Failed to read config file"));
    }

    #[test]
    fn test_error_with_operation() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "access denied",
        ));

        let with_op = result.with_operation("matrix_multiplication");
        assert!(with_op.is_err());
        assert!(with_op
            .unwrap_err()
            .to_string()
            .contains("matrix_multiplication"));
    }

    #[test]
    fn test_sklearn_context() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid data",
        ));

        let with_fit_context = result.fit_context("LinearRegression", 100, 5);
        assert!(with_fit_context.is_err());
        let error_msg = with_fit_context.unwrap_err().to_string();
        assert!(error_msg.contains("LinearRegression"));
        assert!(error_msg.contains("100 samples"));
        assert!(error_msg.contains("5 features"));
    }

    #[test]
    fn test_error_chain() {
        let chain = ErrorChain::new()
            .push_context("Model training")
            .push_context("Data preprocessing")
            .push_error(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "data file missing",
            ))
            .push_context("Feature scaling");

        let error = chain.into_error();
        let error_str = error.to_string();
        assert!(error_str.contains("Model training"));
        assert!(error_str.contains("Data preprocessing"));
        assert!(error_str.contains("Feature scaling"));
        assert!(error_str.contains("data file missing"));
    }

    #[test]
    fn test_validation_context() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "negative value",
        ));

        let with_validation = result.validation_context("learning_rate", "-0.1");
        assert!(with_validation.is_err());
        let error_msg = with_validation.unwrap_err().to_string();
        assert!(error_msg.contains("learning_rate"));
        assert!(error_msg.contains("-0.1"));
    }
}
