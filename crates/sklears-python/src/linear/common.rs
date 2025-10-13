//! Common functionality for linear model Python bindings
//!
//! This module contains shared imports, types, and utilities used
//! across all linear model implementations.

// Re-export commonly used types and traits - Using SciRS2-Core for improved performance
pub use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
pub use pyo3::exceptions::PyValueError;
pub use pyo3::prelude::*;
pub use scirs2_core::ndarray::{Array1, Array2};

// Performance optimization imports
#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

/// Common error type for linear model operations
pub type LinearModelResult<T> = Result<T, PyValueError>;

/// Enhanced error handling for sklears-python
#[derive(Debug)]
pub enum SklearsPythonError {
    /// Input validation errors
    ValidationError(String),
    /// Model fitting errors
    FittingError(String),
    /// Prediction errors
    PredictionError(String),
    /// Memory allocation errors
    MemoryError(String),
    /// Numerical computation errors
    NumericalError(String),
    /// Configuration errors
    ConfigurationError(String),
}

impl std::fmt::Display for SklearsPythonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SklearsPythonError::ValidationError(msg) => write!(f, "Validation Error: {}", msg),
            SklearsPythonError::FittingError(msg) => write!(f, "Model Fitting Error: {}", msg),
            SklearsPythonError::PredictionError(msg) => write!(f, "Prediction Error: {}", msg),
            SklearsPythonError::MemoryError(msg) => write!(f, "Memory Error: {}", msg),
            SklearsPythonError::NumericalError(msg) => write!(f, "Numerical Error: {}", msg),
            SklearsPythonError::ConfigurationError(msg) => {
                write!(f, "Configuration Error: {}", msg)
            }
        }
    }
}

impl std::error::Error for SklearsPythonError {}

impl From<SklearsPythonError> for PyErr {
    fn from(err: SklearsPythonError) -> Self {
        match err {
            SklearsPythonError::ValidationError(msg) => PyValueError::new_err(msg),
            SklearsPythonError::FittingError(msg) => PyRuntimeError::new_err(msg),
            SklearsPythonError::PredictionError(msg) => PyRuntimeError::new_err(msg),
            SklearsPythonError::MemoryError(msg) => {
                use pyo3::exceptions::PyMemoryError;
                PyMemoryError::new_err(msg)
            }
            SklearsPythonError::NumericalError(msg) => PyArithmeticError::new_err(msg),
            SklearsPythonError::ConfigurationError(msg) => PyValueError::new_err(msg),
        }
    }
}

// Import additional exception types
use pyo3::exceptions::{PyArithmeticError, PyRuntimeError};

/// Calculate R² score with optimized array operations
pub fn calculate_r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let y_mean = y_true.mean().unwrap_or(0.0);

    // Use optimized array operations
    let y_centered: Array1<f64> = y_true.mapv(|y| y - y_mean);
    let residuals: Array1<f64> = y_true - y_pred;
    let ss_tot = y_centered.dot(&y_centered);
    let ss_res = residuals.dot(&residuals);

    1.0 - (ss_res / ss_tot)
}

/// Validate input arrays for model fitting
pub fn validate_fit_arrays(x: &Array2<f64>, y: &Array1<f64>) -> PyResult<()> {
    if x.nrows() != y.len() {
        return Err(PyValueError::new_err(format!(
            "X and y have incompatible shapes: X has {} samples, y has {} samples",
            x.nrows(),
            y.len()
        )));
    }

    if x.nrows() == 0 {
        return Err(PyValueError::new_err("X and y must not be empty"));
    }

    if x.ncols() == 0 {
        return Err(PyValueError::new_err("X must have at least one feature"));
    }

    Ok(())
}

/// Validate input arrays for prediction
pub fn validate_predict_array(x: &Array2<f64>) -> PyResult<()> {
    if x.nrows() == 0 {
        return Err(SklearsPythonError::ValidationError("X must not be empty".to_string()).into());
    }

    if x.ncols() == 0 {
        return Err(SklearsPythonError::ValidationError(
            "X must have at least one feature".to_string(),
        )
        .into());
    }

    // Check for invalid values
    validate_finite_values(x)?;

    Ok(())
}

/// Enhanced validation functions with better error handling
pub fn validate_fit_arrays_enhanced(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> Result<(), SklearsPythonError> {
    if x.nrows() != y.len() {
        return Err(SklearsPythonError::ValidationError(format!(
            "X and y have incompatible shapes: X has {} samples, y has {} samples",
            x.nrows(),
            y.len()
        )));
    }

    if x.nrows() == 0 {
        return Err(SklearsPythonError::ValidationError(
            "X and y must not be empty".to_string(),
        ));
    }

    if x.ncols() == 0 {
        return Err(SklearsPythonError::ValidationError(
            "X must have at least one feature".to_string(),
        ));
    }

    // Check for infinite or NaN values
    validate_finite_values(x)?;
    validate_finite_values_1d(y)?;

    // Memory usage validation (warn if arrays are very large)
    check_memory_usage(x, y)?;

    Ok(())
}

/// Validate that array contains only finite values
pub fn validate_finite_values(arr: &Array2<f64>) -> Result<(), SklearsPythonError> {
    for value in arr.iter() {
        if !value.is_finite() {
            return Err(SklearsPythonError::NumericalError(
                "Input array contains non-finite values (NaN or infinite)".to_string(),
            ));
        }
    }
    Ok(())
}

/// Validate that 1D array contains only finite values
pub fn validate_finite_values_1d(arr: &Array1<f64>) -> Result<(), SklearsPythonError> {
    for value in arr.iter() {
        if !value.is_finite() {
            return Err(SklearsPythonError::NumericalError(
                "Target array contains non-finite values (NaN or infinite)".to_string(),
            ));
        }
    }
    Ok(())
}

/// Check memory usage and warn if arrays are very large
pub fn check_memory_usage(x: &Array2<f64>, y: &Array1<f64>) -> Result<(), SklearsPythonError> {
    let x_memory_mb = (x.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    let y_memory_mb = (y.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    let total_memory_mb = x_memory_mb + y_memory_mb;

    // Warn if using more than 1GB of memory
    if total_memory_mb > 1024.0 {
        eprintln!("Warning: Large dataset detected ({:.2} MB). Consider using batch processing or data preprocessing to reduce memory usage.", total_memory_mb);
    }

    // Error if using more than 4GB (likely will cause issues)
    if total_memory_mb > 4096.0 {
        return Err(SklearsPythonError::MemoryError(format!(
            "Dataset is too large ({:.2} MB). Consider using data preprocessing to reduce memory usage.",
            total_memory_mb
        )));
    }

    Ok(())
}

/// Get system memory information for better memory management
pub fn get_available_memory_mb() -> f64 {
    // This is a simplified implementation
    // In a real implementation, you'd use system APIs to get actual available memory
    // For now, we assume 8GB as a reasonable default
    8192.0
}

/// Performance monitoring structure
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub training_time_ms: Option<f64>,
    pub prediction_time_ms: Option<f64>,
    pub memory_usage_mb: Option<f64>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}
