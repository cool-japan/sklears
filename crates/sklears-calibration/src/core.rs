//! Core types and error handling for the calibration framework

use std::fmt;

/// Result type for calibration operations
pub type CalibrationResult<T> = Result<T, CalibrationError>;

/// Error types for calibration operations
#[derive(Debug, Clone)]
pub enum CalibrationError {
    /// Invalid input provided to calibration method
    InvalidInput(String),
    /// Numerical computation error
    NumericalError(String),
    /// Optimization failed to converge
    ConvergenceError(String),
    /// Insufficient data for calibration
    InsufficientData(String),
    /// Mathematical constraint violation
    ConstraintViolation(String),
    /// General calibration error
    General(String),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CalibrationError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            CalibrationError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
            CalibrationError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            CalibrationError::ConstraintViolation(msg) => {
                write!(f, "Constraint violation: {}", msg)
            }
            CalibrationError::General(msg) => write!(f, "Calibration error: {}", msg),
        }
    }
}

impl std::error::Error for CalibrationError {}
