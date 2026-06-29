//! Error types for sklears-compose crate

pub use sklears_core::error::{Result as SklResult, SklearsError};

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, SklearsComposeError>;

/// Compose-specific error types
#[derive(Debug, thiserror::Error)]
pub enum SklearsComposeError {
    /// Core sklears error
    #[error(transparent)]
    Core(#[from] SklearsError),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Invalid data
    #[error("Invalid data: {reason}")]
    /// Field value.
    InvalidData {
        /// The reason.
        reason: String,
    },

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Error type used by the comprehensive_benchmarking module.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BenchmarkError {
    /// Theme not found.
    #[error("Theme not found: {0}")]
    ThemeNotFound(String),

    /// Channel not found.
    #[error("Channel not found: {0}")]
    ChannelNotFound(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Request not found.
    #[error("Request not found: {0}")]
    RequestNotFound(String),

    /// System error.
    #[error("System error: {0}")]
    SystemError(String),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// Result type alias for benchmarking operations.
pub type BenchmarkResult<T> = std::result::Result<T, BenchmarkError>;

impl From<BenchmarkError> for SklearsComposeError {
    fn from(e: BenchmarkError) -> Self {
        Self::Other(e.to_string())
    }
}

impl From<String> for SklearsComposeError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for SklearsComposeError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}
