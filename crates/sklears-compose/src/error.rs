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
    InvalidData { reason: String },

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
