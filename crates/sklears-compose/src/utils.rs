//! Utility functions and types for sklears-compose

use uuid::Uuid;

/// Generate a unique ID string.
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Validate configuration (stub — returns Ok).
pub fn validate_config<T>(_config: &T) -> Result<(), crate::error::SklearsComposeError> {
    Ok(())
}

/// Metrics collection stub.
#[derive(Debug, Clone, Default)]
pub struct MetricsCollector;

impl MetricsCollector {
    /// Create a new `MetricsCollector`.
    pub fn new() -> Self {
        Self
    }
}

/// Security management stub.
#[derive(Debug, Clone, Default)]
pub struct SecurityManager;

impl SecurityManager {
    /// Create a new `SecurityManager`.
    pub fn new() -> Self {
        Self
    }
}
