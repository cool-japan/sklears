//! Circuit Breaker Error Types and Utilities
//!
//! This module contains all error types, implementations, and utility functions
//! for the circuit breaker system. It provides comprehensive error handling
//! with detailed error information and conversion utilities.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Circuit breaker error enumeration
///
/// Comprehensive error type covering all possible failure modes in the
/// circuit breaker system. Each error type provides specific context
/// about the nature of the failure.
///
/// # Error Categories
///
/// - **State Errors**: Circuit is in a state that prevents operation
/// - **Timeout Errors**: Operations exceeded configured timeouts
/// - **Service Errors**: Underlying service failures
/// - **Configuration Errors**: Invalid configuration parameters
/// - **System Errors**: Internal system failures
///
/// # Usage Examples
///
/// ```rust
/// use sklears_compose::circuit_breaker::CircuitBreakerError;
///
/// let result: Result<(), CircuitBreakerError> = Err(CircuitBreakerError::Timeout);
///
/// match result {
///     Err(CircuitBreakerError::CircuitOpen) => {
///         println!("Circuit breaker is open, requests blocked");
///         // Implement fallback logic
///     }
///     Err(CircuitBreakerError::Timeout) => {
///         println!("Operation timed out");
///         // Retry with different parameters
///     }
///     Err(CircuitBreakerError::ServiceUnavailable) => {
///         println!("Service unavailable");
///         // Use cached data or alternative service
///     }
///     Err(other) => {
///         println!("Other error: {:?}", other);
///     }
///     Ok(_) => println!("Service call succeeded"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerError {
    /// Circuit is open (requests blocked)
    ///
    /// The circuit breaker is in the open state and is blocking requests
    /// to protect the downstream service. This indicates that the failure
    /// threshold has been exceeded and the system is in protective mode.
    CircuitOpen,

    /// Request timeout
    ///
    /// The operation exceeded the configured timeout duration. This can
    /// indicate that the downstream service is slow or unresponsive.
    Timeout,

    /// Service unavailable
    ///
    /// The downstream service is not available or not responding. This
    /// is a general service failure indication.
    ServiceUnavailable,

    /// Configuration error
    ///
    /// Invalid configuration parameters were provided. The string contains
    /// details about the specific configuration issue.
    ConfigurationError(String),

    /// Execution error
    ///
    /// An error occurred during operation execution. The string contains
    /// details about the specific execution failure.
    ExecutionError(String),

    /// Recovery error
    ///
    /// An error occurred during circuit breaker recovery operations.
    /// The string contains details about the recovery failure.
    RecoveryError(String),

    /// Analytics error
    ///
    /// An error occurred in the analytics engine. The string contains
    /// details about the analytics failure.
    AnalyticsError(String),

    /// Statistics error
    ///
    /// An error occurred in statistics collection or processing.
    /// The string contains details about the statistics failure.
    StatisticsError(String),

    /// Event system error
    ///
    /// An error occurred in the event recording or publishing system.
    /// The string contains details about the event system failure.
    EventError(String),

    /// Failure detection error
    ///
    /// An error occurred in the failure detection system.
    /// The string contains details about the failure detection error.
    FailureDetectionError(String),

    /// Resource exhaustion
    ///
    /// System resources (memory, threads, etc.) have been exhausted.
    /// The string contains details about which resource was exhausted.
    ResourceExhaustion(String),

    /// Invalid state transition
    ///
    /// An invalid state transition was attempted. The string contains
    /// details about the invalid transition.
    InvalidStateTransition(String),

    /// Concurrency error
    ///
    /// A concurrency-related error occurred (deadlock, race condition, etc.).
    /// The string contains details about the concurrency issue.
    ConcurrencyError(String),

    /// Network error
    ///
    /// A network-related error occurred when communicating with external services.
    /// The string contains details about the network failure.
    NetworkError(String),

    /// Serialization error
    ///
    /// An error occurred during data serialization or deserialization.
    /// The string contains details about the serialization failure.
    SerializationError(String),

    /// Database error
    ///
    /// An error occurred when interacting with a database or persistent storage.
    /// The string contains details about the database failure.
    DatabaseError(String),

    /// Authentication error
    ///
    /// An authentication-related error occurred.
    /// The string contains details about the authentication failure.
    AuthenticationError(String),

    /// Authorization error
    ///
    /// An authorization-related error occurred.
    /// The string contains details about the authorization failure.
    AuthorizationError(String),

    /// Rate limit exceeded
    ///
    /// The operation was rejected due to rate limiting.
    /// The string contains details about the rate limit that was exceeded.
    RateLimitExceeded(String),

    /// Custom error
    ///
    /// A custom error type for application-specific failures.
    /// The string contains the custom error message.
    Custom(String),
}

impl fmt::Display for CircuitBreakerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitBreakerError::CircuitOpen => {
                write!(
                    f,
                    "Circuit breaker is open - requests are being blocked to protect the service"
                )
            }
            CircuitBreakerError::Timeout => {
                write!(
                    f,
                    "Request timeout - operation took longer than configured timeout"
                )
            }
            CircuitBreakerError::ServiceUnavailable => {
                write!(
                    f,
                    "Service unavailable - downstream service is not responding"
                )
            }
            CircuitBreakerError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {msg}")
            }
            CircuitBreakerError::ExecutionError(msg) => {
                write!(f, "Execution error: {msg}")
            }
            CircuitBreakerError::RecoveryError(msg) => {
                write!(f, "Recovery error: {msg}")
            }
            CircuitBreakerError::AnalyticsError(msg) => {
                write!(f, "Analytics error: {msg}")
            }
            CircuitBreakerError::StatisticsError(msg) => {
                write!(f, "Statistics error: {msg}")
            }
            CircuitBreakerError::EventError(msg) => {
                write!(f, "Event system error: {msg}")
            }
            CircuitBreakerError::FailureDetectionError(msg) => {
                write!(f, "Failure detection error: {msg}")
            }
            CircuitBreakerError::ResourceExhaustion(msg) => {
                write!(f, "Resource exhaustion: {msg}")
            }
            CircuitBreakerError::InvalidStateTransition(msg) => {
                write!(f, "Invalid state transition: {msg}")
            }
            CircuitBreakerError::ConcurrencyError(msg) => {
                write!(f, "Concurrency error: {msg}")
            }
            CircuitBreakerError::NetworkError(msg) => {
                write!(f, "Network error: {msg}")
            }
            CircuitBreakerError::SerializationError(msg) => {
                write!(f, "Serialization error: {msg}")
            }
            CircuitBreakerError::DatabaseError(msg) => {
                write!(f, "Database error: {msg}")
            }
            CircuitBreakerError::AuthenticationError(msg) => {
                write!(f, "Authentication error: {msg}")
            }
            CircuitBreakerError::AuthorizationError(msg) => {
                write!(f, "Authorization error: {msg}")
            }
            CircuitBreakerError::RateLimitExceeded(msg) => {
                write!(f, "Rate limit exceeded: {msg}")
            }
            CircuitBreakerError::Custom(msg) => {
                write!(f, "Circuit breaker error: {msg}")
            }
        }
    }
}

impl Error for CircuitBreakerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // Most circuit breaker errors are leaf errors without underlying causes
        None
    }
}

impl CircuitBreakerError {
    /// Check if this error is retryable
    ///
    /// Returns true if the operation that caused this error could potentially
    /// succeed if retried later, false if retrying is unlikely to help.
    ///
    /// # Returns
    /// - `true`: Error is potentially retryable
    /// - `false`: Error is not retryable
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            // These errors might resolve on retry
            CircuitBreakerError::CircuitOpen => true, // Circuit might close
            CircuitBreakerError::Timeout => true,     // Service might respond faster
            CircuitBreakerError::ServiceUnavailable => true, // Service might come back
            CircuitBreakerError::NetworkError(_) => true, // Network might recover
            CircuitBreakerError::ResourceExhaustion(_) => true, // Resources might free up
            CircuitBreakerError::RateLimitExceeded(_) => true, // Rate limit might reset

            // These errors are unlikely to resolve on retry
            CircuitBreakerError::ConfigurationError(_) => false,
            CircuitBreakerError::AuthenticationError(_) => false,
            CircuitBreakerError::AuthorizationError(_) => false,
            CircuitBreakerError::InvalidStateTransition(_) => false,
            CircuitBreakerError::SerializationError(_) => false,

            // These might or might not be retryable depending on context
            CircuitBreakerError::ExecutionError(_) => false, // Depends on the specific error
            CircuitBreakerError::RecoveryError(_) => false,  // Depends on the recovery issue
            CircuitBreakerError::AnalyticsError(_) => false, // Usually not critical to retry
            CircuitBreakerError::StatisticsError(_) => false, // Usually not critical to retry
            CircuitBreakerError::EventError(_) => false,     // Usually not critical to retry
            CircuitBreakerError::FailureDetectionError(_) => false, // System-level issue
            CircuitBreakerError::ConcurrencyError(_) => true, // Might resolve with timing
            CircuitBreakerError::DatabaseError(_) => true,   // Database might recover
            CircuitBreakerError::Custom(_) => false,         // Unknown, assume not retryable
        }
    }

    /// Get the severity level of this error
    ///
    /// Returns a severity level that can be used for logging, alerting,
    /// and error handling decisions.
    ///
    /// # Returns
    /// - `ErrorSeverity`: The severity level of this error
    #[must_use]
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // High severity - immediate attention required
            CircuitBreakerError::ServiceUnavailable => ErrorSeverity::High,
            CircuitBreakerError::ResourceExhaustion(_) => ErrorSeverity::High,
            CircuitBreakerError::ConcurrencyError(_) => ErrorSeverity::High,
            CircuitBreakerError::DatabaseError(_) => ErrorSeverity::High,

            // Medium severity - should be addressed but not critical
            CircuitBreakerError::CircuitOpen => ErrorSeverity::Medium,
            CircuitBreakerError::Timeout => ErrorSeverity::Medium,
            CircuitBreakerError::NetworkError(_) => ErrorSeverity::Medium,
            CircuitBreakerError::RateLimitExceeded(_) => ErrorSeverity::Medium,
            CircuitBreakerError::ExecutionError(_) => ErrorSeverity::Medium,

            // Low severity - informational or configuration issues
            CircuitBreakerError::ConfigurationError(_) => ErrorSeverity::Low,
            CircuitBreakerError::AuthenticationError(_) => ErrorSeverity::Low,
            CircuitBreakerError::AuthorizationError(_) => ErrorSeverity::Low,
            CircuitBreakerError::InvalidStateTransition(_) => ErrorSeverity::Low,
            CircuitBreakerError::SerializationError(_) => ErrorSeverity::Low,

            // Minimal severity - internal system issues that don't affect operation
            CircuitBreakerError::RecoveryError(_) => ErrorSeverity::Minimal,
            CircuitBreakerError::AnalyticsError(_) => ErrorSeverity::Minimal,
            CircuitBreakerError::StatisticsError(_) => ErrorSeverity::Minimal,
            CircuitBreakerError::EventError(_) => ErrorSeverity::Minimal,
            CircuitBreakerError::FailureDetectionError(_) => ErrorSeverity::Minimal,

            // Unknown severity for custom errors
            CircuitBreakerError::Custom(_) => ErrorSeverity::Medium,
        }
    }

    /// Get error category
    ///
    /// Returns a category that groups related error types for easier
    /// handling and analysis.
    ///
    /// # Returns
    /// - `ErrorCategory`: The category this error belongs to
    #[must_use]
    pub fn category(&self) -> ErrorCategory {
        match self {
            CircuitBreakerError::CircuitOpen => ErrorCategory::CircuitBreakerState,
            CircuitBreakerError::InvalidStateTransition(_) => ErrorCategory::CircuitBreakerState,

            CircuitBreakerError::Timeout => ErrorCategory::ServiceCommunication,
            CircuitBreakerError::ServiceUnavailable => ErrorCategory::ServiceCommunication,
            CircuitBreakerError::NetworkError(_) => ErrorCategory::ServiceCommunication,

            CircuitBreakerError::ConfigurationError(_) => ErrorCategory::Configuration,

            CircuitBreakerError::ExecutionError(_) => ErrorCategory::Execution,
            CircuitBreakerError::ResourceExhaustion(_) => ErrorCategory::Execution,
            CircuitBreakerError::ConcurrencyError(_) => ErrorCategory::Execution,

            CircuitBreakerError::RecoveryError(_) => ErrorCategory::InternalSystem,
            CircuitBreakerError::AnalyticsError(_) => ErrorCategory::InternalSystem,
            CircuitBreakerError::StatisticsError(_) => ErrorCategory::InternalSystem,
            CircuitBreakerError::EventError(_) => ErrorCategory::InternalSystem,
            CircuitBreakerError::FailureDetectionError(_) => ErrorCategory::InternalSystem,

            CircuitBreakerError::SerializationError(_) => ErrorCategory::DataProcessing,
            CircuitBreakerError::DatabaseError(_) => ErrorCategory::DataProcessing,

            CircuitBreakerError::AuthenticationError(_) => ErrorCategory::Security,
            CircuitBreakerError::AuthorizationError(_) => ErrorCategory::Security,

            CircuitBreakerError::RateLimitExceeded(_) => ErrorCategory::RateLimit,

            CircuitBreakerError::Custom(_) => ErrorCategory::Custom,
        }
    }

    /// Create a configuration error
    pub fn configuration_error(msg: impl Into<String>) -> Self {
        CircuitBreakerError::ConfigurationError(msg.into())
    }

    /// Create an execution error
    pub fn execution_error(msg: impl Into<String>) -> Self {
        CircuitBreakerError::ExecutionError(msg.into())
    }

    /// Create a recovery error
    pub fn recovery_error(msg: impl Into<String>) -> Self {
        CircuitBreakerError::RecoveryError(msg.into())
    }

    /// Create an analytics error
    pub fn analytics_error(msg: impl Into<String>) -> Self {
        CircuitBreakerError::AnalyticsError(msg.into())
    }

    /// Create a network error
    pub fn network_error(msg: impl Into<String>) -> Self {
        CircuitBreakerError::NetworkError(msg.into())
    }

    /// Create a custom error
    pub fn custom_error(msg: impl Into<String>) -> Self {
        CircuitBreakerError::Custom(msg.into())
    }
}

/// Error severity levels
///
/// Defines the severity of errors for prioritization and handling decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Minimal impact - informational only
    Minimal,
    /// Low impact - should be logged but doesn't require immediate action
    Low,
    /// Medium impact - should be addressed but not critical
    Medium,
    /// High impact - requires immediate attention
    High,
    /// Critical impact - system failure imminent
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Minimal => write!(f, "MINIMAL"),
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error categories for grouping related errors
///
/// Provides a way to group related error types for easier analysis
/// and handling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Circuit breaker state-related errors
    CircuitBreakerState,
    /// Service communication errors
    ServiceCommunication,
    /// Configuration-related errors
    Configuration,
    /// Execution-related errors
    Execution,
    /// Internal system errors
    InternalSystem,
    /// Data processing errors
    DataProcessing,
    /// Security-related errors
    Security,
    /// Rate limiting errors
    RateLimit,
    /// Custom application errors
    Custom,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::CircuitBreakerState => write!(f, "Circuit Breaker State"),
            ErrorCategory::ServiceCommunication => write!(f, "Service Communication"),
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::Execution => write!(f, "Execution"),
            ErrorCategory::InternalSystem => write!(f, "Internal System"),
            ErrorCategory::DataProcessing => write!(f, "Data Processing"),
            ErrorCategory::Security => write!(f, "Security"),
            ErrorCategory::RateLimit => write!(f, "Rate Limit"),
            ErrorCategory::Custom => write!(f, "Custom"),
        }
    }
}

// Conversion implementations for common error types

impl From<std::io::Error> for CircuitBreakerError {
    fn from(error: std::io::Error) -> Self {
        CircuitBreakerError::ExecutionError(error.to_string())
    }
}

impl From<serde_json::Error> for CircuitBreakerError {
    fn from(error: serde_json::Error) -> Self {
        CircuitBreakerError::SerializationError(error.to_string())
    }
}

impl From<std::fmt::Error> for CircuitBreakerError {
    fn from(error: std::fmt::Error) -> Self {
        CircuitBreakerError::SerializationError(error.to_string())
    }
}

impl<T> From<std::sync::PoisonError<T>> for CircuitBreakerError {
    fn from(_error: std::sync::PoisonError<T>) -> Self {
        CircuitBreakerError::ConcurrencyError("Lock poisoned".to_string())
    }
}

impl From<std::sync::mpsc::RecvError> for CircuitBreakerError {
    fn from(error: std::sync::mpsc::RecvError) -> Self {
        CircuitBreakerError::ConcurrencyError(error.to_string())
    }
}

impl<T> From<std::sync::mpsc::SendError<T>> for CircuitBreakerError {
    fn from(_error: std::sync::mpsc::SendError<T>) -> Self {
        CircuitBreakerError::ConcurrencyError("Channel send error".to_string())
    }
}

/// Result type alias for circuit breaker operations
pub type CircuitBreakerResult<T> = Result<T, CircuitBreakerError>;

/// Utility functions for error handling
pub mod utils {
    use super::{CircuitBreakerError, ErrorCategory, ErrorSeverity};

    /// Classify multiple errors by category
    pub fn classify_errors(
        errors: &[CircuitBreakerError],
    ) -> std::collections::HashMap<ErrorCategory, Vec<&CircuitBreakerError>> {
        let mut classified = std::collections::HashMap::new();

        for error in errors {
            classified
                .entry(error.category())
                .or_insert_with(Vec::new)
                .push(error);
        }

        classified
    }

    /// Get the highest severity from a collection of errors
    #[must_use]
    pub fn max_severity(errors: &[CircuitBreakerError]) -> Option<ErrorSeverity> {
        errors
            .iter()
            .map(super::CircuitBreakerError::severity)
            .max()
    }

    /// Count retryable vs non-retryable errors
    #[must_use]
    pub fn count_retryable(errors: &[CircuitBreakerError]) -> (usize, usize) {
        let retryable = errors.iter().filter(|e| e.is_retryable()).count();
        let non_retryable = errors.len() - retryable;
        (retryable, non_retryable)
    }

    /// Create an error summary for logging
    #[must_use]
    pub fn error_summary(error: &CircuitBreakerError) -> String {
        format!(
            "[{}] {} - Category: {}, Retryable: {}",
            error.severity(),
            error,
            error.category(),
            error.is_retryable()
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = CircuitBreakerError::CircuitOpen;
        assert!(error.to_string().contains("Circuit breaker is open"));
    }

    #[test]
    fn test_error_retryable() {
        assert!(CircuitBreakerError::CircuitOpen.is_retryable());
        assert!(CircuitBreakerError::Timeout.is_retryable());
        assert!(!CircuitBreakerError::ConfigurationError("test".to_string()).is_retryable());
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(
            CircuitBreakerError::ServiceUnavailable.severity(),
            ErrorSeverity::High
        );
        assert_eq!(
            CircuitBreakerError::CircuitOpen.severity(),
            ErrorSeverity::Medium
        );
        assert_eq!(
            CircuitBreakerError::AnalyticsError("test".to_string()).severity(),
            ErrorSeverity::Minimal
        );
    }

    #[test]
    fn test_error_category() {
        assert_eq!(
            CircuitBreakerError::CircuitOpen.category(),
            ErrorCategory::CircuitBreakerState
        );
        assert_eq!(
            CircuitBreakerError::NetworkError("test".to_string()).category(),
            ErrorCategory::ServiceCommunication
        );
        assert_eq!(
            CircuitBreakerError::ConfigurationError("test".to_string()).category(),
            ErrorCategory::Configuration
        );
    }

    #[test]
    fn test_error_conversions() {
        let io_error = std::io::Error::new(std::io::ErrorKind::Other, "test");
        let cb_error: CircuitBreakerError = io_error.into();
        match cb_error {
            CircuitBreakerError::ExecutionError(_) => {}
            _ => panic!("Expected ExecutionError"),
        }
    }

    #[test]
    fn test_utility_functions() {
        let errors = vec![
            CircuitBreakerError::CircuitOpen,
            CircuitBreakerError::ConfigurationError("test".to_string()),
            CircuitBreakerError::ServiceUnavailable,
        ];

        let classified = utils::classify_errors(&errors);
        assert!(classified.contains_key(&ErrorCategory::CircuitBreakerState));
        assert!(classified.contains_key(&ErrorCategory::Configuration));
        assert!(classified.contains_key(&ErrorCategory::ServiceCommunication));

        let max_sev = utils::max_severity(&errors);
        assert_eq!(max_sev, Some(ErrorSeverity::High));

        let (retryable, non_retryable) = utils::count_retryable(&errors);
        assert_eq!(retryable, 2); // CircuitOpen and ServiceUnavailable
        assert_eq!(non_retryable, 1); // ConfigurationError
    }
}
