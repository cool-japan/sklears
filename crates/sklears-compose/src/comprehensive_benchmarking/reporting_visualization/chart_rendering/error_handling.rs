//! Error handling and recovery systems
//!
//! This module provides comprehensive error handling and recovery capabilities including:
//! - Chart building and rendering error types
//! - Error recovery strategies and fallback mechanisms
//! - Error reporting and logging configuration
//! - Retry mechanisms with backoff and jitter
//! - Error metrics integration and monitoring

use serde::{Serialize, Deserialize};
use scirs2_core::random::thread_rng;
use chrono::Duration;

/// Primary error type for chart building and rendering operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartBuildError {
    /// Invalid configuration provided to chart builder
    InvalidConfiguration(String),
    /// Missing required property in chart configuration
    MissingProperty(String),
    /// Unsupported chart type requested
    UnsupportedChartType(String),
    /// Data format error during processing
    DataFormatError(String),
    /// Rendering engine error during chart generation
    RenderingError(String),
    /// Custom error with user-defined message
    Custom(String),
}

/// Comprehensive error handling configuration for rendering system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingErrorHandling {
    /// Strategy for error recovery
    pub recovery_strategy: ErrorRecoveryStrategy,
    /// Optional fallback renderer for error situations
    pub fallback_renderer: Option<String>,
    /// Error reporting and logging configuration
    pub error_reporting: ErrorReportingConfig,
    /// Retry mechanism configuration
    pub retry_config: RetryConfig,
}

/// Error recovery strategies for different failure scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorRecoveryStrategy {
    /// Fail immediately without recovery attempt
    FailFast,
    /// Attempt graceful degradation with reduced functionality
    GracefulDegradation,
    /// Retry operation with exponential backoff
    RetryWithBackoff,
    /// Switch to configured fallback renderer
    SwitchToFallback,
    /// Custom recovery strategy with user-defined logic
    Custom(String),
}

/// Error reporting and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReportingConfig {
    /// Enable error reporting system
    pub enabled: bool,
    /// Minimum logging level for error messages
    pub logging_level: ErrorLoggingLevel,
    /// Enable collection of error metrics
    pub metrics_collection: bool,
    /// Enable error notifications to external systems
    pub notifications: bool,
}

/// Logging levels for error classification and filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorLoggingLevel {
    /// Debug level for development and troubleshooting
    Debug,
    /// Info level for general information
    Info,
    /// Warn level for warning conditions
    Warn,
    /// Error level for error conditions
    Error,
    /// Critical level for critical failures
    Critical,
}

/// Retry configuration for error recovery mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay between retry attempts
    pub retry_delay: Duration,
    /// Enable exponential backoff for retry delays
    pub exponential_backoff: bool,
    /// Add jitter to retry timing to avoid thundering herd
    pub jitter: bool,
}

/// Error context information for debugging and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Component where error occurred
    pub component: String,
    /// Operation being performed when error occurred
    pub operation: String,
    /// Additional context information
    pub context_data: std::collections::HashMap<String, String>,
}

/// Error severity classification for prioritization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity - minor issues that don't affect functionality
    Low,
    /// Medium severity - issues that may impact some functionality
    Medium,
    /// High severity - significant issues affecting core functionality
    High,
    /// Critical severity - system-critical failures requiring immediate attention
    Critical,
}

/// Structured error information for comprehensive error tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Primary error
    pub error: ChartBuildError,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error context information
    pub context: ErrorContext,
    /// Recovery attempts made
    pub recovery_attempts: u32,
    /// Whether error was recovered
    pub recovered: bool,
}

/// Error handler trait for custom error processing
pub trait ErrorHandler {
    /// Handle an error with context information
    fn handle_error(&self, error: &ErrorInfo) -> Result<(), ChartBuildError>;

    /// Determine if error should trigger retry
    fn should_retry(&self, error: &ChartBuildError, attempt: u32) -> bool;

    /// Calculate retry delay based on attempt number
    fn calculate_retry_delay(&self, attempt: u32, base_delay: Duration) -> Duration;
}

/// Default error handler implementation
#[derive(Debug, Clone)]
pub struct DefaultErrorHandler {
    /// Configuration for error handling
    pub config: RenderingErrorHandling,
}

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&self, error: &ErrorInfo) -> Result<(), ChartBuildError> {
        match self.config.recovery_strategy {
            ErrorRecoveryStrategy::FailFast => Err(error.error.clone()),
            ErrorRecoveryStrategy::GracefulDegradation => {
                // Log error and continue with degraded functionality
                Ok(())
            }
            ErrorRecoveryStrategy::RetryWithBackoff => {
                if self.should_retry(&error.error, error.recovery_attempts) {
                    Ok(())
                } else {
                    Err(error.error.clone())
                }
            }
            ErrorRecoveryStrategy::SwitchToFallback => {
                // Attempt to switch to fallback renderer
                match &self.config.fallback_renderer {
                    Some(_) => Ok(()),
                    None => Err(error.error.clone()),
                }
            }
            ErrorRecoveryStrategy::Custom(_) => {
                // Custom recovery logic would be implemented here
                Ok(())
            }
        }
    }

    fn should_retry(&self, _error: &ChartBuildError, attempt: u32) -> bool {
        attempt < self.config.retry_config.max_attempts
    }

    fn calculate_retry_delay(&self, attempt: u32, base_delay: Duration) -> Duration {
        let mut delay = base_delay;

        if self.config.retry_config.exponential_backoff {
            // Apply exponential backoff
            let multiplier = 2_u32.pow(attempt.saturating_sub(1));
            delay = delay * multiplier.min(64); // Cap at 64x base delay
        }

        if self.config.retry_config.jitter {
            // Add up to 25% jitter to avoid thundering herd
            let jitter_ms = (delay.num_milliseconds() as f64 * 0.25 * thread_rng().gen::<f64>()) as i64;
            delay = delay + Duration::milliseconds(jitter_ms);
        }

        delay
    }
}

impl std::fmt::Display for ChartBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChartBuildError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            ChartBuildError::MissingProperty(prop) => {
                write!(f, "Missing required property: {}", prop)
            }
            ChartBuildError::UnsupportedChartType(chart_type) => {
                write!(f, "Unsupported chart type: {}", chart_type)
            }
            ChartBuildError::DataFormatError(msg) => {
                write!(f, "Data format error: {}", msg)
            }
            ChartBuildError::RenderingError(msg) => {
                write!(f, "Rendering error: {}", msg)
            }
            ChartBuildError::Custom(msg) => {
                write!(f, "Custom error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ChartBuildError {}

impl Default for RenderingErrorHandling {
    fn default() -> Self {
        Self {
            recovery_strategy: ErrorRecoveryStrategy::RetryWithBackoff,
            fallback_renderer: None,
            error_reporting: ErrorReportingConfig::default(),
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for ErrorReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            logging_level: ErrorLoggingLevel::Error,
            metrics_collection: true,
            notifications: false,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            retry_delay: Duration::milliseconds(1000),
            exponential_backoff: true,
            jitter: true,
        }
    }
}

impl Default for DefaultErrorHandler {
    fn default() -> Self {
        Self {
            config: RenderingErrorHandling::default(),
        }
    }
}