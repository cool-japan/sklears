//! Dashboard error types and error handling
//!
//! This module provides comprehensive error types for dashboard operations including:
//! - Dashboard-specific error classifications
//! - Template and widget error handling
//! - Theme and styling error management
//! - Export and distribution error types
//! - Authentication and permission error handling
//! - I/O and configuration error management

use serde::{Serialize, Deserialize};

/// Dashboard error types for comprehensive
/// error handling and debugging
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
pub enum DashboardError {
    /// Dashboard not found error
    #[error("Dashboard not found: {0}")]
    DashboardNotFound(String),

    /// Template not found error
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    /// Widget not found error
    #[error("Widget not found: {0}")]
    WidgetNotFound(String),

    /// Theme not found error
    #[error("Theme not found: {0}")]
    ThemeNotFound(String),

    /// Distribution channel not found error
    #[error("Channel not found: {0}")]
    ChannelNotFound(String),

    /// Export operation failed error
    #[error("Export failed: {0}")]
    ExportFailed(String),

    /// Distribution operation failed error
    #[error("Distribution failed: {0}")]
    DistributionFailed(String),

    /// Authentication failed error
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Permission denied error
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Network connection error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Database operation error
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Cache operation error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// File system operation error
    #[error("File system error: {0}")]
    FileSystemError(String),

    /// Template compilation error
    #[error("Template compilation error: {0}")]
    TemplateCompilationError(String),

    /// Rendering engine error
    #[error("Rendering engine error: {0}")]
    RenderingEngineError(String),

    /// WebSocket connection error
    #[error("WebSocket error: {0}")]
    WebSocketError(String),

    /// Rate limit exceeded error
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Resource not available error
    #[error("Resource not available: {0}")]
    ResourceNotAvailable(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    TimeoutError(String),

    /// Concurrent modification error
    #[error("Concurrent modification detected: {0}")]
    ConcurrentModificationError(String),

    /// Quota exceeded error
    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    /// Service unavailable error
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Invalid state error
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Dependency error
    #[error("Dependency error: {0}")]
    DependencyError(String),

    /// I/O error wrapper
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Generic error for custom error types
    #[error("Generic error: {0}")]
    Generic(String),
}

/// Error context for enhanced error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error code for programmatic handling
    pub error_code: String,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Component or module where error occurred
    pub component: String,
    /// Operation being performed when error occurred
    pub operation: String,
    /// User identifier if applicable
    pub user_id: Option<String>,
    /// Session identifier if applicable
    pub session_id: Option<String>,
    /// Additional context information
    pub metadata: std::collections::HashMap<String, String>,
}

/// Error severity levels for error classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity - informational or minor issues
    Low,
    /// Medium severity - warnings or recoverable errors
    Medium,
    /// High severity - errors that affect functionality
    High,
    /// Critical severity - system-critical errors
    Critical,
}

/// Structured error information for comprehensive error tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// The actual error
    pub error: DashboardError,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error context information
    pub context: ErrorContext,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
    /// Related error IDs for error correlation
    pub related_errors: Vec<String>,
}

/// Error recovery action for automated error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorRecoveryAction {
    /// Retry the operation
    Retry,
    /// Fall back to alternative method
    Fallback,
    /// Skip the operation
    Skip,
    /// Abort the entire process
    Abort,
    /// Custom recovery action
    Custom(String),
}

/// Error handler trait for custom error processing
pub trait ErrorHandler {
    /// Handle an error with context
    fn handle_error(&self, error_info: &ErrorInfo) -> Result<ErrorRecoveryAction, DashboardError>;

    /// Determine error severity
    fn determine_severity(&self, error: &DashboardError) -> ErrorSeverity;

    /// Generate recovery suggestions
    fn generate_recovery_suggestions(&self, error: &DashboardError) -> Vec<String>;
}

/// Default error handler implementation
#[derive(Debug, Clone)]
pub struct DefaultErrorHandler {
    /// Enable automatic retry for recoverable errors
    pub auto_retry_enabled: bool,
    /// Maximum retry attempts
    pub max_retry_attempts: u32,
    /// Error logging enabled
    pub logging_enabled: bool,
}

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&self, error_info: &ErrorInfo) -> Result<ErrorRecoveryAction, DashboardError> {
        match &error_info.error {
            DashboardError::NetworkError(_) |
            DashboardError::TimeoutError(_) => {
                if self.auto_retry_enabled {
                    Ok(ErrorRecoveryAction::Retry)
                } else {
                    Ok(ErrorRecoveryAction::Abort)
                }
            }
            DashboardError::DashboardNotFound(_) |
            DashboardError::TemplateNotFound(_) |
            DashboardError::WidgetNotFound(_) => {
                Ok(ErrorRecoveryAction::Skip)
            }
            DashboardError::PermissionDenied(_) |
            DashboardError::AuthenticationFailed(_) => {
                Ok(ErrorRecoveryAction::Abort)
            }
            _ => Ok(ErrorRecoveryAction::Fallback)
        }
    }

    fn determine_severity(&self, error: &DashboardError) -> ErrorSeverity {
        match error {
            DashboardError::AuthenticationFailed(_) |
            DashboardError::PermissionDenied(_) |
            DashboardError::ServiceUnavailable(_) => ErrorSeverity::Critical,

            DashboardError::ExportFailed(_) |
            DashboardError::DistributionFailed(_) |
            DashboardError::RenderingEngineError(_) => ErrorSeverity::High,

            DashboardError::ValidationError(_) |
            DashboardError::ConfigurationError(_) |
            DashboardError::TimeoutError(_) => ErrorSeverity::Medium,

            DashboardError::DashboardNotFound(_) |
            DashboardError::TemplateNotFound(_) |
            DashboardError::WidgetNotFound(_) => ErrorSeverity::Low,

            _ => ErrorSeverity::Medium,
        }
    }

    fn generate_recovery_suggestions(&self, error: &DashboardError) -> Vec<String> {
        match error {
            DashboardError::DashboardNotFound(_) => vec![
                "Check if the dashboard ID is correct".to_string(),
                "Verify that the dashboard exists".to_string(),
                "Try refreshing the dashboard list".to_string(),
            ],
            DashboardError::AuthenticationFailed(_) => vec![
                "Check your username and password".to_string(),
                "Verify your account is not locked".to_string(),
                "Contact system administrator if issue persists".to_string(),
            ],
            DashboardError::PermissionDenied(_) => vec![
                "Contact administrator to request access".to_string(),
                "Verify you have the required permissions".to_string(),
                "Check if your role allows this operation".to_string(),
            ],
            DashboardError::NetworkError(_) => vec![
                "Check your network connection".to_string(),
                "Verify server is accessible".to_string(),
                "Try again in a few moments".to_string(),
            ],
            _ => vec![
                "Try the operation again".to_string(),
                "Check system logs for more details".to_string(),
                "Contact support if problem persists".to_string(),
            ],
        }
    }
}

impl Default for DefaultErrorHandler {
    fn default() -> Self {
        Self {
            auto_retry_enabled: true,
            max_retry_attempts: 3,
            logging_enabled: true,
        }
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            error_code: "UNKNOWN".to_string(),
            timestamp: chrono::Utc::now(),
            component: "unknown".to_string(),
            operation: "unknown".to_string(),
            user_id: None,
            session_id: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Utility functions for error handling
impl DashboardError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            DashboardError::NetworkError(_) |
            DashboardError::TimeoutError(_) |
            DashboardError::ServiceUnavailable(_) |
            DashboardError::RateLimitExceeded(_)
        )
    }

    /// Check if error is temporary
    pub fn is_temporary(&self) -> bool {
        matches!(
            self,
            DashboardError::NetworkError(_) |
            DashboardError::TimeoutError(_) |
            DashboardError::ServiceUnavailable(_) |
            DashboardError::RateLimitExceeded(_) |
            DashboardError::ResourceNotAvailable(_)
        )
    }

    /// Get error category
    pub fn category(&self) -> &'static str {
        match self {
            DashboardError::DashboardNotFound(_) |
            DashboardError::TemplateNotFound(_) |
            DashboardError::WidgetNotFound(_) |
            DashboardError::ThemeNotFound(_) |
            DashboardError::ChannelNotFound(_) => "NotFound",

            DashboardError::AuthenticationFailed(_) |
            DashboardError::PermissionDenied(_) => "Security",

            DashboardError::ValidationError(_) |
            DashboardError::ConfigurationError(_) => "Validation",

            DashboardError::NetworkError(_) |
            DashboardError::WebSocketError(_) => "Network",

            DashboardError::ExportFailed(_) |
            DashboardError::DistributionFailed(_) => "Export",

            DashboardError::IoError(_) |
            DashboardError::FileSystemError(_) => "IO",

            _ => "General",
        }
    }
}