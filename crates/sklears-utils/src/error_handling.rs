//! Enhanced error handling utilities for machine learning workflows
//!
//! This module provides comprehensive error handling with context, stack traces,
//! error aggregation, and structured error reporting.

use crate::{UtilsError, UtilsResult};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// ===== ENHANCED ERROR TYPES =====

/// Enhanced error with context and stack trace information
#[derive(Debug, Clone)]
pub struct EnhancedError {
    pub error: UtilsError,
    pub context: Vec<String>,
    pub stack_trace: Vec<String>,
    pub timestamp: std::time::SystemTime,
    pub error_id: String,
    pub metadata: HashMap<String, String>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: UtilsError) -> Self {
        Self {
            error,
            context: Vec::new(),
            stack_trace: Self::capture_stack_trace(),
            timestamp: std::time::SystemTime::now(),
            error_id: Self::generate_error_id(),
            metadata: HashMap::new(),
        }
    }

    /// Add context to the error
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context.push(context.into());
        self
    }

    /// Add metadata to the error
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Generate a unique error ID
    fn generate_error_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("ERR-{timestamp:016x}")
    }

    /// Capture stack trace (simplified implementation)
    fn capture_stack_trace() -> Vec<String> {
        // In a real implementation, you might use backtrace crate
        // For now, we'll create a simple stack trace
        vec![
            "stack_trace: enhanced_error.rs:capture_stack_trace".to_string(),
            "stack_trace: error_handling.rs:new".to_string(),
        ]
    }

    /// Format the error for display
    pub fn format_detailed(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("Error ID: {}\n", self.error_id));
        output.push_str(&format!("Timestamp: {:?}\n", self.timestamp));
        output.push_str(&format!("Error: {}\n", self.error));

        if !self.context.is_empty() {
            output.push_str("Context:\n");
            for (i, ctx) in self.context.iter().enumerate() {
                output.push_str(&format!("  {}: {ctx}\n", i + 1));
            }
        }

        if !self.metadata.is_empty() {
            output.push_str("Metadata:\n");
            for (key, value) in &self.metadata {
                output.push_str(&format!("  {key}: {value}\n"));
            }
        }

        if !self.stack_trace.is_empty() {
            output.push_str("Stack Trace:\n");
            for frame in &self.stack_trace {
                output.push_str(&format!("  {frame}\n"));
            }
        }

        output
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (ID: {})", self.error, self.error_id)
    }
}

impl std::error::Error for EnhancedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// ===== ERROR CONTEXT BUILDER =====

/// Builder for adding context to errors
pub struct ErrorContext {
    operation: String,
    parameters: HashMap<String, String>,
    location: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new<S: Into<String>>(operation: S) -> Self {
        Self {
            operation: operation.into(),
            parameters: HashMap::new(),
            location: None,
        }
    }

    /// Add a parameter to the context
    pub fn with_param<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Add location information
    pub fn at_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Wrap an error with this context
    pub fn wrap_error<E: Into<UtilsError>>(self, error: E) -> EnhancedError {
        let mut enhanced = EnhancedError::new(error.into());

        enhanced = enhanced.with_context(format!("Operation: {}", self.operation));

        if let Some(location) = self.location {
            enhanced = enhanced.with_context(format!("Location: {location}"));
        }

        for (key, value) in self.parameters {
            enhanced = enhanced.with_metadata(key, value);
        }

        enhanced
    }
}

// ===== ERROR AGGREGATION =====

/// Aggregates multiple errors into a single report
#[derive(Debug, Clone)]
pub struct ErrorAggregator {
    errors: Vec<EnhancedError>,
    max_errors: usize,
    continue_on_error: bool,
}

impl ErrorAggregator {
    /// Create a new error aggregator
    pub fn new(max_errors: usize, continue_on_error: bool) -> Self {
        Self {
            errors: Vec::new(),
            max_errors,
            continue_on_error,
        }
    }

    /// Add an error to the aggregator
    pub fn add_error(&mut self, error: EnhancedError) -> UtilsResult<()> {
        self.errors.push(error);

        if self.errors.len() >= self.max_errors {
            if self.continue_on_error {
                // Remove oldest error to make room
                self.errors.remove(0);
            } else {
                return Err(UtilsError::InvalidParameter(
                    "Maximum error count reached".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get the number of errors
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get all errors
    pub fn get_errors(&self) -> &[EnhancedError] {
        &self.errors
    }

    /// Clear all errors
    pub fn clear(&mut self) {
        self.errors.clear();
    }

    /// Generate an error summary
    pub fn generate_summary(&self) -> ErrorSummary {
        let mut summary = ErrorSummary::default();

        for error in &self.errors {
            summary.total_errors += 1;

            match &error.error {
                UtilsError::ShapeMismatch { .. } => summary.shape_errors += 1,
                UtilsError::InvalidParameter(_) => summary.parameter_errors += 1,
                UtilsError::EmptyInput => summary.input_errors += 1,
                UtilsError::InvalidRandomState(_) => summary.random_state_errors += 1,
                UtilsError::InsufficientData { .. } => summary.data_errors += 1,
            }
        }

        summary
    }

    /// Export errors to a structured format
    pub fn export_errors(&self) -> Vec<HashMap<String, String>> {
        self.errors
            .iter()
            .map(|error| {
                let mut export = HashMap::new();
                export.insert("id".to_string(), error.error_id.clone());
                export.insert("error".to_string(), error.error.to_string());
                export.insert("timestamp".to_string(), format!("{:?}", error.timestamp));
                export.insert("context".to_string(), error.context.join("; "));

                for (key, value) in &error.metadata {
                    export.insert(format!("meta_{key}"), value.clone());
                }

                export
            })
            .collect()
    }
}

/// Summary of aggregated errors
#[derive(Debug, Default, Clone)]
pub struct ErrorSummary {
    pub total_errors: usize,
    pub shape_errors: usize,
    pub parameter_errors: usize,
    pub input_errors: usize,
    pub random_state_errors: usize,
    pub data_errors: usize,
}

// ===== ERROR RECOVERY STRATEGIES =====

/// Error recovery strategies for common ML scenarios
pub struct ErrorRecovery;

impl ErrorRecovery {
    /// Attempt to recover from a shape mismatch error
    pub fn recover_shape_mismatch(expected: &[usize], actual: &[usize]) -> Option<Vec<usize>> {
        // Try to suggest a compatible shape
        if expected.len() == actual.len() {
            // Same number of dimensions, might be able to reshape
            let expected_size: usize = expected.iter().product();
            let actual_size: usize = actual.iter().product();

            if expected_size == actual_size {
                return Some(expected.to_vec());
            }
        }

        // Try to add/remove dimensions
        if expected.len() == 1 && actual.len() == 2 {
            // Flatten 2D to 1D
            let total_size: usize = actual.iter().product();
            return Some(vec![total_size]);
        }

        if expected.len() == 2 && actual.len() == 1 {
            // Reshape 1D to 2D
            let size = actual[0];
            // Try to find reasonable 2D shape
            for i in 1..=(size as f64).sqrt() as usize + 1 {
                if size % i == 0 {
                    return Some(vec![i, size / i]);
                }
            }
        }

        None
    }

    /// Attempt to recover from insufficient data
    pub fn recover_insufficient_data(
        required: usize,
        available: usize,
    ) -> Option<RecoveryStrategy> {
        if available == 0 {
            return Some(RecoveryStrategy::GenerateSyntheticData(required));
        }

        if available < required {
            if available >= required / 2 {
                return Some(RecoveryStrategy::ReduceRequirement(available));
            } else {
                return Some(RecoveryStrategy::AugmentData(required - available));
            }
        }

        None
    }

    /// Attempt to fix invalid parameters
    pub fn recover_invalid_parameter(param_name: &str, param_value: &str) -> Option<String> {
        match param_name {
            "n_components" | "n_clusters" | "max_iter" => {
                // Try to parse as number and ensure it's positive
                if let Ok(val) = param_value.parse::<i32>() {
                    if val <= 0 {
                        return Some("1".to_string());
                    }
                }
                Some("10".to_string())
            }
            "random_state" => {
                // Provide a default random state
                Some("42".to_string())
            }
            "tolerance" | "alpha" | "learning_rate" => {
                // Ensure positive float
                if let Ok(val) = param_value.parse::<f64>() {
                    if val <= 0.0 {
                        return Some("0.01".to_string());
                    }
                }
                Some("0.01".to_string())
            }
            _ => None,
        }
    }
}

/// Recovery strategy recommendations
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    GenerateSyntheticData(usize),
    ReduceRequirement(usize),
    AugmentData(usize),
    ReshapeData(Vec<usize>),
    UseDefaultParameter(String),
}

// ===== ERROR REPORTING =====

/// Global error reporter for collecting and analyzing errors
pub struct ErrorReporter {
    errors: Arc<Mutex<Vec<EnhancedError>>>,
    enabled: bool,
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorReporter {
    /// Create a new error reporter
    pub fn new() -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
            enabled: true,
        }
    }

    /// Enable or disable error reporting
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Report an error
    pub fn report_error(&self, error: EnhancedError) {
        if !self.enabled {
            return;
        }

        if let Ok(mut errors) = self.errors.lock() {
            errors.push(error);

            // Keep only the last 1000 errors to prevent memory issues
            if errors.len() > 1000 {
                errors.remove(0);
            }
        }
    }

    /// Get error statistics
    pub fn get_statistics(&self) -> Option<ErrorStatistics> {
        let errors = self.errors.lock().ok()?;

        if errors.is_empty() {
            return None;
        }

        let mut stats = ErrorStatistics {
            total_errors: errors.len(),
            ..Default::default()
        };

        for error in errors.iter() {
            match &error.error {
                UtilsError::ShapeMismatch { .. } => stats.shape_errors += 1,
                UtilsError::InvalidParameter(_) => stats.parameter_errors += 1,
                UtilsError::EmptyInput => stats.input_errors += 1,
                UtilsError::InvalidRandomState(_) => stats.random_state_errors += 1,
                UtilsError::InsufficientData { .. } => stats.data_errors += 1,
            }
        }

        // Calculate error frequency over time windows
        let now = std::time::SystemTime::now();
        let one_hour_ago = now - std::time::Duration::from_secs(3600);
        let one_day_ago = now - std::time::Duration::from_secs(86400);

        stats.errors_last_hour = errors.iter().filter(|e| e.timestamp > one_hour_ago).count();

        stats.errors_last_day = errors.iter().filter(|e| e.timestamp > one_day_ago).count();

        Some(stats)
    }

    /// Clear all reported errors
    pub fn clear(&self) {
        if let Ok(mut errors) = self.errors.lock() {
            errors.clear();
        }
    }
}

/// Error statistics
#[derive(Debug, Default, Clone)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub shape_errors: usize,
    pub parameter_errors: usize,
    pub input_errors: usize,
    pub random_state_errors: usize,
    pub data_errors: usize,
    pub errors_last_hour: usize,
    pub errors_last_day: usize,
}

// ===== CONVENIENCE MACROS AND FUNCTIONS =====

/// Create an enhanced error with context
pub fn create_error<E: Into<UtilsError>>(error: E, operation: &str) -> EnhancedError {
    ErrorContext::new(operation).wrap_error(error)
}

/// Create an enhanced error with context and location
pub fn create_error_at<E: Into<UtilsError>>(
    error: E,
    operation: &str,
    location: &str,
) -> EnhancedError {
    ErrorContext::new(operation)
        .at_location(location)
        .wrap_error(error)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_error() {
        let base_error = UtilsError::InvalidParameter("test".to_string());
        let enhanced = EnhancedError::new(base_error)
            .with_context("Processing data")
            .with_metadata("operation", "test_operation");

        assert!(!enhanced.error_id.is_empty());
        assert_eq!(enhanced.context.len(), 1);
        assert_eq!(enhanced.metadata.len(), 1);

        let formatted = enhanced.format_detailed();
        assert!(formatted.contains("Error ID:"));
        assert!(formatted.contains("Processing data"));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation")
            .with_param("param1", "value1")
            .at_location("test_file.rs:123");

        let base_error = UtilsError::EmptyInput;
        let enhanced = context.wrap_error(base_error);

        assert!(enhanced.context.len() >= 2);
        assert!(enhanced.metadata.contains_key("param1"));
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new(3, false);

        assert!(!aggregator.has_errors());
        assert_eq!(aggregator.error_count(), 0);

        let error1 = EnhancedError::new(UtilsError::EmptyInput);
        let error2 = EnhancedError::new(UtilsError::InvalidParameter("test".to_string()));

        aggregator.add_error(error1).unwrap();
        aggregator.add_error(error2).unwrap();

        assert!(aggregator.has_errors());
        assert_eq!(aggregator.error_count(), 2);

        let summary = aggregator.generate_summary();
        assert_eq!(summary.total_errors, 2);
        assert_eq!(summary.input_errors, 1);
        assert_eq!(summary.parameter_errors, 1);
    }

    #[test]
    fn test_error_recovery() {
        // Test shape mismatch recovery
        let recovery = ErrorRecovery::recover_shape_mismatch(&[10], &[2, 5]);
        assert_eq!(recovery, Some(vec![10]));

        // Test insufficient data recovery
        let strategy = ErrorRecovery::recover_insufficient_data(100, 50);
        match strategy {
            Some(RecoveryStrategy::ReduceRequirement(50)) => (),
            _ => panic!("Expected ReduceRequirement strategy"),
        }

        // Test parameter recovery
        let fixed = ErrorRecovery::recover_invalid_parameter("n_clusters", "-5");
        assert_eq!(fixed, Some("1".to_string()));
    }

    #[test]
    fn test_error_reporter() {
        let reporter = ErrorReporter::new();
        let error = EnhancedError::new(UtilsError::EmptyInput);

        reporter.report_error(error);

        let stats = reporter.get_statistics().unwrap();
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.input_errors, 1);
    }

    #[test]
    fn test_convenience_functions() {
        let error = create_error(UtilsError::EmptyInput, "test_operation");
        assert!(error.context.iter().any(|c| c.contains("test_operation")));

        let error_with_location =
            create_error_at(UtilsError::EmptyInput, "test_operation", "test_file.rs:123");
        assert!(error_with_location.context.len() >= 2);
    }
}
