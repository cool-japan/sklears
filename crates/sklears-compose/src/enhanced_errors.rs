//! Enhanced Error Handling with Actionable Suggestions
//!
//! This module provides comprehensive error handling for pipeline composition
//! with detailed error messages, actionable suggestions, and debugging context.

use sklears_core::prelude::SklearsError;
use std::collections::HashMap;
use std::fmt;

/// Enhanced error types specific to pipeline composition
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// Configuration errors with suggestions
    ConfigurationError {
        message: String,
        suggestions: Vec<String>,
        context: ErrorContext,
    },
    /// Data compatibility issues
    DataCompatibilityError {
        expected: DataShape,
        actual: DataShape,
        stage: String,
        suggestions: Vec<String>,
    },
    /// Pipeline structure errors
    StructureError {
        error_type: StructureErrorType,
        affected_components: Vec<String>,
        suggestions: Vec<String>,
    },
    /// Performance warnings that may impact execution
    PerformanceWarning {
        warning_type: PerformanceWarningType,
        impact_level: ImpactLevel,
        suggestions: Vec<String>,
        metrics: Option<PerformanceMetrics>,
    },
    /// Resource constraint violations
    ResourceError {
        resource_type: ResourceType,
        limit: f64,
        current: f64,
        component: String,
        suggestions: Vec<String>,
    },
    /// Type safety violations in pipeline composition
    TypeSafetyError {
        violation_type: TypeViolationType,
        expected_type: String,
        actual_type: String,
        stage: String,
        suggestions: Vec<String>,
    },
}

/// Error context information for debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub pipeline_stage: String,
    pub component_name: String,
    pub input_shape: Option<(usize, usize)>,
    pub parameters: HashMap<String, String>,
    pub stack_trace: Vec<String>,
}

/// Data shape information for compatibility checking
#[derive(Debug, Clone, PartialEq)]
pub struct DataShape {
    pub samples: usize,
    pub features: usize,
    pub data_type: String,
    pub missing_values: bool,
}

/// Types of pipeline structure errors
#[derive(Debug, Clone)]
pub enum StructureErrorType {
    /// CyclicDependency
    CyclicDependency,
    /// MissingComponent
    MissingComponent,
    /// InvalidConnection
    InvalidConnection,
    /// DanglingNode
    DanglingNode,
    /// InconsistentFlow
    InconsistentFlow,
}

/// Performance warning categories
#[derive(Debug, Clone)]
pub enum PerformanceWarningType {
    /// MemoryUsage
    MemoryUsage,
    /// ComputationalComplexity
    ComputationalComplexity,
    /// NetworkBottleneck
    NetworkBottleneck,
    /// CacheInefficiency
    CacheInefficiency,
    /// SuboptimalConfiguration
    SuboptimalConfiguration,
}

/// Impact levels for warnings and errors
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImpactLevel {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

/// Resource types for constraint checking
#[derive(Debug, Clone)]
pub enum ResourceType {
    /// Memory
    Memory,
    /// CPU
    CPU,
    /// GPU
    GPU,
    /// Disk
    Disk,
    /// NetworkBandwidth
    NetworkBandwidth,
}

/// Performance metrics for context
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub cache_hit_ratio: f64,
}

/// Type safety violation categories
#[derive(Debug, Clone)]
pub enum TypeViolationType {
    /// IncompatibleInputType
    IncompatibleInputType,
    /// MismatchedOutputType
    MismatchedOutputType,
    /// UnsupportedTransformation
    UnsupportedTransformation,
    /// InvalidParameterType
    InvalidParameterType,
}

/// Enhanced error builder for creating detailed error messages
pub struct EnhancedErrorBuilder {
    error_type: Option<PipelineError>,
    suggestions: Vec<String>,
    context: Option<ErrorContext>,
}

impl Default for EnhancedErrorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedErrorBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            error_type: None,
            suggestions: Vec::new(),
            context: None,
        }
    }

    /// Create configuration error with detailed suggestions
    #[must_use]
    pub fn configuration_error(mut self, message: &str) -> Self {
        let suggestions = self.generate_configuration_suggestions(message);
        self.error_type = Some(PipelineError::ConfigurationError {
            message: message.to_string(),
            suggestions,
            context: self.context.clone().unwrap_or_default(),
        });
        self
    }

    /// Create data compatibility error with shape analysis
    #[must_use]
    pub fn data_compatibility_error(
        mut self,
        expected: DataShape,
        actual: DataShape,
        stage: &str,
    ) -> Self {
        let suggestions = self.generate_compatibility_suggestions(&expected, &actual, stage);
        self.error_type = Some(PipelineError::DataCompatibilityError {
            expected,
            actual,
            stage: stage.to_string(),
            suggestions,
        });
        self
    }

    /// Create structure error with component analysis
    #[must_use]
    pub fn structure_error(
        mut self,
        error_type: StructureErrorType,
        affected_components: Vec<String>,
    ) -> Self {
        let suggestions = self.generate_structure_suggestions(&error_type, &affected_components);
        self.error_type = Some(PipelineError::StructureError {
            error_type,
            affected_components,
            suggestions,
        });
        self
    }

    /// Create performance warning with optimization suggestions
    #[must_use]
    pub fn performance_warning(
        mut self,
        warning_type: PerformanceWarningType,
        impact_level: ImpactLevel,
        metrics: Option<PerformanceMetrics>,
    ) -> Self {
        let suggestions =
            self.generate_performance_suggestions(&warning_type, &impact_level, &metrics);
        self.error_type = Some(PipelineError::PerformanceWarning {
            warning_type,
            impact_level,
            suggestions,
            metrics,
        });
        self
    }

    /// Create resource error with scaling suggestions
    #[must_use]
    pub fn resource_error(
        mut self,
        resource_type: ResourceType,
        limit: f64,
        current: f64,
        component: &str,
    ) -> Self {
        let suggestions =
            self.generate_resource_suggestions(&resource_type, limit, current, component);
        self.error_type = Some(PipelineError::ResourceError {
            resource_type,
            limit,
            current,
            component: component.to_string(),
            suggestions,
        });
        self
    }

    /// Create type safety error with type conversion suggestions
    #[must_use]
    pub fn type_safety_error(
        mut self,
        violation_type: TypeViolationType,
        expected_type: &str,
        actual_type: &str,
        stage: &str,
    ) -> Self {
        let suggestions = self.generate_type_safety_suggestions(
            &violation_type,
            expected_type,
            actual_type,
            stage,
        );
        self.error_type = Some(PipelineError::TypeSafetyError {
            violation_type,
            expected_type: expected_type.to_string(),
            actual_type: actual_type.to_string(),
            stage: stage.to_string(),
            suggestions,
        });
        self
    }

    /// Add custom suggestion
    #[must_use]
    pub fn suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    /// Add error context
    #[must_use]
    pub fn context(mut self, context: ErrorContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Build the enhanced error
    #[must_use]
    pub fn build(self) -> PipelineError {
        // Add any custom suggestions to the error and update context
        if let Some(mut error) = self.error_type {
            // Add suggestions to all error types
            match &mut error {
                PipelineError::ConfigurationError { suggestions, .. } => {
                    suggestions.extend(self.suggestions);
                }
                PipelineError::DataCompatibilityError { suggestions, .. } => {
                    suggestions.extend(self.suggestions);
                }
                PipelineError::StructureError { suggestions, .. } => {
                    suggestions.extend(self.suggestions);
                }
                PipelineError::PerformanceWarning { suggestions, .. } => {
                    suggestions.extend(self.suggestions);
                }
                PipelineError::ResourceError { suggestions, .. } => {
                    suggestions.extend(self.suggestions);
                }
                PipelineError::TypeSafetyError { suggestions, .. } => {
                    suggestions.extend(self.suggestions);
                }
            }
            // Update context for ConfigurationError specifically
            if let PipelineError::ConfigurationError { context, .. } = &mut error {
                if let Some(new_context) = self.context {
                    *context = new_context;
                }
            }
            error
        } else {
            // Default error if none specified
            PipelineError::ConfigurationError {
                message: "Unknown pipeline error".to_string(),
                suggestions: self.suggestions,
                context: self.context.unwrap_or_default(),
            }
        }
    }

    /// Generate configuration-specific suggestions
    fn generate_configuration_suggestions(&self, message: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        if message.contains("parameter") {
            suggestions.push("Check parameter names and types in the documentation".to_string());
            suggestions.push("Use the builder pattern to set parameters safely".to_string());
            suggestions.push("Validate parameter ranges before setting".to_string());
        }

        if message.contains("missing") {
            suggestions
                .push("Ensure all required components are added to the pipeline".to_string());
            suggestions.push("Check the pipeline construction order".to_string());
        }

        if message.contains("incompatible") {
            suggestions
                .push("Verify component compatibility using the validation tools".to_string());
            suggestions
                .push("Consider adding adapter components between incompatible stages".to_string());
        }

        suggestions
    }

    /// Generate data compatibility suggestions
    fn generate_compatibility_suggestions(
        &self,
        expected: &DataShape,
        actual: &DataShape,
        stage: &str,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if expected.features != actual.features {
            suggestions.push(format!(
                "Feature count mismatch in '{}': expected {}, got {}",
                stage, expected.features, actual.features
            ));
            suggestions.push(
                "Consider adding feature selection or expansion before this stage".to_string(),
            );
            suggestions.push(
                "Check if previous pipeline stages modified the feature count unexpectedly"
                    .to_string(),
            );
        }

        if expected.samples != actual.samples {
            suggestions.push(format!(
                "Sample count mismatch in '{}': expected {}, got {}",
                stage, expected.samples, actual.samples
            ));
            suggestions.push("Verify data splitting and sampling operations".to_string());
        }

        if expected.data_type != actual.data_type {
            suggestions.push(format!(
                "Data type mismatch in '{}': expected {}, got {}",
                stage, expected.data_type, actual.data_type
            ));
            suggestions.push("Add type conversion transformers before this stage".to_string());
        }

        if actual.missing_values && !expected.missing_values {
            suggestions.push(
                "Handle missing values using imputation or removal before this stage".to_string(),
            );
            suggestions
                .push("Consider using robust algorithms that handle missing data".to_string());
        }

        suggestions
    }

    /// Generate structure-specific suggestions
    fn generate_structure_suggestions(
        &self,
        error_type: &StructureErrorType,
        affected_components: &[String],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        match error_type {
            StructureErrorType::CyclicDependency => {
                suggestions.push("Remove circular dependencies between components".to_string());
                suggestions
                    .push("Use topological sorting to validate pipeline structure".to_string());
                suggestions.push(format!(
                    "Affected components: {}",
                    affected_components.join(", ")
                ));
            }
            StructureErrorType::MissingComponent => {
                suggestions.push("Add the missing component to the pipeline".to_string());
                suggestions.push("Check component names for typos".to_string());
                suggestions
                    .push("Verify component registration in the pipeline builder".to_string());
            }
            StructureErrorType::InvalidConnection => {
                suggestions.push("Check connection compatibility between components".to_string());
                suggestions.push("Ensure output types match input requirements".to_string());
                suggestions.push("Consider adding adapter components".to_string());
            }
            StructureErrorType::DanglingNode => {
                suggestions.push("Connect all nodes to the main pipeline flow".to_string());
                suggestions.push("Remove unused components or connect them properly".to_string());
            }
            StructureErrorType::InconsistentFlow => {
                suggestions.push("Review the entire pipeline flow for consistency".to_string());
                suggestions.push("Use pipeline validation tools before execution".to_string());
            }
        }

        suggestions
    }

    /// Generate performance optimization suggestions
    fn generate_performance_suggestions(
        &self,
        warning_type: &PerformanceWarningType,
        impact_level: &ImpactLevel,
        metrics: &Option<PerformanceMetrics>,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        match warning_type {
            PerformanceWarningType::MemoryUsage => {
                suggestions
                    .push("Consider using streaming processing for large datasets".to_string());
                suggestions.push("Enable memory-efficient pipeline execution".to_string());
                suggestions.push("Use data chunking to reduce memory footprint".to_string());

                if let Some(metrics) = metrics {
                    if metrics.memory_usage_mb > 1000.0 {
                        suggestions.push(
                            "Memory usage exceeds 1GB - consider distributed processing"
                                .to_string(),
                        );
                    }
                }
            }
            PerformanceWarningType::ComputationalComplexity => {
                suggestions.push("Use parallel processing where possible".to_string());
                suggestions.push("Consider simpler algorithms for large datasets".to_string());
                suggestions.push("Enable SIMD optimizations if available".to_string());
            }
            PerformanceWarningType::NetworkBottleneck => {
                suggestions.push("Implement result caching to reduce network calls".to_string());
                suggestions.push("Use connection pooling for external services".to_string());
                suggestions.push("Consider local processing alternatives".to_string());
            }
            PerformanceWarningType::CacheInefficiency => {
                suggestions.push("Optimize cache configuration and size".to_string());
                suggestions.push("Review cache key generation strategy".to_string());
                suggestions.push("Consider different cache eviction policies".to_string());

                if let Some(metrics) = metrics {
                    if metrics.cache_hit_ratio < 0.5 {
                        suggestions.push(format!(
                            "Low cache hit ratio ({:.1}%) - review caching strategy",
                            metrics.cache_hit_ratio * 100.0
                        ));
                    }
                }
            }
            PerformanceWarningType::SuboptimalConfiguration => {
                suggestions
                    .push("Run hyperparameter optimization to find better settings".to_string());
                suggestions.push("Use AutoML tools for automatic configuration tuning".to_string());
                suggestions.push("Profile different configuration options".to_string());
            }
        }

        match impact_level {
            ImpactLevel::Critical => {
                suggestions.insert(
                    0,
                    "🚨 CRITICAL: This issue requires immediate attention".to_string(),
                );
            }
            ImpactLevel::High => {
                suggestions.insert(0, "⚠️ HIGH PRIORITY: Address this issue soon".to_string());
            }
            _ => {}
        }

        suggestions
    }

    /// Generate resource constraint suggestions
    fn generate_resource_suggestions(
        &self,
        resource_type: &ResourceType,
        limit: f64,
        current: f64,
        component: &str,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        let utilization = (current / limit) * 100.0;

        suggestions.push(format!(
            "Resource utilization: {utilization:.1}% ({current:.2}/{limit:.2}) in component '{component}'"
        ));

        match resource_type {
            ResourceType::Memory => {
                suggestions.push("Reduce batch size to lower memory usage".to_string());
                suggestions.push("Enable streaming processing mode".to_string());
                suggestions.push("Use memory-mapped files for large datasets".to_string());
                if utilization > 90.0 {
                    suggestions.push(
                        "Consider upgrading system memory or using distributed processing"
                            .to_string(),
                    );
                }
            }
            ResourceType::CPU => {
                suggestions.push("Reduce computational complexity of algorithms".to_string());
                suggestions.push("Enable parallel processing across multiple cores".to_string());
                suggestions.push("Use approximate algorithms for faster computation".to_string());
            }
            ResourceType::GPU => {
                suggestions.push("Optimize GPU memory allocation and transfers".to_string());
                suggestions
                    .push("Use mixed precision training to reduce GPU memory usage".to_string());
                suggestions.push("Consider model parallelism for large models".to_string());
            }
            ResourceType::Disk => {
                suggestions.push("Enable data compression to reduce disk usage".to_string());
                suggestions.push("Use temporary file cleanup strategies".to_string());
                suggestions.push("Consider cloud storage for large datasets".to_string());
            }
            ResourceType::NetworkBandwidth => {
                suggestions.push("Implement data compression for network transfers".to_string());
                suggestions.push("Use local caching to reduce network usage".to_string());
                suggestions.push("Consider edge processing to minimize data movement".to_string());
            }
        }

        suggestions
    }

    /// Generate type safety suggestions
    fn generate_type_safety_suggestions(
        &self,
        violation_type: &TypeViolationType,
        expected_type: &str,
        actual_type: &str,
        stage: &str,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        suggestions.push(format!(
            "Type mismatch in '{stage}': expected '{expected_type}', got '{actual_type}'"
        ));

        match violation_type {
            TypeViolationType::IncompatibleInputType => {
                suggestions.push("Add type conversion transformer before this stage".to_string());
                suggestions
                    .push("Check the output type of the previous pipeline stage".to_string());
                suggestions.push("Use type adapters for incompatible formats".to_string());
            }
            TypeViolationType::MismatchedOutputType => {
                suggestions.push("Verify the component's output type specification".to_string());
                suggestions.push("Use type casting or conversion as the final step".to_string());
                suggestions.push("Check if the component configuration is correct".to_string());
            }
            TypeViolationType::UnsupportedTransformation => {
                suggestions.push(
                    "Use a different transformation that supports this data type".to_string(),
                );
                suggestions.push("Preprocess the data to a supported type".to_string());
                suggestions.push("Consider using a custom transformer".to_string());
            }
            TypeViolationType::InvalidParameterType => {
                suggestions.push("Check parameter type requirements in documentation".to_string());
                suggestions.push("Use proper type conversion for parameter values".to_string());
                suggestions
                    .push("Validate parameter types before pipeline construction".to_string());
            }
        }

        // Add common type conversion suggestions
        if expected_type.contains("float") && actual_type.contains("int") {
            suggestions
                .push("Convert integer values to float using .mapv(|x| x as f64)".to_string());
        } else if expected_type.contains("int") && actual_type.contains("float") {
            suggestions
                .push("Convert float values to integer (with rounding if needed)".to_string());
        }

        suggestions
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            pipeline_stage: "unknown".to_string(),
            component_name: "unknown".to_string(),
            input_shape: None,
            parameters: HashMap::new(),
            stack_trace: Vec::new(),
        }
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::ConfigurationError {
                message,
                suggestions,
                context,
            } => {
                writeln!(f, "🔧 Configuration Error: {message}")?;
                writeln!(
                    f,
                    "   Component: {} (Stage: {})",
                    context.component_name, context.pipeline_stage
                )?;
                if !suggestions.is_empty() {
                    writeln!(f, "💡 Suggestions:")?;
                    for suggestion in suggestions {
                        writeln!(f, "   • {suggestion}")?;
                    }
                }
            }
            PipelineError::DataCompatibilityError {
                expected,
                actual,
                stage,
                suggestions,
            } => {
                writeln!(f, "📊 Data Compatibility Error in stage '{stage}':")?;
                writeln!(
                    f,
                    "   Expected: {} samples × {} features ({})",
                    expected.samples, expected.features, expected.data_type
                )?;
                writeln!(
                    f,
                    "   Actual:   {} samples × {} features ({})",
                    actual.samples, actual.features, actual.data_type
                )?;
                if !suggestions.is_empty() {
                    writeln!(f, "💡 Suggestions:")?;
                    for suggestion in suggestions {
                        writeln!(f, "   • {suggestion}")?;
                    }
                }
            }
            PipelineError::StructureError {
                error_type,
                affected_components,
                suggestions,
            } => {
                writeln!(f, "🏗️ Pipeline Structure Error: {error_type:?}")?;
                writeln!(
                    f,
                    "   Affected components: {}",
                    affected_components.join(", ")
                )?;
                if !suggestions.is_empty() {
                    writeln!(f, "💡 Suggestions:")?;
                    for suggestion in suggestions {
                        writeln!(f, "   • {suggestion}")?;
                    }
                }
            }
            PipelineError::PerformanceWarning {
                warning_type,
                impact_level,
                suggestions,
                metrics,
            } => {
                writeln!(
                    f,
                    "⚡ Performance Warning: {warning_type:?} (Impact: {impact_level:?})"
                )?;
                if let Some(metrics) = metrics {
                    writeln!(
                        f,
                        "   Execution time: {:.2}ms, Memory: {:.1}MB, CPU: {:.1}%",
                        metrics.execution_time_ms, metrics.memory_usage_mb, metrics.cpu_utilization
                    )?;
                }
                if !suggestions.is_empty() {
                    writeln!(f, "💡 Suggestions:")?;
                    for suggestion in suggestions {
                        writeln!(f, "   • {suggestion}")?;
                    }
                }
            }
            PipelineError::ResourceError {
                resource_type,
                limit,
                current,
                component,
                suggestions,
            } => {
                let utilization = (current / limit) * 100.0;
                writeln!(
                    f,
                    "🔋 Resource Error: {resource_type:?} constraint violated in '{component}'"
                )?;
                writeln!(f, "   Usage: {current:.2}/{limit:.2} ({utilization:.1}%)")?;
                if !suggestions.is_empty() {
                    writeln!(f, "💡 Suggestions:")?;
                    for suggestion in suggestions {
                        writeln!(f, "   • {suggestion}")?;
                    }
                }
            }
            PipelineError::TypeSafetyError {
                violation_type,
                expected_type,
                actual_type,
                stage,
                suggestions,
            } => {
                writeln!(
                    f,
                    "🛡️ Type Safety Error: {violation_type:?} in stage '{stage}'"
                )?;
                writeln!(f, "   Expected: {expected_type}, Got: {actual_type}")?;
                if !suggestions.is_empty() {
                    writeln!(f, "💡 Suggestions:")?;
                    for suggestion in suggestions {
                        writeln!(f, "   • {suggestion}")?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl From<PipelineError> for SklearsError {
    fn from(error: PipelineError) -> Self {
        SklearsError::InvalidInput(error.to_string())
    }
}

/// Convenience functions for creating enhanced errors
impl PipelineError {
    /// Create a configuration error with suggestions
    #[must_use]
    pub fn configuration(message: &str) -> Self {
        EnhancedErrorBuilder::new()
            .configuration_error(message)
            .build()
    }

    /// Create a data compatibility error
    #[must_use]
    pub fn data_compatibility(expected: DataShape, actual: DataShape, stage: &str) -> Self {
        EnhancedErrorBuilder::new()
            .data_compatibility_error(expected, actual, stage)
            .build()
    }

    /// Create a performance warning
    #[must_use]
    pub fn performance_warning(
        warning_type: PerformanceWarningType,
        impact_level: ImpactLevel,
        metrics: Option<PerformanceMetrics>,
    ) -> Self {
        EnhancedErrorBuilder::new()
            .performance_warning(warning_type, impact_level, metrics)
            .build()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configuration_error_creation() {
        let error = PipelineError::configuration("Invalid parameter 'learning_rate'");

        let error_string = error.to_string();
        assert!(error_string.contains("Configuration Error"));
        assert!(error_string.contains("Invalid parameter"));
        assert!(error_string.contains("💡 Suggestions:"));
    }

    #[test]
    fn test_data_compatibility_error() {
        let expected = DataShape {
            samples: 100,
            features: 10,
            data_type: "float64".to_string(),
            missing_values: false,
        };
        let actual = DataShape {
            samples: 100,
            features: 8,
            data_type: "float64".to_string(),
            missing_values: true,
        };

        let error = PipelineError::data_compatibility(expected, actual, "feature_selection");
        let error_string = error.to_string();

        assert!(error_string.contains("Data Compatibility Error"));
        assert!(error_string.contains("feature_selection"));
        assert!(error_string.contains("10 features"));
        assert!(error_string.contains("8 features"));
    }

    #[test]
    fn test_enhanced_error_builder() {
        let context = ErrorContext {
            pipeline_stage: "preprocessing".to_string(),
            component_name: "scaler".to_string(),
            input_shape: Some((100, 5)),
            parameters: HashMap::new(),
            stack_trace: Vec::new(),
        };

        let error = EnhancedErrorBuilder::new()
            .configuration_error("Missing required parameter")
            .suggestion("Check the documentation for required parameters")
            .context(context)
            .build();

        let error_string = error.to_string();
        println!("Error string: {}", error_string);
        assert!(error_string.contains("Configuration Error"));
        assert!(error_string.contains("Stage: preprocessing"));
        assert!(error_string.contains("scaler"));
    }
}
