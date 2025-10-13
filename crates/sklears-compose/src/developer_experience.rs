//! Developer Experience Enhancements
//!
//! This module provides improved developer experience through better error messages,
//! debugging utilities, and pipeline inspection tools. It focuses on making
//! sklearn-compose more ergonomic and easier to debug.

use crate::enhanced_errors::PipelineError;
use serde::{Deserialize, Serialize};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;
use std::fmt;

/// Enhanced error messages with actionable suggestions and context
#[derive(Debug, Clone)]
pub struct DeveloperFriendlyError {
    /// Original error
    pub original_error: PipelineError,
    /// Detailed explanation for developers
    pub explanation: String,
    /// Step-by-step suggestions to fix the issue
    pub fix_suggestions: Vec<FixSuggestion>,
    /// Related documentation links
    pub documentation_links: Vec<String>,
    /// Code examples that might help
    pub code_examples: Vec<CodeExample>,
}

/// A specific suggestion to fix an issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSuggestion {
    /// Priority level of this suggestion
    pub priority: SuggestionPriority,
    /// Short description of the fix
    pub title: String,
    /// Detailed explanation of how to apply the fix
    pub description: String,
    /// Code snippet demonstrating the fix
    pub code_snippet: Option<String>,
    /// Estimated effort to implement (in minutes)
    pub estimated_effort_minutes: u32,
}

/// Priority levels for fix suggestions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SuggestionPriority {
    /// Must fix - will cause failures
    Critical,
    /// Should fix - may cause issues
    High,
    /// Good to fix - improves performance or reliability
    Medium,
    /// Nice to have - cosmetic improvements
    Low,
}

/// Code example to help with debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub title: String,
    pub description: String,
    pub code: String,
    pub expected_output: String,
}

/// Debug utilities for pipeline inspection
#[derive(Debug)]
pub struct PipelineDebugger {
    /// Debug session ID
    pub session_id: String,
    /// Current debug state
    pub state: DebugState,
    /// Breakpoints set by the developer
    pub breakpoints: Vec<Breakpoint>,
    /// Execution trace
    pub execution_trace: Vec<TraceEntry>,
    /// Watch expressions
    pub watch_expressions: HashMap<String, WatchExpression>,
}

/// Current debugging state
#[derive(Debug, Clone)]
pub enum DebugState {
    /// Not debugging
    Idle,
    /// Running with debug enabled
    Running,
    /// Paused at a breakpoint
    Paused {
        breakpoint_id: String,
        context: ExecutionContext,
    },
    /// Stepping through execution
    Stepping { step_type: StepType },
    /// Error occurred during debugging
    Error { error: String },
}

/// Types of stepping through execution
#[derive(Debug, Clone)]
pub enum StepType {
    /// Step to next instruction
    StepNext,
    /// Step into function/component
    StepInto,
    /// Step out of current function/component
    StepOut,
    /// Continue until next breakpoint
    Continue,
}

/// A debugging breakpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    /// Unique ID for this breakpoint
    pub id: String,
    /// Component or stage where breakpoint is set
    pub location: String,
    /// Condition that must be met to trigger breakpoint
    pub condition: Option<String>,
    /// Whether this breakpoint is currently enabled
    pub enabled: bool,
    /// Number of times this breakpoint has been hit
    pub hit_count: u32,
}

/// Execution context when paused at breakpoint
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Current component being executed
    pub current_component: String,
    /// Pipeline stage
    pub stage: String,
    /// Local variables and their values
    pub variables: HashMap<String, String>,
    /// Input data shape at this point
    pub input_shape: Option<(usize, usize)>,
    /// Memory usage at this point
    pub memory_usage_mb: f64,
    /// Execution time so far (milliseconds)
    pub execution_time_ms: f64,
}

/// Entry in the execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Timestamp when this entry was recorded
    pub timestamp: u64,
    /// Component that was executed
    pub component: String,
    /// Operation that was performed
    pub operation: String,
    /// Duration of the operation (milliseconds)
    pub duration_ms: f64,
    /// Input data shape
    pub input_shape: Option<(usize, usize)>,
    /// Output data shape
    pub output_shape: Option<(usize, usize)>,
    /// Memory usage before operation
    pub memory_before_mb: f64,
    /// Memory usage after operation
    pub memory_after_mb: f64,
    /// Any warnings or notes
    pub notes: Vec<String>,
}

/// Watch expression for monitoring values during debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchExpression {
    /// Expression to evaluate
    pub expression: String,
    /// Current value of the expression
    pub current_value: Option<String>,
    /// History of values
    pub value_history: Vec<(u64, String)>, // timestamp, value
    /// Whether this watch is currently active
    pub active: bool,
}

impl PipelineDebugger {
    /// Create a new debugging session
    #[must_use]
    pub fn new() -> Self {
        Self {
            session_id: format!("debug-{}", chrono::Utc::now().timestamp()),
            state: DebugState::Idle,
            breakpoints: Vec::new(),
            execution_trace: Vec::new(),
            watch_expressions: HashMap::new(),
        }
    }

    /// Start a debugging session
    pub fn start_debug_session(&mut self) -> SklResult<()> {
        self.state = DebugState::Running;
        self.execution_trace.clear();
        println!("🐛 Debug session started: {}", self.session_id);
        Ok(())
    }

    /// Add a breakpoint
    pub fn add_breakpoint(&mut self, location: String, condition: Option<String>) -> String {
        let id = format!("bp-{}", self.breakpoints.len());
        let breakpoint = Breakpoint {
            id: id.clone(),
            location,
            condition,
            enabled: true,
            hit_count: 0,
        };
        self.breakpoints.push(breakpoint);
        println!(
            "🔴 Breakpoint added: {} at {}",
            id,
            self.breakpoints.last().unwrap().location
        );
        id
    }

    /// Add a watch expression
    pub fn add_watch(&mut self, name: String, expression: String) {
        let watch = WatchExpression {
            expression: expression.clone(),
            current_value: None,
            value_history: Vec::new(),
            active: true,
        };
        self.watch_expressions.insert(name.clone(), watch);
        println!("👁️  Watch added: {name} -> {expression}");
    }

    /// Record a trace entry
    pub fn record_trace(&mut self, entry: TraceEntry) {
        self.execution_trace.push(entry);

        // Check if we hit any breakpoints
        let current_component = self.execution_trace.last().unwrap().component.clone();
        if let Some(breakpoint) = self.check_breakpoints(&current_component) {
            self.pause_at_breakpoint(breakpoint);
        }
    }

    /// Check if execution should pause at any breakpoints
    fn check_breakpoints(&mut self, current_component: &str) -> Option<String> {
        for breakpoint in &mut self.breakpoints {
            if breakpoint.enabled && breakpoint.location == current_component {
                breakpoint.hit_count += 1;
                return Some(breakpoint.id.clone());
            }
        }
        None
    }

    /// Pause execution at a breakpoint
    fn pause_at_breakpoint(&mut self, breakpoint_id: String) {
        let context = ExecutionContext {
            current_component: "example_component".to_string(), // This would be filled with actual context
            stage: "example_stage".to_string(),
            variables: HashMap::new(),
            input_shape: Some((100, 10)),
            memory_usage_mb: 128.5,
            execution_time_ms: 1500.0,
        };

        self.state = DebugState::Paused {
            breakpoint_id: breakpoint_id.clone(),
            context,
        };
        println!("⏸️  Paused at breakpoint: {breakpoint_id}");
    }

    /// Get debugging summary
    pub fn get_debug_summary(&self) -> DebugSummary {
        DebugSummary {
            session_id: self.session_id.clone(),
            total_trace_entries: self.execution_trace.len(),
            active_breakpoints: self.breakpoints.iter().filter(|bp| bp.enabled).count(),
            active_watches: self
                .watch_expressions
                .iter()
                .filter(|(_, w)| w.active)
                .count(),
            current_state: format!("{:?}", self.state),
            total_execution_time_ms: self.execution_trace.iter().map(|e| e.duration_ms).sum(),
            peak_memory_usage_mb: self
                .execution_trace
                .iter()
                .map(|e| e.memory_after_mb)
                .fold(0.0, f64::max),
        }
    }
}

/// Summary of debugging session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSummary {
    pub session_id: String,
    pub total_trace_entries: usize,
    pub active_breakpoints: usize,
    pub active_watches: usize,
    pub current_state: String,
    pub total_execution_time_ms: f64,
    pub peak_memory_usage_mb: f64,
}

impl Default for PipelineDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Error message enhancer that provides developer-friendly error messages
pub struct ErrorMessageEnhancer;

impl ErrorMessageEnhancer {
    /// Enhance a pipeline error with developer-friendly information
    #[must_use]
    pub fn enhance_error(error: PipelineError) -> DeveloperFriendlyError {
        match &error {
            PipelineError::ConfigurationError { message, .. } => {
                Self::enhance_configuration_error(error.clone(), message)
            }
            PipelineError::DataCompatibilityError {
                expected,
                actual,
                stage,
                ..
            } => Self::enhance_data_compatibility_error(error.clone(), expected, actual, stage),
            PipelineError::StructureError {
                error_type,
                affected_components,
                ..
            } => Self::enhance_structure_error(error.clone(), error_type, affected_components),
            PipelineError::PerformanceWarning {
                warning_type,
                impact_level,
                ..
            } => Self::enhance_performance_warning(error.clone(), warning_type, impact_level),
            PipelineError::ResourceError {
                resource_type,
                limit,
                current,
                component,
                ..
            } => Self::enhance_resource_error(
                error.clone(),
                resource_type,
                *limit,
                *current,
                component,
            ),
            PipelineError::TypeSafetyError {
                violation_type,
                expected_type,
                actual_type,
                ..
            } => Self::enhance_type_safety_error(
                error.clone(),
                violation_type,
                expected_type,
                actual_type,
            ),
        }
    }

    fn enhance_configuration_error(error: PipelineError, message: &str) -> DeveloperFriendlyError {
        DeveloperFriendlyError {
            original_error: error,
            explanation: format!(
                "Configuration Error: {message}\n\n\
                This error occurs when pipeline parameters are incorrectly configured. \
                Common causes include invalid parameter values, missing required parameters, \
                or incompatible parameter combinations."
            ),
            fix_suggestions: vec![
                FixSuggestion {
                    priority: SuggestionPriority::Critical,
                    title: "Check parameter documentation".to_string(),
                    description: "Review the documentation for valid parameter ranges and types".to_string(),
                    code_snippet: Some("pipeline.builder().with_parameter(\"valid_value\").build()".to_string()),
                    estimated_effort_minutes: 5,
                },
                FixSuggestion {
                    priority: SuggestionPriority::High,
                    title: "Use configuration validation".to_string(),
                    description: "Enable configuration validation to catch errors early".to_string(),
                    code_snippet: Some("pipeline.validate_config().expect(\"Valid configuration\")".to_string()),
                    estimated_effort_minutes: 2,
                },
            ],
            documentation_links: vec![
                "https://docs.sklears.com/pipeline-configuration".to_string(),
                "https://docs.sklears.com/parameter-validation".to_string(),
            ],
            code_examples: vec![
                CodeExample {
                    title: "Basic Pipeline Configuration".to_string(),
                    description: "Example of properly configuring a pipeline".to_string(),
                    code: "use sklears_compose::Pipeline;\n\nlet pipeline = Pipeline::builder()\n    .add_step(\"scaler\", StandardScaler::new())\n    .add_step(\"model\", LinearRegression::new())\n    .build()?;".to_string(),
                    expected_output: "Successfully configured pipeline with 2 steps".to_string(),
                },
            ],
        }
    }

    fn enhance_data_compatibility_error(
        error: PipelineError,
        expected: &crate::enhanced_errors::DataShape,
        actual: &crate::enhanced_errors::DataShape,
        stage: &str,
    ) -> DeveloperFriendlyError {
        DeveloperFriendlyError {
            original_error: error,
            explanation: format!(
                "Data Compatibility Error at stage '{}'\n\n\
                Expected: {} samples × {} features ({})\n\
                Actual: {} samples × {} features ({})\n\n\
                This error occurs when data shapes don't match between pipeline stages. \
                This is often caused by incorrect data preprocessing or feature selection.",
                stage,
                expected.samples, expected.features, expected.data_type,
                actual.samples, actual.features, actual.data_type
            ),
            fix_suggestions: vec![
                FixSuggestion {
                    priority: SuggestionPriority::Critical,
                    title: "Check data preprocessing".to_string(),
                    description: "Verify that preprocessing steps maintain expected data shapes".to_string(),
                    code_snippet: Some("print(\"Data shape after preprocessing: {}\", X.shape)".to_string()),
                    estimated_effort_minutes: 10,
                },
                FixSuggestion {
                    priority: SuggestionPriority::High,
                    title: "Add shape validation".to_string(),
                    description: "Add explicit shape validation between pipeline stages".to_string(),
                    code_snippet: Some("pipeline.add_validation_step(ShapeValidator::new(expected_shape))".to_string()),
                    estimated_effort_minutes: 5,
                },
            ],
            documentation_links: vec![
                "https://docs.sklears.com/data-shapes".to_string(),
                "https://docs.sklears.com/pipeline-validation".to_string(),
            ],
            code_examples: vec![
                CodeExample {
                    title: "Data Shape Debugging".to_string(),
                    description: "How to debug data shape issues".to_string(),
                    code: "// Check data shapes at each step\nlet mut pipeline = Pipeline::new();\npipeline.debug_mode(true);\nlet result = pipeline.fit_transform(&X, &y)?;\nprintln!(\"Final shape: {:?}\", result.shape());".to_string(),
                    expected_output: "Detailed shape information at each step".to_string(),
                },
            ],
        }
    }

    fn enhance_structure_error(
        error: PipelineError,
        error_type: &crate::enhanced_errors::StructureErrorType,
        affected_components: &[String],
    ) -> DeveloperFriendlyError {
        DeveloperFriendlyError {
            original_error: error,
            explanation: format!(
                "Pipeline Structure Error: {:?}\n\
                Affected components: {}\n\n\
                This error indicates a problem with the pipeline's structure or component relationships.",
                error_type,
                affected_components.join(", ")
            ),
            fix_suggestions: vec![
                FixSuggestion {
                    priority: SuggestionPriority::Critical,
                    title: "Review pipeline structure".to_string(),
                    description: "Check the logical flow and dependencies between components".to_string(),
                    code_snippet: Some("pipeline.visualize_structure()".to_string()),
                    estimated_effort_minutes: 15,
                },
            ],
            documentation_links: vec![
                "https://docs.sklears.com/pipeline-structure".to_string(),
            ],
            code_examples: vec![],
        }
    }

    fn enhance_performance_warning(
        error: PipelineError,
        warning_type: &crate::enhanced_errors::PerformanceWarningType,
        impact_level: &crate::enhanced_errors::ImpactLevel,
    ) -> DeveloperFriendlyError {
        DeveloperFriendlyError {
            original_error: error,
            explanation: format!(
                "Performance Warning: {warning_type:?} (Impact: {impact_level:?})\n\n\
                This warning indicates potential performance issues that may affect execution."
            ),
            fix_suggestions: vec![FixSuggestion {
                priority: SuggestionPriority::Medium,
                title: "Profile pipeline performance".to_string(),
                description: "Use the built-in profiler to identify bottlenecks".to_string(),
                code_snippet: Some("pipeline.enable_profiling().run_with_profiling()".to_string()),
                estimated_effort_minutes: 20,
            }],
            documentation_links: vec![
                "https://docs.sklears.com/performance-optimization".to_string()
            ],
            code_examples: vec![],
        }
    }

    fn enhance_resource_error(
        error: PipelineError,
        resource_type: &crate::enhanced_errors::ResourceType,
        limit: f64,
        current: f64,
        component: &str,
    ) -> DeveloperFriendlyError {
        DeveloperFriendlyError {
            original_error: error,
            explanation: format!(
                "Resource Constraint Error in component '{component}'\n\
                Resource: {resource_type:?}\n\
                Limit: {limit:.2}\n\
                Current usage: {current:.2}\n\n\
                The component is exceeding available resources."
            ),
            fix_suggestions: vec![FixSuggestion {
                priority: SuggestionPriority::High,
                title: "Optimize memory usage".to_string(),
                description:
                    "Consider using streaming or batch processing to reduce memory footprint"
                        .to_string(),
                code_snippet: Some("pipeline.set_batch_size(1000).enable_streaming()".to_string()),
                estimated_effort_minutes: 30,
            }],
            documentation_links: vec!["https://docs.sklears.com/memory-optimization".to_string()],
            code_examples: vec![],
        }
    }

    fn enhance_type_safety_error(
        error: PipelineError,
        violation_type: &crate::enhanced_errors::TypeViolationType,
        expected_type: &str,
        actual_type: &str,
    ) -> DeveloperFriendlyError {
        DeveloperFriendlyError {
            original_error: error,
            explanation: format!(
                "Type Safety Error: {violation_type:?}\n\
                Expected type: {expected_type}\n\
                Actual type: {actual_type}\n\n\
                This error occurs when there's a type mismatch between pipeline components."
            ),
            fix_suggestions: vec![FixSuggestion {
                priority: SuggestionPriority::Critical,
                title: "Check type compatibility".to_string(),
                description: "Ensure all pipeline components have compatible input/output types"
                    .to_string(),
                code_snippet: Some("pipeline.validate_type_safety()".to_string()),
                estimated_effort_minutes: 10,
            }],
            documentation_links: vec!["https://docs.sklears.com/type-safety".to_string()],
            code_examples: vec![],
        }
    }
}

/// Display implementations for better developer experience
impl fmt::Display for DeveloperFriendlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "🚨 {}", self.explanation)?;

        if !self.fix_suggestions.is_empty() {
            writeln!(f, "\n💡 Suggested fixes:")?;
            for (i, suggestion) in self.fix_suggestions.iter().enumerate() {
                writeln!(
                    f,
                    "  {}. [{}] {} (≈{}min)",
                    i + 1,
                    match suggestion.priority {
                        SuggestionPriority::Critical => "🔴 CRITICAL",
                        SuggestionPriority::High => "🟡 HIGH",
                        SuggestionPriority::Medium => "🔵 MEDIUM",
                        SuggestionPriority::Low => "⚪ LOW",
                    },
                    suggestion.title,
                    suggestion.estimated_effort_minutes
                )?;
                writeln!(f, "     {}", suggestion.description)?;
                if let Some(code) = &suggestion.code_snippet {
                    writeln!(f, "     Example: {code}")?;
                }
            }
        }

        if !self.documentation_links.is_empty() {
            writeln!(f, "\n📚 Helpful documentation:")?;
            for link in &self.documentation_links {
                writeln!(f, "  • {link}")?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for SuggestionPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SuggestionPriority::Critical => write!(f, "Critical"),
            SuggestionPriority::High => write!(f, "High"),
            SuggestionPriority::Medium => write!(f, "Medium"),
            SuggestionPriority::Low => write!(f, "Low"),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_errors::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = PipelineDebugger::new();
        assert!(matches!(debugger.state, DebugState::Idle));
        assert_eq!(debugger.breakpoints.len(), 0);
        assert_eq!(debugger.execution_trace.len(), 0);
    }

    #[test]
    fn test_add_breakpoint() {
        let mut debugger = PipelineDebugger::new();
        let bp_id = debugger.add_breakpoint("test_component".to_string(), None);

        assert_eq!(debugger.breakpoints.len(), 1);
        assert_eq!(debugger.breakpoints[0].id, bp_id);
        assert_eq!(debugger.breakpoints[0].location, "test_component");
        assert!(debugger.breakpoints[0].enabled);
    }

    #[test]
    fn test_add_watch() {
        let mut debugger = PipelineDebugger::new();
        debugger.add_watch("test_var".to_string(), "x.shape".to_string());

        assert_eq!(debugger.watch_expressions.len(), 1);
        assert!(debugger.watch_expressions.contains_key("test_var"));
        assert_eq!(debugger.watch_expressions["test_var"].expression, "x.shape");
    }

    #[test]
    fn test_error_enhancement() {
        let error = PipelineError::ConfigurationError {
            message: "Invalid parameter value".to_string(),
            suggestions: vec!["Check documentation".to_string()],
            context: ErrorContext {
                pipeline_stage: "test".to_string(),
                component_name: "test_component".to_string(),
                input_shape: Some((100, 10)),
                parameters: std::collections::HashMap::new(),
                stack_trace: vec![],
            },
        };

        let enhanced = ErrorMessageEnhancer::enhance_error(error);
        assert!(enhanced.explanation.contains("Configuration Error"));
        assert!(!enhanced.fix_suggestions.is_empty());
        assert!(enhanced.fix_suggestions[0].priority as u8 <= SuggestionPriority::Critical as u8);
    }

    #[test]
    fn test_debug_summary() {
        let debugger = PipelineDebugger::new();
        let summary = debugger.get_debug_summary();

        assert_eq!(summary.total_trace_entries, 0);
        assert_eq!(summary.active_breakpoints, 0);
        assert_eq!(summary.active_watches, 0);
        assert_eq!(summary.total_execution_time_ms, 0.0);
    }
}
