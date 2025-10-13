//! Configuration management and validation for optimization
//!
//! This module provides comprehensive configuration management including:
//! - Optimizer configuration and parameter validation
//! - Problem validation and transformation
//! - Configuration templates and presets
//! - Parameter tuning and optimization
//! - Validation rules and constraint checking

use std::collections::HashMap;
use std::time::Duration;

use scirs2_core::ndarray::Array1;
use crate::core::SklResult;
use super::optimization_core::{OptimizationProblem, ProblemValidation, ValidationIssue, ComplexityEstimate, OptimizerParameters};

/// Optimizer configuration management
#[derive(Debug, Clone)]
pub struct OptimizerConfiguration {
    /// Configuration identifier
    pub config_id: String,
    /// Algorithm-specific parameters
    pub algorithm_parameters: OptimizerParameters,
    /// Stopping criteria
    pub stopping_criteria: StoppingCriteria,
    /// Performance settings
    pub performance_settings: PerformanceSettings,
    /// Logging configuration
    pub logging_settings: LoggingSettings,
    /// Parallel execution settings
    pub parallel_settings: ParallelSettings,
    /// Validation settings
    pub validation_settings: ValidationSettings,
    /// Custom configuration values
    pub custom_settings: HashMap<String, ConfigValue>,
}

/// Generic configuration value
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Boolean(bool),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of integers
    IntArray(Vec<i64>),
    /// Nested configuration
    Config(HashMap<String, ConfigValue>),
}

/// Stopping criteria configuration
#[derive(Debug, Clone)]
pub struct StoppingCriteria {
    /// Maximum iterations
    pub max_iterations: Option<u64>,
    /// Maximum function evaluations
    pub max_function_evaluations: Option<u64>,
    /// Maximum execution time
    pub max_time: Option<Duration>,
    /// Target objective value
    pub target_objective: Option<f64>,
    /// Objective tolerance
    pub objective_tolerance: f64,
    /// Gradient tolerance
    pub gradient_tolerance: f64,
    /// Parameter tolerance
    pub parameter_tolerance: f64,
    /// Stagnation threshold
    pub stagnation_threshold: u64,
    /// Relative tolerance enabled
    pub relative_tolerance: bool,
}

/// Performance optimization settings
#[derive(Debug, Clone)]
pub struct PerformanceSettings {
    /// Precision level for computations
    pub precision_level: PrecisionLevel,
    /// Enable SIMD acceleration
    pub simd_enabled: bool,
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
    /// Memory usage limit (bytes)
    pub memory_limit: Option<usize>,
    /// Cache size for function evaluations
    pub cache_size: usize,
    /// Enable adaptive performance tuning
    pub adaptive_tuning: bool,
    /// Performance monitoring enabled
    pub monitoring_enabled: bool,
}

/// Precision levels for numerical computations
#[derive(Debug, Clone)]
pub enum PrecisionLevel {
    /// Single precision (32-bit)
    Single,
    /// Double precision (64-bit)
    Double,
    /// Extended precision
    Extended,
    /// Arbitrary precision
    Arbitrary,
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingSettings {
    /// Enable logging
    pub enabled: bool,
    /// Log level
    pub level: LogLevel,
    /// Log frequency
    pub frequency: LogFrequency,
    /// Log file path
    pub log_file: Option<String>,
    /// Enable console output
    pub console_output: bool,
    /// Log format
    pub format: String,
    /// Buffer size for log messages
    pub buffer_size: usize,
}

/// Logging levels
#[derive(Debug, Clone)]
pub enum LogLevel {
    /// Error messages only
    Error,
    /// Warning and error messages
    Warning,
    /// Informational messages
    Info,
    /// Debug messages
    Debug,
    /// Trace-level messages
    Trace,
}

/// Log frequency settings
#[derive(Debug, Clone)]
pub enum LogFrequency {
    /// Log every iteration
    EveryIteration,
    /// Log every N iterations
    EveryN(u64),
    /// Log on significant events
    OnEvent,
    /// Log on convergence milestones
    OnMilestone,
}

/// Parallel execution settings
#[derive(Debug, Clone)]
pub struct ParallelSettings {
    /// Enable parallel execution
    pub enabled: bool,
    /// Number of threads to use
    pub thread_count: Option<usize>,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Communication protocol
    pub communication_protocol: CommunicationProtocol,
    /// Synchronization method
    pub synchronization_method: String,
    /// Enable distributed execution
    pub distributed: bool,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    Static,
    Dynamic,
    WorkStealing,
    RoundRobin,
}

/// Communication protocols for distributed execution
#[derive(Debug, Clone)]
pub enum CommunicationProtocol {
    /// Message Passing Interface
    MPI,
    /// TCP/IP sockets
    TCP,
    /// UDP sockets
    UDP,
    /// Shared memory
    SharedMemory,
    /// Custom protocol
    Custom(String),
}

/// Validation settings for problems and solutions
#[derive(Debug, Clone)]
pub struct ValidationSettings {
    /// Enable problem validation
    pub problem_validation: bool,
    /// Enable solution validation
    pub solution_validation: bool,
    /// Numerical tolerance for validation
    pub numerical_tolerance: f64,
    /// Enable constraint checking
    pub constraint_checking: bool,
    /// Enable bounds checking
    pub bounds_checking: bool,
    /// Validation frequency
    pub validation_frequency: u64,
    /// Fail on validation errors
    pub fail_on_error: bool,
}

/// Optimization validator trait
pub trait OptimizationValidator: Send + Sync {
    /// Validate optimization problem
    fn validate_problem(&self, problem: &OptimizationProblem) -> SklResult<ProblemValidation>;

    /// Validate optimizer configuration
    fn validate_configuration(&self, config: &OptimizerConfiguration) -> SklResult<ConfigValidation>;

    /// Get validator name
    fn get_name(&self) -> &str;

    /// Get validation rules
    fn get_rules(&self) -> Vec<ValidationRule>;
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ConfigValidation {
    /// Whether configuration is valid
    pub is_valid: bool,
    /// Validation score (0-1)
    pub score: f64,
    /// Configuration issues found
    pub issues: Vec<ConfigValidationIssue>,
    /// Recommended improvements
    pub recommendations: Vec<String>,
    /// Estimated performance impact
    pub performance_impact: f64,
}

/// Configuration validation issue
#[derive(Debug, Clone)]
pub struct ConfigValidationIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Issue description
    pub description: String,
    /// Parameter name (if applicable)
    pub parameter_name: Option<String>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone)]
pub enum IssueSeverity {
    /// Minor issue
    Minor,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical issue
    Critical,
}

/// Issue categories
#[derive(Debug, Clone)]
pub enum IssueCategory {
    /// Parameter value out of range
    ParameterRange,
    /// Incompatible parameter combination
    Compatibility,
    /// Performance concern
    Performance,
    /// Numerical stability issue
    Stability,
    /// Resource constraint
    Resource,
    /// Configuration syntax error
    Syntax,
}

/// Validation rule definition
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub rule_id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule parameters
    pub parameters: HashMap<String, f64>,
    /// Rule enabled
    pub enabled: bool,
}

/// Types of validation rules
#[derive(Debug, Clone)]
pub enum ValidationRuleType {
    /// Range validation
    Range,
    /// Type validation
    Type,
    /// Compatibility validation
    Compatibility,
    /// Performance validation
    Performance,
    /// Custom validation rule
    Custom(String),
}

impl Default for OptimizerConfiguration {
    fn default() -> Self {
        Self {
            config_id: format!("config_{}", std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()),
            algorithm_parameters: OptimizerParameters::default(),
            stopping_criteria: StoppingCriteria::default(),
            performance_settings: PerformanceSettings::default(),
            logging_settings: LoggingSettings::default(),
            parallel_settings: ParallelSettings::default(),
            validation_settings: ValidationSettings::default(),
            custom_settings: HashMap::new(),
        }
    }
}

impl Default for StoppingCriteria {
    fn default() -> Self {
        Self {
            max_iterations: Some(1000),
            max_function_evaluations: Some(10000),
            max_time: Some(Duration::from_secs(3600)), // 1 hour
            target_objective: None,
            objective_tolerance: 1e-6,
            gradient_tolerance: 1e-8,
            parameter_tolerance: 1e-10,
            stagnation_threshold: 100,
            relative_tolerance: true,
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            precision_level: PrecisionLevel::Double,
            simd_enabled: true,
            gpu_enabled: false,
            memory_limit: None,
            cache_size: 1000,
            adaptive_tuning: true,
            monitoring_enabled: true,
        }
    }
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            level: LogLevel::Info,
            frequency: LogFrequency::EveryN(10),
            log_file: None,
            console_output: true,
            format: "%(timestamp)s - %(level)s - %(message)s".to_string(),
            buffer_size: 1024,
        }
    }
}

impl Default for ParallelSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            thread_count: None, // Use system default
            load_balancing: LoadBalancingStrategy::Dynamic,
            communication_protocol: CommunicationProtocol::SharedMemory,
            synchronization_method: "barrier".to_string(),
            distributed: false,
        }
    }
}

impl Default for ValidationSettings {
    fn default() -> Self {
        Self {
            problem_validation: true,
            solution_validation: true,
            numerical_tolerance: 1e-12,
            constraint_checking: true,
            bounds_checking: true,
            validation_frequency: 1,
            fail_on_error: false,
        }
    }
}

impl OptimizerConfiguration {
    /// Create new optimizer configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration
    pub fn validate(&self) -> SklResult<ConfigValidation> {
        let mut issues = Vec::new();
        let mut score = 1.0;

        // Validate stopping criteria
        if let Some(max_iter) = self.stopping_criteria.max_iterations {
            if max_iter == 0 {
                issues.push(ConfigValidationIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::ParameterRange,
                    description: "Maximum iterations must be greater than 0".to_string(),
                    parameter_name: Some("max_iterations".to_string()),
                    suggested_fix: Some("Set max_iterations to a positive value".to_string()),
                });
                score -= 0.2;
            }
        }

        // Validate tolerances
        if self.stopping_criteria.objective_tolerance <= 0.0 {
            issues.push(ConfigValidationIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::ParameterRange,
                description: "Objective tolerance must be positive".to_string(),
                parameter_name: Some("objective_tolerance".to_string()),
                suggested_fix: Some("Set objective_tolerance to a small positive value".to_string()),
            });
            score -= 0.2;
        }

        let is_valid = issues.iter().all(|issue| !matches!(issue.severity, IssueSeverity::Error | IssueSeverity::Critical));

        Ok(ConfigValidation {
            is_valid,
            score: score.max(0.0),
            issues,
            recommendations: vec![
                "Consider enabling SIMD acceleration for better performance".to_string(),
                "Set appropriate memory limits for large problems".to_string(),
            ],
            performance_impact: self.estimate_performance_impact(),
        })
    }

    /// Estimate performance impact of configuration
    fn estimate_performance_impact(&self) -> f64 {
        let mut impact = 1.0;

        // SIMD acceleration impact
        if self.performance_settings.simd_enabled {
            impact *= 0.7; // 30% speedup
        }

        // GPU acceleration impact
        if self.performance_settings.gpu_enabled {
            impact *= 0.5; // 50% speedup
        }

        // Parallel execution impact
        if self.parallel_settings.enabled {
            if let Some(threads) = self.parallel_settings.thread_count {
                impact *= 1.0 / (threads as f64).sqrt(); // Diminishing returns
            }
        }

        impact
    }

    /// Get configuration summary
    pub fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();

        summary.insert("config_id".to_string(), self.config_id.clone());
        summary.insert("simd_enabled".to_string(), self.performance_settings.simd_enabled.to_string());
        summary.insert("gpu_enabled".to_string(), self.performance_settings.gpu_enabled.to_string());
        summary.insert("parallel_enabled".to_string(), self.parallel_settings.enabled.to_string());
        summary.insert("logging_enabled".to_string(), self.logging_settings.enabled.to_string());

        if let Some(max_iter) = self.stopping_criteria.max_iterations {
            summary.insert("max_iterations".to_string(), max_iter.to_string());
        }

        summary
    }
}

impl ConfigValue {
    /// Convert to integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(i) => Some(*i),
            ConfigValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Convert to float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Convert to string if possible
    pub fn as_string(&self) -> Option<String> {
        match self {
            ConfigValue::String(s) => Some(s.clone()),
            ConfigValue::Integer(i) => Some(i.to_string()),
            ConfigValue::Float(f) => Some(f.to_string()),
            ConfigValue::Boolean(b) => Some(b.to_string()),
            _ => None,
        }
    }

    /// Convert to boolean if possible
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(b) => Some(*b),
            ConfigValue::Integer(i) => Some(*i != 0),
            ConfigValue::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }
}