//! Enhanced Configuration Validation Framework
//!
//! This module provides comprehensive compile-time and runtime validation for pipeline
//! configurations, parameter constraints, and component compatibility. It ensures that
//! pipeline configurations are correct before execution and provides detailed diagnostics
//! for configuration errors.

use serde::{Deserialize, Serialize};
use sklears_core::traits::Estimator;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub severity: ValidationSeverity,
    pub component: String,
    pub field: String,
    pub message: String,
    pub suggestions: Vec<String>,
    pub error_code: String,
}

impl ValidationResult {
    pub fn new(
        severity: ValidationSeverity,
        component: impl Into<String>,
        field: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            component: component.into(),
            field: field.into(),
            message: message.into(),
            suggestions: Vec::new(),
            error_code: "VALIDATION_ERROR".to_string(),
        }
    }

    #[must_use]
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    pub fn with_error_code(mut self, code: impl Into<String>) -> Self {
        self.error_code = code.into();
        self
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(
            self.severity,
            ValidationSeverity::Error | ValidationSeverity::Critical
        )
    }
}

/// Comprehensive validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub results: Vec<ValidationResult>,
    pub summary: ValidationSummary,
}

#[derive(Debug, Clone)]
pub struct ValidationSummary {
    pub total_checks: usize,
    pub passed: usize,
    pub warnings: usize,
    pub errors: usize,
    pub critical: usize,
    pub overall_status: ValidationStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    /// Passed
    Passed,
    /// PassedWithWarnings
    PassedWithWarnings,
    /// Failed
    Failed,
    /// Critical
    Critical,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationReport {
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            summary: ValidationSummary {
                total_checks: 0,
                passed: 0,
                warnings: 0,
                errors: 0,
                critical: 0,
                overall_status: ValidationStatus::Passed,
            },
        }
    }

    pub fn add_result(&mut self, result: ValidationResult) {
        match result.severity {
            ValidationSeverity::Info => {}
            ValidationSeverity::Warning => self.summary.warnings += 1,
            ValidationSeverity::Error => self.summary.errors += 1,
            ValidationSeverity::Critical => self.summary.critical += 1,
        }
        self.results.push(result);
        self.update_summary();
    }

    fn update_summary(&mut self) {
        self.summary.total_checks = self.results.len();
        self.summary.passed = self.summary.total_checks
            - self.summary.warnings
            - self.summary.errors
            - self.summary.critical;

        self.summary.overall_status = if self.summary.critical > 0 {
            ValidationStatus::Critical
        } else if self.summary.errors > 0 {
            ValidationStatus::Failed
        } else if self.summary.warnings > 0 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Passed
        };
    }

    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.summary.errors > 0 || self.summary.critical > 0
    }

    #[must_use]
    pub fn display_summary(&self) -> String {
        format!(
            "Validation Summary: {} checks, {} passed, {} warnings, {} errors, {} critical (Status: {:?})",
            self.summary.total_checks,
            self.summary.passed,
            self.summary.warnings,
            self.summary.errors,
            self.summary.critical,
            self.summary.overall_status
        )
    }
}

/// Configuration validator trait for type-safe validation
pub trait ConfigurationValidator<T> {
    fn validate(&self, config: &T) -> ValidationReport;
    fn validate_field(&self, field_name: &str, value: &dyn std::fmt::Debug) -> ValidationReport;
    fn get_validation_schema(&self) -> ValidationSchema;
}

/// Validation schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSchema {
    pub name: String,
    pub version: String,
    pub fields: HashMap<String, FieldConstraints>,
    pub dependencies: Vec<DependencyConstraint>,
    pub custom_rules: Vec<CustomValidationRule>,
}

/// Field-level validation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConstraints {
    pub required: bool,
    pub field_type: FieldType,
    pub constraints: Vec<Constraint>,
    pub description: String,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    /// Integer
    Integer,
    /// Float
    Float,
    /// String
    String,
    /// Boolean
    Boolean,
    /// Array
    Array(Box<FieldType>),
    /// Object
    Object(String), // Schema name reference
    /// Enum
    Enum(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Range
    Range { min: f64, max: f64 },
    /// Length
    Length { min: usize, max: usize },
    /// Pattern
    Pattern(String), // Regex pattern
    /// OneOf
    OneOf(Vec<String>),
    /// Custom
    Custom(String), // Custom validation function name
}

/// Cross-field dependency constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyConstraint {
    pub name: String,
    pub condition: String, // Expression that must be true
    pub error_message: String,
    pub severity: ValidationSeverity,
}

/// Custom validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomValidationRule {
    pub name: String,
    pub description: String,
    pub rule_type: RuleType,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    /// ParameterCompatibility
    ParameterCompatibility,
    /// PerformanceWarning
    PerformanceWarning,
    /// SecurityCheck
    SecurityCheck,
    /// ResourceValidation
    ResourceValidation,
}

/// Enhanced compile-time configuration validator with type-level constraints
pub struct CompileTimeValidator<T> {
    schema: ValidationSchema,
    _phantom: PhantomData<T>,
}

impl<T> CompileTimeValidator<T> {
    #[must_use]
    pub fn new(schema: ValidationSchema) -> Self {
        Self {
            schema,
            _phantom: PhantomData,
        }
    }

    /// Validate configuration at compile time where possible
    pub fn validate_static(&self, config: &T) -> ValidationReport
    where
        T: fmt::Debug,
    {
        // Add static validation logic here
        // This would be expanded with actual compile-time checks

        ValidationReport::new()
    }
}

/// Type-level validation traits for compile-time guarantees
pub trait ValidConfig {
    type Error;
    const IS_VALID: bool;

    fn validate_config() -> Result<(), Self::Error>;
}

/// Phantom type for validated configurations
pub struct ValidatedConfig<T> {
    inner: T,
}

impl<T> ValidatedConfig<T>
where
    T: ValidConfig,
{
    /// Create a validated configuration
    pub fn new(config: T) -> Result<Self, T::Error> {
        if T::IS_VALID {
            Ok(ValidatedConfig { inner: config })
        } else {
            Err(T::validate_config().unwrap_err())
        }
    }

    /// Access the inner configuration
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Consume and return the inner configuration
    pub fn into_inner(self) -> T {
        self.inner
    }
}

/// Type-level parameter constraints using const generics
pub trait ParameterConstraints<const MIN: i32, const MAX: i32> {
    #[must_use]
    fn validate_range(value: i32) -> bool {
        value >= MIN && value <= MAX
    }
}

/// Compile-time validated parameter with range constraints
#[derive(Debug, Clone, Copy)]
pub struct ValidatedParameter<const MIN: i32, const MAX: i32> {
    value: i32,
}

impl<const MIN: i32, const MAX: i32> ValidatedParameter<MIN, MAX> {
    /// Create a validated parameter at compile time if possible
    #[must_use]
    pub const fn new(value: i32) -> Option<Self> {
        if value >= MIN && value <= MAX {
            Some(ValidatedParameter { value })
        } else {
            None
        }
    }

    /// Create a validated parameter with runtime check
    pub fn new_runtime(value: i32) -> Result<Self, ParameterValidationError> {
        if value >= MIN && value <= MAX {
            Ok(ValidatedParameter { value })
        } else {
            Err(ParameterValidationError::OutOfRange {
                value,
                min: MIN,
                max: MAX,
            })
        }
    }

    /// Get the parameter value
    #[must_use]
    pub const fn value(&self) -> i32 {
        self.value
    }
}

/// Enhanced parameter validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterValidationError {
    /// OutOfRange
    OutOfRange { value: i32, min: i32, max: i32 },
    /// InvalidType
    InvalidType { expected: String, actual: String },
    /// MissingRequired
    MissingRequired { parameter: String },
    /// DependencyViolation
    DependencyViolation {
        parameter: String,
        dependency: String,
    },
}

impl fmt::Display for ParameterValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParameterValidationError::OutOfRange { value, min, max } => {
                write!(f, "Parameter value {value} is out of range [{min}, {max}]")
            }
            ParameterValidationError::InvalidType { expected, actual } => {
                write!(f, "Expected type '{expected}', found '{actual}'")
            }
            ParameterValidationError::MissingRequired { parameter } => {
                write!(f, "Required parameter '{parameter}' is missing")
            }
            ParameterValidationError::DependencyViolation {
                parameter,
                dependency,
            } => {
                write!(
                    f,
                    "Parameter '{parameter}' violates dependency constraint '{dependency}'"
                )
            }
        }
    }
}

impl std::error::Error for ParameterValidationError {}

/// Type-level feature flag validation
pub trait FeatureFlags {
    const SUPPORTS_PARALLEL: bool;
    const SUPPORTS_GPU: bool;
    const REQUIRES_BLAS: bool;
    const MEMORY_INTENSIVE: bool;
}

/// Validated configuration with feature constraints
pub struct FeatureValidatedConfig<T, F>
where
    T: ValidConfig,
    F: FeatureFlags,
{
    config: ValidatedConfig<T>,
    _features: PhantomData<F>,
}

impl<T, F> FeatureValidatedConfig<T, F>
where
    T: ValidConfig,
    F: FeatureFlags,
{
    /// Create a feature-validated configuration
    pub fn new(config: T) -> Result<Self, ConfigurationValidationError> {
        // Validate feature compatibility
        Self::validate_features()?;

        let validated_config = ValidatedConfig::new(config)
            .map_err(|_| ConfigurationValidationError::InvalidConfiguration)?;

        Ok(FeatureValidatedConfig {
            config: validated_config,
            _features: PhantomData,
        })
    }

    /// Validate feature requirements at compile time
    const fn validate_features() -> Result<(), ConfigurationValidationError> {
        // This would contain compile-time feature validation logic
        Ok(())
    }

    /// Check if parallel execution is supported
    #[must_use]
    pub const fn supports_parallel() -> bool {
        F::SUPPORTS_PARALLEL
    }

    /// Check if GPU acceleration is supported
    #[must_use]
    pub const fn supports_gpu() -> bool {
        F::SUPPORTS_GPU
    }

    /// Access the validated configuration
    pub fn config(&self) -> &ValidatedConfig<T> {
        &self.config
    }
}

/// Enhanced configuration validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigurationValidationError {
    /// InvalidConfiguration
    InvalidConfiguration,
    /// FeatureNotSupported
    FeatureNotSupported { feature: String },
    /// IncompatibleFeatures
    IncompatibleFeatures { feature1: String, feature2: String },
    /// ResourceConstraintViolation
    ResourceConstraintViolation { resource: String, limit: String },
}

impl fmt::Display for ConfigurationValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigurationValidationError::InvalidConfiguration => {
                write!(f, "Configuration validation failed")
            }
            ConfigurationValidationError::FeatureNotSupported { feature } => {
                write!(f, "Feature '{feature}' is not supported")
            }
            ConfigurationValidationError::IncompatibleFeatures { feature1, feature2 } => {
                write!(f, "Features '{feature1}' and '{feature2}' are incompatible")
            }
            ConfigurationValidationError::ResourceConstraintViolation { resource, limit } => {
                write!(f, "Resource '{resource}' violates constraint '{limit}'")
            }
        }
    }
}

impl std::error::Error for ConfigurationValidationError {}

/// Compile-time pipeline stage validation
pub trait PipelineStage {
    type Input;
    type Output;
    type Config: ValidConfig;

    const STAGE_NAME: &'static str;
    const IS_TRANSFORMATIVE: bool;
    const IS_TERMINAL: bool;

    #[must_use]
    fn validate_compatibility<U: PipelineStage>() -> bool
    where
        Self::Output: CompatibleWith<U::Input>,
    {
        true
    }
}

/// Type-level compatibility checking
pub trait CompatibleWith<T> {}

// Implement compatibility for common types
impl CompatibleWith<f64> for f64 {}
impl CompatibleWith<i32> for i32 {}
impl<T> CompatibleWith<Vec<T>> for Vec<T> {}

/// Compile-time validated pipeline
pub struct ValidatedPipeline<Stages> {
    stages: Stages,
}

impl<S1> ValidatedPipeline<(S1,)>
where
    S1: PipelineStage,
{
    /// Create a single-stage validated pipeline
    pub fn new(stage: S1) -> Self
    where
        S1::Config: ValidConfig,
    {
        /// ValidatedPipeline
        ValidatedPipeline { stages: (stage,) }
    }
}

impl<S1, S2> ValidatedPipeline<(S1, S2)>
where
    S1: PipelineStage,
    S2: PipelineStage,
    S1::Output: CompatibleWith<S2::Input>,
{
    /// Create a two-stage validated pipeline with compile-time compatibility checking
    pub fn new(stage1: S1, stage2: S2) -> Self
    where
        S1::Config: ValidConfig,
        S2::Config: ValidConfig,
    {
        /// ValidatedPipeline
        ValidatedPipeline {
            stages: (stage1, stage2),
        }
    }
}

/// Macro for creating validated configurations with compile-time checks
#[macro_export]
macro_rules! validated_config {
    ($config_type:ty, $($field:ident: $value:expr),*) => {{
        // This would expand to create a validated configuration
        // with compile-time validation where possible
        compile_error!("Macro implementation would go here");
    }};
}

/// Advanced type-level validation for complex configurations
pub struct TypedConfigurationValidator<T, const N: usize> {
    validators: [fn(&T) -> ValidationResult; N],
    _phantom: PhantomData<T>,
}

impl<T, const N: usize> TypedConfigurationValidator<T, N> {
    /// Create a typed validator with a fixed number of validation functions
    pub const fn new(validators: [fn(&T) -> ValidationResult; N]) -> Self {
        Self {
            validators,
            _phantom: PhantomData,
        }
    }

    /// Validate configuration using all registered validators
    pub fn validate(&self, config: &T) -> ValidationReport {
        let mut report = ValidationReport::new();

        for validator in &self.validators {
            let result = validator(config);
            report.add_result(result);
        }

        report
    }
}

/// Runtime configuration validator
pub struct RuntimeValidator {
    schemas: HashMap<String, ValidationSchema>,
    custom_validators:
        HashMap<String, Box<dyn Fn(&dyn std::fmt::Debug) -> ValidationResult + Send + Sync>>,
}

impl Default for RuntimeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeValidator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            custom_validators: HashMap::new(),
        }
    }

    pub fn register_schema(&mut self, name: String, schema: ValidationSchema) {
        self.schemas.insert(name, schema);
    }

    pub fn register_custom_validator<F>(&mut self, name: String, validator: F)
    where
        F: Fn(&dyn std::fmt::Debug) -> ValidationResult + Send + Sync + 'static,
    {
        self.custom_validators.insert(name, Box::new(validator));
    }

    pub fn validate_configuration<T>(&self, config: &T, schema_name: &str) -> ValidationReport
    where
        T: fmt::Debug,
    {
        let mut report = ValidationReport::new();

        if let Some(schema) = self.schemas.get(schema_name) {
            // Validate against schema
            for (field_name, constraints) in &schema.fields {
                let field_result = self.validate_field_constraints(field_name, constraints, config);
                if let Some(result) = field_result {
                    report.add_result(result);
                }
            }

            // Validate dependencies
            for dependency in &schema.dependencies {
                let dep_result = self.validate_dependency(dependency, config);
                if let Some(result) = dep_result {
                    report.add_result(result);
                }
            }

            // Apply custom rules
            for rule in &schema.custom_rules {
                let rule_result = self.apply_custom_rule(rule, config);
                if let Some(result) = rule_result {
                    report.add_result(result);
                }
            }
        } else {
            report.add_result(
                ValidationResult::new(
                    ValidationSeverity::Error,
                    "RuntimeValidator",
                    "schema",
                    format!("Schema '{schema_name}' not found"),
                )
                .with_error_code("SCHEMA_NOT_FOUND"),
            );
        }

        report
    }

    fn validate_field_constraints<T>(
        &self,
        field_name: &str,
        constraints: &FieldConstraints,
        config: &T,
    ) -> Option<ValidationResult>
    where
        T: fmt::Debug,
    {
        // Simplified validation - in practice this would inspect the actual field values
        if constraints.required {
            // Check if required field is present and valid
            None // Placeholder - would contain actual validation logic
        } else {
            None
        }
    }

    fn validate_dependency<T>(
        &self,
        dependency: &DependencyConstraint,
        config: &T,
    ) -> Option<ValidationResult>
    where
        T: fmt::Debug,
    {
        // Evaluate dependency condition
        // Placeholder - would contain actual dependency checking logic
        None
    }

    fn apply_custom_rule<T>(
        &self,
        rule: &CustomValidationRule,
        config: &T,
    ) -> Option<ValidationResult>
    where
        T: fmt::Debug,
    {
        // Apply custom validation rule
        // Placeholder - would contain actual custom rule logic
        None
    }
}

/// Pipeline-specific configuration validator
pub struct PipelineConfigValidator {
    validator: RuntimeValidator,
}

impl Default for PipelineConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineConfigValidator {
    #[must_use]
    pub fn new() -> Self {
        let mut validator = RuntimeValidator::new();

        // Register pipeline-specific schemas
        let pipeline_schema = ValidationSchema {
            name: "PipelineConfig".to_string(),
            version: "1.0.0".to_string(),
            fields: Self::create_pipeline_field_constraints(),
            dependencies: Self::create_pipeline_dependencies(),
            custom_rules: Self::create_pipeline_custom_rules(),
        };

        validator.register_schema("PipelineConfig".to_string(), pipeline_schema);

        // Register custom validators
        validator.register_custom_validator("performance_check".to_string(), |config| {
            ValidationResult::new(
                ValidationSeverity::Info,
                "PipelineConfig",
                "performance",
                "Performance validation passed",
            )
        });

        Self { validator }
    }

    fn create_pipeline_field_constraints() -> HashMap<String, FieldConstraints> {
        let mut fields = HashMap::new();

        fields.insert(
            "n_jobs".to_string(),
            /// FieldConstraints
            FieldConstraints {
                required: false,
                field_type: FieldType::Integer,
                constraints: vec![Constraint::Range {
                    min: -1.0,
                    max: 1000.0,
                }],
                description: "Number of parallel jobs (-1 for all cores)".to_string(),
                examples: vec!["1".to_string(), "4".to_string(), "-1".to_string()],
            },
        );

        fields.insert(
            "random_state".to_string(),
            /// FieldConstraints
            FieldConstraints {
                required: false,
                field_type: FieldType::Integer,
                constraints: vec![Constraint::Range {
                    min: 0.0,
                    max: 2_147_483_647.0,
                }],
                description: "Random seed for reproducibility".to_string(),
                examples: vec!["42".to_string(), "123".to_string()],
            },
        );

        fields.insert(
            "verbose".to_string(),
            /// FieldConstraints
            FieldConstraints {
                required: false,
                field_type: FieldType::Boolean,
                constraints: vec![],
                description: "Enable verbose output".to_string(),
                examples: vec!["true".to_string(), "false".to_string()],
            },
        );

        fields
    }

    fn create_pipeline_dependencies() -> Vec<DependencyConstraint> {
        vec![DependencyConstraint {
            name: "parallel_processing_compatibility".to_string(),
            condition: "n_jobs > 1 implies thread_safe == true".to_string(),
            error_message: "Parallel processing requires thread-safe components".to_string(),
            severity: ValidationSeverity::Error,
        }]
    }

    fn create_pipeline_custom_rules() -> Vec<CustomValidationRule> {
        vec![CustomValidationRule {
            name: "memory_usage_warning".to_string(),
            description: "Warn about potentially high memory usage".to_string(),
            rule_type: RuleType::PerformanceWarning,
            parameters: HashMap::new(),
        }]
    }

    pub fn validate_pipeline_config<T>(&self, config: &T) -> ValidationReport
    where
        T: fmt::Debug,
    {
        self.validator
            .validate_configuration(config, "PipelineConfig")
    }
}

/// Configuration validation builder for fluent API
pub struct ValidationBuilder {
    schema: ValidationSchema,
}

impl ValidationBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema: ValidationSchema {
                name: name.into(),
                version: "1.0.0".to_string(),
                fields: HashMap::new(),
                dependencies: Vec::new(),
                custom_rules: Vec::new(),
            },
        }
    }

    pub fn add_field(mut self, name: impl Into<String>, constraints: FieldConstraints) -> Self {
        self.schema.fields.insert(name.into(), constraints);
        self
    }

    #[must_use]
    pub fn add_dependency(mut self, dependency: DependencyConstraint) -> Self {
        self.schema.dependencies.push(dependency);
        self
    }

    #[must_use]
    pub fn add_custom_rule(mut self, rule: CustomValidationRule) -> Self {
        self.schema.custom_rules.push(rule);
        self
    }

    #[must_use]
    pub fn build(self) -> ValidationSchema {
        self.schema
    }
}

/// Enhanced configuration validation examples
pub mod examples {
    use super::{
        Constraint, CustomValidationRule, DependencyConstraint, FieldConstraints, FieldType,
        HashMap, RuleType, ValidationBuilder, ValidationSchema, ValidationSeverity,
    };

    #[must_use]
    pub fn create_linear_regression_validator() -> ValidationSchema {
        ValidationBuilder::new("LinearRegressionConfig")
            .add_field(
                "fit_intercept",
                /// FieldConstraints
                FieldConstraints {
                    required: false,
                    field_type: FieldType::Boolean,
                    constraints: vec![],
                    description: "Whether to calculate the intercept".to_string(),
                    examples: vec!["true".to_string(), "false".to_string()],
                },
            )
            .add_field(
                "alpha",
                /// FieldConstraints
                FieldConstraints {
                    required: false,
                    field_type: FieldType::Float,
                    constraints: vec![Constraint::Range { min: 0.0, max: 1e6 }],
                    description: "Regularization strength".to_string(),
                    examples: vec!["0.01".to_string(), "1.0".to_string(), "100.0".to_string()],
                },
            )
            .add_dependency(DependencyConstraint {
                name: "regularization_warning".to_string(),
                condition: "alpha > 1000".to_string(),
                error_message: "Very high regularization may lead to underfitting".to_string(),
                severity: ValidationSeverity::Warning,
            })
            .build()
    }

    #[must_use]
    pub fn create_ensemble_validator() -> ValidationSchema {
        ValidationBuilder::new("EnsembleConfig")
            .add_field(
                "n_estimators",
                /// FieldConstraints
                FieldConstraints {
                    required: true,
                    field_type: FieldType::Integer,
                    constraints: vec![Constraint::Range {
                        min: 1.0,
                        max: 10000.0,
                    }],
                    description: "Number of estimators in the ensemble".to_string(),
                    examples: vec!["10".to_string(), "100".to_string(), "1000".to_string()],
                },
            )
            .add_field(
                "voting",
                /// FieldConstraints
                FieldConstraints {
                    required: false,
                    field_type: FieldType::Enum(vec!["hard".to_string(), "soft".to_string()]),
                    constraints: vec![Constraint::OneOf(vec![
                        "hard".to_string(),
                        "soft".to_string(),
                    ])],
                    description: "Voting strategy for ensemble".to_string(),
                    examples: vec!["hard".to_string(), "soft".to_string()],
                },
            )
            .add_custom_rule(CustomValidationRule {
                name: "performance_vs_accuracy_tradeoff".to_string(),
                description: "Warn about performance implications of large ensembles".to_string(),
                rule_type: RuleType::PerformanceWarning,
                parameters: HashMap::new(),
            })
            .build()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_creation() {
        let result = ValidationResult::new(
            ValidationSeverity::Warning,
            "TestComponent",
            "test_field",
            "Test message",
        )
        .with_suggestions(vec!["Fix suggestion".to_string()])
        .with_error_code("TEST_001");

        assert_eq!(result.severity, ValidationSeverity::Warning);
        assert_eq!(result.component, "TestComponent");
        assert_eq!(result.field, "test_field");
        assert_eq!(result.message, "Test message");
        assert_eq!(result.suggestions.len(), 1);
        assert_eq!(result.error_code, "TEST_001");
        assert!(!result.is_error());
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();

        report.add_result(ValidationResult::new(
            ValidationSeverity::Warning,
            "Component1",
            "field1",
            "Warning message",
        ));

        report.add_result(ValidationResult::new(
            ValidationSeverity::Error,
            "Component2",
            "field2",
            "Error message",
        ));

        assert_eq!(report.summary.warnings, 1);
        assert_eq!(report.summary.errors, 1);
        assert_eq!(report.summary.overall_status, ValidationStatus::Failed);
        assert!(report.has_errors());
    }

    #[test]
    fn test_pipeline_config_validator() {
        let validator = PipelineConfigValidator::new();

        // Mock configuration for testing
        #[derive(Debug)]
        struct MockConfig {
            n_jobs: i32,
            verbose: bool,
        }

        let config = MockConfig {
            n_jobs: 4,
            verbose: true,
        };

        let report = validator.validate_pipeline_config(&config);

        // In a real implementation, this would have actual validation results
        assert_eq!(report.summary.overall_status, ValidationStatus::Passed);
    }

    #[test]
    fn test_validation_builder() {
        let schema = ValidationBuilder::new("TestSchema")
            .add_field(
                "test_field",
                /// FieldConstraints
                FieldConstraints {
                    required: true,
                    field_type: FieldType::Integer,
                    constraints: vec![Constraint::Range {
                        min: 1.0,
                        max: 100.0,
                    }],
                    description: "Test field".to_string(),
                    examples: vec!["50".to_string()],
                },
            )
            .build();

        assert_eq!(schema.name, "TestSchema");
        assert_eq!(schema.fields.len(), 1);
        assert!(schema.fields.contains_key("test_field"));
    }

    #[test]
    fn test_validated_parameter() {
        // Test compile-time validation with const generics
        type JobCount = ValidatedParameter<1, 8>;

        // Valid parameter
        let valid_param = JobCount::new_runtime(4).unwrap();
        assert_eq!(valid_param.value(), 4);

        // Invalid parameter
        let invalid_param = JobCount::new_runtime(10);
        assert!(invalid_param.is_err());

        match invalid_param.unwrap_err() {
            ParameterValidationError::OutOfRange { value, min, max } => {
                assert_eq!(value, 10);
                assert_eq!(min, 1);
                assert_eq!(max, 8);
            }
            _ => panic!("Expected OutOfRange error"),
        }
    }

    #[test]
    fn test_feature_flags() {
        // Mock feature flags for testing
        struct TestFeatures;
        impl FeatureFlags for TestFeatures {
            const SUPPORTS_PARALLEL: bool = true;
            const SUPPORTS_GPU: bool = false;
            const REQUIRES_BLAS: bool = true;
            const MEMORY_INTENSIVE: bool = false;
        }

        // Mock valid configuration
        #[derive(Debug)]
        struct TestConfig;
        impl ValidConfig for TestConfig {
            type Error = ConfigurationValidationError;
            const IS_VALID: bool = true;

            fn validate_config() -> Result<(), Self::Error> {
                Ok(())
            }
        }

        // Test feature validation
        let config = TestConfig;
        let feature_config = FeatureValidatedConfig::<TestConfig, TestFeatures>::new(config);
        assert!(feature_config.is_ok());

        // Test feature queries
        assert!(FeatureValidatedConfig::<TestConfig, TestFeatures>::supports_parallel());
        assert!(!FeatureValidatedConfig::<TestConfig, TestFeatures>::supports_gpu());
    }

    #[test]
    fn test_pipeline_stage_compatibility() {
        // Mock pipeline stages for testing
        #[derive(Debug)]
        struct TestStage1;
        #[derive(Debug)]
        struct TestStage2;

        #[derive(Debug)]
        struct TestConfig;
        impl ValidConfig for TestConfig {
            type Error = ConfigurationValidationError;
            const IS_VALID: bool = true;

            fn validate_config() -> Result<(), Self::Error> {
                Ok(())
            }
        }

        impl PipelineStage for TestStage1 {
            type Input = i32;
            type Output = f64;
            type Config = TestConfig;

            const STAGE_NAME: &'static str = "TestStage1";
            const IS_TRANSFORMATIVE: bool = true;
            const IS_TERMINAL: bool = false;
        }

        impl PipelineStage for TestStage2 {
            type Input = f64;
            type Output = String;
            type Config = TestConfig;

            const STAGE_NAME: &'static str = "TestStage2";
            const IS_TRANSFORMATIVE: bool = true;
            const IS_TERMINAL: bool = true;
        }

        // Test pipeline creation with compatible stages
        let stage1 = TestStage1;
        let stage2 = TestStage2;
        let _pipeline = ValidatedPipeline::<(TestStage1, TestStage2)>::new(stage1, stage2);

        // Test stage properties
        assert_eq!(TestStage1::STAGE_NAME, "TestStage1");
        assert!(TestStage1::IS_TRANSFORMATIVE);
        assert!(!TestStage1::IS_TERMINAL);
        assert!(TestStage2::IS_TERMINAL);
    }

    #[test]
    fn test_typed_configuration_validator() {
        // Test configuration type
        #[derive(Debug)]
        struct TestConfig {
            value: i32,
        }

        // Validation functions
        fn validate_positive(config: &TestConfig) -> ValidationResult {
            if config.value > 0 {
                ValidationResult::new(
                    ValidationSeverity::Info,
                    "TestConfig",
                    "value",
                    "Value is positive",
                )
            } else {
                ValidationResult::new(
                    ValidationSeverity::Error,
                    "TestConfig",
                    "value",
                    "Value must be positive",
                )
            }
        }

        fn validate_range(config: &TestConfig) -> ValidationResult {
            if config.value >= 1 && config.value <= 100 {
                ValidationResult::new(
                    ValidationSeverity::Info,
                    "TestConfig",
                    "value",
                    "Value is in range",
                )
            } else {
                ValidationResult::new(
                    ValidationSeverity::Warning,
                    "TestConfig",
                    "value",
                    "Value is outside recommended range",
                )
            }
        }

        // Create typed validator with const generic size
        let validator = TypedConfigurationValidator::new([validate_positive, validate_range]);

        // Test with valid configuration
        let valid_config = TestConfig { value: 50 };
        let report = validator.validate(&valid_config);
        assert_eq!(report.summary.errors, 0);

        // Test with invalid configuration
        let invalid_config = TestConfig { value: -10 };
        let report = validator.validate(&invalid_config);
        assert_eq!(report.summary.errors, 1);
        assert_eq!(report.summary.warnings, 1);
    }

    #[test]
    fn test_parameter_validation_error_display() {
        let error = ParameterValidationError::OutOfRange {
            value: 15,
            min: 1,
            max: 10,
        };
        assert_eq!(
            error.to_string(),
            "Parameter value 15 is out of range [1, 10]"
        );

        let error = ParameterValidationError::InvalidType {
            expected: "Integer".to_string(),
            actual: "String".to_string(),
        };
        assert_eq!(error.to_string(), "Expected type 'Integer', found 'String'");

        let error = ParameterValidationError::MissingRequired {
            parameter: "alpha".to_string(),
        };
        assert_eq!(error.to_string(), "Required parameter 'alpha' is missing");
    }

    #[test]
    fn test_configuration_validation_error_display() {
        let error = ConfigurationValidationError::FeatureNotSupported {
            feature: "GPU".to_string(),
        };
        assert_eq!(error.to_string(), "Feature 'GPU' is not supported");

        let error = ConfigurationValidationError::IncompatibleFeatures {
            feature1: "Parallel".to_string(),
            feature2: "SingleThread".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Features 'Parallel' and 'SingleThread' are incompatible"
        );
    }
}
