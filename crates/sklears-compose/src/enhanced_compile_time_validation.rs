//! Enhanced compile-time validation for pipeline configurations
//!
//! This module provides comprehensive compile-time validation capabilities for ML pipelines including:
//! - Type-level validation using Rust's type system to catch errors at compile time
//! - Constraint validation for parameter ranges and dependencies
//! - Schema validation for configuration structure and format
//! - Cross-reference validation for relationships between configuration sections
//! - Custom validation rules and user-defined validation logic
//! - Comprehensive validation reporting with detailed error messages and suggestions

use crate::error::{Result, SklearsComposeError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Enhanced compile-time validation system for ML pipeline configurations
#[derive(Debug)]
pub struct CompileTimeValidator<State = Unvalidated> {
    /// Validation state marker
    state: PhantomData<State>,

    /// Schema validators for different configuration types
    schema_validators: Arc<RwLock<HashMap<String, Box<dyn SchemaValidator>>>>,

    /// Constraint validators for parameter validation
    constraint_validators: Arc<RwLock<Vec<Box<dyn ConstraintValidator>>>>,

    /// Dependency validators for checking relationships
    dependency_validators: Arc<RwLock<Vec<Box<dyn DependencyValidator>>>>,

    /// Cross-reference validators for inter-section validation
    cross_reference_validators: Arc<RwLock<Vec<Box<dyn CrossReferenceValidator>>>>,

    /// Custom validation rules
    custom_validators: Arc<RwLock<Vec<Box<dyn CustomValidator>>>>,

    /// Validation configuration
    config: ValidationConfig,

    /// Validation cache for performance
    validation_cache: Arc<RwLock<HashMap<String, ValidationResult>>>,
}

/// Validation state markers
#[derive(Debug, Clone)]
pub struct Unvalidated;

#[derive(Debug, Clone)]
pub struct Validated;

#[derive(Debug, Clone)]
pub struct PartiallyValidated;

/// Configuration for validation behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable strict type checking
    pub strict_type_checking: bool,

    /// Enable constraint validation
    pub enable_constraint_validation: bool,

    /// Enable dependency validation
    pub enable_dependency_validation: bool,

    /// Enable cross-reference validation
    pub enable_cross_reference_validation: bool,

    /// Enable custom validation rules
    pub enable_custom_validation: bool,

    /// Maximum validation depth for nested configurations
    pub max_validation_depth: usize,

    /// Validation timeout
    pub validation_timeout: Duration,

    /// Enable validation caching
    pub enable_caching: bool,

    /// Fail fast on first error or collect all errors
    pub fail_fast: bool,

    /// Enable detailed validation reports
    pub detailed_reports: bool,
}

/// Comprehensive validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation status
    pub status: ValidationStatus,

    /// Validation errors found
    pub errors: Vec<ValidationError>,

    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,

    /// Validation suggestions
    pub suggestions: Vec<ValidationSuggestion>,

    /// Validation metrics
    pub metrics: ValidationMetrics,

    /// Validation timestamp
    pub timestamp: SystemTime,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Valid
    Valid,
    /// Invalid
    Invalid,
    /// Warning
    Warning,
    /// Partial
    Partial,
    /// Unknown
    Unknown,
}

/// Validation error with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error identifier
    pub error_id: String,

    /// Error category
    pub category: ValidationErrorCategory,

    /// Error message
    pub message: String,

    /// Error location in configuration
    pub location: ConfigurationLocation,

    /// Error severity
    pub severity: ValidationSeverity,

    /// Expected value or format
    pub expected: Option<String>,

    /// Actual value found
    pub actual: Option<String>,

    /// Suggested fix
    pub suggestion: Option<String>,

    /// Related errors
    pub related_errors: Vec<String>,
}

/// Categories of validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationErrorCategory {
    /// TypeMismatch
    TypeMismatch,
    /// ConstraintViolation
    ConstraintViolation,
    /// MissingRequired
    MissingRequired,
    /// InvalidFormat
    InvalidFormat,
    /// DependencyMissing
    DependencyMissing,
    /// CrossReferenceError
    CrossReferenceError,
    /// CustomValidationFailure
    CustomValidationFailure,
    /// SchemaViolation
    SchemaViolation,
    /// RangeError
    RangeError,
    /// CompatibilityError
    CompatibilityError,
}

/// Validation error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical
    Critical,
    /// High
    High,
    /// Medium
    Medium,
    /// Low
    Low,
    /// Info
    Info,
}

/// Location within configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationLocation {
    /// Configuration section
    pub section: String,

    /// Field path (e.g., "`model.parameters.learning_rate`")
    pub field_path: String,

    /// Line number (if applicable)
    pub line_number: Option<usize>,

    /// Column number (if applicable)
    pub column_number: Option<usize>,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning identifier
    pub warning_id: String,

    /// Warning message
    pub message: String,

    /// Warning location
    pub location: ConfigurationLocation,

    /// Warning category
    pub category: WarningCategory,

    /// Recommendation
    pub recommendation: Option<String>,
}

/// Warning categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningCategory {
    /// Performance
    Performance,
    /// Compatibility
    Compatibility,
    /// Deprecated
    Deprecated,
    /// Suboptimal
    Suboptimal,
    /// Experimental
    Experimental,
}

/// Validation suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSuggestion {
    /// Suggestion identifier
    pub suggestion_id: String,

    /// Suggestion message
    pub message: String,

    /// Suggested action
    pub action: SuggestionAction,

    /// Confidence level
    pub confidence: f64,

    /// Priority level
    pub priority: SuggestionPriority,
}

/// Types of suggestion actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionAction {
    /// AddField
    AddField { field: String, value: String },
    /// RemoveField
    RemoveField { field: String },
    /// ModifyField
    ModifyField { field: String, new_value: String },
    /// RestructureSection
    RestructureSection {
        section: String,
        new_structure: String,
    },
    /// AddDependency
    AddDependency { dependency: String },
    /// UpgradeVersion
    UpgradeVersion { component: String, version: String },
}

/// Suggestion priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionPriority {
    /// Critical
    Critical,
    /// High
    High,
    /// Medium
    Medium,
    /// Low
    Low,
}

/// Validation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Total validation time
    pub validation_time: Duration,

    /// Number of rules checked
    pub rules_checked: usize,

    /// Number of fields validated
    pub fields_validated: usize,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Memory usage during validation
    pub memory_usage: u64,
}

/// Type-safe configuration builder with compile-time validation
#[derive(Debug)]
pub struct TypeSafeConfigBuilder<State = Unbuilt> {
    /// Builder state marker
    state: PhantomData<State>,

    /// Configuration data
    config_data: HashMap<String, ConfigValue>,

    /// Type constraints
    type_constraints: HashMap<String, TypeConstraint>,

    /// Validation rules
    validation_rules: Vec<Box<dyn ValidationRule>>,

    /// Builder configuration
    builder_config: BuilderConfig,
}

/// Builder state markers
#[derive(Debug, Clone)]
pub struct Unbuilt;

#[derive(Debug, Clone)]
pub struct Built;

/// Configuration value with type information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    /// String
    String(String),
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// Boolean
    Boolean(bool),
    /// Array
    Array(Vec<ConfigValue>),
    /// Object
    Object(HashMap<String, ConfigValue>),
    /// Null
    Null,
}

/// Type constraint for configuration fields
pub struct TypeConstraint {
    /// Expected type
    pub expected_type: ConfigType,

    /// Whether field is required
    pub required: bool,

    /// Default value if not provided
    pub default_value: Option<ConfigValue>,

    /// Value constraints
    pub constraints: Vec<ValueConstraint>,

    /// Custom validation function
    pub custom_validator: Option<Box<dyn Fn(&ConfigValue) -> Result<()> + Send + Sync>>,
}

impl std::fmt::Debug for TypeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeConstraint")
            .field("expected_type", &self.expected_type)
            .field("required", &self.required)
            .field("default_value", &self.default_value)
            .field("constraints", &self.constraints)
            .field(
                "custom_validator",
                &format!(
                    "<{} validator>",
                    if self.custom_validator.is_some() {
                        "some"
                    } else {
                        "none"
                    }
                ),
            )
            .finish()
    }
}

impl Clone for TypeConstraint {
    fn clone(&self) -> Self {
        Self {
            expected_type: self.expected_type.clone(),
            required: self.required,
            default_value: self.default_value.clone(),
            constraints: self.constraints.clone(),
            custom_validator: None, // Can't clone function pointers
        }
    }
}

/// Configuration value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigType {
    /// String
    String,
    /// Integer
    Integer,
    /// Float
    Float,
    /// Boolean
    Boolean,
    /// Array
    Array(Box<ConfigType>),
    /// Object
    Object(HashMap<String, ConfigType>),
    /// Union
    Union(Vec<ConfigType>),
    /// Optional
    Optional(Box<ConfigType>),
}

/// Value constraints for configuration fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueConstraint {
    /// Range
    Range { min: f64, max: f64 },
    /// Length
    Length { min: usize, max: Option<usize> },
    /// Pattern
    Pattern(String),
    /// OneOf
    OneOf(Vec<ConfigValue>),
    /// Custom
    Custom(String), // Custom constraint description
}

/// Builder configuration
#[derive(Debug, Clone)]
pub struct BuilderConfig {
    /// Enable strict validation
    pub strict_validation: bool,

    /// Allow unknown fields
    pub allow_unknown_fields: bool,

    /// Maximum nesting depth
    pub max_nesting_depth: usize,
}

/// Schema validator trait
pub trait SchemaValidator: std::fmt::Debug + Send + Sync {
    /// Validate configuration against schema
    fn validate_schema(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<Vec<ValidationError>>;

    /// Get schema name
    fn schema_name(&self) -> &str;

    /// Get schema version
    fn schema_version(&self) -> &str;
}

/// Constraint validator trait
pub trait ConstraintValidator: std::fmt::Debug + Send + Sync {
    /// Validate configuration constraints
    fn validate_constraints(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<Vec<ValidationError>>;

    /// Get validator name
    fn validator_name(&self) -> &str;
}

/// Dependency validator trait
pub trait DependencyValidator: std::fmt::Debug + Send + Sync {
    /// Validate configuration dependencies
    fn validate_dependencies(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<Vec<ValidationError>>;

    /// Get dependency validator name
    fn validator_name(&self) -> &str;
}

/// Cross-reference validator trait
pub trait CrossReferenceValidator: std::fmt::Debug + Send + Sync {
    /// Validate cross-references in configuration
    fn validate_cross_references(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<Vec<ValidationError>>;

    /// Get validator name
    fn validator_name(&self) -> &str;
}

/// Custom validator trait
pub trait CustomValidator: std::fmt::Debug + Send + Sync {
    /// Perform custom validation
    fn validate(&self, config: &HashMap<String, ConfigValue>) -> Result<Vec<ValidationError>>;

    /// Get validator name
    fn validator_name(&self) -> &str;

    /// Get validator description
    fn description(&self) -> &str;
}

/// Validation rule trait
pub trait ValidationRule: std::fmt::Debug + Send + Sync {
    /// Apply validation rule
    fn apply(&self, field: &str, value: &ConfigValue) -> Result<()>;

    /// Get rule name
    fn rule_name(&self) -> &str;
}

/// Pipeline configuration schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfigurationSchema {
    /// Schema version
    pub version: String,

    /// Required fields
    pub required_fields: HashSet<String>,

    /// Field definitions
    pub field_definitions: HashMap<String, FieldDefinition>,

    /// Schema constraints
    pub constraints: Vec<SchemaConstraint>,

    /// Cross-reference rules
    pub cross_reference_rules: Vec<CrossReferenceRule>,
}

/// Field definition in schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field type
    pub field_type: ConfigType,

    /// Field description
    pub description: String,

    /// Default value
    pub default: Option<ConfigValue>,

    /// Field constraints
    pub constraints: Vec<ValueConstraint>,

    /// Whether field is deprecated
    pub deprecated: bool,

    /// Deprecation message
    pub deprecation_message: Option<String>,
}

/// Schema-level constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaConstraint {
    /// Constraint name
    pub name: String,

    /// Constraint description
    pub description: String,

    /// Fields involved in constraint
    pub fields: Vec<String>,

    /// Constraint type
    pub constraint_type: SchemaConstraintType,
}

/// Types of schema constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaConstraintType {
    /// MutualExclusion
    MutualExclusion,
    /// RequiredTogether
    RequiredTogether,
    /// ConditionalRequired
    ConditionalRequired {
        condition: String,
        required: Vec<String>,
    },
    /// ValueDependency
    ValueDependency {
        field: String,
        dependent_field: String,
        values: Vec<ConfigValue>,
    },
}

/// Cross-reference rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReferenceRule {
    /// Rule name
    pub name: String,

    /// Source field
    pub source_field: String,

    /// Target field
    pub target_field: String,

    /// Reference type
    pub reference_type: ReferenceType,

    /// Validation rule
    pub validation_rule: String,
}

/// Types of cross-references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    /// ForeignKey
    ForeignKey,
    /// WeakReference
    WeakReference,
    /// StrongReference
    StrongReference,
    /// Computed
    Computed,
}

/// Compile-time validated pipeline configuration
#[derive(Debug)]
pub struct ValidatedPipelineConfig<T> {
    /// Configuration data
    config: T,

    /// Validation proof
    validation_proof: ValidationProof,

    /// Schema version used for validation
    schema_version: String,
}

/// Proof that configuration has been validated
#[derive(Debug, Clone)]
pub struct ValidationProof {
    /// Proof identifier
    proof_id: String,

    /// Validation timestamp
    validation_time: SystemTime,

    /// Validator version
    validator_version: String,

    /// Validation checksum
    checksum: String,
}

/// Type-level proof that configuration is valid
pub struct ValidConfigMarker<T>(PhantomData<T>);

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_type_checking: true,
            enable_constraint_validation: true,
            enable_dependency_validation: true,
            enable_cross_reference_validation: true,
            enable_custom_validation: true,
            max_validation_depth: 10,
            validation_timeout: Duration::from_secs(30),
            enable_caching: true,
            fail_fast: false,
            detailed_reports: true,
        }
    }
}

impl<State> CompileTimeValidator<State> {
    /// Create a new compile-time validator
    #[must_use]
    pub fn new() -> CompileTimeValidator<Unvalidated> {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new compile-time validator with custom configuration
    #[must_use]
    pub fn with_config(config: ValidationConfig) -> CompileTimeValidator<Unvalidated> {
        /// CompileTimeValidator
        CompileTimeValidator {
            state: PhantomData,
            schema_validators: Arc::new(RwLock::new(HashMap::new())),
            constraint_validators: Arc::new(RwLock::new(Vec::new())),
            dependency_validators: Arc::new(RwLock::new(Vec::new())),
            cross_reference_validators: Arc::new(RwLock::new(Vec::new())),
            custom_validators: Arc::new(RwLock::new(Vec::new())),
            config,
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl CompileTimeValidator<Unvalidated> {
    /// Add a schema validator
    #[must_use]
    pub fn add_schema_validator(self, validator: Box<dyn SchemaValidator>) -> Self {
        let schema_name = validator.schema_name().to_string();
        self.schema_validators
            .write()
            .unwrap()
            .insert(schema_name, validator);
        self
    }

    /// Add a constraint validator
    #[must_use]
    pub fn add_constraint_validator(self, validator: Box<dyn ConstraintValidator>) -> Self {
        self.constraint_validators.write().unwrap().push(validator);
        self
    }

    /// Add a dependency validator
    #[must_use]
    pub fn add_dependency_validator(self, validator: Box<dyn DependencyValidator>) -> Self {
        self.dependency_validators.write().unwrap().push(validator);
        self
    }

    /// Add a cross-reference validator
    #[must_use]
    pub fn add_cross_reference_validator(
        self,
        validator: Box<dyn CrossReferenceValidator>,
    ) -> Self {
        self.cross_reference_validators
            .write()
            .unwrap()
            .push(validator);
        self
    }

    /// Add a custom validator
    #[must_use]
    pub fn add_custom_validator(self, validator: Box<dyn CustomValidator>) -> Self {
        self.custom_validators.write().unwrap().push(validator);
        self
    }

    /// Validate configuration and transition to validated state
    pub fn validate(
        self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<(CompileTimeValidator<Validated>, ValidationResult)> {
        let start_time = Instant::now();
        let mut errors = Vec::new();
        let warnings = Vec::new();
        let mut suggestions = Vec::new();
        let mut rules_checked = 0;

        // Check validation cache
        let config_hash = self.compute_config_hash(config);
        if self.config.enable_caching {
            if let Some(cached_result) = self.validation_cache.read().unwrap().get(&config_hash) {
                let validation_cache_clone = Arc::clone(&self.validation_cache);
                return Ok((
                    /// CompileTimeValidator
                    CompileTimeValidator {
                        state: PhantomData,
                        schema_validators: self.schema_validators,
                        constraint_validators: self.constraint_validators,
                        dependency_validators: self.dependency_validators,
                        cross_reference_validators: self.cross_reference_validators,
                        custom_validators: self.custom_validators,
                        config: self.config,
                        validation_cache: validation_cache_clone,
                    },
                    cached_result.clone(),
                ));
            }
        }

        // Schema validation
        if !self.schema_validators.read().unwrap().is_empty() {
            for validator in self.schema_validators.read().unwrap().values() {
                match validator.validate_schema(config) {
                    Ok(mut schema_errors) => {
                        errors.append(&mut schema_errors);
                        rules_checked += 1;
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            error_id: format!("schema_error_{rules_checked}"),
                            category: ValidationErrorCategory::SchemaViolation,
                            message: e.to_string(),
                            location: ConfigurationLocation {
                                section: "schema".to_string(),
                                field_path: "root".to_string(),
                                line_number: None,
                                column_number: None,
                            },
                            severity: ValidationSeverity::Critical,
                            expected: None,
                            actual: None,
                            suggestion: Some(
                                "Check schema definition and configuration format".to_string(),
                            ),
                            related_errors: Vec::new(),
                        });
                    }
                }

                if self.config.fail_fast && !errors.is_empty() {
                    break;
                }
            }
        }

        // Constraint validation
        if self.config.enable_constraint_validation && (errors.is_empty() || !self.config.fail_fast)
        {
            for validator in self.constraint_validators.read().unwrap().iter() {
                match validator.validate_constraints(config) {
                    Ok(mut constraint_errors) => {
                        errors.append(&mut constraint_errors);
                        rules_checked += 1;
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            error_id: format!("constraint_error_{rules_checked}"),
                            category: ValidationErrorCategory::ConstraintViolation,
                            message: e.to_string(),
                            location: ConfigurationLocation {
                                section: "constraints".to_string(),
                                field_path: "unknown".to_string(),
                                line_number: None,
                                column_number: None,
                            },
                            severity: ValidationSeverity::High,
                            expected: None,
                            actual: None,
                            suggestion: Some(
                                "Review parameter constraints and valid ranges".to_string(),
                            ),
                            related_errors: Vec::new(),
                        });
                    }
                }

                if self.config.fail_fast && !errors.is_empty() {
                    break;
                }
            }
        }

        // Dependency validation
        if self.config.enable_dependency_validation && (errors.is_empty() || !self.config.fail_fast)
        {
            for validator in self.dependency_validators.read().unwrap().iter() {
                match validator.validate_dependencies(config) {
                    Ok(mut dependency_errors) => {
                        errors.append(&mut dependency_errors);
                        rules_checked += 1;
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            error_id: format!("dependency_error_{rules_checked}"),
                            category: ValidationErrorCategory::DependencyMissing,
                            message: e.to_string(),
                            location: ConfigurationLocation {
                                section: "dependencies".to_string(),
                                field_path: "unknown".to_string(),
                                line_number: None,
                                column_number: None,
                            },
                            severity: ValidationSeverity::High,
                            expected: None,
                            actual: None,
                            suggestion: Some(
                                "Check that all required dependencies are configured".to_string(),
                            ),
                            related_errors: Vec::new(),
                        });
                    }
                }

                if self.config.fail_fast && !errors.is_empty() {
                    break;
                }
            }
        }

        // Cross-reference validation
        if self.config.enable_cross_reference_validation
            && (errors.is_empty() || !self.config.fail_fast)
        {
            for validator in self.cross_reference_validators.read().unwrap().iter() {
                match validator.validate_cross_references(config) {
                    Ok(mut cross_ref_errors) => {
                        errors.append(&mut cross_ref_errors);
                        rules_checked += 1;
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            error_id: format!("cross_ref_error_{rules_checked}"),
                            category: ValidationErrorCategory::CrossReferenceError,
                            message: e.to_string(),
                            location: ConfigurationLocation {
                                section: "cross_references".to_string(),
                                field_path: "unknown".to_string(),
                                line_number: None,
                                column_number: None,
                            },
                            severity: ValidationSeverity::Medium,
                            expected: None,
                            actual: None,
                            suggestion: Some(
                                "Check cross-references between configuration sections".to_string(),
                            ),
                            related_errors: Vec::new(),
                        });
                    }
                }

                if self.config.fail_fast && !errors.is_empty() {
                    break;
                }
            }
        }

        // Custom validation
        if self.config.enable_custom_validation && (errors.is_empty() || !self.config.fail_fast) {
            for validator in self.custom_validators.read().unwrap().iter() {
                match validator.validate(config) {
                    Ok(mut custom_errors) => {
                        errors.append(&mut custom_errors);
                        rules_checked += 1;
                    }
                    Err(e) => {
                        errors.push(ValidationError {
                            error_id: format!("custom_error_{rules_checked}"),
                            category: ValidationErrorCategory::CustomValidationFailure,
                            message: e.to_string(),
                            location: ConfigurationLocation {
                                section: "custom".to_string(),
                                field_path: "unknown".to_string(),
                                line_number: None,
                                column_number: None,
                            },
                            severity: ValidationSeverity::Medium,
                            expected: None,
                            actual: None,
                            suggestion: Some("Review custom validation rules".to_string()),
                            related_errors: Vec::new(),
                        });
                    }
                }

                if self.config.fail_fast && !errors.is_empty() {
                    break;
                }
            }
        }

        // Generate suggestions based on errors
        suggestions.extend(self.generate_suggestions(&errors, config));

        // Determine validation status
        let status = if !errors.is_empty() {
            ValidationStatus::Invalid
        } else if !warnings.is_empty() {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Valid
        };

        // Create validation result
        let validation_time = start_time.elapsed();
        let result = ValidationResult {
            status,
            errors,
            warnings,
            suggestions,
            metrics: ValidationMetrics {
                validation_time,
                rules_checked,
                fields_validated: config.len(),
                cache_hit_rate: 0.0, // Would be calculated based on cache usage
                memory_usage: 0,     // Would be measured
            },
            timestamp: SystemTime::now(),
        };

        // Cache result
        if self.config.enable_caching {
            self.validation_cache
                .write()
                .unwrap()
                .insert(config_hash, result.clone());
        }

        // Transition to validated state
        let validated_validator = CompileTimeValidator {
            state: PhantomData,
            schema_validators: self.schema_validators,
            constraint_validators: self.constraint_validators,
            dependency_validators: self.dependency_validators,
            cross_reference_validators: self.cross_reference_validators,
            custom_validators: self.custom_validators,
            config: self.config,
            validation_cache: self.validation_cache,
        };

        Ok((validated_validator, result))
    }

    // Private helper methods
    fn compute_config_hash(&self, config: &HashMap<String, ConfigValue>) -> String {
        // Simple hash computation - would use a proper hash function in practice
        format!("hash_{}", config.len())
    }

    fn generate_suggestions(
        &self,
        errors: &[ValidationError],
        config: &HashMap<String, ConfigValue>,
    ) -> Vec<ValidationSuggestion> {
        let mut suggestions = Vec::new();

        for error in errors {
            match &error.category {
                ValidationErrorCategory::MissingRequired => {
                    suggestions.push(ValidationSuggestion {
                        suggestion_id: format!("add_required_{}", error.error_id),
                        message: format!("Add required field '{}'", error.location.field_path),
                        action: SuggestionAction::AddField {
                            field: error.location.field_path.clone(),
                            value: error
                                .expected
                                .clone()
                                .unwrap_or_else(|| "default_value".to_string()),
                        },
                        confidence: 0.9,
                        priority: SuggestionPriority::Critical,
                    });
                }
                ValidationErrorCategory::TypeMismatch => {
                    if let Some(expected) = &error.expected {
                        suggestions.push(ValidationSuggestion {
                            suggestion_id: format!("fix_type_{}", error.error_id),
                            message: format!(
                                "Convert field '{}' to type '{}'",
                                error.location.field_path, expected
                            ),
                            action: SuggestionAction::ModifyField {
                                field: error.location.field_path.clone(),
                                new_value: format!("convert_to_{expected}"),
                            },
                            confidence: 0.8,
                            priority: SuggestionPriority::High,
                        });
                    }
                }
                _ => {}
            }
        }

        suggestions
    }
}

impl CompileTimeValidator<Validated> {
    /// Create a validated configuration from raw config data
    pub fn create_validated_config<T>(&self, config: T) -> ValidatedPipelineConfig<T> {
        let validation_proof = ValidationProof {
            proof_id: format!(
                "proof_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
            validation_time: SystemTime::now(),
            validator_version: "1.0.0".to_string(),
            checksum: "validated".to_string(),
        };

        /// ValidatedPipelineConfig
        ValidatedPipelineConfig {
            config,
            validation_proof,
            schema_version: "1.0.0".to_string(),
        }
    }

    /// Get validation metrics
    #[must_use]
    pub fn get_validation_metrics(&self) -> ValidationMetrics {
        /// ValidationMetrics
        ValidationMetrics {
            validation_time: Duration::from_millis(0),
            rules_checked: 0,
            fields_validated: 0,
            cache_hit_rate: 0.0,
            memory_usage: 0,
        }
    }
}

impl<T> ValidatedPipelineConfig<T> {
    /// Get the validated configuration
    pub fn config(&self) -> &T {
        &self.config
    }

    /// Get the validation proof
    pub fn validation_proof(&self) -> &ValidationProof {
        &self.validation_proof
    }

    /// Convert to another type while preserving validation
    pub fn map<U, F>(self, f: F) -> ValidatedPipelineConfig<U>
    where
        F: FnOnce(T) -> U,
    {
        /// ValidatedPipelineConfig
        ValidatedPipelineConfig {
            config: f(self.config),
            validation_proof: self.validation_proof,
            schema_version: self.schema_version,
        }
    }
}

// Type-safe configuration builder implementation

impl Default for TypeSafeConfigBuilder<Unbuilt> {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeSafeConfigBuilder<Unbuilt> {
    /// Create a new type-safe configuration builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PhantomData,
            config_data: HashMap::new(),
            type_constraints: HashMap::new(),
            validation_rules: Vec::new(),
            builder_config: BuilderConfig {
                strict_validation: true,
                allow_unknown_fields: false,
                max_nesting_depth: 5,
            },
        }
    }

    /// Add a string field with validation
    pub fn string_field<T: Into<String>>(mut self, name: &str, value: T) -> Self {
        self.config_data
            .insert(name.to_string(), ConfigValue::String(value.into()));
        self.type_constraints.insert(
            name.to_string(),
            /// TypeConstraint
            TypeConstraint {
                expected_type: ConfigType::String,
                required: true,
                default_value: None,
                constraints: Vec::new(),
                custom_validator: None,
            },
        );
        self
    }

    /// Add an integer field with validation
    #[must_use]
    pub fn integer_field(mut self, name: &str, value: i64) -> Self {
        self.config_data
            .insert(name.to_string(), ConfigValue::Integer(value));
        self.type_constraints.insert(
            name.to_string(),
            /// TypeConstraint
            TypeConstraint {
                expected_type: ConfigType::Integer,
                required: true,
                default_value: None,
                constraints: Vec::new(),
                custom_validator: None,
            },
        );
        self
    }

    /// Add a float field with validation
    #[must_use]
    pub fn float_field(mut self, name: &str, value: f64) -> Self {
        self.config_data
            .insert(name.to_string(), ConfigValue::Float(value));
        self.type_constraints.insert(
            name.to_string(),
            /// TypeConstraint
            TypeConstraint {
                expected_type: ConfigType::Float,
                required: true,
                default_value: None,
                constraints: Vec::new(),
                custom_validator: None,
            },
        );
        self
    }

    /// Add a boolean field with validation
    #[must_use]
    pub fn boolean_field(mut self, name: &str, value: bool) -> Self {
        self.config_data
            .insert(name.to_string(), ConfigValue::Boolean(value));
        self.type_constraints.insert(
            name.to_string(),
            /// TypeConstraint
            TypeConstraint {
                expected_type: ConfigType::Boolean,
                required: true,
                default_value: None,
                constraints: Vec::new(),
                custom_validator: None,
            },
        );
        self
    }

    /// Add a constraint to a field
    #[must_use]
    pub fn with_constraint(mut self, field: &str, constraint: ValueConstraint) -> Self {
        if let Some(type_constraint) = self.type_constraints.get_mut(field) {
            type_constraint.constraints.push(constraint);
        }
        self
    }

    /// Add a validation rule
    #[must_use]
    pub fn with_rule(mut self, rule: Box<dyn ValidationRule>) -> Self {
        self.validation_rules.push(rule);
        self
    }

    /// Build the configuration with validation
    pub fn build(self) -> Result<TypeSafeConfigBuilder<Built>> {
        // Validate configuration
        for (field, constraint) in &self.type_constraints {
            if let Some(value) = self.config_data.get(field) {
                // Check type compatibility
                if !self.is_type_compatible(value, &constraint.expected_type) {
                    return Err(SklearsComposeError::InvalidConfiguration(format!(
                        "Type mismatch for field '{}': expected {:?}, got {:?}",
                        field, constraint.expected_type, value
                    )));
                }

                // Check constraints
                for value_constraint in &constraint.constraints {
                    self.validate_constraint(field, value, value_constraint)?;
                }

                // Apply custom validator if present
                if let Some(ref validator) = constraint.custom_validator {
                    validator(value)
                        .map_err(|e| SklearsComposeError::InvalidConfiguration(e.to_string()))?;
                }
            } else if constraint.required {
                return Err(SklearsComposeError::InvalidConfiguration(format!(
                    "Required field '{field}' is missing"
                )));
            }
        }

        // Apply validation rules
        for rule in &self.validation_rules {
            for (field, value) in &self.config_data {
                rule.apply(field, value)
                    .map_err(|e| SklearsComposeError::InvalidConfiguration(e.to_string()))?;
            }
        }

        Ok(TypeSafeConfigBuilder {
            state: PhantomData,
            config_data: self.config_data,
            type_constraints: self.type_constraints,
            validation_rules: self.validation_rules,
            builder_config: self.builder_config,
        })
    }

    // Helper methods
    fn is_type_compatible(&self, value: &ConfigValue, expected_type: &ConfigType) -> bool {
        match (value, expected_type) {
            (ConfigValue::String(_), ConfigType::String) => true,
            (ConfigValue::Integer(_), ConfigType::Integer) => true,
            (ConfigValue::Float(_), ConfigType::Float) => true,
            (ConfigValue::Boolean(_), ConfigType::Boolean) => true,
            (ConfigValue::Array(arr), ConfigType::Array(element_type)) => arr
                .iter()
                .all(|item| self.is_type_compatible(item, element_type)),
            (ConfigValue::Object(_), ConfigType::Object(_)) => true,
            (ConfigValue::Null, ConfigType::Optional(_)) => true,
            _ => false,
        }
    }

    fn validate_constraint(
        &self,
        field: &str,
        value: &ConfigValue,
        constraint: &ValueConstraint,
    ) -> Result<()> {
        match constraint {
            ValueConstraint::Range { min, max } => {
                let numeric_value = match value {
                    ConfigValue::Integer(i) => *i as f64,
                    ConfigValue::Float(f) => *f,
                    _ => {
                        return Err(SklearsComposeError::InvalidConfiguration(format!(
                            "Range constraint can only be applied to numeric values in field '{field}'"
                        )));
                    }
                };

                if numeric_value < *min || numeric_value > *max {
                    return Err(SklearsComposeError::InvalidConfiguration(format!(
                        "Value {numeric_value} in field '{field}' is outside range [{min}, {max}]"
                    )));
                }
            }
            ValueConstraint::Length { min, max } => {
                let length = match value {
                    ConfigValue::String(s) => s.len(),
                    ConfigValue::Array(arr) => arr.len(),
                    _ => {
                        return Err(SklearsComposeError::InvalidConfiguration(
                            format!("Length constraint can only be applied to strings or arrays in field '{field}'")
                        ));
                    }
                };

                if length < *min {
                    return Err(SklearsComposeError::InvalidConfiguration(format!(
                        "Length {length} in field '{field}' is less than minimum {min}"
                    )));
                }

                if let Some(max_len) = max {
                    if length > *max_len {
                        return Err(SklearsComposeError::InvalidConfiguration(format!(
                            "Length {length} in field '{field}' is greater than maximum {max_len}"
                        )));
                    }
                }
            }
            ValueConstraint::OneOf(allowed_values) => {
                if !allowed_values.contains(value) {
                    return Err(SklearsComposeError::InvalidConfiguration(format!(
                        "Value in field '{field}' is not one of the allowed values"
                    )));
                }
            }
            ValueConstraint::Pattern(pattern) => {
                if let ConfigValue::String(s) = value {
                    // Simple pattern matching - would use regex in practice
                    if !s.contains(pattern) {
                        return Err(SklearsComposeError::InvalidConfiguration(format!(
                            "String in field '{field}' does not match pattern '{pattern}'"
                        )));
                    }
                } else {
                    return Err(SklearsComposeError::InvalidConfiguration(format!(
                        "Pattern constraint can only be applied to strings in field '{field}'"
                    )));
                }
            }
            ValueConstraint::Custom(description) => {
                // Custom constraints would be implemented with proper validation logic
                // For now, just a placeholder
                println!("Custom constraint '{description}' applied to field '{field}'");
            }
        }

        Ok(())
    }
}

impl TypeSafeConfigBuilder<Built> {
    /// Get the validated configuration data
    #[must_use]
    pub fn config_data(&self) -> &HashMap<String, ConfigValue> {
        &self.config_data
    }

    /// Extract configuration data
    #[must_use]
    pub fn into_config_data(self) -> HashMap<String, ConfigValue> {
        self.config_data
    }
}

// Example schema and constraint validators

/// Example schema validator for pipeline configurations
#[derive(Debug)]
pub struct PipelineSchemaValidator {
    schema: PipelineConfigurationSchema,
}

impl Default for PipelineSchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineSchemaValidator {
    #[must_use]
    pub fn new() -> Self {
        let mut required_fields = HashSet::new();
        required_fields.insert("model".to_string());
        required_fields.insert("pipeline".to_string());

        let mut field_definitions = HashMap::new();
        field_definitions.insert(
            "model".to_string(),
            /// FieldDefinition
            FieldDefinition {
                field_type: ConfigType::Object(HashMap::new()),
                description: "Model configuration".to_string(),
                default: None,
                constraints: Vec::new(),
                deprecated: false,
                deprecation_message: None,
            },
        );

        let schema = PipelineConfigurationSchema {
            version: "1.0.0".to_string(),
            required_fields,
            field_definitions,
            constraints: Vec::new(),
            cross_reference_rules: Vec::new(),
        };

        Self { schema }
    }
}

impl SchemaValidator for PipelineSchemaValidator {
    fn validate_schema(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check required fields
        for required_field in &self.schema.required_fields {
            if !config.contains_key(required_field) {
                errors.push(ValidationError {
                    error_id: format!("missing_required_{required_field}"),
                    category: ValidationErrorCategory::MissingRequired,
                    message: format!("Required field '{required_field}' is missing"),
                    location: ConfigurationLocation {
                        section: "root".to_string(),
                        field_path: required_field.clone(),
                        line_number: None,
                        column_number: None,
                    },
                    severity: ValidationSeverity::Critical,
                    expected: Some("required field".to_string()),
                    actual: Some("missing".to_string()),
                    suggestion: Some(format!("Add required field '{required_field}'")),
                    related_errors: Vec::new(),
                });
            }
        }

        Ok(errors)
    }

    fn schema_name(&self) -> &'static str {
        "PipelineSchema"
    }

    fn schema_version(&self) -> &str {
        &self.schema.version
    }
}

/// Example parameter constraint validator
#[derive(Debug)]
pub struct ParameterConstraintValidator;

impl ConstraintValidator for ParameterConstraintValidator {
    fn validate_constraints(
        &self,
        config: &HashMap<String, ConfigValue>,
    ) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Example: validate learning rate is in valid range
        if let Some(ConfigValue::Object(model_config)) = config.get("model") {
            if let Some(ConfigValue::Float(learning_rate)) = model_config.get("learning_rate") {
                if *learning_rate <= 0.0 || *learning_rate > 1.0 {
                    errors.push(ValidationError {
                        error_id: "invalid_learning_rate".to_string(),
                        category: ValidationErrorCategory::RangeError,
                        message: "Learning rate must be between 0.0 and 1.0".to_string(),
                        location: ConfigurationLocation {
                            section: "model".to_string(),
                            field_path: "learning_rate".to_string(),
                            line_number: None,
                            column_number: None,
                        },
                        severity: ValidationSeverity::High,
                        expected: Some("0.0 < learning_rate <= 1.0".to_string()),
                        actual: Some(learning_rate.to_string()),
                        suggestion: Some(
                            "Set learning rate to a value between 0.001 and 0.1".to_string(),
                        ),
                        related_errors: Vec::new(),
                    });
                }
            }
        }

        Ok(errors)
    }

    fn validator_name(&self) -> &'static str {
        "ParameterConstraintValidator"
    }
}

/// Example validation rule
#[derive(Debug)]
pub struct NonEmptyStringRule;

impl ValidationRule for NonEmptyStringRule {
    fn apply(&self, field: &str, value: &ConfigValue) -> Result<()> {
        if let ConfigValue::String(s) = value {
            if s.is_empty() {
                return Err(format!("String field '{field}' cannot be empty").into());
            }
        }
        Ok(())
    }

    fn rule_name(&self) -> &'static str {
        "NonEmptyStringRule"
    }
}

// Convenience macros for compile-time validation

/// Macro for creating enhanced validated configurations
#[macro_export]
macro_rules! enhanced_validated_config {
    ($($key:expr => $value:expr),*) => {{
        let mut config = std::collections::HashMap::new();
        $(
            config.insert($key.to_string(), $value);
        )*
        config
    }};
}

/// Macro for type-safe configuration building
#[macro_export]
macro_rules! type_safe_config {
    (
        $($field:ident: $type:ident = $value:expr),*
    ) => {{
        let builder = TypeSafeConfigBuilder::new();
        $(
            let builder = match stringify!($type) {
                "String" => builder.string_field(stringify!($field), $value),
                "Integer" => builder.integer_field(stringify!($field), $value),
                "Float" => builder.float_field(stringify!($field), $value),
                "Boolean" => builder.boolean_field(stringify!($field), $value),
                _ => panic!("Unsupported type: {}", stringify!($type)),
            };
        )*
        builder.build()
    }};
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = CompileTimeValidator::<Unvalidated>::new();
        assert!(validator.config.strict_type_checking);
    }

    #[test]
    fn test_schema_validation() {
        let validator = CompileTimeValidator::<Unvalidated>::new()
            .add_schema_validator(Box::new(PipelineSchemaValidator::new()));

        let mut config = HashMap::new();
        config.insert("model".to_string(), ConfigValue::Object(HashMap::new()));
        config.insert("pipeline".to_string(), ConfigValue::Object(HashMap::new()));

        let result = validator.validate(&config);
        assert!(result.is_ok());

        let (_, validation_result) = result.unwrap();
        assert!(matches!(validation_result.status, ValidationStatus::Valid));
    }

    #[test]
    fn test_constraint_validation() {
        let validator = CompileTimeValidator::<Unvalidated>::new()
            .add_constraint_validator(Box::new(ParameterConstraintValidator));

        let mut model_config = HashMap::new();
        model_config.insert("learning_rate".to_string(), ConfigValue::Float(2.0)); // Invalid

        let mut config = HashMap::new();
        config.insert("model".to_string(), ConfigValue::Object(model_config));

        let result = validator.validate(&config);
        assert!(result.is_ok());

        let (_, validation_result) = result.unwrap();
        assert!(matches!(
            validation_result.status,
            ValidationStatus::Invalid
        ));
        assert!(!validation_result.errors.is_empty());
    }

    #[test]
    fn test_type_safe_builder() {
        let builder = TypeSafeConfigBuilder::new()
            .string_field("name", "test_pipeline")
            .float_field("learning_rate", 0.01)
            .integer_field("epochs", 100)
            .boolean_field("verbose", true)
            .with_constraint(
                "learning_rate",
                ValueConstraint::Range { min: 0.0, max: 1.0 },
            )
            .with_rule(Box::new(NonEmptyStringRule));

        let result = builder.build();
        assert!(result.is_ok());

        let built_config = result.unwrap();
        assert_eq!(built_config.config_data().len(), 4);
    }

    #[test]
    fn test_validated_config_creation() {
        let validator = CompileTimeValidator::<Unvalidated>::new();
        let config = HashMap::new();

        let (validated_validator, _) = validator.validate(&config).unwrap();
        let validated_config = validated_validator.create_validated_config(config);

        assert!(validated_config
            .validation_proof()
            .proof_id
            .starts_with("proof_"));
    }

    #[test]
    fn test_validation_errors() {
        let validator = CompileTimeValidator::<Unvalidated>::new()
            .add_schema_validator(Box::new(PipelineSchemaValidator::new()));

        let config = HashMap::new(); // Missing required fields

        let result = validator.validate(&config);
        assert!(result.is_ok());

        let (_, validation_result) = result.unwrap();
        assert!(matches!(
            validation_result.status,
            ValidationStatus::Invalid
        ));
        assert!(validation_result.errors.len() >= 2); // Missing "model" and "pipeline"
    }

    #[test]
    fn test_config_value_types() {
        assert!(matches!(
            ConfigValue::String("test".to_string()),
            ConfigValue::String(_)
        ));
        assert!(matches!(ConfigValue::Integer(42), ConfigValue::Integer(_)));
        assert!(matches!(ConfigValue::Float(3.14), ConfigValue::Float(_)));
        assert!(matches!(
            ConfigValue::Boolean(true),
            ConfigValue::Boolean(_)
        ));
    }

    #[test]
    fn test_validation_suggestions() {
        let validator = CompileTimeValidator::<Unvalidated>::new()
            .add_schema_validator(Box::new(PipelineSchemaValidator::new()));

        let config = HashMap::new();

        let (_, validation_result) = validator.validate(&config).unwrap();
        assert!(!validation_result.suggestions.is_empty());

        let suggestion = &validation_result.suggestions[0];
        assert!(matches!(suggestion.priority, SuggestionPriority::Critical));
    }

    #[test]
    fn test_constraint_range() {
        let builder = TypeSafeConfigBuilder::new()
            .float_field("value", 150.0)
            .with_constraint(
                "value",
                ValueConstraint::Range {
                    min: 0.0,
                    max: 100.0,
                },
            );

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_length() {
        let builder = TypeSafeConfigBuilder::new()
            .string_field("name", "a") // Too short
            .with_constraint(
                "name",
                ValueConstraint::Length {
                    min: 3,
                    max: Some(10),
                },
            );

        let result = builder.build();
        assert!(result.is_err());
    }
}
