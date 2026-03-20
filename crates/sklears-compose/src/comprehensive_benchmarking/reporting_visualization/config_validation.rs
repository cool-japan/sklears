//! Configuration validation system
//!
//! This module provides comprehensive validation capabilities including rules, schemas,
//! custom validators, caching, and migration support for configuration management.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::Duration;

use super::config_presets::{ValidationStatus, ValidationError, ValidationSeverity};

/// Configuration validation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationValidation {
    /// Validation enabled
    pub validation_enabled: bool,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
    /// Strict validation mode
    pub strict_mode: bool,
    /// Validation schemas
    pub schemas: ValidationSchemas,
    /// Custom validators
    pub custom_validators: Vec<CustomValidator>,
    /// Validation cache
    pub cache: ValidationCache,
}

impl Default for ConfigurationValidation {
    fn default() -> Self {
        Self {
            validation_enabled: true,
            validation_rules: vec![
                ValidationRule {
                    rule_id: "format_compatibility".to_string(),
                    rule_name: "Format Compatibility".to_string(),
                    rule_description: "Validates format compatibility with settings".to_string(),
                    rule_type: ValidationRuleType::FormatValidation,
                    conditions: vec![
                        ValidationCondition {
                            field: "format".to_string(),
                            operator: ValidationOperator::IsValid,
                            value: "supported_format".to_string(),
                            error_message: "Unsupported export format".to_string(),
                        },
                    ],
                    severity: ValidationSeverity::Error,
                    enabled: true,
                },
                ValidationRule {
                    rule_id: "quality_range".to_string(),
                    rule_name: "Quality Range".to_string(),
                    rule_description: "Validates quality settings are within acceptable range".to_string(),
                    rule_type: ValidationRuleType::RangeValidation,
                    conditions: vec![
                        ValidationCondition {
                            field: "quality".to_string(),
                            operator: ValidationOperator::InRange,
                            value: "1-100".to_string(),
                            error_message: "Quality must be between 1 and 100".to_string(),
                        },
                    ],
                    severity: ValidationSeverity::Warning,
                    enabled: true,
                },
            ],
            strict_mode: false,
            schemas: ValidationSchemas::default(),
            custom_validators: vec![],
            cache: ValidationCache::default(),
        }
    }
}

/// Validation rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Unique rule identifier
    pub rule_id: String,
    /// Human-readable rule name
    pub rule_name: String,
    /// Rule description
    pub rule_description: String,
    /// Type of validation rule
    pub rule_type: ValidationRuleType,
    /// Validation conditions
    pub conditions: Vec<ValidationCondition>,
    /// Severity of validation failure
    pub severity: ValidationSeverity,
    /// Whether rule is enabled
    pub enabled: bool,
}

/// Types of validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Format-specific validation
    FormatValidation,
    /// Range validation
    RangeValidation,
    /// Required field validation
    RequiredValidation,
    /// Pattern validation (regex)
    PatternValidation,
    /// Cross-field validation
    CrossFieldValidation,
    /// Custom validation
    CustomValidation,
    /// Performance validation
    PerformanceValidation,
    /// Security validation
    SecurityValidation,
}

/// Validation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCondition {
    /// Field to validate
    pub field: String,
    /// Validation operator
    pub operator: ValidationOperator,
    /// Expected value or pattern
    pub value: String,
    /// Error message for validation failure
    pub error_message: String,
}

/// Validation operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationOperator {
    /// Equals
    Equals,
    /// Not equals
    NotEquals,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// In range
    InRange,
    /// Matches pattern
    MatchesPattern,
    /// Is valid (custom validation)
    IsValid,
    /// Is required
    IsRequired,
    /// Is one of
    IsOneOf,
    /// Contains
    Contains,
}

/// Validation schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSchemas {
    /// Template schemas
    pub template_schemas: HashMap<String, TemplateSchema>,
    /// Preset schemas
    pub preset_schemas: HashMap<String, PresetSchema>,
    /// Format schemas
    pub format_schemas: HashMap<String, FormatSchema>,
    /// Schema versioning
    pub schema_versioning: SchemaVersioning,
}

impl Default for ValidationSchemas {
    fn default() -> Self {
        Self {
            template_schemas: HashMap::new(),
            preset_schemas: HashMap::new(),
            format_schemas: HashMap::new(),
            schema_versioning: SchemaVersioning::default(),
        }
    }
}

/// Template schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSchema {
    /// Schema identifier
    pub schema_id: String,
    /// Schema version
    pub version: String,
    /// Required fields
    pub required_fields: Vec<String>,
    /// Optional fields
    pub optional_fields: Vec<String>,
    /// Field definitions
    pub field_definitions: HashMap<String, FieldDefinition>,
    /// Schema constraints
    pub constraints: Vec<SchemaConstraint>,
}

/// Preset schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetSchema {
    /// Schema identifier
    pub schema_id: String,
    /// Schema version
    pub version: String,
    /// Base template schema
    pub base_template_schema: Option<String>,
    /// Allowed overrides
    pub allowed_overrides: Vec<String>,
    /// Validation rules
    pub validation_rules: Vec<String>,
}

/// Format schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatSchema {
    /// Schema identifier
    pub schema_id: String,
    /// Format type
    pub format_type: String,
    /// Schema version
    pub version: String,
    /// Format-specific fields
    pub format_fields: HashMap<String, FieldDefinition>,
    /// Format constraints
    pub format_constraints: Vec<FormatConstraint>,
}

/// Field definition in schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Field description
    pub description: String,
    /// Default value
    pub default_value: Option<String>,
    /// Allowed values
    pub allowed_values: Option<Vec<String>>,
    /// Value constraints
    pub constraints: Vec<FieldConstraint>,
}

/// Field types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// Boolean type
    Boolean,
    /// Array type
    Array(Box<FieldType>),
    /// Object type
    Object(HashMap<String, FieldDefinition>),
    /// Enum type
    Enum(Vec<String>),
    /// Custom type
    Custom(String),
}

/// Field constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldConstraint {
    /// Minimum value constraint
    MinValue(f64),
    /// Maximum value constraint
    MaxValue(f64),
    /// Minimum length constraint
    MinLength(usize),
    /// Maximum length constraint
    MaxLength(usize),
    /// Pattern constraint
    Pattern(String),
    /// Custom constraint
    Custom(String),
}

/// Schema constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaConstraint {
    /// Conditional constraint
    Conditional {
        /// Condition field
        condition_field: String,
        /// Condition value
        condition_value: String,
        /// Required fields when condition is met
        required_fields: Vec<String>,
    },
    /// Mutual exclusion constraint
    MutualExclusion {
        /// Mutually exclusive fields
        fields: Vec<String>,
    },
    /// Custom constraint
    Custom(String),
}

/// Format-specific constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormatConstraint {
    /// Resolution constraint
    Resolution {
        /// Minimum width
        min_width: Option<u32>,
        /// Maximum width
        max_width: Option<u32>,
        /// Minimum height
        min_height: Option<u32>,
        /// Maximum height
        max_height: Option<u32>,
    },
    /// Color depth constraint
    ColorDepth(Vec<u8>),
    /// Quality constraint
    Quality {
        /// Minimum quality
        min_quality: f64,
        /// Maximum quality
        max_quality: f64,
    },
    /// Custom format constraint
    Custom(String),
}

/// Schema versioning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersioning {
    /// Current schema versions
    pub current_versions: HashMap<String, String>,
    /// Schema migration rules
    pub migration_rules: Vec<SchemaMigrationRule>,
    /// Backward compatibility
    pub backward_compatibility: BackwardCompatibility,
}

impl Default for SchemaVersioning {
    fn default() -> Self {
        Self {
            current_versions: HashMap::new(),
            migration_rules: vec![],
            backward_compatibility: BackwardCompatibility::default(),
        }
    }
}

/// Schema migration rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMigrationRule {
    /// Source version
    pub from_version: String,
    /// Target version
    pub to_version: String,
    /// Migration steps
    pub migration_steps: Vec<MigrationStep>,
    /// Rollback steps
    pub rollback_steps: Vec<MigrationStep>,
}

/// Migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStep {
    /// Add field
    AddField {
        /// Field name
        field_name: String,
        /// Field definition
        field_definition: FieldDefinition,
    },
    /// Remove field
    RemoveField {
        /// Field name
        field_name: String,
    },
    /// Rename field
    RenameField {
        /// Old field name
        old_name: String,
        /// New field name
        new_name: String,
    },
    /// Transform field
    TransformField {
        /// Field name
        field_name: String,
        /// Transformation function
        transformation: String,
    },
    /// Custom migration step
    Custom(String),
}

/// Backward compatibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardCompatibility {
    /// Support legacy versions
    pub support_legacy: bool,
    /// Supported versions
    pub supported_versions: Vec<String>,
    /// Deprecation warnings
    pub deprecation_warnings: bool,
    /// Auto-migration enabled
    pub auto_migration: bool,
}

impl Default for BackwardCompatibility {
    fn default() -> Self {
        Self {
            support_legacy: true,
            supported_versions: vec!["1.0.0".to_string()],
            deprecation_warnings: true,
            auto_migration: false,
        }
    }
}

/// Custom validator definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomValidator {
    /// Validator identifier
    pub validator_id: String,
    /// Validator name
    pub validator_name: String,
    /// Validator description
    pub description: String,
    /// Validator implementation
    pub implementation: ValidatorImplementation,
    /// Validator configuration
    pub configuration: HashMap<String, String>,
    /// Validator enabled
    pub enabled: bool,
}

/// Validator implementation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidatorImplementation {
    /// JavaScript function
    JavaScript(String),
    /// Python script
    Python(String),
    /// External executable
    External(String),
    /// Built-in validator
    BuiltIn(String),
    /// Custom implementation
    Custom(String),
}

/// Validation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCache {
    /// Cache enabled
    pub enabled: bool,
    /// Cache size limit
    pub max_entries: usize,
    /// Cache TTL (time to live)
    pub ttl: Duration,
    /// Cache statistics
    pub statistics: CacheStatistics,
}

impl Default for ValidationCache {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            ttl: Duration::hours(1),
            statistics: CacheStatistics::default(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Total requests
    pub total_requests: u64,
    /// Cache size
    pub current_size: usize,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            total_requests: 0,
            current_size: 0,
        }
    }
}

/// Configuration validation operations
impl ConfigurationValidation {
    /// Creates a new configuration validation system
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables or disables validation
    pub fn set_validation_enabled(&mut self, enabled: bool) {
        self.validation_enabled = enabled;
    }

    /// Enables or disables strict mode
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Adds a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.push(rule);
    }

    /// Removes a validation rule by ID
    pub fn remove_rule(&mut self, rule_id: &str) {
        self.validation_rules.retain(|rule| rule.rule_id != rule_id);
    }

    /// Enables or disables a validation rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) {
        if let Some(rule) = self.validation_rules.iter_mut().find(|r| r.rule_id == rule_id) {
            rule.enabled = enabled;
        }
    }

    /// Gets validation rules by type
    pub fn get_rules_by_type(&self, rule_type: &ValidationRuleType) -> Vec<&ValidationRule> {
        self.validation_rules
            .iter()
            .filter(|rule| rule.enabled && std::mem::discriminant(&rule.rule_type) == std::mem::discriminant(rule_type))
            .collect()
    }

    /// Validates a configuration value against rules
    pub fn validate_value(&self, field: &str, value: &str) -> ValidationStatus {
        if !self.validation_enabled {
            return ValidationStatus::Valid;
        }

        let mut errors = vec![];

        for rule in &self.validation_rules {
            if !rule.enabled {
                continue;
            }

            for condition in &rule.conditions {
                if condition.field == field {
                    let validation_result = self.validate_condition(condition, value);
                    if let ValidationStatus::Invalid(mut condition_errors) = validation_result {
                        errors.append(&mut condition_errors);
                    }
                }
            }
        }

        if errors.is_empty() {
            ValidationStatus::Valid
        } else {
            ValidationStatus::Invalid(errors)
        }
    }

    /// Validates a single condition
    fn validate_condition(&self, condition: &ValidationCondition, value: &str) -> ValidationStatus {
        let is_valid = match &condition.operator {
            ValidationOperator::Equals => value == condition.value,
            ValidationOperator::NotEquals => value != condition.value,
            ValidationOperator::GreaterThan => {
                if let (Ok(val), Ok(cond_val)) = (value.parse::<f64>(), condition.value.parse::<f64>()) {
                    val > cond_val
                } else {
                    false
                }
            }
            ValidationOperator::LessThan => {
                if let (Ok(val), Ok(cond_val)) = (value.parse::<f64>(), condition.value.parse::<f64>()) {
                    val < cond_val
                } else {
                    false
                }
            }
            ValidationOperator::InRange => {
                if let Some((min, max)) = condition.value.split_once('-') {
                    if let (Ok(val), Ok(min_val), Ok(max_val)) = (value.parse::<f64>(), min.parse::<f64>(), max.parse::<f64>()) {
                        val >= min_val && val <= max_val
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            ValidationOperator::MatchesPattern => {
                // In a real implementation, you'd use regex
                value.contains(&condition.value)
            }
            ValidationOperator::IsValid => {
                // Custom validation logic would go here
                true
            }
            ValidationOperator::IsRequired => !value.is_empty(),
            ValidationOperator::IsOneOf => {
                condition.value.split(',').any(|opt| opt.trim() == value)
            }
            ValidationOperator::Contains => value.contains(&condition.value),
        };

        if is_valid {
            ValidationStatus::Valid
        } else {
            ValidationStatus::Invalid(vec![ValidationError {
                code: format!("VALIDATION_{}", condition.operator.get_operator_code()),
                message: condition.error_message.clone(),
                severity: ValidationSeverity::Error,
                field: Some(condition.field.clone()),
                suggested_fix: None,
            }])
        }
    }

    /// Clears validation cache
    pub fn clear_cache(&mut self) {
        self.cache.statistics = CacheStatistics::default();
    }

    /// Gets cache hit rate
    pub fn get_cache_hit_rate(&self) -> f64 {
        if self.cache.statistics.total_requests == 0 {
            0.0
        } else {
            self.cache.statistics.hits as f64 / self.cache.statistics.total_requests as f64
        }
    }
}

impl ValidationOperator {
    /// Gets the operator code for error messages
    pub fn get_operator_code(&self) -> &'static str {
        match self {
            ValidationOperator::Equals => "EQUALS",
            ValidationOperator::NotEquals => "NOT_EQUALS",
            ValidationOperator::GreaterThan => "GREATER_THAN",
            ValidationOperator::LessThan => "LESS_THAN",
            ValidationOperator::InRange => "IN_RANGE",
            ValidationOperator::MatchesPattern => "MATCHES_PATTERN",
            ValidationOperator::IsValid => "IS_VALID",
            ValidationOperator::IsRequired => "IS_REQUIRED",
            ValidationOperator::IsOneOf => "IS_ONE_OF",
            ValidationOperator::Contains => "CONTAINS",
        }
    }
}

impl ValidationRule {
    /// Creates a new validation rule
    pub fn new(
        rule_id: String,
        rule_name: String,
        rule_type: ValidationRuleType,
        severity: ValidationSeverity,
    ) -> Self {
        Self {
            rule_id,
            rule_name,
            rule_description: String::new(),
            rule_type,
            conditions: vec![],
            severity,
            enabled: true,
        }
    }

    /// Adds a condition to the rule
    pub fn add_condition(mut self, condition: ValidationCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Sets the rule description
    pub fn with_description(mut self, description: String) -> Self {
        self.rule_description = description;
        self
    }

    /// Sets whether the rule is enabled
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl ValidationCondition {
    /// Creates a new validation condition
    pub fn new(
        field: String,
        operator: ValidationOperator,
        value: String,
        error_message: String,
    ) -> Self {
        Self {
            field,
            operator,
            value,
            error_message,
        }
    }
}

impl FieldDefinition {
    /// Creates a new field definition
    pub fn new(name: String, field_type: FieldType, description: String) -> Self {
        Self {
            name,
            field_type,
            description,
            default_value: None,
            allowed_values: None,
            constraints: vec![],
        }
    }

    /// Sets the default value
    pub fn with_default(mut self, default_value: String) -> Self {
        self.default_value = Some(default_value);
        self
    }

    /// Sets allowed values
    pub fn with_allowed_values(mut self, allowed_values: Vec<String>) -> Self {
        self.allowed_values = Some(allowed_values);
        self
    }

    /// Adds a constraint
    pub fn with_constraint(mut self, constraint: FieldConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }
}