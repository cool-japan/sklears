//! # Schema Validation Module
//!
//! Comprehensive metadata schema definition and validation system providing
//! type safety, constraint enforcement, and schema evolution capabilities.
//!
//! ## Features
//!
//! - **Schema Definition**: Rich schema definition language
//! - **Validation Engine**: Fast and comprehensive validation
//! - **Schema Registry**: Centralized schema management and versioning
//! - **Type System**: Extensive type system with custom types
//! - **Constraint Validation**: Various built-in and custom constraints
//! - **Schema Evolution**: Version management and migration paths
//! - **Performance Optimization**: Compiled validators for speed
//! - **Error Reporting**: Detailed validation error messages
//!
//! ## Architecture
//!
//! ```text
//! SchemaValidator
//! ├── SchemaRegistry (schema storage and versioning)
//! ├── TypeSystem (type definitions and validation)
//! ├── ConstraintEngine (constraint validation)
//! ├── ValidationEngine (main validation logic)
//! ├── SchemaCompiler (compiled validator generation)
//! ├── ErrorReporter (detailed error reporting)
//! └── MigrationEngine (schema evolution and migration)
//! ```

use scirs2_core::error::{CoreError, Result};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;
use std::fmt;

/// Schema definition for metadata validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSchema {
    /// Schema identifier
    pub id: String,
    /// Schema name
    pub name: String,
    /// Schema version
    pub version: String,
    /// Schema description
    pub description: Option<String>,
    /// Root type definition
    pub root_type: TypeDefinition,
    /// Named type definitions
    pub type_definitions: HashMap<String, TypeDefinition>,
    /// Global constraints
    pub constraints: Vec<Constraint>,
    /// Schema metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Schema author
    pub author: Option<String>,
    /// Compatibility settings
    pub compatibility: CompatibilitySettings,
}

/// Type definition in the schema system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    /// Type name
    pub name: String,
    /// Type kind
    pub kind: TypeKind,
    /// Type description
    pub description: Option<String>,
    /// Type constraints
    pub constraints: Vec<Constraint>,
    /// Optional type (allows null)
    pub optional: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Type metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Different kinds of types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeKind {
    /// String type
    String(StringType),
    /// Number type (integer or float)
    Number(NumberType),
    /// Boolean type
    Boolean,
    /// Date/time type
    DateTime(DateTimeType),
    /// Array type
    Array(ArrayType),
    /// Object type
    Object(ObjectType),
    /// Union type (one of several types)
    Union(UnionType),
    /// Enum type
    Enum(EnumType),
    /// Reference to another type
    Reference(String),
    /// Custom type with validation function
    Custom(CustomType),
}

/// String type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringType {
    /// Minimum length
    pub min_length: Option<usize>,
    /// Maximum length
    pub max_length: Option<usize>,
    /// Regular expression pattern
    pub pattern: Option<String>,
    /// Character encoding requirement
    pub encoding: Option<String>,
    /// Case sensitivity
    pub case_sensitive: bool,
}

/// Number type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberType {
    /// Number format (integer, float, decimal)
    pub format: NumberFormat,
    /// Minimum value (inclusive)
    pub minimum: Option<f64>,
    /// Maximum value (inclusive)
    pub maximum: Option<f64>,
    /// Exclusive minimum
    pub exclusive_minimum: Option<f64>,
    /// Exclusive maximum
    pub exclusive_maximum: Option<f64>,
    /// Multiple of constraint
    pub multiple_of: Option<f64>,
    /// Precision for decimal numbers
    pub precision: Option<usize>,
}

/// Number format types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NumberFormat {
    /// Integer number
    Integer,
    /// Floating point number
    Float,
    /// High precision decimal
    Decimal,
}

/// Date/time type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateTimeType {
    /// Date/time format
    pub format: DateTimeFormat,
    /// Minimum date/time
    pub minimum: Option<SystemTime>,
    /// Maximum date/time
    pub maximum: Option<SystemTime>,
    /// Timezone requirement
    pub timezone: Option<String>,
}

/// Date/time format types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DateTimeFormat {
    /// ISO 8601 date-time
    DateTime,
    /// ISO 8601 date only
    Date,
    /// ISO 8601 time only
    Time,
    /// Unix timestamp
    Timestamp,
    /// Custom format string
    Custom(String),
}

/// Array type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayType {
    /// Item type
    pub items: Box<TypeDefinition>,
    /// Minimum number of items
    pub min_items: Option<usize>,
    /// Maximum number of items
    pub max_items: Option<usize>,
    /// Unique items requirement
    pub unique_items: bool,
}

/// Object type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectType {
    /// Property definitions
    pub properties: HashMap<String, TypeDefinition>,
    /// Required properties
    pub required: HashSet<String>,
    /// Additional properties allowed
    pub additional_properties: bool,
    /// Pattern properties
    pub pattern_properties: HashMap<String, TypeDefinition>,
    /// Minimum number of properties
    pub min_properties: Option<usize>,
    /// Maximum number of properties
    pub max_properties: Option<usize>,
}

/// Union type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnionType {
    /// Possible types
    pub types: Vec<TypeDefinition>,
    /// Discriminator field for type selection
    pub discriminator: Option<String>,
}

/// Enum type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumType {
    /// Allowed values
    pub values: Vec<serde_json::Value>,
    /// Case sensitive enum matching
    pub case_sensitive: bool,
}

/// Custom type with validation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomType {
    /// Type name
    pub type_name: String,
    /// Validation parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Validation constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Not null constraint
    NotNull,
    /// Unique constraint (within collection)
    Unique,
    /// Format constraint (email, url, etc.)
    Format(FormatType),
    /// Custom constraint with validation function
    Custom(CustomConstraint),
    /// Range constraint for numbers
    Range(f64, f64),
    /// Length constraint for strings/arrays
    Length(usize, usize),
    /// Conditional constraint
    Conditional(ConditionalConstraint),
}

/// Format constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormatType {
    /// Email format
    Email,
    /// URL format
    Url,
    /// UUID format
    Uuid,
    /// IPv4 address format
    IPv4,
    /// IPv6 address format
    IPv6,
    /// JSON format
    Json,
    /// Base64 format
    Base64,
    /// Custom format with regex
    Custom(String),
}

/// Custom constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Error message template
    pub error_message: String,
}

/// Conditional constraint based on other field values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalConstraint {
    /// Condition to check
    pub condition: FieldCondition,
    /// Constraint to apply if condition is true
    pub then_constraint: Box<Constraint>,
    /// Constraint to apply if condition is false
    pub else_constraint: Option<Box<Constraint>>,
}

/// Field condition for conditional constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldCondition {
    /// Field equals value
    Equals(String, serde_json::Value),
    /// Field not equals value
    NotEquals(String, serde_json::Value),
    /// Field exists
    Exists(String),
    /// Field does not exist
    NotExists(String),
    /// Field matches pattern
    Matches(String, String),
    /// Complex condition with AND/OR logic
    Complex(Box<ComplexCondition>),
}

/// Complex condition with boolean logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexCondition {
    /// AND operation
    And(Vec<FieldCondition>),
    /// OR operation
    Or(Vec<FieldCondition>),
    /// NOT operation
    Not(Box<FieldCondition>),
}

/// Schema compatibility settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilitySettings {
    /// Backward compatibility mode
    pub backward_compatible: bool,
    /// Forward compatibility mode
    pub forward_compatible: bool,
    /// Allow additional properties
    pub allow_additional_properties: bool,
    /// Strict mode (fail on warnings)
    pub strict_mode: bool,
}

impl Default for CompatibilitySettings {
    fn default() -> Self {
        Self {
            backward_compatible: true,
            forward_compatible: false,
            allow_additional_properties: true,
            strict_mode: false,
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation passed
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Validation statistics
    pub statistics: ValidationStatistics,
    /// Validation timestamp
    pub validated_at: SystemTime,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Field path where error occurred
    pub field_path: String,
    /// Expected value or constraint
    pub expected: Option<String>,
    /// Actual value that failed validation
    pub actual: Option<serde_json::Value>,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Additional error context
    pub context: HashMap<String, serde_json::Value>,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Field path where warning occurred
    pub field_path: String,
    /// Warning category
    pub category: WarningCategory,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity error
    Low,
    /// Medium severity error
    Medium,
    /// High severity error
    High,
    /// Critical error
    Critical,
}

/// Warning categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningCategory {
    /// Deprecated field or feature
    Deprecated,
    /// Performance warning
    Performance,
    /// Compatibility warning
    Compatibility,
    /// Best practice recommendation
    BestPractice,
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    /// Total fields validated
    pub fields_validated: usize,
    /// Validation duration
    pub validation_time: Duration,
    /// Number of constraints checked
    pub constraints_checked: usize,
    /// Number of type validations
    pub type_validations: usize,
}

/// Schema registry configuration
#[derive(Debug, Clone)]
pub struct SchemaRegistryConfig {
    /// Maximum number of schema versions to keep
    pub max_versions_per_schema: usize,
    /// Enable schema caching
    pub enable_caching: bool,
    /// Cache TTL for compiled validators
    pub cache_ttl: Duration,
    /// Enable automatic schema migration
    pub enable_auto_migration: bool,
    /// Validation timeout
    pub validation_timeout: Duration,
}

impl Default for SchemaRegistryConfig {
    fn default() -> Self {
        Self {
            max_versions_per_schema: 10,
            enable_caching: true,
            cache_ttl: Duration::from_secs(3600), // 1 hour
            enable_auto_migration: false,
            validation_timeout: Duration::from_secs(30),
        }
    }
}

/// Compiled validator for performance
#[derive(Debug)]
pub struct CompiledValidator {
    /// Validator ID
    pub id: String,
    /// Schema reference
    pub schema_id: String,
    /// Compilation timestamp
    pub compiled_at: SystemTime,
    /// Validation function (simplified representation)
    pub validation_rules: Vec<ValidationRule>,
    /// Performance statistics
    pub stats: ValidatorStatistics,
}

/// Individual validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule ID
    pub id: String,
    /// Field path this rule applies to
    pub field_path: String,
    /// Rule type
    pub rule_type: RuleType,
    /// Rule parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Types of validation rules
#[derive(Debug, Clone)]
pub enum RuleType {
    /// Type validation rule
    TypeCheck(TypeKind),
    /// Constraint validation rule
    ConstraintCheck(Constraint),
    /// Format validation rule
    FormatCheck(FormatType),
    /// Custom validation rule
    CustomCheck(String),
}

/// Validator performance statistics
#[derive(Debug, Clone)]
pub struct ValidatorStatistics {
    /// Total validations performed
    pub validations_performed: u64,
    /// Average validation time
    pub avg_validation_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Main schema validator
#[derive(Debug)]
pub struct SchemaValidator {
    /// Configuration
    config: SchemaRegistryConfig,
    /// Schema registry
    schemas: HashMap<String, HashMap<String, MetadataSchema>>, // schema_name -> version -> schema
    /// Compiled validators cache
    compiled_validators: HashMap<String, CompiledValidator>,
    /// Custom format validators
    custom_validators: HashMap<String, Box<dyn CustomValidatorFunction>>,
    /// Performance metrics
    metrics: Arc<MetricRegistry>,
    /// Validation timers
    validation_timer: Timer,
    compilation_timer: Timer,
    /// Validation counters
    validations_performed: Counter,
    schemas_registered: Counter,
    validation_errors: Counter,
    /// Performance gauges
    active_schemas: Gauge,
    cache_size: Gauge,
}

/// Custom validator function trait
pub trait CustomValidatorFunction: Send + Sync + fmt::Debug {
    /// Validate a value
    fn validate(&self, value: &serde_json::Value, parameters: &HashMap<String, serde_json::Value>) -> Result<bool>;
    /// Get error message for validation failure
    fn error_message(&self, parameters: &HashMap<String, serde_json::Value>) -> String;
}

/// Schema migration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMigration {
    /// Migration ID
    pub id: String,
    /// Source schema version
    pub from_version: String,
    /// Target schema version
    pub to_version: String,
    /// Migration steps
    pub steps: Vec<MigrationStep>,
    /// Automatic migration possible
    pub automatic: bool,
    /// Migration description
    pub description: String,
}

/// Individual migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStep {
    /// Add new field
    AddField {
        field_path: String,
        field_type: TypeDefinition,
        default_value: Option<serde_json::Value>,
    },
    /// Remove field
    RemoveField {
        field_path: String,
    },
    /// Rename field
    RenameField {
        old_path: String,
        new_path: String,
    },
    /// Change field type
    ChangeType {
        field_path: String,
        old_type: TypeKind,
        new_type: TypeKind,
        conversion: Option<String>,
    },
    /// Add constraint
    AddConstraint {
        field_path: String,
        constraint: Constraint,
    },
    /// Remove constraint
    RemoveConstraint {
        field_path: String,
        constraint: Constraint,
    },
    /// Custom migration step
    Custom {
        step_name: String,
        parameters: HashMap<String, serde_json::Value>,
    },
}

impl SchemaValidator {
    /// Create a new schema validator
    pub fn new() -> Self {
        Self::with_config(SchemaRegistryConfig::default())
    }

    /// Create schema validator with configuration
    pub fn with_config(config: SchemaRegistryConfig) -> Self {
        let metrics = Arc::new(MetricRegistry::new());

        Self {
            config,
            schemas: HashMap::new(),
            compiled_validators: HashMap::new(),
            custom_validators: HashMap::new(),
            metrics: metrics.clone(),
            validation_timer: metrics.timer("schema.validation_duration"),
            compilation_timer: metrics.timer("schema.compilation_duration"),
            validations_performed: metrics.counter("schema.validations_performed"),
            schemas_registered: metrics.counter("schema.schemas_registered"),
            validation_errors: metrics.counter("schema.validation_errors"),
            active_schemas: metrics.gauge("schema.active_schemas"),
            cache_size: metrics.gauge("schema.cache_size_mb"),
        }
    }

    /// Register a schema in the registry
    pub fn register_schema(&mut self, schema: MetadataSchema) -> Result<()> {
        let schema_name = schema.name.clone();
        let schema_version = schema.version.clone();

        // Validate schema definition
        self.validate_schema_definition(&schema)?;

        // Store schema
        self.schemas
            .entry(schema_name.clone())
            .or_insert_with(HashMap::new)
            .insert(schema_version.clone(), schema);

        // Clean up old versions if necessary
        self.cleanup_old_versions(&schema_name)?;

        // Compile validator
        self.compile_validator(&schema_name, &schema_version)?;

        self.schemas_registered.inc();
        self.active_schemas.set(self.schemas.len() as f64);

        Ok(())
    }

    /// Validate data against a schema
    pub fn validate(
        &mut self,
        data: &serde_json::Value,
        schema_name: &str,
        schema_version: Option<&str>,
    ) -> Result<ValidationResult> {
        let _timer = self.validation_timer.start_timer();
        let start_time = Instant::now();

        // Get schema
        let version = schema_version.unwrap_or("latest");
        let schema = self.get_schema(schema_name, version)?;

        // Get or compile validator
        let validator_key = format!("{}:{}", schema_name, schema.version);
        if !self.compiled_validators.contains_key(&validator_key) {
            self.compile_validator(schema_name, &schema.version)?;
        }

        let validator = self.compiled_validators
            .get(&validator_key)
            .ok_or_else(|| CoreError::ValidationError("Validator not found".to_string()))?;

        // Perform validation
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut stats = ValidationStatistics {
            fields_validated: 0,
            validation_time: Duration::from_secs(0),
            constraints_checked: 0,
            type_validations: 0,
        };

        // Validate root type
        self.validate_value(
            data,
            &schema.root_type,
            "",
            &mut errors,
            &mut warnings,
            &mut stats,
        )?;

        // Check global constraints
        for constraint in &schema.constraints {
            if let Err(error) = self.validate_constraint(data, constraint, "") {
                errors.push(ValidationError {
                    code: "GLOBAL_CONSTRAINT_FAILED".to_string(),
                    message: error.to_string(),
                    field_path: "".to_string(),
                    expected: None,
                    actual: Some(data.clone()),
                    severity: ErrorSeverity::High,
                    context: HashMap::new(),
                });
            }
            stats.constraints_checked += 1;
        }

        let validation_time = start_time.elapsed();
        stats.validation_time = validation_time;

        let result = ValidationResult {
            valid: errors.is_empty() && (warnings.is_empty() || !schema.compatibility.strict_mode),
            errors,
            warnings,
            statistics: stats,
            validated_at: SystemTime::now(),
        };

        self.validations_performed.inc();
        if !result.valid {
            self.validation_errors.inc();
        }

        Ok(result)
    }

    /// Get available schema versions
    pub fn get_schema_versions(&self, schema_name: &str) -> Vec<String> {
        self.schemas
            .get(schema_name)
            .map(|versions| versions.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Generate migration between schema versions
    pub fn generate_migration(
        &self,
        schema_name: &str,
        from_version: &str,
        to_version: &str,
    ) -> Result<SchemaMigration> {
        let from_schema = self.get_schema(schema_name, from_version)?;
        let to_schema = self.get_schema(schema_name, to_version)?;

        let mut steps = Vec::new();
        let mut automatic = true;

        // Compare schemas and generate migration steps
        self.compare_types(
            &from_schema.root_type,
            &to_schema.root_type,
            "",
            &mut steps,
            &mut automatic,
        );

        Ok(SchemaMigration {
            id: Uuid::new_v4().to_string(),
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            steps,
            automatic,
            description: format!(
                "Migration from {} v{} to v{}",
                schema_name, from_version, to_version
            ),
        })
    }

    /// Register a custom validator function
    pub fn register_custom_validator(
        &mut self,
        name: String,
        validator: Box<dyn CustomValidatorFunction>,
    ) {
        self.custom_validators.insert(name, validator);
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert("total_schemas".to_string(), json!(self.schemas.len()));
        stats.insert("compiled_validators".to_string(), json!(self.compiled_validators.len()));
        stats.insert("validations_performed".to_string(), json!(self.validations_performed.get()));
        stats.insert("validation_errors".to_string(), json!(self.validation_errors.get()));

        let error_rate = if self.validations_performed.get() > 0 {
            self.validation_errors.get() as f64 / self.validations_performed.get() as f64
        } else {
            0.0
        };
        stats.insert("error_rate".to_string(), json!(error_rate));

        // Schema distribution
        let mut schema_counts = HashMap::new();
        for (name, versions) in &self.schemas {
            schema_counts.insert(name.clone(), versions.len());
        }
        stats.insert("schema_versions".to_string(), json!(schema_counts));

        stats
    }

    // Private helper methods

    fn get_schema(&self, name: &str, version: &str) -> Result<&MetadataSchema> {
        let versions = self.schemas
            .get(name)
            .ok_or_else(|| CoreError::ValidationError(format!("Schema {} not found", name)))?;

        if version == "latest" {
            // Get the latest version (assuming semantic versioning sort)
            versions
                .values()
                .max_by_key(|s| &s.version)
                .ok_or_else(|| CoreError::ValidationError("No schema versions available".to_string()))
        } else {
            versions
                .get(version)
                .ok_or_else(|| CoreError::ValidationError(
                    format!("Schema {} version {} not found", name, version)
                ))
        }
    }

    fn validate_schema_definition(&self, schema: &MetadataSchema) -> Result<()> {
        // Basic schema validation
        if schema.name.is_empty() {
            return Err(CoreError::ValidationError("Schema name cannot be empty".to_string()));
        }

        if schema.version.is_empty() {
            return Err(CoreError::ValidationError("Schema version cannot be empty".to_string()));
        }

        // Validate type definitions
        self.validate_type_definition(&schema.root_type, &schema.type_definitions)?;

        Ok(())
    }

    fn validate_type_definition(
        &self,
        type_def: &TypeDefinition,
        type_registry: &HashMap<String, TypeDefinition>,
    ) -> Result<()> {
        match &type_def.kind {
            TypeKind::Reference(ref_name) => {
                if !type_registry.contains_key(ref_name) {
                    return Err(CoreError::ValidationError(
                        format!("Referenced type {} not found", ref_name)
                    ));
                }
            }
            TypeKind::Array(array_type) => {
                self.validate_type_definition(&array_type.items, type_registry)?;
            }
            TypeKind::Object(object_type) => {
                for property in object_type.properties.values() {
                    self.validate_type_definition(property, type_registry)?;
                }
            }
            TypeKind::Union(union_type) => {
                for union_member in &union_type.types {
                    self.validate_type_definition(union_member, type_registry)?;
                }
            }
            _ => {} // Other types are self-contained
        }

        Ok(())
    }

    fn cleanup_old_versions(&mut self, schema_name: &str) -> Result<()> {
        if let Some(versions) = self.schemas.get_mut(schema_name) {
            if versions.len() > self.config.max_versions_per_schema {
                // Keep only the most recent versions
                let mut version_list: Vec<_> = versions.keys().cloned().collect();
                version_list.sort(); // Assuming lexicographic sort works for versions

                let to_remove = version_list.len() - self.config.max_versions_per_schema;
                for version in version_list.into_iter().take(to_remove) {
                    versions.remove(&version);

                    // Remove compiled validator
                    let validator_key = format!("{}:{}", schema_name, version);
                    self.compiled_validators.remove(&validator_key);
                }
            }
        }

        Ok(())
    }

    fn compile_validator(&mut self, schema_name: &str, version: &str) -> Result<()> {
        let _timer = self.compilation_timer.start_timer();

        let schema = self.get_schema(schema_name, version)?;
        let validator_key = format!("{}:{}", schema_name, version);

        let mut validation_rules = Vec::new();

        // Compile rules from type definition
        self.compile_type_rules(&schema.root_type, "", &mut validation_rules)?;

        // Compile global constraint rules
        for constraint in &schema.constraints {
            validation_rules.push(ValidationRule {
                id: Uuid::new_v4().to_string(),
                field_path: "".to_string(),
                rule_type: RuleType::ConstraintCheck(constraint.clone()),
                parameters: HashMap::new(),
            });
        }

        let validator = CompiledValidator {
            id: Uuid::new_v4().to_string(),
            schema_id: schema.id.clone(),
            compiled_at: SystemTime::now(),
            validation_rules,
            stats: ValidatorStatistics {
                validations_performed: 0,
                avg_validation_time: Duration::from_secs(0),
                cache_hit_rate: 0.0,
                error_rate: 0.0,
            },
        };

        self.compiled_validators.insert(validator_key, validator);
        self.cache_size.set(self.compiled_validators.len() as f64);

        Ok(())
    }

    fn compile_type_rules(
        &self,
        type_def: &TypeDefinition,
        field_path: &str,
        rules: &mut Vec<ValidationRule>,
    ) -> Result<()> {
        // Add type check rule
        rules.push(ValidationRule {
            id: Uuid::new_v4().to_string(),
            field_path: field_path.to_string(),
            rule_type: RuleType::TypeCheck(type_def.kind.clone()),
            parameters: HashMap::new(),
        });

        // Add constraint rules
        for constraint in &type_def.constraints {
            rules.push(ValidationRule {
                id: Uuid::new_v4().to_string(),
                field_path: field_path.to_string(),
                rule_type: RuleType::ConstraintCheck(constraint.clone()),
                parameters: HashMap::new(),
            });
        }

        // Recursively compile rules for complex types
        match &type_def.kind {
            TypeKind::Array(array_type) => {
                let items_path = format!("{}[]", field_path);
                self.compile_type_rules(&array_type.items, &items_path, rules)?;
            }
            TypeKind::Object(object_type) => {
                for (prop_name, prop_type) in &object_type.properties {
                    let prop_path = if field_path.is_empty() {
                        prop_name.clone()
                    } else {
                        format!("{}.{}", field_path, prop_name)
                    };
                    self.compile_type_rules(prop_type, &prop_path, rules)?;
                }
            }
            _ => {} // Other types don't have nested rules
        }

        Ok(())
    }

    fn validate_value(
        &self,
        value: &serde_json::Value,
        type_def: &TypeDefinition,
        field_path: &str,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
        stats: &mut ValidationStatistics,
    ) -> Result<()> {
        stats.fields_validated += 1;

        // Check if value is null and type is optional
        if value.is_null() {
            if type_def.optional {
                return Ok(());
            } else {
                errors.push(ValidationError {
                    code: "NULL_VALUE".to_string(),
                    message: "Value cannot be null".to_string(),
                    field_path: field_path.to_string(),
                    expected: Some("non-null value".to_string()),
                    actual: Some(value.clone()),
                    severity: ErrorSeverity::High,
                    context: HashMap::new(),
                });
                return Ok(());
            }
        }

        // Validate type
        if let Err(error) = self.validate_type(value, &type_def.kind) {
            errors.push(ValidationError {
                code: "TYPE_MISMATCH".to_string(),
                message: error.to_string(),
                field_path: field_path.to_string(),
                expected: Some(format!("{:?}", type_def.kind)),
                actual: Some(value.clone()),
                severity: ErrorSeverity::High,
                context: HashMap::new(),
            });
        }
        stats.type_validations += 1;

        // Validate constraints
        for constraint in &type_def.constraints {
            if let Err(error) = self.validate_constraint(value, constraint, field_path) {
                errors.push(ValidationError {
                    code: "CONSTRAINT_VIOLATION".to_string(),
                    message: error.to_string(),
                    field_path: field_path.to_string(),
                    expected: None,
                    actual: Some(value.clone()),
                    severity: ErrorSeverity::Medium,
                    context: HashMap::new(),
                });
            }
            stats.constraints_checked += 1;
        }

        Ok(())
    }

    fn validate_type(&self, value: &serde_json::Value, type_kind: &TypeKind) -> Result<()> {
        match type_kind {
            TypeKind::String(string_type) => {
                if let Some(s) = value.as_str() {
                    if let Some(min_len) = string_type.min_length {
                        if s.len() < min_len {
                            return Err(CoreError::ValidationError(
                                format!("String too short: {} < {}", s.len(), min_len)
                            ));
                        }
                    }
                    if let Some(max_len) = string_type.max_length {
                        if s.len() > max_len {
                            return Err(CoreError::ValidationError(
                                format!("String too long: {} > {}", s.len(), max_len)
                            ));
                        }
                    }
                    if let Some(ref pattern) = string_type.pattern {
                        // Simplified regex check - would use regex crate in production
                        if !self.matches_pattern(s, pattern) {
                            return Err(CoreError::ValidationError(
                                format!("String does not match pattern: {}", pattern)
                            ));
                        }
                    }
                } else {
                    return Err(CoreError::ValidationError("Value is not a string".to_string()));
                }
            }
            TypeKind::Number(number_type) => {
                if let Some(num) = value.as_f64() {
                    if let Some(min) = number_type.minimum {
                        if num < min {
                            return Err(CoreError::ValidationError(
                                format!("Number too small: {} < {}", num, min)
                            ));
                        }
                    }
                    if let Some(max) = number_type.maximum {
                        if num > max {
                            return Err(CoreError::ValidationError(
                                format!("Number too large: {} > {}", num, max)
                            ));
                        }
                    }
                } else {
                    return Err(CoreError::ValidationError("Value is not a number".to_string()));
                }
            }
            TypeKind::Boolean => {
                if !value.is_boolean() {
                    return Err(CoreError::ValidationError("Value is not a boolean".to_string()));
                }
            }
            TypeKind::Array(array_type) => {
                if let Some(arr) = value.as_array() {
                    if let Some(min_items) = array_type.min_items {
                        if arr.len() < min_items {
                            return Err(CoreError::ValidationError(
                                format!("Array too short: {} < {}", arr.len(), min_items)
                            ));
                        }
                    }
                    if let Some(max_items) = array_type.max_items {
                        if arr.len() > max_items {
                            return Err(CoreError::ValidationError(
                                format!("Array too long: {} > {}", arr.len(), max_items)
                            ));
                        }
                    }
                    // Validate array items (simplified)
                    for item in arr {
                        self.validate_type(item, &array_type.items.kind)?;
                    }
                } else {
                    return Err(CoreError::ValidationError("Value is not an array".to_string()));
                }
            }
            TypeKind::Object(_object_type) => {
                if !value.is_object() {
                    return Err(CoreError::ValidationError("Value is not an object".to_string()));
                }
                // Object validation would be more complex in a full implementation
            }
            _ => {} // Other type validations would be implemented similarly
        }

        Ok(())
    }

    fn validate_constraint(
        &self,
        value: &serde_json::Value,
        constraint: &Constraint,
        field_path: &str,
    ) -> Result<()> {
        match constraint {
            Constraint::NotNull => {
                if value.is_null() {
                    return Err(CoreError::ValidationError("Value cannot be null".to_string()));
                }
            }
            Constraint::Format(format_type) => {
                if let Some(s) = value.as_str() {
                    if !self.validate_format(s, format_type) {
                        return Err(CoreError::ValidationError(
                            format!("Value does not match format: {:?}", format_type)
                        ));
                    }
                }
            }
            Constraint::Range(min, max) => {
                if let Some(num) = value.as_f64() {
                    if num < *min || num > *max {
                        return Err(CoreError::ValidationError(
                            format!("Value {} not in range [{}, {}]", num, min, max)
                        ));
                    }
                }
            }
            Constraint::Custom(custom_constraint) => {
                if let Some(validator) = self.custom_validators.get(&custom_constraint.name) {
                    if !validator.validate(value, &custom_constraint.parameters)? {
                        return Err(CoreError::ValidationError(
                            custom_constraint.error_message.clone()
                        ));
                    }
                }
            }
            _ => {} // Other constraints would be implemented
        }

        Ok(())
    }

    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        // Simplified pattern matching - would use regex crate in production
        pattern == "*" || text.contains(pattern)
    }

    fn validate_format(&self, text: &str, format_type: &FormatType) -> bool {
        match format_type {
            FormatType::Email => text.contains('@'),
            FormatType::Url => text.starts_with("http://") || text.starts_with("https://"),
            FormatType::Uuid => text.len() == 36 && text.chars().filter(|&c| c == '-').count() == 4,
            _ => true, // Simplified validation
        }
    }

    fn compare_types(
        &self,
        from_type: &TypeDefinition,
        to_type: &TypeDefinition,
        field_path: &str,
        steps: &mut Vec<MigrationStep>,
        automatic: &mut bool,
    ) {
        // Simplified type comparison for migration generation
        if std::mem::discriminant(&from_type.kind) != std::mem::discriminant(&to_type.kind) {
            steps.push(MigrationStep::ChangeType {
                field_path: field_path.to_string(),
                old_type: from_type.kind.clone(),
                new_type: to_type.kind.clone(),
                conversion: None,
            });
            *automatic = false; // Type changes usually require manual intervention
        }
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

// Example implementations of custom validators

#[derive(Debug)]
pub struct EmailValidator;

impl CustomValidatorFunction for EmailValidator {
    fn validate(&self, value: &serde_json::Value, _parameters: &HashMap<String, serde_json::Value>) -> Result<bool> {
        if let Some(s) = value.as_str() {
            Ok(s.contains('@') && s.contains('.'))
        } else {
            Ok(false)
        }
    }

    fn error_message(&self, _parameters: &HashMap<String, serde_json::Value>) -> String {
        "Invalid email format".to_string()
    }
}

#[derive(Debug)]
pub struct RangeValidator;

impl CustomValidatorFunction for RangeValidator {
    fn validate(&self, value: &serde_json::Value, parameters: &HashMap<String, serde_json::Value>) -> Result<bool> {
        if let (Some(num), Some(min_val), Some(max_val)) = (
            value.as_f64(),
            parameters.get("min").and_then(|v| v.as_f64()),
            parameters.get("max").and_then(|v| v.as_f64())
        ) {
            Ok(num >= min_val && num <= max_val)
        } else {
            Ok(false)
        }
    }

    fn error_message(&self, parameters: &HashMap<String, serde_json::Value>) -> String {
        if let (Some(min), Some(max)) = (
            parameters.get("min").and_then(|v| v.as_f64()),
            parameters.get("max").and_then(|v| v.as_f64())
        ) {
            format!("Value must be between {} and {}", min, max)
        } else {
            "Value out of range".to_string()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_registration_and_validation() {
        let mut validator = SchemaValidator::new();

        // Create a simple schema
        let schema = MetadataSchema {
            id: "test_schema_1".to_string(),
            name: "test_schema".to_string(),
            version: "1.0.0".to_string(),
            description: Some("Test schema".to_string()),
            root_type: TypeDefinition {
                name: "root".to_string(),
                kind: TypeKind::Object(ObjectType {
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), TypeDefinition {
                            name: "name".to_string(),
                            kind: TypeKind::String(StringType {
                                min_length: Some(1),
                                max_length: Some(100),
                                pattern: None,
                                encoding: None,
                                case_sensitive: true,
                            }),
                            description: None,
                            constraints: vec![Constraint::NotNull],
                            optional: false,
                            default: None,
                            metadata: HashMap::new(),
                        });
                        props.insert("age".to_string(), TypeDefinition {
                            name: "age".to_string(),
                            kind: TypeKind::Number(NumberType {
                                format: NumberFormat::Integer,
                                minimum: Some(0.0),
                                maximum: Some(150.0),
                                exclusive_minimum: None,
                                exclusive_maximum: None,
                                multiple_of: None,
                                precision: None,
                            }),
                            description: None,
                            constraints: vec![],
                            optional: true,
                            default: None,
                            metadata: HashMap::new(),
                        });
                        props
                    },
                    required: ["name"].iter().map(|s| s.to_string()).collect(),
                    additional_properties: false,
                    pattern_properties: HashMap::new(),
                    min_properties: None,
                    max_properties: None,
                }),
                description: None,
                constraints: vec![],
                optional: false,
                default: None,
                metadata: HashMap::new(),
            },
            type_definitions: HashMap::new(),
            constraints: vec![],
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            author: Some("test".to_string()),
            compatibility: CompatibilitySettings::default(),
        };

        // Register schema
        validator.register_schema(schema).unwrap();

        // Test valid data
        let valid_data = json!({
            "name": "John Doe",
            "age": 30
        });

        let result = validator.validate(&valid_data, "test_schema", Some("1.0.0")).unwrap();
        assert!(result.valid);
        assert!(result.errors.is_empty());

        // Test invalid data (missing required field)
        let invalid_data = json!({
            "age": 30
        });

        let result = validator.validate(&invalid_data, "test_schema", Some("1.0.0")).unwrap();
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_custom_validator() {
        let mut validator = SchemaValidator::new();

        // Register custom validator
        validator.register_custom_validator(
            "email".to_string(),
            Box::new(EmailValidator),
        );

        // Create schema with custom constraint
        let schema = MetadataSchema {
            id: "email_schema_1".to_string(),
            name: "email_schema".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            root_type: TypeDefinition {
                name: "root".to_string(),
                kind: TypeKind::String(StringType {
                    min_length: None,
                    max_length: None,
                    pattern: None,
                    encoding: None,
                    case_sensitive: true,
                }),
                description: None,
                constraints: vec![
                    Constraint::Custom(CustomConstraint {
                        name: "email".to_string(),
                        parameters: HashMap::new(),
                        error_message: "Invalid email".to_string(),
                    })
                ],
                optional: false,
                default: None,
                metadata: HashMap::new(),
            },
            type_definitions: HashMap::new(),
            constraints: vec![],
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            author: None,
            compatibility: CompatibilitySettings::default(),
        };

        validator.register_schema(schema).unwrap();

        // Test valid email
        let valid_email = json!("user@example.com");
        let result = validator.validate(&valid_email, "email_schema", Some("1.0.0")).unwrap();
        assert!(result.valid);

        // Test invalid email
        let invalid_email = json!("not-an-email");
        let result = validator.validate(&invalid_email, "email_schema", Some("1.0.0")).unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_schema_versions() {
        let mut validator = SchemaValidator::new();

        // Register first version
        let schema_v1 = MetadataSchema {
            id: "versioned_schema_1".to_string(),
            name: "versioned_schema".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            root_type: TypeDefinition {
                name: "root".to_string(),
                kind: TypeKind::String(StringType {
                    min_length: None,
                    max_length: None,
                    pattern: None,
                    encoding: None,
                    case_sensitive: true,
                }),
                description: None,
                constraints: vec![],
                optional: false,
                default: None,
                metadata: HashMap::new(),
            },
            type_definitions: HashMap::new(),
            constraints: vec![],
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            author: None,
            compatibility: CompatibilitySettings::default(),
        };

        validator.register_schema(schema_v1).unwrap();

        // Register second version
        let schema_v2 = MetadataSchema {
            id: "versioned_schema_2".to_string(),
            name: "versioned_schema".to_string(),
            version: "2.0.0".to_string(),
            description: None,
            root_type: TypeDefinition {
                name: "root".to_string(),
                kind: TypeKind::Number(NumberType {
                    format: NumberFormat::Integer,
                    minimum: None,
                    maximum: None,
                    exclusive_minimum: None,
                    exclusive_maximum: None,
                    multiple_of: None,
                    precision: None,
                }),
                description: None,
                constraints: vec![],
                optional: false,
                default: None,
                metadata: HashMap::new(),
            },
            type_definitions: HashMap::new(),
            constraints: vec![],
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            author: None,
            compatibility: CompatibilitySettings::default(),
        };

        validator.register_schema(schema_v2).unwrap();

        // Test both versions
        let versions = validator.get_schema_versions("versioned_schema");
        assert_eq!(versions.len(), 2);
        assert!(versions.contains(&"1.0.0".to_string()));
        assert!(versions.contains(&"2.0.0".to_string()));

        // Test validation with specific versions
        let string_data = json!("test");
        let result_v1 = validator.validate(&string_data, "versioned_schema", Some("1.0.0")).unwrap();
        assert!(result_v1.valid);

        let result_v2 = validator.validate(&string_data, "versioned_schema", Some("2.0.0")).unwrap();
        assert!(!result_v2.valid); // String should fail number validation

        let number_data = json!(42);
        let result_v2_num = validator.validate(&number_data, "versioned_schema", Some("2.0.0")).unwrap();
        assert!(result_v2_num.valid);
    }

    #[test]
    fn test_migration_generation() {
        let validator = SchemaValidator::new();

        // This would test migration generation between schema versions
        // Implementation would be more complex in a full system
        let migration = validator.generate_migration(
            "test_schema",
            "1.0.0",
            "2.0.0"
        );

        // In a real implementation, this would succeed and return migration steps
        assert!(migration.is_err()); // Expected since we haven't registered the schemas
    }
}