//! Schema management system
//!
//! This module provides comprehensive schema management capabilities including registry,
//! validation, evolution, and documentation for configuration schemas.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_validation::{SchemaConstraint, SchemaMigrationRule};

/// Schema management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaManager {
    /// Schema registry
    pub registry: SchemaRegistry,
    /// Schema validation
    pub validation: SchemaValidation,
    /// Schema evolution
    pub evolution: SchemaEvolution,
    /// Schema documentation
    pub documentation: SchemaDocumentation,
}

impl Default for SchemaManager {
    fn default() -> Self {
        Self {
            registry: SchemaRegistry::default(),
            validation: SchemaValidation::default(),
            evolution: SchemaEvolution::default(),
            documentation: SchemaDocumentation::default(),
        }
    }
}

/// Schema registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistry {
    /// Registered schemas
    pub schemas: HashMap<String, ConfigurationSchema>,
    /// Schema relationships
    pub relationships: HashMap<String, Vec<String>>,
    /// Schema metadata
    pub metadata: HashMap<String, SchemaMetadata>,
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self {
            schemas: HashMap::new(),
            relationships: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Configuration schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationSchema {
    /// Schema identifier
    pub schema_id: String,
    /// Schema name
    pub name: String,
    /// Schema version
    pub version: String,
    /// Schema definition
    pub definition: SchemaDefinition,
    /// Schema constraints
    pub constraints: Vec<SchemaConstraint>,
    /// Schema examples
    pub examples: Vec<String>,
}

/// Schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Schema type
    pub schema_type: String,
    /// Schema properties
    pub properties: HashMap<String, PropertyDefinition>,
    /// Required properties
    pub required: Vec<String>,
    /// Additional properties allowed
    pub additional_properties: bool,
}

/// Property definition in schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDefinition {
    /// Property type
    pub property_type: String,
    /// Property description
    pub description: String,
    /// Property format
    pub format: Option<String>,
    /// Default value
    pub default: Option<String>,
    /// Allowed values
    pub enum_values: Option<Vec<String>>,
    /// Property constraints
    pub constraints: Vec<PropertyConstraint>,
}

/// Property constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyConstraint {
    /// Minimum value
    Minimum(f64),
    /// Maximum value
    Maximum(f64),
    /// Minimum length
    MinLength(usize),
    /// Maximum length
    MaxLength(usize),
    /// Pattern constraint
    Pattern(String),
    /// Unique constraint
    Unique,
    /// Custom constraint
    Custom(String),
}

/// Schema metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMetadata {
    /// Schema author
    pub author: String,
    /// Creation date
    pub created_at: DateTime<Utc>,
    /// Last modified date
    pub modified_at: DateTime<Utc>,
    /// Schema description
    pub description: String,
    /// Schema tags
    pub tags: Vec<String>,
    /// Schema license
    pub license: Option<String>,
}

/// Schema validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidation {
    /// Validation enabled
    pub enabled: bool,
    /// Validation mode
    pub mode: SchemaValidationMode,
    /// Error handling
    pub error_handling: SchemaErrorHandling,
    /// Validation cache
    pub cache: SchemaValidationCache,
}

impl Default for SchemaValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: SchemaValidationMode::Strict,
            error_handling: SchemaErrorHandling::FailFast,
            cache: SchemaValidationCache::default(),
        }
    }
}

/// Schema validation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaValidationMode {
    /// Strict validation
    Strict,
    /// Lenient validation
    Lenient,
    /// Advisory validation
    Advisory,
    /// Custom validation
    Custom(String),
}

/// Schema error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaErrorHandling {
    /// Fail fast on first error
    FailFast,
    /// Collect all errors
    CollectAll,
    /// Log and continue
    LogAndContinue,
    /// Custom handling
    Custom(String),
}

/// Schema validation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationCache {
    /// Cache enabled
    pub enabled: bool,
    /// Cache size
    pub max_size: usize,
    /// Cache TTL
    pub ttl: Duration,
}

impl Default for SchemaValidationCache {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1000,
            ttl: Duration::hours(1),
        }
    }
}

/// Schema evolution management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaEvolution {
    /// Evolution enabled
    pub enabled: bool,
    /// Migration strategies
    pub migration_strategies: Vec<SchemaMigrationStrategy>,
    /// Compatibility checking
    pub compatibility_checking: CompatibilityChecking,
    /// Evolution history
    pub history: EvolutionHistory,
}

impl Default for SchemaEvolution {
    fn default() -> Self {
        Self {
            enabled: true,
            migration_strategies: vec![],
            compatibility_checking: CompatibilityChecking::default(),
            history: EvolutionHistory::default(),
        }
    }
}

/// Schema migration strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMigrationStrategy {
    /// Strategy name
    pub name: String,
    /// Source version pattern
    pub source_pattern: String,
    /// Target version pattern
    pub target_pattern: String,
    /// Migration rules
    pub migration_rules: Vec<SchemaMigrationRule>,
}

/// Compatibility checking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityChecking {
    /// Checking enabled
    pub enabled: bool,
    /// Compatibility level
    pub level: CompatibilityLevel,
    /// Breaking change detection
    pub breaking_change_detection: bool,
    /// Compatibility reports
    pub reports: bool,
}

impl Default for CompatibilityChecking {
    fn default() -> Self {
        Self {
            enabled: true,
            level: CompatibilityLevel::Backward,
            breaking_change_detection: true,
            reports: true,
        }
    }
}

/// Compatibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    /// Backward compatibility
    Backward,
    /// Forward compatibility
    Forward,
    /// Full compatibility
    Full,
    /// No compatibility checks
    None,
}

/// Evolution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionHistory {
    /// History entries
    pub entries: Vec<EvolutionEntry>,
    /// Maximum history size
    pub max_size: usize,
    /// History retention
    pub retention: Duration,
}

impl Default for EvolutionHistory {
    fn default() -> Self {
        Self {
            entries: vec![],
            max_size: 1000,
            retention: Duration::days(365),
        }
    }
}

/// Evolution history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEntry {
    /// Entry identifier
    pub id: String,
    /// Schema identifier
    pub schema_id: String,
    /// Source version
    pub from_version: String,
    /// Target version
    pub to_version: String,
    /// Migration applied
    pub migration: String,
    /// Migration timestamp
    pub timestamp: DateTime<Utc>,
    /// Migration status
    pub status: MigrationStatus,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStatus {
    /// Migration pending
    Pending,
    /// Migration in progress
    InProgress,
    /// Migration completed
    Completed,
    /// Migration failed
    Failed(String),
    /// Migration rolled back
    RolledBack,
}

/// Schema documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDocumentation {
    /// Documentation enabled
    pub enabled: bool,
    /// Documentation format
    pub format: DocumentationFormat,
    /// Auto-generation
    pub auto_generation: bool,
    /// Documentation templates
    pub templates: HashMap<String, String>,
}

impl Default for SchemaDocumentation {
    fn default() -> Self {
        Self {
            enabled: true,
            format: DocumentationFormat::Markdown,
            auto_generation: true,
            templates: HashMap::new(),
        }
    }
}

/// Documentation formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentationFormat {
    /// Markdown format
    Markdown,
    /// HTML format
    HTML,
    /// JSON format
    JSON,
    /// XML format
    XML,
    /// Custom format
    Custom(String),
}

/// Schema management operations
impl SchemaManager {
    /// Creates a new schema manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new schema
    pub fn register_schema(&mut self, schema: ConfigurationSchema) -> Result<(), String> {
        if self.registry.schemas.contains_key(&schema.schema_id) {
            return Err(format!("Schema with ID '{}' already exists", schema.schema_id));
        }

        let metadata = SchemaMetadata {
            author: "system".to_string(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
            description: format!("Schema for {}", schema.name),
            tags: vec![],
            license: None,
        };

        self.registry.metadata.insert(schema.schema_id.clone(), metadata);
        self.registry.schemas.insert(schema.schema_id.clone(), schema);
        Ok(())
    }

    /// Unregisters a schema
    pub fn unregister_schema(&mut self, schema_id: &str) -> Result<ConfigurationSchema, String> {
        self.registry.metadata.remove(schema_id);
        self.registry.relationships.remove(schema_id);

        self.registry.schemas
            .remove(schema_id)
            .ok_or_else(|| format!("Schema with ID '{}' not found", schema_id))
    }

    /// Gets a schema by ID
    pub fn get_schema(&self, schema_id: &str) -> Option<&ConfigurationSchema> {
        self.registry.schemas.get(schema_id)
    }

    /// Lists all schemas
    pub fn list_schemas(&self) -> Vec<&ConfigurationSchema> {
        self.registry.schemas.values().collect()
    }

    /// Gets schema metadata
    pub fn get_schema_metadata(&self, schema_id: &str) -> Option<&SchemaMetadata> {
        self.registry.metadata.get(schema_id)
    }

    /// Updates schema metadata
    pub fn update_schema_metadata(&mut self, schema_id: &str, metadata: SchemaMetadata) {
        self.registry.metadata.insert(schema_id.to_string(), metadata);
    }

    /// Adds a relationship between schemas
    pub fn add_schema_relationship(&mut self, from_schema: &str, to_schema: &str) {
        self.registry.relationships
            .entry(from_schema.to_string())
            .or_insert_with(Vec::new)
            .push(to_schema.to_string());
    }

    /// Gets related schemas
    pub fn get_related_schemas(&self, schema_id: &str) -> Vec<&str> {
        self.registry.relationships
            .get(schema_id)
            .map(|relations| relations.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Validates a configuration against a schema
    pub fn validate_configuration(&self, schema_id: &str, configuration: &HashMap<String, String>) -> Result<bool, String> {
        if !self.validation.enabled {
            return Ok(true);
        }

        let schema = self.get_schema(schema_id)
            .ok_or_else(|| format!("Schema '{}' not found", schema_id))?;

        // Validate required properties
        for required_prop in &schema.definition.required {
            if !configuration.contains_key(required_prop) {
                return Err(format!("Required property '{}' is missing", required_prop));
            }
        }

        // Validate property types and constraints
        for (prop_name, prop_value) in configuration {
            if let Some(prop_def) = schema.definition.properties.get(prop_name) {
                // Validate constraints
                for constraint in &prop_def.constraints {
                    match constraint {
                        PropertyConstraint::MinLength(min_len) => {
                            if prop_value.len() < *min_len {
                                return Err(format!("Property '{}' minimum length is {}", prop_name, min_len));
                            }
                        }
                        PropertyConstraint::MaxLength(max_len) => {
                            if prop_value.len() > *max_len {
                                return Err(format!("Property '{}' maximum length is {}", prop_name, max_len));
                            }
                        }
                        PropertyConstraint::Pattern(pattern) => {
                            // In a real implementation, you'd use regex
                            if !prop_value.contains(pattern) {
                                return Err(format!("Property '{}' does not match pattern '{}'", prop_name, pattern));
                            }
                        }
                        _ => {} // Other constraints would be implemented here
                    }
                }

                // Validate enum values
                if let Some(enum_values) = &prop_def.enum_values {
                    if !enum_values.contains(prop_value) {
                        return Err(format!("Property '{}' value '{}' is not in allowed values: {:?}", prop_name, prop_value, enum_values));
                    }
                }
            } else if !schema.definition.additional_properties {
                return Err(format!("Property '{}' is not allowed", prop_name));
            }
        }

        Ok(true)
    }

    /// Migrates a schema to a new version
    pub fn migrate_schema(&mut self, schema_id: &str, target_version: &str) -> Result<(), String> {
        if !self.evolution.enabled {
            return Err("Schema evolution is disabled".to_string());
        }

        let schema = self.get_schema(schema_id)
            .ok_or_else(|| format!("Schema '{}' not found", schema_id))?;

        let current_version = &schema.version;

        // Find migration strategy
        let strategy = self.evolution.migration_strategies
            .iter()
            .find(|s| s.source_pattern == *current_version && s.target_pattern == *target_version)
            .ok_or_else(|| format!("No migration strategy found from '{}' to '{}'", current_version, target_version))?;

        // Record migration attempt
        let migration_entry = EvolutionEntry {
            id: format!("{}_{}_to_{}", schema_id, current_version, target_version),
            schema_id: schema_id.to_string(),
            from_version: current_version.clone(),
            to_version: target_version.to_string(),
            migration: strategy.name.clone(),
            timestamp: Utc::now(),
            status: MigrationStatus::InProgress,
        };

        self.evolution.history.entries.push(migration_entry);

        // Apply migration (simplified implementation)
        if let Some(schema_mut) = self.registry.schemas.get_mut(schema_id) {
            schema_mut.version = target_version.to_string();
        }

        // Update migration status
        if let Some(entry) = self.evolution.history.entries.last_mut() {
            entry.status = MigrationStatus::Completed;
        }

        Ok(())
    }

    /// Checks compatibility between schema versions
    pub fn check_compatibility(&self, schema_id: &str, from_version: &str, to_version: &str) -> Result<bool, String> {
        if !self.evolution.compatibility_checking.enabled {
            return Ok(true);
        }

        match self.evolution.compatibility_checking.level {
            CompatibilityLevel::None => Ok(true),
            CompatibilityLevel::Backward | CompatibilityLevel::Forward | CompatibilityLevel::Full => {
                // Simplified compatibility check
                // In a real implementation, this would perform detailed schema comparison
                Ok(from_version <= to_version)
            }
        }
    }

    /// Generates documentation for a schema
    pub fn generate_documentation(&self, schema_id: &str) -> Result<String, String> {
        if !self.documentation.enabled {
            return Err("Schema documentation is disabled".to_string());
        }

        let schema = self.get_schema(schema_id)
            .ok_or_else(|| format!("Schema '{}' not found", schema_id))?;

        let metadata = self.get_schema_metadata(schema_id);

        match self.documentation.format {
            DocumentationFormat::Markdown => {
                let mut doc = format!("# Schema: {}\n\n", schema.name);
                doc.push_str(&format!("**Version:** {}\n\n", schema.version));

                if let Some(meta) = metadata {
                    doc.push_str(&format!("**Description:** {}\n\n", meta.description));
                    doc.push_str(&format!("**Author:** {}\n\n", meta.author));
                }

                doc.push_str("## Properties\n\n");
                for (prop_name, prop_def) in &schema.definition.properties {
                    doc.push_str(&format!("### {}\n\n", prop_name));
                    doc.push_str(&format!("- **Type:** {}\n", prop_def.property_type));
                    doc.push_str(&format!("- **Description:** {}\n", prop_def.description));

                    if let Some(default) = &prop_def.default {
                        doc.push_str(&format!("- **Default:** {}\n", default));
                    }

                    if let Some(enum_values) = &prop_def.enum_values {
                        doc.push_str(&format!("- **Allowed Values:** {:?}\n", enum_values));
                    }

                    doc.push('\n');
                }

                Ok(doc)
            }
            DocumentationFormat::JSON => {
                // Simplified JSON documentation
                Ok(format!(r#"{{"schema": "{}", "version": "{}", "properties": {}}}"#, schema.name, schema.version, schema.definition.properties.len()))
            }
            _ => Err("Unsupported documentation format".to_string()),
        }
    }

    /// Clears validation cache
    pub fn clear_validation_cache(&mut self) {
        // Cache clearing would be implemented here
    }

    /// Gets validation cache statistics
    pub fn get_cache_statistics(&self) -> (usize, Duration) {
        (self.validation.cache.max_size, self.validation.cache.ttl)
    }
}

impl ConfigurationSchema {
    /// Creates a new configuration schema
    pub fn new(schema_id: String, name: String, version: String) -> Self {
        Self {
            schema_id,
            name,
            version,
            definition: SchemaDefinition {
                schema_type: "object".to_string(),
                properties: HashMap::new(),
                required: vec![],
                additional_properties: true,
            },
            constraints: vec![],
            examples: vec![],
        }
    }

    /// Adds a property to the schema
    pub fn add_property(mut self, name: String, property: PropertyDefinition) -> Self {
        self.definition.properties.insert(name, property);
        self
    }

    /// Marks a property as required
    pub fn require_property(mut self, name: String) -> Self {
        if !self.definition.required.contains(&name) {
            self.definition.required.push(name);
        }
        self
    }

    /// Sets whether additional properties are allowed
    pub fn allow_additional_properties(mut self, allow: bool) -> Self {
        self.definition.additional_properties = allow;
        self
    }

    /// Adds a constraint to the schema
    pub fn add_constraint(mut self, constraint: SchemaConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Adds an example to the schema
    pub fn add_example(mut self, example: String) -> Self {
        self.examples.push(example);
        self
    }
}

impl PropertyDefinition {
    /// Creates a new property definition
    pub fn new(property_type: String, description: String) -> Self {
        Self {
            property_type,
            description,
            format: None,
            default: None,
            enum_values: None,
            constraints: vec![],
        }
    }

    /// Sets the property format
    pub fn with_format(mut self, format: String) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the default value
    pub fn with_default(mut self, default: String) -> Self {
        self.default = Some(default);
        self
    }

    /// Sets allowed enum values
    pub fn with_enum_values(mut self, enum_values: Vec<String>) -> Self {
        self.enum_values = Some(enum_values);
        self
    }

    /// Adds a constraint
    pub fn with_constraint(mut self, constraint: PropertyConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }
}