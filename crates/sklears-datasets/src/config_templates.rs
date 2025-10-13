//! Dataset configuration templates and YAML/JSON support
//!
//! This module provides a comprehensive configuration system for dataset generation
//! using YAML and JSON templates. It supports parameter inheritance, validation,
//! and experiment tracking.

use crate::traits::{ConfigValue, GeneratorConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Configuration system errors
#[derive(Error, Debug)]
pub enum ConfigTemplateError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parsing error: {0}")]
    Yaml(String),
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Parameter override error: {0}")]
    ParameterOverride(String),
    #[error("Inheritance cycle detected: {0}")]
    InheritanceCycle(String),
    #[error("Missing required parameter: {0}")]
    MissingParameter(String),
}

pub type ConfigTemplateResult<T> = Result<T, ConfigTemplateError>;

/// Dataset template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetTemplate {
    /// Template metadata
    pub metadata: TemplateMetadata,

    /// Base template to inherit from
    pub extends: Option<String>,

    /// Generator configuration
    pub generator: GeneratorTemplateConfig,

    /// Parameter definitions and constraints
    pub parameters: HashMap<String, ParameterTemplate>,

    /// Validation rules
    pub validation: Option<ValidationRules>,

    /// Export settings
    pub export: Option<ExportTemplate>,

    /// Experiment tracking settings
    pub experiment: Option<ExperimentTemplate>,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub created: Option<String>,
    pub modified: Option<String>,
    pub tags: Vec<String>,
    pub category: Option<String>,
}

/// Generator configuration in template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorTemplateConfig {
    pub name: String,
    pub n_samples: Option<usize>,
    pub n_features: Option<usize>,
    pub random_state: Option<u64>,
    pub parameters: HashMap<String, serde_yaml::Value>,
}

/// Parameter template with constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterTemplate {
    pub description: String,
    pub param_type: ParameterTypeTemplate,
    pub required: bool,
    pub default: Option<serde_yaml::Value>,
    pub constraints: Option<ParameterConstraints>,
    pub dependencies: Vec<String>,
}

/// Parameter type definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ParameterTypeTemplate {
    /// Integer

    Integer {

        min: Option<i64>,

        max: Option<i64>,
    },
    /// Float

    Float {

        min: Option<f64>,

        max: Option<f64>,
    },
    /// String

    String {

        pattern: Option<String>,
    },
    Boolean,
    Array {
        element_type: Box<ParameterTypeTemplate>,
    },
    Enum {
        values: Vec<serde_yaml::Value>,
    },
    Object {
        properties: HashMap<String, ParameterTemplate>,
    },
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub custom: Vec<String>,
}

/// Validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    pub dataset_size: Option<DatasetSizeConstraints>,
    pub statistical: Option<StatisticalConstraints>,
    pub custom: Vec<String>,
}

/// Dataset size constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSizeConstraints {
    pub min_samples: Option<usize>,
    pub max_samples: Option<usize>,
    pub min_features: Option<usize>,
    pub max_features: Option<usize>,
    pub max_memory_mb: Option<usize>,
}

/// Statistical constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConstraints {
    pub target_correlation: Option<f64>,
    pub noise_level: Option<f64>,
    pub class_balance: Option<ClassBalanceConstraints>,
}

/// Class balance constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassBalanceConstraints {
    pub min_ratio: f64,
    pub max_ratio: f64,
    pub enforce_balance: bool,
}

/// Export template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTemplate {
    pub formats: Vec<String>,
    pub filename_pattern: String,
    pub include_metadata: bool,
    pub compression: Option<String>,
}

/// Experiment tracking template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTemplate {
    pub name: String,
    pub description: String,
    pub metrics: Vec<String>,
    pub track_parameters: bool,
    pub track_outputs: bool,
    pub storage_path: Option<String>,
}

/// Template library for managing collections of templates
pub struct TemplateLibrary {
    templates: HashMap<String, DatasetTemplate>,
    search_paths: Vec<String>,
}

impl TemplateLibrary {
    /// Create a new template library
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            search_paths: vec![
                "./templates".to_string(),
                "~/.sklears/templates".to_string(),
            ],
        }
    }

    /// Add a search path for templates
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths
            .push(path.as_ref().to_string_lossy().to_string());
    }

    /// Load a template from file
    pub fn load_template<P: AsRef<Path>>(&mut self, path: P) -> ConfigTemplateResult<String> {
        let content = fs::read_to_string(&path)?;
        let template: DatasetTemplate = if path.as_ref().extension().and_then(|ext| ext.to_str())
            == Some("json")
        {
            serde_json::from_str(&content)?
        } else {
            serde_yaml::from_str(&content).map_err(|e| ConfigTemplateError::Yaml(e.to_string()))?
        };

        let name = template.metadata.name.clone();
        self.templates.insert(name.clone(), template);
        Ok(name)
    }

    /// Load all templates from a directory
    pub fn load_directory<P: AsRef<Path>>(&mut self, dir: P) -> ConfigTemplateResult<Vec<String>> {
        let mut loaded = Vec::new();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "yaml" || ext == "yml" || ext == "json" {
                            if let Ok(name) = self.load_template(&path) {
                                loaded.push(name);
                            }
                        }
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Register a template programmatically
    pub fn register_template(&mut self, template: DatasetTemplate) {
        let name = template.metadata.name.clone();
        self.templates.insert(name, template);
    }

    /// Get a template by name
    pub fn get_template(&self, name: &str) -> Option<&DatasetTemplate> {
        self.templates.get(name)
    }

    /// List all available templates
    pub fn list_templates(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }

    /// Search templates by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<String> {
        self.templates
            .iter()
            .filter(|(_, template)| template.metadata.tags.contains(&tag.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Search templates by category
    pub fn search_by_category(&self, category: &str) -> Vec<String> {
        self.templates
            .iter()
            .filter(|(_, template)| {
                template.metadata.category.as_ref() == Some(&category.to_string())
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Resolve template inheritance and create final configuration
    pub fn resolve_template(
        &self,
        name: &str,
        overrides: Option<HashMap<String, serde_yaml::Value>>,
    ) -> ConfigTemplateResult<GeneratorConfig> {
        let template = self
            .get_template(name)
            .ok_or_else(|| ConfigTemplateError::TemplateNotFound(name.to_string()))?;

        // Resolve inheritance chain
        let resolved = self.resolve_inheritance(template)?;

        // Create generator config
        let mut config = GeneratorConfig::new(
            resolved.generator.n_samples.unwrap_or(1000),
            resolved.generator.n_features.unwrap_or(10),
        );

        if let Some(seed) = resolved.generator.random_state {
            config = config.with_random_state(seed);
        }

        // Apply parameters from template
        for (key, value) in &resolved.generator.parameters {
            if let Ok(config_value) = self.convert_yaml_value(value) {
                config.set_parameter(key.clone(), config_value);
            }
        }

        // Apply overrides
        if let Some(overrides) = overrides {
            for (key, value) in overrides {
                if let Ok(config_value) = self.convert_yaml_value(&value) {
                    config.set_parameter(key, config_value);
                }
            }
        }

        // Validate configuration
        self.validate_config(&resolved, &config)?;

        Ok(config)
    }

    /// Resolve template inheritance
    fn resolve_inheritance(
        &self,
        template: &DatasetTemplate,
    ) -> ConfigTemplateResult<DatasetTemplate> {
        let mut resolved = template.clone();
        let mut visited = std::collections::HashSet::new();

        self.resolve_inheritance_recursive(&mut resolved, &mut visited)?;
        Ok(resolved)
    }

    fn resolve_inheritance_recursive(
        &self,
        template: &mut DatasetTemplate,
        visited: &mut std::collections::HashSet<String>,
    ) -> ConfigTemplateResult<()> {
        if let Some(ref base_name) = template.extends.clone() {
            if visited.contains(base_name) {
                return Err(ConfigTemplateError::InheritanceCycle(base_name.clone()));
            }

            visited.insert(base_name.clone());

            let base_template = self
                .get_template(base_name)
                .ok_or_else(|| ConfigTemplateError::TemplateNotFound(base_name.clone()))?;

            let mut base = base_template.clone();
            self.resolve_inheritance_recursive(&mut base, visited)?;

            // Merge base into current template
            self.merge_templates(&mut base, template);
            *template = base;

            visited.remove(base_name);
        }

        Ok(())
    }

    /// Merge two templates (child overrides parent)
    fn merge_templates(&self, parent: &mut DatasetTemplate, child: &DatasetTemplate) {
        // Merge generator config
        if child.generator.n_samples.is_some() {
            parent.generator.n_samples = child.generator.n_samples;
        }
        if child.generator.n_features.is_some() {
            parent.generator.n_features = child.generator.n_features;
        }
        if child.generator.random_state.is_some() {
            parent.generator.random_state = child.generator.random_state;
        }

        // Merge parameters
        for (key, value) in &child.generator.parameters {
            parent
                .generator
                .parameters
                .insert(key.clone(), value.clone());
        }

        // Merge parameter definitions
        for (key, value) in &child.parameters {
            parent.parameters.insert(key.clone(), value.clone());
        }

        // Override other fields
        if child.validation.is_some() {
            parent.validation = child.validation.clone();
        }
        if child.export.is_some() {
            parent.export = child.export.clone();
        }
        if child.experiment.is_some() {
            parent.experiment = child.experiment.clone();
        }
    }

    /// Convert YAML value to ConfigValue
    fn convert_yaml_value(&self, value: &serde_yaml::Value) -> ConfigTemplateResult<ConfigValue> {
        match value {
            serde_yaml::Value::Number(n) => {
                if n.is_i64() {
                    Ok(ConfigValue::Int(n.as_i64().unwrap()))
                } else if n.is_f64() {
                    Ok(ConfigValue::Float(n.as_f64().unwrap()))
                } else {
                    Err(ConfigTemplateError::ParameterOverride(
                        "Invalid number type".to_string(),
                    ))
                }
            }
            serde_yaml::Value::String(s) => Ok(ConfigValue::String(s.clone())),
            serde_yaml::Value::Bool(b) => Ok(ConfigValue::Bool(*b)),
            serde_yaml::Value::Sequence(seq) => {
                let mut int_vec = Vec::new();
                let mut float_vec = Vec::new();
                let mut is_int = true;

                for item in seq {
                    match item {
                        serde_yaml::Value::Number(n) => {
                            if n.is_i64() {
                                int_vec.push(n.as_i64().unwrap());
                                float_vec.push(n.as_f64().unwrap());
                            } else if n.is_f64() {
                                is_int = false;
                                float_vec.push(n.as_f64().unwrap());
                            }
                        }
                        _ => {
                            return Err(ConfigTemplateError::ParameterOverride(
                                "Array must contain only numbers".to_string(),
                            ))
                        }
                    }
                }

                if is_int {
                    Ok(ConfigValue::IntArray(int_vec))
                } else {
                    Ok(ConfigValue::FloatArray(float_vec))
                }
            }
            _ => Err(ConfigTemplateError::ParameterOverride(
                "Unsupported value type".to_string(),
            )),
        }
    }

    /// Validate configuration against template constraints
    fn validate_config(
        &self,
        template: &DatasetTemplate,
        config: &GeneratorConfig,
    ) -> ConfigTemplateResult<()> {
        // Validate required parameters
        for (param_name, param_def) in &template.parameters {
            if param_def.required && config.get_parameter(param_name).is_none() {
                return Err(ConfigTemplateError::MissingParameter(param_name.clone()));
            }
        }

        // Validate parameter constraints
        for (param_name, param_def) in &template.parameters {
            if let Some(value) = config.get_parameter(param_name) {
                self.validate_parameter_value(param_def, value)?;
            }
        }

        // Validate dataset size constraints
        if let Some(validation) = &template.validation {
            if let Some(size_constraints) = &validation.dataset_size {
                self.validate_size_constraints(size_constraints, config)?;
            }
        }

        Ok(())
    }

    /// Validate parameter value against constraints
    fn validate_parameter_value(
        &self,
        param_def: &ParameterTemplate,
        value: &ConfigValue,
    ) -> ConfigTemplateResult<()> {
        match (&param_def.param_type, value) {
            (ParameterTypeTemplate::Integer { min, max }, ConfigValue::Int(v)) => {
                if let Some(min_val) = min {
                    if *v < *min_val {
                        return Err(ConfigTemplateError::Validation(format!(
                            "Value {} below minimum {}",
                            v, min_val
                        )));
                    }
                }
                if let Some(max_val) = max {
                    if *v > *max_val {
                        return Err(ConfigTemplateError::Validation(format!(
                            "Value {} above maximum {}",
                            v, max_val
                        )));
                    }
                }
            }
            (ParameterTypeTemplate::Float { min, max }, ConfigValue::Float(v)) => {
                if let Some(min_val) = min {
                    if *v < *min_val {
                        return Err(ConfigTemplateError::Validation(format!(
                            "Value {} below minimum {}",
                            v, min_val
                        )));
                    }
                }
                if let Some(max_val) = max {
                    if *v > *max_val {
                        return Err(ConfigTemplateError::Validation(format!(
                            "Value {} above maximum {}",
                            v, max_val
                        )));
                    }
                }
            }
            (ParameterTypeTemplate::Boolean, ConfigValue::Bool(_)) => {
                // Always valid
            }
            _ => {
                // Type mismatch - could add more detailed checking
            }
        }

        Ok(())
    }

    /// Validate size constraints
    fn validate_size_constraints(
        &self,
        constraints: &DatasetSizeConstraints,
        config: &GeneratorConfig,
    ) -> ConfigTemplateResult<()> {
        if let Some(min_samples) = constraints.min_samples {
            if config.n_samples < min_samples {
                return Err(ConfigTemplateError::Validation(format!(
                    "n_samples {} below minimum {}",
                    config.n_samples, min_samples
                )));
            }
        }

        if let Some(max_samples) = constraints.max_samples {
            if config.n_samples > max_samples {
                return Err(ConfigTemplateError::Validation(format!(
                    "n_samples {} above maximum {}",
                    config.n_samples, max_samples
                )));
            }
        }

        if let Some(min_features) = constraints.min_features {
            if config.n_features < min_features {
                return Err(ConfigTemplateError::Validation(format!(
                    "n_features {} below minimum {}",
                    config.n_features, min_features
                )));
            }
        }

        if let Some(max_features) = constraints.max_features {
            if config.n_features > max_features {
                return Err(ConfigTemplateError::Validation(format!(
                    "n_features {} above maximum {}",
                    config.n_features, max_features
                )));
            }
        }

        Ok(())
    }

    /// Export template to YAML
    pub fn export_template_yaml(&self, name: &str) -> ConfigTemplateResult<String> {
        let template = self
            .get_template(name)
            .ok_or_else(|| ConfigTemplateError::TemplateNotFound(name.to_string()))?;

        serde_yaml::to_string(template).map_err(|e| ConfigTemplateError::Yaml(e.to_string()))
    }

    /// Export template to JSON
    pub fn export_template_json(&self, name: &str) -> ConfigTemplateResult<String> {
        let template = self
            .get_template(name)
            .ok_or_else(|| ConfigTemplateError::TemplateNotFound(name.to_string()))?;

        Ok(serde_json::to_string_pretty(template)?)
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Template builder for programmatic template creation
pub struct TemplateBuilder {
    template: DatasetTemplate,
}

impl TemplateBuilder {
    /// Create a new template builder
    pub fn new(name: &str) -> Self {
        Self {
            template: DatasetTemplate {
                metadata: TemplateMetadata {
                    name: name.to_string(),
                    version: "1.0.0".to_string(),
                    description: "".to_string(),
                    author: "".to_string(),
                    created: None,
                    modified: None,
                    tags: Vec::new(),
                    category: None,
                },
                extends: None,
                generator: GeneratorTemplateConfig {
                    name: "classification".to_string(),
                    n_samples: Some(1000),
                    n_features: Some(10),
                    random_state: None,
                    parameters: HashMap::new(),
                },
                parameters: HashMap::new(),
                validation: None,
                export: None,
                experiment: None,
            },
        }
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: TemplateMetadata) -> Self {
        self.template.metadata = metadata;
        self
    }

    /// Set generator
    pub fn generator(mut self, generator: GeneratorTemplateConfig) -> Self {
        self.template.generator = generator;
        self
    }

    /// Add parameter definition
    pub fn parameter(mut self, name: String, param: ParameterTemplate) -> Self {
        self.template.parameters.insert(name, param);
        self
    }

    /// Set validation rules
    pub fn validation(mut self, validation: ValidationRules) -> Self {
        self.template.validation = Some(validation);
        self
    }

    /// Set inheritance
    pub fn extends(mut self, base: String) -> Self {
        self.template.extends = Some(base);
        self
    }

    /// Build the template
    pub fn build(self) -> DatasetTemplate {
        self.template
    }
}

/// Convenience functions for creating common templates
pub fn create_classification_template() -> DatasetTemplate {
    TemplateBuilder::new("classification_basic")
        .metadata(TemplateMetadata {
            name: "classification_basic".to_string(),
            version: "1.0.0".to_string(),
            description: "Basic classification dataset template".to_string(),
            author: "sklears".to_string(),
            created: None,
            modified: None,
            tags: vec!["classification".to_string(), "basic".to_string()],
            category: Some("classification".to_string()),
        })
        .parameter(
            "n_classes".to_string(),
            ParameterTemplate {
                description: "Number of classes".to_string(),
                param_type: ParameterTypeTemplate::Integer {
                    min: Some(2),
                    max: Some(10),
                },
                required: false,
                default: Some(serde_yaml::Value::Number(serde_yaml::Number::from(2))),
                constraints: None,
                dependencies: vec![],
            },
        )
        .parameter(
            "n_informative".to_string(),
            ParameterTemplate {
                description: "Number of informative features".to_string(),
                param_type: ParameterTypeTemplate::Integer {
                    min: Some(1),
                    max: None,
                },
                required: false,
                default: None,
                constraints: None,
                dependencies: vec![],
            },
        )
        .build()
}

pub fn create_regression_template() -> DatasetTemplate {
    TemplateBuilder::new("regression_basic")
        .metadata(TemplateMetadata {
            name: "regression_basic".to_string(),
            version: "1.0.0".to_string(),
            description: "Basic regression dataset template".to_string(),
            author: "sklears".to_string(),
            created: None,
            modified: None,
            tags: vec!["regression".to_string(), "basic".to_string()],
            category: Some("regression".to_string()),
        })
        .generator(GeneratorTemplateConfig {
            name: "regression".to_string(),
            n_samples: Some(1000),
            n_features: Some(5),
            random_state: Some(42),
            parameters: HashMap::new(),
        })
        .parameter(
            "noise".to_string(),
            ParameterTemplate {
                description: "Standard deviation of gaussian noise".to_string(),
                param_type: ParameterTypeTemplate::Float {
                    min: Some(0.0),
                    max: Some(1.0),
                },
                required: false,
                default: Some(serde_yaml::Value::Number(serde_yaml::Number::from(0.1))),
                constraints: None,
                dependencies: vec![],
            },
        )
        .build()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_template_builder() {
        let template = TemplateBuilder::new("test_template")
            .metadata(TemplateMetadata {
                name: "test_template".to_string(),
                version: "1.0.0".to_string(),
                description: "Test template".to_string(),
                author: "Test".to_string(),
                created: None,
                modified: None,
                tags: vec!["test".to_string()],
                category: Some("test".to_string()),
            })
            .build();

        assert_eq!(template.metadata.name, "test_template");
        assert_eq!(template.metadata.version, "1.0.0");
    }

    #[test]
    fn test_template_library() {
        let mut library = TemplateLibrary::new();

        // Register a template
        let template = create_classification_template();
        library.register_template(template);

        // Check if registered
        assert!(library.get_template("classification_basic").is_some());
        assert!(library
            .list_templates()
            .contains(&"classification_basic".to_string()));

        // Search by tag
        let classification_templates = library.search_by_tag("classification");
        assert!(classification_templates.contains(&"classification_basic".to_string()));
    }

    #[test]
    fn test_template_resolution() {
        let mut library = TemplateLibrary::new();
        library.register_template(create_classification_template());

        // Resolve template
        let config = library
            .resolve_template("classification_basic", None)
            .unwrap();
        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.n_features, 10);

        // Test with overrides
        let mut overrides = HashMap::new();
        overrides.insert(
            "n_classes".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(5)),
        );

        let config_with_overrides = library
            .resolve_template("classification_basic", Some(overrides))
            .unwrap();

        if let Some(crate::traits::ConfigValue::Int(n_classes)) =
            config_with_overrides.get_parameter("n_classes")
        {
            assert_eq!(*n_classes, 5);
        }
    }

    #[test]
    fn test_yaml_serialization() {
        let template = create_regression_template();
        let yaml = serde_yaml::to_string(&template).unwrap();

        // Verify it can be parsed back
        let parsed: DatasetTemplate = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.metadata.name, "regression_basic");
    }

    #[test]
    fn test_template_inheritance() {
        let mut library = TemplateLibrary::new();

        // Create base template
        let base = create_classification_template();
        library.register_template(base);

        // Create derived template
        let derived = TemplateBuilder::new("classification_advanced")
            .extends("classification_basic".to_string())
            .generator(GeneratorTemplateConfig {
                name: "classification".to_string(),
                n_samples: Some(5000), // Override
                n_features: Some(20),  // Override
                random_state: Some(123),
                parameters: HashMap::new(),
            })
            .build();

        library.register_template(derived);

        // Resolve derived template
        let config = library
            .resolve_template("classification_advanced", None)
            .unwrap();
        assert_eq!(config.n_samples, 5000); // Should be overridden
        assert_eq!(config.n_features, 20); // Should be overridden
    }

    #[test]
    fn test_parameter_validation() {
        let mut library = TemplateLibrary::new();
        library.register_template(create_classification_template());

        // Test valid override
        let mut valid_overrides = HashMap::new();
        valid_overrides.insert(
            "n_classes".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(3)),
        );

        assert!(library
            .resolve_template("classification_basic", Some(valid_overrides))
            .is_ok());

        // Test invalid override (out of range)
        let mut invalid_overrides = HashMap::new();
        invalid_overrides.insert(
            "n_classes".to_string(),
            serde_yaml::Value::Number(serde_yaml::Number::from(15)), // Above max of 10
        );

        assert!(library
            .resolve_template("classification_basic", Some(invalid_overrides))
            .is_err());
    }
}
