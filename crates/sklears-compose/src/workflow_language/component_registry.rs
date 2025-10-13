//! Component Registry for Pipeline Components
//!
//! This module provides component registration and discovery capabilities for the
//! workflow system, including component metadata management, parameter schemas,
//! validation rules, and component lifecycle management.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{BTreeMap, HashMap};

use super::workflow_definitions::{DataType, ParameterValue, StepType};

/// Component registry for available pipeline components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRegistry {
    /// Registered components by name
    pub components: HashMap<String, ComponentDefinition>,
}

/// Component definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDefinition {
    /// Component name
    pub name: String,
    /// Component type
    pub component_type: StepType,
    /// Component description
    pub description: String,
    /// Component category
    pub category: ComponentCategory,
    /// Input parameters schema
    pub parameters: BTreeMap<String, ParameterSchema>,
    /// Input ports
    pub inputs: Vec<PortDefinition>,
    /// Output ports
    pub outputs: Vec<PortDefinition>,
    /// Component version
    pub version: String,
    /// Whether component is deprecated
    pub deprecated: bool,
    /// Performance characteristics
    pub performance: PerformanceCharacteristics,
    /// Implementation details
    pub implementation: ImplementationDetails,
}

/// Parameter schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter data type
    pub param_type: DataType,
    /// Default value
    pub default: Option<ParameterValue>,
    /// Parameter description
    pub description: String,
    /// Validation rules
    pub validation: Option<ValidationRule>,
    /// Whether parameter is required
    pub required: bool,
    /// Parameter hints for UI
    pub ui_hints: Option<UIHints>,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule parameters
    pub parameters: BTreeMap<String, String>,
    /// Custom validation function
    pub custom_validator: Option<String>,
}

/// Types of validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Range validation for numeric values
    Range { min: Option<f64>, max: Option<f64> },
    /// Length validation for strings/arrays
    Length {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Pattern validation for strings
    Pattern(String),
    /// Enum validation (allowed values)
    Enum(Vec<String>),
    /// Cross-parameter validation
    CrossParameter(String),
    /// Custom validation function
    Custom(String),
}

/// Port definition for inputs/outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortDefinition {
    /// Port name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Whether port is optional
    pub optional: bool,
    /// Port description
    pub description: String,
    /// Shape constraints
    pub shape_constraints: Option<String>,
}

/// Component categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentCategory {
    /// Data input/output
    DataIO,
    /// Preprocessing
    Preprocessing,
    /// Feature engineering
    FeatureEngineering,
    /// Model training
    ModelTraining,
    /// Model evaluation
    ModelEvaluation,
    /// Visualization
    Visualization,
    /// Utilities
    Utilities,
    /// Custom category
    Custom(String),
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Time complexity (Big O notation)
    pub time_complexity: String,
    /// Space complexity (Big O notation)
    pub space_complexity: String,
    /// Whether component supports parallel execution
    pub parallel_capable: bool,
    /// Whether component supports GPU acceleration
    pub gpu_accelerated: bool,
    /// Typical memory usage
    pub memory_usage: MemoryUsage,
    /// Scalability characteristics
    pub scalability: ScalabilityInfo,
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Base memory overhead
    pub base_overhead_mb: f64,
    /// Memory scaling factor with data size
    pub scaling_factor: f64,
    /// Peak memory multiplier
    pub peak_multiplier: f64,
}

/// Scalability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityInfo {
    /// Maximum recommended data size
    pub max_data_size: Option<usize>,
    /// Scaling behavior
    pub scaling_behavior: ScalingBehavior,
    /// Bottleneck description
    pub bottlenecks: Vec<String>,
}

/// Scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingBehavior {
    /// Linear scaling with data size
    Linear,
    /// Logarithmic scaling
    Logarithmic,
    /// Polynomial scaling
    Polynomial(f64),
    /// Exponential scaling
    Exponential,
    /// Constant (doesn't scale with data)
    Constant,
}

/// Implementation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationDetails {
    /// Implementation language
    pub language: String,
    /// Required dependencies
    pub dependencies: Vec<String>,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// License information
    pub license: String,
    /// Source location
    pub source: Option<String>,
}

/// UI hints for parameter display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIHints {
    /// Widget type for parameter input
    pub widget_type: WidgetType,
    /// Display order
    pub display_order: Option<i32>,
    /// Grouping information
    pub group: Option<String>,
    /// Help text
    pub help_text: Option<String>,
    /// Placeholder text
    pub placeholder: Option<String>,
}

/// Widget types for UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    /// Text input
    TextInput,
    /// Number input
    NumberInput,
    /// Checkbox
    Checkbox,
    /// Dropdown/select
    Dropdown(Vec<String>),
    /// Slider
    Slider { min: f64, max: f64, step: f64 },
    /// File picker
    FilePicker,
    /// Color picker
    ColorPicker,
    /// Custom widget
    Custom(String),
}

impl ComponentRegistry {
    /// Create a new component registry with default components
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            components: HashMap::new(),
        };

        // Register default components
        registry.register_default_components();
        registry
    }

    /// Register a component
    pub fn register_component(&mut self, component: ComponentDefinition) -> SklResult<()> {
        if self.components.contains_key(&component.name) {
            return Err(SklearsError::InvalidInput(format!(
                "Component '{}' already registered",
                component.name
            )));
        }

        self.components.insert(component.name.clone(), component);
        Ok(())
    }

    /// Get a component definition
    #[must_use]
    pub fn get_component(&self, name: &str) -> Option<&ComponentDefinition> {
        self.components.get(name)
    }

    /// Check if a component exists
    #[must_use]
    pub fn has_component(&self, name: &str) -> bool {
        self.components.contains_key(name)
    }

    /// List all available components
    #[must_use]
    pub fn list_components(&self) -> Vec<&str> {
        self.components
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Get components by category
    #[must_use]
    pub fn get_components_by_category(
        &self,
        category: &ComponentCategory,
    ) -> Vec<&ComponentDefinition> {
        self.components
            .values()
            .filter(|comp| {
                std::mem::discriminant(&comp.category) == std::mem::discriminant(category)
            })
            .collect()
    }

    /// Search components by name or description
    #[must_use]
    pub fn search_components(&self, query: &str) -> Vec<&ComponentDefinition> {
        let query_lower = query.to_lowercase();
        self.components
            .values()
            .filter(|comp| {
                comp.name.to_lowercase().contains(&query_lower)
                    || comp.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Validate component parameters
    pub fn validate_parameters(
        &self,
        component_name: &str,
        parameters: &BTreeMap<String, ParameterValue>,
    ) -> SklResult<()> {
        let component = self.get_component(component_name).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Component '{component_name}' not found"))
        })?;

        // Check required parameters
        for (param_name, param_schema) in &component.parameters {
            if param_schema.required && !parameters.contains_key(param_name) {
                return Err(SklearsError::InvalidInput(format!(
                    "Required parameter '{param_name}' missing for component '{component_name}'"
                )));
            }
        }

        // Validate parameter values
        for (param_name, param_value) in parameters {
            if let Some(param_schema) = component.parameters.get(param_name) {
                self.validate_parameter_value(param_schema, param_value)?;
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown parameter '{param_name}' for component '{component_name}'"
                )));
            }
        }

        Ok(())
    }

    /// Validate a single parameter value
    fn validate_parameter_value(
        &self,
        schema: &ParameterSchema,
        value: &ParameterValue,
    ) -> SklResult<()> {
        // Type validation
        let types_match = match (&schema.param_type, value) {
            (DataType::Float32 | DataType::Float64, ParameterValue::Float(_)) => true,
            (DataType::Int32 | DataType::Int64, ParameterValue::Int(_)) => true,
            (DataType::Boolean, ParameterValue::Bool(_)) => true,
            (DataType::String, ParameterValue::String(_)) => true,
            (DataType::Array(_), ParameterValue::Array(_)) => true,
            _ => false,
        };

        if !types_match {
            return Err(SklearsError::InvalidInput(format!(
                "Parameter type mismatch: expected {:?}, got {:?}",
                schema.param_type, value
            )));
        }

        // Validation rules
        if let Some(validation) = &schema.validation {
            self.apply_validation_rule(validation, value)?;
        }

        Ok(())
    }

    /// Apply validation rule to a parameter value
    fn apply_validation_rule(
        &self,
        rule: &ValidationRule,
        value: &ParameterValue,
    ) -> SklResult<()> {
        match &rule.rule_type {
            ValidationRuleType::Range { min, max } => {
                if let ParameterValue::Float(val) = value {
                    if let Some(min_val) = min {
                        if *val < *min_val {
                            return Err(SklearsError::InvalidInput(format!(
                                "Value {val} is below minimum {min_val}"
                            )));
                        }
                    }
                    if let Some(max_val) = max {
                        if *val > *max_val {
                            return Err(SklearsError::InvalidInput(format!(
                                "Value {val} is above maximum {max_val}"
                            )));
                        }
                    }
                } else if let ParameterValue::Int(val) = value {
                    if let Some(min_val) = min {
                        if (*val as f64) < *min_val {
                            return Err(SklearsError::InvalidInput(format!(
                                "Value {val} is below minimum {min_val}"
                            )));
                        }
                    }
                    if let Some(max_val) = max {
                        if (*val as f64) > *max_val {
                            return Err(SklearsError::InvalidInput(format!(
                                "Value {val} is above maximum {max_val}"
                            )));
                        }
                    }
                }
            }
            ValidationRuleType::Length { min, max } => {
                let length = match value {
                    ParameterValue::String(s) => s.len(),
                    ParameterValue::Array(arr) => arr.len(),
                    _ => return Ok(()), // Skip length validation for non-string/array types
                };

                if let Some(min_len) = min {
                    if length < *min_len {
                        return Err(SklearsError::InvalidInput(format!(
                            "Length {length} is below minimum {min_len}"
                        )));
                    }
                }
                if let Some(max_len) = max {
                    if length > *max_len {
                        return Err(SklearsError::InvalidInput(format!(
                            "Length {length} is above maximum {max_len}"
                        )));
                    }
                }
            }
            ValidationRuleType::Enum(allowed_values) => {
                if let ParameterValue::String(val) = value {
                    if !allowed_values.contains(val) {
                        return Err(SklearsError::InvalidInput(format!(
                            "Value '{val}' is not in allowed values: {allowed_values:?}"
                        )));
                    }
                }
            }
            _ => {
                // Other validation rules not implemented in this example
            }
        }

        Ok(())
    }

    /// Register default components
    fn register_default_components(&mut self) {
        // StandardScaler
        let standard_scaler = ComponentDefinition {
            name: "StandardScaler".to_string(),
            component_type: StepType::Transformer,
            description: "Standardize features by removing the mean and scaling to unit variance"
                .to_string(),
            category: ComponentCategory::Preprocessing,
            parameters: {
                let mut params = BTreeMap::new();
                params.insert(
                    "with_mean".to_string(),
                    ParameterSchema {
                        param_type: DataType::Boolean,
                        default: Some(ParameterValue::Bool(true)),
                        description: "Center the data before scaling".to_string(),
                        validation: None,
                        required: false,
                        ui_hints: Some(UIHints {
                            widget_type: WidgetType::Checkbox,
                            display_order: Some(1),
                            group: Some("Scaling Options".to_string()),
                            help_text: Some(
                                "Whether to center the data before scaling".to_string(),
                            ),
                            placeholder: None,
                        }),
                    },
                );
                params.insert(
                    "with_std".to_string(),
                    ParameterSchema {
                        param_type: DataType::Boolean,
                        default: Some(ParameterValue::Bool(true)),
                        description: "Scale the data to unit variance".to_string(),
                        validation: None,
                        required: false,
                        ui_hints: Some(UIHints {
                            widget_type: WidgetType::Checkbox,
                            display_order: Some(2),
                            group: Some("Scaling Options".to_string()),
                            help_text: Some(
                                "Whether to scale the data to unit variance".to_string(),
                            ),
                            placeholder: None,
                        }),
                    },
                );
                params
            },
            inputs: vec![PortDefinition {
                name: "X".to_string(),
                data_type: DataType::Matrix(Box::new(DataType::Float64)),
                optional: false,
                description: "Input feature matrix".to_string(),
                shape_constraints: Some("[n_samples, n_features]".to_string()),
            }],
            outputs: vec![PortDefinition {
                name: "X_scaled".to_string(),
                data_type: DataType::Matrix(Box::new(DataType::Float64)),
                optional: false,
                description: "Scaled feature matrix".to_string(),
                shape_constraints: Some("[n_samples, n_features]".to_string()),
            }],
            version: "1.0.0".to_string(),
            deprecated: false,
            performance: PerformanceCharacteristics {
                time_complexity: "O(n*m)".to_string(),
                space_complexity: "O(m)".to_string(),
                parallel_capable: true,
                gpu_accelerated: false,
                memory_usage: MemoryUsage {
                    base_overhead_mb: 1.0,
                    scaling_factor: 0.1,
                    peak_multiplier: 1.2,
                },
                scalability: ScalabilityInfo {
                    max_data_size: None,
                    scaling_behavior: ScalingBehavior::Linear,
                    bottlenecks: vec!["Memory bandwidth".to_string()],
                },
            },
            implementation: ImplementationDetails {
                language: "Rust".to_string(),
                dependencies: vec!["ndarray".to_string(), "sklears-core".to_string()],
                platforms: vec![
                    "Linux".to_string(),
                    "macOS".to_string(),
                    "Windows".to_string(),
                ],
                license: "MIT".to_string(),
                source: None,
            },
        };

        // LinearRegression
        let linear_regression = ComponentDefinition {
            name: "LinearRegression".to_string(),
            component_type: StepType::Trainer,
            description: "Ordinary least squares Linear Regression".to_string(),
            category: ComponentCategory::ModelTraining,
            parameters: {
                let mut params = BTreeMap::new();
                params.insert(
                    "fit_intercept".to_string(),
                    ParameterSchema {
                        param_type: DataType::Boolean,
                        default: Some(ParameterValue::Bool(true)),
                        description: "Whether to fit an intercept term".to_string(),
                        validation: None,
                        required: false,
                        ui_hints: Some(UIHints {
                            widget_type: WidgetType::Checkbox,
                            display_order: Some(1),
                            group: None,
                            help_text: Some(
                                "Whether to calculate the intercept for this model".to_string(),
                            ),
                            placeholder: None,
                        }),
                    },
                );
                params
            },
            inputs: vec![
                PortDefinition {
                    name: "X".to_string(),
                    data_type: DataType::Matrix(Box::new(DataType::Float64)),
                    optional: false,
                    description: "Training data".to_string(),
                    shape_constraints: Some("[n_samples, n_features]".to_string()),
                },
                PortDefinition {
                    name: "y".to_string(),
                    data_type: DataType::Array(Box::new(DataType::Float64)),
                    optional: false,
                    description: "Target values".to_string(),
                    shape_constraints: Some("[n_samples]".to_string()),
                },
            ],
            outputs: vec![PortDefinition {
                name: "model".to_string(),
                data_type: DataType::Custom("LinearRegressionModel".to_string()),
                optional: false,
                description: "Trained linear regression model".to_string(),
                shape_constraints: None,
            }],
            version: "1.0.0".to_string(),
            deprecated: false,
            performance: PerformanceCharacteristics {
                time_complexity: "O(n*m^2)".to_string(),
                space_complexity: "O(m^2)".to_string(),
                parallel_capable: true,
                gpu_accelerated: true,
                memory_usage: MemoryUsage {
                    base_overhead_mb: 2.0,
                    scaling_factor: 0.2,
                    peak_multiplier: 1.5,
                },
                scalability: ScalabilityInfo {
                    max_data_size: Some(1_000_000),
                    scaling_behavior: ScalingBehavior::Polynomial(2.0),
                    bottlenecks: vec!["Matrix inversion".to_string()],
                },
            },
            implementation: ImplementationDetails {
                language: "Rust".to_string(),
                dependencies: vec!["ndarray".to_string(), "ndarray-linalg".to_string()],
                platforms: vec![
                    "Linux".to_string(),
                    "macOS".to_string(),
                    "Windows".to_string(),
                ],
                license: "MIT".to_string(),
                source: None,
            },
        };

        // Register components
        let _ = self.register_component(standard_scaler);
        let _ = self.register_component(linear_regression);
    }

    /// Get component metadata summary
    #[must_use]
    pub fn get_component_summary(&self, name: &str) -> Option<ComponentSummary> {
        self.get_component(name).map(|comp| ComponentSummary {
            name: comp.name.clone(),
            component_type: comp.component_type.clone(),
            description: comp.description.clone(),
            category: comp.category.clone(),
            version: comp.version.clone(),
            deprecated: comp.deprecated,
            parameter_count: comp.parameters.len(),
            input_count: comp.inputs.len(),
            output_count: comp.outputs.len(),
        })
    }

    /// Get all component summaries
    #[must_use]
    pub fn get_all_summaries(&self) -> Vec<ComponentSummary> {
        self.components
            .keys()
            .filter_map(|name| self.get_component_summary(name))
            .collect()
    }
}

/// Component summary for quick overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSummary {
    /// Component name
    pub name: String,
    /// Component type
    pub component_type: StepType,
    /// Description
    pub description: String,
    /// Category
    pub category: ComponentCategory,
    /// Version
    pub version: String,
    /// Whether deprecated
    pub deprecated: bool,
    /// Number of parameters
    pub parameter_count: usize,
    /// Number of inputs
    pub input_count: usize,
    /// Number of outputs
    pub output_count: usize,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Component discovery service for finding available components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDiscovery {
    /// Available component registries
    pub registries: Vec<String>,
    /// Search paths for components
    pub search_paths: Vec<String>,
    /// Discovery configuration
    pub config: DiscoveryConfig,
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable automatic discovery
    pub auto_discovery: bool,
    /// Discovery timeout in seconds
    pub timeout_sec: u64,
    /// Cache discovery results
    pub cache_results: bool,
}

/// Component metadata for extended information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component identifier
    pub id: String,
    /// Display name
    pub display_name: String,
    /// Icon or image reference
    pub icon: Option<String>,
    /// Documentation URL
    pub documentation_url: Option<String>,
    /// Example usage
    pub examples: Vec<String>,
    /// Keywords for search
    pub keywords: Vec<String>,
    /// Maintainer information
    pub maintainer: Option<String>,
    /// Creation timestamp
    pub created_at: String,
    /// Last updated timestamp
    pub updated_at: String,
}

/// Component signature for type checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSignature {
    /// Input signature
    pub inputs: Vec<TypeSignature>,
    /// Output signature
    pub outputs: Vec<TypeSignature>,
    /// Parameter signature
    pub parameters: Vec<ParameterSignature>,
}

/// Type signature for inputs/outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSignature {
    /// Type name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Shape information
    pub shape: Option<String>,
    /// Type constraints
    pub constraints: Vec<String>,
}

/// Parameter signature for type checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSignature {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: DataType,
    /// Whether required
    pub required: bool,
    /// Type constraints
    pub constraints: Vec<String>,
}

/// Component type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentType {
    /// Data loading component
    DataLoader,
    /// Data transformation component
    Transformer,
    /// Model training component
    Trainer,
    /// Model inference component
    Predictor,
    /// Evaluation component
    Evaluator,
    /// Visualization component
    Visualizer,
    /// Utility component
    Utility,
    /// Custom component type
    Custom(String),
}

/// Component validator for validation logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentValidator {
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Custom validation function
    pub custom_validator: Option<String>,
    /// Validation context
    pub context: ValidationContext,
}

/// Validation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    /// Current workflow context
    pub workflow_id: Option<String>,
    /// Available component instances
    pub available_components: Vec<String>,
    /// Global parameters
    pub global_params: BTreeMap<String, ParameterValue>,
}

/// Component version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentVersion {
    /// Version string
    pub version: String,
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
    /// Pre-release identifier
    pub pre_release: Option<String>,
    /// Build metadata
    pub build: Option<String>,
}

/// Registry error types
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum RegistryError {
    /// Component not found
    #[error("Component not found: {0}")]
    ComponentNotFound(String),
    /// Component already exists
    #[error("Component already exists: {0}")]
    ComponentExists(String),
    /// Invalid component definition
    #[error("Invalid component definition: {0}")]
    InvalidComponent(String),
    /// Version conflict
    #[error("Version conflict: {0}")]
    VersionConflict(String),
    /// Dependency error
    #[error("Dependency error: {0}")]
    DependencyError(String),
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
    /// IO error
    #[error("IO error: {0}")]
    IoError(String),
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_registry_creation() {
        let registry = ComponentRegistry::new();
        assert!(registry.has_component("StandardScaler"));
        assert!(registry.has_component("LinearRegression"));
        assert!(!registry.has_component("NonExistentComponent"));
    }

    #[test]
    fn test_get_component() {
        let registry = ComponentRegistry::new();
        let component = registry.get_component("StandardScaler");
        assert!(component.is_some());

        let comp = component.unwrap();
        assert_eq!(comp.name, "StandardScaler");
        assert_eq!(comp.component_type, StepType::Transformer);
    }

    #[test]
    fn test_validate_parameters() {
        let registry = ComponentRegistry::new();

        let mut params = BTreeMap::new();
        params.insert("with_mean".to_string(), ParameterValue::Bool(true));
        params.insert("with_std".to_string(), ParameterValue::Bool(false));

        let result = registry.validate_parameters("StandardScaler", &params);
        assert!(result.is_ok());

        // Test invalid parameter
        params.insert("invalid_param".to_string(), ParameterValue::Bool(true));
        let result = registry.validate_parameters("StandardScaler", &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_components() {
        let registry = ComponentRegistry::new();
        let results = registry.search_components("scale");
        assert!(!results.is_empty());
        assert!(results.iter().any(|comp| comp.name == "StandardScaler"));
    }

    #[test]
    fn test_get_components_by_category() {
        let registry = ComponentRegistry::new();
        let preprocessing_components =
            registry.get_components_by_category(&ComponentCategory::Preprocessing);
        assert!(!preprocessing_components.is_empty());
        assert!(preprocessing_components
            .iter()
            .any(|comp| comp.name == "StandardScaler"));
    }

    #[test]
    fn test_component_summary() {
        let registry = ComponentRegistry::new();
        let summary = registry.get_component_summary("LinearRegression");
        assert!(summary.is_some());

        let sum = summary.unwrap();
        assert_eq!(sum.name, "LinearRegression");
        assert_eq!(sum.component_type, StepType::Trainer);
        assert!(!sum.deprecated);
    }
}
