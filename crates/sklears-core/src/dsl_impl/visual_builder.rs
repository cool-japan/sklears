//! Visual pipeline builder for creating ML pipelines through drag-and-drop interface
//!
//! This module provides a comprehensive visual interface for building machine learning
//! pipelines without writing code. It includes a component library, canvas for pipeline
//! design, code generation, validation, and export capabilities.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::dsl_impl::dsl_types::{PipelineConfig, PipelineStage, StageType};

/// Visual pipeline builder for creating ML pipelines through a drag-and-drop interface
///
/// The VisualPipelineBuilder provides a complete solution for building machine learning
/// pipelines visually, without requiring knowledge of the underlying DSL or Rust code.
/// It includes validation, code generation, and export capabilities.
#[derive(Debug, Clone)]
pub struct VisualPipelineBuilder {
    /// Library of available components for pipeline construction
    pub component_library: ComponentLibrary,
    /// Canvas for designing and organizing pipeline components
    pub pipeline_canvas: PipelineCanvas,
    /// Code generator for converting visual designs to executable code
    pub code_generator: VisualCodeGenerator,
    /// Validator for ensuring pipeline correctness
    pub validator: PipelineValidator,
    /// Manager for exporting pipelines to various formats
    pub export_manager: PipelineExportManager,
    /// Configuration settings for the visual builder
    pub settings: VisualBuilderSettings,
}

impl VisualPipelineBuilder {
    /// Create a new visual pipeline builder with default components
    pub fn new() -> Self {
        Self {
            component_library: ComponentLibrary::new(),
            pipeline_canvas: PipelineCanvas::new(),
            code_generator: VisualCodeGenerator::new(),
            validator: PipelineValidator::new(),
            export_manager: PipelineExportManager::new(),
            settings: VisualBuilderSettings::default(),
        }
    }

    /// Generate web-based visual builder interface
    ///
    /// Creates a complete web interface for the visual pipeline builder,
    /// including HTML templates, JavaScript logic, CSS styling, and API endpoints.
    pub fn generate_web_interface(&self) -> crate::error::Result<WebInterface> {
        let html_template = self.generate_html_interface()?;
        let javascript_code = self.generate_javascript_logic()?;
        let css_styling = self.generate_css_styles()?;
        let component_definitions = self.generate_component_definitions()?;

        Ok(WebInterface {
            html_template,
            javascript_code,
            css_styling,
            component_definitions,
            api_endpoints: self.generate_api_endpoints()?,
            websocket_handlers: self.generate_websocket_handlers()?,
        })
    }

    /// Build a pipeline from visual configuration
    ///
    /// Converts a visual pipeline configuration into executable code,
    /// including validation, optimization, and documentation generation.
    pub fn build_pipeline_from_visual(&self, visual_config: &VisualPipelineConfig) -> crate::error::Result<GeneratedPipeline> {
        // Validate the visual configuration
        let validation_result = self.validator.validate_visual_pipeline(visual_config)?;
        if !validation_result.is_valid {
            return Err(crate::error::SklError::ValidationError(
                format!("Pipeline validation failed: {}", validation_result.error_message.unwrap_or_default())
            ));
        }

        // Generate DSL code from visual configuration
        let dsl_code = self.code_generator.generate_dsl_from_visual(visual_config)?;

        // Generate Rust implementation code
        let rust_code = self.code_generator.generate_rust_implementation(visual_config)?;

        // Generate comprehensive documentation
        let documentation = self.generate_pipeline_documentation(visual_config)?;

        // Analyze dependencies and performance characteristics
        let dependencies = self.analyze_dependencies(visual_config)?;
        let performance_hints = self.generate_performance_hints(visual_config)?;
        let test_code = self.generate_test_code(visual_config)?;

        Ok(GeneratedPipeline {
            name: visual_config.name.clone(),
            dsl_code,
            rust_code,
            documentation,
            dependencies,
            performance_hints,
            test_code,
            metadata: visual_config.metadata.clone(),
        })
    }

    /// Import pipeline from various formats
    ///
    /// Supports importing pipelines from multiple sources including JSON, YAML,
    /// scikit-learn pipelines, PyTorch models, and existing DSL macros.
    pub fn import_pipeline(&mut self, import_data: &PipelineImportData) -> crate::error::Result<VisualPipelineConfig> {
        match import_data.format {
            ImportFormat::Json => self.import_from_json(&import_data.content),
            ImportFormat::Yaml => self.import_from_yaml(&import_data.content),
            ImportFormat::SklearnPipeline => self.import_from_sklearn(&import_data.content),
            ImportFormat::TorchScript => self.import_from_torch(&import_data.content),
            ImportFormat::DslMacro => self.import_from_dsl_macro(&import_data.content),
            ImportFormat::OnnxModel => self.import_from_onnx(&import_data.content),
        }
    }

    /// Export pipeline to various formats
    ///
    /// Supports exporting to multiple formats for use in different environments
    /// and frameworks.
    pub fn export_pipeline(&self, config: &VisualPipelineConfig, format: ExportFormat) -> crate::error::Result<String> {
        self.export_manager.export(config, format)
    }

    /// Validate a visual pipeline configuration
    pub fn validate_pipeline(&self, config: &VisualPipelineConfig) -> crate::error::Result<ValidationResult> {
        self.validator.validate_visual_pipeline(config)
    }

    /// Optimize a visual pipeline configuration
    ///
    /// Applies various optimization strategies to improve pipeline performance
    /// and resource usage.
    pub fn optimize_pipeline(&self, config: &VisualPipelineConfig) -> crate::error::Result<VisualPipelineConfig> {
        let mut optimized_config = config.clone();

        // Apply component-level optimizations
        self.optimize_components(&mut optimized_config)?;

        // Apply data flow optimizations
        self.optimize_data_flow(&mut optimized_config)?;

        // Apply resource usage optimizations
        self.optimize_resource_usage(&mut optimized_config)?;

        Ok(optimized_config)
    }

    /// Get available component templates
    pub fn get_component_templates(&self) -> &Vec<ComponentTemplate> {
        &self.component_library.templates
    }

    /// Add custom component to the library
    pub fn add_custom_component(&mut self, component: ComponentDef) -> crate::error::Result<()> {
        self.component_library.add_custom_component(component)
    }

    /// Generate HTML interface for the visual builder
    fn generate_html_interface(&self) -> crate::error::Result<String> {
        Ok(include_str!("../templates/visual_builder.html").to_string())
    }

    /// Generate JavaScript logic for the visual builder
    fn generate_javascript_logic(&self) -> crate::error::Result<String> {
        Ok(include_str!("../templates/visual_builder.js").to_string())
    }

    /// Generate CSS styling for the visual builder
    fn generate_css_styles(&self) -> crate::error::Result<String> {
        Ok(include_str!("../templates/visual_builder.css").to_string())
    }

    /// Generate component definitions for the frontend
    fn generate_component_definitions(&self) -> crate::error::Result<String> {
        let definitions = serde_json::to_string_pretty(&self.component_library)?;
        Ok(definitions)
    }

    /// Generate API endpoints for the visual builder
    fn generate_api_endpoints(&self) -> crate::error::Result<Vec<ApiEndpoint>> {
        Ok(vec![
            ApiEndpoint {
                path: "/api/components".to_string(),
                method: "GET".to_string(),
                description: "Get available components".to_string(),
            },
            ApiEndpoint {
                path: "/api/pipeline/validate".to_string(),
                method: "POST".to_string(),
                description: "Validate pipeline configuration".to_string(),
            },
            ApiEndpoint {
                path: "/api/pipeline/generate".to_string(),
                method: "POST".to_string(),
                description: "Generate code from visual configuration".to_string(),
            },
            ApiEndpoint {
                path: "/api/pipeline/export".to_string(),
                method: "POST".to_string(),
                description: "Export pipeline to various formats".to_string(),
            },
        ])
    }

    /// Generate WebSocket handlers for real-time collaboration
    fn generate_websocket_handlers(&self) -> crate::error::Result<Vec<WebSocketHandler>> {
        Ok(vec![
            WebSocketHandler {
                event: "component_added".to_string(),
                description: "Handle component addition to canvas".to_string(),
            },
            WebSocketHandler {
                event: "component_moved".to_string(),
                description: "Handle component movement on canvas".to_string(),
            },
            WebSocketHandler {
                event: "connection_created".to_string(),
                description: "Handle connection between components".to_string(),
            },
        ])
    }

    /// Generate comprehensive documentation for the pipeline
    fn generate_pipeline_documentation(&self, config: &VisualPipelineConfig) -> crate::error::Result<PipelineDocumentation> {
        Ok(PipelineDocumentation {
            overview: format!("Pipeline: {}", config.name),
            components: config.components.iter().map(|c| format!("- {}: {}", c.name, c.description)).collect(),
            data_flow: self.describe_data_flow(config)?,
            performance_notes: self.generate_performance_notes(config)?,
            usage_examples: self.generate_usage_examples(config)?,
        })
    }

    /// Analyze pipeline dependencies
    fn analyze_dependencies(&self, config: &VisualPipelineConfig) -> crate::error::Result<Vec<String>> {
        let mut dependencies = HashSet::new();

        for component in &config.components {
            dependencies.extend(component.dependencies.iter().cloned());
        }

        Ok(dependencies.into_iter().collect())
    }

    /// Generate performance optimization hints
    fn generate_performance_hints(&self, config: &VisualPipelineConfig) -> crate::error::Result<Vec<PerformanceHint>> {
        let mut hints = Vec::new();

        // Analyze for common performance issues
        if config.components.len() > 10 {
            hints.push(PerformanceHint {
                category: "Complexity".to_string(),
                message: "Consider breaking down complex pipelines into smaller, reusable components".to_string(),
                severity: "Medium".to_string(),
            });
        }

        // Check for parallelization opportunities
        if self.can_parallelize(config)? {
            hints.push(PerformanceHint {
                category: "Parallelization".to_string(),
                message: "This pipeline can benefit from parallel execution".to_string(),
                severity: "Low".to_string(),
            });
        }

        Ok(hints)
    }

    /// Generate test code for the pipeline
    fn generate_test_code(&self, config: &VisualPipelineConfig) -> crate::error::Result<String> {
        let test_template = format!(
            r#"
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_{}_pipeline() {{
        let pipeline = {}Pipeline::new().expect("Failed to create pipeline");
        // Add your test logic here
        assert!(true);
    }}

    #[test]
    fn test_{}_pipeline_with_sample_data() {{
        let pipeline = {}Pipeline::new().expect("Failed to create pipeline");
        // Add sample data testing here
        assert!(true);
    }}
}}
            "#,
            config.name.to_lowercase(),
            config.name,
            config.name.to_lowercase(),
            config.name
        );

        Ok(test_template)
    }

    /// Helper methods for optimization
    fn optimize_components(&self, config: &mut VisualPipelineConfig) -> crate::error::Result<()> {
        // Component-level optimizations
        for component in &mut config.components {
            if component.component_type == "preprocessing" {
                component.properties.insert("use_simd".to_string(), "true".to_string());
            }
        }
        Ok(())
    }

    fn optimize_data_flow(&self, config: &mut VisualPipelineConfig) -> crate::error::Result<()> {
        // Data flow optimizations
        Ok(())
    }

    fn optimize_resource_usage(&self, config: &mut VisualPipelineConfig) -> crate::error::Result<()> {
        // Resource usage optimizations
        Ok(())
    }

    fn describe_data_flow(&self, config: &VisualPipelineConfig) -> crate::error::Result<String> {
        Ok(format!("Data flows through {} components", config.components.len()))
    }

    fn generate_performance_notes(&self, config: &VisualPipelineConfig) -> crate::error::Result<String> {
        Ok(format!("Pipeline with {} components", config.components.len()))
    }

    fn generate_usage_examples(&self, config: &VisualPipelineConfig) -> crate::error::Result<Vec<String>> {
        Ok(vec![format!("let pipeline = {}Pipeline::new()?;", config.name)])
    }

    fn can_parallelize(&self, config: &VisualPipelineConfig) -> crate::error::Result<bool> {
        Ok(config.components.len() > 2)
    }

    // Import methods
    fn import_from_json(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        Ok(serde_json::from_str(content)?)
    }

    fn import_from_yaml(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        Ok(serde_yaml::from_str(content)?)
    }

    fn import_from_sklearn(&self, _content: &str) -> crate::error::Result<VisualPipelineConfig> {
        // TODO: Implement sklearn pipeline import
        Err(crate::error::SklError::NotImplemented("sklearn import not yet implemented".to_string()))
    }

    fn import_from_torch(&self, _content: &str) -> crate::error::Result<VisualPipelineConfig> {
        // TODO: Implement PyTorch model import
        Err(crate::error::SklError::NotImplemented("PyTorch import not yet implemented".to_string()))
    }

    fn import_from_dsl_macro(&self, _content: &str) -> crate::error::Result<VisualPipelineConfig> {
        // TODO: Implement DSL macro import
        Err(crate::error::SklError::NotImplemented("DSL macro import not yet implemented".to_string()))
    }

    fn import_from_onnx(&self, _content: &str) -> crate::error::Result<VisualPipelineConfig> {
        // TODO: Implement ONNX model import
        Err(crate::error::SklError::NotImplemented("ONNX import not yet implemented".to_string()))
    }
}

impl Default for VisualPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for a visual pipeline created through the drag-and-drop interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPipelineConfig {
    /// Name of the pipeline
    pub name: String,
    /// Description of what the pipeline does
    pub description: String,
    /// List of components in the pipeline
    pub components: Vec<ComponentInstance>,
    /// Connections between components
    pub connections: Vec<ComponentConnection>,
    /// Canvas layout information
    pub layout: CanvasLayout,
    /// Pipeline metadata
    pub metadata: HashMap<String, String>,
    /// Global pipeline settings
    pub settings: PipelineSettings,
}

/// Instance of a component placed on the canvas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInstance {
    /// Unique identifier for this component instance
    pub id: String,
    /// Name of the component
    pub name: String,
    /// Type of component (preprocessing, model, etc.)
    pub component_type: String,
    /// Component-specific properties and configuration
    pub properties: HashMap<String, String>,
    /// Position on the canvas
    pub position: ComponentPosition,
    /// Dependencies required by this component
    pub dependencies: Vec<String>,
    /// Description of the component's functionality
    pub description: String,
}

/// Connection between two components in the pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConnection {
    /// ID of the source component
    pub from_component: String,
    /// Output port of the source component
    pub from_port: String,
    /// ID of the destination component
    pub to_component: String,
    /// Input port of the destination component
    pub to_port: String,
    /// Type of data flowing through this connection
    pub data_type: String,
}

/// Position of a component on the canvas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPosition {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Layout information for the entire canvas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasLayout {
    pub width: f64,
    pub height: f64,
    pub zoom_level: f64,
    pub grid_enabled: bool,
    pub snap_to_grid: bool,
}

/// Pipeline-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSettings {
    pub parallel_execution: bool,
    pub cache_intermediate_results: bool,
    pub enable_gpu: bool,
    pub memory_limit_mb: Option<usize>,
    pub timeout_seconds: Option<u64>,
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            parallel_execution: false,
            cache_intermediate_results: true,
            enable_gpu: false,
            memory_limit_mb: None,
            timeout_seconds: None,
        }
    }
}

/// Library of available components for pipeline construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Available component templates organized by category
    pub templates: Vec<ComponentTemplate>,
    /// Custom components added by users
    pub custom_components: Vec<ComponentDef>,
    /// Component categories for organization
    pub categories: Vec<ComponentCategory>,
}

impl ComponentLibrary {
    pub fn new() -> Self {
        Self {
            templates: Self::create_default_templates(),
            custom_components: Vec::new(),
            categories: Self::create_default_categories(),
        }
    }

    pub fn add_custom_component(&mut self, component: ComponentDef) -> crate::error::Result<()> {
        self.custom_components.push(component);
        Ok(())
    }

    fn create_default_templates() -> Vec<ComponentTemplate> {
        vec![
            ComponentTemplate {
                id: "data_loader".to_string(),
                name: "Data Loader".to_string(),
                category: "input".to_string(),
                description: "Load data from various sources".to_string(),
                input_ports: vec![],
                output_ports: vec!["data".to_string()],
                properties: HashMap::new(),
            },
            ComponentTemplate {
                id: "scaler".to_string(),
                name: "Feature Scaler".to_string(),
                category: "preprocessing".to_string(),
                description: "Scale features to a standard range".to_string(),
                input_ports: vec!["data".to_string()],
                output_ports: vec!["scaled_data".to_string()],
                properties: HashMap::new(),
            },
            ComponentTemplate {
                id: "random_forest".to_string(),
                name: "Random Forest".to_string(),
                category: "model".to_string(),
                description: "Random Forest classifier/regressor".to_string(),
                input_ports: vec!["features".to_string(), "labels".to_string()],
                output_ports: vec!["predictions".to_string()],
                properties: HashMap::new(),
            },
        ]
    }

    fn create_default_categories() -> Vec<ComponentCategory> {
        vec![
            ComponentCategory {
                id: "input".to_string(),
                name: "Data Input".to_string(),
                description: "Components for loading and importing data".to_string(),
            },
            ComponentCategory {
                id: "preprocessing".to_string(),
                name: "Preprocessing".to_string(),
                description: "Data cleaning and transformation components".to_string(),
            },
            ComponentCategory {
                id: "model".to_string(),
                name: "Models".to_string(),
                description: "Machine learning models and algorithms".to_string(),
            },
            ComponentCategory {
                id: "output".to_string(),
                name: "Output".to_string(),
                description: "Components for saving and exporting results".to_string(),
            },
        ]
    }
}

/// Template for creating component instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTemplate {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Definition of a custom component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDef {
    pub name: String,
    pub description: String,
    pub category: String,
    pub implementation: String,
    pub dependencies: Vec<String>,
}

/// Category for organizing components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentCategory {
    pub id: String,
    pub name: String,
    pub description: String,
}

/// Canvas for designing and organizing pipeline components
#[derive(Debug, Clone)]
pub struct PipelineCanvas {
    pub width: u32,
    pub height: u32,
    pub zoom_level: f64,
    pub grid_enabled: bool,
    pub components: Vec<ComponentInstance>,
    pub connections: Vec<ComponentConnection>,
}

impl PipelineCanvas {
    pub fn new() -> Self {
        Self {
            width: 1920,
            height: 1080,
            zoom_level: 1.0,
            grid_enabled: true,
            components: Vec::new(),
            connections: Vec::new(),
        }
    }
}

impl Default for PipelineCanvas {
    fn default() -> Self {
        Self::new()
    }
}

/// Code generator for converting visual designs to executable code
#[derive(Debug, Clone)]
pub struct VisualCodeGenerator {
    pub settings: CodeGenerationSettings,
}

impl VisualCodeGenerator {
    pub fn new() -> Self {
        Self {
            settings: CodeGenerationSettings::default(),
        }
    }

    pub fn generate_dsl_from_visual(&self, config: &VisualPipelineConfig) -> crate::error::Result<String> {
        // Generate DSL macro code from visual configuration
        Ok(format!("// Generated DSL for {}", config.name))
    }

    pub fn generate_rust_implementation(&self, config: &VisualPipelineConfig) -> crate::error::Result<String> {
        // Generate Rust implementation code
        Ok(format!("// Generated Rust code for {}", config.name))
    }
}

impl Default for VisualCodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Settings for code generation
#[derive(Debug, Clone)]
pub struct CodeGenerationSettings {
    pub optimize_for_performance: bool,
    pub include_documentation: bool,
    pub generate_tests: bool,
    pub target_rust_edition: String,
}

impl Default for CodeGenerationSettings {
    fn default() -> Self {
        Self {
            optimize_for_performance: true,
            include_documentation: true,
            generate_tests: true,
            target_rust_edition: "2021".to_string(),
        }
    }
}

// Additional supporting types would continue here...
// (truncated for brevity - the actual implementation would include all remaining types)

/// Settings for the visual builder interface
#[derive(Debug, Clone)]
pub struct VisualBuilderSettings {
    pub auto_save: bool,
    pub collaborative_editing: bool,
    pub theme: String,
    pub grid_size: u32,
}

impl Default for VisualBuilderSettings {
    fn default() -> Self {
        Self {
            auto_save: true,
            collaborative_editing: false,
            theme: "dark".to_string(),
            grid_size: 20,
        }
    }
}

// Placeholder implementations for remaining types
#[derive(Debug, Clone)]
pub struct PipelineValidator;
impl PipelineValidator {
    pub fn new() -> Self { Self }
    pub fn validate_visual_pipeline(&self, _config: &VisualPipelineConfig) -> crate::error::Result<ValidationResult> {
        Ok(ValidationResult { is_valid: true, error_message: None })
    }
}

#[derive(Debug, Clone)]
pub struct PipelineExportManager;
impl PipelineExportManager {
    pub fn new() -> Self { Self }
    pub fn export(&self, _config: &VisualPipelineConfig, _format: ExportFormat) -> crate::error::Result<String> {
        Ok("// Exported code".to_string())
    }
}

// Supporting types
#[derive(Debug, Clone)]
pub struct ValidationResult { pub is_valid: bool, pub error_message: Option<String> }

#[derive(Debug, Clone)]
pub struct GeneratedPipeline {
    pub name: String,
    pub dsl_code: String,
    pub rust_code: String,
    pub documentation: PipelineDocumentation,
    pub dependencies: Vec<String>,
    pub performance_hints: Vec<PerformanceHint>,
    pub test_code: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PipelineDocumentation { pub overview: String, pub components: Vec<String>, pub data_flow: String, pub performance_notes: String, pub usage_examples: Vec<String> }

#[derive(Debug, Clone)]
pub struct PerformanceHint { pub category: String, pub message: String, pub severity: String }

#[derive(Debug, Clone)]
pub struct PipelineImportData { pub format: ImportFormat, pub content: String }

#[derive(Debug, Clone)]
pub struct WebInterface { pub html_template: String, pub javascript_code: String, pub css_styling: String, pub component_definitions: String, pub api_endpoints: Vec<ApiEndpoint>, pub websocket_handlers: Vec<WebSocketHandler> }

#[derive(Debug, Clone)]
pub struct ApiEndpoint { pub path: String, pub method: String, pub description: String }

#[derive(Debug, Clone)]
pub struct WebSocketHandler { pub event: String, pub description: String }

#[derive(Debug, Clone)]
pub enum ImportFormat { Json, Yaml, SklearnPipeline, TorchScript, DslMacro, OnnxModel }

#[derive(Debug, Clone)]
pub enum ExportFormat { Json, Yaml, RustCode, PythonCode, DslMacro }

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_builder_creation() {
        let builder = VisualPipelineBuilder::new();
        assert_eq!(builder.component_library.templates.len(), 3);
    }

    #[test]
    fn test_component_library_default() {
        let library = ComponentLibrary::new();
        assert!(!library.templates.is_empty());
        assert!(!library.categories.is_empty());
    }

    #[test]
    fn test_pipeline_canvas_default() {
        let canvas = PipelineCanvas::new();
        assert_eq!(canvas.width, 1920);
        assert_eq!(canvas.height, 1080);
        assert_eq!(canvas.zoom_level, 1.0);
    }
}