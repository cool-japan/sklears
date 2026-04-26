//! Visual pipeline builder for creating ML pipelines through drag-and-drop interface
//!
//! This module provides a comprehensive visual interface for building machine learning
//! pipelines without writing code. It includes a component library, canvas for pipeline
//! design, code generation, validation, and export capabilities.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "serde")]
extern crate serde_yaml;

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
    pub fn build_pipeline_from_visual(
        &self,
        visual_config: &VisualPipelineConfig,
    ) -> crate::error::Result<GeneratedPipeline> {
        // Validate the visual configuration
        let validation_result = self.validator.validate_visual_pipeline(visual_config)?;
        if !validation_result.is_valid {
            return Err(crate::error::SklearsError::ValidationError(format!(
                "Pipeline validation failed: {}",
                validation_result.error_message.unwrap_or_default()
            )));
        }

        // Generate DSL code from visual configuration
        let dsl_code = self
            .code_generator
            .generate_dsl_from_visual(visual_config)?;

        // Generate Rust implementation code
        let rust_code = self
            .code_generator
            .generate_rust_implementation(visual_config)?;

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
    pub fn import_pipeline(
        &mut self,
        import_data: &PipelineImportData,
    ) -> crate::error::Result<VisualPipelineConfig> {
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
    pub fn export_pipeline(
        &self,
        config: &VisualPipelineConfig,
        format: ExportFormat,
    ) -> crate::error::Result<String> {
        self.export_manager.export(config, format)
    }

    /// Validate a visual pipeline configuration
    pub fn validate_pipeline(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<ValidationResult> {
        self.validator.validate_visual_pipeline(config)
    }

    /// Optimize a visual pipeline configuration
    ///
    /// Applies various optimization strategies to improve pipeline performance
    /// and resource usage.
    pub fn optimize_pipeline(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<VisualPipelineConfig> {
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
        Ok(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Visual Pipeline Builder</title>
    <link rel="stylesheet" href="visual_builder.css">
</head>
<body>
    <div id="pipeline-canvas"></div>
    <div id="component-library"></div>
    <script src="visual_builder.js"></script>
</body>
</html>"#
            .to_string())
    }

    /// Generate JavaScript logic for the visual builder
    fn generate_javascript_logic(&self) -> crate::error::Result<String> {
        Ok(r#"
// Visual Pipeline Builder JavaScript
console.log('Visual Pipeline Builder loaded');
"#
        .to_string())
    }

    /// Generate CSS styling for the visual builder
    fn generate_css_styles(&self) -> crate::error::Result<String> {
        Ok(r#"
/* Visual Pipeline Builder CSS */
#pipeline-canvas { width: 100%; height: 600px; border: 1px solid #ccc; }
"#
        .to_string())
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
    fn generate_pipeline_documentation(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<PipelineDocumentation> {
        Ok(PipelineDocumentation {
            overview: format!("Pipeline: {}", config.name),
            components: config
                .components
                .iter()
                .map(|c| format!("- {}: {}", c.name, c.description))
                .collect(),
            data_flow: self.describe_data_flow(config)?,
            performance_notes: self.generate_performance_notes(config)?,
            usage_examples: self.generate_usage_examples(config)?,
        })
    }

    /// Analyze pipeline dependencies
    fn analyze_dependencies(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<Vec<String>> {
        let mut dependencies = HashSet::new();

        for component in &config.components {
            dependencies.extend(component.dependencies.iter().cloned());
        }

        Ok(dependencies.into_iter().collect())
    }

    /// Generate performance optimization hints
    fn generate_performance_hints(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<Vec<PerformanceHint>> {
        let mut hints = Vec::new();

        // Analyze for common performance issues
        if config.components.len() > 10 {
            hints.push(PerformanceHint {
                category: "Complexity".to_string(),
                message:
                    "Consider breaking down complex pipelines into smaller, reusable components"
                        .to_string(),
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
                component
                    .properties
                    .insert("use_simd".to_string(), "true".to_string());
            }
        }
        Ok(())
    }

    fn optimize_data_flow(&self, _config: &mut VisualPipelineConfig) -> crate::error::Result<()> {
        // Data flow optimizations
        Ok(())
    }

    fn optimize_resource_usage(
        &self,
        _config: &mut VisualPipelineConfig,
    ) -> crate::error::Result<()> {
        // Resource usage optimizations
        Ok(())
    }

    fn describe_data_flow(&self, config: &VisualPipelineConfig) -> crate::error::Result<String> {
        Ok(format!(
            "Data flows through {} components",
            config.components.len()
        ))
    }

    fn generate_performance_notes(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<String> {
        Ok(format!(
            "Pipeline with {} components",
            config.components.len()
        ))
    }

    fn generate_usage_examples(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<Vec<String>> {
        Ok(vec![format!(
            "let pipeline = {}Pipeline::new()?;",
            config.name
        )])
    }

    fn can_parallelize(&self, config: &VisualPipelineConfig) -> crate::error::Result<bool> {
        Ok(config.components.len() > 2)
    }

    // Import methods
    fn import_from_json(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        Ok(serde_json::from_str(content)?)
    }

    fn import_from_yaml(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        #[cfg(feature = "serde")]
        {
            serde_yaml::from_str(content)
                .map_err(|e| crate::error::SklearsError::SerializationError(e.to_string()))
        }
        #[cfg(not(feature = "serde"))]
        {
            let _ = content;
            Err(crate::error::SklearsError::NotImplemented(
                "YAML import requires the 'serde' feature".to_string(),
            ))
        }
    }

    /// Import a scikit-learn pipeline from its JSON interchange representation.
    ///
    /// Expected schema:
    /// ```json
    /// {
    ///   "type": "Pipeline",
    ///   "name": "my_pipeline",
    ///   "steps": [
    ///     { "name": "scaler",  "type": "StandardScaler", "params": {} },
    ///     { "name": "clf",     "type": "RandomForestClassifier",
    ///       "params": { "n_estimators": "100", "max_depth": "5" } }
    ///   ]
    /// }
    /// ```
    fn import_from_sklearn(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        // Parse the top-level JSON object.
        let root: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| crate::error::SklearsError::SerializationError(e.to_string()))?;

        let pipeline_type = root
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("Pipeline");

        if pipeline_type != "Pipeline" {
            return Err(crate::error::SklearsError::InvalidOperation(format!(
                "sklearn import: expected top-level type 'Pipeline', got '{pipeline_type}'"
            )));
        }

        let name = root
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("sklearn_pipeline")
            .to_string();

        let steps = root
            .get("steps")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                crate::error::SklearsError::InvalidOperation(
                    "sklearn import: 'steps' array is required".to_string(),
                )
            })?;

        let mut components = Vec::new();
        let mut connections: Vec<ComponentConnection> = Vec::new();
        let layout_x_step = 200.0_f64;

        for (idx, step) in steps.iter().enumerate() {
            let step_name = step
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("step")
                .to_string();

            let step_type = step
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();

            // Classify each sklearn estimator into a SkleaRS component category.
            let component_type = classify_sklearn_type(&step_type);

            // Convert params object into a HashMap<String, String>.
            let mut properties: HashMap<String, String> = HashMap::new();
            properties.insert("sklearn_type".to_string(), step_type.clone());
            if let Some(params) = step.get("params").and_then(|v| v.as_object()) {
                for (k, v) in params {
                    let val_str = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    properties.insert(k.clone(), val_str);
                }
            }

            let instance = ComponentInstance {
                id: format!("step_{idx}"),
                name: step_name.clone(),
                component_type,
                properties,
                position: ComponentPosition {
                    x: layout_x_step * idx as f64,
                    y: 100.0,
                    width: 160.0,
                    height: 80.0,
                },
                dependencies: vec![],
                description: format!("sklearn {step_type}"),
            };

            // Connect each step to the previous one.
            if idx > 0 {
                connections.push(ComponentConnection {
                    from_component: format!("step_{}", idx - 1),
                    from_port: "output".to_string(),
                    to_component: format!("step_{idx}"),
                    to_port: "input".to_string(),
                    data_type: "array".to_string(),
                });
            }

            components.push(instance);
        }

        Ok(VisualPipelineConfig {
            name,
            description: "Imported from scikit-learn pipeline JSON".to_string(),
            components,
            connections,
            layout: CanvasLayout {
                width: 1920.0,
                height: 1080.0,
                zoom_level: 1.0,
                grid_enabled: true,
                snap_to_grid: false,
            },
            metadata: {
                let mut m = HashMap::new();
                m.insert("source".to_string(), "sklearn".to_string());
                m
            },
            settings: PipelineSettings::default(),
        })
    }

    /// Import a PyTorch model from its JSON description.
    ///
    /// Expected schema (subset of TorchScript JSON export or a custom
    /// SkleaRS-defined interchange format):
    /// ```json
    /// {
    ///   "type": "Sequential",
    ///   "name": "my_model",
    ///   "layers": [
    ///     { "type": "Linear",   "params": { "in_features": "784", "out_features": "256" } },
    ///     { "type": "ReLU",     "params": {} },
    ///     { "type": "Linear",   "params": { "in_features": "256", "out_features": "10"  } },
    ///     { "type": "Softmax",  "params": { "dim": "1" } }
    ///   ]
    /// }
    /// ```
    fn import_from_torch(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        let root: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| crate::error::SklearsError::SerializationError(e.to_string()))?;

        let model_type = root
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("Sequential");

        // We support Sequential and ModuleList at the top level.
        if !matches!(model_type, "Sequential" | "ModuleList" | "Module") {
            return Err(crate::error::SklearsError::InvalidOperation(format!(
                "torch import: unsupported top-level model type '{model_type}'. \
                 Supported: Sequential, ModuleList, Module"
            )));
        }

        let name = root
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("torch_model")
            .to_string();

        let layers = root
            .get("layers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                crate::error::SklearsError::InvalidOperation(
                    "torch import: 'layers' array is required".to_string(),
                )
            })?;

        let mut components = Vec::new();
        let mut connections: Vec<ComponentConnection> = Vec::new();

        for (idx, layer) in layers.iter().enumerate() {
            let layer_type = layer
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();

            let component_type = classify_torch_layer(&layer_type);

            let mut properties: HashMap<String, String> = HashMap::new();
            properties.insert("torch_layer_type".to_string(), layer_type.clone());
            if let Some(params) = layer.get("params").and_then(|v| v.as_object()) {
                for (k, v) in params {
                    let val_str = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    properties.insert(k.clone(), val_str);
                }
            }

            let instance = ComponentInstance {
                id: format!("layer_{idx}"),
                name: format!("{layer_type}_{idx}"),
                component_type,
                properties,
                position: ComponentPosition {
                    x: 200.0 * idx as f64,
                    y: 100.0,
                    width: 160.0,
                    height: 80.0,
                },
                dependencies: vec![],
                description: format!("PyTorch {layer_type} layer"),
            };

            if idx > 0 {
                connections.push(ComponentConnection {
                    from_component: format!("layer_{}", idx - 1),
                    from_port: "output".to_string(),
                    to_component: format!("layer_{idx}"),
                    to_port: "input".to_string(),
                    data_type: "tensor".to_string(),
                });
            }

            components.push(instance);
        }

        Ok(VisualPipelineConfig {
            name,
            description: format!("Imported from PyTorch {model_type} JSON"),
            components,
            connections,
            layout: CanvasLayout {
                width: 1920.0,
                height: 1080.0,
                zoom_level: 1.0,
                grid_enabled: true,
                snap_to_grid: false,
            },
            metadata: {
                let mut m = HashMap::new();
                m.insert("source".to_string(), "torch".to_string());
                m.insert("model_type".to_string(), model_type.to_string());
                m
            },
            settings: PipelineSettings::default(),
        })
    }

    /// Import a pipeline from its SkleaRS DSL macro text representation.
    ///
    /// The DSL macro format is a minimal line-oriented description:
    /// ```text
    /// pipeline my_pipeline {
    ///   step standard_scaler  type=preprocessing
    ///   step random_forest    type=model  n_estimators=100
    ///   connect standard_scaler -> random_forest
    /// }
    /// ```
    ///
    /// Lines beginning with `step` define components; lines beginning with
    /// `connect` define directed edges.  All other lines (including the outer
    /// `pipeline { }` braces) are treated as structural delimiters or comments.
    fn import_from_dsl_macro(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        let mut name = "dsl_pipeline".to_string();
        let mut components: Vec<ComponentInstance> = Vec::new();
        let mut connections: Vec<ComponentConnection> = Vec::new();
        let mut x_pos = 0.0_f64;

        for raw_line in content.lines() {
            let line = raw_line.trim();

            // Outer `pipeline <name> {` block opener.
            if let Some(rest) = line.strip_prefix("pipeline ") {
                let block_name = rest.trim_end_matches('{').trim().to_string();
                if !block_name.is_empty() {
                    name = block_name;
                }
                continue;
            }

            // `step <id>  key=val ...`
            if let Some(rest) = line.strip_prefix("step ") {
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if tokens.is_empty() {
                    continue;
                }
                let step_id = tokens[0].to_string();
                let mut properties: HashMap<String, String> = HashMap::new();
                let mut component_type = "generic".to_string();

                for token in tokens.iter().skip(1) {
                    if let Some((k, v)) = token.split_once('=') {
                        if k == "type" {
                            component_type = v.to_string();
                        } else {
                            properties.insert(k.to_string(), v.to_string());
                        }
                    }
                }

                components.push(ComponentInstance {
                    id: step_id.clone(),
                    name: step_id,
                    component_type,
                    properties,
                    position: ComponentPosition {
                        x: x_pos,
                        y: 100.0,
                        width: 160.0,
                        height: 80.0,
                    },
                    dependencies: vec![],
                    description: "Imported from DSL macro".to_string(),
                });
                x_pos += 200.0;
                continue;
            }

            // `connect <from> -> <to> [data_type=<type>]`
            if let Some(rest) = line.strip_prefix("connect ") {
                // Split on '->' to get from and to (with optional trailing params).
                let parts: Vec<&str> = rest.splitn(2, "->").collect();
                if parts.len() != 2 {
                    continue;
                }
                let from_id = parts[0].trim().to_string();
                let to_part = parts[1].trim();

                // The `to` side may contain `data_type=…` after whitespace.
                let (to_id, data_type) =
                    if let Some((id_part, kv_part)) = to_part.split_once(char::is_whitespace) {
                        let dt = kv_part
                            .split_whitespace()
                            .find(|t| t.starts_with("data_type="))
                            .and_then(|t| t.split_once('='))
                            .map(|(_, v)| v.to_string())
                            .unwrap_or_else(|| "array".to_string());
                        (id_part.trim().to_string(), dt)
                    } else {
                        (to_part.to_string(), "array".to_string())
                    };

                connections.push(ComponentConnection {
                    from_component: from_id,
                    from_port: "output".to_string(),
                    to_component: to_id,
                    to_port: "input".to_string(),
                    data_type,
                });
                continue;
            }

            // Skip `}`, blank lines, and comment lines (`//`, `#`).
        }

        if components.is_empty() {
            return Err(crate::error::SklearsError::InvalidOperation(
                "dsl_macro import: no 'step' declarations found in input".to_string(),
            ));
        }

        Ok(VisualPipelineConfig {
            name,
            description: "Imported from SkleaRS DSL macro".to_string(),
            components,
            connections,
            layout: CanvasLayout {
                width: 1920.0,
                height: 1080.0,
                zoom_level: 1.0,
                grid_enabled: true,
                snap_to_grid: true,
            },
            metadata: {
                let mut m = HashMap::new();
                m.insert("source".to_string(), "dsl_macro".to_string());
                m
            },
            settings: PipelineSettings::default(),
        })
    }

    /// Import a model from the ONNX JSON interchange format.
    ///
    /// ONNX has an official JSON representation (produced by `onnx.helper` or
    /// `protoc --encode=onnx.ModelProto`).  We parse the subset relevant for
    /// inference graph reconstruction:
    ///
    /// ```json
    /// {
    ///   "irVersion": 8,
    ///   "opsetImport": [{ "version": 17 }],
    ///   "graph": {
    ///     "name": "my_graph",
    ///     "node": [
    ///       { "opType": "Conv",    "name": "conv1", "input": ["X"],     "output": ["Y"] },
    ///       { "opType": "Relu",    "name": "relu1", "input": ["Y"],     "output": ["Z"] },
    ///       { "opType": "Flatten", "name": "flat1", "input": ["Z"],     "output": ["W"] },
    ///       { "opType": "Gemm",    "name": "gemm1", "input": ["W","B"], "output": ["O"] }
    ///     ]
    ///   }
    /// }
    /// ```
    fn import_from_onnx(&self, content: &str) -> crate::error::Result<VisualPipelineConfig> {
        let root: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| crate::error::SklearsError::SerializationError(e.to_string()))?;

        // Validate this looks like an ONNX ModelProto JSON.
        if root.get("graph").is_none() {
            return Err(crate::error::SklearsError::InvalidOperation(
                "onnx import: top-level 'graph' field is required".to_string(),
            ));
        }

        let graph = &root["graph"];
        let name = graph
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("onnx_graph")
            .to_string();

        let ir_version = root.get("irVersion").and_then(|v| v.as_u64()).unwrap_or(0);

        let nodes = graph
            .get("node")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                crate::error::SklearsError::InvalidOperation(
                    "onnx import: 'graph.node' array is required".to_string(),
                )
            })?;

        // Build a map from ONNX tensor name → component id so we can reconstruct
        // the dataflow graph from input/output tensor names.
        let mut output_tensor_to_node: HashMap<String, String> = HashMap::new();
        let mut components = Vec::new();
        let mut connections: Vec<ComponentConnection> = Vec::new();

        for (idx, node) in nodes.iter().enumerate() {
            let op_type = node
                .get("opType")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();

            let node_name = node
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{op_type}_{idx}"));

            let node_id = format!("node_{idx}");
            let component_type = classify_onnx_op(&op_type);

            // Collect node attributes as component properties.
            let mut properties: HashMap<String, String> = HashMap::new();
            properties.insert("onnx_op_type".to_string(), op_type.clone());
            properties.insert("onnx_ir_version".to_string(), ir_version.to_string());
            if let Some(attrs) = node.get("attribute").and_then(|v| v.as_array()) {
                for attr in attrs {
                    let attr_name = attr.get("name").and_then(|v| v.as_str()).unwrap_or("attr");
                    let attr_val = attr
                        .get("f")
                        .or_else(|| attr.get("i"))
                        .or_else(|| attr.get("s"))
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    properties.insert(attr_name.to_string(), attr_val);
                }
            }

            // Register every output tensor produced by this node.
            if let Some(outputs) = node.get("output").and_then(|v| v.as_array()) {
                for out in outputs {
                    if let Some(tensor_name) = out.as_str() {
                        output_tensor_to_node.insert(tensor_name.to_string(), node_id.clone());
                    }
                }
            }

            components.push(ComponentInstance {
                id: node_id,
                name: node_name,
                component_type,
                properties,
                position: ComponentPosition {
                    x: 200.0 * idx as f64,
                    y: 100.0,
                    width: 160.0,
                    height: 80.0,
                },
                dependencies: vec![],
                description: format!("ONNX op: {op_type}"),
            });
        }

        // Second pass: resolve input tensor names → source node edges.
        for (idx, node) in nodes.iter().enumerate() {
            let to_id = format!("node_{idx}");
            if let Some(inputs) = node.get("input").and_then(|v| v.as_array()) {
                for inp in inputs {
                    if let Some(tensor_name) = inp.as_str() {
                        if let Some(from_id) = output_tensor_to_node.get(tensor_name) {
                            connections.push(ComponentConnection {
                                from_component: from_id.clone(),
                                from_port: tensor_name.to_string(),
                                to_component: to_id.clone(),
                                to_port: tensor_name.to_string(),
                                data_type: "tensor".to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(VisualPipelineConfig {
            name,
            description: "Imported from ONNX model JSON".to_string(),
            components,
            connections,
            layout: CanvasLayout {
                width: 1920.0,
                height: 1080.0,
                zoom_level: 1.0,
                grid_enabled: true,
                snap_to_grid: false,
            },
            metadata: {
                let mut m = HashMap::new();
                m.insert("source".to_string(), "onnx".to_string());
                m.insert("ir_version".to_string(), ir_version.to_string());
                m
            },
            settings: PipelineSettings::default(),
        })
    }
}

impl Default for VisualPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Classification helpers for the import functions
// ---------------------------------------------------------------------------

/// Map a scikit-learn estimator class name to a SkleaRS component_type string.
fn classify_sklearn_type(sklearn_type: &str) -> String {
    match sklearn_type {
        // Preprocessing / transformers
        "StandardScaler"
        | "MinMaxScaler"
        | "RobustScaler"
        | "MaxAbsScaler"
        | "Normalizer"
        | "PowerTransformer"
        | "QuantileTransformer" => "preprocessing".to_string(),
        "PCA" | "TruncatedSVD" | "FastICA" | "NMF" | "FactorAnalysis" => {
            "dimensionality_reduction".to_string()
        }
        "SimpleImputer" | "KNNImputer" | "IterativeImputer" => "imputer".to_string(),
        "OneHotEncoder" | "OrdinalEncoder" | "LabelEncoder" | "TargetEncoder" => {
            "encoder".to_string()
        }
        "SelectKBest" | "SelectPercentile" | "VarianceThreshold" | "RFE" => {
            "feature_selection".to_string()
        }
        // Classifiers
        "LogisticRegression"
        | "SVC"
        | "LinearSVC"
        | "NuSVC"
        | "RandomForestClassifier"
        | "GradientBoostingClassifier"
        | "AdaBoostClassifier"
        | "BaggingClassifier"
        | "KNeighborsClassifier"
        | "DecisionTreeClassifier"
        | "GaussianNB"
        | "BernoulliNB"
        | "MultinomialNB"
        | "MLPClassifier"
        | "ExtraTreesClassifier" => "classifier".to_string(),
        // Regressors
        "LinearRegression"
        | "Ridge"
        | "Lasso"
        | "ElasticNet"
        | "SVR"
        | "NuSVR"
        | "RandomForestRegressor"
        | "GradientBoostingRegressor"
        | "AdaBoostRegressor"
        | "KNeighborsRegressor"
        | "DecisionTreeRegressor"
        | "MLPRegressor"
        | "ExtraTreesRegressor" => "regressor".to_string(),
        // Clustering
        "KMeans"
        | "MiniBatchKMeans"
        | "DBSCAN"
        | "AgglomerativeClustering"
        | "SpectralClustering"
        | "MeanShift"
        | "Birch"
        | "OPTICS" => "clustering".to_string(),
        // Pipelines / meta-estimators
        "Pipeline" | "FeatureUnion" | "ColumnTransformer" => "pipeline".to_string(),
        // Fallback
        _ => "model".to_string(),
    }
}

/// Map a PyTorch layer type name to a SkleaRS component_type string.
fn classify_torch_layer(layer_type: &str) -> String {
    match layer_type {
        "Linear" | "Bilinear" | "LazyLinear" => "linear".to_string(),
        "Conv1d" | "Conv2d" | "Conv3d" | "ConvTranspose1d" | "ConvTranspose2d"
        | "ConvTranspose3d" | "LazyConv2d" => "convolution".to_string(),
        "RNN" | "LSTM" | "GRU" | "RNNCell" | "LSTMCell" | "GRUCell" => "recurrent".to_string(),
        "MultiheadAttention" | "Transformer" | "TransformerEncoder" | "TransformerDecoder" => {
            "attention".to_string()
        }
        "ReLU" | "LeakyReLU" | "PReLU" | "ELU" | "SELU" | "GELU" | "Sigmoid" | "Tanh"
        | "Softmax" | "LogSoftmax" | "Mish" | "SiLU" => "activation".to_string(),
        "BatchNorm1d" | "BatchNorm2d" | "BatchNorm3d" | "LayerNorm" | "GroupNorm"
        | "InstanceNorm1d" | "InstanceNorm2d" => "normalization".to_string(),
        "Dropout" | "Dropout2d" | "AlphaDropout" => "regularization".to_string(),
        "MaxPool1d" | "MaxPool2d" | "AvgPool1d" | "AvgPool2d" | "AdaptiveAvgPool2d"
        | "AdaptiveMaxPool2d" => "pooling".to_string(),
        "Flatten" | "Reshape" | "Permute" | "Embedding" | "EmbeddingBag" => "reshape".to_string(),
        _ => "layer".to_string(),
    }
}

/// Map an ONNX operator type to a SkleaRS component_type string.
fn classify_onnx_op(op_type: &str) -> String {
    match op_type {
        "Gemm" | "MatMul" => "linear".to_string(),
        "Conv" | "ConvTranspose" | "ConvInteger" => "convolution".to_string(),
        "LSTM" | "GRU" | "RNN" => "recurrent".to_string(),
        "Attention" => "attention".to_string(),
        "Relu" | "LeakyRelu" | "PRelu" | "Elu" | "Selu" | "Gelu" | "Sigmoid" | "Tanh"
        | "Softmax" | "LogSoftmax" | "Mish" => "activation".to_string(),
        "BatchNormalization"
        | "InstanceNormalization"
        | "LayerNormalization"
        | "GroupNormalization" => "normalization".to_string(),
        "Dropout" => "regularization".to_string(),
        "MaxPool" | "AveragePool" | "GlobalAveragePool" | "GlobalMaxPool" => "pooling".to_string(),
        "Flatten" | "Reshape" | "Transpose" | "Squeeze" | "Unsqueeze" => "reshape".to_string(),
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Sqrt" | "Exp" | "Log" | "Abs" | "Neg"
        | "Floor" | "Ceil" | "Round" => "elementwise".to_string(),
        "Concat" | "Split" | "Slice" | "Gather" | "Scatter" | "Pad" | "Tile" => {
            "tensor_ops".to_string()
        }
        "Constant" | "Identity" => "utility".to_string(),
        _ => "op".to_string(),
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

impl Default for ComponentLibrary {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for PipelineCanvas {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn generate_dsl_from_visual(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<String> {
        // Generate DSL macro code from visual configuration
        Ok(format!("// Generated DSL for {}", config.name))
    }

    pub fn generate_rust_implementation(
        &self,
        config: &VisualPipelineConfig,
    ) -> crate::error::Result<String> {
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

impl Default for PipelineValidator {
    fn default() -> Self {
        Self
    }
}

impl PipelineValidator {
    pub fn new() -> Self {
        Self
    }
    pub fn validate_visual_pipeline(
        &self,
        _config: &VisualPipelineConfig,
    ) -> crate::error::Result<ValidationResult> {
        Ok(ValidationResult {
            is_valid: true,
            error_message: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PipelineExportManager;

impl Default for PipelineExportManager {
    fn default() -> Self {
        Self
    }
}

impl PipelineExportManager {
    pub fn new() -> Self {
        Self
    }
    pub fn export(
        &self,
        _config: &VisualPipelineConfig,
        _format: ExportFormat,
    ) -> crate::error::Result<String> {
        Ok("// Exported code".to_string())
    }
}

// Supporting types
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub error_message: Option<String>,
}

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
pub struct PipelineDocumentation {
    pub overview: String,
    pub components: Vec<String>,
    pub data_flow: String,
    pub performance_notes: String,
    pub usage_examples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceHint {
    pub category: String,
    pub message: String,
    pub severity: String,
}

#[derive(Debug, Clone)]
pub struct PipelineImportData {
    pub format: ImportFormat,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct WebInterface {
    pub html_template: String,
    pub javascript_code: String,
    pub css_styling: String,
    pub component_definitions: String,
    pub api_endpoints: Vec<ApiEndpoint>,
    pub websocket_handlers: Vec<WebSocketHandler>,
}

#[derive(Debug, Clone)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct WebSocketHandler {
    pub event: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum ImportFormat {
    Json,
    Yaml,
    SklearnPipeline,
    TorchScript,
    DslMacro,
    OnnxModel,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Yaml,
    RustCode,
    PythonCode,
    DslMacro,
}

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

    // -----------------------------------------------------------------
    // import_from_sklearn tests
    // -----------------------------------------------------------------

    #[test]
    fn test_import_from_sklearn_minimal_pipeline() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{
            "type": "Pipeline",
            "name": "iris_pipeline",
            "steps": [
                { "name": "scaler", "type": "StandardScaler", "params": {} },
                { "name": "clf",    "type": "LogisticRegression",
                  "params": { "max_iter": "1000", "C": "1.0" } }
            ]
        }"#;

        let config = builder
            .import_pipeline(&PipelineImportData {
                format: ImportFormat::SklearnPipeline,
                content: json.to_string(),
            })
            .expect("sklearn import should succeed");

        assert_eq!(config.name, "iris_pipeline");
        assert_eq!(config.components.len(), 2);
        assert_eq!(config.connections.len(), 1);

        // scaler is preprocessing
        assert_eq!(config.components[0].name, "scaler");
        assert_eq!(config.components[0].component_type, "preprocessing");

        // classifier is a classifier
        assert_eq!(config.components[1].name, "clf");
        assert_eq!(config.components[1].component_type, "classifier");
        assert_eq!(
            config.components[1]
                .properties
                .get("max_iter")
                .map(|s| s.as_str()),
            Some("1000")
        );

        // Connection links step 0 → step 1.
        assert_eq!(config.connections[0].from_component, "step_0");
        assert_eq!(config.connections[0].to_component, "step_1");

        assert_eq!(
            config.metadata.get("source").map(|s| s.as_str()),
            Some("sklearn")
        );
    }

    #[test]
    fn test_import_from_sklearn_wrong_type_fails() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{ "type": "RandomForestClassifier", "steps": [] }"#;
        let result = builder.import_pipeline(&PipelineImportData {
            format: ImportFormat::SklearnPipeline,
            content: json.to_string(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_import_from_sklearn_missing_steps_fails() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{ "type": "Pipeline", "name": "empty" }"#;
        let result = builder.import_pipeline(&PipelineImportData {
            format: ImportFormat::SklearnPipeline,
            content: json.to_string(),
        });
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // import_from_torch tests
    // -----------------------------------------------------------------

    #[test]
    fn test_import_from_torch_sequential() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{
            "type": "Sequential",
            "name": "two_layer_net",
            "layers": [
                { "type": "Linear",  "params": { "in_features": "784", "out_features": "256" } },
                { "type": "ReLU",    "params": {} },
                { "type": "Linear",  "params": { "in_features": "256", "out_features": "10" } },
                { "type": "Softmax", "params": { "dim": "1" } }
            ]
        }"#;

        let config = builder
            .import_pipeline(&PipelineImportData {
                format: ImportFormat::TorchScript,
                content: json.to_string(),
            })
            .expect("torch import should succeed");

        assert_eq!(config.name, "two_layer_net");
        assert_eq!(config.components.len(), 4);
        // Connections: 3 (each layer connected to next).
        assert_eq!(config.connections.len(), 3);

        assert_eq!(config.components[0].component_type, "linear");
        assert_eq!(config.components[1].component_type, "activation");

        assert_eq!(
            config.metadata.get("source").map(|s| s.as_str()),
            Some("torch")
        );
        assert_eq!(
            config.metadata.get("model_type").map(|s| s.as_str()),
            Some("Sequential")
        );
    }

    #[test]
    fn test_import_from_torch_unsupported_type_fails() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{ "type": "DataParallel", "layers": [] }"#;
        let result = builder.import_pipeline(&PipelineImportData {
            format: ImportFormat::TorchScript,
            content: json.to_string(),
        });
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // import_from_dsl_macro tests
    // -----------------------------------------------------------------

    #[test]
    fn test_import_from_dsl_macro_basic() {
        let mut builder = VisualPipelineBuilder::new();
        let dsl = "
pipeline iris_dsl {
    step scaler type=preprocessing
    step clf    type=classifier n_estimators=100
    connect scaler -> clf
}
";
        let config = builder
            .import_pipeline(&PipelineImportData {
                format: ImportFormat::DslMacro,
                content: dsl.to_string(),
            })
            .expect("dsl_macro import should succeed");

        assert_eq!(config.name, "iris_dsl");
        assert_eq!(config.components.len(), 2);
        assert_eq!(config.connections.len(), 1);

        assert_eq!(config.components[0].id, "scaler");
        assert_eq!(config.components[0].component_type, "preprocessing");
        assert_eq!(config.components[1].component_type, "classifier");
        assert_eq!(
            config.components[1]
                .properties
                .get("n_estimators")
                .map(|s| s.as_str()),
            Some("100")
        );

        assert_eq!(config.connections[0].from_component, "scaler");
        assert_eq!(config.connections[0].to_component, "clf");

        assert_eq!(
            config.metadata.get("source").map(|s| s.as_str()),
            Some("dsl_macro")
        );
    }

    #[test]
    fn test_import_from_dsl_macro_empty_fails() {
        let mut builder = VisualPipelineBuilder::new();
        let result = builder.import_pipeline(&PipelineImportData {
            format: ImportFormat::DslMacro,
            content: "pipeline empty { }".to_string(),
        });
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // import_from_onnx tests
    // -----------------------------------------------------------------

    #[test]
    fn test_import_from_onnx_minimal_graph() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{
            "irVersion": 8,
            "opsetImport": [{ "version": 17 }],
            "graph": {
                "name": "simple_fc",
                "node": [
                    {
                        "opType": "Gemm",
                        "name": "gemm0",
                        "input": ["X", "W0", "B0"],
                        "output": ["Y0"]
                    },
                    {
                        "opType": "Relu",
                        "name": "relu0",
                        "input": ["Y0"],
                        "output": ["Z0"]
                    },
                    {
                        "opType": "Gemm",
                        "name": "gemm1",
                        "input": ["Z0", "W1", "B1"],
                        "output": ["out"]
                    }
                ]
            }
        }"#;

        let config = builder
            .import_pipeline(&PipelineImportData {
                format: ImportFormat::OnnxModel,
                content: json.to_string(),
            })
            .expect("onnx import should succeed");

        assert_eq!(config.name, "simple_fc");
        assert_eq!(config.components.len(), 3);

        // Gemm → linear, Relu → activation.
        assert_eq!(config.components[0].component_type, "linear");
        assert_eq!(config.components[1].component_type, "activation");
        assert_eq!(config.components[2].component_type, "linear");

        // relu0 receives from gemm0 (Y0 tensor).
        let relu_conn = config
            .connections
            .iter()
            .find(|c| c.to_component == "node_1")
            .expect("should have connection to relu0");
        assert_eq!(relu_conn.from_component, "node_0");
        assert_eq!(relu_conn.data_type, "tensor");

        assert_eq!(
            config.metadata.get("source").map(|s| s.as_str()),
            Some("onnx")
        );
        assert_eq!(
            config.metadata.get("ir_version").map(|s| s.as_str()),
            Some("8")
        );
    }

    #[test]
    fn test_import_from_onnx_missing_graph_fails() {
        let mut builder = VisualPipelineBuilder::new();
        let json = r#"{ "irVersion": 8 }"#;
        let result = builder.import_pipeline(&PipelineImportData {
            format: ImportFormat::OnnxModel,
            content: json.to_string(),
        });
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // classifier helper tests
    // -----------------------------------------------------------------

    #[test]
    fn test_classify_sklearn_type() {
        assert_eq!(classify_sklearn_type("StandardScaler"), "preprocessing");
        assert_eq!(
            classify_sklearn_type("RandomForestClassifier"),
            "classifier"
        );
        assert_eq!(classify_sklearn_type("LinearRegression"), "regressor");
        assert_eq!(classify_sklearn_type("KMeans"), "clustering");
        assert_eq!(classify_sklearn_type("PCA"), "dimensionality_reduction");
        assert_eq!(classify_sklearn_type("UnknownEstimator"), "model");
    }

    #[test]
    fn test_classify_torch_layer() {
        assert_eq!(classify_torch_layer("Linear"), "linear");
        assert_eq!(classify_torch_layer("ReLU"), "activation");
        assert_eq!(classify_torch_layer("BatchNorm2d"), "normalization");
        assert_eq!(classify_torch_layer("Dropout"), "regularization");
        assert_eq!(classify_torch_layer("MaxPool2d"), "pooling");
        assert_eq!(classify_torch_layer("Flatten"), "reshape");
        assert_eq!(classify_torch_layer("SomeUnknownLayer"), "layer");
    }

    #[test]
    fn test_classify_onnx_op() {
        assert_eq!(classify_onnx_op("Gemm"), "linear");
        assert_eq!(classify_onnx_op("Conv"), "convolution");
        assert_eq!(classify_onnx_op("Relu"), "activation");
        assert_eq!(classify_onnx_op("BatchNormalization"), "normalization");
        assert_eq!(classify_onnx_op("MaxPool"), "pooling");
        assert_eq!(classify_onnx_op("Add"), "elementwise");
        assert_eq!(classify_onnx_op("Concat"), "tensor_ops");
        assert_eq!(classify_onnx_op("UnknownOp"), "op");
    }
}
