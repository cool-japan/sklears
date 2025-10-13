//! Code Generation for Workflows
//!
//! This module provides code generation capabilities for converting workflow definitions
//! into executable code in various programming languages including Rust, Python, and JSON.
//! Supports different code styles, optimization levels, and deployment targets.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{BTreeMap, HashMap};

use super::workflow_definitions::{ExecutionMode, StepDefinition, WorkflowDefinition};

/// Code generation target language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeLanguage {
    /// Rust language
    Rust,
    /// Python language
    Python,
    /// JSON format
    Json,
    /// YAML format
    Yaml,
    /// JavaScript/TypeScript
    JavaScript,
    /// C++ language
    Cpp,
}

/// File format for saving/loading workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    /// Json
    Json,
    /// Yaml
    Yaml,
    /// Toml
    Toml,
    /// Binary
    Binary,
}

/// Code generation configuration
#[derive(Debug, Clone)]
pub struct CodeGenerationConfig {
    /// Target language
    pub language: CodeLanguage,
    /// Code style preferences
    pub style: CodeStyle,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Include comments
    pub include_comments: bool,
    /// Include type annotations
    pub include_type_annotations: bool,
    /// Target deployment environment
    pub deployment_target: DeploymentTarget,
    /// Custom templates
    pub custom_templates: HashMap<String, String>,
}

/// Code style preferences
#[derive(Debug, Clone)]
pub struct CodeStyle {
    /// Indentation size
    pub indent_size: usize,
    /// Use tabs instead of spaces
    pub use_tabs: bool,
    /// Maximum line length
    pub max_line_length: usize,
    /// Naming convention
    pub naming_convention: NamingConvention,
    /// Include error handling
    pub include_error_handling: bool,
}

/// Naming conventions
#[derive(Debug, Clone)]
pub enum NamingConvention {
    /// `snake_case`
    SnakeCase,
    CamelCase,
    /// `PascalCase`
    PascalCase,
    /// kebab-case
    KebabCase,
}

/// Optimization levels
#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    /// No optimization, readable code
    None,
    /// Basic optimizations
    Basic,
    /// Aggressive optimizations
    Aggressive,
    /// Production-ready optimizations
    Production,
}

/// Deployment targets
#[derive(Debug, Clone)]
pub enum DeploymentTarget {
    /// Local development
    Local,
    /// Docker container
    Docker,
    /// Kubernetes
    Kubernetes,
    /// Cloud functions
    CloudFunction,
    /// WebAssembly
    WebAssembly,
    /// Embedded systems
    Embedded,
}

/// Code generator for workflows
#[derive(Debug)]
pub struct CodeGenerator {
    /// Generation configuration
    config: CodeGenerationConfig,
    /// Template engine
    templates: TemplateEngine,
    /// Generated code statistics
    stats: GenerationStatistics,
}

/// Template engine for code generation
#[derive(Debug)]
pub struct TemplateEngine {
    /// Language templates
    templates: HashMap<CodeLanguage, LanguageTemplate>,
    /// Custom template overrides
    custom_overrides: HashMap<String, String>,
}

/// Language-specific templates
#[derive(Debug, Clone)]
pub struct LanguageTemplate {
    /// File header template
    pub header: String,
    /// Import/use statements template
    pub imports: String,
    /// Function definition template
    pub function_def: String,
    /// Step execution template
    pub step_execution: String,
    /// Connection template
    pub connection: String,
    /// Footer template
    pub footer: String,
}

/// Code generation statistics
#[derive(Debug, Clone)]
pub struct GenerationStatistics {
    pub total_lines: usize,
    pub code_lines: usize,
    pub comment_lines: usize,
    pub function_count: usize,
    pub import_count: usize,
    pub generation_time: std::time::Duration,
}

/// Generated code result
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// Generated source code
    pub source_code: String,
    /// Language of generated code
    pub language: CodeLanguage,
    /// Required dependencies
    pub dependencies: Vec<String>,
    /// Compilation/execution instructions
    pub instructions: String,
    /// Generation statistics
    pub statistics: GenerationStatistics,
}

impl CodeGenerator {
    /// Create a new code generator
    #[must_use]
    pub fn new(config: CodeGenerationConfig) -> Self {
        Self {
            config,
            templates: TemplateEngine::new(),
            stats: GenerationStatistics::new(),
        }
    }

    /// Get the code generation language
    #[must_use]
    pub fn language(&self) -> &CodeLanguage {
        &self.config.language
    }

    /// Check if comments are included
    #[must_use]
    pub fn include_comments(&self) -> bool {
        self.config.include_comments
    }

    /// Generate code from workflow definition
    pub fn generate_code(&mut self, workflow: &WorkflowDefinition) -> SklResult<GeneratedCode> {
        let generation_start = std::time::Instant::now();

        let source_code = match self.config.language {
            CodeLanguage::Rust => self.generate_rust_code(workflow)?,
            CodeLanguage::Python => self.generate_python_code(workflow)?,
            CodeLanguage::Json => self.generate_json_code(workflow)?,
            CodeLanguage::Yaml => self.generate_yaml_code(workflow)?,
            CodeLanguage::JavaScript => self.generate_javascript_code(workflow)?,
            CodeLanguage::Cpp => self.generate_cpp_code(workflow)?,
        };

        self.stats.generation_time = generation_start.elapsed();
        self.update_statistics(&source_code);

        Ok(GeneratedCode {
            source_code,
            language: self.config.language.clone(),
            dependencies: self.get_required_dependencies(workflow),
            instructions: self.generate_instructions(),
            statistics: self.stats.clone(),
        })
    }

    /// Generate Rust code
    fn generate_rust_code(&self, workflow: &WorkflowDefinition) -> SklResult<String> {
        let mut code = String::new();

        // Header and imports
        code.push_str(&self.generate_rust_header(workflow));
        code.push_str(&self.generate_rust_imports(workflow));

        // Main function
        code.push_str(&self.generate_rust_main_function(workflow));

        // Step functions
        for step in &workflow.steps {
            code.push_str(&self.generate_rust_step_function(step, workflow)?);
        }

        Ok(code)
    }

    /// Generate Rust header
    fn generate_rust_header(&self, workflow: &WorkflowDefinition) -> String {
        let mut header = String::new();

        if self.config.include_comments {
            header.push_str(&format!(
                "//! Generated Rust code for workflow: {}\n",
                workflow.metadata.name
            ));
            header.push_str(&format!("//! Version: {}\n", workflow.metadata.version));
            if let Some(description) = &workflow.metadata.description {
                header.push_str(&format!("//! Description: {description}\n"));
            }
            header.push_str("//!\n");
            header.push_str(
                "//! This code was automatically generated from a workflow definition.\n",
            );
            header.push_str("//! Do not edit this file directly.\n\n");
        }

        header
    }

    /// Generate Rust imports
    fn generate_rust_imports(&self, _workflow: &WorkflowDefinition) -> String {
        let mut imports = String::new();

        imports.push_str("use sklears_core::{\n");
        imports.push_str("    error::{Result as SklResult, SklearsError},\n");
        imports.push_str("    types::Float,\n");
        imports.push_str("};\n");
        imports.push_str("use scirs2_core::ndarray::{Array1, Array2};\n");
        imports.push_str("use std::collections::HashMap;\n");
        imports.push_str("use serde::{Serialize, Deserialize};\n\n");

        imports
    }

    /// Generate Rust main function
    fn generate_rust_main_function(&self, workflow: &WorkflowDefinition) -> String {
        let function_name = self.convert_name(
            &workflow.metadata.name,
            &self.config.style.naming_convention,
        );
        let mut code = String::new();

        if self.config.include_comments {
            code.push_str(&format!(
                "/// Execute the {} workflow\n",
                workflow.metadata.name
            ));
        }

        code.push_str(&format!(
            "pub fn {function_name}() -> SklResult<HashMap<String, Array2<Float>>> {{\n"
        ));

        // Execution mode setup
        match workflow.execution.mode {
            ExecutionMode::Parallel => {
                code.push_str("    // Parallel execution mode\n");
            }
            ExecutionMode::Sequential => {
                code.push_str("    // Sequential execution mode\n");
            }
            _ => {
                code.push_str("    // Default execution mode\n");
            }
        }

        code.push_str("    let mut results = HashMap::new();\n\n");

        // Generate step calls in execution order
        for step in &workflow.steps {
            let step_func_name = self.convert_name(&step.id, &self.config.style.naming_convention);
            code.push_str(&format!(
                "    let {step_func_name} = {step_func_name}()?;\n"
            ));
        }

        code.push_str("\n    Ok(results)\n");
        code.push_str("}\n\n");

        code
    }

    /// Generate Rust step function
    fn generate_rust_step_function(
        &self,
        step: &StepDefinition,
        _workflow: &WorkflowDefinition,
    ) -> SklResult<String> {
        let function_name = self.convert_name(&step.id, &self.config.style.naming_convention);
        let mut code = String::new();

        if self.config.include_comments {
            code.push_str(&format!(
                "/// Execute step: {} ({})\n",
                step.id, step.algorithm
            ));
            if let Some(description) = &step.description {
                code.push_str(&format!("/// {description}\n"));
            }
        }

        code.push_str(&format!(
            "fn {function_name}() -> SklResult<Array2<Float>> {{\n"
        ));

        // Algorithm-specific implementation
        match step.algorithm.as_str() {
            "StandardScaler" => {
                code.push_str("    // StandardScaler implementation\n");
                code.push_str("    // TODO: Implement actual scaling logic\n");
                code.push_str("    let scaled_data = Array2::zeros((0, 0));\n");
                code.push_str("    Ok(scaled_data)\n");
            }
            "LinearRegression" => {
                code.push_str("    // LinearRegression implementation\n");
                code.push_str("    // TODO: Implement actual regression logic\n");
                code.push_str("    let predictions = Array2::zeros((0, 0));\n");
                code.push_str("    Ok(predictions)\n");
            }
            _ => {
                code.push_str(&format!("    // {} implementation\n", step.algorithm));
                code.push_str("    // TODO: Implement component logic\n");
                code.push_str("    let result = Array2::zeros((0, 0));\n");
                code.push_str("    Ok(result)\n");
            }
        }

        code.push_str("}\n\n");

        Ok(code)
    }

    /// Generate Python code
    fn generate_python_code(&self, workflow: &WorkflowDefinition) -> SklResult<String> {
        let mut code = String::new();

        // Header and imports
        code.push_str(&self.generate_python_header(workflow));
        code.push_str(&self.generate_python_imports());

        // Main function
        code.push_str(&self.generate_python_main_function(workflow));

        // Step functions
        for step in &workflow.steps {
            code.push_str(&self.generate_python_step_function(step)?);
        }

        Ok(code)
    }

    /// Generate Python header
    fn generate_python_header(&self, workflow: &WorkflowDefinition) -> String {
        let mut header = String::new();

        if self.config.include_comments {
            header.push_str(&format!(
                "\"\"\"Generated Python code for workflow: {}\n",
                workflow.metadata.name
            ));
            header.push_str(&format!("Version: {}\n", workflow.metadata.version));
            if let Some(description) = &workflow.metadata.description {
                header.push_str(&format!("Description: {description}\n"));
            }
            header
                .push_str("\nThis code was automatically generated from a workflow definition.\n");
            header.push_str("Do not edit this file directly.\n");
            header.push_str("\"\"\"\n\n");
        }

        header
    }

    /// Generate Python imports
    fn generate_python_imports(&self) -> String {
        let mut imports = String::new();

        imports.push_str("import numpy as np\n");
        imports.push_str("from sklearn.preprocessing import StandardScaler\n");
        imports.push_str("from sklearn.linear_model import LinearRegression\n");
        imports.push_str("from typing import Dict, Any, Optional\n\n");

        imports
    }

    /// Generate Python main function
    fn generate_python_main_function(&self, workflow: &WorkflowDefinition) -> String {
        let function_name =
            self.convert_name(&workflow.metadata.name, &NamingConvention::SnakeCase);
        let mut code = String::new();

        if self.config.include_comments {
            code.push_str(&format!("def {function_name}() -> Dict[str, Any]:\n"));
            code.push_str(&format!(
                "    \"\"\"Execute the {} workflow.\"\"\"\n",
                workflow.metadata.name
            ));
        } else {
            code.push_str(&format!("def {function_name}():\n"));
        }

        code.push_str("    results = {}\n\n");

        // Generate step calls
        for step in &workflow.steps {
            let step_func_name = self.convert_name(&step.id, &NamingConvention::SnakeCase);
            code.push_str(&format!(
                "    results['{}'] = {}()\n",
                step.id, step_func_name
            ));
        }

        code.push_str("\n    return results\n\n");

        code
    }

    /// Generate Python step function
    fn generate_python_step_function(&self, step: &StepDefinition) -> SklResult<String> {
        let function_name = self.convert_name(&step.id, &NamingConvention::SnakeCase);
        let mut code = String::new();

        if self.config.include_comments {
            code.push_str(&format!("def {function_name}():\n"));
            code.push_str(&format!(
                "    \"\"\"Execute step: {} ({}).\"\"\"\n",
                step.id, step.algorithm
            ));
        } else {
            code.push_str(&format!("def {function_name}():\n"));
        }

        // Algorithm-specific implementation
        match step.algorithm.as_str() {
            "StandardScaler" => {
                code.push_str("    scaler = StandardScaler()\n");
                code.push_str("    # TODO: Implement actual scaling logic\n");
                code.push_str("    return scaler\n");
            }
            "LinearRegression" => {
                code.push_str("    model = LinearRegression()\n");
                code.push_str("    # TODO: Implement actual training logic\n");
                code.push_str("    return model\n");
            }
            _ => {
                code.push_str(&format!("    # {} implementation\n", step.algorithm));
                code.push_str("    # TODO: Implement component logic\n");
                code.push_str("    return None\n");
            }
        }

        code.push('\n');

        Ok(code)
    }

    /// Generate JSON code
    fn generate_json_code(&self, workflow: &WorkflowDefinition) -> SklResult<String> {
        match serde_json::to_string_pretty(workflow) {
            Ok(json) => Ok(json),
            Err(e) => Err(SklearsError::InvalidInput(format!(
                "JSON serialization failed: {e}"
            ))),
        }
    }

    /// Generate YAML code
    fn generate_yaml_code(&self, workflow: &WorkflowDefinition) -> SklResult<String> {
        match serde_yaml::to_string(workflow) {
            Ok(yaml) => Ok(yaml),
            Err(e) => Err(SklearsError::InvalidInput(format!(
                "YAML serialization failed: {e}"
            ))),
        }
    }

    /// Generate JavaScript code
    fn generate_javascript_code(&self, workflow: &WorkflowDefinition) -> SklResult<String> {
        let mut code = String::new();

        code.push_str(&format!(
            "// Generated JavaScript code for workflow: {}\n\n",
            workflow.metadata.name
        ));

        code.push_str(&format!(
            "function {}() {{\n",
            self.convert_name(&workflow.metadata.name, &NamingConvention::CamelCase)
        ));

        code.push_str("    const results = {};\n\n");

        for step in &workflow.steps {
            code.push_str(&format!(
                "    results['{}'] = {}();\n",
                step.id,
                self.convert_name(&step.id, &NamingConvention::CamelCase)
            ));
        }

        code.push_str("\n    return results;\n");
        code.push_str("}\n\n");

        // Step functions
        for step in &workflow.steps {
            code.push_str(&format!(
                "function {}() {{\n",
                self.convert_name(&step.id, &NamingConvention::CamelCase)
            ));
            code.push_str(&format!("    // {} implementation\n", step.algorithm));
            code.push_str("    // TODO: Implement component logic\n");
            code.push_str("    return null;\n");
            code.push_str("}\n\n");
        }

        Ok(code)
    }

    /// Generate C++ code
    fn generate_cpp_code(&self, workflow: &WorkflowDefinition) -> SklResult<String> {
        let mut code = String::new();

        // Header
        code.push_str(&format!(
            "// Generated C++ code for workflow: {}\n\n",
            workflow.metadata.name
        ));

        // Includes
        code.push_str("#include <iostream>\n");
        code.push_str("#include <vector>\n");
        code.push_str("#include <map>\n");
        code.push_str("#include <string>\n\n");

        // Main function
        code.push_str(&format!(
            "std::map<std::string, void*> {}() {{\n",
            self.convert_name(&workflow.metadata.name, &NamingConvention::SnakeCase)
        ));

        code.push_str("    std::map<std::string, void*> results;\n\n");

        for step in &workflow.steps {
            code.push_str(&format!(
                "    results[\"{}\"] = {}();\n",
                step.id,
                self.convert_name(&step.id, &NamingConvention::SnakeCase)
            ));
        }

        code.push_str("\n    return results;\n");
        code.push_str("}\n\n");

        // Step functions
        for step in &workflow.steps {
            code.push_str(&format!(
                "void* {}() {{\n",
                self.convert_name(&step.id, &NamingConvention::SnakeCase)
            ));
            code.push_str(&format!("    // {} implementation\n", step.algorithm));
            code.push_str("    // TODO: Implement component logic\n");
            code.push_str("    return nullptr;\n");
            code.push_str("}\n\n");
        }

        Ok(code)
    }

    /// Convert name according to naming convention
    #[must_use]
    pub fn convert_name(&self, name: &str, convention: &NamingConvention) -> String {
        match convention {
            NamingConvention::SnakeCase => name
                .split(|c: char| c.is_whitespace() || c == '-' || c == '_')
                .filter(|segment| !segment.is_empty())
                .map(|segment| segment.to_lowercase())
                .collect::<Vec<_>>()
                .join("_"),
            NamingConvention::CamelCase => {
                let mut result = String::new();
                let mut capitalize_next = false;
                for (i, c) in name.chars().enumerate() {
                    if c.is_whitespace() || c == '_' || c == '-' {
                        capitalize_next = true;
                    } else if i == 0 {
                        result.push(c.to_lowercase().next().unwrap());
                    } else if capitalize_next {
                        result.push(c.to_uppercase().next().unwrap());
                        capitalize_next = false;
                    } else {
                        result.push(c.to_lowercase().next().unwrap());
                    }
                }
                result
            }
            NamingConvention::PascalCase => {
                let mut result = String::new();
                let mut capitalize_next = true;
                for c in name.chars() {
                    if c.is_whitespace() || c == '_' || c == '-' {
                        capitalize_next = true;
                    } else if capitalize_next {
                        result.push(c.to_uppercase().next().unwrap());
                        capitalize_next = false;
                    } else {
                        result.push(c.to_lowercase().next().unwrap());
                    }
                }
                result
            }
            NamingConvention::KebabCase => name
                .to_lowercase()
                .chars()
                .map(|c| {
                    if c.is_whitespace() || c == '_' {
                        '-'
                    } else {
                        c
                    }
                })
                .collect::<String>()
                .replace("--", "-"),
        }
    }

    /// Get required dependencies for workflow
    fn get_required_dependencies(&self, workflow: &WorkflowDefinition) -> Vec<String> {
        let mut dependencies = Vec::new();

        match self.config.language {
            CodeLanguage::Rust => {
                dependencies.push("sklears-core".to_string());
                dependencies.push("scirs2-autograd".to_string());
                dependencies.push("serde".to_string());
            }
            CodeLanguage::Python => {
                dependencies.push("numpy".to_string());
                dependencies.push("scikit-learn".to_string());
            }
            _ => {}
        }

        // Add algorithm-specific dependencies
        for step in &workflow.steps {
            match step.algorithm.as_str() {
                "StandardScaler" | "LinearRegression" => {
                    if matches!(self.config.language, CodeLanguage::Python)
                        && !dependencies.contains(&"scikit-learn".to_string())
                    {
                        dependencies.push("scikit-learn".to_string());
                    }
                }
                _ => {}
            }
        }

        dependencies
    }

    /// Generate compilation/execution instructions
    fn generate_instructions(&self) -> String {
        match self.config.language {
            CodeLanguage::Rust => "To compile and run:\n\
                1. Add dependencies to Cargo.toml\n\
                2. Run 'cargo build'\n\
                3. Run 'cargo run'"
                .to_string(),
            CodeLanguage::Python => "To run:\n\
                1. Install dependencies: pip install numpy scikit-learn\n\
                2. Run: python workflow.py"
                .to_string(),
            CodeLanguage::JavaScript => "To run:\n\
                1. Install Node.js\n\
                2. Run: node workflow.js"
                .to_string(),
            CodeLanguage::Cpp => "To compile and run:\n\
                1. Compile: g++ -o workflow workflow.cpp\n\
                2. Run: ./workflow"
                .to_string(),
            _ => "See language-specific documentation".to_string(),
        }
    }

    /// Update generation statistics
    fn update_statistics(&mut self, source_code: &str) {
        let lines: Vec<&str> = source_code.lines().collect();
        self.stats.total_lines = lines.len();

        self.stats.code_lines = lines
            .iter()
            .filter(|line| {
                !line.trim().is_empty()
                    && !line.trim().starts_with("//")
                    && !line.trim().starts_with('#')
            })
            .count();

        self.stats.comment_lines = lines
            .iter()
            .filter(|line| line.trim().starts_with("//") || line.trim().starts_with('#'))
            .count();

        // Count functions (approximate)
        self.stats.function_count = source_code.matches("fn ").count()
            + source_code.matches("def ").count()
            + source_code.matches("function ").count();

        // Count imports (approximate)
        self.stats.import_count = source_code.matches("use ").count()
            + source_code.matches("import ").count()
            + source_code.matches("#include").count();
    }
}

impl TemplateEngine {
    fn new() -> Self {
        Self {
            templates: HashMap::new(),
            custom_overrides: HashMap::new(),
        }
    }
}

impl GenerationStatistics {
    fn new() -> Self {
        Self {
            total_lines: 0,
            code_lines: 0,
            comment_lines: 0,
            function_count: 0,
            import_count: 0,
            generation_time: std::time::Duration::from_secs(0),
        }
    }
}

impl Default for CodeGenerationConfig {
    fn default() -> Self {
        Self {
            language: CodeLanguage::Rust,
            style: CodeStyle::default(),
            optimization_level: OptimizationLevel::Basic,
            include_comments: true,
            include_type_annotations: true,
            deployment_target: DeploymentTarget::Local,
            custom_templates: HashMap::new(),
        }
    }
}

impl Default for CodeStyle {
    fn default() -> Self {
        Self {
            indent_size: 4,
            use_tabs: false,
            max_line_length: 100,
            naming_convention: NamingConvention::SnakeCase,
            include_error_handling: true,
        }
    }
}

/// Code generation error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum CodeGenerationError {
    /// Template compilation error
    #[error("Template compilation error: {0}")]
    TemplateError(String),
    /// Language backend error
    #[error("Language backend error: {0}")]
    LanguageError(String),
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    /// Workflow validation error
    #[error("Workflow validation error: {0}")]
    WorkflowError(String),
    /// IO error
    #[error("IO error: {0}")]
    IoError(String),
    /// Syntax error in generated code
    #[error("Syntax error in generated code: {0}")]
    SyntaxError(String),
    /// Dependency resolution error
    #[error("Dependency resolution error: {0}")]
    DependencyError(String),
    /// Sklears error
    #[error("Sklears error: {0}")]
    SklearsError(#[from] sklears_core::error::SklearsError),
}

/// Code template for code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTemplate {
    /// Template name
    pub name: String,
    /// Template content
    pub content: String,
    /// Template language
    pub language: CodeLanguage,
    /// Template variables
    pub variables: Vec<String>,
    /// Template metadata
    pub metadata: TemplateMetadata,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Author of the template
    pub author: Option<String>,
    /// Template version
    pub version: String,
    /// Template description
    pub description: Option<String>,
    /// Template tags
    pub tags: Vec<String>,
}

/// Language backend for code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageBackend {
    /// Language name
    pub language: CodeLanguage,
    /// Code generator
    pub generator: String,
    /// Template engine
    pub template_engine: String,
    /// Supported features
    pub features: Vec<String>,
    /// Backend configuration
    pub config: BTreeMap<String, String>,
}

/// Type alias for target language (compatibility)
pub type TargetLanguage = CodeLanguage;

/// Template context for variable substitution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemplateContext {
    /// Template variables
    pub variables: BTreeMap<String, TemplateValue>,
    /// Global constants
    pub constants: BTreeMap<String, String>,
    /// Include paths
    pub include_paths: Vec<String>,
    /// Conditional flags
    pub flags: BTreeMap<String, bool>,
}

/// Template value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateValue {
    /// String value
    String(String),
    /// Number value
    Number(f64),
    /// Boolean value
    Boolean(bool),
    /// Array value
    Array(Vec<TemplateValue>),
    /// Object value
    Object(BTreeMap<String, TemplateValue>),
}

/// Template registry for managing templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateRegistry {
    /// Registered templates
    pub templates: BTreeMap<String, CodeTemplate>,
    /// Template categories
    pub categories: BTreeMap<String, Vec<String>>,
    /// Registry metadata
    pub metadata: RegistryMetadata,
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Registry name
    pub name: String,
    /// Registry version
    pub version: String,
    /// Last updated timestamp
    pub updated_at: String,
    /// Total template count
    pub template_count: usize,
}

impl TemplateRegistry {
    /// Create a new template registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: BTreeMap::new(),
            categories: BTreeMap::new(),
            metadata: RegistryMetadata {
                name: "Default Registry".to_string(),
                version: "1.0.0".to_string(),
                updated_at: chrono::Utc::now().to_rfc3339(),
                template_count: 0,
            },
        }
    }

    /// Register a new template
    pub fn register_template(&mut self, template: CodeTemplate) {
        self.templates.insert(template.name.clone(), template);
        self.metadata.template_count = self.templates.len();
        self.metadata.updated_at = chrono::Utc::now().to_rfc3339();
    }

    /// Get a template by name
    #[must_use]
    pub fn get_template(&self, name: &str) -> Option<&CodeTemplate> {
        self.templates.get(name)
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_language::workflow_definitions::{StepType, WorkflowMetadata};

    #[test]
    fn test_code_generator_creation() {
        let config = CodeGenerationConfig::default();
        let generator = CodeGenerator::new(config);
        assert_eq!(generator.stats.total_lines, 0);
    }

    #[test]
    fn test_rust_code_generation() {
        let mut generator = CodeGenerator::new(CodeGenerationConfig::default());
        let mut workflow = WorkflowDefinition::default();
        workflow.metadata.name = "test_workflow".to_string();
        workflow.steps.push(StepDefinition::new(
            "step1",
            StepType::Transformer,
            "StandardScaler",
        ));

        let result = generator.generate_code(&workflow);
        assert!(result.is_ok());

        let generated = result.unwrap();
        assert!(!generated.source_code.is_empty());
        assert!(matches!(generated.language, CodeLanguage::Rust));
        assert!(!generated.dependencies.is_empty());
    }

    #[test]
    fn test_python_code_generation() {
        let config = CodeGenerationConfig {
            language: CodeLanguage::Python,
            ..Default::default()
        };
        let mut generator = CodeGenerator::new(config);
        let mut workflow = WorkflowDefinition::default();
        workflow.metadata.name = "test_workflow".to_string();
        workflow.steps.push(StepDefinition::new(
            "step1",
            StepType::Transformer,
            "StandardScaler",
        ));

        let result = generator.generate_code(&workflow);
        assert!(result.is_ok());

        let generated = result.unwrap();
        assert!(!generated.source_code.is_empty());
        assert!(matches!(generated.language, CodeLanguage::Python));
        assert!(generated.source_code.contains("def "));
    }

    #[test]
    fn test_json_code_generation() {
        let config = CodeGenerationConfig {
            language: CodeLanguage::Json,
            ..Default::default()
        };
        let mut generator = CodeGenerator::new(config);
        let workflow = WorkflowDefinition::default();

        let result = generator.generate_code(&workflow);
        assert!(result.is_ok());

        let generated = result.unwrap();
        assert!(!generated.source_code.is_empty());
        assert!(generated.source_code.contains("{"));
    }

    #[test]
    fn test_naming_convention_conversion() {
        let config = CodeGenerationConfig::default();
        let generator = CodeGenerator::new(config);

        assert_eq!(
            generator.convert_name("Test Workflow", &NamingConvention::SnakeCase),
            "test_workflow"
        );
        assert_eq!(
            generator.convert_name("test_workflow", &NamingConvention::CamelCase),
            "testWorkflow"
        );
        assert_eq!(
            generator.convert_name("test_workflow", &NamingConvention::PascalCase),
            "TestWorkflow"
        );
        assert_eq!(
            generator.convert_name("test_workflow", &NamingConvention::KebabCase),
            "test-workflow"
        );
    }
}
