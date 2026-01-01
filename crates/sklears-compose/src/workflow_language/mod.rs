//! Workflow Language System
//!
//! A comprehensive system for describing, building, and executing machine learning workflows
//! through multiple interfaces including visual builders, domain-specific language (DSL),
//! and programmatic APIs. This module provides enterprise-grade workflow orchestration
//! with support for complex dependencies, resource management, and multi-language code generation.
//!
//! # Architecture Overview
//!
//! The workflow language system is built on a modular architecture with clear separation
//! of concerns:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                         Workflow Language System                                │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │  Visual Builder  │  DSL Language   │  Component Registry  │  Code Generation   │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │           Workflow Definitions (Core Data Structures)                          │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │                       Workflow Execution Engine                                │
//! ├─────────────────────────────────────────────────────────────────────────────────┤
//! │                     Comprehensive Test Infrastructure                          │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Core Components
//!
//! ## Workflow Definitions
//! Core data structures and type definitions that form the foundation of all workflow
//! representations. Includes metadata, input/output specifications, step definitions,
//! and execution configuration.
//!
//! ## Visual Builder
//! Interactive visual interface for building workflows through drag-and-drop operations.
//! Supports real-time validation, undo/redo functionality, and collaborative editing.
//!
//! ## DSL Language
//! Domain-specific language for expressing workflows in a human-readable text format.
//! Includes a complete lexer/parser implementation with syntax highlighting and
//! intelligent auto-completion.
//!
//! ## Component Registry
//! Centralized registry for managing workflow components with support for versioning,
//! dependency resolution, and plugin architecture.
//!
//! ## Workflow Execution
//! High-performance execution engine with support for parallel processing, resource
//! management, fault tolerance, and comprehensive monitoring.
//!
//! ## Code Generation
//! Multi-language code generation system supporting Rust, Python, JavaScript, and C++
//! with customizable templates and optimization strategies.
//!
//! # Usage Examples
//!
//! ## Creating a Workflow Programmatically
//!
//! ```rust
//! use sklears_compose::workflow_language::{
//!     ExecutionConfig, WorkflowDefinition, WorkflowMetadata,
//! };
//!
//! let workflow = WorkflowDefinition {
//!     metadata: WorkflowMetadata {
//!         name: "ml_pipeline".to_string(),
//!         version: "1.0.0".to_string(),
//!         description: Some("End-to-end ML pipeline".to_string()),
//!         ..Default::default()
//!     },
//!     inputs: vec![],
//!     outputs: vec![],
//!     steps: vec![],
//!     connections: vec![],
//!     execution: ExecutionConfig::default(),
//! };
//! ```
//!
//! ## Using the Visual Builder
//!
//! ```rust
//! use sklears_compose::workflow_language::{
//!     Connection, StepDefinition, StepType, VisualPipelineBuilder,
//! };
//!
//! let mut builder = VisualPipelineBuilder::new();
//!
//! let data_loader = StepDefinition::new("data_loader", StepType::Input, "CsvReader")
//!     .with_output("dataset");
//! builder.add_step(data_loader).unwrap();
//!
//! let trainer = StepDefinition::new("trainer", StepType::Trainer, "RandomForest")
//!     .with_input("dataset")
//!     .with_output("model");
//! builder.add_step(trainer).unwrap();
//!
//! builder
//!     .add_connection(Connection::direct("data_loader", "dataset", "trainer", "dataset"))
//!     .unwrap();
//! ```
//!
//! ## Parsing DSL
//!
//! ```rust
//! use sklears_compose::workflow_language::PipelineDSL;
//!
//! let dsl_code = r#"pipeline "ml_workflow" {
//!     version "1.0.0"
//!     input data: Matrix<Float64>
//!     output predictions
//! }"#;
//!
//! let mut dsl = PipelineDSL::new();
//! let workflow = dsl.parse(dsl_code).unwrap();
//! assert_eq!(workflow.metadata.name, "ml_workflow");
//! assert_eq!(workflow.inputs[0].name, "data");
//! ```
//!
//! ## Code Generation
//!
//! ```rust
//! use sklears_compose::workflow_language::{
//!     CodeGenerationConfig, CodeGenerator, CodeLanguage, GeneratedCode, WorkflowDefinition,
//! };
//!
//! let mut generator = CodeGenerator::new(CodeGenerationConfig::default());
//! let workflow = WorkflowDefinition::default();
//! let GeneratedCode { language, .. } = generator.generate_code(&workflow).unwrap();
//! assert!(matches!(language, CodeLanguage::Rust));
//! ```

// Core module declarations
pub mod code_generation;
pub mod component_registry;
pub mod dsl_language;
pub mod visual_builder;
pub mod workflow_definitions;
pub mod workflow_execution;

// Test infrastructure (conditional compilation for tests)
#[allow(non_snake_case)]
#[cfg(test)]
pub mod workflow_tests;

// Re-export core types and traits for easy access
pub use workflow_definitions::{
    Connection, ConnectionType, DataType, ExecutionConfig, ExecutionMode, InputDefinition,
    OutputDefinition, ParameterDefinition, ParameterValue, ResourceRequirements, StepDefinition,
    StepStatus, StepType, ValidationResult, WorkflowDefinition, WorkflowMetadata, WorkflowStatus,
};

pub use visual_builder::{
    CanvasConfig, CanvasInteraction, ComponentPosition, DragState, GridConfig, Position,
    SelectionState, UndoRedoManager, ValidationState, ViewportConfig, VisualPipelineBuilder,
    WorkflowHistory, WorkflowSnapshot, ZoomConfig,
};

pub use component_registry::{
    ComponentDefinition, ComponentDiscovery, ComponentMetadata, ComponentRegistry,
    ComponentSignature, ComponentType, ComponentValidator, ComponentVersion, ParameterSchema,
    PortDefinition, RegistryError,
};

pub use workflow_execution::{
    ExecutionContext, ExecutionResult, ExecutionState, ExecutionStatistics, ExecutionTracker,
    ParallelExecutionConfig, ResourceAllocation, ResourceManager, StepExecutionResult,
    WorkflowExecutionError, WorkflowExecutor,
};

pub use code_generation::{
    CodeGenerationConfig, CodeGenerationError, CodeGenerator, CodeLanguage, CodeTemplate,
    FileFormat, GeneratedCode, GenerationStatistics, LanguageBackend, OptimizationLevel,
    TargetLanguage, TemplateContext, TemplateEngine, TemplateRegistry,
};

pub use dsl_language::{
    AstNode, AutoCompleter, DslConfig, DslError, DslLexer, DslParser, LexError, ParseError,
    ParseResult, PipelineDSL, SemanticAnalyzer, SymbolTable, SyntaxHighlighter, Token, TokenType,
    TypeChecker,
};

// Type aliases for common usage patterns
pub type WorkflowResult<T> = Result<T, WorkflowError>;

// Common error types unified interface
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Execution error: {0}")]
    Execution(#[from] WorkflowExecutionError),

    #[error("Code generation error: {0}")]
    CodeGeneration(#[from] CodeGenerationError),

    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("Registry error: {0}")]
    Registry(#[from] RegistryError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Sklears error: {0}")]
    Sklears(#[from] sklears_core::error::SklearsError),
}

// Builder patterns and factory functions for common operations
impl WorkflowDefinition {
    /// Create a new workflow builder with fluent interface
    #[must_use]
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Create a workflow from DSL string
    pub fn from_dsl(dsl_code: &str) -> Result<Self, ParseError> {
        let mut dsl = PipelineDSL::new();
        dsl.parse(dsl_code)
            .map_err(|e| ParseError::InvalidSyntax(format!("DSL parse error: {e}"), 0, 0))
    }

    /// Generate code for this workflow in the specified language
    pub fn generate_code(&self, language: TargetLanguage) -> Result<String, CodeGenerationError> {
        let config = CodeGenerationConfig {
            language,
            ..Default::default()
        };
        let mut generator = CodeGenerator::new(config);
        let generated = generator.generate_code(self)?;
        Ok(generated.source_code)
    }

    /// Execute this workflow with the given context
    #[must_use]
    pub fn execute(&self, context: ExecutionContext) -> ExecutionResult {
        let mut executor = WorkflowExecutor::new();
        // TODO: Configure executor with context
        executor.execute_workflow(self.clone()).unwrap_or_default()
    }
}

/// Builder pattern for constructing workflows programmatically
pub struct WorkflowBuilder {
    metadata: WorkflowMetadata,
    inputs: Vec<InputDefinition>,
    outputs: Vec<OutputDefinition>,
    steps: Vec<StepDefinition>,
    connections: Vec<Connection>,
    execution: ExecutionConfig,
}

impl WorkflowBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: WorkflowMetadata::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            steps: Vec::new(),
            connections: Vec::new(),
            execution: ExecutionConfig::default(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.metadata.name = name.into();
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.metadata.version = version.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.metadata.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn add_input(mut self, input: InputDefinition) -> Self {
        self.inputs.push(input);
        self
    }

    #[must_use]
    pub fn add_output(mut self, output: OutputDefinition) -> Self {
        self.outputs.push(output);
        self
    }

    #[must_use]
    pub fn add_step(mut self, step: StepDefinition) -> Self {
        self.steps.push(step);
        self
    }

    #[must_use]
    pub fn add_connection(mut self, connection: Connection) -> Self {
        self.connections.push(connection);
        self
    }

    #[must_use]
    pub fn execution_config(mut self, config: ExecutionConfig) -> Self {
        self.execution = config;
        self
    }

    #[must_use]
    pub fn build(self) -> WorkflowDefinition {
        /// WorkflowDefinition
        WorkflowDefinition {
            metadata: self.metadata,
            inputs: self.inputs,
            outputs: self.outputs,
            steps: self.steps,
            connections: self.connections,
            execution: self.execution,
        }
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Convenience functions for quick workflow operations
#[must_use]
pub fn create_workflow(name: &str) -> WorkflowBuilder {
    WorkflowBuilder::new().name(name)
}

pub fn parse_workflow(dsl_code: &str) -> Result<WorkflowDefinition, ParseError> {
    WorkflowDefinition::from_dsl(dsl_code)
}

pub fn create_visual_builder() -> Result<VisualPipelineBuilder, WorkflowError> {
    Ok(VisualPipelineBuilder::new())
}

pub fn create_code_generator() -> Result<CodeGenerator, WorkflowError> {
    Ok(CodeGenerator::new(CodeGenerationConfig::default()))
}

// Integration utilities for cross-module operations
pub struct WorkflowIntegration;

impl WorkflowIntegration {
    /// Convert a visual workflow to DSL representation
    pub fn visual_to_dsl(builder: &VisualPipelineBuilder) -> Result<String, WorkflowError> {
        let dsl = PipelineDSL::new();
        Ok(dsl.generate(&builder.workflow))
    }

    /// Load a workflow from visual builder into execution engine
    pub fn visual_to_execution(
        builder: &VisualPipelineBuilder,
        context: ExecutionContext,
    ) -> Result<WorkflowExecutor, WorkflowError> {
        let executor = WorkflowExecutor::new();
        // TODO: Configure executor with registry and context as needed
        Ok(executor)
    }

    /// Generate code from visual workflow
    pub fn visual_to_code(
        builder: &VisualPipelineBuilder,
        language: TargetLanguage,
    ) -> Result<String, WorkflowError> {
        let config = CodeGenerationConfig {
            language,
            ..Default::default()
        };
        let mut generator = CodeGenerator::new(config);
        let generated = generator.generate_code(&builder.workflow)?;
        Ok(generated.source_code)
    }

    /// Parse DSL and generate code in one operation
    pub fn dsl_to_code(dsl_code: &str, language: TargetLanguage) -> Result<String, WorkflowError> {
        let workflow = WorkflowDefinition::from_dsl(dsl_code)?;
        let config = CodeGenerationConfig {
            language,
            ..Default::default()
        };
        let mut generator = CodeGenerator::new(config);
        let generated = generator.generate_code(&workflow)?;
        Ok(generated.source_code)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_builder() {
        let workflow = WorkflowBuilder::new()
            .name("test_workflow")
            .version("1.0.0")
            .description("Test workflow for builder pattern")
            .build();

        assert_eq!(workflow.metadata.name, "test_workflow");
        assert_eq!(workflow.metadata.version, "1.0.0");
        assert_eq!(
            workflow.metadata.description,
            Some("Test workflow for builder pattern".to_string())
        );
    }

    #[test]
    fn test_convenience_functions() {
        let builder = create_workflow("test");
        let workflow = builder.build();
        assert_eq!(workflow.metadata.name, "test");
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules are properly accessible
        let _registry = ComponentRegistry::default();
        let _config = CanvasConfig::default();
        let _exec_config = ExecutionConfig::default();

        // Verify error types work together
        let _result: WorkflowResult<()> = Ok(());
    }
}
