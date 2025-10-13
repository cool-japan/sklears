//! Comprehensive Test Suite for Workflow Language
//!
//! This module provides comprehensive testing infrastructure for the workflow language
//! including unit tests, integration tests, property-based tests, and performance tests
//! for all workflow components and functionality.

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::{
        code_generation::*, component_registry::*, dsl_language::*, visual_builder::*,
        workflow_definitions::*, workflow_execution::*,
    };
    use std::collections::{BTreeMap, HashMap};
    use std::time::Duration;

    // ======================
    // Workflow Definitions Tests
    // ======================

    #[test]
    fn test_workflow_definition_creation() {
        let workflow = WorkflowDefinition::default();
        assert_eq!(workflow.metadata.name, "Untitled Workflow");
        assert_eq!(workflow.metadata.version, "1.0.0");
        assert!(workflow.steps.is_empty());
        assert!(workflow.connections.is_empty());
        assert_eq!(workflow.execution.mode, ExecutionMode::Sequential);
    }

    #[test]
    fn test_step_definition_builder() {
        let step = StepDefinition::new("test_step", StepType::Transformer, "StandardScaler")
            .with_parameter("with_mean", ParameterValue::Bool(true))
            .with_parameter("with_std", ParameterValue::Bool(true))
            .with_input("X")
            .with_output("X_scaled")
            .with_description("Standard scaling transformation");

        assert_eq!(step.id, "test_step");
        assert_eq!(step.step_type, StepType::Transformer);
        assert_eq!(step.algorithm, "StandardScaler");
        assert_eq!(step.parameters.len(), 2);
        assert_eq!(step.inputs.len(), 1);
        assert_eq!(step.outputs.len(), 1);
        assert!(step.description.is_some());

        // Verify parameters
        if let Some(ParameterValue::Bool(val)) = step.parameters.get("with_mean") {
            assert!(*val);
        } else {
            panic!("Expected with_mean parameter");
        }
    }

    #[test]
    fn test_connection_builder() {
        let connection = Connection::direct("step1", "output", "step2", "input")
            .with_transform("normalize")
            .with_type(ConnectionType::Split);

        assert_eq!(connection.from_step, "step1");
        assert_eq!(connection.from_output, "output");
        assert_eq!(connection.to_step, "step2");
        assert_eq!(connection.to_input, "input");
        assert_eq!(connection.connection_type, ConnectionType::Split);
        assert_eq!(connection.transform, Some("normalize".to_string()));
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(format!("{}", DataType::Float32), "f32");
        assert_eq!(format!("{}", DataType::Float64), "f64");
        assert_eq!(
            format!("{}", DataType::Array(Box::new(DataType::Float64))),
            "Array<f64>"
        );
        assert_eq!(
            format!("{}", DataType::Matrix(Box::new(DataType::Int32))),
            "Matrix<i32>"
        );
        assert_eq!(
            format!("{}", DataType::Custom("CustomType".to_string())),
            "CustomType"
        );
    }

    #[test]
    fn test_workflow_metadata_operations() {
        let mut metadata = WorkflowMetadata::new("Test Workflow");
        let original_time = metadata.modified_at.clone();

        // Wait a bit to ensure timestamp difference
        std::thread::sleep(Duration::from_millis(10));
        metadata.touch();

        assert_eq!(metadata.name, "Test Workflow");
        assert_eq!(metadata.version, "1.0.0");
        assert_ne!(metadata.modified_at, original_time);
    }

    #[test]
    fn test_input_output_definitions() {
        let input = InputDefinition::new("features", DataType::Matrix(Box::new(DataType::Float64)))
            .with_shape(ShapeConstraint::fixed(vec![1000, 20]))
            .with_constraints(ValueConstraints::new().with_range(-1.0, 1.0))
            .with_description("Feature matrix for training");

        let output =
            OutputDefinition::new("predictions", DataType::Array(Box::new(DataType::Float64)))
                .with_shape(ShapeConstraint::fixed(vec![1000]))
                .with_description("Model predictions");

        assert_eq!(input.name, "features");
        assert!(matches!(input.data_type, DataType::Matrix(_)));
        assert!(input.shape.is_some());
        assert!(input.constraints.is_some());
        assert!(input.description.is_some());

        assert_eq!(output.name, "predictions");
        assert!(matches!(output.data_type, DataType::Array(_)));
        assert!(output.shape.is_some());
        assert!(output.description.is_some());
    }

    #[test]
    fn test_execution_config() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Parallel,
            parallel: Some(ParallelConfig {
                num_workers: 4,
                chunk_size: Some(1000),
                load_balancing: "round_robin".to_string(),
            }),
            resources: Some(ResourceLimits {
                max_memory_mb: Some(1024),
                max_cpu_time_sec: Some(300),
                max_wall_time_sec: Some(600),
            }),
            caching: Some(CachingConfig {
                enable_step_caching: true,
                cache_directory: Some("/tmp/workflow_cache".to_string()),
                cache_ttl_sec: Some(3600),
                max_cache_size_mb: Some(512),
            }),
        };

        assert_eq!(config.mode, ExecutionMode::Parallel);
        assert!(config.parallel.is_some());
        assert!(config.resources.is_some());
        assert!(config.caching.is_some());

        let parallel_config = config.parallel.unwrap();
        assert_eq!(parallel_config.num_workers, 4);
        assert_eq!(parallel_config.chunk_size, Some(1000));
    }

    // ======================
    // Visual Builder Tests
    // ======================

    #[test]
    fn test_visual_pipeline_builder_creation() {
        let builder = VisualPipelineBuilder::new();
        assert_eq!(builder.workflow.steps.len(), 0);
        assert_eq!(builder.component_positions.len(), 0);
        assert!(builder.validation_state.is_valid);
        assert_eq!(builder.history.len(), 1); // Initial snapshot
        assert_eq!(builder.history_index, 0);
    }

    #[test]
    fn test_visual_builder_add_step() {
        let mut builder = VisualPipelineBuilder::new();
        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");

        let result = builder.add_step(step);
        assert!(result.is_ok());
        assert_eq!(builder.workflow.steps.len(), 1);
        assert_eq!(builder.component_positions.len(), 1);
        assert!(builder.component_positions.contains_key("step1"));
        assert_eq!(builder.history.len(), 2); // Initial + add step
    }

    #[test]
    fn test_visual_builder_duplicate_step_error() {
        let mut builder = VisualPipelineBuilder::new();
        let step1 = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        let step2 = StepDefinition::new("step1", StepType::Predictor, "LinearRegression");

        assert!(builder.add_step(step1).is_ok());
        assert!(builder.add_step(step2).is_err());
    }

    #[test]
    fn test_visual_builder_remove_step() {
        let mut builder = VisualPipelineBuilder::new();
        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");

        builder.add_step(step).unwrap();
        assert_eq!(builder.workflow.steps.len(), 1);

        let result = builder.remove_step("step1");
        assert!(result.is_ok());
        assert_eq!(builder.workflow.steps.len(), 0);
        assert!(!builder.component_positions.contains_key("step1"));
    }

    #[test]
    fn test_visual_builder_connections() {
        let mut builder = VisualPipelineBuilder::new();

        let step1 = StepDefinition::new("step1", StepType::Transformer, "StandardScaler")
            .with_output("X_scaled");
        let step2 =
            StepDefinition::new("step2", StepType::Predictor, "LinearRegression").with_input("X");

        builder.add_step(step1).unwrap();
        builder.add_step(step2).unwrap();

        let connection = Connection::direct("step1", "X_scaled", "step2", "X");
        let result = builder.add_connection(connection);

        assert!(result.is_ok());
        assert_eq!(builder.workflow.connections.len(), 1);
    }

    #[test]
    fn test_visual_builder_undo_redo() {
        let mut builder = VisualPipelineBuilder::new();
        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");

        builder.add_step(step).unwrap();
        assert_eq!(builder.workflow.steps.len(), 1);
        assert_eq!(builder.history_index, 1);

        // Undo
        builder.undo().unwrap();
        assert_eq!(builder.workflow.steps.len(), 0);
        assert_eq!(builder.history_index, 0);

        // Redo
        builder.redo().unwrap();
        assert_eq!(builder.workflow.steps.len(), 1);
        assert_eq!(builder.history_index, 1);

        // Test undo/redo bounds
        assert!(builder.undo().is_ok());
        assert!(builder.undo().is_err()); // Already at beginning
        builder.redo().unwrap();
        assert!(builder.redo().is_err()); // Already at end
    }

    #[test]
    fn test_visual_builder_component_positioning() {
        let mut builder = VisualPipelineBuilder::new();

        let position = Position {
            x: 100.0,
            y: 200.0,
            width: 120.0,
            height: 80.0,
        };
        let step = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");

        builder.add_step(step).unwrap();
        builder.move_component("step1", position.clone()).unwrap();

        let stored_position = builder.component_positions.get("step1").unwrap();
        assert_eq!(stored_position.x, position.x);
        assert_eq!(stored_position.y, position.y);
    }

    #[test]
    fn test_visual_builder_snap_to_grid() {
        let builder = VisualPipelineBuilder::new();
        let position = Position {
            x: 23.7,
            y: 47.3,
            width: 100.0,
            height: 80.0,
        };

        let snapped = builder.snap_to_grid(position);
        assert_eq!(snapped.x, 20.0); // Grid size is 20.0
        assert_eq!(snapped.y, 40.0);
        assert_eq!(snapped.width, 100.0);
        assert_eq!(snapped.height, 80.0);
    }

    #[test]
    fn test_visual_builder_validation() {
        let mut builder = VisualPipelineBuilder::new();

        // Add disconnected components
        let step1 = StepDefinition::new("step1", StepType::Transformer, "StandardScaler");
        let step2 = StepDefinition::new("step2", StepType::Predictor, "LinearRegression");

        builder.add_step(step1).unwrap();
        builder.add_step(step2).unwrap();

        // Validation should fail due to disconnected components
        assert!(!builder.validation_state.is_valid);
        assert!(!builder.validation_state.errors.is_empty());
    }

    // ======================
    // Component Registry Tests
    // ======================

    #[test]
    fn test_component_registry_creation() {
        let registry = ComponentRegistry::new();
        assert!(registry.has_component("StandardScaler"));
        assert!(registry.has_component("LinearRegression"));
        assert!(!registry.has_component("NonExistentComponent"));
    }

    #[test]
    fn test_component_registry_queries() {
        let registry = ComponentRegistry::new();

        // Test component existence
        let component = registry.get_component("StandardScaler");
        assert!(component.is_some());

        let comp = component.unwrap();
        assert_eq!(comp.name, "StandardScaler");
        assert_eq!(comp.component_type, StepType::Transformer);
        assert!(!comp.deprecated);

        // Test component listing
        let components = registry.list_components();
        assert!(!components.is_empty());
        assert!(components.contains(&"StandardScaler"));
        assert!(components.contains(&"LinearRegression"));
    }

    #[test]
    fn test_component_registry_search() {
        let registry = ComponentRegistry::new();

        let results = registry.search_components("scale");
        assert!(!results.is_empty());
        assert!(results.iter().any(|comp| comp.name == "StandardScaler"));

        let results = registry.search_components("regression");
        assert!(!results.is_empty());
        assert!(results.iter().any(|comp| comp.name == "LinearRegression"));

        let results = registry.search_components("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_component_registry_category_filtering() {
        let registry = ComponentRegistry::new();

        let preprocessing_components =
            registry.get_components_by_category(&ComponentCategory::Preprocessing);
        assert!(!preprocessing_components.is_empty());
        assert!(preprocessing_components
            .iter()
            .any(|comp| comp.name == "StandardScaler"));

        let training_components =
            registry.get_components_by_category(&ComponentCategory::ModelTraining);
        assert!(!training_components.is_empty());
        assert!(training_components
            .iter()
            .any(|comp| comp.name == "LinearRegression"));
    }

    #[test]
    fn test_component_parameter_validation() {
        let registry = ComponentRegistry::new();

        // Valid parameters
        let mut params = BTreeMap::new();
        params.insert("with_mean".to_string(), ParameterValue::Bool(true));
        params.insert("with_std".to_string(), ParameterValue::Bool(false));

        let result = registry.validate_parameters("StandardScaler", &params);
        assert!(result.is_ok());

        // Invalid parameter name
        params.insert("invalid_param".to_string(), ParameterValue::Bool(true));
        let result = registry.validate_parameters("StandardScaler", &params);
        assert!(result.is_err());

        // Invalid parameter type
        let mut params = BTreeMap::new();
        params.insert(
            "with_mean".to_string(),
            ParameterValue::String("invalid".to_string()),
        );
        let result = registry.validate_parameters("StandardScaler", &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_component_summaries() {
        let registry = ComponentRegistry::new();

        let summary = registry.get_component_summary("LinearRegression");
        assert!(summary.is_some());

        let sum = summary.unwrap();
        assert_eq!(sum.name, "LinearRegression");
        assert_eq!(sum.component_type, StepType::Trainer);
        assert!(!sum.deprecated);
        assert!(sum.parameter_count > 0);
        assert!(sum.input_count > 0);
        assert!(sum.output_count > 0);

        let all_summaries = registry.get_all_summaries();
        assert!(!all_summaries.is_empty());
        assert!(all_summaries.iter().any(|s| s.name == "StandardScaler"));
        assert!(all_summaries.iter().any(|s| s.name == "LinearRegression"));
    }

    // ======================
    // Workflow Execution Tests
    // ======================

    #[test]
    fn test_workflow_executor_creation() {
        let executor = WorkflowExecutor::new();
        assert_eq!(executor.get_statistics().total_executions, 0);
        assert_eq!(executor.get_statistics().successful_executions, 0);
        assert_eq!(executor.get_statistics().failed_executions, 0);
    }

    #[test]
    fn test_empty_workflow_validation() {
        let executor = WorkflowExecutor::new();
        let workflow = WorkflowDefinition::default();

        let validation = executor.validate_workflow(&workflow);
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
        assert_eq!(validation.errors[0].error_type, "EmptyWorkflow");
        assert!(validation.execution_order.is_none());
    }

    #[test]
    fn test_valid_workflow_validation() {
        let executor = WorkflowExecutor::new();
        let mut workflow = WorkflowDefinition::default();

        workflow.steps.push(StepDefinition::new(
            "step1",
            StepType::Transformer,
            "StandardScaler",
        ));

        let validation = executor.validate_workflow(&workflow);
        assert!(validation.is_valid);
        assert!(validation.errors.is_empty());
        assert!(validation.execution_order.is_some());
        assert_eq!(
            validation.execution_order.unwrap(),
            vec!["step1".to_string()]
        );
    }

    #[test]
    fn test_unknown_component_validation() {
        let executor = WorkflowExecutor::new();
        let mut workflow = WorkflowDefinition::default();

        workflow.steps.push(StepDefinition::new(
            "step1",
            StepType::Transformer,
            "UnknownComponent",
        ));

        let validation = executor.validate_workflow(&workflow);
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
        assert_eq!(validation.errors[0].error_type, "UnknownComponent");
    }

    #[test]
    fn test_execution_order_determination() {
        let executor = WorkflowExecutor::new();
        let mut workflow = WorkflowDefinition::default();

        // Add steps with dependencies
        workflow.steps.push(
            StepDefinition::new("step1", StepType::Transformer, "StandardScaler")
                .with_output("X_scaled"),
        );
        workflow.steps.push(
            StepDefinition::new("step2", StepType::Trainer, "LinearRegression").with_input("X"),
        );

        // Add connection to create dependency
        workflow
            .connections
            .push(Connection::direct("step1", "X_scaled", "step2", "X"));

        let order = executor.determine_execution_order(&workflow).unwrap();
        assert_eq!(order, vec!["step1".to_string(), "step2".to_string()]);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let executor = WorkflowExecutor::new();
        let mut workflow = WorkflowDefinition::default();

        // Add steps
        workflow.steps.push(StepDefinition::new(
            "step1",
            StepType::Transformer,
            "StandardScaler",
        ));
        workflow.steps.push(StepDefinition::new(
            "step2",
            StepType::Trainer,
            "LinearRegression",
        ));

        // Add circular connections
        workflow
            .connections
            .push(Connection::direct("step1", "output", "step2", "input"));
        workflow
            .connections
            .push(Connection::direct("step2", "output", "step1", "input"));

        let result = executor.check_circular_dependencies(&workflow);
        assert!(result.is_err());
    }

    // ======================
    // Code Generation Tests
    // ======================

    #[test]
    fn test_code_generator_creation() {
        let config = CodeGenerationConfig::default();
        let generator = CodeGenerator::new(config);
        assert!(matches!(generator.language(), CodeLanguage::Rust));
        assert!(generator.include_comments());
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
        assert!(generated.source_code.contains("use sklears_core"));
        assert!(generated.source_code.contains("pub fn test_workflow"));
        assert!(generated.source_code.contains("StandardScaler"));
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
        assert!(generated.source_code.contains("import numpy"));
        assert!(generated.source_code.contains("def test_workflow"));
        assert!(generated.source_code.contains("StandardScaler"));
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
        assert!(matches!(generated.language, CodeLanguage::Json));
        assert!(generated.source_code.contains("{"));
        assert!(generated.source_code.contains("\"metadata\""));
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

        // Test edge cases
        assert_eq!(
            generator.convert_name("Multiple   Spaces", &NamingConvention::SnakeCase),
            "multiple_spaces"
        );
        assert_eq!(
            generator.convert_name("kebab-case-name", &NamingConvention::CamelCase),
            "kebabCaseName"
        );
    }

    // ======================
    // DSL Language Tests
    // ======================

    #[test]
    fn test_dsl_lexer_basic_tokens() {
        let mut lexer = DslLexer::new();
        let tokens = lexer.tokenize("pipeline { }").unwrap();

        assert_eq!(tokens[0], Token::Pipeline);
        assert_eq!(tokens[1], Token::LeftBrace);
        assert_eq!(tokens[2], Token::RightBrace);
        assert_eq!(tokens[3], Token::Eof);
    }

    #[test]
    fn test_dsl_lexer_string_literal() {
        let mut lexer = DslLexer::new();
        let tokens = lexer.tokenize("\"hello world\"").unwrap();

        if let Token::StringLiteral(s) = &tokens[0] {
            assert_eq!(s, "hello world");
        } else {
            panic!("Expected string literal, got {:?}", tokens[0]);
        }
    }

    #[test]
    fn test_dsl_lexer_number_literal() {
        let mut lexer = DslLexer::new();

        // Test integer
        let tokens = lexer.tokenize("42").unwrap();
        if let Token::NumberLiteral(n) = &tokens[0] {
            assert_eq!(*n, 42.0);
        } else {
            panic!("Expected number literal");
        }

        // Test float
        let tokens = lexer.tokenize("42.5").unwrap();
        if let Token::NumberLiteral(n) = &tokens[0] {
            assert_eq!(*n, 42.5);
        } else {
            panic!("Expected number literal");
        }
    }

    #[test]
    fn test_dsl_lexer_boolean_literals() {
        let mut lexer = DslLexer::new();

        let tokens = lexer.tokenize("true false").unwrap();
        assert_eq!(tokens[0], Token::BooleanLiteral(true));
        assert_eq!(tokens[1], Token::BooleanLiteral(false));
    }

    #[test]
    fn test_dsl_lexer_keywords() {
        let mut lexer = DslLexer::new();
        let tokens = lexer.tokenize("pipeline step flow execute").unwrap();

        assert_eq!(tokens[0], Token::Pipeline);
        assert_eq!(tokens[1], Token::Step);
        assert_eq!(tokens[2], Token::Flow);
        assert_eq!(tokens[3], Token::Execute);
    }

    #[test]
    fn test_dsl_parser_simple_pipeline() {
        let mut dsl = PipelineDSL::new();
        let input = r#"
            pipeline "Test Pipeline" {
                version "1.0.0"
                author "Test Author"
                step scaler: StandardScaler {
                    with_mean: true,
                    with_std: false
                }
            }
        "#;

        let workflow = dsl.parse(input).unwrap();
        assert_eq!(workflow.metadata.name, "Test Pipeline");
        assert_eq!(workflow.metadata.version, "1.0.0");
        assert_eq!(workflow.metadata.author, Some("Test Author".to_string()));
        assert_eq!(workflow.steps.len(), 1);
        assert_eq!(workflow.steps[0].algorithm, "StandardScaler");
        assert_eq!(workflow.steps[0].parameters.len(), 2);
    }

    #[test]
    fn test_dsl_parser_with_connections() {
        let mut dsl = PipelineDSL::new();
        let input = r#"
            pipeline "Connected Pipeline" {
                version "1.0.0"
                step scaler: StandardScaler { }
                step model: LinearRegression { }
                flow scaler.output -> model.input
            }
        "#;

        let workflow = dsl.parse(input).unwrap();
        assert_eq!(workflow.steps.len(), 2);
        assert_eq!(workflow.connections.len(), 1);

        let connection = &workflow.connections[0];
        assert_eq!(connection.from_step, "scaler");
        assert_eq!(connection.from_output, "output");
        assert_eq!(connection.to_step, "model");
        assert_eq!(connection.to_input, "input");
    }

    #[test]
    fn test_dsl_generation() {
        let mut workflow = WorkflowDefinition::default();
        workflow.metadata.name = "Generated Workflow".to_string();
        workflow.metadata.version = "2.0.0".to_string();
        workflow.metadata.author = Some("Generator".to_string());

        let step = StepDefinition::new("scaler", StepType::Transformer, "StandardScaler")
            .with_parameter("with_mean", ParameterValue::Bool(true))
            .with_parameter("threshold", ParameterValue::Float(0.5));
        workflow.steps.push(step);

        let dsl = PipelineDSL::new();
        let generated = dsl.generate(&workflow);

        assert!(generated.contains("pipeline \"Generated Workflow\""));
        assert!(generated.contains("version \"2.0.0\""));
        assert!(generated.contains("author \"Generator\""));
        assert!(generated.contains("step scaler: StandardScaler"));
        assert!(generated.contains("with_mean: true"));
        assert!(generated.contains("threshold: 0.5"));
    }

    #[test]
    fn test_dsl_syntax_validation() {
        let mut dsl = PipelineDSL::new();

        // Valid syntax
        let valid_input = r#"
            pipeline "Valid Pipeline" {
                version "1.0.0"
                step test: StandardScaler { }
            }
        "#;
        let errors = dsl.validate_syntax(valid_input).unwrap();
        assert!(errors.is_empty());

        // Invalid syntax - missing closing brace
        let invalid_input = r#"
            pipeline "Invalid Pipeline" {
                version "1.0.0"
                step test: StandardScaler {
        "#;
        let errors = dsl.validate_syntax(invalid_input).unwrap();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_dsl_error_reporting() {
        let error = DslError {
            message: "Test error".to_string(),
            line: 5,
            column: 10,
        };

        let error_string = format!("{}", error);
        assert!(error_string.contains("Test error"));
        assert!(error_string.contains("line 5"));
        assert!(error_string.contains("column 10"));
    }

    // ======================
    // Integration Tests
    // ======================

    #[test]
    fn test_end_to_end_workflow_processing() {
        // Create workflow via visual builder
        let mut builder = VisualPipelineBuilder::new();

        let step1 = StepDefinition::new("preprocessor", StepType::Transformer, "StandardScaler")
            .with_parameter("with_mean", ParameterValue::Bool(true))
            .with_output("X_scaled");

        let step2 = StepDefinition::new("model", StepType::Trainer, "LinearRegression")
            .with_parameter("fit_intercept", ParameterValue::Bool(true))
            .with_input("X");

        builder.add_step(step1).unwrap();
        builder.add_step(step2).unwrap();

        let connection = Connection::direct("preprocessor", "X_scaled", "model", "X");
        builder.add_connection(connection).unwrap();

        let workflow = builder.get_workflow().clone();

        // Validate with executor
        let executor = WorkflowExecutor::new();
        let validation = executor.validate_workflow(&workflow);
        assert!(validation.is_valid);

        // Generate code
        let mut generator = CodeGenerator::new(CodeGenerationConfig::default());
        let generated = generator.generate_code(&workflow).unwrap();
        assert!(!generated.source_code.is_empty());

        // Generate DSL
        let dsl = PipelineDSL::new();
        let dsl_text = dsl.generate(&workflow);
        assert!(dsl_text.contains("preprocessor"));
        assert!(dsl_text.contains("model"));
        assert!(dsl_text.contains("flow"));
    }

    #[test]
    fn test_dsl_round_trip() {
        // Create workflow via DSL
        let mut dsl = PipelineDSL::new();
        let original_dsl = r#"
            pipeline "Round Trip Test" {
                version "1.0.0"
                author "Test Suite"

                step preprocessor: StandardScaler {
                    with_mean: true,
                    with_std: true
                }

                step model: LinearRegression {
                    fit_intercept: true
                }

                flow preprocessor.X_scaled -> model.X

                execute {
                    mode: parallel,
                    workers: 4
                }
            }
        "#;

        let workflow = dsl.parse(original_dsl).unwrap();

        // Generate DSL back
        let generated_dsl = dsl.generate(&workflow);

        // Parse generated DSL
        let round_trip_workflow = dsl.parse(&generated_dsl).unwrap();

        // Verify round trip
        assert_eq!(workflow.metadata.name, round_trip_workflow.metadata.name);
        assert_eq!(
            workflow.metadata.version,
            round_trip_workflow.metadata.version
        );
        assert_eq!(workflow.steps.len(), round_trip_workflow.steps.len());
        assert_eq!(
            workflow.connections.len(),
            round_trip_workflow.connections.len()
        );
    }

    #[test]
    fn test_cross_module_compatibility() {
        // Test that all modules work together correctly
        let registry = ComponentRegistry::new();
        let mut executor = WorkflowExecutor::with_registry(registry);
        let mut builder = VisualPipelineBuilder::new();
        let dsl = PipelineDSL::new();

        // Create workflow through visual builder
        let step = StepDefinition::new("test_step", StepType::Transformer, "StandardScaler");
        builder.add_step(step).unwrap();

        let workflow = builder.get_workflow().clone();

        // Validate with executor
        let validation = executor.validate_workflow(&workflow);
        assert!(validation.is_valid);

        // Generate DSL
        let dsl_text = dsl.generate(&workflow);
        assert!(dsl_text.contains("test_step"));
        assert!(dsl_text.contains("StandardScaler"));

        // Parse DSL back
        let mut parsed_dsl = PipelineDSL::new();
        let parsed_workflow = parsed_dsl.parse(&dsl_text).unwrap();
        assert_eq!(parsed_workflow.metadata.name, workflow.metadata.name);
    }

    // ======================
    // Performance Tests
    // ======================

    #[test]
    fn test_large_workflow_performance() {
        let mut builder = VisualPipelineBuilder::new();
        let start_time = std::time::Instant::now();

        // Create a large workflow with many steps
        for i in 0..100 {
            let step = StepDefinition::new(
                &format!("step_{}", i),
                StepType::Transformer,
                "StandardScaler",
            );
            builder.add_step(step).unwrap();
        }

        let creation_time = start_time.elapsed();
        assert!(creation_time < Duration::from_millis(1000)); // Should be fast

        // Validate performance
        let executor = WorkflowExecutor::new();
        let validation_start = std::time::Instant::now();
        let validation = executor.validate_workflow(builder.get_workflow());
        let validation_time = validation_start.elapsed();

        assert!(validation.is_valid);
        assert!(validation_time < Duration::from_millis(100)); // Should be fast
    }

    #[test]
    fn test_dsl_parsing_performance() {
        // Generate large DSL text
        let mut large_dsl = String::from(r#"pipeline "Performance Test" { version "1.0.0""#);

        for i in 0..50 {
            large_dsl.push_str(&format!(
                r#"
                step step_{}: StandardScaler {{
                    with_mean: true,
                    with_std: false
                }}"#,
                i
            ));
        }

        large_dsl.push_str("}");

        let mut dsl = PipelineDSL::new();
        let start_time = std::time::Instant::now();
        let workflow = dsl.parse(&large_dsl).unwrap();
        let parse_time = start_time.elapsed();

        assert_eq!(workflow.steps.len(), 50);
        assert!(parse_time < Duration::from_millis(100)); // Should be fast
    }
}
