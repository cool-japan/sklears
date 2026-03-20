//! Core Workflow Definitions and Data Structures
//!
//! This module provides the fundamental data structures for workflow definitions,
//! including workflow metadata, input/output schemas, step definitions, data types,
//! constraints, and execution configurations for machine learning pipelines.

use chrono;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Workflow description language for pipeline definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Workflow metadata
    pub metadata: WorkflowMetadata,
    /// Input schema definition
    pub inputs: Vec<InputDefinition>,
    /// Output schema definition
    pub outputs: Vec<OutputDefinition>,
    /// Pipeline steps
    pub steps: Vec<StepDefinition>,
    /// Data flow connections
    pub connections: Vec<Connection>,
    /// Execution configuration
    pub execution: ExecutionConfig,
}

/// Workflow metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMetadata {
    /// Workflow name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: String,
    /// Last modified timestamp
    pub modified_at: String,
}

/// Input definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputDefinition {
    /// Input name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Shape constraints
    pub shape: Option<ShapeConstraint>,
    /// Value constraints
    pub constraints: Option<ValueConstraints>,
    /// Description
    pub description: Option<String>,
}

/// Output definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputDefinition {
    /// Output name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Shape
    pub shape: Option<ShapeConstraint>,
    /// Description
    pub description: Option<String>,
}

/// Step definition in the workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDefinition {
    /// Step identifier
    pub id: String,
    /// Step type
    pub step_type: StepType,
    /// Algorithm/component name
    pub algorithm: String,
    /// Parameters
    pub parameters: BTreeMap<String, ParameterValue>,
    /// Input mappings
    pub inputs: Vec<String>,
    /// Output mappings
    pub outputs: Vec<String>,
    /// Conditional execution
    pub condition: Option<ExecutionCondition>,
    /// Description
    pub description: Option<String>,
}

/// Data type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// Boolean
    Boolean,
    /// String
    String,
    /// Array of specific type
    Array(Box<DataType>),
    /// Matrix of specific type
    Matrix(Box<DataType>),
    /// Custom type
    Custom(String),
}

/// Shape constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeConstraint {
    /// Dimensions
    pub dimensions: Vec<DimensionConstraint>,
    /// Optional shape validation
    pub validation: Option<String>,
}

/// Dimension constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionConstraint {
    /// Fixed size
    Fixed(usize),
    /// Range of sizes
    Range { min: usize, max: Option<usize> },
    /// Any size
    Any,
    /// Size matches another dimension
    MatchesDimension { step_id: String, dimension: usize },
    /// Dynamic size determined at runtime
    Dynamic(String),
}

/// Value constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueConstraints {
    /// Minimum value
    pub min: Option<f64>,
    /// Maximum value
    pub max: Option<f64>,
    /// Allowed values
    pub allowed_values: Option<Vec<String>>,
    /// Regular expression pattern for strings
    pub pattern: Option<String>,
    /// Custom validation function name
    pub custom_validator: Option<String>,
}

/// Step type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepType {
    /// Data input step
    Input,
    /// Data output step
    Output,
    /// Data preprocessing step
    Preprocessor,
    /// Feature transformation step
    Transformer,
    /// Model training step
    Trainer,
    /// Model prediction step
    Predictor,
    /// Evaluation/metrics step
    Evaluator,
    /// Custom processing step
    Custom(String),
}

/// Parameter value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    /// Float value
    Float(f64),
    /// Integer value
    Int(i64),
    /// Boolean value
    Bool(bool),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<ParameterValue>),
}

/// Connection between steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source step identifier
    pub from_step: String,
    /// Source output name
    pub from_output: String,
    /// Target step identifier
    pub to_step: String,
    /// Target input name
    pub to_input: String,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Optional transformation applied to data
    pub transform: Option<String>,
}

/// Connection type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    /// Direct data flow
    Direct,
    /// Data splitting/broadcasting
    Split,
    /// Data aggregation/joining
    Join,
    /// Conditional connection
    Conditional,
}

/// Execution condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCondition {
    /// Condition expression
    pub expression: String,
    /// Variables referenced in the condition
    pub variables: Vec<String>,
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Execution mode
    pub mode: ExecutionMode,
    /// Parallel execution configuration
    pub parallel: Option<ParallelConfig>,
    /// Resource limits
    pub resources: Option<ResourceLimits>,
    /// Caching configuration
    pub caching: Option<CachingConfig>,
}

/// Execution mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionMode {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Distributed execution
    Distributed,
    /// GPU-accelerated execution
    GPU,
    /// Adaptive execution (choose based on data size)
    Adaptive,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Chunk size for data parallelism
    pub chunk_size: Option<usize>,
    /// Load balancing strategy
    pub load_balancing: String,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
    /// Maximum CPU time in seconds
    pub max_cpu_time_sec: Option<usize>,
    /// Maximum wall clock time in seconds
    pub max_wall_time_sec: Option<usize>,
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable step-level caching
    pub enable_step_caching: bool,
    /// Cache directory
    pub cache_directory: Option<String>,
    /// Cache expiration time in seconds
    pub cache_ttl_sec: Option<usize>,
    /// Maximum cache size in MB
    pub max_cache_size_mb: Option<usize>,
}

impl Default for WorkflowDefinition {
    fn default() -> Self {
        Self {
            metadata: WorkflowMetadata {
                name: "Untitled Workflow".to_string(),
                version: "1.0.0".to_string(),
                description: None,
                author: None,
                tags: Vec::new(),
                created_at: chrono::Utc::now().to_rfc3339(),
                modified_at: chrono::Utc::now().to_rfc3339(),
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            steps: Vec::new(),
            connections: Vec::new(),
            execution: ExecutionConfig {
                mode: ExecutionMode::Sequential,
                parallel: None,
                resources: None,
                caching: None,
            },
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Float32 => write!(f, "f32"),
            DataType::Float64 => write!(f, "f64"),
            DataType::Int32 => write!(f, "i32"),
            DataType::Int64 => write!(f, "i64"),
            DataType::Boolean => write!(f, "bool"),
            DataType::String => write!(f, "String"),
            DataType::Array(inner) => write!(f, "Array<{inner}>"),
            DataType::Matrix(inner) => write!(f, "Matrix<{inner}>"),
            DataType::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl WorkflowMetadata {
    /// Create new workflow metadata with defaults
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: None,
            author: None,
            tags: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            modified_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Update the modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = chrono::Utc::now().to_rfc3339();
    }
}

impl StepDefinition {
    /// Create a new step definition
    #[must_use]
    pub fn new(id: &str, step_type: StepType, algorithm: &str) -> Self {
        Self {
            id: id.to_string(),
            step_type,
            algorithm: algorithm.to_string(),
            parameters: BTreeMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            condition: None,
            description: None,
        }
    }

    /// Add a parameter to the step
    #[must_use]
    pub fn with_parameter(mut self, name: &str, value: ParameterValue) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Add an input mapping
    #[must_use]
    pub fn with_input(mut self, input: &str) -> Self {
        self.inputs.push(input.to_string());
        self
    }

    /// Add an output mapping
    #[must_use]
    pub fn with_output(mut self, output: &str) -> Self {
        self.outputs.push(output.to_string());
        self
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }
}

impl Connection {
    /// Create a new direct connection between steps
    #[must_use]
    pub fn direct(from_step: &str, from_output: &str, to_step: &str, to_input: &str) -> Self {
        Self {
            from_step: from_step.to_string(),
            from_output: from_output.to_string(),
            to_step: to_step.to_string(),
            to_input: to_input.to_string(),
            connection_type: ConnectionType::Direct,
            transform: None,
        }
    }

    /// Create a new connection with transformation
    #[must_use]
    pub fn with_transform(mut self, transform: &str) -> Self {
        self.transform = Some(transform.to_string());
        self
    }

    /// Set connection type
    #[must_use]
    pub fn with_type(mut self, connection_type: ConnectionType) -> Self {
        self.connection_type = connection_type;
        self
    }
}

impl InputDefinition {
    /// Create a new input definition
    #[must_use]
    pub fn new(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            shape: None,
            constraints: None,
            description: None,
        }
    }

    /// Add shape constraints
    #[must_use]
    pub fn with_shape(mut self, shape: ShapeConstraint) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Add value constraints
    #[must_use]
    pub fn with_constraints(mut self, constraints: ValueConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Add description
    #[must_use]
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }
}

impl OutputDefinition {
    /// Create a new output definition
    #[must_use]
    pub fn new(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            shape: None,
            description: None,
        }
    }

    /// Add shape constraints
    #[must_use]
    pub fn with_shape(mut self, shape: ShapeConstraint) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Add description
    #[must_use]
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }
}

impl ShapeConstraint {
    /// Create a new shape constraint with fixed dimensions
    pub fn fixed(dimensions: Vec<usize>) -> Self {
        Self {
            dimensions: dimensions
                .into_iter()
                .map(DimensionConstraint::Fixed)
                .collect(),
            validation: None,
        }
    }

    /// Create a shape constraint with dynamic dimensions
    #[must_use]
    pub fn dynamic(constraints: Vec<DimensionConstraint>) -> Self {
        Self {
            dimensions: constraints,
            validation: None,
        }
    }

    /// Add validation expression
    #[must_use]
    pub fn with_validation(mut self, validation: &str) -> Self {
        self.validation = Some(validation.to_string());
        self
    }
}

impl ValueConstraints {
    /// Create new value constraints
    #[must_use]
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            allowed_values: None,
            pattern: None,
            custom_validator: None,
        }
    }

    /// Set minimum value
    #[must_use]
    pub fn with_min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum value
    #[must_use]
    pub fn with_max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set range
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Set allowed values
    #[must_use]
    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = Some(values);
        self
    }

    /// Set pattern for string validation
    #[must_use]
    pub fn with_pattern(mut self, pattern: &str) -> Self {
        self.pattern = Some(pattern.to_string());
        self
    }
}

impl Default for ValueConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter definition for step configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Data type of the parameter
    pub data_type: DataType,
    /// Default value if not specified
    pub default_value: Option<ParameterValue>,
    /// Whether the parameter is required
    pub required: bool,
    /// Parameter description
    pub description: Option<String>,
    /// Value constraints
    pub constraints: Option<ValueConstraints>,
}

/// Resource requirements for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: Option<usize>,
    /// Memory required in MB
    pub memory_mb: Option<usize>,
    /// GPU memory required in MB
    pub gpu_memory_mb: Option<usize>,
    /// Disk space required in MB
    pub disk_space_mb: Option<usize>,
    /// Network bandwidth required in Mbps
    pub network_bandwidth_mbps: Option<f64>,
}

/// Step execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepStatus {
    /// Step is pending execution
    Pending,
    /// Step is currently running
    Running,
    /// Step completed successfully
    Completed,
    /// Step failed with error
    Failed,
    /// Step was skipped
    Skipped,
    /// Step was cancelled
    Cancelled,
}

/// Validation result for workflow components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors found
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Step or component ID that was validated
    pub component_id: Option<String>,
}

/// Overall workflow status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkflowStatus {
    /// Workflow is being initialized
    Initializing,
    /// Workflow is ready to run
    Ready,
    /// Workflow is currently executing
    Running,
    /// Workflow paused execution
    Paused,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed during execution
    Failed,
    /// Workflow was cancelled
    Cancelled,
}

impl Default for WorkflowMetadata {
    fn default() -> Self {
        Self {
            name: "Untitled Workflow".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            author: None,
            tags: vec![],
            created_at: chrono::Utc::now().to_rfc3339(),
            modified_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Sequential,
            parallel: None,
            resources: None,
            caching: None,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_definition_default() {
        let workflow = WorkflowDefinition::default();
        assert_eq!(workflow.metadata.name, "Untitled Workflow");
        assert_eq!(workflow.metadata.version, "1.0.0");
        assert_eq!(workflow.steps.len(), 0);
        assert_eq!(workflow.execution.mode, ExecutionMode::Sequential);
    }

    #[test]
    fn test_step_definition_builder() {
        let step = StepDefinition::new("test_step", StepType::Transformer, "StandardScaler")
            .with_parameter("with_mean", ParameterValue::Bool(true))
            .with_parameter("with_std", ParameterValue::Bool(true))
            .with_input("X")
            .with_output("X_scaled")
            .with_description("Standard scaling step");

        assert_eq!(step.id, "test_step");
        assert_eq!(step.algorithm, "StandardScaler");
        assert_eq!(step.parameters.len(), 2);
        assert_eq!(step.inputs.len(), 1);
        assert_eq!(step.outputs.len(), 1);
        assert!(step.description.is_some());
    }

    #[test]
    fn test_connection_builder() {
        let connection = Connection::direct("step1", "output", "step2", "input")
            .with_transform("normalize")
            .with_type(ConnectionType::Split);

        assert_eq!(connection.from_step, "step1");
        assert_eq!(connection.to_step, "step2");
        assert_eq!(connection.connection_type, ConnectionType::Split);
        assert!(connection.transform.is_some());
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(format!("{}", DataType::Float32), "f32");
        assert_eq!(
            format!("{}", DataType::Array(Box::new(DataType::Float64))),
            "Array<f64>"
        );
        assert_eq!(
            format!("{}", DataType::Matrix(Box::new(DataType::Int32))),
            "Matrix<i32>"
        );
    }

    #[test]
    fn test_input_output_definitions() {
        let input = InputDefinition::new("features", DataType::Matrix(Box::new(DataType::Float64)))
            .with_shape(ShapeConstraint::fixed(vec![1000, 20]))
            .with_constraints(ValueConstraints::new().with_range(-1.0, 1.0))
            .with_description("Feature matrix");

        let output =
            OutputDefinition::new("predictions", DataType::Array(Box::new(DataType::Float64)))
                .with_shape(ShapeConstraint::fixed(vec![1000]))
                .with_description("Model predictions");

        assert_eq!(input.name, "features");
        assert!(input.shape.is_some());
        assert!(input.constraints.is_some());

        assert_eq!(output.name, "predictions");
        assert!(output.shape.is_some());
    }

    #[test]
    fn test_workflow_metadata() {
        let mut metadata = WorkflowMetadata::new("Test Workflow");
        let original_time = metadata.modified_at.clone();

        std::thread::sleep(std::time::Duration::from_millis(10));
        metadata.touch();

        assert_eq!(metadata.name, "Test Workflow");
        assert_ne!(metadata.modified_at, original_time);
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
    }
}
