//! Workflow Execution and Validation
//!
//! This module provides workflow execution capabilities including validation,
//! runtime execution, dependency resolution, and execution monitoring for
//! machine learning pipeline workflows.

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use super::component_registry::ComponentRegistry;
use super::workflow_definitions::{Connection, ExecutionMode, StepDefinition, WorkflowDefinition};

/// Workflow execution engine
#[derive(Debug)]
pub struct WorkflowExecutor {
    /// Component registry
    registry: ComponentRegistry,
    /// Execution context
    context: ExecutionContext,
    /// Execution statistics
    stats: ExecutionStatistics,
}

/// Execution context for workflow runs
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Execution ID
    pub execution_id: String,
    /// Workflow being executed
    pub workflow: WorkflowDefinition,
    /// Data flow between steps
    pub data_flow: HashMap<String, StepData>,
    /// Execution start time
    pub start_time: Instant,
    /// Current execution step
    pub current_step: Option<String>,
    /// Execution mode
    pub execution_mode: ExecutionMode,
}

/// Data passed between workflow steps
#[derive(Debug, Clone)]
pub struct StepData {
    /// Step identifier
    pub step_id: String,
    /// Output port name
    pub port_name: String,
    /// Data matrices
    pub matrices: HashMap<String, Array2<Float>>,
    /// Data arrays
    pub arrays: HashMap<String, Array1<Float>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when data was produced
    pub timestamp: Instant,
}

/// Execution result for a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Execution ID
    pub execution_id: String,
    /// Whether execution succeeded
    pub success: bool,
    /// Execution duration
    pub duration: Duration,
    /// Step results
    pub step_results: Vec<StepExecutionResult>,
    /// Final outputs
    pub outputs: HashMap<String, String>,
    /// Error message if execution failed
    pub error: Option<String>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self {
            execution_id: "unknown".to_string(),
            success: false,
            duration: Duration::from_secs(0),
            step_results: Vec::new(),
            outputs: HashMap::new(),
            error: Some("Execution failed".to_string()),
            performance: PerformanceMetrics::default(),
        }
    }
}

/// Result of executing a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecutionResult {
    /// Step ID
    pub step_id: String,
    /// Step name/algorithm
    pub algorithm: String,
    /// Whether step succeeded
    pub success: bool,
    /// Step execution duration
    pub duration: Duration,
    /// Memory usage during step
    pub memory_usage: u64,
    /// Output data sizes
    pub output_sizes: HashMap<String, usize>,
    /// Error message if step failed
    pub error: Option<String>,
}

/// Performance metrics for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total execution time
    pub total_time: Duration,
    /// Peak memory usage
    pub peak_memory: u64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Parallelism efficiency
    pub parallelism_efficiency: f64,
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStatistics {
    /// Total executions
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Step execution counts
    pub step_execution_counts: HashMap<String, u64>,
}

/// Workflow validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether workflow is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Execution order
    pub execution_order: Option<Vec<String>>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Related step ID
    pub step_id: Option<String>,
    /// Related connection
    pub connection: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: String,
    /// Warning message
    pub message: String,
    /// Related step ID
    pub step_id: Option<String>,
}

impl WorkflowExecutor {
    /// Create a new workflow executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: ComponentRegistry::new(),
            context: ExecutionContext::new(),
            stats: ExecutionStatistics::new(),
        }
    }

    /// Create executor with custom registry
    #[must_use]
    pub fn with_registry(registry: ComponentRegistry) -> Self {
        Self {
            registry,
            context: ExecutionContext::new(),
            stats: ExecutionStatistics::new(),
        }
    }

    /// Validate a workflow
    #[must_use]
    pub fn validate_workflow(&self, workflow: &WorkflowDefinition) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for empty workflow
        if workflow.steps.is_empty() {
            errors.push(ValidationError {
                error_type: "EmptyWorkflow".to_string(),
                message: "Workflow has no steps".to_string(),
                step_id: None,
                connection: None,
            });
            return ValidationResult {
                is_valid: false,
                errors,
                warnings,
                execution_order: None,
            };
        }

        // Validate steps
        for step in &workflow.steps {
            self.validate_step(step, &mut errors, &mut warnings);
        }

        // Validate connections
        for connection in &workflow.connections {
            self.validate_connection(connection, workflow, &mut errors, &mut warnings);
        }

        // Check for circular dependencies
        if let Err(cycle_error) = self.check_circular_dependencies(workflow) {
            errors.push(ValidationError {
                error_type: "CircularDependency".to_string(),
                message: cycle_error,
                step_id: None,
                connection: None,
            });
        }

        // Determine execution order
        let execution_order = if errors.is_empty() {
            self.determine_execution_order(workflow).ok()
        } else {
            None
        };

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            execution_order,
        }
    }

    /// Validate a single step
    fn validate_step(
        &self,
        step: &StepDefinition,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        // Check if component exists
        if !self.registry.has_component(&step.algorithm) {
            errors.push(ValidationError {
                error_type: "UnknownComponent".to_string(),
                message: format!("Component '{}' not found in registry", step.algorithm),
                step_id: Some(step.id.clone()),
                connection: None,
            });
            return;
        }

        // Validate parameters
        if let Err(param_error) = self
            .registry
            .validate_parameters(&step.algorithm, &step.parameters)
        {
            errors.push(ValidationError {
                error_type: "InvalidParameters".to_string(),
                message: param_error.to_string(),
                step_id: Some(step.id.clone()),
                connection: None,
            });
        }

        // Check for deprecated components
        if let Some(component) = self.registry.get_component(&step.algorithm) {
            if component.deprecated {
                warnings.push(ValidationWarning {
                    warning_type: "DeprecatedComponent".to_string(),
                    message: format!("Component '{}' is deprecated", step.algorithm),
                    step_id: Some(step.id.clone()),
                });
            }
        }
    }

    /// Validate a connection
    fn validate_connection(
        &self,
        connection: &Connection,
        workflow: &WorkflowDefinition,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationWarning>,
    ) {
        // Check if source step exists
        let source_step = workflow.steps.iter().find(|s| s.id == connection.from_step);
        if source_step.is_none() {
            errors.push(ValidationError {
                error_type: "InvalidConnection".to_string(),
                message: format!("Source step '{}' not found", connection.from_step),
                step_id: None,
                connection: Some(format!(
                    "{}:{} -> {}:{}",
                    connection.from_step,
                    connection.from_output,
                    connection.to_step,
                    connection.to_input
                )),
            });
            return;
        }

        // Check if target step exists
        let target_step = workflow.steps.iter().find(|s| s.id == connection.to_step);
        if target_step.is_none() {
            errors.push(ValidationError {
                error_type: "InvalidConnection".to_string(),
                message: format!("Target step '{}' not found", connection.to_step),
                step_id: None,
                connection: Some(format!(
                    "{}:{} -> {}:{}",
                    connection.from_step,
                    connection.from_output,
                    connection.to_step,
                    connection.to_input
                )),
            });
            return;
        }

        // Validate that source step has the output port
        let source = source_step.unwrap();
        if !source.outputs.contains(&connection.from_output) {
            errors.push(ValidationError {
                error_type: "InvalidConnection".to_string(),
                message: format!(
                    "Step '{}' does not have output '{}'",
                    connection.from_step, connection.from_output
                ),
                step_id: Some(source.id.clone()),
                connection: Some(format!(
                    "{}:{} -> {}:{}",
                    connection.from_step,
                    connection.from_output,
                    connection.to_step,
                    connection.to_input
                )),
            });
        }

        // Validate that target step has the input port
        let target = target_step.unwrap();
        if !target.inputs.contains(&connection.to_input) {
            errors.push(ValidationError {
                error_type: "InvalidConnection".to_string(),
                message: format!(
                    "Step '{}' does not have input '{}'",
                    connection.to_step, connection.to_input
                ),
                step_id: Some(target.id.clone()),
                connection: Some(format!(
                    "{}:{} -> {}:{}",
                    connection.from_step,
                    connection.from_output,
                    connection.to_step,
                    connection.to_input
                )),
            });
        }
    }

    /// Check for circular dependencies
    pub fn check_circular_dependencies(&self, workflow: &WorkflowDefinition) -> Result<(), String> {
        let mut graph = HashMap::new();

        // Build dependency graph
        for step in &workflow.steps {
            graph.insert(step.id.clone(), HashSet::new());
        }

        for connection in &workflow.connections {
            if let Some(dependencies) = graph.get_mut(&connection.to_step) {
                dependencies.insert(connection.from_step.clone());
            }
        }

        // Check for cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for step_id in graph.keys() {
            if !visited.contains(step_id)
                && self.has_cycle_dfs(step_id, &graph, &mut visited, &mut rec_stack)
            {
                return Err(format!(
                    "Circular dependency detected involving step '{step_id}'"
                ));
            }
        }

        Ok(())
    }

    /// DFS helper for cycle detection
    fn has_cycle_dfs(
        &self,
        step_id: &str,
        graph: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(step_id.to_string());
        rec_stack.insert(step_id.to_string());

        if let Some(dependencies) = graph.get(step_id) {
            for dep in dependencies {
                if !visited.contains(dep) {
                    if self.has_cycle_dfs(dep, graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }

        rec_stack.remove(step_id);
        false
    }

    /// Determine execution order using topological sort
    pub fn determine_execution_order(
        &self,
        workflow: &WorkflowDefinition,
    ) -> SklResult<Vec<String>> {
        let mut in_degree = HashMap::new();
        let mut adj_list = HashMap::new();

        // Initialize
        for step in &workflow.steps {
            in_degree.insert(step.id.clone(), 0);
            adj_list.insert(step.id.clone(), Vec::new());
        }

        // Build graph and calculate in-degrees
        for connection in &workflow.connections {
            *in_degree.get_mut(&connection.to_step).unwrap() += 1;
            adj_list
                .get_mut(&connection.from_step)
                .unwrap()
                .push(connection.to_step.clone());
        }

        // Topological sort using Kahn's algorithm
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Find all nodes with in-degree 0
        for (step_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(step_id.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            result.push(current.clone());

            // Reduce in-degree of adjacent nodes
            for neighbor in &adj_list[&current] {
                *in_degree.get_mut(neighbor).unwrap() -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor.clone());
                }
            }
        }

        if result.len() != workflow.steps.len() {
            return Err(SklearsError::InvalidInput(
                "Circular dependency detected".to_string(),
            ));
        }

        Ok(result)
    }

    /// Execute a workflow
    pub fn execute_workflow(&mut self, workflow: WorkflowDefinition) -> SklResult<ExecutionResult> {
        let execution_start = Instant::now();
        let execution_id = uuid::Uuid::new_v4().to_string();

        // Validate workflow first
        let validation = self.validate_workflow(&workflow);
        if !validation.is_valid {
            return Ok(ExecutionResult {
                execution_id,
                success: false,
                duration: execution_start.elapsed(),
                step_results: Vec::new(),
                outputs: HashMap::new(),
                error: Some(format!(
                    "Workflow validation failed: {:?}",
                    validation.errors
                )),
                performance: PerformanceMetrics::default(),
            });
        }

        // Setup execution context
        self.context = ExecutionContext {
            execution_id: execution_id.clone(),
            workflow: workflow.clone(),
            data_flow: HashMap::new(),
            start_time: execution_start,
            current_step: None,
            execution_mode: workflow.execution.mode.clone(),
        };

        let execution_order = validation.execution_order.unwrap();
        let mut step_results = Vec::new();
        let mut success = true;
        let mut error_message = None;

        // Execute steps in order
        for step_id in execution_order {
            let step = workflow.steps.iter().find(|s| s.id == step_id).unwrap();
            self.context.current_step = Some(step_id.clone());

            match self.execute_step(step) {
                Ok(step_result) => {
                    step_results.push(step_result);
                }
                Err(e) => {
                    success = false;
                    error_message = Some(e.to_string());
                    step_results.push(StepExecutionResult {
                        step_id: step_id.clone(),
                        algorithm: step.algorithm.clone(),
                        success: false,
                        duration: Duration::from_millis(0),
                        memory_usage: 0,
                        output_sizes: HashMap::new(),
                        error: Some(e.to_string()),
                    });
                    break;
                }
            }
        }

        // Update statistics
        self.stats.total_executions += 1;
        if success {
            self.stats.successful_executions += 1;
        } else {
            self.stats.failed_executions += 1;
        }

        let total_duration = execution_start.elapsed();
        self.stats.average_execution_time = Duration::from_millis(
            (((self.stats.average_execution_time.as_millis()
                * u128::from(self.stats.total_executions - 1))
                + total_duration.as_millis())
                / u128::from(self.stats.total_executions))
            .try_into()
            .unwrap_or(u64::MAX),
        );

        Ok(ExecutionResult {
            execution_id,
            success,
            duration: total_duration,
            step_results,
            outputs: self.extract_final_outputs(&workflow),
            error: error_message,
            performance: self.calculate_performance_metrics(execution_start),
        })
    }

    /// Execute a single step
    fn execute_step(&mut self, step: &StepDefinition) -> SklResult<StepExecutionResult> {
        let step_start = Instant::now();

        // Get component definition
        let component = self
            .registry
            .get_component(&step.algorithm)
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!("Component '{}' not found", step.algorithm))
            })?;

        // Prepare input data
        let input_data = self.prepare_step_input(step)?;

        // Simulate step execution (in real implementation, this would call actual components)
        let output_data = self.simulate_step_execution(step, &input_data)?;

        // Store output data in context
        self.store_step_output(step, output_data.clone());

        // Update step execution count
        *self
            .stats
            .step_execution_counts
            .entry(step.algorithm.clone())
            .or_insert(0) += 1;

        Ok(StepExecutionResult {
            step_id: step.id.clone(),
            algorithm: step.algorithm.clone(),
            success: true,
            duration: step_start.elapsed(),
            memory_usage: self.estimate_memory_usage(&output_data),
            output_sizes: output_data
                .matrices
                .iter()
                .map(|(k, v)| (k.clone(), v.len()))
                .collect(),
            error: None,
        })
    }

    /// Prepare input data for a step
    fn prepare_step_input(&self, step: &StepDefinition) -> SklResult<StepData> {
        let mut input_data = StepData {
            step_id: step.id.clone(),
            port_name: "input".to_string(),
            matrices: HashMap::new(),
            arrays: HashMap::new(),
            metadata: HashMap::new(),
            timestamp: Instant::now(),
        };

        // For each input connection, get data from previous steps
        for connection in &self.context.workflow.connections {
            if connection.to_step == step.id {
                let source_data_key =
                    format!("{}:{}", connection.from_step, connection.from_output);
                if let Some(source_data) = self.context.data_flow.get(&source_data_key) {
                    // Copy relevant data based on connection mapping
                    for (key, matrix) in &source_data.matrices {
                        input_data.matrices.insert(key.clone(), matrix.clone());
                    }
                    for (key, array) in &source_data.arrays {
                        input_data.arrays.insert(key.clone(), array.clone());
                    }
                }
            }
        }

        Ok(input_data)
    }

    /// Simulate step execution (placeholder for actual component execution)
    fn simulate_step_execution(
        &self,
        step: &StepDefinition,
        input_data: &StepData,
    ) -> SklResult<StepData> {
        // This is a placeholder implementation
        // In a real system, this would delegate to the actual component implementation

        let mut output_data = StepData {
            step_id: step.id.clone(),
            port_name: "output".to_string(),
            matrices: HashMap::new(),
            arrays: HashMap::new(),
            metadata: HashMap::new(),
            timestamp: Instant::now(),
        };

        // Simple simulation based on component type
        match step.algorithm.as_str() {
            "StandardScaler" => {
                // Simulate scaling operation
                if let Some(input_matrix) = input_data.matrices.get("X") {
                    let scaled_matrix = input_matrix.clone(); // Placeholder
                    output_data
                        .matrices
                        .insert("X_scaled".to_string(), scaled_matrix);
                }
            }
            "LinearRegression" => {
                // Simulate training
                if input_data.matrices.contains_key("X") && input_data.arrays.contains_key("y") {
                    // Create dummy model output
                    output_data
                        .metadata
                        .insert("model_type".to_string(), "LinearRegression".to_string());
                    output_data
                        .metadata
                        .insert("trained".to_string(), "true".to_string());
                }
            }
            _ => {
                // Default behavior: pass through input data
                output_data.matrices = input_data.matrices.clone();
                output_data.arrays = input_data.arrays.clone();
            }
        }

        Ok(output_data)
    }

    /// Store step output in execution context
    fn store_step_output(&mut self, step: &StepDefinition, output_data: StepData) {
        for output_name in &step.outputs {
            let key = format!("{}:{}", step.id, output_name);
            self.context.data_flow.insert(key, output_data.clone());
        }
    }

    /// Extract final outputs from workflow execution
    fn extract_final_outputs(&self, workflow: &WorkflowDefinition) -> HashMap<String, String> {
        let mut outputs = HashMap::new();

        for output in &workflow.outputs {
            // Find the step that produces this output
            for step in &workflow.steps {
                if step.outputs.contains(&output.name) {
                    let key = format!("{}:{}", step.id, output.name);
                    if let Some(data) = self.context.data_flow.get(&key) {
                        outputs.insert(
                            output.name.clone(),
                            format!("Data from step '{}' port '{}'", step.id, output.name),
                        );
                    }
                }
            }
        }

        outputs
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(&self, start_time: Instant) -> PerformanceMetrics {
        PerformanceMetrics {
            total_time: start_time.elapsed(),
            peak_memory: 0,              // Placeholder
            cpu_utilization: 0.0,        // Placeholder
            throughput: 0.0,             // Placeholder
            parallelism_efficiency: 1.0, // Placeholder
        }
    }

    /// Estimate memory usage for step data
    fn estimate_memory_usage(&self, _data: &StepData) -> u64 {
        // Placeholder implementation
        1024 * 1024 // 1MB
    }

    /// Get execution statistics
    #[must_use]
    pub fn get_statistics(&self) -> &ExecutionStatistics {
        &self.stats
    }
}

impl ExecutionContext {
    fn new() -> Self {
        Self {
            execution_id: String::new(),
            workflow: WorkflowDefinition::default(),
            data_flow: HashMap::new(),
            start_time: Instant::now(),
            current_step: None,
            execution_mode: ExecutionMode::Sequential,
        }
    }
}

impl ExecutionStatistics {
    fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_execution_time: Duration::from_secs(0),
            step_execution_counts: HashMap::new(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time: Duration::from_secs(0),
            peak_memory: 0,
            cpu_utilization: 0.0,
            throughput: 0.0,
            parallelism_efficiency: 0.0,
        }
    }
}

impl Default for WorkflowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution state tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionState {
    /// Execution is initializing
    Initializing,
    /// Execution is preparing
    Preparing,
    /// Execution is running
    Running,
    /// Execution is paused
    Paused,
    /// Execution completed successfully
    Completed,
    /// Execution failed
    Failed,
    /// Execution was cancelled
    Cancelled,
    /// Execution timed out
    TimedOut,
}

/// Execution tracker for monitoring workflow progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTracker {
    /// Current execution state
    pub state: ExecutionState,
    /// Progress percentage (0-100)
    pub progress: f64,
    /// Currently executing step
    pub current_step: Option<String>,
    /// Completed steps
    pub completed_steps: Vec<String>,
    /// Failed steps
    pub failed_steps: Vec<String>,
    /// Execution start time
    pub start_time: String,
    /// Estimated completion time
    pub estimated_completion: Option<String>,
    /// Error messages
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionConfig {
    /// Maximum number of parallel workers
    pub max_workers: usize,
    /// Task queue size
    pub queue_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Thread pool configuration
    pub thread_pool: ThreadPoolConfig,
    /// Resource sharing strategy
    pub resource_sharing: ResourceSharingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least loaded worker
    LeastLoaded,
    /// Random assignment
    Random,
    /// Work stealing
    WorkStealing,
    /// Custom strategy
    Custom(String),
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Core thread count
    pub core_threads: usize,
    /// Maximum thread count
    pub max_threads: usize,
    /// Thread keep-alive time in seconds
    pub keep_alive_sec: u64,
    /// Thread stack size in bytes
    pub stack_size: Option<usize>,
}

/// Resource sharing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceSharingStrategy {
    /// Exclusive resource access
    Exclusive,
    /// Shared resource access
    Shared,
    /// Copy-on-write sharing
    CopyOnWrite,
    /// Memory mapped sharing
    MemoryMapped,
}

/// Resource allocation for workflow execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU allocation
    pub cpu: CpuAllocation,
    /// Memory allocation
    pub memory: MemoryAllocation,
    /// GPU allocation
    pub gpu: Option<GpuAllocation>,
    /// Disk allocation
    pub disk: Option<DiskAllocation>,
    /// Network allocation
    pub network: Option<NetworkAllocation>,
}

/// CPU resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAllocation {
    /// Number of cores allocated
    pub cores: usize,
    /// CPU utilization limit (0.0-1.0)
    pub utilization_limit: f64,
    /// CPU affinity settings
    pub affinity: Vec<usize>,
}

/// Memory resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Maximum memory in MB
    pub max_memory_mb: usize,
    /// Memory type preference
    pub memory_type: MemoryType,
    /// Swap allowance
    pub allow_swap: bool,
}

/// Memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    /// System RAM
    Ram,
    /// High bandwidth memory
    Hbm,
    /// Non-volatile memory
    Nvram,
    /// Any available memory
    Any,
}

/// GPU resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    /// GPU device IDs
    pub device_ids: Vec<usize>,
    /// Memory per GPU in MB
    pub memory_per_gpu_mb: usize,
    /// Compute capability requirement
    pub min_compute_capability: f64,
}

/// Disk resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskAllocation {
    /// Temporary storage in MB
    pub temp_storage_mb: usize,
    /// Storage paths
    pub storage_paths: Vec<String>,
    /// I/O bandwidth limit in MB/s
    pub io_bandwidth_mbs: Option<f64>,
}

/// Network resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAllocation {
    /// Bandwidth limit in Mbps
    pub bandwidth_mbps: f64,
    /// Connection limit
    pub max_connections: usize,
    /// Network interfaces
    pub interfaces: Vec<String>,
}

/// Resource manager for managing workflow resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManager {
    /// Available resource pool
    pub available_resources: ResourcePool,
    /// Current allocations
    pub allocations: HashMap<String, ResourceAllocation>,
    /// Resource monitoring
    pub monitoring: ResourceMonitoring,
    /// Resource scheduling strategy
    pub scheduling_strategy: ResourceSchedulingStrategy,
}

/// Resource pool available for allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    /// Total CPU cores
    pub total_cpu_cores: usize,
    /// Total memory in MB
    pub total_memory_mb: usize,
    /// Available GPUs
    pub gpus: Vec<GpuInfo>,
    /// Available disk space in MB
    pub disk_space_mb: usize,
    /// Network bandwidth in Mbps
    pub network_bandwidth_mbps: f64,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU ID
    pub id: usize,
    /// GPU name
    pub name: String,
    /// Memory in MB
    pub memory_mb: usize,
    /// Compute capability
    pub compute_capability: f64,
    /// Whether currently available
    pub available: bool,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoring {
    /// Enable resource monitoring
    pub enabled: bool,
    /// Monitoring interval in seconds
    pub interval_sec: u64,
    /// Resource usage thresholds
    pub thresholds: ResourceThresholds,
}

/// Resource usage thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// CPU usage warning threshold
    pub cpu_warning: f64,
    /// Memory usage warning threshold
    pub memory_warning: f64,
    /// Disk usage warning threshold
    pub disk_warning: f64,
}

/// Resource scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceSchedulingStrategy {
    /// First-come, first-served
    Fcfs,
    /// Shortest job first
    Sjf,
    /// Round-robin
    RoundRobin,
    /// Priority-based
    Priority,
    /// Fair share
    FairShare,
    /// Custom strategy
    Custom(String),
}

/// Workflow execution error types
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum WorkflowExecutionError {
    /// Validation error
    #[error("Workflow validation error: {0}")]
    ValidationError(String),
    /// Resource allocation failed
    #[error("Resource allocation failed: {0}")]
    ResourceAllocationError(String),
    /// Step execution failed
    #[error("Step execution failed for '{0}': {1}")]
    StepExecutionError(String, String), // step_id, error_message
    /// Dependency resolution failed
    #[error("Dependency resolution failed: {0}")]
    DependencyError(String),
    /// Timeout occurred
    #[error("Workflow timeout: {0}")]
    TimeoutError(String),
    /// Cancellation requested
    #[error("Workflow cancelled: {0}")]
    CancellationError(String),
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    /// System error
    #[error("System error: {0}")]
    SystemError(String),
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_language::workflow_definitions::{DataType, StepType};

    #[test]
    fn test_workflow_executor_creation() {
        let executor = WorkflowExecutor::new();
        assert_eq!(executor.stats.total_executions, 0);
    }

    #[test]
    fn test_empty_workflow_validation() {
        let executor = WorkflowExecutor::new();
        let workflow = WorkflowDefinition::default();

        let validation = executor.validate_workflow(&workflow);
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
        assert_eq!(validation.errors[0].error_type, "EmptyWorkflow");
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

        // Add steps
        workflow.steps.push(
            StepDefinition::new("step1", StepType::Transformer, "StandardScaler")
                .with_output("X_scaled"),
        );
        workflow.steps.push(
            StepDefinition::new("step2", StepType::Trainer, "LinearRegression").with_input("X"),
        );

        // Add connection
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
}
