//! Pipeline System for Modular Composition
//!
//! This module provides comprehensive pipeline configuration, execution strategies,
//! and composition patterns for building complex data processing workflows using
//! modular components with support for parallel execution, error handling, and monitoring.

use serde::{Deserialize, Serialize};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;

use super::component_framework::{ComponentConfig, PluggableComponent};
use super::event_system::EventBus;

/// Pipeline builder for creating modular data processing pipelines
///
/// Provides fluent API for pipeline construction with component composition,
/// execution strategies, error handling, and monitoring capabilities.
#[derive(Debug)]
pub struct PipelineBuilder {
    /// Pipeline stages
    stages: Vec<PipelineStage>,
    /// Pipeline configuration
    config: PipelineConfiguration,
    /// Error handling strategy
    error_strategy: ErrorHandlingStrategy,
    /// Execution strategy
    execution_strategy: ExecutionStrategy,
    /// Pipeline metadata
    metadata: PipelineMetadata,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfiguration::default(),
            error_strategy: ErrorHandlingStrategy::FailFast,
            execution_strategy: ExecutionStrategy::Sequential,
            metadata: PipelineMetadata::new(),
        }
    }

    /// Add a component stage to the pipeline
    #[must_use]
    pub fn add_stage(mut self, component_type: &str, config: ComponentConfig) -> Self {
        let stage = PipelineStage {
            stage_id: format!("stage_{}", self.stages.len()),
            component_type: component_type.to_string(),
            component_config: config,
            stage_type: StageType::Component,
            parallel_branches: Vec::new(),
            conditional_execution: None,
            retry_config: None,
            timeout: None,
        };

        self.stages.push(stage);
        self
    }

    /// Add a parallel stage with multiple branches
    #[must_use]
    pub fn add_parallel_stage(mut self, branches: Vec<ParallelBranch>) -> Self {
        let stage = PipelineStage {
            stage_id: format!("parallel_stage_{}", self.stages.len()),
            component_type: "parallel".to_string(),
            component_config: ComponentConfig::new("parallel", "parallel"),
            stage_type: StageType::Parallel,
            parallel_branches: branches,
            conditional_execution: None,
            retry_config: None,
            timeout: None,
        };

        self.stages.push(stage);
        self
    }

    /// Add a conditional stage
    #[must_use]
    pub fn add_conditional_stage(
        mut self,
        component_type: &str,
        config: ComponentConfig,
        condition: Box<dyn ConditionalExecution>,
    ) -> Self {
        let stage = PipelineStage {
            stage_id: format!("conditional_stage_{}", self.stages.len()),
            component_type: component_type.to_string(),
            component_config: config,
            stage_type: StageType::Conditional,
            parallel_branches: Vec::new(),
            conditional_execution: Some(condition),
            retry_config: None,
            timeout: None,
        };

        self.stages.push(stage);
        self
    }

    /// Set error handling strategy
    #[must_use]
    pub fn with_error_strategy(mut self, strategy: ErrorHandlingStrategy) -> Self {
        self.error_strategy = strategy;
        self
    }

    /// Set execution strategy
    #[must_use]
    pub fn with_execution_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.execution_strategy = strategy;
        self
    }

    /// Set pipeline timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.pipeline_timeout = Some(timeout);
        self
    }

    /// Set retry configuration
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfiguration) -> Self {
        self.config.retry_config = Some(config);
        self
    }

    /// Add pipeline metadata
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata
            .custom_metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Build the pipeline
    pub fn build(self) -> SklResult<Pipeline> {
        self.validate_pipeline()?;

        Ok(Pipeline {
            pipeline_id: uuid::Uuid::new_v4().to_string(),
            stages: self.stages,
            config: self.config,
            error_strategy: self.error_strategy,
            execution_strategy: self.execution_strategy,
            metadata: self.metadata,
            state: PipelineState::Created,
            components: Arc::new(RwLock::new(HashMap::new())),
            event_bus: Arc::new(RwLock::new(EventBus::new())),
            execution_context: Arc::new(RwLock::new(ExecutionContext::new())),
            metrics: Arc::new(Mutex::new(PipelineMetrics::new())),
        })
    }

    /// Validate pipeline configuration
    fn validate_pipeline(&self) -> SklResult<()> {
        if self.stages.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Pipeline must have at least one stage".to_string(),
            ));
        }

        // Validate stage dependencies
        for stage in &self.stages {
            if stage.component_type.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "Stage component type cannot be empty".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Check if the pipeline has no stages
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

/// Complete pipeline with stages and execution context
pub struct Pipeline {
    /// Unique pipeline identifier
    pub pipeline_id: String,
    /// Pipeline stages
    pub stages: Vec<PipelineStage>,
    /// Pipeline configuration
    pub config: PipelineConfiguration,
    /// Error handling strategy
    pub error_strategy: ErrorHandlingStrategy,
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Pipeline metadata
    pub metadata: PipelineMetadata,
    /// Current pipeline state
    pub state: PipelineState,
    /// Component instances
    pub components: Arc<RwLock<HashMap<String, Box<dyn PluggableComponent>>>>,
    /// Event bus for component communication
    pub event_bus: Arc<RwLock<EventBus>>,
    /// Execution context
    pub execution_context: Arc<RwLock<ExecutionContext>>,
    /// Pipeline metrics
    pub metrics: Arc<Mutex<PipelineMetrics>>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("pipeline_id", &self.pipeline_id)
            .field("stages", &self.stages)
            .field("config", &self.config)
            .field("error_strategy", &self.error_strategy)
            .field("execution_strategy", &self.execution_strategy)
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .field("components", &"[components: <RwLock>]".to_string())
            .field("event_bus", &"[event_bus: <RwLock>]".to_string())
            .field(
                "execution_context",
                &"[execution_context: <RwLock>]".to_string(),
            )
            .field("metrics", &"[metrics: <Mutex>]".to_string())
            .finish()
    }
}

impl Pipeline {
    /// Execute the pipeline
    pub async fn execute(&mut self, input_data: PipelineData) -> SklResult<PipelineResult> {
        let start_time = Instant::now();
        self.state = PipelineState::Running;

        let mut metrics = self.metrics.lock().unwrap();
        metrics.execution_count += 1;
        metrics.last_execution_start = Some(start_time);
        drop(metrics);

        // Initialize execution context
        {
            let mut context = self.execution_context.write().unwrap();
            context.input_data = input_data;
            context.execution_id = uuid::Uuid::new_v4().to_string();
            context.start_time = start_time;
        }

        let execution_result = match self.execution_strategy {
            ExecutionStrategy::Sequential => self.execute_sequential().await,
            ExecutionStrategy::Parallel => self.execute_parallel().await,
            ExecutionStrategy::Adaptive => self.execute_adaptive().await,
        };

        let end_time = Instant::now();
        let execution_duration = end_time.duration_since(start_time);

        // Update metrics
        let mut metrics = self.metrics.lock().unwrap();
        metrics.last_execution_end = Some(end_time);
        metrics.total_execution_time += execution_duration;

        if execution_result.is_ok() {
            metrics.successful_executions += 1;
            self.state = PipelineState::Completed;
        } else {
            metrics.failed_executions += 1;
            self.state = PipelineState::Failed;
        }

        execution_result
    }

    /// Execute pipeline sequentially
    async fn execute_sequential(&mut self) -> SklResult<PipelineResult> {
        let mut stage_results = Vec::new();
        let mut current_data = {
            let context = self.execution_context.read().unwrap();
            context.input_data.clone()
        };

        let stages = self.stages.clone(); // Clone to avoid borrow conflicts
        for (index, stage) in stages.iter().enumerate() {
            let stage_start = Instant::now();

            match self.execute_stage(stage, &current_data).await {
                Ok(stage_result) => {
                    current_data = stage_result.output_data.clone();
                    stage_results.push(stage_result);
                }
                Err(error) => {
                    let stage_result = StageResult {
                        stage_id: stage.stage_id.clone(),
                        success: false,
                        execution_time: stage_start.elapsed(),
                        error: Some(error.to_string()),
                        output_data: PipelineData::empty(),
                        metrics: StageMetrics::new(),
                    };
                    stage_results.push(stage_result);

                    return self.handle_stage_error(index, error);
                }
            }
        }

        Ok(PipelineResult {
            pipeline_id: self.pipeline_id.clone(),
            success: true,
            stage_results,
            final_output: current_data,
            execution_time: {
                let context = self.execution_context.read().unwrap();
                context.start_time.elapsed()
            },
            error: None,
        })
    }

    /// Execute pipeline in parallel where possible
    async fn execute_parallel(&mut self) -> SklResult<PipelineResult> {
        // Placeholder for parallel execution
        // In a real implementation, this would analyze stage dependencies
        // and execute independent stages in parallel
        self.execute_sequential().await
    }

    /// Execute pipeline with adaptive strategy
    async fn execute_adaptive(&mut self) -> SklResult<PipelineResult> {
        // Adaptive execution would choose between sequential and parallel
        // based on resource availability and stage characteristics
        if self.stages.len() > 3 {
            self.execute_parallel().await
        } else {
            self.execute_sequential().await
        }
    }

    /// Execute a single stage
    async fn execute_stage(
        &mut self,
        stage: &PipelineStage,
        input_data: &PipelineData,
    ) -> SklResult<StageResult> {
        let stage_start = Instant::now();

        // Check conditional execution
        if let Some(condition) = &stage.conditional_execution {
            if !condition.should_execute(input_data)? {
                return Ok(StageResult {
                    stage_id: stage.stage_id.clone(),
                    success: true,
                    execution_time: stage_start.elapsed(),
                    error: None,
                    output_data: input_data.clone(),
                    metrics: StageMetrics::new(),
                });
            }
        }

        let execution_result = match stage.stage_type {
            StageType::Component => self.execute_component_stage(stage, input_data).await,
            StageType::Parallel => self.execute_parallel_stage(stage, input_data).await,
            StageType::Conditional => self.execute_component_stage(stage, input_data).await,
        };

        let mut stage_result = match execution_result {
            Ok(result) => result,
            Err(error) => StageResult {
                stage_id: stage.stage_id.clone(),
                success: false,
                execution_time: stage_start.elapsed(),
                error: Some(error.to_string()),
                output_data: PipelineData::empty(),
                metrics: StageMetrics::new(),
            },
        };

        stage_result.execution_time = stage_start.elapsed();
        Ok(stage_result)
    }

    /// Execute a component stage
    async fn execute_component_stage(
        &mut self,
        stage: &PipelineStage,
        input_data: &PipelineData,
    ) -> SklResult<StageResult> {
        // Get or create component instance
        let component_key = format!("{}_{}", stage.component_type, stage.stage_id);

        // For now, return a placeholder result
        // In a real implementation, this would:
        // 1. Get component from registry
        // 2. Initialize component with stage config
        // 3. Process input data through component
        // 4. Return processed output

        Ok(StageResult {
            stage_id: stage.stage_id.clone(),
            success: true,
            execution_time: Duration::from_millis(10),
            error: None,
            output_data: input_data.clone(),
            metrics: StageMetrics::new(),
        })
    }

    /// Execute a parallel stage
    async fn execute_parallel_stage(
        &mut self,
        stage: &PipelineStage,
        input_data: &PipelineData,
    ) -> SklResult<StageResult> {
        let mut branch_results = Vec::new();

        // Execute all branches in parallel
        for branch in &stage.parallel_branches {
            // In a real implementation, this would spawn async tasks
            let branch_result = self.execute_parallel_branch(branch, input_data).await?;
            branch_results.push(branch_result);
        }

        // Combine branch results
        let combined_output = self.combine_parallel_results(&branch_results)?;

        Ok(StageResult {
            stage_id: stage.stage_id.clone(),
            success: true,
            execution_time: Duration::from_millis(20),
            error: None,
            output_data: combined_output,
            metrics: StageMetrics::new(),
        })
    }

    /// Execute a parallel branch
    async fn execute_parallel_branch(
        &mut self,
        branch: &ParallelBranch,
        input_data: &PipelineData,
    ) -> SklResult<PipelineData> {
        // Placeholder for branch execution
        Ok(input_data.clone())
    }

    /// Combine results from parallel branches
    fn combine_parallel_results(&self, results: &[PipelineData]) -> SklResult<PipelineData> {
        // Placeholder for result combination
        if let Some(first) = results.first() {
            Ok(first.clone())
        } else {
            Ok(PipelineData::empty())
        }
    }

    /// Handle stage execution error
    fn handle_stage_error(
        &self,
        stage_index: usize,
        error: SklearsError,
    ) -> SklResult<PipelineResult> {
        match self.error_strategy {
            ErrorHandlingStrategy::FailFast => Err(error),
            ErrorHandlingStrategy::ContinueOnError => {
                // Return partial result
                Ok(PipelineResult {
                    pipeline_id: self.pipeline_id.clone(),
                    success: false,
                    stage_results: Vec::new(),
                    final_output: PipelineData::empty(),
                    execution_time: Duration::from_secs(0),
                    error: Some(error.to_string()),
                })
            }
            ErrorHandlingStrategy::Retry => {
                // Implement retry logic
                Err(error)
            }
        }
    }

    /// Get pipeline metrics
    #[must_use]
    pub fn get_metrics(&self) -> PipelineMetrics {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }

    /// Get pipeline state
    #[must_use]
    pub fn get_state(&self) -> PipelineState {
        self.state.clone()
    }
}

/// Pipeline stage configuration
#[derive(Debug)]
pub struct PipelineStage {
    /// Stage identifier
    pub stage_id: String,
    /// Component type for this stage
    pub component_type: String,
    /// Component configuration
    pub component_config: ComponentConfig,
    /// Stage type
    pub stage_type: StageType,
    /// Parallel branches (for parallel stages)
    pub parallel_branches: Vec<ParallelBranch>,
    /// Conditional execution logic
    pub conditional_execution: Option<Box<dyn ConditionalExecution>>,
    /// Retry configuration
    pub retry_config: Option<RetryConfiguration>,
    /// Stage timeout
    pub timeout: Option<Duration>,
}

impl Clone for PipelineStage {
    fn clone(&self) -> Self {
        Self {
            stage_id: self.stage_id.clone(),
            component_type: self.component_type.clone(),
            component_config: self.component_config.clone(),
            stage_type: self.stage_type.clone(),
            parallel_branches: self.parallel_branches.clone(),
            conditional_execution: None, // Skip trait object cloning
            retry_config: self.retry_config.clone(),
            timeout: self.timeout,
        }
    }
}

/// Stage types
#[derive(Debug, Clone, PartialEq)]
pub enum StageType {
    /// Single component stage
    Component,
    /// Parallel execution stage
    Parallel,
    /// Conditional execution stage
    Conditional,
}

/// Parallel branch configuration
#[derive(Debug, Clone)]
pub struct ParallelBranch {
    /// Branch identifier
    pub branch_id: String,
    /// Component type for this branch
    pub component_type: String,
    /// Branch configuration
    pub config: ComponentConfig,
    /// Branch weight for load balancing
    pub weight: f64,
}

/// Conditional execution trait
pub trait ConditionalExecution: Send + Sync + std::fmt::Debug {
    /// Check if stage should execute based on input data
    fn should_execute(&self, input_data: &PipelineData) -> SklResult<bool>;

    /// Get condition description
    fn description(&self) -> String;
}

/// Error handling strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ErrorHandlingStrategy {
    /// Stop pipeline execution on first error
    #[default]
    FailFast,
    /// Continue pipeline execution despite errors
    ContinueOnError,
    /// Retry failed stages
    Retry,
}

/// Execution strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ExecutionStrategy {
    #[default]
    /// Execute stages sequentially
    Sequential,
    /// Execute stages in parallel where possible
    Parallel,
    /// Adaptively choose execution strategy
    Adaptive,
}

/// Pipeline states
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineState {
    /// Pipeline created but not started
    Created,
    /// Pipeline is running
    Running,
    /// Pipeline completed successfully
    Completed,
    /// Pipeline failed
    Failed,
    /// Pipeline was cancelled
    Cancelled,
    /// Pipeline is paused
    Paused,
}

/// Pipeline data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineData {
    /// Data payload
    pub data: HashMap<String, serde_json::Value>,
    /// Data metadata
    pub metadata: HashMap<String, String>,
    /// Data timestamp
    pub timestamp: Option<String>,
}

impl PipelineData {
    /// Create empty pipeline data
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: HashMap::new(),
            metadata: HashMap::new(),
            timestamp: None,
        }
    }

    /// Create pipeline data with initial data
    #[must_use]
    pub fn new(data: HashMap<String, serde_json::Value>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
        }
    }

    /// Add data field
    #[must_use]
    pub fn with_data(mut self, key: &str, value: serde_json::Value) -> Self {
        self.data.insert(key.to_string(), value);
        self
    }

    /// Add metadata field
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get data field
    #[must_use]
    pub fn get_data(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }

    /// Get metadata field
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Pipeline execution result
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Pipeline identifier
    pub pipeline_id: String,
    /// Execution success status
    pub success: bool,
    /// Results from each stage
    pub stage_results: Vec<StageResult>,
    /// Final pipeline output
    pub final_output: PipelineData,
    /// Total execution time
    pub execution_time: Duration,
    /// Error message if execution failed
    pub error: Option<String>,
}

/// Stage execution result
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage identifier
    pub stage_id: String,
    /// Stage execution success
    pub success: bool,
    /// Stage execution time
    pub execution_time: Duration,
    /// Error message if stage failed
    pub error: Option<String>,
    /// Stage output data
    pub output_data: PipelineData,
    /// Stage metrics
    pub metrics: StageMetrics,
}

/// Stage execution metrics
#[derive(Debug, Clone)]
pub struct StageMetrics {
    /// Memory usage during stage execution
    pub memory_usage: u64,
    /// CPU usage during stage execution
    pub cpu_usage: f64,
    /// Number of processed items
    pub processed_items: u64,
    /// Custom stage metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl StageMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            memory_usage: 0,
            cpu_usage: 0.0,
            processed_items: 0,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfiguration {
    /// Pipeline timeout
    pub pipeline_timeout: Option<Duration>,
    /// Maximum parallel stages
    pub max_parallel_stages: usize,
    /// Retry configuration
    pub retry_config: Option<RetryConfiguration>,
    /// Enable pipeline metrics collection
    pub enable_metrics: bool,
    /// Enable stage profiling
    pub enable_profiling: bool,
}

impl Default for PipelineConfiguration {
    fn default() -> Self {
        Self {
            pipeline_timeout: None,
            max_parallel_stages: 4,
            retry_config: None,
            enable_metrics: true,
            enable_profiling: false,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfiguration {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay between retries
    pub max_delay: Duration,
}

/// Pipeline metadata
#[derive(Debug, Clone)]
pub struct PipelineMetadata {
    /// Pipeline name
    pub name: Option<String>,
    /// Pipeline description
    pub description: Option<String>,
    /// Pipeline version
    pub version: Option<String>,
    /// Pipeline author
    pub author: Option<String>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
}

impl PipelineMetadata {
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            version: None,
            author: None,
            created_at: Instant::now(),
            custom_metadata: HashMap::new(),
        }
    }
}

/// Execution context
#[derive(Debug)]
pub struct ExecutionContext {
    /// Execution identifier
    pub execution_id: String,
    /// Input data
    pub input_data: PipelineData,
    /// Execution start time
    pub start_time: Instant,
    /// Context variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Execution trace
    pub trace: Vec<ExecutionTrace>,
}

impl ExecutionContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            execution_id: String::new(),
            input_data: PipelineData::empty(),
            start_time: Instant::now(),
            variables: HashMap::new(),
            trace: Vec::new(),
        }
    }

    /// Add execution trace entry
    pub fn add_trace(&mut self, stage_id: &str, event: &str, data: Option<serde_json::Value>) {
        self.trace.push(ExecutionTrace {
            timestamp: Instant::now(),
            stage_id: stage_id.to_string(),
            event: event.to_string(),
            data,
        });
    }
}

/// Execution trace entry
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Trace timestamp
    pub timestamp: Instant,
    /// Stage identifier
    pub stage_id: String,
    /// Trace event
    pub event: String,
    /// Optional trace data
    pub data: Option<serde_json::Value>,
}

/// Pipeline metrics
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Total number of executions
    pub execution_count: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Last execution start time
    pub last_execution_start: Option<Instant>,
    /// Last execution end time
    pub last_execution_end: Option<Instant>,
    /// Average execution time
    pub average_execution_time: Duration,
}

impl PipelineMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            execution_count: 0,
            successful_executions: 0,
            failed_executions: 0,
            total_execution_time: Duration::from_secs(0),
            last_execution_start: None,
            last_execution_end: None,
            average_execution_time: Duration::from_secs(0),
        }
    }

    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.execution_count as f64
        }
    }

    /// Update average execution time
    pub fn update_average(&mut self) {
        if self.execution_count > 0 {
            self.average_execution_time = self.total_execution_time / self.execution_count as u32;
        }
    }
}

/// Pipeline system errors
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Pipeline validation failed: {0}")]
    ValidationFailed(String),

    #[error("Stage execution failed: {stage_id}: {error}")]
    StageExecutionFailed { stage_id: String, error: String },

    #[error("Pipeline timeout exceeded: {0:?}")]
    TimeoutExceeded(Duration),

    #[error("Invalid pipeline state: {0}")]
    InvalidState(String),

    #[error("Component not found: {0}")]
    ComponentNotFound(String),
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PipelineMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for StageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ========== Missing Pipeline Types ==========

/// Modular pipeline for component composition
#[derive(Debug, Clone)]
pub struct ModularPipeline {
    /// Pipeline identifier
    pub id: String,
    /// Pipeline stages
    pub stages: Vec<PipelineStage>,
    /// Pipeline configuration
    pub config: PipelineConfig,
    /// Pipeline metadata
    pub metadata: PipelineMetadata,
    /// Execution context
    pub execution_context: Arc<Mutex<ExecutionContext>>,
}

/// Type alias for modular pipeline builder
pub type ModularPipelineBuilder = PipelineBuilder;

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name
    pub name: String,
    /// Pipeline description
    pub description: Option<String>,
    /// Execution strategy
    #[serde(skip)]
    pub execution_strategy: ExecutionStrategy,
    /// Error handling strategy
    #[serde(skip)]
    pub error_handling: ErrorHandlingStrategy,
    /// Resource constraints
    pub resource_constraints: Option<ResourceConstraints>,
    /// Timeout configuration
    pub timeout_config: Option<TimeoutConfig>,
    /// Retry configuration
    pub retry_config: Option<RetryConfig>,
}

/// Pipeline step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    /// Step identifier
    pub id: String,
    /// Step name
    pub name: String,
    /// Component type for this step
    pub component_type: String,
    /// Step configuration
    pub config: ComponentConfig,
    /// Step dependencies
    pub dependencies: Vec<String>,
    /// Step condition
    pub condition: Option<String>,
    /// Step retry policy
    pub retry_policy: Option<RetryPolicy>,
}

/// Resource constraints for pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
    /// Maximum CPU cores
    pub max_cpu_cores: Option<usize>,
    /// Maximum execution time in seconds
    pub max_execution_time_sec: Option<u64>,
    /// Maximum concurrent steps
    pub max_concurrent_steps: Option<usize>,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Step timeout in seconds
    pub step_timeout_sec: u64,
    /// Pipeline timeout in seconds
    pub pipeline_timeout_sec: u64,
    /// Timeout action
    pub timeout_action: TimeoutAction,
}

/// Timeout actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeoutAction {
    /// Fail the pipeline
    Fail,
    /// Skip the step
    Skip,
    /// Retry the step
    Retry,
    /// Use default value
    UseDefault,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
    /// Retryable error types
    pub retryable_errors: Vec<String>,
}

/// Backoff strategies for retries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay
    Fixed,
    /// Linear backoff
    Linear,
    /// Exponential backoff
    Exponential,
    /// Custom backoff
    Custom(String),
}

/// Retry policy for individual steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Enable retry for this step
    pub enabled: bool,
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Retry delay in milliseconds
    pub delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl ModularPipeline {
    /// Create a new modular pipeline
    #[must_use]
    pub fn new(id: String, config: PipelineConfig) -> Self {
        Self {
            id,
            stages: Vec::new(),
            config,
            metadata: PipelineMetadata::new(),
            execution_context: Arc::new(Mutex::new(ExecutionContext::new())),
        }
    }

    /// Add a pipeline step
    pub fn add_step(&mut self, step: PipelineStep) {
        // Convert PipelineStep to PipelineStage for internal use
        let stage = PipelineStage {
            stage_id: step.id.clone(),
            component_type: step.component_type,
            component_config: step.config,
            stage_type: StageType::Component,
            parallel_branches: Vec::new(),
            conditional_execution: None,
            retry_config: None,
            timeout: None,
        };
        self.stages.push(stage);
    }

    /// Get pipeline step by ID
    #[must_use]
    pub fn get_step(&self, step_id: &str) -> Option<PipelineStep> {
        self.stages
            .iter()
            .find(|stage| stage.stage_id == step_id)
            .map(|stage| {
                /// PipelineStep
                PipelineStep {
                    id: stage.stage_id.clone(),
                    name: stage.stage_id.clone(), // Use stage_id as name for simplicity
                    component_type: stage.component_type.clone(),
                    config: stage.component_config.clone(),
                    dependencies: Vec::new(), // Would need to track this separately
                    condition: None,
                    retry_policy: None,
                }
            })
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            name: "DefaultPipeline".to_string(),
            description: None,
            execution_strategy: ExecutionStrategy::Sequential,
            error_handling: ErrorHandlingStrategy::FailFast,
            resource_constraints: None,
            timeout_config: None,
            retry_config: None,
        }
    }
}

impl Default for TimeoutAction {
    fn default() -> Self {
        Self::Fail
    }
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self::Exponential
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .add_stage("test_component", ComponentConfig::new("test", "test_type"))
            .with_error_strategy(ErrorHandlingStrategy::FailFast)
            .with_execution_strategy(ExecutionStrategy::Sequential)
            .build();

        assert!(pipeline.is_ok());
        let pipeline = pipeline.unwrap();
        assert_eq!(pipeline.stages.len(), 1);
        assert_eq!(pipeline.error_strategy, ErrorHandlingStrategy::FailFast);
    }

    #[test]
    fn test_pipeline_data() {
        let mut data = HashMap::new();
        data.insert(
            "key1".to_string(),
            serde_json::Value::String("value1".to_string()),
        );

        let pipeline_data = PipelineData::new(data)
            .with_metadata("source", "test")
            .with_data(
                "key2",
                serde_json::Value::Number(serde_json::Number::from(42)),
            );

        assert_eq!(pipeline_data.get_metadata("source"), Some("test"));
        assert!(pipeline_data.get_data("key1").is_some());
        assert!(pipeline_data.get_data("key2").is_some());
    }

    #[test]
    fn test_pipeline_metrics() {
        let mut metrics = PipelineMetrics::new();
        metrics.execution_count = 10;
        metrics.successful_executions = 8;
        metrics.failed_executions = 2;

        assert_eq!(metrics.success_rate(), 0.8);
    }

    #[test]
    fn test_execution_context() {
        let mut context = ExecutionContext::new();
        context.add_trace("stage1", "started", None);
        context.add_trace(
            "stage1",
            "completed",
            Some(serde_json::Value::String("success".to_string())),
        );

        assert_eq!(context.trace.len(), 2);
        assert_eq!(context.trace[0].stage_id, "stage1");
        assert_eq!(context.trace[0].event, "started");
    }

    #[test]
    fn test_empty_pipeline_validation() {
        let pipeline = PipelineBuilder::new().build();
        assert!(pipeline.is_err());
    }
}
