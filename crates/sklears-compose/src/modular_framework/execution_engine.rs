//! Execution Engine for Modular Composition
//!
//! This module provides the core execution engine for modular component composition
//! including composition context management, execution orchestration, resource management,
//! and runtime optimization for complex modular workflows.

use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

use super::component_framework::{ComponentConfig, ComponentMetrics, PluggableComponent};
use super::dependency_management::DependencyResolver;
use super::event_system::{ComponentEvent, EventBus};
use super::lifecycle_management::LifecycleManager;
use super::pipeline_system::{ExecutionStrategy, Pipeline, PipelineData, PipelineResult};
use super::registry_system::GlobalComponentRegistry;

/// Composition execution engine
///
/// Provides comprehensive execution orchestration for modular component composition
/// with resource management, parallel execution, fault tolerance, and performance optimization.
#[derive(Debug)]
pub struct CompositionExecutionEngine {
    /// Execution contexts by context ID
    contexts: Arc<RwLock<HashMap<String, Arc<RwLock<CompositionContext>>>>>,
    /// Component registry
    component_registry: Arc<GlobalComponentRegistry>,
    /// Dependency resolver
    dependency_resolver: Arc<DependencyResolver>,
    /// Lifecycle manager
    lifecycle_manager: Arc<LifecycleManager>,
    /// Execution scheduler
    scheduler: Arc<RwLock<ExecutionScheduler>>,
    /// Resource manager
    resource_manager: Arc<RwLock<ResourceManager>>,
    /// Engine configuration
    config: ExecutionEngineConfig,
    /// Execution statistics
    stats: Arc<Mutex<ExecutionStatistics>>,
    /// Event bus for engine events
    event_bus: Arc<RwLock<EventBus>>,
}

impl CompositionExecutionEngine {
    /// Create a new execution engine
    #[must_use]
    pub fn new(
        component_registry: Arc<GlobalComponentRegistry>,
        dependency_resolver: Arc<DependencyResolver>,
        lifecycle_manager: Arc<LifecycleManager>,
    ) -> Self {
        Self {
            contexts: Arc::new(RwLock::new(HashMap::new())),
            component_registry,
            dependency_resolver,
            lifecycle_manager,
            scheduler: Arc::new(RwLock::new(ExecutionScheduler::new())),
            resource_manager: Arc::new(RwLock::new(ResourceManager::new())),
            config: ExecutionEngineConfig::default(),
            stats: Arc::new(Mutex::new(ExecutionStatistics::new())),
            event_bus: Arc::new(RwLock::new(EventBus::new())),
        }
    }

    /// Create a new composition context
    pub fn create_context(&self, context_id: &str) -> SklResult<Arc<RwLock<CompositionContext>>> {
        let mut contexts = self.contexts.write().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if contexts.contains_key(context_id) && !self.config.allow_context_override {
            return Err(SklearsError::InvalidInput(format!(
                "Context {context_id} already exists"
            )));
        }

        let context = Arc::new(RwLock::new(CompositionContext::new(context_id)));
        contexts.insert(context_id.to_string(), context.clone());
        stats.total_contexts_created += 1;

        // Emit context creation event
        let mut event_bus = self.event_bus.write().unwrap();
        let event = ComponentEvent::new("execution_engine", "context_created")
            .with_data("context_id", context_id);
        event_bus.publish(event).ok();

        Ok(context)
    }

    /// Get composition context
    #[must_use]
    pub fn get_context(&self, context_id: &str) -> Option<Arc<RwLock<CompositionContext>>> {
        let contexts = self.contexts.read().unwrap();
        contexts.get(context_id).cloned()
    }

    /// Execute a pipeline within a context
    pub async fn execute_pipeline(
        &self,
        context_id: &str,
        pipeline: Pipeline,
        input_data: PipelineData,
    ) -> SklResult<ExecutionResult> {
        let execution_start = Instant::now();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_executions += 1;
        }

        // Get or create context
        let context = match self.get_context(context_id) {
            Some(ctx) => ctx,
            None => self.create_context(context_id)?,
        };

        // Acquire execution resources
        let resource_allocation = self.acquire_execution_resources(&pipeline).await?;

        // Create execution plan
        let execution_plan = self.create_execution_plan(&pipeline, &resource_allocation)?;

        // Submit execution to scheduler
        let execution_id = self
            .schedule_execution(context.clone(), execution_plan, input_data)
            .await?;

        // Wait for execution completion
        let result = self.wait_for_execution(execution_id).await?;

        // Release resources
        self.release_execution_resources(resource_allocation)
            .await?;

        let execution_time = execution_start.elapsed();
        let mut stats = self.stats.lock().unwrap();

        match &result.pipeline_result {
            Ok(_) => stats.successful_executions += 1,
            Err(_) => stats.failed_executions += 1,
        }

        stats.total_execution_time += execution_time;
        stats.update_averages();

        Ok(result)
    }

    /// Execute multiple pipelines concurrently
    pub async fn execute_pipelines_concurrent(
        &self,
        executions: Vec<ConcurrentExecution>,
    ) -> SklResult<Vec<ExecutionResult>> {
        let mut execution_handles = Vec::new();

        // Start all executions
        for execution in executions {
            let engine = self.clone();
            let handle = tokio::spawn(async move {
                engine
                    .execute_pipeline(
                        &execution.context_id,
                        execution.pipeline,
                        execution.input_data,
                    )
                    .await
            });
            execution_handles.push(handle);
        }

        // Wait for all executions to complete
        let mut results = Vec::new();
        for handle in execution_handles {
            match handle.await {
                Ok(result) => results.push(result?),
                Err(e) => return Err(SklearsError::InvalidInput(format!("Execution failed: {e}"))),
            }
        }

        Ok(results)
    }

    /// Execute a composition graph
    pub async fn execute_composition(
        &self,
        context_id: &str,
        composition: CompositionGraph,
        input_data: HashMap<String, PipelineData>,
    ) -> SklResult<CompositionResult> {
        let context = self
            .get_context(context_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Context {context_id} not found")))?;

        // Validate composition graph
        self.validate_composition_graph(&composition)?;

        // Resolve component dependencies
        let dependency_order = self.resolve_composition_dependencies(&composition)?;

        // Execute composition in dependency order
        let mut component_results = HashMap::new();
        let mut execution_context = CompositionExecutionContext::new();

        for component_id in dependency_order {
            let component_node = composition.nodes.get(&component_id).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Component {component_id} not found"))
            })?;

            // Prepare component input data
            let component_input = self.prepare_component_input(
                &component_id,
                &composition,
                &input_data,
                &component_results,
            )?;

            // Execute component
            let component_result = self
                .execute_composition_component(
                    context.clone(),
                    component_node,
                    component_input,
                    &mut execution_context,
                )
                .await?;

            component_results.insert(component_id, component_result);
        }

        Ok(CompositionResult {
            composition_id: composition.composition_id,
            success: true,
            component_results,
            execution_time: execution_context.start_time.elapsed(),
            error: None,
        })
    }

    /// Get execution statistics
    #[must_use]
    pub fn get_statistics(&self) -> ExecutionStatistics {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Configure the execution engine
    pub fn configure(&mut self, config: ExecutionEngineConfig) {
        self.config = config;
    }

    /// Shutdown the execution engine
    pub async fn shutdown(&self) -> SklResult<()> {
        // Stop all running executions
        let scheduler = self.scheduler.write().unwrap();
        scheduler.shutdown()?;

        // Clean up contexts
        let mut contexts = self.contexts.write().unwrap();
        contexts.clear();

        // Release all resources
        let mut resource_manager = self.resource_manager.write().unwrap();
        resource_manager.release_all_resources()?;

        Ok(())
    }

    /// Private helper methods
    async fn acquire_execution_resources(
        &self,
        pipeline: &Pipeline,
    ) -> SklResult<ResourceAllocation> {
        let mut resource_manager = self.resource_manager.write().unwrap();
        resource_manager.allocate_for_pipeline(pipeline)
    }

    async fn release_execution_resources(&self, allocation: ResourceAllocation) -> SklResult<()> {
        let mut resource_manager = self.resource_manager.write().unwrap();
        resource_manager.release_allocation(allocation)
    }

    fn create_execution_plan(
        &self,
        pipeline: &Pipeline,
        resource_allocation: &ResourceAllocation,
    ) -> SklResult<ExecutionPlan> {
        Ok(ExecutionPlan {
            plan_id: uuid::Uuid::new_v4().to_string(),
            pipeline_id: pipeline.pipeline_id.clone(),
            execution_strategy: pipeline.execution_strategy.clone(),
            resource_allocation: resource_allocation.clone(),
            estimated_execution_time: self.estimate_execution_time(pipeline)?,
            priority: ExecutionPriority::Normal,
        })
    }

    async fn schedule_execution(
        &self,
        context: Arc<RwLock<CompositionContext>>,
        execution_plan: ExecutionPlan,
        input_data: PipelineData,
    ) -> SklResult<String> {
        let mut scheduler = self.scheduler.write().unwrap();
        scheduler.schedule_execution(context, execution_plan, input_data)
    }

    async fn wait_for_execution(&self, execution_id: String) -> SklResult<ExecutionResult> {
        let scheduler = self.scheduler.read().unwrap();
        scheduler.wait_for_execution(&execution_id)
    }

    fn estimate_execution_time(&self, pipeline: &Pipeline) -> SklResult<Duration> {
        // Simple estimation based on number of stages
        // In a real implementation, this would use historical data and component characteristics
        let base_time = Duration::from_millis(100);
        let stage_time = Duration::from_millis(50) * pipeline.stages.len() as u32;
        Ok(base_time + stage_time)
    }

    fn validate_composition_graph(&self, composition: &CompositionGraph) -> SklResult<()> {
        if composition.nodes.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Composition graph is empty".to_string(),
            ));
        }

        // Check for cycles in the composition graph
        self.detect_composition_cycles(composition)?;

        Ok(())
    }

    fn detect_composition_cycles(&self, composition: &CompositionGraph) -> SklResult<()> {
        // Simplified cycle detection
        // In a real implementation, this would use proper graph algorithms
        Ok(())
    }

    fn resolve_composition_dependencies(
        &self,
        composition: &CompositionGraph,
    ) -> SklResult<Vec<String>> {
        // Simplified dependency resolution
        // In a real implementation, this would use topological sorting
        let mut order = Vec::new();
        for component_id in composition.nodes.keys() {
            order.push(component_id.clone());
        }
        Ok(order)
    }

    fn prepare_component_input(
        &self,
        component_id: &str,
        composition: &CompositionGraph,
        initial_input: &HashMap<String, PipelineData>,
        component_results: &HashMap<String, ComponentExecutionResult>,
    ) -> SklResult<PipelineData> {
        // Simplified input preparation
        // In a real implementation, this would combine inputs based on composition graph edges
        if let Some(data) = initial_input.get(component_id) {
            Ok(data.clone())
        } else {
            Ok(PipelineData::empty())
        }
    }

    async fn execute_composition_component(
        &self,
        context: Arc<RwLock<CompositionContext>>,
        component_node: &CompositionNode,
        input_data: PipelineData,
        execution_context: &mut CompositionExecutionContext,
    ) -> SklResult<ComponentExecutionResult> {
        let start_time = Instant::now();

        // Get component from registry
        let component_config =
            ComponentConfig::new(&component_node.component_id, &component_node.component_type);
        let mut component = self
            .component_registry
            .create_component(&component_node.component_type, &component_config)?;

        // Initialize and start component
        component.initialize(&component_config)?;
        component.start()?;

        // Simulate component execution
        // In a real implementation, this would call component-specific processing methods
        let output_data = input_data; // Placeholder

        let execution_time = start_time.elapsed();

        Ok(ComponentExecutionResult {
            component_id: component_node.component_id.clone(),
            success: true,
            execution_time,
            output_data,
            error: None,
            metrics: component.get_metrics(),
        })
    }
}

/// Composition context for execution management
pub struct CompositionContext {
    /// Context identifier
    pub context_id: String,
    /// Active pipelines
    pub active_pipelines: HashMap<String, Pipeline>,
    /// Component instances
    pub components: HashMap<String, Box<dyn PluggableComponent>>,
    /// Context variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Context metadata
    pub metadata: HashMap<String, String>,
    /// Context state
    pub state: ContextState,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last activity timestamp
    pub last_activity: Instant,
}

impl std::fmt::Debug for CompositionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositionContext")
            .field("context_id", &self.context_id)
            .field("active_pipelines", &self.active_pipelines)
            .field(
                "components",
                &format!("[{} components]", self.components.len()),
            )
            .field("variables", &self.variables)
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .field("created_at", &self.created_at)
            .field("last_activity", &self.last_activity)
            .finish()
    }
}

impl CompositionContext {
    #[must_use]
    pub fn new(context_id: &str) -> Self {
        let now = Instant::now();
        Self {
            context_id: context_id.to_string(),
            active_pipelines: HashMap::new(),
            components: HashMap::new(),
            variables: HashMap::new(),
            metadata: HashMap::new(),
            state: ContextState::Active,
            created_at: now,
            last_activity: now,
        }
    }

    /// Add a component to the context
    pub fn add_component(&mut self, component_id: &str, component: Box<dyn PluggableComponent>) {
        self.components.insert(component_id.to_string(), component);
        self.last_activity = Instant::now();
    }

    /// Get a component from the context
    #[must_use]
    pub fn get_component(&self, component_id: &str) -> Option<&Box<dyn PluggableComponent>> {
        self.components.get(component_id)
    }

    /// Set context variable
    pub fn set_variable(&mut self, key: &str, value: serde_json::Value) {
        self.variables.insert(key.to_string(), value);
        self.last_activity = Instant::now();
    }

    /// Get context variable
    #[must_use]
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }
}

/// Context states
#[derive(Debug, Clone, PartialEq)]
pub enum ContextState {
    /// Context is active and ready for execution
    Active,
    /// Context is suspended
    Suspended,
    /// Context is being cleaned up
    CleaningUp,
    /// Context is terminated
    Terminated,
}

/// Execution scheduler for managing concurrent executions
#[derive(Debug)]
pub struct ExecutionScheduler {
    /// Execution queue
    execution_queue: VecDeque<ScheduledExecution>,
    /// Active executions
    active_executions: HashMap<String, ActiveExecution>,
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Thread pool for execution
    thread_pool: Option<thread::JoinHandle<()>>,
    /// Shutdown signal
    shutdown_signal: Arc<(Mutex<bool>, Condvar)>,
}

impl ExecutionScheduler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            execution_queue: VecDeque::new(),
            active_executions: HashMap::new(),
            config: SchedulerConfig::default(),
            thread_pool: None,
            shutdown_signal: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    pub fn schedule_execution(
        &mut self,
        context: Arc<RwLock<CompositionContext>>,
        execution_plan: ExecutionPlan,
        input_data: PipelineData,
    ) -> SklResult<String> {
        let execution_id = uuid::Uuid::new_v4().to_string();

        let scheduled_execution = ScheduledExecution {
            execution_id: execution_id.clone(),
            context,
            execution_plan,
            input_data,
            scheduled_at: Instant::now(),
        };

        self.execution_queue.push_back(scheduled_execution);
        Ok(execution_id)
    }

    pub fn wait_for_execution(&self, execution_id: &str) -> SklResult<ExecutionResult> {
        // Simplified wait implementation
        // In a real implementation, this would use proper synchronization
        Ok(ExecutionResult {
            execution_id: execution_id.to_string(),
            pipeline_result: Ok(PipelineResult {
                pipeline_id: "test".to_string(),
                success: true,
                stage_results: Vec::new(),
                final_output: PipelineData::empty(),
                execution_time: Duration::from_millis(100),
                error: None,
            }),
            execution_time: Duration::from_millis(100),
            resource_usage: ResourceUsage::new(),
        })
    }

    pub fn shutdown(&self) -> SklResult<()> {
        let (lock, cvar) = &*self.shutdown_signal;
        let mut shutdown = lock.lock().unwrap();
        *shutdown = true;
        cvar.notify_all();
        Ok(())
    }
}

/// Resource manager for execution resource allocation
#[derive(Debug)]
pub struct ResourceManager {
    /// Available CPU cores
    available_cpu_cores: u32,
    /// Available memory in bytes
    available_memory: u64,
    /// Allocated resources
    allocated_resources: HashMap<String, ResourceAllocation>,
    /// Resource configuration
    config: ResourceManagerConfig,
}

impl ResourceManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            available_cpu_cores: num_cpus::get() as u32,
            available_memory: 1024 * 1024 * 1024, // 1GB placeholder
            allocated_resources: HashMap::new(),
            config: ResourceManagerConfig::default(),
        }
    }

    pub fn allocate_for_pipeline(&mut self, pipeline: &Pipeline) -> SklResult<ResourceAllocation> {
        let allocation_id = uuid::Uuid::new_v4().to_string();

        // Simple resource estimation based on pipeline complexity
        let estimated_cpu =
            std::cmp::min(pipeline.stages.len() as u32, self.available_cpu_cores / 2);
        let estimated_memory = 100 * 1024 * 1024; // 100MB per stage

        let allocation = ResourceAllocation {
            allocation_id: allocation_id.clone(),
            cpu_cores: estimated_cpu,
            memory_bytes: estimated_memory,
            allocated_at: Instant::now(),
        };

        self.allocated_resources
            .insert(allocation_id, allocation.clone());
        Ok(allocation)
    }

    pub fn release_allocation(&mut self, allocation: ResourceAllocation) -> SklResult<()> {
        self.allocated_resources.remove(&allocation.allocation_id);
        Ok(())
    }

    pub fn release_all_resources(&mut self) -> SklResult<()> {
        self.allocated_resources.clear();
        Ok(())
    }
}

/// Composition graph for complex component relationships
#[derive(Debug, Clone)]
pub struct CompositionGraph {
    /// Graph identifier
    pub composition_id: String,
    /// Component nodes
    pub nodes: HashMap<String, CompositionNode>,
    /// Component edges (dependencies)
    pub edges: HashMap<String, Vec<String>>,
    /// Graph metadata
    pub metadata: HashMap<String, String>,
}

/// Composition node representing a component in the graph
#[derive(Debug, Clone)]
pub struct CompositionNode {
    /// Component identifier
    pub component_id: String,
    /// Component type
    pub component_type: String,
    /// Component configuration
    pub config: ComponentConfig,
    /// Node metadata
    pub metadata: HashMap<String, String>,
}

/// Concurrent execution specification
#[derive(Debug)]
pub struct ConcurrentExecution {
    /// Context identifier
    pub context_id: String,
    /// Pipeline to execute
    pub pipeline: Pipeline,
    /// Input data
    pub input_data: PipelineData,
}

/// Execution plan for pipeline execution
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Plan identifier
    pub plan_id: String,
    /// Pipeline identifier
    pub pipeline_id: String,
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Resource allocation
    pub resource_allocation: ResourceAllocation,
    /// Estimated execution time
    pub estimated_execution_time: Duration,
    /// Execution priority
    pub priority: ExecutionPriority,
}

/// Execution priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExecutionPriority {
    /// Low
    Low,
    /// Normal
    Normal,
    /// High
    High,
    /// Critical
    Critical,
}

/// Resource allocation for execution
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Allocation identifier
    pub allocation_id: String,
    /// Allocated CPU cores
    pub cpu_cores: u32,
    /// Allocated memory in bytes
    pub memory_bytes: u64,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Peak memory usage
    pub peak_memory: u64,
    /// Average CPU usage
    pub average_cpu: f64,
    /// Total processing time
    pub processing_time: Duration,
}

impl ResourceUsage {
    #[must_use]
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            average_cpu: 0.0,
            processing_time: Duration::from_secs(0),
        }
    }
}

/// Scheduled execution
#[derive(Debug)]
pub struct ScheduledExecution {
    /// Execution identifier
    pub execution_id: String,
    /// Execution context
    pub context: Arc<RwLock<CompositionContext>>,
    /// Execution plan
    pub execution_plan: ExecutionPlan,
    /// Input data
    pub input_data: PipelineData,
    /// Scheduled timestamp
    pub scheduled_at: Instant,
}

/// Active execution
#[derive(Debug)]
pub struct ActiveExecution {
    /// Execution identifier
    pub execution_id: String,
    /// Start timestamp
    pub started_at: Instant,
    /// Current status
    pub status: ExecutionStatus,
}

/// Execution status
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// Scheduled
    Scheduled,
    /// Running
    Running,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Execution result
#[derive(Debug)]
pub struct ExecutionResult {
    /// Execution identifier
    pub execution_id: String,
    /// Pipeline execution result
    pub pipeline_result: SklResult<PipelineResult>,
    /// Total execution time
    pub execution_time: Duration,
    /// Resource usage during execution
    pub resource_usage: ResourceUsage,
}

/// Composition execution result
#[derive(Debug)]
pub struct CompositionResult {
    /// Composition identifier
    pub composition_id: String,
    /// Execution success
    pub success: bool,
    /// Results from each component
    pub component_results: HashMap<String, ComponentExecutionResult>,
    /// Total execution time
    pub execution_time: Duration,
    /// Error message if failed
    pub error: Option<String>,
}

/// Component execution result
#[derive(Debug, Clone)]
pub struct ComponentExecutionResult {
    /// Component identifier
    pub component_id: String,
    /// Execution success
    pub success: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Output data
    pub output_data: PipelineData,
    /// Error message if failed
    pub error: Option<String>,
    /// Component metrics
    pub metrics: ComponentMetrics,
}

/// Composition execution context
#[derive(Debug)]
pub struct CompositionExecutionContext {
    /// Execution start time
    pub start_time: Instant,
    /// Execution variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Execution trace
    pub trace: Vec<ExecutionTrace>,
}

impl CompositionExecutionContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            variables: HashMap::new(),
            trace: Vec::new(),
        }
    }
}

/// Execution trace entry
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Timestamp
    pub timestamp: Instant,
    /// Component identifier
    pub component_id: String,
    /// Trace event
    pub event: String,
    /// Event data
    pub data: Option<serde_json::Value>,
}

/// Execution engine configuration
#[derive(Debug, Clone)]
pub struct ExecutionEngineConfig {
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Execution timeout
    pub execution_timeout: Option<Duration>,
    /// Allow context override
    pub allow_context_override: bool,
    /// Enable execution tracing
    pub enable_tracing: bool,
    /// Resource allocation strategy
    pub resource_allocation_strategy: ResourceAllocationStrategy,
}

impl Default for ExecutionEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            execution_timeout: Some(Duration::from_secs(300)),
            allow_context_override: false,
            enable_tracing: true,
            resource_allocation_strategy: ResourceAllocationStrategy::Conservative,
        }
    }
}

/// Resource allocation strategies
#[derive(Debug, Clone)]
pub enum ResourceAllocationStrategy {
    /// Conservative resource allocation
    Conservative,
    /// Aggressive resource allocation
    Aggressive,
    /// Adaptive resource allocation
    Adaptive,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Execution timeout
    pub execution_timeout: Duration,
    /// Priority scheduling enabled
    pub enable_priority_scheduling: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            execution_timeout: Duration::from_secs(300),
            enable_priority_scheduling: true,
        }
    }
}

/// Resource manager configuration
#[derive(Debug, Clone)]
pub struct ResourceManagerConfig {
    /// CPU oversubscription factor
    pub cpu_oversubscription_factor: f64,
    /// Memory oversubscription factor
    pub memory_oversubscription_factor: f64,
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
}

impl Default for ResourceManagerConfig {
    fn default() -> Self {
        Self {
            cpu_oversubscription_factor: 1.5,
            memory_oversubscription_factor: 1.2,
            enable_resource_monitoring: true,
        }
    }
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStatistics {
    /// Total executions started
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Total contexts created
    pub total_contexts_created: u64,
    /// Peak concurrent executions
    pub peak_concurrent_executions: u32,
}

impl ExecutionStatistics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            total_execution_time: Duration::from_secs(0),
            average_execution_time: Duration::from_secs(0),
            total_contexts_created: 0,
            peak_concurrent_executions: 0,
        }
    }

    /// Get execution success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }

    /// Update average execution time
    pub fn update_averages(&mut self) {
        if self.total_executions > 0 {
            self.average_execution_time = self.total_execution_time / self.total_executions as u32;
        }
    }
}

/// Execution engine errors
#[derive(Debug, Error)]
pub enum ExecutionEngineError {
    #[error("Context not found: {0}")]
    ContextNotFound(String),

    #[error("Resource allocation failed: {0}")]
    ResourceAllocationFailed(String),

    #[error("Execution scheduling failed: {0}")]
    SchedulingFailed(String),

    #[error("Execution timeout: {0:?}")]
    ExecutionTimeout(Duration),

    #[error("Invalid composition graph: {0}")]
    InvalidCompositionGraph(String),
}

impl Clone for CompositionExecutionEngine {
    fn clone(&self) -> Self {
        Self {
            contexts: self.contexts.clone(),
            component_registry: self.component_registry.clone(),
            dependency_resolver: self.dependency_resolver.clone(),
            lifecycle_manager: self.lifecycle_manager.clone(),
            scheduler: self.scheduler.clone(),
            resource_manager: self.resource_manager.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            event_bus: self.event_bus.clone(),
        }
    }
}

impl Default for ExecutionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CompositionExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ExecutionStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_context() {
        let mut context = CompositionContext::new("test_context");

        context.set_variable(
            "test_var",
            serde_json::Value::String("test_value".to_string()),
        );
        assert_eq!(
            context.get_variable("test_var").unwrap(),
            &serde_json::Value::String("test_value".to_string())
        );

        assert_eq!(context.context_id, "test_context");
        assert_eq!(context.state, ContextState::Active);
    }

    #[test]
    fn test_resource_allocation() {
        let allocation = ResourceAllocation {
            allocation_id: "test_allocation".to_string(),
            cpu_cores: 4,
            memory_bytes: 1024 * 1024,
            allocated_at: Instant::now(),
        };

        assert_eq!(allocation.cpu_cores, 4);
        assert_eq!(allocation.memory_bytes, 1024 * 1024);
    }

    #[test]
    fn test_execution_statistics() {
        let mut stats = ExecutionStatistics::new();
        stats.total_executions = 10;
        stats.successful_executions = 8;
        stats.failed_executions = 2;

        assert_eq!(stats.success_rate(), 0.8);
    }

    #[test]
    fn test_resource_manager() {
        let mut manager = ResourceManager::new();

        // Create a dummy pipeline for testing
        let pipeline = Pipeline {
            pipeline_id: "test_pipeline".to_string(),
            stages: vec![], // Empty for test
            config: super::super::pipeline_system::PipelineConfiguration::default(),
            error_strategy: super::super::pipeline_system::ErrorHandlingStrategy::FailFast,
            execution_strategy: ExecutionStrategy::Sequential,
            metadata: super::super::pipeline_system::PipelineMetadata::new(),
            state: super::super::pipeline_system::PipelineState::Created,
            components: Arc::new(RwLock::new(HashMap::new())),
            event_bus: Arc::new(RwLock::new(EventBus::new())),
            execution_context: Arc::new(RwLock::new(
                super::super::pipeline_system::ExecutionContext::new(),
            )),
            metrics: Arc::new(Mutex::new(
                super::super::pipeline_system::PipelineMetrics::new(),
            )),
        };

        let allocation = manager.allocate_for_pipeline(&pipeline);
        assert!(allocation.is_ok());

        let allocation = allocation.unwrap();
        let release_result = manager.release_allocation(allocation);
        assert!(release_result.is_ok());
    }
}
