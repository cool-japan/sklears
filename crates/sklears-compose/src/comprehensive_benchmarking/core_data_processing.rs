use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;

/// Core data transformation engine providing fundamental processing capabilities,
/// pipeline orchestration, and stage management for complex data workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformationEngine {
    /// Data processing pipelines for structured transformation workflows
    processing_pipelines: HashMap<String, DataProcessingPipeline>,
    /// Pipeline execution engine
    execution_engine: PipelineExecutionEngine,
    /// Stage factory for creating processing stages
    stage_factory: ProcessingStageFactory,
    /// Resource manager for pipeline execution
    resource_manager: Arc<RwLock<PipelineResourceManager>>,
    /// Dependency resolver for pipeline orchestration
    dependency_resolver: DependencyResolver,
}

impl DataTransformationEngine {
    /// Create a new data transformation engine with default configuration
    pub fn new() -> Self {
        Self {
            processing_pipelines: HashMap::new(),
            execution_engine: PipelineExecutionEngine::new(),
            stage_factory: ProcessingStageFactory::new(),
            resource_manager: Arc::new(RwLock::new(PipelineResourceManager::new())),
            dependency_resolver: DependencyResolver::new(),
        }
    }

    /// Register a new processing pipeline
    pub fn register_pipeline(&mut self, pipeline: DataProcessingPipeline) -> Result<(), ProcessingError> {
        // Validate pipeline configuration
        self.validate_pipeline(&pipeline)?;

        // Register dependencies
        self.dependency_resolver.register_dependencies(&pipeline)?;

        // Store pipeline
        self.processing_pipelines.insert(pipeline.pipeline_id.clone(), pipeline);

        Ok(())
    }

    /// Execute a processing pipeline
    pub async fn execute_pipeline(&self, pipeline_id: &str, input_data: TransformationData) -> Result<TransformationData, ProcessingError> {
        let pipeline = self.processing_pipelines.get(pipeline_id)
            .ok_or_else(|| ProcessingError::PipelineNotFound(pipeline_id.to_string()))?;

        // Check dependencies
        self.dependency_resolver.check_dependencies(pipeline_id)?;

        // Execute pipeline through execution engine
        self.execution_engine.execute(pipeline, input_data).await
    }

    /// Get pipeline status
    pub fn get_pipeline_status(&self, pipeline_id: &str) -> Option<PipelineStatus> {
        self.execution_engine.get_status(pipeline_id)
    }

    /// Validate pipeline configuration
    fn validate_pipeline(&self, pipeline: &DataProcessingPipeline) -> Result<(), ProcessingError> {
        // Validate stages
        for stage in &pipeline.stages {
            self.validate_stage(stage)?;
        }

        // Validate pipeline dependencies
        self.validate_pipeline_dependencies(pipeline)?;

        Ok(())
    }

    /// Validate individual processing stage
    fn validate_stage(&self, stage: &ProcessingStage) -> Result<(), ProcessingError> {
        // Validate stage configuration
        stage.configuration.validate()?;

        // Validate input/output compatibility
        self.validate_stage_compatibility(stage)?;

        Ok(())
    }

    /// Validate stage input/output compatibility
    fn validate_stage_compatibility(&self, stage: &ProcessingStage) -> Result<(), ProcessingError> {
        // Check input requirements against available data
        stage.input_requirements.validate()?;

        // Check output specifications
        stage.output_specifications.validate()?;

        Ok(())
    }

    /// Validate pipeline dependencies
    fn validate_pipeline_dependencies(&self, pipeline: &DataProcessingPipeline) -> Result<(), ProcessingError> {
        for dependency in &pipeline.metadata.dependencies {
            // Check if dependency pipeline exists
            if !self.processing_pipelines.contains_key(&dependency.pipeline_id) {
                return Err(ProcessingError::DependencyNotFound(dependency.pipeline_id.clone()));
            }
        }

        Ok(())
    }
}

/// Data processing pipeline configuration for complex
/// multi-stage data transformation workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessingPipeline {
    /// Unique pipeline identifier
    pub pipeline_id: String,
    /// Human-readable pipeline name
    pub pipeline_name: String,
    /// Pipeline description and documentation
    pub description: String,
    /// Ordered list of processing stages
    pub stages: Vec<ProcessingStage>,
    /// Pipeline configuration settings
    pub configuration: PipelineConfiguration,
    /// Input data schema specification
    pub input_schema: DataSchema,
    /// Output data schema specification
    pub output_schema: DataSchema,
    /// Pipeline metadata and versioning
    pub metadata: PipelineMetadata,
}

impl DataProcessingPipeline {
    /// Create a new pipeline with basic configuration
    pub fn new(id: String, name: String, description: String) -> Self {
        Self {
            pipeline_id: id,
            pipeline_name: name,
            description,
            stages: Vec::new(),
            configuration: PipelineConfiguration::default(),
            input_schema: DataSchema::default(),
            output_schema: DataSchema::default(),
            metadata: PipelineMetadata::default(),
        }
    }

    /// Add a processing stage to the pipeline
    pub fn add_stage(&mut self, stage: ProcessingStage) {
        self.stages.push(stage);
    }

    /// Get total estimated processing time
    pub fn estimate_processing_time(&self) -> Duration {
        self.stages.iter()
            .map(|stage| stage.estimate_execution_time())
            .sum()
    }

    /// Get pipeline complexity score
    pub fn complexity_score(&self) -> f64 {
        let stage_complexity: f64 = self.stages.iter()
            .map(|stage| stage.complexity_score())
            .sum();

        let dependency_complexity = self.metadata.dependencies.len() as f64 * 0.1;

        stage_complexity + dependency_complexity
    }
}

/// Individual processing stage configuration
/// for granular transformation control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStage {
    /// Unique stage identifier
    pub stage_id: String,
    /// Stage name and description
    pub stage_name: String,
    /// Type of processing operation
    pub stage_type: ProcessingStageType,
    /// Stage-specific configuration
    pub configuration: StageConfiguration,
    /// Input requirements for this stage
    pub input_requirements: StageInputRequirements,
    /// Output specifications for this stage
    pub output_specifications: StageOutputSpecifications,
    /// Error handling configuration
    pub error_handling: StageErrorHandling,
    /// Performance settings for the stage
    pub performance_settings: StagePerformanceSettings,
}

impl ProcessingStage {
    /// Create a new processing stage
    pub fn new(id: String, name: String, stage_type: ProcessingStageType) -> Self {
        Self {
            stage_id: id,
            stage_name: name,
            stage_type,
            configuration: StageConfiguration::default(),
            input_requirements: StageInputRequirements::default(),
            output_specifications: StageOutputSpecifications::default(),
            error_handling: StageErrorHandling::default(),
            performance_settings: StagePerformanceSettings::default(),
        }
    }

    /// Estimate execution time for this stage
    pub fn estimate_execution_time(&self) -> Duration {
        match self.stage_type {
            ProcessingStageType::Validation => Duration::from_millis(100),
            ProcessingStageType::Cleaning => Duration::from_millis(500),
            ProcessingStageType::Conversion => Duration::from_millis(200),
            ProcessingStageType::Aggregation => Duration::from_millis(800),
            ProcessingStageType::Enrichment => Duration::from_millis(1000),
            ProcessingStageType::Statistical => Duration::from_millis(1500),
            ProcessingStageType::Filtering => Duration::from_millis(300),
            ProcessingStageType::Sorting => Duration::from_millis(600),
            ProcessingStageType::Grouping => Duration::from_millis(400),
            ProcessingStageType::Joining => Duration::from_millis(1200),
            ProcessingStageType::Custom(_) => Duration::from_millis(1000),
        }
    }

    /// Calculate complexity score for this stage
    pub fn complexity_score(&self) -> f64 {
        let base_complexity = match self.stage_type {
            ProcessingStageType::Validation => 1.0,
            ProcessingStageType::Cleaning => 2.5,
            ProcessingStageType::Conversion => 1.5,
            ProcessingStageType::Aggregation => 3.0,
            ProcessingStageType::Enrichment => 4.0,
            ProcessingStageType::Statistical => 5.0,
            ProcessingStageType::Filtering => 1.2,
            ProcessingStageType::Sorting => 2.0,
            ProcessingStageType::Grouping => 2.2,
            ProcessingStageType::Joining => 3.5,
            ProcessingStageType::Custom(_) => 3.0,
        };

        // Adjust for configuration complexity
        let config_multiplier = 1.0 + (self.configuration.parameters.len() as f64 * 0.1);

        base_complexity * config_multiplier
    }

    /// Validate stage configuration
    pub fn validate(&self) -> Result<(), ProcessingError> {
        self.configuration.validate()?;
        self.input_requirements.validate()?;
        self.output_specifications.validate()?;
        self.error_handling.validate()?;
        Ok(())
    }
}

/// Processing stage type enumeration for
/// different transformation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStageType {
    /// Data validation and quality checks
    Validation,
    /// Data cleaning and preprocessing
    Cleaning,
    /// Data format conversion
    Conversion,
    /// Data aggregation and summarization
    Aggregation,
    /// Data enrichment and augmentation
    Enrichment,
    /// Statistical analysis and computation
    Statistical,
    /// Data filtering and selection
    Filtering,
    /// Data sorting and ordering
    Sorting,
    /// Data grouping and categorization
    Grouping,
    /// Data joining and merging
    Joining,
    /// Custom processing implementation
    Custom(String),
}

/// Pipeline execution engine for orchestrating
/// complex multi-stage data processing workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineExecutionEngine {
    /// Currently executing pipelines
    active_executions: Arc<RwLock<HashMap<String, PipelineExecution>>>,
    /// Execution scheduler
    scheduler: ExecutionScheduler,
    /// Stage executor pool
    stage_executor_pool: StageExecutorPool,
}

impl PipelineExecutionEngine {
    /// Create a new execution engine
    pub fn new() -> Self {
        Self {
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            scheduler: ExecutionScheduler::new(),
            stage_executor_pool: StageExecutorPool::new(),
        }
    }

    /// Execute a pipeline asynchronously
    pub async fn execute(&self, pipeline: &DataProcessingPipeline, input_data: TransformationData) -> Result<TransformationData, ProcessingError> {
        let execution_id = uuid::Uuid::new_v4().to_string();

        // Create execution context
        let execution = PipelineExecution::new(
            execution_id.clone(),
            pipeline.clone(),
            input_data,
        );

        // Register execution
        {
            let mut executions = self.active_executions.write().unwrap();
            executions.insert(execution_id.clone(), execution);
        }

        // Schedule execution
        let result = self.scheduler.schedule_execution(&execution_id, pipeline).await;

        // Remove execution from active list
        {
            let mut executions = self.active_executions.write().unwrap();
            executions.remove(&execution_id);
        }

        result
    }

    /// Get execution status
    pub fn get_status(&self, pipeline_id: &str) -> Option<PipelineStatus> {
        let executions = self.active_executions.read().unwrap();
        executions.values()
            .find(|exec| exec.pipeline.pipeline_id == pipeline_id)
            .map(|exec| exec.status.clone())
    }
}

/// Pipeline execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineExecution {
    /// Unique execution identifier
    pub execution_id: String,
    /// Pipeline being executed
    pub pipeline: DataProcessingPipeline,
    /// Input data for processing
    pub input_data: TransformationData,
    /// Current execution status
    pub status: PipelineStatus,
    /// Execution start time
    pub start_time: DateTime<Utc>,
    /// Current stage index
    pub current_stage: usize,
    /// Intermediate results
    pub intermediate_data: Option<TransformationData>,
}

impl PipelineExecution {
    /// Create a new pipeline execution
    pub fn new(execution_id: String, pipeline: DataProcessingPipeline, input_data: TransformationData) -> Self {
        Self {
            execution_id,
            pipeline,
            input_data,
            status: PipelineStatus::Pending,
            start_time: Utc::now(),
            current_stage: 0,
            intermediate_data: None,
        }
    }
}

/// Pipeline execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStatus {
    /// Pipeline is queued for execution
    Pending,
    /// Pipeline is currently running
    Running {
        current_stage: usize,
        progress: f64,
    },
    /// Pipeline completed successfully
    Completed {
        duration: Duration,
        stages_completed: usize,
    },
    /// Pipeline failed with error
    Failed {
        error: String,
        failed_stage: usize,
    },
    /// Pipeline was cancelled
    Cancelled {
        reason: String,
    },
}

/// Execution scheduler for managing pipeline execution order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionScheduler {
    /// Scheduling configuration
    config: SchedulerConfig,
    /// Priority queue for pipeline execution
    execution_queue: Arc<Mutex<Vec<ScheduledExecution>>>,
}

impl ExecutionScheduler {
    /// Create a new execution scheduler
    pub fn new() -> Self {
        Self {
            config: SchedulerConfig::default(),
            execution_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Schedule a pipeline execution
    pub async fn schedule_execution(&self, execution_id: &str, pipeline: &DataProcessingPipeline) -> Result<TransformationData, ProcessingError> {
        // Create scheduled execution
        let scheduled = ScheduledExecution {
            execution_id: execution_id.to_string(),
            pipeline: pipeline.clone(),
            priority: self.calculate_priority(pipeline),
            scheduled_time: Utc::now(),
        };

        // Add to queue
        {
            let mut queue = self.execution_queue.lock().unwrap();
            queue.push(scheduled);
            queue.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        }

        // Execute immediately for now (simplified)
        self.execute_immediately(pipeline).await
    }

    /// Calculate execution priority
    fn calculate_priority(&self, pipeline: &DataProcessingPipeline) -> f64 {
        let complexity_factor = 1.0 / (1.0 + pipeline.complexity_score());
        let urgency_factor = 1.0; // Could be based on deadline

        complexity_factor * urgency_factor
    }

    /// Execute pipeline immediately (simplified implementation)
    async fn execute_immediately(&self, pipeline: &DataProcessingPipeline) -> Result<TransformationData, ProcessingError> {
        // Simplified execution - would be more complex in real implementation
        Ok(TransformationData {
            records: Vec::new(),
            metadata: HashMap::new(),
            schema: DataSchema::default(),
            quality_metrics: None,
        })
    }
}

/// Scheduled execution entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledExecution {
    /// Execution identifier
    pub execution_id: String,
    /// Pipeline to execute
    pub pipeline: DataProcessingPipeline,
    /// Execution priority
    pub priority: f64,
    /// Scheduled time
    pub scheduled_time: DateTime<Utc>,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Default execution timeout
    pub default_timeout: Duration,
    /// Priority calculation weights
    pub priority_weights: PriorityWeights,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            default_timeout: Duration::from_secs(3600),
            priority_weights: PriorityWeights::default(),
        }
    }
}

/// Priority calculation weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityWeights {
    /// Weight for complexity factor
    pub complexity_weight: f64,
    /// Weight for urgency factor
    pub urgency_weight: f64,
    /// Weight for resource requirements
    pub resource_weight: f64,
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            complexity_weight: 0.3,
            urgency_weight: 0.5,
            resource_weight: 0.2,
        }
    }
}

/// Stage executor pool for parallel stage execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageExecutorPool {
    /// Pool configuration
    config: PoolConfig,
    /// Available executors
    executors: Arc<RwLock<Vec<StageExecutor>>>,
}

impl StageExecutorPool {
    /// Create a new executor pool
    pub fn new() -> Self {
        Self {
            config: PoolConfig::default(),
            executors: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get an available executor
    pub fn get_executor(&self) -> Option<StageExecutor> {
        let mut executors = self.executors.write().unwrap();
        executors.pop()
    }

    /// Return an executor to the pool
    pub fn return_executor(&self, executor: StageExecutor) {
        let mut executors = self.executors.write().unwrap();
        executors.push(executor);
    }
}

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Minimum pool size
    pub min_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 5,
            max_size: 20,
            min_size: 2,
        }
    }
}

/// Stage executor for individual stage processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageExecutor {
    /// Executor identifier
    pub executor_id: String,
    /// Supported stage types
    pub supported_types: Vec<ProcessingStageType>,
    /// Current status
    pub status: ExecutorStatus,
}

/// Executor status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutorStatus {
    /// Available for work
    Available,
    /// Currently busy
    Busy,
    /// Maintenance mode
    Maintenance,
    /// Failed state
    Failed,
}

/// Processing stage factory for creating stage instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStageFactory {
    /// Stage templates
    templates: HashMap<String, ProcessingStage>,
    /// Factory configuration
    config: FactoryConfig,
}

impl ProcessingStageFactory {
    /// Create a new stage factory
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            config: FactoryConfig::default(),
        }
    }

    /// Create a stage from template
    pub fn create_stage(&self, template_id: &str, stage_id: String) -> Result<ProcessingStage, ProcessingError> {
        let template = self.templates.get(template_id)
            .ok_or_else(|| ProcessingError::TemplateNotFound(template_id.to_string()))?;

        let mut stage = template.clone();
        stage.stage_id = stage_id;

        Ok(stage)
    }

    /// Register a new stage template
    pub fn register_template(&mut self, template_id: String, template: ProcessingStage) {
        self.templates.insert(template_id, template);
    }
}

/// Factory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryConfig {
    /// Default stage settings
    pub default_settings: HashMap<String, ConfigurationValue>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

impl Default for FactoryConfig {
    fn default() -> Self {
        Self {
            default_settings: HashMap::new(),
            validation_rules: Vec::new(),
        }
    }
}

/// Dependency resolver for pipeline orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolver {
    /// Dependency graph
    dependency_graph: HashMap<String, Vec<PipelineDependency>>,
    /// Resolution cache
    resolution_cache: Arc<RwLock<HashMap<String, DependencyResolution>>>,
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new() -> Self {
        Self {
            dependency_graph: HashMap::new(),
            resolution_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register pipeline dependencies
    pub fn register_dependencies(&mut self, pipeline: &DataProcessingPipeline) -> Result<(), ProcessingError> {
        self.dependency_graph.insert(
            pipeline.pipeline_id.clone(),
            pipeline.metadata.dependencies.clone()
        );
        Ok(())
    }

    /// Check pipeline dependencies
    pub fn check_dependencies(&self, pipeline_id: &str) -> Result<(), ProcessingError> {
        if let Some(dependencies) = self.dependency_graph.get(pipeline_id) {
            for dependency in dependencies {
                self.validate_dependency(dependency)?;
            }
        }
        Ok(())
    }

    /// Validate individual dependency
    fn validate_dependency(&self, dependency: &PipelineDependency) -> Result<(), ProcessingError> {
        // Check if dependency is satisfied
        match dependency.dependency_type {
            DependencyType::Required => {
                // Must be available
                if !self.is_dependency_available(&dependency.pipeline_id) {
                    return Err(ProcessingError::DependencyNotAvailable(dependency.pipeline_id.clone()));
                }
            },
            DependencyType::Optional => {
                // Can proceed without it
            },
            DependencyType::Conditional => {
                // Check condition
                if self.should_check_conditional_dependency(dependency) {
                    if !self.is_dependency_available(&dependency.pipeline_id) {
                        return Err(ProcessingError::ConditionalDependencyNotMet(dependency.pipeline_id.clone()));
                    }
                }
            },
        }
        Ok(())
    }

    /// Check if dependency is available
    fn is_dependency_available(&self, pipeline_id: &str) -> bool {
        // Simplified check - would integrate with actual pipeline registry
        self.dependency_graph.contains_key(pipeline_id)
    }

    /// Check if conditional dependency should be evaluated
    fn should_check_conditional_dependency(&self, dependency: &PipelineDependency) -> bool {
        // Simplified logic - would evaluate actual conditions
        true
    }
}

/// Dependency resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolution {
    /// Pipeline identifier
    pub pipeline_id: String,
    /// Resolution status
    pub status: ResolutionStatus,
    /// Resolved dependencies
    pub resolved_dependencies: Vec<String>,
    /// Resolution time
    pub resolution_time: DateTime<Utc>,
}

/// Resolution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStatus {
    /// All dependencies resolved
    Resolved,
    /// Partial resolution
    PartiallyResolved,
    /// Failed to resolve
    Failed,
    /// Resolution in progress
    InProgress,
}

/// Pipeline resource manager for resource allocation and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResourceManager {
    /// Resource allocations
    allocations: HashMap<String, ResourceAllocation>,
    /// Resource limits
    limits: ResourceLimits,
    /// Usage monitoring
    usage_monitor: ResourceUsageMonitor,
}

impl PipelineResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            limits: ResourceLimits::default(),
            usage_monitor: ResourceUsageMonitor::new(),
        }
    }

    /// Allocate resources for pipeline execution
    pub fn allocate_resources(&mut self, pipeline_id: &str, requirements: &ResourceRequirements) -> Result<ResourceAllocation, ProcessingError> {
        // Check if resources are available
        if !self.can_allocate(requirements) {
            return Err(ProcessingError::InsufficientResources);
        }

        // Create allocation
        let allocation = ResourceAllocation {
            pipeline_id: pipeline_id.to_string(),
            cpu_allocation: CpuAllocation {
                cores: requirements.cpu_cores,
                priority: requirements.cpu_priority,
            },
            memory_allocation: MemoryAllocation {
                heap_size: requirements.memory_mb,
                growth_strategy: MemoryGrowthStrategy::Conservative,
            },
            io_allocation: IoAllocation {
                read_bandwidth: requirements.io_bandwidth_mbps,
                write_bandwidth: requirements.io_bandwidth_mbps,
                priority: IoPriority::Normal,
            },
            network_allocation: NetworkAllocation {
                bandwidth: requirements.network_bandwidth_mbps,
                timeout_settings: NetworkTimeoutSettings::default(),
            },
        };

        // Store allocation
        self.allocations.insert(pipeline_id.to_string(), allocation.clone());

        Ok(allocation)
    }

    /// Check if resources can be allocated
    fn can_allocate(&self, requirements: &ResourceRequirements) -> bool {
        let current_usage = self.usage_monitor.get_current_usage();

        current_usage.cpu_usage + requirements.cpu_cores <= self.limits.max_cpu_cores &&
        current_usage.memory_usage + requirements.memory_mb <= self.limits.max_memory_mb &&
        current_usage.io_usage + requirements.io_bandwidth_mbps <= self.limits.max_io_bandwidth
    }

    /// Release resources
    pub fn release_resources(&mut self, pipeline_id: &str) {
        self.allocations.remove(pipeline_id);
    }
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Required CPU cores
    pub cpu_cores: f64,
    /// CPU priority
    pub cpu_priority: CpuPriority,
    /// Required memory in MB
    pub memory_mb: u64,
    /// I/O bandwidth in MB/s
    pub io_bandwidth_mbps: f64,
    /// Network bandwidth in MB/s
    pub network_bandwidth_mbps: f64,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU cores
    pub max_cpu_cores: f64,
    /// Maximum memory in MB
    pub max_memory_mb: u64,
    /// Maximum I/O bandwidth
    pub max_io_bandwidth: f64,
    /// Maximum network bandwidth
    pub max_network_bandwidth: f64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_cores: 16.0,
            max_memory_mb: 32768,
            max_io_bandwidth: 1000.0,
            max_network_bandwidth: 1000.0,
        }
    }
}

/// Resource usage monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageMonitor {
    /// Current usage statistics
    current_usage: ResourceUsage,
    /// Usage history
    usage_history: Vec<ResourceUsageSnapshot>,
}

impl ResourceUsageMonitor {
    /// Create a new usage monitor
    pub fn new() -> Self {
        Self {
            current_usage: ResourceUsage::default(),
            usage_history: Vec::new(),
        }
    }

    /// Get current resource usage
    pub fn get_current_usage(&self) -> &ResourceUsage {
        &self.current_usage
    }

    /// Update usage statistics
    pub fn update_usage(&mut self, usage: ResourceUsage) {
        self.current_usage = usage.clone();
        self.usage_history.push(ResourceUsageSnapshot {
            usage,
            timestamp: Utc::now(),
        });

        // Limit history size
        if self.usage_history.len() > 1000 {
            self.usage_history.drain(0..500);
        }
    }
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage
    pub cpu_usage: f64,
    /// Memory usage in MB
    pub memory_usage: u64,
    /// I/O usage
    pub io_usage: f64,
    /// Network usage
    pub network_usage: f64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            io_usage: 0.0,
            network_usage: 0.0,
        }
    }
}

/// Resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSnapshot {
    /// Usage at this point in time
    pub usage: ResourceUsage,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Transformation data structure for
/// data processing input and output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationData {
    /// Data records
    pub records: Vec<DataRecord>,
    /// Data metadata
    pub metadata: HashMap<String, String>,
    /// Data schema
    pub schema: DataSchema,
    /// Data quality metrics
    pub quality_metrics: Option<HashMap<String, f64>>,
}

/// Data record for individual
/// data entries in transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    /// Record identifier
    pub id: String,
    /// Record fields
    pub fields: HashMap<String, DataValue>,
    /// Record timestamp
    pub timestamp: Option<DateTime<Utc>>,
    /// Record metadata
    pub metadata: HashMap<String, String>,
}

/// Data value for field values
/// in transformation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Date value
    Date(DateTime<Utc>),
    /// Array value
    Array(Vec<DataValue>),
    /// Object value
    Object(HashMap<String, DataValue>),
    /// Null value
    Null,
}

/// Processing error types for comprehensive
/// error handling and debugging
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Pipeline not found: {0}")]
    PipelineNotFound(String),

    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    #[error("Stage execution failed: {0}")]
    StageExecutionFailed(String),

    #[error("Dependency not found: {0}")]
    DependencyNotFound(String),

    #[error("Dependency not available: {0}")]
    DependencyNotAvailable(String),

    #[error("Conditional dependency not met: {0}")]
    ConditionalDependencyNotMet(String),

    #[error("Insufficient resources")]
    InsufficientResources,

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Type alias for processing results
pub type ProcessingResult<T> = Result<T, ProcessingError>;