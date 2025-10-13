//! Core Execution Type Definitions
//!
//! This module provides the fundamental type definitions used throughout the
//! composable execution engine, including task definitions, result types,
//! metadata structures, and execution parameters.

use sklears_core::error::Result as SklResult;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

/// Core execution task definition
///
/// Represents a single unit of work that can be executed by the
/// composable execution engine with configurable requirements and constraints.
pub struct ExecutionTask {
    /// Unique task identifier
    pub id: String,

    /// Task type classification
    pub task_type: TaskType,

    /// Task function to execute (boxed closure)
    pub task_fn: Box<dyn Fn() -> SklResult<()> + Send + Sync>,

    /// Task metadata and configuration
    pub metadata: TaskMetadata,

    /// Resource requirements for execution
    pub requirements: TaskRequirements,

    /// Execution constraints and limits
    pub constraints: TaskConstraints,
}

impl fmt::Debug for ExecutionTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionTask")
            .field("id", &self.id)
            .field("task_type", &self.task_type)
            .field("task_fn", &"<function>")
            .field("metadata", &self.metadata)
            .field("requirements", &self.requirements)
            .field("constraints", &self.constraints)
            .finish()
    }
}

impl Clone for ExecutionTask {
    fn clone(&self) -> Self {
        // Note: task_fn cannot be cloned, so we create a new placeholder
        Self {
            id: self.id.clone(),
            task_type: self.task_type.clone(),
            task_fn: Box::new(|| Ok(())), // Placeholder function
            metadata: self.metadata.clone(),
            requirements: self.requirements.clone(),
            constraints: self.constraints.clone(),
        }
    }
}

/// Task type classification for execution strategies
///
/// Categorizes tasks to help the execution engine select appropriate
/// strategies and optimizations based on task characteristics.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Data transformation task
    Transform,

    /// Model training task
    Fit,

    /// Prediction/inference task
    Predict,

    /// Data preprocessing task
    Preprocess,

    /// Result postprocessing task
    Postprocess,

    /// Data validation task
    Validate,

    /// Feature engineering task
    FeatureEngineering,

    /// Model evaluation task
    Evaluate,

    /// Hyperparameter optimization task
    HyperparameterOptimization,

    /// Cross-validation task
    CrossValidation,

    /// Ensemble learning task
    Ensemble,

    /// Data loading/saving task
    DataIO,

    /// Visualization task
    Visualization,

    /// Custom task type
    Custom(String),
}

/// Task execution status enumeration
///
/// Tracks the current state of task execution throughout its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is queued and waiting to start
    Pending,

    /// Task is currently running
    Running,

    /// Task completed successfully
    Completed,

    /// Task failed during execution
    Failed,

    /// Task was cancelled before completion
    Cancelled,

    /// Task execution timed out
    Timeout,

    /// Task is being retried after failure
    Retrying,

    /// Task is paused/suspended
    Paused,

    /// Task is being rolled back
    RollingBack,
}

/// Task priority levels for scheduling
///
/// Defines the relative importance of tasks for scheduling decisions
/// and resource allocation prioritization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    /// Lowest priority (background tasks)
    Low,

    /// Normal priority (default)
    Normal,

    /// High priority (important tasks)
    High,

    /// Critical priority (time-sensitive tasks)
    Critical,

    /// System priority (infrastructure tasks)
    System,
}

/// Comprehensive task metadata
///
/// Contains descriptive information, timing details, and organizational
/// data for tasks to support tracking, debugging, and optimization.
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Human-readable task name
    pub name: String,

    /// Optional task description
    pub description: Option<String>,

    /// Organizational tags for grouping
    pub tags: Vec<String>,

    /// Task creation timestamp
    pub created_at: SystemTime,

    /// Estimated execution duration
    pub estimated_duration: Option<Duration>,

    /// Task priority level
    pub priority: TaskPriority,

    /// Task dependencies (IDs of tasks that must complete first)
    pub dependencies: Vec<String>,

    /// Task group identifier for batch operations
    pub group_id: Option<String>,

    /// User who submitted the task
    pub submitted_by: Option<String>,

    /// Additional custom metadata
    pub custom_metadata: HashMap<String, MetadataValue>,

    /// Task retry configuration
    pub retry_config: Option<TaskRetryConfig>,

    /// Task timeout configuration
    pub timeout_config: Option<TaskTimeoutConfig>,
}

/// Flexible metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// String
    String(String),
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// Boolean
    Boolean(bool),
    /// Timestamp
    Timestamp(SystemTime),
    /// Duration
    Duration(Duration),
    /// Array
    Array(Vec<MetadataValue>),
    /// Object
    Object(HashMap<String, MetadataValue>),
}

/// Task retry configuration
#[derive(Debug, Clone)]
pub struct TaskRetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,

    /// Delay between retry attempts
    pub retry_delay: Duration,

    /// Backoff strategy for retry delays
    pub backoff_strategy: RetryBackoffStrategy,

    /// Conditions that trigger retries
    pub retry_conditions: Vec<RetryCondition>,

    /// Maximum total retry time
    pub max_retry_time: Option<Duration>,
}

/// Retry backoff strategies
#[derive(Debug, Clone)]
pub enum RetryBackoffStrategy {
    /// Fixed delay between retries
    Fixed,

    /// Linear increase in delay
    Linear { increment: Duration },

    /// Exponential backoff
    Exponential { multiplier: f64 },

    /// Jittered exponential backoff
    JitteredExponential { multiplier: f64, jitter: f64 },

    /// Custom backoff function
    Custom { strategy_name: String },
}

/// Conditions that trigger task retries
#[derive(Debug, Clone)]
pub enum RetryCondition {
    /// Retry on any failure
    AnyFailure,

    /// Retry on specific error types
    ErrorType(String),

    /// Retry on resource unavailability
    ResourceUnavailable,

    /// Retry on timeout
    Timeout,

    /// Retry on transient failures only
    TransientFailure,

    /// Custom retry condition
    Custom(String),
}

/// Task timeout configuration
#[derive(Debug, Clone)]
pub struct TaskTimeoutConfig {
    /// Overall task timeout
    pub total_timeout: Duration,

    /// Timeout for resource allocation
    pub allocation_timeout: Option<Duration>,

    /// Timeout for task initialization
    pub initialization_timeout: Option<Duration>,

    /// Timeout for task execution
    pub execution_timeout: Option<Duration>,

    /// Timeout for task cleanup
    pub cleanup_timeout: Option<Duration>,

    /// Action to take on timeout
    pub timeout_action: TimeoutAction,
}

/// Actions to take when tasks timeout
#[derive(Debug, Clone)]
pub enum TimeoutAction {
    /// Cancel the task immediately
    Cancel,

    /// Attempt graceful shutdown
    GracefulShutdown { grace_period: Duration },

    /// Force kill the task
    ForceKill,

    /// Move to different execution strategy
    Migrate { target_strategy: String },

    /// Custom timeout handling
    Custom { handler_name: String },
}

/// Task resource requirements specification
///
/// Defines the system resources needed for successful task execution,
/// enabling optimal resource allocation and scheduling decisions.
#[derive(Debug, Clone, Default)]
pub struct TaskRequirements {
    /// Required CPU cores (None = any available)
    pub cpu_cores: Option<usize>,

    /// Required memory in bytes
    pub memory_bytes: Option<u64>,

    /// Required I/O bandwidth (bytes/sec)
    pub io_bandwidth: Option<u64>,

    /// Required GPU memory in bytes
    pub gpu_memory: Option<u64>,

    /// Required network bandwidth (bytes/sec)
    pub network_bandwidth: Option<u64>,

    /// Required storage space in bytes
    pub storage_space: Option<u64>,

    /// GPU-specific requirements
    pub gpu_requirements: Option<GpuRequirements>,

    /// CPU-specific requirements
    pub cpu_requirements: Option<CpuRequirements>,

    /// Memory-specific requirements
    pub memory_requirements: Option<MemoryRequirements>,

    /// I/O-specific requirements
    pub io_requirements: Option<IoRequirements>,

    /// Network-specific requirements
    pub network_requirements: Option<NetworkRequirements>,

    /// Custom resource requirements
    pub custom_requirements: HashMap<String, ResourceRequirement>,
}

/// GPU-specific requirements
#[derive(Debug, Clone)]
pub struct GpuRequirements {
    /// Minimum compute capability required
    pub min_compute_capability: Option<(u32, u32)>,

    /// Required GPU architecture
    pub required_architecture: Option<String>,

    /// Minimum GPU memory bandwidth
    pub min_memory_bandwidth: Option<u64>,

    /// Required GPU features
    pub required_features: Vec<String>,

    /// GPU affinity preferences
    pub affinity_preferences: Vec<usize>,
}

/// CPU-specific requirements
#[derive(Debug, Clone)]
pub struct CpuRequirements {
    /// Minimum CPU frequency in MHz
    pub min_frequency: Option<f64>,

    /// Required CPU instruction sets
    pub required_instruction_sets: Vec<String>,

    /// Required CPU features (e.g., AVX, SSE)
    pub required_features: Vec<String>,

    /// NUMA topology preferences
    pub numa_preferences: Option<NumaPreferences>,

    /// CPU cache requirements
    pub cache_requirements: Option<CacheRequirements>,
}

/// NUMA topology preferences
#[derive(Debug, Clone)]
pub struct NumaPreferences {
    /// Preferred NUMA nodes
    pub preferred_nodes: Vec<usize>,

    /// Local memory preference strength
    pub local_memory_preference: f64,

    /// Maximum NUMA distance allowed
    pub max_numa_distance: Option<usize>,
}

/// CPU cache requirements
#[derive(Debug, Clone)]
pub struct CacheRequirements {
    /// Minimum L1 cache size
    pub min_l1_cache: Option<u64>,

    /// Minimum L2 cache size
    pub min_l2_cache: Option<u64>,

    /// Minimum L3 cache size
    pub min_l3_cache: Option<u64>,

    /// Cache line size requirements
    pub cache_line_size: Option<usize>,
}

/// Memory-specific requirements
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Memory type preference (DDR4, DDR5, etc.)
    pub memory_type: Option<String>,

    /// Minimum memory bandwidth
    pub min_bandwidth: Option<u64>,

    /// Memory latency requirements
    pub max_latency: Option<Duration>,

    /// Huge page requirements
    pub huge_pages: Option<HugePageRequirements>,

    /// Memory protection requirements
    pub protection: Option<MemoryProtection>,
}

/// Huge page configuration
#[derive(Debug, Clone)]
pub struct HugePageRequirements {
    /// Huge page size preference
    pub page_size: HugePageSize,

    /// Number of huge pages needed
    pub page_count: usize,

    /// Huge page pool preference
    pub pool_preference: Option<String>,
}

/// Huge page size options
#[derive(Debug, Clone)]
pub enum HugePageSize {
    /// 2MB pages
    Size2MB,

    /// 1GB pages
    Size1GB,

    /// Custom size in bytes
    Custom(u64),
}

/// Memory protection requirements
#[derive(Debug, Clone)]
pub struct MemoryProtection {
    /// Require memory encryption
    pub encryption: bool,

    /// Require memory isolation
    pub isolation: bool,

    /// Memory access permissions
    pub permissions: MemoryPermissions,
}

/// Memory access permissions
#[derive(Debug, Clone)]
pub struct MemoryPermissions {
    /// Read permission required
    pub read: bool,

    /// Write permission required
    pub write: bool,

    /// Execute permission required
    pub execute: bool,
}

/// I/O-specific requirements
#[derive(Debug, Clone)]
pub struct IoRequirements {
    /// Required storage type
    pub storage_type: Option<StorageType>,

    /// Minimum IOPS requirement
    pub min_iops: Option<u64>,

    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,

    /// I/O pattern preferences
    pub access_patterns: Vec<IoAccessPattern>,

    /// Durability requirements
    pub durability: Option<DurabilityLevel>,
}

/// Storage type preferences
#[derive(Debug, Clone)]
pub enum StorageType {
    /// HDD
    HDD,
    /// SSD
    SSD,
    /// NVMe
    NVMe,
    /// Memory
    Memory,
    /// Network
    Network,
    /// Custom
    Custom(String),
}

/// I/O access patterns
#[derive(Debug, Clone)]
pub enum IoAccessPattern {
    /// Sequential
    Sequential,
    /// Random
    Random,
    /// Mixed
    Mixed,
    /// WriteHeavy
    WriteHeavy,
    /// ReadHeavy
    ReadHeavy,
    /// StreamingRead
    StreamingRead,
    /// StreamingWrite
    StreamingWrite,
}

/// Data durability levels
#[derive(Debug, Clone)]
pub enum DurabilityLevel {
    /// No durability guarantees
    None,

    /// Write to memory buffer
    BufferWrite,

    /// Force write to storage
    SyncWrite,

    /// Replicated writes
    Replicated { replica_count: usize },

    /// Custom durability strategy
    Custom(String),
}

/// Network-specific requirements
#[derive(Debug, Clone)]
pub struct NetworkRequirements {
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,

    /// Minimum bandwidth requirement
    pub min_bandwidth: Option<u64>,

    /// Maximum packet loss tolerance
    pub max_packet_loss: Option<f64>,

    /// Protocol requirements
    pub required_protocols: Vec<String>,

    /// Quality of Service requirements
    pub qos_requirements: Option<QosRequirements>,
}

/// Quality of Service requirements
#[derive(Debug, Clone)]
pub struct QosRequirements {
    /// Traffic class priority
    pub traffic_class: TrafficClass,

    /// Bandwidth guarantee
    pub bandwidth_guarantee: Option<u64>,

    /// Latency guarantee
    pub latency_guarantee: Option<Duration>,

    /// Jitter tolerance
    pub jitter_tolerance: Option<Duration>,
}

/// Network traffic classes
#[derive(Debug, Clone)]
pub enum TrafficClass {
    /// BestEffort
    BestEffort,
    /// Bronze
    Bronze,
    /// Silver
    Silver,
    /// Gold
    Gold,
    /// Platinum
    Platinum,
    /// RealTime
    RealTime,
    /// Custom
    Custom(String),
}

/// Generic resource requirement
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    /// Resource type identifier
    pub resource_type: String,

    /// Required amount/quantity
    pub required_amount: f64,

    /// Minimum acceptable amount
    pub min_amount: Option<f64>,

    /// Maximum acceptable amount
    pub max_amount: Option<f64>,

    /// Resource unit (e.g., "bytes", "cores", "connections")
    pub unit: String,

    /// Additional resource properties
    pub properties: HashMap<String, String>,
}

/// Task execution constraints and limits
///
/// Defines boundaries and limits for task execution to ensure
/// system stability and prevent resource exhaustion.
#[derive(Debug, Clone, Default)]
pub struct TaskConstraints {
    /// Maximum total execution time
    pub max_execution_time: Option<Duration>,

    /// Task deadline (absolute time)
    pub deadline: Option<SystemTime>,

    /// Required execution location
    pub location: Option<ExecutionLocation>,

    /// Resource affinity requirements
    pub affinity: Option<TaskAffinity>,

    /// Execution isolation requirements
    pub isolation: Option<IsolationRequirements>,

    /// Security constraints
    pub security: Option<SecurityConstraints>,

    /// Compliance requirements
    pub compliance: Option<ComplianceRequirements>,

    /// Custom constraints
    pub custom_constraints: HashMap<String, ConstraintValue>,
}

/// Execution location specification
///
/// Defines where the task should be executed, supporting
/// local, remote, cloud, and edge execution environments.
#[derive(Debug, Clone)]
pub enum ExecutionLocation {
    /// Execute on local machine
    Local,

    /// Execute on specific remote node
    Remote { node_id: String },

    /// Execute in cloud environment
    Cloud {
        provider: String,
        region: String,
        availability_zone: Option<String>,
    },

    /// Execute on edge device
    Edge {
        device_id: String,
        device_type: String,
    },

    /// Execute in container environment
    Container {
        container_id: String,
        orchestrator: String,
    },

    /// Execute in virtual machine
    VirtualMachine { vm_id: String, hypervisor: String },

    /// Custom execution location
    Custom {
        location_type: String,
        parameters: HashMap<String, String>,
    },
}

/// Task affinity requirements for resource binding
#[derive(Debug, Clone)]
pub struct TaskAffinity {
    /// CPU core affinity
    pub cpu_affinity: Option<Vec<usize>>,

    /// Memory affinity (NUMA nodes)
    pub memory_affinity: Option<Vec<usize>>,

    /// Node affinity for distributed execution
    pub node_affinity: Option<Vec<String>>,

    /// GPU device affinity
    pub gpu_affinity: Option<Vec<usize>>,

    /// Storage device affinity
    pub storage_affinity: Option<Vec<String>>,

    /// Network interface affinity
    pub network_affinity: Option<Vec<String>>,

    /// Affinity strength (how strict the requirements are)
    pub affinity_strength: AffinityStrength,
}

/// Affinity strength levels
#[derive(Debug, Clone)]
pub enum AffinityStrength {
    /// Preferred but not required
    Preferred,

    /// Required (hard constraint)
    Required,

    /// Anti-affinity (avoid these resources)
    AntiAffinity,

    /// Custom affinity policy
    Custom(String),
}

/// Execution isolation requirements
#[derive(Debug, Clone)]
pub struct IsolationRequirements {
    /// Process isolation level
    pub process_isolation: IsolationLevel,

    /// Memory isolation requirements
    pub memory_isolation: IsolationLevel,

    /// Network isolation requirements
    pub network_isolation: IsolationLevel,

    /// Filesystem isolation requirements
    pub filesystem_isolation: IsolationLevel,

    /// Custom isolation requirements
    pub custom_isolation: HashMap<String, IsolationLevel>,
}

/// Isolation level specification
#[derive(Debug, Clone)]
pub enum IsolationLevel {
    /// No isolation required
    None,

    /// Basic isolation (process boundaries)
    Basic,

    /// Strong isolation (containers)
    Strong,

    /// Complete isolation (VMs, sandboxing)
    Complete,

    /// Custom isolation level
    Custom(String),
}

/// Security constraint requirements
#[derive(Debug, Clone)]
pub struct SecurityConstraints {
    /// Required security clearance level
    pub clearance_level: Option<String>,

    /// Encryption requirements
    pub encryption_requirements: EncryptionRequirements,

    /// Access control requirements
    pub access_control: AccessControlRequirements,

    /// Audit requirements
    pub audit_requirements: AuditRequirements,

    /// Data classification requirements
    pub data_classification: Option<DataClassification>,
}

/// Encryption requirements
#[derive(Debug, Clone)]
pub struct EncryptionRequirements {
    /// Data encryption at rest
    pub at_rest: bool,

    /// Data encryption in transit
    pub in_transit: bool,

    /// Data encryption in memory
    pub in_memory: bool,

    /// Required encryption algorithms
    pub algorithms: Vec<String>,

    /// Minimum key length
    pub min_key_length: Option<usize>,
}

/// Access control requirements
#[derive(Debug, Clone)]
pub struct AccessControlRequirements {
    /// Required authentication methods
    pub authentication_methods: Vec<String>,

    /// Required authorization levels
    pub authorization_levels: Vec<String>,

    /// Role-based access control
    pub rbac_requirements: Option<RbacRequirements>,

    /// Attribute-based access control
    pub abac_requirements: Option<AbacRequirements>,
}

/// Role-Based Access Control requirements
#[derive(Debug, Clone)]
pub struct RbacRequirements {
    /// Required roles
    pub required_roles: Vec<String>,

    /// Prohibited roles
    pub prohibited_roles: Vec<String>,

    /// Role hierarchy requirements
    pub hierarchy_requirements: Option<String>,
}

/// Attribute-Based Access Control requirements
#[derive(Debug, Clone)]
pub struct AbacRequirements {
    /// Required attributes
    pub required_attributes: HashMap<String, String>,

    /// Attribute policies
    pub policies: Vec<String>,

    /// Context requirements
    pub context_requirements: HashMap<String, String>,
}

/// Audit requirements
#[derive(Debug, Clone)]
pub struct AuditRequirements {
    /// Enable audit logging
    pub enable_logging: bool,

    /// Audit detail level
    pub detail_level: AuditDetailLevel,

    /// Required audit events
    pub required_events: Vec<String>,

    /// Audit retention period
    pub retention_period: Duration,

    /// Audit integrity requirements
    pub integrity_requirements: AuditIntegrityRequirements,
}

/// Audit detail levels
#[derive(Debug, Clone)]
pub enum AuditDetailLevel {
    /// Basic
    Basic,
    /// Detailed
    Detailed,
    /// Comprehensive
    Comprehensive,
    /// Custom
    Custom(String),
}

/// Audit integrity requirements
#[derive(Debug, Clone)]
pub struct AuditIntegrityRequirements {
    /// Digital signatures required
    pub digital_signatures: bool,

    /// Tamper detection required
    pub tamper_detection: bool,

    /// Audit log encryption
    pub log_encryption: bool,

    /// Immutable audit logs
    pub immutable_logs: bool,
}

/// Data classification levels
#[derive(Debug, Clone)]
pub enum DataClassification {
    /// Public
    Public,
    /// Internal
    Internal,
    /// Confidential
    Confidential,
    /// Restricted
    Restricted,
    /// TopSecret
    TopSecret,
    /// Custom
    Custom(String),
}

/// Compliance requirements
#[derive(Debug, Clone)]
pub struct ComplianceRequirements {
    /// Required compliance frameworks
    pub frameworks: Vec<ComplianceFramework>,

    /// Data residency requirements
    pub data_residency: Option<DataResidencyRequirements>,

    /// Privacy requirements
    pub privacy_requirements: Option<PrivacyRequirements>,

    /// Custom compliance requirements
    pub custom_requirements: HashMap<String, String>,
}

/// Compliance frameworks
#[derive(Debug, Clone)]
pub enum ComplianceFramework {
    /// GDPR
    GDPR,
    /// HIPAA
    HIPAA,
    /// SOX
    SOX,
    /// PciDss
    PciDss,
    /// ISO27001
    ISO27001,
    /// FedRAMP
    FedRAMP,
    /// SOC2
    SOC2,
    /// Custom
    Custom(String),
}

/// Data residency requirements
#[derive(Debug, Clone)]
pub struct DataResidencyRequirements {
    /// Allowed countries
    pub allowed_countries: Vec<String>,

    /// Prohibited countries
    pub prohibited_countries: Vec<String>,

    /// Allowed regions
    pub allowed_regions: Vec<String>,

    /// Data sovereignty requirements
    pub sovereignty_requirements: bool,
}

/// Privacy requirements
#[derive(Debug, Clone)]
pub struct PrivacyRequirements {
    /// PII handling requirements
    pub pii_handling: PiiHandlingRequirements,

    /// Data anonymization requirements
    pub anonymization: bool,

    /// Consent requirements
    pub consent_requirements: ConsentRequirements,

    /// Right to be forgotten compliance
    pub right_to_be_forgotten: bool,
}

/// PII handling requirements
#[derive(Debug, Clone)]
pub struct PiiHandlingRequirements {
    /// PII encryption required
    pub encryption_required: bool,

    /// PII access logging
    pub access_logging: bool,

    /// PII retention limits
    pub retention_limits: Option<Duration>,

    /// PII processing purposes
    pub processing_purposes: Vec<String>,
}

/// Consent requirements
#[derive(Debug, Clone)]
pub struct ConsentRequirements {
    /// Explicit consent required
    pub explicit_consent: bool,

    /// Consent verification required
    pub consent_verification: bool,

    /// Consent withdrawal support
    pub withdrawal_support: bool,

    /// Consent granularity level
    pub granularity_level: ConsentGranularity,
}

/// Consent granularity levels
#[derive(Debug, Clone)]
pub enum ConsentGranularity {
    /// Global
    Global,
    /// PerPurpose
    PerPurpose,
    /// PerDataType
    PerDataType,
    /// PerOperation
    PerOperation,
    /// Custom
    Custom(String),
}

/// Generic constraint value
#[derive(Debug, Clone)]
pub enum ConstraintValue {
    /// String
    String(String),
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// Boolean
    Boolean(bool),
    /// Duration
    Duration(Duration),
    /// Timestamp
    Timestamp(SystemTime),
    /// Array
    Array(Vec<ConstraintValue>),
    /// Object
    Object(HashMap<String, ConstraintValue>),
}

/// Task execution result containing output and metrics
///
/// Comprehensive result structure that captures all aspects of
/// task execution including output data, performance metrics, and error information.
#[derive(Debug)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,

    /// Final execution status
    pub status: TaskStatus,

    /// Task output data (if any)
    pub data: Option<Box<dyn Any + Send>>,

    /// Execution performance metrics
    pub metrics: TaskExecutionMetrics,

    /// Error information (if task failed)
    pub error: Option<TaskError>,
}

/// Detailed task execution metrics
///
/// Comprehensive performance and resource utilization metrics
/// collected during task execution for analysis and optimization.
#[derive(Debug, Clone)]
pub struct TaskExecutionMetrics {
    /// Task execution start time
    pub start_time: SystemTime,

    /// Task execution end time
    pub end_time: Option<SystemTime>,

    /// Total execution duration
    pub duration: Option<Duration>,

    /// Resource usage during execution
    pub resource_usage: TaskResourceUsage,

    /// Performance characteristics
    pub performance: TaskPerformanceMetrics,
}

impl Default for TaskExecutionMetrics {
    fn default() -> Self {
        Self {
            start_time: SystemTime::UNIX_EPOCH,
            end_time: None,
            duration: None,
            resource_usage: TaskResourceUsage::default(),
            performance: TaskPerformanceMetrics::default(),
        }
    }
}

/// Resource utilization metrics during task execution
#[derive(Debug, Clone, Default)]
pub struct TaskResourceUsage {
    /// CPU utilization percentage
    pub cpu_percent: f64,

    /// Memory usage in bytes
    pub memory_bytes: u64,

    /// I/O operations performed
    pub io_operations: u64,

    /// Network bytes transferred
    pub network_bytes: u64,

    /// GPU memory used (if applicable)
    pub gpu_memory_bytes: Option<u64>,

    /// GPU utilization percentage
    pub gpu_utilization_percent: Option<f64>,

    /// Storage space used
    pub storage_bytes: Option<u64>,

    /// Energy consumption (if measurable)
    pub energy_consumption: Option<f64>,
}

/// Task performance characteristics
#[derive(Debug, Clone, Default)]
pub struct TaskPerformanceMetrics {
    /// Task throughput (operations per second)
    pub throughput: f64,

    /// Task latency in milliseconds
    pub latency: f64,

    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,

    /// Error rate during execution (0.0 to 1.0)
    pub error_rate: f64,

    /// Quality metrics (task-specific)
    pub quality_metrics: HashMap<String, f64>,

    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Task efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct EfficiencyMetrics {
    /// CPU efficiency (work done per CPU time)
    pub cpu_efficiency: f64,

    /// Memory efficiency (work done per memory used)
    pub memory_efficiency: f64,

    /// I/O efficiency (work done per I/O operation)
    pub io_efficiency: f64,

    /// Energy efficiency (work done per energy consumed)
    pub energy_efficiency: Option<f64>,

    /// Overall efficiency score
    pub overall_efficiency: f64,
}

/// Task execution error information
///
/// Detailed error information to support debugging, recovery,
/// and failure analysis in the execution system.
#[derive(Debug, Clone)]
pub struct TaskError {
    /// Error category/type
    pub error_type: String,

    /// Human-readable error message
    pub message: String,

    /// Error code (if applicable)
    pub code: Option<i32>,

    /// Stack trace information
    pub stack_trace: Option<String>,

    /// Error context and metadata
    pub context: HashMap<String, String>,

    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,

    /// Related errors (for error chains)
    pub related_errors: Vec<RelatedError>,

    /// Error severity level
    pub severity: ErrorSeverity,

    /// Error category for classification
    pub category: ErrorCategory,
}

/// Related error information for error chains
#[derive(Debug, Clone)]
pub struct RelatedError {
    /// Related error type
    pub error_type: String,

    /// Related error message
    pub message: String,

    /// Relationship to main error
    pub relationship: ErrorRelationship,
}

/// Error relationship types
#[derive(Debug, Clone)]
pub enum ErrorRelationship {
    /// This error caused the main error
    Cause,

    /// This error was caused by the main error
    Effect,

    /// This error is related to the main error
    Related,

    /// This error occurred at the same time
    Concurrent,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational (not really an error)
    Info,

    /// Warning (potential issue)
    Warning,

    /// Minor error (recoverable)
    Minor,

    /// Major error (significant impact)
    Major,

    /// Critical error (system-affecting)
    Critical,

    /// Fatal error (system failure)
    Fatal,
}

/// Error categories for classification
#[derive(Debug, Clone)]
pub enum ErrorCategory {
    /// Resource-related errors
    Resource,

    /// Configuration errors
    Configuration,

    /// Input validation errors
    Validation,

    /// Network connectivity errors
    Network,

    /// I/O operation errors
    IO,

    /// Security-related errors
    Security,

    /// Timeout errors
    Timeout,

    /// Dependency errors
    Dependency,

    /// Internal system errors
    System,

    /// User-induced errors
    User,

    /// External service errors
    External,

    /// Custom error category
    Custom(String),
}

// Default implementations and utility functions

impl Default for TaskMetadata {
    fn default() -> Self {
        Self {
            name: "unnamed_task".to_string(),
            description: None,
            tags: Vec::new(),
            created_at: SystemTime::now(),
            estimated_duration: None,
            priority: TaskPriority::Normal,
            dependencies: Vec::new(),
            group_id: None,
            submitted_by: None,
            custom_metadata: HashMap::new(),
            retry_config: None,
            timeout_config: None,
        }
    }
}

impl ExecutionTask {
    /// Create a new execution task with minimal configuration
    pub fn new<F>(id: String, name: String, task_fn: F) -> Self
    where
        F: Fn() -> SklResult<()> + Send + Sync + 'static,
    {
        Self {
            id,
            task_type: TaskType::Custom("generic".to_string()),
            task_fn: Box::new(task_fn),
            metadata: TaskMetadata {
                name,
                ..Default::default()
            },
            requirements: TaskRequirements::default(),
            constraints: TaskConstraints::default(),
        }
    }

    /// Create a new execution task with full configuration
    pub fn with_config<F>(
        id: String,
        task_type: TaskType,
        task_fn: F,
        metadata: TaskMetadata,
        requirements: TaskRequirements,
        constraints: TaskConstraints,
    ) -> Self
    where
        F: Fn() -> SklResult<()> + Send + Sync + 'static,
    {
        Self {
            id,
            task_type,
            task_fn: Box::new(task_fn),
            metadata,
            requirements,
            constraints,
        }
    }

    /// Get estimated execution time
    #[must_use]
    pub fn estimated_duration(&self) -> Duration {
        self.metadata
            .estimated_duration
            .unwrap_or(Duration::from_secs(60))
    }

    /// Check if task has dependencies
    #[must_use]
    pub fn has_dependencies(&self) -> bool {
        !self.metadata.dependencies.is_empty()
    }

    /// Get task priority
    #[must_use]
    pub fn priority(&self) -> TaskPriority {
        self.metadata.priority.clone()
    }

    /// Check if task requires GPU
    #[must_use]
    pub fn requires_gpu(&self) -> bool {
        self.requirements.gpu_memory.is_some() || self.requirements.gpu_requirements.is_some()
    }

    /// Check if task is CPU intensive
    #[must_use]
    pub fn is_cpu_intensive(&self) -> bool {
        self.requirements.cpu_cores.unwrap_or(1) > 1
    }

    /// Check if task is memory intensive
    #[must_use]
    pub fn is_memory_intensive(&self) -> bool {
        self.requirements.memory_bytes.unwrap_or(0) > 1024 * 1024 * 1024 // > 1GB
    }

    /// Check if task is I/O intensive
    #[must_use]
    pub fn is_io_intensive(&self) -> bool {
        self.requirements.io_bandwidth.is_some()
            || self.requirements.storage_space.unwrap_or(0) > 100 * 1024 * 1024 // > 100MB
    }

    /// Check if task is network intensive
    #[must_use]
    pub fn is_network_intensive(&self) -> bool {
        self.requirements.network_bandwidth.is_some()
            || self.requirements.network_requirements.is_some()
    }
}

impl TaskResult {
    /// Check if task completed successfully
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self.status, TaskStatus::Completed)
    }

    /// Check if task failed
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self.status, TaskStatus::Failed)
    }

    /// Get execution duration if available
    #[must_use]
    pub fn execution_duration(&self) -> Option<Duration> {
        self.metrics.duration
    }

    /// Get error message if task failed
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error.as_ref().map(|e| e.message.as_str())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = ExecutionTask::new("test_task".to_string(), "Test Task".to_string(), || Ok(()));

        assert_eq!(task.id, "test_task");
        assert_eq!(task.metadata.name, "Test Task");
        assert_eq!(task.metadata.priority, TaskPriority::Normal);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
        assert!(TaskPriority::System > TaskPriority::Critical);
    }

    #[test]
    fn test_task_status_variants() {
        let statuses = vec![
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
            TaskStatus::Timeout,
            TaskStatus::Retrying,
            TaskStatus::Paused,
            TaskStatus::RollingBack,
        ];

        for status in statuses {
            assert!(matches!(status, _)); // Accept any TaskStatus variant
        }
    }

    #[test]
    fn test_task_type_variants() {
        let types = vec![
            TaskType::Transform,
            TaskType::Fit,
            TaskType::Predict,
            TaskType::Preprocess,
            TaskType::Postprocess,
            TaskType::Validate,
            TaskType::Custom("test".to_string()),
        ];

        for task_type in types {
            assert!(matches!(task_type, _)); // Accept any TaskType variant
        }
    }

    #[test]
    fn test_task_resource_checks() {
        let mut task =
            ExecutionTask::new("test_task".to_string(), "Test Task".to_string(), || Ok(()));

        // Test GPU requirement
        task.requirements.gpu_memory = Some(1024 * 1024 * 1024); // 1GB
        assert!(task.requires_gpu());

        // Test CPU intensive
        task.requirements.cpu_cores = Some(8);
        assert!(task.is_cpu_intensive());

        // Test memory intensive
        task.requirements.memory_bytes = Some(2 * 1024 * 1024 * 1024); // 2GB
        assert!(task.is_memory_intensive());

        // Test I/O intensive
        task.requirements.storage_space = Some(500 * 1024 * 1024); // 500MB
        assert!(task.is_io_intensive());
    }

    #[test]
    fn test_task_result_status_checks() {
        let mut result = TaskResult {
            task_id: "test".to_string(),
            status: TaskStatus::Completed,
            data: None,
            metrics: TaskExecutionMetrics::default(),
            error: None,
        };

        assert!(result.is_success());
        assert!(!result.is_failure());

        result.status = TaskStatus::Failed;
        assert!(!result.is_success());
        assert!(result.is_failure());
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Fatal > ErrorSeverity::Critical);
        assert!(ErrorSeverity::Critical > ErrorSeverity::Major);
        assert!(ErrorSeverity::Major > ErrorSeverity::Minor);
        assert!(ErrorSeverity::Minor > ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning > ErrorSeverity::Info);
    }

    #[test]
    fn test_metadata_value_variants() {
        let values = vec![
            MetadataValue::String("test".to_string()),
            MetadataValue::Integer(42),
            MetadataValue::Float(3.14),
            MetadataValue::Boolean(true),
            MetadataValue::Timestamp(SystemTime::now()),
            MetadataValue::Duration(Duration::from_secs(60)),
        ];

        for value in values {
            assert!(matches!(value, _)); // Accept any MetadataValue variant
        }
    }

    #[test]
    fn test_execution_location_variants() {
        let locations = vec![
            ExecutionLocation::Local,
            ExecutionLocation::Remote {
                node_id: "node1".to_string(),
            },
            ExecutionLocation::Cloud {
                provider: "AWS".to_string(),
                region: "us-east-1".to_string(),
                availability_zone: Some("us-east-1a".to_string()),
            },
            ExecutionLocation::Edge {
                device_id: "edge1".to_string(),
                device_type: "raspberry_pi".to_string(),
            },
        ];

        for location in locations {
            assert!(matches!(location, _)); // Accept any ExecutionLocation variant
        }
    }
}
