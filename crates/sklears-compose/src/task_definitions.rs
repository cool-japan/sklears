//! Task Definition System for Composable Execution Engine
//!
//! This module provides comprehensive task definition and management capabilities for the
//! composable execution engine. It includes task metadata, requirements, constraints,
//! results, and execution context management with full lifecycle support from task
//! creation through completion and cleanup.
//!
//! # Task Architecture
//!
//! The task system is built around several core concepts:
//!
//! ```text
//! ExecutionTask
//! ├── TaskMetadata          // Task identification and classification
//! ├── TaskRequirements      // Resource and dependency requirements
//! ├── TaskConstraints       // Execution limits and constraints
//! ├── TaskContext          // Runtime execution context
//! └── TaskResult           // Execution output and metrics
//! ```
//!
//! # Task Lifecycle
//!
//! Tasks progress through well-defined states:
//!
//! ```text
//! Created → Queued → Scheduled → Running → Completed/Failed/Cancelled
//!     ↓        ↓        ↓         ↓            ↓
//!   Metadata  Priority  Resources  Context    Results
//! ```
//!
//! # Usage Examples
//!
//! ## Basic Task Creation
//! ```rust,ignore
//! use sklears_compose::task_definitions::*;
//!
//! // Create a simple machine learning training task
//! let task = ExecutionTask::builder()
//!     .name("train_model")
//!     .task_type(TaskType::Fit)
//!     .priority(TaskPriority::High)
//!     .requirements(TaskRequirements {
//!         cpu_cores: Some(8),
//!         memory: Some(16 * 1024 * 1024 * 1024), // 16GB
//!         gpu_memory: Some(8 * 1024 * 1024 * 1024), // 8GB GPU memory
//!         max_duration: Some(Duration::from_secs(3600)), // 1 hour
//!         ..Default::default()
//!     })
//!     .constraints(TaskConstraints {
//!         can_be_preempted: false,
//!         requires_exclusive_access: true,
//!         location_constraint: Some(ExecutionLocation::OnPremise),
//!         ..Default::default()
//!     })
//!     .build();
//! ```
//!
//! ## Data Processing Pipeline Task
//! ```rust,ignore
//! let preprocessing_task = ExecutionTask::builder()
//!     .name("preprocess_data")
//!     .task_type(TaskType::Transform)
//!     .priority(TaskPriority::Normal)
//!     .requirements(TaskRequirements {
//!         cpu_cores: Some(4),
//!         memory: Some(8 * 1024 * 1024 * 1024), // 8GB
//!         disk_space: Some(100 * 1024 * 1024 * 1024), // 100GB
//!         network_bandwidth: Some(100 * 1024 * 1024), // 100MB/s
//!         dependencies: vec!["data_ingestion".to_string()],
//!         ..Default::default()
//!     })
//!     .constraints(TaskConstraints {
//!         can_be_preempted: true,
//!         retry_policy: Some(RetryPolicy {
//!             max_attempts: 3,
//!             backoff_strategy: BackoffStrategy::Exponential { base: 1000, max: 30000 },
//!         }),
//!         timeout: Some(Duration::from_secs(1800)), // 30 minutes
//!         ..Default::default()
//!     })
//!     .build();
//! ```
//!
//! ## Real-time Inference Task
//! ```rust,ignore
//! let inference_task = ExecutionTask::builder()
//!     .name("realtime_inference")
//!     .task_type(TaskType::Predict)
//!     .priority(TaskPriority::Critical)
//!     .requirements(TaskRequirements {
//!         cpu_cores: Some(2),
//!         memory: Some(4 * 1024 * 1024 * 1024), // 4GB
//!         max_latency: Some(Duration::from_millis(10)), // 10ms SLA
//!         gpu_devices: vec!["cuda:0".to_string()],
//!         ..Default::default()
//!     })
//!     .constraints(TaskConstraints {
//!         can_be_preempted: false,
//!         requires_exclusive_access: false,
//!         affinity: Some(TaskAffinity::GpuOptimized),
//!         location_constraint: Some(ExecutionLocation::Edge),
//!         ..Default::default()
//!     })
//!     .build();
//! ```
//!
//! ## Distributed Training Task
//! ```rust,ignore
//! let distributed_training = ExecutionTask::builder()
//!     .name("distributed_training")
//!     .task_type(TaskType::Fit)
//!     .priority(TaskPriority::High)
//!     .requirements(TaskRequirements {
//!         cpu_cores: Some(32),
//!         memory: Some(128 * 1024 * 1024 * 1024), // 128GB
//!         gpu_devices: vec!["cuda:0", "cuda:1", "cuda:2", "cuda:3"].iter().map(|s| s.to_string()).collect(),
//!         network_bandwidth: Some(10 * 1024 * 1024 * 1024), // 10GB/s for model sync
//!         dependencies: vec!["data_preparation".to_string(), "model_setup".to_string()],
//!         ..Default::default()
//!     })
//!     .constraints(TaskConstraints {
//!         can_be_preempted: false,
//!         requires_exclusive_access: true,
//!         affinity: Some(TaskAffinity::CpuOptimized),
//!         location_constraint: Some(ExecutionLocation::CloudCluster),
//!         checkpoint_interval: Some(Duration::from_secs(600)), // Checkpoint every 10 minutes
//!         ..Default::default()
//!     })
//!     .metadata(TaskMetadata {
//!         created_at: SystemTime::now(),
//!         owner: "ml_team".to_string(),
//!         project: "large_model_training".to_string(),
//!         tags: vec!["distributed", "gpu", "training"].iter().map(|s| s.to_string()).collect(),
//!         estimated_duration: Some(Duration::from_secs(86400)), // 24 hours
//!         cost_budget: Some(1000.0), // $1000 budget
//!         ..Default::default()
//!     })
//!     .build();
//! ```

use scirs2_core::ndarray::Array2;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Main execution task definition with comprehensive metadata and requirements
#[derive(Clone)]
pub struct ExecutionTask {
    /// Task metadata and identification
    pub metadata: TaskMetadata,
    /// Resource requirements for execution
    pub requirements: TaskRequirements,
    /// Execution constraints and policies
    pub constraints: TaskConstraints,
    /// Task execution function or closure
    pub execution_fn: Option<TaskExecutionFunction>,
    /// Current task status
    pub status: TaskStatus,
    /// Task execution context
    pub context: Option<TaskContext>,
    /// Task execution results (when completed)
    pub result: Option<TaskResult>,
}

impl std::fmt::Debug for ExecutionTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionTask")
            .field("metadata", &self.metadata)
            .field("requirements", &self.requirements)
            .field("constraints", &self.constraints)
            .field(
                "execution_fn",
                &match &self.execution_fn {
                    Some(_) => "<function>",
                    None => "<none>",
                },
            )
            .field("status", &self.status)
            .field("context", &self.context)
            .field("result", &self.result)
            .finish()
    }
}

/// Task execution function type
pub type TaskExecutionFunction =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send>> + Send + Sync>;

/// Comprehensive task metadata for identification and classification
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Unique task identifier
    pub id: String,
    /// Human-readable task name
    pub name: String,
    /// Task type classification
    pub task_type: TaskType,
    /// Task priority level
    pub priority: TaskPriority,
    /// Task creation timestamp
    pub created_at: SystemTime,
    /// Task owner or creator
    pub owner: String,
    /// Project or workspace association
    pub project: String,
    /// Task description
    pub description: String,
    /// Task tags for categorization
    pub tags: Vec<String>,
    /// Estimated execution duration
    pub estimated_duration: Option<Duration>,
    /// Expected output size
    pub estimated_output_size: Option<u64>,
    /// Cost budget allocation
    pub cost_budget: Option<f64>,
    /// Task version for reproducibility
    pub version: String,
    /// Task dependencies (other task IDs)
    pub dependencies: Vec<String>,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

/// Resource requirements specification for task execution
#[derive(Debug, Clone, Default)]
pub struct TaskRequirements {
    /// Required CPU cores
    pub cpu_cores: Option<usize>,
    /// Required memory in bytes
    pub memory: Option<u64>,
    /// Required GPU devices
    pub gpu_devices: Vec<String>,
    /// Required GPU memory per device
    pub gpu_memory: Option<u64>,
    /// Required disk space in bytes
    pub disk_space: Option<u64>,
    /// Required network bandwidth in bytes/sec
    pub network_bandwidth: Option<u64>,
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,
    /// Maximum execution duration
    pub max_duration: Option<Duration>,
    /// Minimum required performance score
    pub min_performance_score: Option<f64>,
    /// Required software dependencies
    pub software_dependencies: Vec<SoftwareDependency>,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// Data dependencies
    pub data_dependencies: Vec<DataDependency>,
    /// Service dependencies
    pub service_dependencies: Vec<ServiceDependency>,
    /// Task dependencies (execution order)
    pub dependencies: Vec<String>,
    /// Input data size estimation
    pub input_data_size: Option<u64>,
    /// Output data size estimation
    pub output_data_size: Option<u64>,
}

/// Software dependency specification
#[derive(Debug, Clone)]
pub struct SoftwareDependency {
    /// Software name
    pub name: String,
    /// Required version or version constraint
    pub version: String,
    /// Package manager or source
    pub source: String,
    /// Is this dependency optional?
    pub optional: bool,
    /// Installation command or script
    pub install_command: Option<String>,
}

/// Hardware requirements specification
#[derive(Debug, Clone, Default)]
pub struct HardwareRequirements {
    /// CPU architecture requirement
    pub cpu_architecture: Option<CpuArchitecture>,
    /// Minimum CPU frequency
    pub min_cpu_frequency: Option<f64>,
    /// Required instruction sets
    pub required_instruction_sets: Vec<InstructionSet>,
    /// GPU architecture requirement
    pub gpu_architecture: Option<GpuArchitecture>,
    /// Minimum GPU compute capability
    pub min_gpu_compute_capability: Option<String>,
    /// Memory type preference
    pub memory_type: Option<MemoryType>,
    /// Storage type requirement
    pub storage_type: Option<StorageType>,
    /// Network requirements
    pub network_requirements: NetworkRequirements,
}

/// CPU architecture types
#[derive(Debug, Clone, PartialEq)]
pub enum CpuArchitecture {
    /// X86_64
    X86_64,
    /// Aarch64
    Aarch64,
    /// Arm
    Arm,
    /// Riscv64
    Riscv64,
    /// Power
    Power,
    /// Sparc
    Sparc,
}

/// Instruction set requirements
#[derive(Debug, Clone, PartialEq)]
pub enum InstructionSet {
    /// AVX
    AVX,
    /// AVX2
    AVX2,
    /// AVX512
    AVX512,
    /// SSE2
    SSE2,
    /// SSE3
    SSE3,
    /// SSE4_1
    SSE4_1,
    /// SSE4_2
    SSE4_2,
    /// NEON
    NEON,
    /// SVE
    SVE,
}

/// GPU architecture types
#[derive(Debug, Clone, PartialEq)]
pub enum GpuArchitecture {
    /// CUDA
    CUDA,
    /// ROCm
    ROCm,
    /// OpenCL
    OpenCL,
    /// Metal
    Metal,
    /// Vulkan
    Vulkan,
    /// DirectML
    DirectML,
}

/// Memory type preferences
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryType {
    /// DDR4
    DDR4,
    /// DDR5
    DDR5,
    /// HBM2
    HBM2,
    /// HBM3
    HBM3,
    /// LPDDR5
    LPDDR5,
}

/// Storage type requirements
#[derive(Debug, Clone, PartialEq)]
pub enum StorageType {
    /// SSD
    SSD,
    /// NVMe
    NVMe,
    /// HDD
    HDD,
    /// MemoryMapped
    MemoryMapped,
    /// Network
    Network,
}

/// Network requirements specification
#[derive(Debug, Clone, Default)]
pub struct NetworkRequirements {
    /// Minimum bandwidth
    pub min_bandwidth: Option<u64>,
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,
    /// Required protocols
    pub protocols: Vec<NetworkProtocol>,
    /// Security requirements
    pub security_requirements: Vec<SecurityRequirement>,
}

/// Network protocol types
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkProtocol {
    /// TCP
    TCP,
    /// UDP
    UDP,
    /// HTTP
    HTTP,
    /// HTTPS
    HTTPS,
    gRPC,
    /// WebSocket
    WebSocket,
    /// MQTT
    MQTT,
    /// InfiniBand
    InfiniBand,
}

/// Security requirement types
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityRequirement {
    /// TLS
    TLS,
    mTLS,
    /// OAuth2
    OAuth2,
    /// JWT
    JWT,
    /// RBAC
    RBAC,
    /// Encryption
    Encryption,
    /// VPN
    VPN,
}

/// Data dependency specification
#[derive(Debug, Clone)]
pub struct DataDependency {
    /// Data source identifier
    pub source: String,
    /// Data format
    pub format: DataFormat,
    /// Data size estimation
    pub size: Option<u64>,
    /// Data access pattern
    pub access_pattern: DataAccessPattern,
    /// Data locality preference
    pub locality_preference: DataLocality,
    /// Required data freshness
    pub freshness_requirement: Option<Duration>,
}

/// Data format types
#[derive(Debug, Clone, PartialEq)]
pub enum DataFormat {
    /// CSV
    CSV,
    /// Parquet
    Parquet,
    /// JSON
    JSON,
    /// JSONL
    JSONL,
    /// Avro
    Avro,
    /// ORC
    ORC,
    /// HDF5
    HDF5,
    /// NumPy
    NumPy,
    /// Arrow
    Arrow,
    /// Custom
    Custom(String),
}

/// Data access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum DataAccessPattern {
    /// Sequential
    Sequential,
    /// Random
    Random,
    /// Streaming
    Streaming,
    /// Batch
    Batch,
    /// Interactive
    Interactive,
}

/// Data locality preferences
#[derive(Debug, Clone, PartialEq)]
pub enum DataLocality {
    /// Local
    Local,
    /// Regional
    Regional,
    /// Global
    Global,
    /// Edge
    Edge,
    /// NoPreference
    NoPreference,
}

/// Service dependency specification
#[derive(Debug, Clone)]
pub struct ServiceDependency {
    /// Service name or identifier
    pub name: String,
    /// Service endpoint URL
    pub endpoint: String,
    /// Required service version
    pub version: Option<String>,
    /// Authentication requirements
    pub auth_requirements: Vec<AuthRequirement>,
    /// Service level agreement requirements
    pub sla_requirements: SlaRequirements,
}

/// Authentication requirement types
#[derive(Debug, Clone, PartialEq)]
pub enum AuthRequirement {
    /// ApiKey
    ApiKey,
    /// Bearer
    Bearer,
    /// Basic
    Basic,
    /// OAuth2
    OAuth2,
    /// Custom
    Custom(String),
}

/// Service level agreement requirements
#[derive(Debug, Clone)]
pub struct SlaRequirements {
    /// Maximum response time
    pub max_response_time: Option<Duration>,
    /// Minimum availability percentage
    pub min_availability: Option<f64>,
    /// Maximum error rate
    pub max_error_rate: Option<f64>,
}

/// Task execution constraints and policies
#[derive(Debug, Clone)]
pub struct TaskConstraints {
    /// Can the task be preempted?
    pub can_be_preempted: bool,
    /// Does the task require exclusive resource access?
    pub requires_exclusive_access: bool,
    /// Maximum execution time before timeout
    pub timeout: Option<Duration>,
    /// Retry policy configuration
    pub retry_policy: Option<RetryPolicy>,
    /// Resource affinity preferences
    pub affinity: Option<TaskAffinity>,
    /// Anti-affinity constraints
    pub anti_affinity: Option<TaskAntiAffinity>,
    /// Execution location constraints
    pub location_constraint: Option<ExecutionLocation>,
    /// Security constraints
    pub security_constraints: SecurityConstraints,
    /// Checkpointing configuration
    pub checkpoint_interval: Option<Duration>,
    /// Cleanup policy after completion
    pub cleanup_policy: CleanupPolicy,
    /// Resource scaling constraints
    pub scaling_constraints: ScalingConstraints,
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Backoff strategy
    pub backoff_strategy: BackoffStrategy,
    /// Conditions that trigger retries
    pub retry_conditions: Vec<RetryCondition>,
    /// Maximum total retry time
    pub max_retry_time: Option<Duration>,
}

/// Backoff strategies for retries
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed { delay: u64 },
    /// Linear increase in delay
    Linear { base: u64, increment: u64 },
    /// Exponential backoff
    Exponential { base: u64, max: u64 },
    /// Exponential with jitter
    ExponentialJitter { base: u64, max: u64, jitter: f64 },
}

/// Conditions that trigger task retries
#[derive(Debug, Clone, PartialEq)]
pub enum RetryCondition {
    /// Network-related failures
    NetworkFailure,
    /// Resource unavailability
    ResourceUnavailable,
    /// Timeout failures
    Timeout,
    /// Transient errors
    TransientError,
    /// Custom error condition
    Custom(String),
}

/// Task affinity preferences for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum TaskAffinity {
    /// CPU-optimized execution
    CpuOptimized,
    /// GPU-optimized execution
    GpuOptimized,
    /// Memory-optimized execution
    MemoryOptimized,
    /// Network-optimized execution
    NetworkOptimized,
    /// Storage-optimized execution
    StorageOptimized,
    /// Energy-efficient execution
    EnergyEfficient,
    /// Node affinity
    NodeAffinity(String),
    /// Custom affinity
    Custom(String),
}

/// Task anti-affinity constraints
#[derive(Debug, Clone)]
pub struct TaskAntiAffinity {
    /// Tasks that cannot run on the same node
    pub node_anti_affinity: Vec<String>,
    /// Tasks that cannot run simultaneously
    pub temporal_anti_affinity: Vec<String>,
    /// Resource conflicts to avoid
    pub resource_anti_affinity: Vec<String>,
}

/// Execution location constraints
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionLocation {
    /// On-premise execution
    OnPremise,
    /// Cloud execution
    Cloud,
    /// Edge device execution
    Edge,
    /// Specific cloud provider
    CloudProvider(String),
    /// Specific region
    Region(String),
    /// Specific availability zone
    AvailabilityZone(String),
    /// Specific cluster
    Cluster(String),
    /// Specific node
    Node(String),
    /// Hybrid execution
    Hybrid,
    /// Cloud cluster execution
    CloudCluster,
}

/// Security constraints for task execution
#[derive(Debug, Clone)]
pub struct SecurityConstraints {
    /// Required security level
    pub security_level: SecurityLevel,
    /// Data classification level
    pub data_classification: DataClassification,
    /// Required compliance standards
    pub compliance_requirements: Vec<ComplianceStandard>,
    /// Encryption requirements
    pub encryption_requirements: EncryptionRequirements,
    /// Access control requirements
    pub access_control: AccessControlRequirements,
}

/// Security levels
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityLevel {
    /// Public
    Public,
    /// Internal
    Internal,
    /// Confidential
    Confidential,
    /// Secret
    Secret,
    /// TopSecret
    TopSecret,
}

/// Data classification levels
#[derive(Debug, Clone, PartialEq)]
pub enum DataClassification {
    /// Public
    Public,
    /// Internal
    Internal,
    /// Sensitive
    Sensitive,
    /// Restricted
    Restricted,
    /// Confidential
    Confidential,
}

/// Compliance standards
#[derive(Debug, Clone, PartialEq)]
pub enum ComplianceStandard {
    /// GDPR
    GDPR,
    /// HIPAA
    HIPAA,
    /// SOC2
    SOC2,
    /// ISO27001
    ISO27001,
    /// FISMA
    FISMA,
    /// PCI_DSS
    PCI_DSS,
    /// Custom
    Custom(String),
}

/// Encryption requirements
#[derive(Debug, Clone, Default)]
pub struct EncryptionRequirements {
    /// Encryption at rest required
    pub at_rest: bool,
    /// Encryption in transit required
    pub in_transit: bool,
    /// Encryption in use (processing) required
    pub in_use: bool,
    /// Required encryption algorithms
    pub algorithms: Vec<EncryptionAlgorithm>,
}

/// Encryption algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum EncryptionAlgorithm {
    AES256,
    AES128,
    RSA2048,
    RSA4096,
    ChaCha20,
    Custom(String),
}

/// Access control requirements
#[derive(Debug, Clone, Default)]
pub struct AccessControlRequirements {
    /// Required authentication methods
    pub authentication: Vec<AuthenticationMethod>,
    /// Required authorization policies
    pub authorization: Vec<AuthorizationPolicy>,
    /// Audit requirements
    pub audit_requirements: AuditRequirements,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq)]
pub enum AuthenticationMethod {
    /// Password
    Password,
    /// MFA
    MFA,
    /// Certificate
    Certificate,
    /// Biometric
    Biometric,
    /// SSO
    SSO,
    /// LDAP
    LDAP,
    /// Custom
    Custom(String),
}

/// Authorization policy types
#[derive(Debug, Clone, PartialEq)]
pub enum AuthorizationPolicy {
    /// RBAC
    RBAC,
    /// ABAC
    ABAC,
    /// DAC
    DAC,
    /// MAC
    MAC,
    /// Custom
    Custom(String),
}

/// Audit requirements
#[derive(Debug, Clone, Default)]
pub struct AuditRequirements {
    /// Enable audit logging
    pub enabled: bool,
    /// Audit log retention period
    pub retention_period: Option<Duration>,
    /// Audit events to track
    pub events: Vec<AuditEvent>,
}

/// Audit event types
#[derive(Debug, Clone, PartialEq)]
pub enum AuditEvent {
    /// Access
    Access,
    /// Modification
    Modification,
    /// Deletion
    Deletion,
    /// Export
    Export,
    /// Authentication
    Authentication,
    /// Authorization
    Authorization,
    /// Custom
    Custom(String),
}

/// Cleanup policy after task completion
#[derive(Debug, Clone)]
pub struct CleanupPolicy {
    /// Cleanup strategy
    pub strategy: CleanupStrategy,
    /// Cleanup delay after completion
    pub delay: Option<Duration>,
    /// Resources to preserve
    pub preserve_resources: Vec<String>,
    /// Cleanup timeout
    pub timeout: Option<Duration>,
}

/// Cleanup strategies
#[derive(Debug, Clone, PartialEq)]
pub enum CleanupStrategy {
    /// Immediate cleanup
    Immediate,
    /// Delayed cleanup
    Delayed,
    /// Manual cleanup
    Manual,
    /// Conditional cleanup
    Conditional(String),
    /// No cleanup
    None,
}

/// Resource scaling constraints
#[derive(Debug, Clone)]
pub struct ScalingConstraints {
    /// Enable horizontal scaling
    pub horizontal_scaling: bool,
    /// Enable vertical scaling
    pub vertical_scaling: bool,
    /// Minimum instances
    pub min_instances: Option<usize>,
    /// Maximum instances
    pub max_instances: Option<usize>,
    /// Scaling triggers
    pub scaling_triggers: Vec<ScalingTrigger>,
    /// Scaling cooldown period
    pub cooldown_period: Option<Duration>,
}

/// Scaling trigger types
#[derive(Debug, Clone)]
pub enum ScalingTrigger {
    /// CPU utilization threshold
    CpuUtilization(f64),
    /// Memory utilization threshold
    MemoryUtilization(f64),
    /// Queue depth threshold
    QueueDepth(usize),
    /// Custom metric threshold
    CustomMetric { name: String, threshold: f64 },
}

/// Task execution status
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    /// Task has been created but not queued
    Created,
    /// Task is queued for execution
    Queued,
    /// Task is scheduled for execution
    Scheduled,
    /// Task is currently running
    Running,
    /// Task is paused
    Paused,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed(TaskError),
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
    /// Task is being retried
    Retrying,
    /// Task is being cleaned up
    CleaningUp,
}

/// Task type classification for execution strategies
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Data preprocessing task
    Preprocess,
    /// Model training task
    Fit,
    /// Prediction/inference task
    Predict,
    /// Data transformation task
    Transform,
    /// Model evaluation task
    Evaluate,
    /// Data validation task
    Validate,
    /// Model deployment task
    Deploy,
    /// Monitoring task
    Monitor,
    /// Cleanup task
    Cleanup,
    /// Custom task type
    Custom(String),
}

/// Task priority levels for scheduling
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Lowest priority
    Lowest,
    /// Low priority
    Low,
    /// Normal priority (default)
    Normal,
    /// High priority
    High,
    /// Highest priority
    Highest,
    /// Critical priority (preempts others)
    Critical,
}

/// Task execution error types
#[derive(Debug, Clone, PartialEq)]
pub enum TaskError {
    /// Resource allocation failure
    ResourceAllocation(String),
    /// Dependency failure
    DependencyFailure(String),
    /// Execution timeout
    Timeout,
    /// Runtime error during execution
    RuntimeError(String),
    /// Validation error
    ValidationError(String),
    /// Infrastructure error
    InfrastructureError(String),
    /// Security violation
    SecurityViolation(String),
    /// Custom error
    Custom(String),
}

/// Task execution result with comprehensive metrics
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,
    /// Task execution status
    pub status: TaskStatus,
    /// Execution output data
    pub output: Option<TaskOutput>,
    /// Execution metrics
    pub metrics: TaskExecutionMetrics,
    /// Resource usage during execution
    pub resource_usage: TaskResourceUsage,
    /// Performance metrics
    pub performance_metrics: TaskPerformanceMetrics,
    /// Error information (if failed)
    pub error: Option<TaskError>,
    /// Execution logs
    pub logs: Vec<LogEntry>,
    /// Execution artifacts
    pub artifacts: Vec<Artifact>,
    /// Task execution time (convenience field)
    pub execution_time: Option<Duration>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

/// Task execution output
#[derive(Debug)]
pub enum TaskOutput {
    /// Numerical array output
    Array(Array2<f64>),
    /// String output
    Text(String),
    /// Binary data output
    Binary(Vec<u8>),
    /// JSON-serialized output
    Json(String),
    /// File path to output
    FilePath(String),
    /// Multiple outputs
    Multiple(HashMap<String, Box<TaskOutput>>),
    /// Custom output type
    Custom(Box<dyn Any + Send + Sync>),
}

impl Clone for TaskOutput {
    fn clone(&self) -> Self {
        match self {
            TaskOutput::Array(arr) => TaskOutput::Array(arr.clone()),
            TaskOutput::Text(text) => TaskOutput::Text(text.clone()),
            TaskOutput::Binary(data) => TaskOutput::Binary(data.clone()),
            TaskOutput::Json(json) => TaskOutput::Json(json.clone()),
            TaskOutput::FilePath(path) => TaskOutput::FilePath(path.clone()),
            TaskOutput::Multiple(map) => TaskOutput::Multiple(map.clone()),
            TaskOutput::Custom(_) => {
                // Cannot clone trait objects, provide a placeholder
                TaskOutput::Text("Custom output (clone not supported)".to_string())
            }
        }
    }
}

/// Task resource usage metrics
#[derive(Debug, Clone, Default)]
pub struct TaskResourceUsage {
    /// CPU time used (in seconds)
    pub cpu_time: f64,
    /// Memory usage (in bytes)
    pub memory_usage: u64,
    /// Peak memory usage (in bytes)
    pub peak_memory_usage: u64,
    /// Disk I/O operations
    pub disk_io_operations: u64,
    /// Network bandwidth used (in bytes)
    pub network_usage: u64,
    /// GPU usage percentage
    pub gpu_usage: Option<f64>,
    /// GPU memory used (in bytes)
    pub gpu_memory_usage: Option<u64>,
}

/// Task performance metrics
#[derive(Debug, Clone, Default)]
pub struct TaskPerformanceMetrics {
    /// Operations per second
    pub operations_per_second: f64,
    /// Throughput (items processed per second)
    pub throughput: f64,
    /// Latency (average response time)
    pub latency: Duration,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// Cache hit rate (percentage)
    pub cache_hit_rate: Option<f64>,
    /// Efficiency score (0.0-1.0)
    pub efficiency_score: f64,
}

/// Log entry for task execution
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Log timestamp
    pub timestamp: SystemTime,
    /// Log level
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Log source component
    pub source: String,
}

/// Log levels for task execution
#[derive(Debug, Clone)]
pub enum LogLevel {
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
}

/// Task execution artifact
#[derive(Debug, Clone)]
pub struct Artifact {
    /// Artifact name
    pub name: String,
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// File path or location
    pub location: String,
    /// Artifact size in bytes
    pub size: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Content hash for verification
    pub content_hash: Option<String>,
}

/// Types of task artifacts
#[derive(Debug, Clone)]
pub enum ArtifactType {
    /// Output file
    OutputFile,
    /// Log file
    LogFile,
    /// Intermediate result
    IntermediateResult,
    /// Cache file
    CacheFile,
    /// Report or summary
    Report,
    /// Custom artifact
    Custom(String),
}

/// Comprehensive task execution metrics
#[derive(Debug, Clone)]
pub struct TaskExecutionMetrics {
    /// Task start time
    pub start_time: SystemTime,
    /// Task end time
    pub end_time: Option<SystemTime>,
    /// Total execution duration
    pub duration: Option<Duration>,
    /// Queue wait time
    pub queue_wait_time: Duration,
    /// Scheduling time
    pub scheduling_time: Duration,
    /// Setup time
    pub setup_time: Duration,
    /// Cleanup time
    pub cleanup_time: Duration,
    /// Number of retry attempts
    pub retry_attempts: u32,
    /// Checkpoint count
    pub checkpoint_count: u32,
    /// Task completion percentage
    pub completion_percentage: f64,
    /// Task efficiency score
    pub efficiency_score: Option<f64>,
}

impl Default for TaskExecutionMetrics {
    fn default() -> Self {
        Self {
            start_time: SystemTime::now(),
            end_time: None,
            duration: None,
            queue_wait_time: Duration::from_secs(0),
            scheduling_time: Duration::from_secs(0),
            setup_time: Duration::from_secs(0),
            cleanup_time: Duration::from_secs(0),
            retry_attempts: 0,
            checkpoint_count: 0,
            completion_percentage: 0.0,
            efficiency_score: None,
        }
    }
}

/// Task execution context for runtime state management
#[derive(Debug, Clone)]
pub struct TaskContext {
    /// Context identifier
    pub id: String,
    /// Execution environment variables
    pub environment: HashMap<String, String>,
    /// Working directory
    pub working_directory: String,
    /// Allocated resources
    pub allocated_resources: AllocatedResources,
    /// Execution state
    pub state: TaskExecutionState,
    /// Progress information
    pub progress: TaskProgress,
    /// Communication channels
    pub channels: CommunicationChannels,
}

/// Allocated resources for task execution
#[derive(Debug, Clone)]
pub struct AllocatedResources {
    /// Allocated CPU cores
    pub cpu_cores: Vec<usize>,
    /// Allocated memory in bytes
    pub memory: u64,
    /// Allocated GPU devices
    pub gpu_devices: Vec<String>,
    /// Allocated storage space
    pub storage_space: u64,
    /// Allocated network bandwidth
    pub network_bandwidth: u64,
    /// Resource allocation timestamp
    pub allocated_at: SystemTime,
}

/// Task execution state
#[derive(Debug, Clone)]
pub struct TaskExecutionState {
    /// Current execution phase
    pub phase: ExecutionPhase,
    /// Execution progress percentage
    pub progress_percentage: f64,
    /// Current operation being performed
    pub current_operation: String,
    /// Estimated time remaining
    pub estimated_time_remaining: Option<Duration>,
    /// Execution metrics snapshots
    pub metrics_snapshots: Vec<MetricsSnapshot>,
}

/// Execution phases
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionPhase {
    /// Initialization
    Initialization,
    /// Setup
    Setup,
    /// Execution
    Execution,
    /// Cleanup
    Cleanup,
    /// Finalization
    Finalization,
}

/// Task progress information
#[derive(Debug, Clone)]
pub struct TaskProgress {
    /// Total work units
    pub total_work: Option<u64>,
    /// Completed work units
    pub completed_work: u64,
    /// Progress percentage
    pub percentage: f64,
    /// Current milestone
    pub current_milestone: String,
    /// Estimated completion time
    pub estimated_completion: Option<SystemTime>,
}

/// Communication channels for task coordination
#[derive(Debug, Clone)]
pub struct CommunicationChannels {
    /// Command channel for control messages
    pub command_channel: Option<String>,
    /// Status update channel
    pub status_channel: Option<String>,
    /// Progress reporting channel
    pub progress_channel: Option<String>,
    /// Error reporting channel
    pub error_channel: Option<String>,
}

/// Metrics snapshot for monitoring
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    /// CPU usage at snapshot time
    pub cpu_usage: f64,
    /// Memory usage at snapshot time
    pub memory_usage: u64,
    /// GPU usage at snapshot time
    pub gpu_usage: Option<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Task builder for convenient task creation
pub struct TaskBuilder {
    metadata: TaskMetadata,
    requirements: TaskRequirements,
    constraints: TaskConstraints,
    execution_fn: Option<TaskExecutionFunction>,
}

impl std::fmt::Debug for TaskBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskBuilder")
            .field("metadata", &self.metadata)
            .field("requirements", &self.requirements)
            .field("constraints", &self.constraints)
            .field(
                "execution_fn",
                &match &self.execution_fn {
                    Some(_) => "<function>",
                    None => "<none>",
                },
            )
            .finish()
    }
}

impl Default for TaskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskBuilder {
    /// Create a new task builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: TaskMetadata::default(),
            requirements: TaskRequirements::default(),
            constraints: TaskConstraints::default(),
            execution_fn: None,
        }
    }

    /// Set task name
    #[must_use]
    pub fn name(mut self, name: &str) -> Self {
        self.metadata.name = name.to_string();
        self
    }

    /// Set task type
    #[must_use]
    pub fn task_type(mut self, task_type: TaskType) -> Self {
        self.metadata.task_type = task_type;
        self
    }

    /// Set task priority
    #[must_use]
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.metadata.priority = priority;
        self
    }

    /// Set task requirements
    #[must_use]
    pub fn requirements(mut self, requirements: TaskRequirements) -> Self {
        self.requirements = requirements;
        self
    }

    /// Set task constraints
    #[must_use]
    pub fn constraints(mut self, constraints: TaskConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set task metadata
    #[must_use]
    pub fn metadata(mut self, metadata: TaskMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set execution function
    pub fn execution_fn(mut self, func: TaskExecutionFunction) -> Self {
        self.execution_fn = Some(func);
        self
    }

    /// Build the task
    #[must_use]
    pub fn build(self) -> ExecutionTask {
        /// ExecutionTask
        ExecutionTask {
            metadata: self.metadata,
            requirements: self.requirements,
            constraints: self.constraints,
            execution_fn: self.execution_fn,
            status: TaskStatus::Created,
            context: None,
            result: None,
        }
    }
}

/// Task validator for ensuring task correctness
pub struct TaskValidator;

impl TaskValidator {
    /// Validate a task for correctness and consistency
    pub fn validate(task: &ExecutionTask) -> SklResult<()> {
        // Validate metadata
        Self::validate_metadata(&task.metadata)?;

        // Validate requirements
        Self::validate_requirements(&task.requirements)?;

        // Validate constraints
        Self::validate_constraints(&task.constraints)?;

        Ok(())
    }

    fn validate_metadata(metadata: &TaskMetadata) -> SklResult<()> {
        if metadata.name.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Task name cannot be empty".to_string(),
            ));
        }

        if metadata.id.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Task ID cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_requirements(requirements: &TaskRequirements) -> SklResult<()> {
        if let Some(cores) = requirements.cpu_cores {
            if cores == 0 {
                return Err(SklearsError::InvalidInput(
                    "CPU cores must be greater than 0".to_string(),
                ));
            }
        }

        if let Some(memory) = requirements.memory {
            if memory == 0 {
                return Err(SklearsError::InvalidInput(
                    "Memory must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn validate_constraints(constraints: &TaskConstraints) -> SklResult<()> {
        if let Some(timeout) = constraints.timeout {
            if timeout.as_secs() == 0 {
                return Err(SklearsError::InvalidInput(
                    "Timeout must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

// Default implementations
impl Default for TaskMetadata {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: "unnamed_task".to_string(),
            task_type: TaskType::Custom("generic".to_string()),
            priority: TaskPriority::Normal,
            created_at: SystemTime::now(),
            owner: "unknown".to_string(),
            project: "default".to_string(),
            description: String::new(),
            tags: Vec::new(),
            estimated_duration: None,
            estimated_output_size: None,
            cost_budget: None,
            version: "1.0.0".to_string(),
            dependencies: Vec::new(),
            custom_fields: HashMap::new(),
        }
    }
}

impl Default for TaskConstraints {
    fn default() -> Self {
        Self {
            can_be_preempted: true,
            requires_exclusive_access: false,
            timeout: None,
            retry_policy: None,
            affinity: None,
            anti_affinity: None,
            location_constraint: None,
            security_constraints: SecurityConstraints::default(),
            checkpoint_interval: None,
            cleanup_policy: CleanupPolicy::default(),
            scaling_constraints: ScalingConstraints::default(),
        }
    }
}

impl Default for SecurityConstraints {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Internal,
            data_classification: DataClassification::Internal,
            compliance_requirements: Vec::new(),
            encryption_requirements: EncryptionRequirements::default(),
            access_control: AccessControlRequirements::default(),
        }
    }
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self {
            strategy: CleanupStrategy::Delayed,
            delay: Some(Duration::from_secs(300)), // 5 minutes
            preserve_resources: Vec::new(),
            timeout: Some(Duration::from_secs(600)), // 10 minutes
        }
    }
}

impl Default for ScalingConstraints {
    fn default() -> Self {
        Self {
            horizontal_scaling: false,
            vertical_scaling: false,
            min_instances: Some(1),
            max_instances: Some(10),
            scaling_triggers: Vec::new(),
            cooldown_period: Some(Duration::from_secs(300)), // 5 minutes
        }
    }
}

impl ExecutionTask {
    /// Create a new task builder
    #[must_use]
    pub fn builder() -> TaskBuilder {
        TaskBuilder::new()
    }

    /// Create a simple task with basic parameters
    pub fn new(name: &str, task_type: TaskType, execution_fn: TaskExecutionFunction) -> Self {
        Self::builder()
            .name(name)
            .task_type(task_type)
            .execution_fn(execution_fn)
            .build()
    }

    /// Get task ID
    #[must_use]
    pub fn id(&self) -> &str {
        &self.metadata.id
    }

    /// Get task name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get task status
    #[must_use]
    pub fn status(&self) -> &TaskStatus {
        &self.status
    }

    /// Update task status
    pub fn set_status(&mut self, status: TaskStatus) {
        self.status = status;
    }

    /// Check if task is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            TaskStatus::Completed | TaskStatus::Failed(_) | TaskStatus::Cancelled
        )
    }

    /// Check if task can be retried
    #[must_use]
    pub fn can_retry(&self) -> bool {
        if let Some(retry_policy) = &self.constraints.retry_policy {
            if let TaskStatus::Failed(_) = self.status {
                return retry_policy.max_attempts > 0;
            }
        }
        false
    }

    /// Validate the task
    pub fn validate(&self) -> SklResult<()> {
        TaskValidator::validate(self)
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskError::ResourceAllocation(msg) => write!(f, "Resource allocation error: {msg}"),
            TaskError::DependencyFailure(msg) => write!(f, "Dependency failure: {msg}"),
            TaskError::Timeout => write!(f, "Task timeout"),
            TaskError::RuntimeError(msg) => write!(f, "Runtime error: {msg}"),
            TaskError::ValidationError(msg) => write!(f, "Validation error: {msg}"),
            TaskError::InfrastructureError(msg) => write!(f, "Infrastructure error: {msg}"),
            TaskError::SecurityViolation(msg) => write!(f, "Security violation: {msg}"),
            TaskError::Custom(msg) => write!(f, "Custom error: {msg}"),
        }
    }
}

impl std::error::Error for TaskError {}

// External crate declarations
extern crate uuid;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_builder() {
        let task = ExecutionTask::builder()
            .name("test_task")
            .task_type(TaskType::Preprocess)
            .priority(TaskPriority::High)
            .build();

        assert_eq!(task.name(), "test_task");
        assert_eq!(task.metadata.task_type, TaskType::Preprocess);
        assert_eq!(task.metadata.priority, TaskPriority::High);
        assert_eq!(task.status, TaskStatus::Created);
    }

    #[test]
    fn test_task_validation() {
        let task = ExecutionTask::builder()
            .name("valid_task")
            .task_type(TaskType::Fit)
            .build();

        assert!(task.validate().is_ok());
    }

    #[test]
    fn test_task_requirements() {
        let requirements = TaskRequirements {
            cpu_cores: Some(4),
            memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            gpu_devices: vec!["cuda:0".to_string()],
            ..Default::default()
        };

        assert_eq!(requirements.cpu_cores, Some(4));
        assert_eq!(requirements.memory, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(requirements.gpu_devices.len(), 1);
    }

    #[test]
    fn test_task_constraints() {
        let constraints = TaskConstraints {
            can_be_preempted: false,
            requires_exclusive_access: true,
            timeout: Some(Duration::from_secs(3600)),
            ..Default::default()
        };

        assert!(!constraints.can_be_preempted);
        assert!(constraints.requires_exclusive_access);
        assert_eq!(constraints.timeout, Some(Duration::from_secs(3600)));
    }

    #[test]
    fn test_task_priorities() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
        assert!(TaskPriority::Low > TaskPriority::Lowest);
    }

    #[test]
    fn test_task_status_transitions() {
        let mut task = ExecutionTask::builder().name("status_test").build();

        assert_eq!(task.status(), &TaskStatus::Created);

        task.set_status(TaskStatus::Queued);
        assert_eq!(task.status(), &TaskStatus::Queued);

        task.set_status(TaskStatus::Running);
        assert_eq!(task.status(), &TaskStatus::Running);

        task.set_status(TaskStatus::Completed);
        assert_eq!(task.status(), &TaskStatus::Completed);
        assert!(task.is_complete());
    }

    #[test]
    fn test_metadata_default() {
        let metadata = TaskMetadata::default();
        assert!(!metadata.id.is_empty());
        assert_eq!(metadata.name, "unnamed_task");
        assert_eq!(metadata.priority, TaskPriority::Normal);
        assert_eq!(metadata.owner, "unknown");
    }

    #[test]
    fn test_retry_policy() {
        let retry_policy = RetryPolicy {
            max_attempts: 3,
            backoff_strategy: BackoffStrategy::Exponential {
                base: 1000,
                max: 30000,
            },
            retry_conditions: vec![RetryCondition::NetworkFailure, RetryCondition::Timeout],
            max_retry_time: Some(Duration::from_secs(300)),
        };

        assert_eq!(retry_policy.max_attempts, 3);
        assert!(matches!(
            retry_policy.backoff_strategy,
            BackoffStrategy::Exponential { .. }
        ));
        assert_eq!(retry_policy.retry_conditions.len(), 2);
    }

    #[test]
    fn test_security_constraints() {
        let security = SecurityConstraints {
            security_level: SecurityLevel::Confidential,
            data_classification: DataClassification::Sensitive,
            compliance_requirements: vec![ComplianceStandard::GDPR, ComplianceStandard::HIPAA],
            ..Default::default()
        };

        assert_eq!(security.security_level, SecurityLevel::Confidential);
        assert_eq!(security.data_classification, DataClassification::Sensitive);
        assert_eq!(security.compliance_requirements.len(), 2);
    }
}
