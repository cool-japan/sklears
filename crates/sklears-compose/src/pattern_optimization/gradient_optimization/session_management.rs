//! Session Management Module for Gradient Optimization
//!
//! This module provides comprehensive session lifecycle management, including
//! session creation, configuration, state tracking, resource allocation,
//! progress monitoring, and event handling for optimization sessions.

use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use std::fmt;
use scirs2_core::error::{CoreError, Result as SklResult};
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, rng};
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use serde::{Deserialize, Serialize};

// Import from other specialized modules
use super::configuration_management::{OptimizationConfig};

/// Session lifecycle states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SessionState {
    /// Session is being initialized
    Initializing,
    /// Session is ready to start optimization
    Ready,
    /// Session is actively running optimization
    Running,
    /// Session is paused and can be resumed
    Paused,
    /// Session is stopped and requires restart
    Stopped,
    /// Session completed successfully
    Completed,
    /// Session failed with errors
    Failed { reason: String },
    /// Session is being cleaned up
    Cleanup,
}

/// Safety constraints for optimization sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraints {
    pub max_memory_usage: usize,
    pub max_cpu_utilization: f64,
    pub max_execution_time: Duration,
    pub parameter_change_limits: HashMap<String, (f64, f64)>,
    pub rollback_triggers: Vec<RollbackTrigger>,
    pub emergency_stop_conditions: Vec<EmergencyCondition>,
    pub monitoring_frequency: Duration,
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_memory_usage: 8 * 1024 * 1024 * 1024, // 8GB
            max_cpu_utilization: 90.0,
            max_execution_time: Duration::from_hours(12),
            parameter_change_limits: HashMap::new(),
            rollback_triggers: vec![
                RollbackTrigger::PerformanceDegradation { threshold: 0.2 },
                RollbackTrigger::ErrorRateIncrease { threshold: 0.1 },
            ],
            emergency_stop_conditions: vec![
                EmergencyCondition::MemoryExhaustion { threshold: 0.95 },
                EmergencyCondition::SystemInstability,
            ],
            monitoring_frequency: Duration::from_seconds(5),
        }
    }
}

impl SafetyConstraints {
    /// Create strict safety constraints for production use
    pub fn strict() -> Self {
        Self {
            max_memory_usage: 4 * 1024 * 1024 * 1024, // 4GB
            max_cpu_utilization: 80.0,
            max_execution_time: Duration::from_hours(6),
            parameter_change_limits: HashMap::new(),
            rollback_triggers: vec![
                RollbackTrigger::PerformanceDegradation { threshold: 0.1 },
                RollbackTrigger::ErrorRateIncrease { threshold: 0.05 },
                RollbackTrigger::MemoryPressure { threshold: 0.8 },
            ],
            emergency_stop_conditions: vec![
                EmergencyCondition::MemoryExhaustion { threshold: 0.85 },
                EmergencyCondition::SystemInstability,
                EmergencyCondition::CriticalError,
            ],
            monitoring_frequency: Duration::from_seconds(1),
        }
    }

    /// Create relaxed safety constraints for development
    pub fn relaxed() -> Self {
        Self {
            max_memory_usage: 16 * 1024 * 1024 * 1024, // 16GB
            max_cpu_utilization: 95.0,
            max_execution_time: Duration::from_hours(48),
            parameter_change_limits: HashMap::new(),
            rollback_triggers: vec![
                RollbackTrigger::PerformanceDegradation { threshold: 0.5 },
                RollbackTrigger::ErrorRateIncrease { threshold: 0.2 },
            ],
            emergency_stop_conditions: vec![
                EmergencyCondition::MemoryExhaustion { threshold: 0.98 },
            ],
            monitoring_frequency: Duration::from_seconds(10),
        }
    }
}

/// Rollback trigger conditions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RollbackTrigger {
    PerformanceDegradation { threshold: f64 },
    ErrorRateIncrease { threshold: f64 },
    MemoryPressure { threshold: f64 },
    ResourceExhaustion,
    UserTriggered,
    TimeoutExpired,
    Custom { condition_name: String, parameters: HashMap<String, f64> },
}

/// Emergency stop conditions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmergencyCondition {
    MemoryExhaustion { threshold: f64 },
    SystemInstability,
    CriticalError,
    ExternalShutdown,
    Custom { condition_name: String },
}

/// Optimization session with comprehensive state management
pub struct OptimizationSession {
    pub session_id: String,
    pub created_at: Instant,
    pub state: SessionState,
    pub config: OptimizationConfig,
    pub safety_constraints: SafetyConstraints,
    pub coordinator_handle: CoordinatorSessionHandle,
    pub execution_context: ExecutionContext,
    pub resource_allocation: ResourceAllocation,
    pub progress_tracker: ProgressTracker,
    pub session_metrics: SessionMetrics,
    pub event_log: Arc<RwLock<VecDeque<SessionEvent>>>,
    pub checkpoint_manager: CheckpointManager,
    pub data_manager: SessionDataManager,
}

impl OptimizationSession {
    /// Create a new optimization session
    pub fn new(
        session_id: String,
        config: OptimizationConfig,
        safety_constraints: SafetyConstraints,
        coordinator_handle: CoordinatorSessionHandle,
    ) -> Self {
        Self {
            session_id: session_id.clone(),
            created_at: Instant::now(),
            state: SessionState::Initializing,
            config,
            safety_constraints,
            coordinator_handle,
            execution_context: ExecutionContext::new(),
            resource_allocation: ResourceAllocation::new(),
            progress_tracker: ProgressTracker::new(session_id.clone()),
            session_metrics: SessionMetrics::new(),
            event_log: Arc::new(RwLock::new(VecDeque::new())),
            checkpoint_manager: CheckpointManager::new(session_id.clone()),
            data_manager: SessionDataManager::new(session_id),
        }
    }

    /// Start the optimization session
    pub async fn start(&mut self) -> SklResult<()> {
        self.log_event(SessionEvent::StateTransition {
            from: self.state.clone(),
            to: SessionState::Running,
            timestamp: Instant::now(),
            reason: "Session started".to_string(),
        });

        self.state = SessionState::Running;
        self.session_metrics.start_time = Some(Instant::now());
        self.progress_tracker.start();

        Ok(())
    }

    /// Pause the optimization session
    pub async fn pause(&mut self) -> SklResult<()> {
        if self.state != SessionState::Running {
            return Err(CoreError::InvalidOperation(
                "Cannot pause session that is not running".to_string(),
            ));
        }

        self.log_event(SessionEvent::StateTransition {
            from: self.state.clone(),
            to: SessionState::Paused,
            timestamp: Instant::now(),
            reason: "Session paused by user".to_string(),
        });

        self.state = SessionState::Paused;
        self.progress_tracker.pause();

        Ok(())
    }

    /// Resume the optimization session
    pub async fn resume(&mut self) -> SklResult<()> {
        if self.state != SessionState::Paused {
            return Err(CoreError::InvalidOperation(
                "Cannot resume session that is not paused".to_string(),
            ));
        }

        self.log_event(SessionEvent::StateTransition {
            from: self.state.clone(),
            to: SessionState::Running,
            timestamp: Instant::now(),
            reason: "Session resumed by user".to_string(),
        });

        self.state = SessionState::Running;
        self.progress_tracker.resume();

        Ok(())
    }

    /// Stop the optimization session
    pub async fn stop(&mut self) -> SklResult<()> {
        self.log_event(SessionEvent::StateTransition {
            from: self.state.clone(),
            to: SessionState::Stopped,
            timestamp: Instant::now(),
            reason: "Session stopped by user".to_string(),
        });

        self.state = SessionState::Stopped;
        self.session_metrics.end_time = Some(Instant::now());
        self.progress_tracker.stop();

        Ok(())
    }

    /// Complete the optimization session successfully
    pub async fn complete(&mut self) -> SklResult<()> {
        self.log_event(SessionEvent::StateTransition {
            from: self.state.clone(),
            to: SessionState::Completed,
            timestamp: Instant::now(),
            reason: "Session completed successfully".to_string(),
        });

        self.state = SessionState::Completed;
        self.session_metrics.end_time = Some(Instant::now());
        self.progress_tracker.complete();

        Ok(())
    }

    /// Fail the optimization session with reason
    pub async fn fail(&mut self, reason: String) -> SklResult<()> {
        self.log_event(SessionEvent::StateTransition {
            from: self.state.clone(),
            to: SessionState::Failed { reason: reason.clone() },
            timestamp: Instant::now(),
            reason: format!("Session failed: {}", reason),
        });

        self.state = SessionState::Failed { reason };
        self.session_metrics.end_time = Some(Instant::now());
        self.progress_tracker.fail();

        Ok(())
    }

    /// Get session duration
    pub fn duration(&self) -> Option<Duration> {
        self.session_metrics.start_time.map(|start| {
            self.session_metrics.end_time
                .unwrap_or_else(Instant::now)
                .duration_since(start)
        })
    }

    /// Check if session is active
    pub fn is_active(&self) -> bool {
        matches!(self.state, SessionState::Running | SessionState::Paused)
    }

    /// Create checkpoint
    pub async fn create_checkpoint(&mut self, name: String) -> SklResult<String> {
        self.checkpoint_manager.create_checkpoint(name, &self.state, &self.progress_tracker).await
    }

    /// Restore from checkpoint
    pub async fn restore_checkpoint(&mut self, checkpoint_id: String) -> SklResult<()> {
        let checkpoint = self.checkpoint_manager.restore_checkpoint(checkpoint_id).await?;

        self.log_event(SessionEvent::CheckpointRestored {
            checkpoint_id: checkpoint.id.clone(),
            timestamp: Instant::now(),
        });

        Ok(())
    }

    /// Log session event
    fn log_event(&self, event: SessionEvent) {
        if let Ok(mut log) = self.event_log.write() {
            log.push_back(event);
            // Keep only last 1000 events to prevent memory bloat
            if log.len() > 1000 {
                log.pop_front();
            }
        }
    }

    /// Get recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<SessionEvent> {
        if let Ok(log) = self.event_log.read() {
            log.iter().rev().take(count).cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Update session metrics
    pub fn update_metrics(&mut self, metrics_update: SessionMetricsUpdate) {
        self.session_metrics.update(metrics_update);
    }

    /// Get session summary
    pub fn get_summary(&self) -> SessionSummary {
        SessionSummary {
            session_id: self.session_id.clone(),
            state: self.state.clone(),
            created_at: self.created_at,
            duration: self.duration(),
            metrics: self.session_metrics.clone(),
            progress: self.progress_tracker.get_progress(),
            resource_usage: self.resource_allocation.get_current_usage(),
            checkpoint_count: self.checkpoint_manager.get_checkpoint_count(),
            event_count: self.event_log.read().map(|log| log.len()).unwrap_or(0),
        }
    }
}

/// Session handle for coordinator interaction
#[derive(Debug)]
pub struct CoordinatorSessionHandle {
    pub coordinator_ref: String,
    pub session_token: String,
    pub permissions: SessionPermissions,
    pub resource_limits: ResourceLimits,
}

impl CoordinatorSessionHandle {
    /// Create a new coordinator session handle
    pub fn new(
        coordinator_ref: String,
        session_token: String,
        permissions: SessionPermissions,
        resource_limits: ResourceLimits,
    ) -> Self {
        Self {
            coordinator_ref,
            session_token,
            permissions,
            resource_limits,
        }
    }

    /// Check if operation is permitted
    pub fn can_perform_operation(&self, operation: &SessionOperation) -> bool {
        match operation {
            SessionOperation::ModifyConfig => self.permissions.can_modify_config,
            SessionOperation::AccessRawData => self.permissions.can_access_raw_data,
            SessionOperation::TriggerRollback => self.permissions.can_trigger_rollback,
            SessionOperation::EmergencyStop => self.permissions.can_emergency_stop,
            SessionOperation::AccessSubsystem(subsystem) => {
                self.permissions.allowed_subsystems.contains(subsystem)
            }
        }
    }
}

/// Session permissions
#[derive(Debug, Clone)]
pub struct SessionPermissions {
    pub can_modify_config: bool,
    pub can_access_raw_data: bool,
    pub can_trigger_rollback: bool,
    pub can_emergency_stop: bool,
    pub allowed_subsystems: HashSet<String>,
}

impl Default for SessionPermissions {
    fn default() -> Self {
        Self {
            can_modify_config: false,
            can_access_raw_data: false,
            can_trigger_rollback: true,
            can_emergency_stop: true,
            allowed_subsystems: HashSet::new(),
        }
    }
}

impl SessionPermissions {
    /// Create full permissions for administrative sessions
    pub fn admin() -> Self {
        Self {
            can_modify_config: true,
            can_access_raw_data: true,
            can_trigger_rollback: true,
            can_emergency_stop: true,
            allowed_subsystems: ["factory_core", "configuration_manager", "performance_tracker",
                               "sync_policies", "error_handler", "memory_manager", "adaptive_system"]
                .iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create read-only permissions
    pub fn read_only() -> Self {
        Self {
            can_modify_config: false,
            can_access_raw_data: false,
            can_trigger_rollback: false,
            can_emergency_stop: false,
            allowed_subsystems: HashSet::new(),
        }
    }

    /// Create user permissions for regular optimization sessions
    pub fn user() -> Self {
        Self {
            can_modify_config: false,
            can_access_raw_data: false,
            can_trigger_rollback: true,
            can_emergency_stop: true,
            allowed_subsystems: ["performance_tracker"].iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Resource limits for sessions
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory: usize,
    pub max_cpu_cores: usize,
    pub max_gpu_memory: Option<usize>,
    pub max_network_bandwidth: Option<f64>,
    pub max_storage: Option<usize>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 2 * 1024 * 1024 * 1024, // 2GB
            max_cpu_cores: 4,
            max_gpu_memory: None,
            max_network_bandwidth: None,
            max_storage: Some(10 * 1024 * 1024 * 1024), // 10GB
        }
    }
}

impl ResourceLimits {
    /// Create unlimited resource limits
    pub fn unlimited() -> Self {
        Self {
            max_memory: usize::MAX,
            max_cpu_cores: usize::MAX,
            max_gpu_memory: None,
            max_network_bandwidth: None,
            max_storage: None,
        }
    }

    /// Create minimal resource limits
    pub fn minimal() -> Self {
        Self {
            max_memory: 512 * 1024 * 1024, // 512MB
            max_cpu_cores: 1,
            max_gpu_memory: None,
            max_network_bandwidth: Some(1.0), // 1 Mbps
            max_storage: Some(1024 * 1024 * 1024), // 1GB
        }
    }
}

/// Execution context for optimization
#[derive(Debug)]
pub struct ExecutionContext {
    pub environment: ExecutionEnvironment,
    pub hardware_profile: HardwareProfile,
    pub software_stack: SoftwareStack,
    pub network_configuration: NetworkConfiguration,
    pub security_context: SecurityContext,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Self {
        Self {
            environment: ExecutionEnvironment::new(),
            hardware_profile: HardwareProfile::detect(),
            software_stack: SoftwareStack::current(),
            network_configuration: NetworkConfiguration::default(),
            security_context: SecurityContext::default(),
        }
    }
}

/// Execution environment details
#[derive(Debug, Clone)]
pub struct ExecutionEnvironment {
    pub environment_type: EnvironmentType,
    pub isolation_level: IsolationLevel,
    pub resource_availability: ResourceAvailability,
}

impl ExecutionEnvironment {
    pub fn new() -> Self {
        Self {
            environment_type: EnvironmentType::Local,
            isolation_level: IsolationLevel::Process,
            resource_availability: ResourceAvailability::assess(),
        }
    }
}

/// Environment type
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentType {
    Local,
    Distributed,
    Cloud { provider: String, region: String },
    Hybrid,
    Custom { environment_name: String },
}

/// Isolation level for execution
#[derive(Debug, Clone, PartialEq)]
pub enum IsolationLevel {
    None,
    Process,
    Container,
    VirtualMachine,
    HardwareLevel,
}

/// Resource availability assessment
#[derive(Debug, Clone)]
pub struct ResourceAvailability {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_count: usize,
    pub storage_gb: f64,
    pub network_mbps: f64,
    pub availability_score: f64,
}

impl ResourceAvailability {
    /// Assess current resource availability
    pub fn assess() -> Self {
        // In a real implementation, this would probe system resources
        Self {
            cpu_cores: num_cpus::get(),
            memory_gb: 8.0, // Default assumption
            gpu_count: 0,   // Default assumption
            storage_gb: 100.0, // Default assumption
            network_mbps: 100.0, // Default assumption
            availability_score: 0.8, // Default assumption
        }
    }
}

/// Hardware profile information
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_info: CpuInfo,
    pub memory_info: MemoryInfo,
    pub gpu_info: Option<GpuInfo>,
    pub storage_info: StorageInfo,
    pub network_info: NetworkInfo,
}

impl HardwareProfile {
    /// Detect current hardware profile
    pub fn detect() -> Self {
        Self {
            cpu_info: CpuInfo::detect(),
            memory_info: MemoryInfo::detect(),
            gpu_info: GpuInfo::detect(),
            storage_info: StorageInfo::detect(),
            network_info: NetworkInfo::detect(),
        }
    }
}

/// CPU Information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model: String,
    pub cores: usize,
    pub threads: usize,
    pub base_frequency: f64,
    pub cache_size: usize,
}

impl CpuInfo {
    pub fn detect() -> Self {
        Self {
            model: "Unknown CPU".to_string(),
            cores: num_cpus::get(),
            threads: num_cpus::get(), // Simplified
            base_frequency: 2.4, // Default GHz
            cache_size: 8 * 1024 * 1024, // 8MB default
        }
    }
}

/// Memory Information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total: usize,
    pub available: usize,
    pub memory_type: String,
    pub bandwidth: f64,
}

impl MemoryInfo {
    pub fn detect() -> Self {
        Self {
            total: 8 * 1024 * 1024 * 1024, // 8GB default
            available: 6 * 1024 * 1024 * 1024, // 6GB default
            memory_type: "DDR4".to_string(),
            bandwidth: 25.6, // GB/s default
        }
    }
}

/// GPU Information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub model: String,
    pub memory: usize,
    pub compute_capability: String,
    pub cores: usize,
}

impl GpuInfo {
    pub fn detect() -> Option<Self> {
        // In a real implementation, this would detect actual GPU hardware
        None
    }
}

/// Storage Information
#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub total_space: usize,
    pub available_space: usize,
    pub storage_type: StorageType,
    pub read_speed: f64,
    pub write_speed: f64,
}

impl StorageInfo {
    pub fn detect() -> Self {
        Self {
            total_space: 1024 * 1024 * 1024 * 1024, // 1TB default
            available_space: 512 * 1024 * 1024 * 1024, // 512GB default
            storage_type: StorageType::SSD,
            read_speed: 500.0, // MB/s
            write_speed: 450.0, // MB/s
        }
    }
}

/// Storage Type
#[derive(Debug, Clone, PartialEq)]
pub enum StorageType {
    HDD,
    SSD,
    NVMe,
    Network,
    Memory,
}

/// Network Information
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    pub interfaces: Vec<NetworkInterface>,
    pub primary_interface: String,
    pub bandwidth: f64,
    pub latency: f64,
}

impl NetworkInfo {
    pub fn detect() -> Self {
        Self {
            interfaces: vec![NetworkInterface {
                name: "eth0".to_string(),
                interface_type: NetworkInterfaceType::Ethernet,
                speed_mbps: 1000.0,
                is_active: true,
            }],
            primary_interface: "eth0".to_string(),
            bandwidth: 1000.0, // Mbps
            latency: 1.0, // ms
        }
    }
}

/// Network Interface
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    pub name: String,
    pub interface_type: NetworkInterfaceType,
    pub speed_mbps: f64,
    pub is_active: bool,
}

/// Network Interface Type
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkInterfaceType {
    Ethernet,
    WiFi,
    Loopback,
    InfiniBand,
    Custom(String),
}

/// Software Stack Information
#[derive(Debug, Clone)]
pub struct SoftwareStack {
    pub operating_system: OperatingSystemInfo,
    pub runtime_environment: RuntimeEnvironment,
    pub libraries: Vec<LibraryInfo>,
    pub frameworks: Vec<FrameworkInfo>,
}

impl SoftwareStack {
    pub fn current() -> Self {
        Self {
            operating_system: OperatingSystemInfo::detect(),
            runtime_environment: RuntimeEnvironment::current(),
            libraries: vec![], // Would be populated with actual libraries
            frameworks: vec![], // Would be populated with actual frameworks
        }
    }
}

/// Operating System Information
#[derive(Debug, Clone)]
pub struct OperatingSystemInfo {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub kernel_version: String,
}

impl OperatingSystemInfo {
    pub fn detect() -> Self {
        Self {
            name: std::env::consts::OS.to_string(),
            version: "Unknown".to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            kernel_version: "Unknown".to_string(),
        }
    }
}

/// Runtime Environment
#[derive(Debug, Clone)]
pub struct RuntimeEnvironment {
    pub language: String,
    pub version: String,
    pub features: Vec<String>,
    pub optimization_level: OptimizationLevel,
}

impl RuntimeEnvironment {
    pub fn current() -> Self {
        Self {
            language: "Rust".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            features: vec![], // Would list enabled features
            optimization_level: OptimizationLevel::Release,
        }
    }
}

/// Optimization Level
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Profile,
    Custom(String),
}

/// Library Information
#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
}

/// Framework Information
#[derive(Debug, Clone)]
pub struct FrameworkInfo {
    pub name: String,
    pub version: String,
    pub components: Vec<String>,
}

/// Network Configuration
#[derive(Debug, Clone)]
pub struct NetworkConfiguration {
    pub cluster_config: Option<ClusterConfiguration>,
    pub communication_settings: CommunicationSettings,
    pub security_settings: NetworkSecuritySettings,
}

impl Default for NetworkConfiguration {
    fn default() -> Self {
        Self {
            cluster_config: None,
            communication_settings: CommunicationSettings::default(),
            security_settings: NetworkSecuritySettings::default(),
        }
    }
}

/// Cluster Configuration
#[derive(Debug, Clone)]
pub struct ClusterConfiguration {
    pub node_id: String,
    pub cluster_size: usize,
    pub node_roles: Vec<NodeRole>,
    pub discovery_mechanism: DiscoveryMechanism,
}

/// Node Role
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    Master,
    Worker,
    Coordinator,
    Storage,
    Custom(String),
}

/// Discovery Mechanism
#[derive(Debug, Clone)]
pub enum DiscoveryMechanism {
    Static { endpoints: Vec<String> },
    DNS { service_name: String },
    Consul { endpoint: String },
    Kubernetes { namespace: String },
    Custom { mechanism: String },
}

/// Communication Settings
#[derive(Debug, Clone)]
pub struct CommunicationSettings {
    pub protocol: CommunicationProtocol,
    pub compression: Option<CompressionType>,
    pub serialization: SerializationType,
    pub timeout: Duration,
    pub retry_policy: RetryPolicy,
}

impl Default for CommunicationSettings {
    fn default() -> Self {
        Self {
            protocol: CommunicationProtocol::HTTP,
            compression: Some(CompressionType::Gzip),
            serialization: SerializationType::JSON,
            timeout: Duration::from_seconds(30),
            retry_policy: RetryPolicy::default(),
        }
    }
}

/// Communication Protocol
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationProtocol {
    HTTP,
    HTTPS,
    gRPC,
    WebSocket,
    TCP,
    UDP,
    Custom(String),
}

/// Compression Type
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionType {
    None,
    Gzip,
    Deflate,
    Brotli,
    LZ4,
    Custom(String),
}

/// Serialization Type
#[derive(Debug, Clone, PartialEq)]
pub enum SerializationType {
    JSON,
    Binary,
    MessagePack,
    Protobuf,
    Custom(String),
}

/// Retry Policy
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_seconds(30),
            backoff_factor: 2.0,
        }
    }
}

/// Network Security Settings
#[derive(Debug, Clone)]
pub struct NetworkSecuritySettings {
    pub encryption_enabled: bool,
    pub authentication_required: bool,
    pub certificate_validation: bool,
    pub allowed_ciphers: Vec<String>,
}

impl Default for NetworkSecuritySettings {
    fn default() -> Self {
        Self {
            encryption_enabled: true,
            authentication_required: true,
            certificate_validation: true,
            allowed_ciphers: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
                "TLS_AES_128_GCM_SHA256".to_string(),
            ],
        }
    }
}

/// Security Context
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub user_identity: UserIdentity,
    pub access_controls: AccessControls,
    pub audit_settings: AuditSettings,
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            user_identity: UserIdentity::default(),
            access_controls: AccessControls::default(),
            audit_settings: AuditSettings::default(),
        }
    }
}

/// User Identity
#[derive(Debug, Clone)]
pub struct UserIdentity {
    pub user_id: String,
    pub username: String,
    pub groups: Vec<String>,
    pub roles: Vec<String>,
}

impl Default for UserIdentity {
    fn default() -> Self {
        Self {
            user_id: "unknown".to_string(),
            username: "unknown".to_string(),
            groups: vec![],
            roles: vec!["user".to_string()],
        }
    }
}

/// Access Controls
#[derive(Debug, Clone)]
pub struct AccessControls {
    pub read_permissions: Vec<String>,
    pub write_permissions: Vec<String>,
    pub execute_permissions: Vec<String>,
    pub admin_permissions: Vec<String>,
}

impl Default for AccessControls {
    fn default() -> Self {
        Self {
            read_permissions: vec!["session_info".to_string()],
            write_permissions: vec![],
            execute_permissions: vec!["basic_operations".to_string()],
            admin_permissions: vec![],
        }
    }
}

/// Audit Settings
#[derive(Debug, Clone)]
pub struct AuditSettings {
    pub audit_enabled: bool,
    pub audit_level: AuditLevel,
    pub audit_targets: Vec<String>,
    pub retention_period: Duration,
}

impl Default for AuditSettings {
    fn default() -> Self {
        Self {
            audit_enabled: true,
            audit_level: AuditLevel::Standard,
            audit_targets: vec!["session_events".to_string(), "security_events".to_string()],
            retention_period: Duration::from_days(30),
        }
    }
}

/// Audit Level
#[derive(Debug, Clone, PartialEq)]
pub enum AuditLevel {
    None,
    Basic,
    Standard,
    Detailed,
    Comprehensive,
}

/// Resource allocation tracking
#[derive(Debug)]
pub struct ResourceAllocation {
    pub memory_allocation: MemoryAllocation,
    pub cpu_allocation: CpuAllocation,
    pub gpu_allocation: Option<GpuAllocation>,
    pub storage_allocation: StorageAllocation,
    pub network_allocation: NetworkAllocation,
    pub allocation_history: VecDeque<AllocationSnapshot>,
}

impl ResourceAllocation {
    pub fn new() -> Self {
        Self {
            memory_allocation: MemoryAllocation::new(),
            cpu_allocation: CpuAllocation::new(),
            gpu_allocation: None,
            storage_allocation: StorageAllocation::new(),
            network_allocation: NetworkAllocation::new(),
            allocation_history: VecDeque::new(),
        }
    }

    pub fn get_current_usage(&self) -> ResourceUsage {
        ResourceUsage {
            memory_used: self.memory_allocation.used,
            cpu_used: self.cpu_allocation.used_cores,
            gpu_memory_used: self.gpu_allocation.as_ref().map(|g| g.memory_used),
            storage_used: self.storage_allocation.used,
            network_used: self.network_allocation.bandwidth_used,
        }
    }

    pub fn take_snapshot(&mut self) {
        let snapshot = AllocationSnapshot {
            timestamp: Instant::now(),
            memory: self.memory_allocation.used,
            cpu: self.cpu_allocation.used_cores,
            gpu_memory: self.gpu_allocation.as_ref().map(|g| g.memory_used),
            storage: self.storage_allocation.used,
            network: self.network_allocation.bandwidth_used,
        };

        self.allocation_history.push_back(snapshot);
        if self.allocation_history.len() > 1000 {
            self.allocation_history.pop_front();
        }
    }
}

/// Memory Allocation
#[derive(Debug)]
pub struct MemoryAllocation {
    pub allocated: usize,
    pub used: usize,
    pub peak_usage: usize,
    pub allocation_breakdown: HashMap<String, usize>,
}

impl MemoryAllocation {
    pub fn new() -> Self {
        Self {
            allocated: 0,
            used: 0,
            peak_usage: 0,
            allocation_breakdown: HashMap::new(),
        }
    }
}

/// CPU Allocation
#[derive(Debug)]
pub struct CpuAllocation {
    pub allocated_cores: usize,
    pub used_cores: usize,
    pub utilization: f64,
    pub core_affinity: Vec<usize>,
}

impl CpuAllocation {
    pub fn new() -> Self {
        Self {
            allocated_cores: 0,
            used_cores: 0,
            utilization: 0.0,
            core_affinity: vec![],
        }
    }
}

/// GPU Allocation
#[derive(Debug)]
pub struct GpuAllocation {
    pub allocated_memory: usize,
    pub memory_used: usize,
    pub utilization: f64,
    pub device_ids: Vec<usize>,
}

/// Storage Allocation
#[derive(Debug)]
pub struct StorageAllocation {
    pub allocated: usize,
    pub used: usize,
    pub temp_usage: usize,
    pub persistent_usage: usize,
}

impl StorageAllocation {
    pub fn new() -> Self {
        Self {
            allocated: 0,
            used: 0,
            temp_usage: 0,
            persistent_usage: 0,
        }
    }
}

/// Network Allocation
#[derive(Debug)]
pub struct NetworkAllocation {
    pub allocated_bandwidth: f64,
    pub bandwidth_used: f64,
    pub connection_count: usize,
    pub data_transferred: usize,
}

impl NetworkAllocation {
    pub fn new() -> Self {
        Self {
            allocated_bandwidth: 0.0,
            bandwidth_used: 0.0,
            connection_count: 0,
            data_transferred: 0,
        }
    }
}

/// Allocation Snapshot
#[derive(Debug, Clone)]
pub struct AllocationSnapshot {
    pub timestamp: Instant,
    pub memory: usize,
    pub cpu: usize,
    pub gpu_memory: Option<usize>,
    pub storage: usize,
    pub network: f64,
}

/// Resource Usage Summary
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_used: usize,
    pub cpu_used: usize,
    pub gpu_memory_used: Option<usize>,
    pub storage_used: usize,
    pub network_used: f64,
}

/// Progress tracking for optimization sessions
#[derive(Debug)]
pub struct ProgressTracker {
    pub session_id: String,
    pub total_steps: Option<usize>,
    pub completed_steps: usize,
    pub current_phase: OptimizationPhase,
    pub phase_history: VecDeque<PhaseTransition>,
    pub start_time: Option<Instant>,
    pub estimated_completion: Option<Instant>,
    pub progress_callbacks: Vec<ProgressCallback>,
}

impl ProgressTracker {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            total_steps: None,
            completed_steps: 0,
            current_phase: OptimizationPhase::Initialization,
            phase_history: VecDeque::new(),
            start_time: None,
            estimated_completion: None,
            progress_callbacks: vec![],
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.transition_to_phase(OptimizationPhase::Running);
    }

    pub fn pause(&mut self) {
        self.transition_to_phase(OptimizationPhase::Paused);
    }

    pub fn resume(&mut self) {
        self.transition_to_phase(OptimizationPhase::Running);
    }

    pub fn stop(&mut self) {
        self.transition_to_phase(OptimizationPhase::Stopped);
    }

    pub fn complete(&mut self) {
        self.transition_to_phase(OptimizationPhase::Completed);
    }

    pub fn fail(&mut self) {
        self.transition_to_phase(OptimizationPhase::Failed);
    }

    pub fn set_total_steps(&mut self, total: usize) {
        self.total_steps = Some(total);
        self.update_estimated_completion();
    }

    pub fn increment_progress(&mut self, steps: usize) {
        self.completed_steps += steps;
        self.update_estimated_completion();
        self.notify_progress_callbacks();
    }

    pub fn get_progress(&self) -> Progress {
        Progress {
            completed_steps: self.completed_steps,
            total_steps: self.total_steps,
            current_phase: self.current_phase.clone(),
            percentage: self.calculate_percentage(),
            estimated_completion: self.estimated_completion,
            elapsed_time: self.start_time.map(|start| start.elapsed()),
        }
    }

    fn transition_to_phase(&mut self, new_phase: OptimizationPhase) {
        let transition = PhaseTransition {
            from: self.current_phase.clone(),
            to: new_phase.clone(),
            timestamp: Instant::now(),
        };

        self.phase_history.push_back(transition);
        if self.phase_history.len() > 100 {
            self.phase_history.pop_front();
        }

        self.current_phase = new_phase;
    }

    fn calculate_percentage(&self) -> Option<f64> {
        self.total_steps.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.completed_steps as f64 / total as f64 * 100.0).min(100.0)
            }
        })
    }

    fn update_estimated_completion(&mut self) {
        if let (Some(start), Some(total)) = (self.start_time, self.total_steps) {
            if self.completed_steps > 0 && total > self.completed_steps {
                let elapsed = start.elapsed();
                let steps_per_second = self.completed_steps as f64 / elapsed.as_secs_f64();
                let remaining_steps = total - self.completed_steps;
                let estimated_remaining = Duration::from_secs_f64(remaining_steps as f64 / steps_per_second);
                self.estimated_completion = Some(Instant::now() + estimated_remaining);
            }
        }
    }

    fn notify_progress_callbacks(&self) {
        let progress = self.get_progress();
        for callback in &self.progress_callbacks {
            (callback.function)(&progress);
        }
    }
}

/// Optimization Phase
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationPhase {
    Initialization,
    DataLoading,
    ModelSetup,
    Running,
    Paused,
    Validation,
    Finalization,
    Completed,
    Failed,
    Stopped,
}

/// Phase Transition
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    pub from: OptimizationPhase,
    pub to: OptimizationPhase,
    pub timestamp: Instant,
}

/// Progress Information
#[derive(Debug, Clone)]
pub struct Progress {
    pub completed_steps: usize,
    pub total_steps: Option<usize>,
    pub current_phase: OptimizationPhase,
    pub percentage: Option<f64>,
    pub estimated_completion: Option<Instant>,
    pub elapsed_time: Option<Duration>,
}

/// Progress Callback
#[derive(Debug)]
pub struct ProgressCallback {
    pub name: String,
    pub function: Box<dyn Fn(&Progress) + Send + Sync>,
}

/// Session metrics tracking
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub start_time: Option<Instant>,
    pub end_time: Option<Instant>,
    pub optimization_iterations: usize,
    pub convergence_rate: f64,
    pub best_loss: Option<f64>,
    pub current_loss: Option<f64>,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub parameter_count: usize,
    pub gradient_norm: f64,
    pub memory_usage: MemoryUsageMetrics,
    pub performance_metrics: PerformanceMetrics,
}

impl SessionMetrics {
    pub fn new() -> Self {
        Self {
            start_time: None,
            end_time: None,
            optimization_iterations: 0,
            convergence_rate: 0.0,
            best_loss: None,
            current_loss: None,
            learning_rate: 0.001,
            batch_size: 32,
            parameter_count: 0,
            gradient_norm: 0.0,
            memory_usage: MemoryUsageMetrics::new(),
            performance_metrics: PerformanceMetrics::new(),
        }
    }

    pub fn update(&mut self, update: SessionMetricsUpdate) {
        if let Some(iterations) = update.optimization_iterations {
            self.optimization_iterations = iterations;
        }
        if let Some(rate) = update.convergence_rate {
            self.convergence_rate = rate;
        }
        if let Some(loss) = update.current_loss {
            self.current_loss = Some(loss);
            if self.best_loss.map_or(true, |best| loss < best) {
                self.best_loss = Some(loss);
            }
        }
        if let Some(lr) = update.learning_rate {
            self.learning_rate = lr;
        }
        if let Some(norm) = update.gradient_norm {
            self.gradient_norm = norm;
        }
        if let Some(memory) = update.memory_usage {
            self.memory_usage = memory;
        }
        if let Some(performance) = update.performance_metrics {
            self.performance_metrics = performance;
        }
    }
}

/// Session Metrics Update
#[derive(Debug, Clone)]
pub struct SessionMetricsUpdate {
    pub optimization_iterations: Option<usize>,
    pub convergence_rate: Option<f64>,
    pub current_loss: Option<f64>,
    pub learning_rate: Option<f64>,
    pub gradient_norm: Option<f64>,
    pub memory_usage: Option<MemoryUsageMetrics>,
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Memory Usage Metrics
#[derive(Debug, Clone)]
pub struct MemoryUsageMetrics {
    pub allocated: usize,
    pub used: usize,
    pub peak: usize,
    pub fragmentation: f64,
}

impl MemoryUsageMetrics {
    pub fn new() -> Self {
        Self {
            allocated: 0,
            used: 0,
            peak: 0,
            fragmentation: 0.0,
        }
    }
}

/// Performance Metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub cpu_utilization: f64,
    pub gpu_utilization: Option<f64>,
    pub network_utilization: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::from_millis(0),
            cpu_utilization: 0.0,
            gpu_utilization: None,
            network_utilization: 0.0,
        }
    }
}

/// Session events for audit trail
#[derive(Debug, Clone)]
pub enum SessionEvent {
    Created {
        timestamp: Instant,
        config: String, // Serialized config for audit
    },
    StateTransition {
        from: SessionState,
        to: SessionState,
        timestamp: Instant,
        reason: String,
    },
    ConfigurationChanged {
        timestamp: Instant,
        changes: Vec<ConfigChange>,
    },
    ProgressUpdate {
        timestamp: Instant,
        completed_steps: usize,
        total_steps: Option<usize>,
    },
    ResourceAllocation {
        timestamp: Instant,
        resource_type: String,
        amount: usize,
    },
    ErrorOccurred {
        timestamp: Instant,
        error: String,
        severity: ErrorSeverity,
    },
    CheckpointCreated {
        checkpoint_id: String,
        timestamp: Instant,
    },
    CheckpointRestored {
        checkpoint_id: String,
        timestamp: Instant,
    },
    UserAction {
        timestamp: Instant,
        user_id: String,
        action: String,
    },
}

/// Configuration Change
#[derive(Debug, Clone)]
pub struct ConfigChange {
    pub parameter: String,
    pub old_value: String,
    pub new_value: String,
}

/// Error Severity
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Checkpoint management
#[derive(Debug)]
pub struct CheckpointManager {
    pub session_id: String,
    pub checkpoints: HashMap<String, SessionCheckpoint>,
    pub max_checkpoints: usize,
    pub checkpoint_interval: Option<Duration>,
    pub last_checkpoint_time: Option<Instant>,
}

impl CheckpointManager {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            checkpoints: HashMap::new(),
            max_checkpoints: 10,
            checkpoint_interval: None,
            last_checkpoint_time: None,
        }
    }

    pub async fn create_checkpoint(
        &mut self,
        name: String,
        state: &SessionState,
        progress: &ProgressTracker,
    ) -> SklResult<String> {
        let checkpoint_id = format!("{}_{}", self.session_id, Instant::now().elapsed().as_millis());

        let checkpoint = SessionCheckpoint {
            id: checkpoint_id.clone(),
            name,
            created_at: Instant::now(),
            session_state: state.clone(),
            progress_snapshot: progress.get_progress(),
            metadata: HashMap::new(),
        };

        self.checkpoints.insert(checkpoint_id.clone(), checkpoint);
        self.last_checkpoint_time = Some(Instant::now());

        // Remove old checkpoints if limit exceeded
        if self.checkpoints.len() > self.max_checkpoints {
            let oldest_checkpoint = self.checkpoints
                .iter()
                .min_by_key(|(_, cp)| cp.created_at)
                .map(|(id, _)| id.clone());

            if let Some(oldest_id) = oldest_checkpoint {
                self.checkpoints.remove(&oldest_id);
            }
        }

        Ok(checkpoint_id)
    }

    pub async fn restore_checkpoint(&self, checkpoint_id: String) -> SklResult<&SessionCheckpoint> {
        self.checkpoints.get(&checkpoint_id)
            .ok_or_else(|| CoreError::InvalidOperation(
                format!("Checkpoint {} not found", checkpoint_id)
            ))
    }

    pub fn get_checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    pub fn list_checkpoints(&self) -> Vec<&SessionCheckpoint> {
        self.checkpoints.values().collect()
    }
}

/// Session Checkpoint
#[derive(Debug, Clone)]
pub struct SessionCheckpoint {
    pub id: String,
    pub name: String,
    pub created_at: Instant,
    pub session_state: SessionState,
    pub progress_snapshot: Progress,
    pub metadata: HashMap<String, String>,
}

/// Session data management
#[derive(Debug)]
pub struct SessionDataManager {
    pub session_id: String,
    pub data_storage: HashMap<String, SessionData>,
    pub temporary_data: HashMap<String, TemporaryData>,
    pub data_access_log: VecDeque<DataAccessEvent>,
}

impl SessionDataManager {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            data_storage: HashMap::new(),
            temporary_data: HashMap::new(),
            data_access_log: VecDeque::new(),
        }
    }

    pub fn store_data(&mut self, key: String, data: SessionData) {
        self.log_access(DataAccessEvent {
            timestamp: Instant::now(),
            operation: DataOperation::Store,
            key: key.clone(),
            size: data.size(),
        });

        self.data_storage.insert(key, data);
    }

    pub fn retrieve_data(&mut self, key: &str) -> Option<&SessionData> {
        self.log_access(DataAccessEvent {
            timestamp: Instant::now(),
            operation: DataOperation::Retrieve,
            key: key.to_string(),
            size: 0, // Size unknown for retrieval
        });

        self.data_storage.get(key)
    }

    pub fn remove_data(&mut self, key: &str) -> Option<SessionData> {
        self.log_access(DataAccessEvent {
            timestamp: Instant::now(),
            operation: DataOperation::Remove,
            key: key.to_string(),
            size: 0,
        });

        self.data_storage.remove(key)
    }

    fn log_access(&mut self, event: DataAccessEvent) {
        self.data_access_log.push_back(event);
        if self.data_access_log.len() > 1000 {
            self.data_access_log.pop_front();
        }
    }
}

/// Session Data
#[derive(Debug, Clone)]
pub struct SessionData {
    pub data_type: SessionDataType,
    pub content: Vec<u8>,
    pub metadata: HashMap<String, String>,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
}

impl SessionData {
    pub fn size(&self) -> usize {
        self.content.len()
    }
}

/// Session Data Type
#[derive(Debug, Clone, PartialEq)]
pub enum SessionDataType {
    Model,
    Dataset,
    Checkpoint,
    Logs,
    Metrics,
    Configuration,
    Temporary,
}

/// Temporary Data
#[derive(Debug, Clone)]
pub struct TemporaryData {
    pub content: Vec<u8>,
    pub expires_at: Instant,
}

/// Data Access Event
#[derive(Debug, Clone)]
pub struct DataAccessEvent {
    pub timestamp: Instant,
    pub operation: DataOperation,
    pub key: String,
    pub size: usize,
}

/// Data Operation
#[derive(Debug, Clone, PartialEq)]
pub enum DataOperation {
    Store,
    Retrieve,
    Remove,
    Update,
}

/// Session operations for permission checking
#[derive(Debug, Clone, PartialEq)]
pub enum SessionOperation {
    ModifyConfig,
    AccessRawData,
    TriggerRollback,
    EmergencyStop,
    AccessSubsystem(String),
}

/// Session summary for reporting
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: String,
    pub state: SessionState,
    pub created_at: Instant,
    pub duration: Option<Duration>,
    pub metrics: SessionMetrics,
    pub progress: Progress,
    pub resource_usage: ResourceUsage,
    pub checkpoint_count: usize,
    pub event_count: usize,
}

/// Session builder for creating optimization sessions
pub struct SessionBuilder {
    session_id: Option<String>,
    config: Option<OptimizationConfig>,
    safety_constraints: Option<SafetyConstraints>,
    permissions: Option<SessionPermissions>,
    resource_limits: Option<ResourceLimits>,
    coordinator_ref: Option<String>,
}

impl SessionBuilder {
    /// Create a new session builder
    pub fn new() -> Self {
        Self {
            session_id: None,
            config: None,
            safety_constraints: None,
            permissions: None,
            resource_limits: None,
            coordinator_ref: None,
        }
    }

    /// Set session ID
    pub fn session_id(mut self, id: String) -> Self {
        self.session_id = Some(id);
        self
    }

    /// Set optimization configuration
    pub fn config(mut self, config: OptimizationConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set safety constraints
    pub fn safety_constraints(mut self, constraints: SafetyConstraints) -> Self {
        self.safety_constraints = Some(constraints);
        self
    }

    /// Set session permissions
    pub fn permissions(mut self, permissions: SessionPermissions) -> Self {
        self.permissions = Some(permissions);
        self
    }

    /// Set resource limits
    pub fn resource_limits(mut self, limits: ResourceLimits) -> Self {
        self.resource_limits = Some(limits);
        self
    }

    /// Set coordinator reference
    pub fn coordinator_ref(mut self, coordinator_ref: String) -> Self {
        self.coordinator_ref = Some(coordinator_ref);
        self
    }

    /// Build the optimization session
    pub fn build(self) -> SklResult<OptimizationSession> {
        let session_id = self.session_id.unwrap_or_else(|| {
            format!("session_{}", Instant::now().elapsed().as_millis())
        });

        let config = self.config.unwrap_or_default();
        let safety_constraints = self.safety_constraints.unwrap_or_default();
        let permissions = self.permissions.unwrap_or_default();
        let resource_limits = self.resource_limits.unwrap_or_default();
        let coordinator_ref = self.coordinator_ref.unwrap_or_else(|| "default".to_string());

        let coordinator_handle = CoordinatorSessionHandle::new(
            coordinator_ref,
            format!("token_{}", session_id),
            permissions,
            resource_limits,
        );

        Ok(OptimizationSession::new(
            session_id,
            config,
            safety_constraints,
            coordinator_handle,
        ))
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Extension trait for Duration to add convenience methods
trait DurationExt {
    fn from_hours(hours: u64) -> Duration;
    fn from_days(days: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn from_days(days: u64) -> Duration {
        Duration::from_secs(days * 24 * 3600)
    }
}