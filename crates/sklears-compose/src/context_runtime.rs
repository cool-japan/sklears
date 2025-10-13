//! Runtime environment and process management for execution contexts
//!
//! This module provides comprehensive runtime management capabilities including
//! process lifecycle, environment configuration, resource monitoring, and
//! system integration.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock, Mutex},
    time::{Duration, Instant, SystemTime},
    process::{Child, Command, Stdio},
    path::PathBuf,
    env,
    thread,
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextError, ContextResult,
    ContextMetadata, ContextConfig, ResourceUsage, ResourceLimits, ContextEvent,
};

/// Runtime context for managing execution environments
#[derive(Debug)]
pub struct RuntimeContext {
    /// Runtime identifier
    pub id: String,
    /// Runtime type
    pub runtime_type: RuntimeType,
    /// Current state
    pub state: Arc<RwLock<RuntimeState>>,
    /// Configuration
    pub config: Arc<RwLock<RuntimeConfig>>,
    /// Environment variables
    pub environment: Arc<RwLock<HashMap<String, String>>>,
    /// Active processes
    pub processes: Arc<RwLock<HashMap<ProcessId, ProcessHandle>>>,
    /// Resource monitor
    pub resource_monitor: Arc<Mutex<ResourceMonitor>>,
    /// Runtime metrics
    pub metrics: Arc<Mutex<RuntimeMetrics>>,
    /// Health checker
    pub health_checker: Arc<Mutex<RuntimeHealthChecker>>,
    /// Created timestamp
    pub created_at: SystemTime,
}

/// Runtime types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeType {
    /// Native process runtime
    Native,
    /// Container runtime (Docker, Podman)
    Container(ContainerRuntime),
    /// Virtual machine runtime
    VirtualMachine(VmRuntime),
    /// Kubernetes pod runtime
    Kubernetes,
    /// WebAssembly runtime
    WebAssembly,
    /// Serverless function runtime
    Serverless(ServerlessRuntime),
    /// Custom runtime
    Custom(String),
}

/// Container runtime types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerRuntime {
    /// Docker container runtime
    Docker,
    /// Podman container runtime
    Podman,
    /// Containerd runtime
    Containerd,
    /// CRI-O runtime
    CriO,
    /// Custom container runtime
    Custom(String),
}

/// Virtual machine runtime types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VmRuntime {
    /// QEMU/KVM virtual machine
    QemuKvm,
    /// VMware virtual machine
    VMware,
    /// VirtualBox virtual machine
    VirtualBox,
    /// Hyper-V virtual machine
    HyperV,
    /// Cloud VM instance
    CloudVm(String),
    /// Custom VM runtime
    Custom(String),
}

/// Serverless runtime types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServerlessRuntime {
    /// AWS Lambda
    AwsLambda,
    /// Google Cloud Functions
    GoogleCloudFunctions,
    /// Azure Functions
    AzureFunctions,
    /// Vercel Functions
    VercelFunctions,
    /// Custom serverless runtime
    Custom(String),
}

/// Runtime states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeState {
    /// Runtime is initializing
    Initializing,
    /// Runtime is ready
    Ready,
    /// Runtime is running
    Running,
    /// Runtime is paused
    Paused,
    /// Runtime is stopping
    Stopping,
    /// Runtime is stopped
    Stopped,
    /// Runtime is in error state
    Error,
    /// Runtime is being migrated
    Migrating,
}

impl Display for RuntimeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeState::Initializing => write!(f, "initializing"),
            RuntimeState::Ready => write!(f, "ready"),
            RuntimeState::Running => write!(f, "running"),
            RuntimeState::Paused => write!(f, "paused"),
            RuntimeState::Stopping => write!(f, "stopping"),
            RuntimeState::Stopped => write!(f, "stopped"),
            RuntimeState::Error => write!(f, "error"),
            RuntimeState::Migrating => write!(f, "migrating"),
        }
    }
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Working directory
    pub working_directory: Option<PathBuf>,
    /// PATH environment variable override
    pub path_override: Option<String>,
    /// Default timeout for operations
    pub default_timeout: Duration,
    /// Maximum concurrent processes
    pub max_concurrent_processes: usize,
    /// Auto-restart failed processes
    pub auto_restart: bool,
    /// Resource limits
    pub resource_limits: RuntimeResourceLimits,
    /// Monitoring configuration
    pub monitoring: RuntimeMonitoringConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Security configuration
    pub security: RuntimeSecurityConfig,
    /// Custom configuration
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            working_directory: None,
            path_override: None,
            default_timeout: Duration::from_secs(30),
            max_concurrent_processes: 100,
            auto_restart: false,
            resource_limits: RuntimeResourceLimits::default(),
            monitoring: RuntimeMonitoringConfig::default(),
            network: NetworkConfig::default(),
            storage: StorageConfig::default(),
            security: RuntimeSecurityConfig::default(),
            custom: HashMap::new(),
        }
    }
}

/// Runtime resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeResourceLimits {
    /// Maximum memory in bytes
    pub max_memory: Option<usize>,
    /// Maximum CPU cores
    pub max_cpu_cores: Option<usize>,
    /// Maximum disk I/O rate (bytes/sec)
    pub max_disk_io: Option<usize>,
    /// Maximum network bandwidth (bytes/sec)
    pub max_network_bandwidth: Option<usize>,
    /// Maximum file descriptors
    pub max_file_descriptors: Option<usize>,
    /// Maximum processes/threads
    pub max_processes: Option<usize>,
    /// Custom limits
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for RuntimeResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: Some(2 * 1024 * 1024 * 1024), // 2GB
            max_cpu_cores: Some(4),
            max_disk_io: Some(100 * 1024 * 1024), // 100MB/s
            max_network_bandwidth: Some(50 * 1024 * 1024), // 50MB/s
            max_file_descriptors: Some(4096),
            max_processes: Some(1000),
            custom: HashMap::new(),
        }
    }
}

/// Runtime monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMonitoringConfig {
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Enable distributed tracing
    pub enable_tracing: bool,
}

impl Default for RuntimeMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_resource_monitoring: true,
            monitoring_interval: Duration::from_secs(5),
            enable_health_checks: true,
            health_check_interval: Duration::from_secs(30),
            enable_metrics: true,
            metrics_interval: Duration::from_secs(10),
            enable_tracing: false,
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Enable network access
    pub enable_network: bool,
    /// Allowed ports
    pub allowed_ports: Vec<u16>,
    /// Blocked hosts
    pub blocked_hosts: Vec<String>,
    /// DNS servers
    pub dns_servers: Vec<String>,
    /// HTTP proxy
    pub http_proxy: Option<String>,
    /// HTTPS proxy
    pub https_proxy: Option<String>,
    /// No proxy hosts
    pub no_proxy: Vec<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enable_network: true,
            allowed_ports: vec![80, 443, 8080, 8443],
            blocked_hosts: Vec::new(),
            dns_servers: vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()],
            http_proxy: None,
            https_proxy: None,
            no_proxy: vec!["localhost".to_string(), "127.0.0.1".to_string()],
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Temporary directory
    pub temp_dir: Option<PathBuf>,
    /// Data directory
    pub data_dir: Option<PathBuf>,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Log directory
    pub log_dir: Option<PathBuf>,
    /// Maximum storage usage
    pub max_storage_usage: Option<usize>,
    /// Enable compression
    pub enable_compression: bool,
    /// Enable encryption
    pub enable_encryption: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            temp_dir: Some(env::temp_dir()),
            data_dir: None,
            cache_dir: None,
            log_dir: None,
            max_storage_usage: Some(10 * 1024 * 1024 * 1024), // 10GB
            enable_compression: false,
            enable_encryption: false,
        }
    }
}

/// Runtime security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSecurityConfig {
    /// Enable sandboxing
    pub enable_sandboxing: bool,
    /// Allowed system calls
    pub allowed_syscalls: Vec<String>,
    /// Blocked system calls
    pub blocked_syscalls: Vec<String>,
    /// Enable capabilities restriction
    pub enable_capabilities_restriction: bool,
    /// Allowed capabilities
    pub allowed_capabilities: Vec<String>,
    /// Enable SELinux/AppArmor
    pub enable_mandatory_access_control: bool,
}

impl Default for RuntimeSecurityConfig {
    fn default() -> Self {
        Self {
            enable_sandboxing: false,
            allowed_syscalls: Vec::new(),
            blocked_syscalls: Vec::new(),
            enable_capabilities_restriction: false,
            allowed_capabilities: Vec::new(),
            enable_mandatory_access_control: false,
        }
    }
}

/// Process identifier
pub type ProcessId = Uuid;

/// Process handle for managing child processes
#[derive(Debug)]
pub struct ProcessHandle {
    /// Process ID
    pub id: ProcessId,
    /// System process ID
    pub pid: Option<u32>,
    /// Process command
    pub command: String,
    /// Process arguments
    pub args: Vec<String>,
    /// Working directory
    pub working_dir: Option<PathBuf>,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Process child handle
    pub child: Option<Child>,
    /// Process state
    pub state: ProcessState,
    /// Start time
    pub started_at: SystemTime,
    /// End time
    pub ended_at: Option<SystemTime>,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Resource usage
    pub resource_usage: ProcessResourceUsage,
}

/// Process states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessState {
    /// Process is starting
    Starting,
    /// Process is running
    Running,
    /// Process is paused
    Paused,
    /// Process completed successfully
    Completed,
    /// Process failed
    Failed,
    /// Process was killed
    Killed,
    /// Process timed out
    TimedOut,
}

impl Display for ProcessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessState::Starting => write!(f, "starting"),
            ProcessState::Running => write!(f, "running"),
            ProcessState::Paused => write!(f, "paused"),
            ProcessState::Completed => write!(f, "completed"),
            ProcessState::Failed => write!(f, "failed"),
            ProcessState::Killed => write!(f, "killed"),
            ProcessState::TimedOut => write!(f, "timed_out"),
        }
    }
}

/// Process resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessResourceUsage {
    /// CPU time used (user + system)
    pub cpu_time: Duration,
    /// Maximum resident set size (memory)
    pub max_rss: usize,
    /// Number of page faults
    pub page_faults: usize,
    /// Number of context switches
    pub context_switches: usize,
    /// Number of file system inputs
    pub fs_inputs: usize,
    /// Number of file system outputs
    pub fs_outputs: usize,
    /// Network bytes sent
    pub network_sent: usize,
    /// Network bytes received
    pub network_received: usize,
}

/// Resource monitor for tracking system resources
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Monitor state
    pub state: MonitorState,
    /// Monitoring thread handle
    pub monitor_thread: Option<thread::JoinHandle<()>>,
    /// Current resource usage
    pub current_usage: ResourceUsage,
    /// Historical usage data
    pub history: Vec<(SystemTime, ResourceUsage)>,
    /// Usage alerts
    pub alerts: Vec<ResourceAlert>,
    /// Configuration
    pub config: ResourceMonitorConfig,
}

/// Monitor states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorState {
    /// Monitor is stopped
    Stopped,
    /// Monitor is running
    Running,
    /// Monitor is paused
    Paused,
    /// Monitor is in error state
    Error,
}

/// Resource alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    /// Alert type
    pub alert_type: ResourceAlertType,
    /// Resource type
    pub resource_type: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Alert message
    pub message: String,
}

/// Resource alert types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAlertType {
    /// Resource usage exceeded threshold
    ThresholdExceeded,
    /// Resource usage approaching limit
    ApproachingLimit,
    /// Resource became unavailable
    Unavailable,
    /// Resource performance degraded
    PerformanceDegraded,
}

/// Resource monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitorConfig {
    /// Monitoring interval
    pub interval: Duration,
    /// Maximum history size
    pub max_history_size: usize,
    /// Memory usage threshold (percentage)
    pub memory_threshold: f32,
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f32,
    /// Disk usage threshold (percentage)
    pub disk_threshold: f32,
    /// Network usage threshold (bytes/sec)
    pub network_threshold: usize,
    /// Enable alerts
    pub enable_alerts: bool,
}

impl Default for ResourceMonitorConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            max_history_size: 1000,
            memory_threshold: 80.0,
            cpu_threshold: 90.0,
            disk_threshold: 85.0,
            network_threshold: 100 * 1024 * 1024, // 100MB/s
            enable_alerts: true,
        }
    }
}

/// Runtime metrics collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    /// Total processes started
    pub processes_started: usize,
    /// Total processes completed
    pub processes_completed: usize,
    /// Total processes failed
    pub processes_failed: usize,
    /// Current active processes
    pub active_processes: usize,
    /// Average process duration
    pub avg_process_duration: Duration,
    /// Total CPU time used
    pub total_cpu_time: Duration,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Total network I/O
    pub total_network_io: usize,
    /// Total disk I/O
    pub total_disk_io: usize,
    /// Uptime
    pub uptime: Duration,
    /// Custom metrics
    pub custom: HashMap<String, serde_json::Value>,
}

/// Runtime health checker
#[derive(Debug)]
pub struct RuntimeHealthChecker {
    /// Health checks
    pub health_checks: Vec<Box<dyn RuntimeHealthCheck>>,
    /// Current health status
    pub health_status: HealthStatus,
    /// Last health check time
    pub last_check_time: Option<SystemTime>,
    /// Health check history
    pub health_history: Vec<(SystemTime, HealthStatus)>,
    /// Configuration
    pub config: HealthCheckConfig,
}

/// Health check trait
pub trait RuntimeHealthCheck: Send + Sync {
    /// Health check name
    fn name(&self) -> &str;

    /// Perform health check
    fn check(&self, runtime: &RuntimeContext) -> ContextResult<HealthStatus>;

    /// Check timeout
    fn timeout(&self) -> Duration;
}

/// Health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Runtime is healthy
    Healthy,
    /// Runtime is degraded but functional
    Degraded,
    /// Runtime is unhealthy
    Unhealthy,
    /// Health check failed
    Unknown,
}

impl Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Maximum failures before marking unhealthy
    pub max_failures: usize,
    /// Enable health monitoring
    pub enable_monitoring: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            max_failures: 3,
            enable_monitoring: true,
        }
    }
}

impl RuntimeContext {
    /// Create a new runtime context
    pub fn new(
        id: String,
        runtime_type: RuntimeType,
        config: RuntimeConfig,
    ) -> Self {
        Self {
            id,
            runtime_type,
            state: Arc::new(RwLock::new(RuntimeState::Initializing)),
            config: Arc::new(RwLock::new(config)),
            environment: Arc::new(RwLock::new(env::vars().collect())),
            processes: Arc::new(RwLock::new(HashMap::new())),
            resource_monitor: Arc::new(Mutex::new(ResourceMonitor::new())),
            metrics: Arc::new(Mutex::new(RuntimeMetrics::default())),
            health_checker: Arc::new(Mutex::new(RuntimeHealthChecker::new())),
            created_at: SystemTime::now(),
        }
    }

    /// Initialize the runtime context
    pub fn initialize(&self) -> ContextResult<()> {
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        if *state != RuntimeState::Initializing {
            return Err(ContextError::custom("invalid_state",
                format!("Cannot initialize runtime in state: {}", state)));
        }

        // Initialize environment
        self.initialize_environment()?;

        // Initialize resource monitoring
        self.start_resource_monitoring()?;

        // Initialize health checking
        self.start_health_checking()?;

        // Update state
        *state = RuntimeState::Ready;

        Ok(())
    }

    /// Start the runtime
    pub fn start(&self) -> ContextResult<()> {
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        match *state {
            RuntimeState::Ready | RuntimeState::Paused => {
                *state = RuntimeState::Running;
                Ok(())
            }
            _ => Err(ContextError::custom("invalid_state",
                format!("Cannot start runtime in state: {}", state)))
        }
    }

    /// Stop the runtime
    pub fn stop(&self) -> ContextResult<()> {
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        *state = RuntimeState::Stopping;
        drop(state);

        // Stop all processes
        self.terminate_all_processes()?;

        // Stop resource monitoring
        self.stop_resource_monitoring()?;

        // Stop health checking
        self.stop_health_checking()?;

        // Update state
        let mut state = self.state.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;
        *state = RuntimeState::Stopped;

        Ok(())
    }

    /// Spawn a new process
    pub fn spawn_process(
        &self,
        command: String,
        args: Vec<String>,
        working_dir: Option<PathBuf>,
        env_vars: Option<HashMap<String, String>>,
    ) -> ContextResult<ProcessId> {
        let state = self.state.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;

        if *state != RuntimeState::Running {
            return Err(ContextError::custom("invalid_state",
                format!("Cannot spawn process in state: {}", state)));
        }

        drop(state);

        let process_id = Uuid::new_v4();
        let mut processes = self.processes.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire processes lock: {}", e)))?;

        // Check process limits
        let config = self.config.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;

        if processes.len() >= config.max_concurrent_processes {
            return Err(ContextError::custom("process_limit_exceeded",
                format!("Maximum concurrent processes limit {} exceeded", config.max_concurrent_processes)));
        }

        drop(config);

        // Prepare environment
        let mut process_env = self.environment.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?.clone();

        if let Some(additional_env) = env_vars {
            process_env.extend(additional_env);
        }

        // Create process handle
        let mut process_handle = ProcessHandle {
            id: process_id,
            pid: None,
            command: command.clone(),
            args: args.clone(),
            working_dir: working_dir.clone(),
            environment: process_env.clone(),
            child: None,
            state: ProcessState::Starting,
            started_at: SystemTime::now(),
            ended_at: None,
            exit_code: None,
            resource_usage: ProcessResourceUsage::default(),
        };

        // Spawn the process
        let mut cmd = Command::new(&command);
        cmd.args(&args);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.envs(&process_env);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(mut child) => {
                process_handle.pid = child.id();
                process_handle.state = ProcessState::Running;
                process_handle.child = Some(child);

                // Store process handle
                processes.insert(process_id, process_handle);

                // Update metrics
                let mut metrics = self.metrics.lock().map_err(|e|
                    ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
                metrics.processes_started += 1;
                metrics.active_processes += 1;

                Ok(process_id)
            }
            Err(e) => {
                process_handle.state = ProcessState::Failed;
                process_handle.ended_at = Some(SystemTime::now());
                processes.insert(process_id, process_handle);

                Err(ContextError::internal(format!("Failed to spawn process: {}", e)))
            }
        }
    }

    /// Terminate a process
    pub fn terminate_process(&self, process_id: ProcessId) -> ContextResult<()> {
        let mut processes = self.processes.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire processes lock: {}", e)))?;

        let process = processes.get_mut(&process_id)
            .ok_or_else(|| ContextError::not_found(format!("process:{}", process_id)))?;

        if let Some(ref mut child) = process.child {
            match child.kill() {
                Ok(()) => {
                    process.state = ProcessState::Killed;
                    process.ended_at = Some(SystemTime::now());
                    if let Ok(exit_status) = child.wait() {
                        process.exit_code = exit_status.code();
                    }
                }
                Err(e) => {
                    return Err(ContextError::internal(format!("Failed to kill process: {}", e)));
                }
            }
        }

        // Update metrics
        let mut metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.active_processes = metrics.active_processes.saturating_sub(1);

        Ok(())
    }

    /// Get process status
    pub fn get_process_status(&self, process_id: ProcessId) -> ContextResult<Option<ProcessState>> {
        let processes = self.processes.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire processes lock: {}", e)))?;

        Ok(processes.get(&process_id).map(|p| p.state))
    }

    /// List all processes
    pub fn list_processes(&self) -> ContextResult<Vec<ProcessId>> {
        let processes = self.processes.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire processes lock: {}", e)))?;

        Ok(processes.keys().cloned().collect())
    }

    /// Get runtime state
    pub fn get_state(&self) -> ContextResult<RuntimeState> {
        let state = self.state.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire state lock: {}", e)))?;
        Ok(*state)
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> ContextResult<RuntimeMetrics> {
        let metrics = self.metrics.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        Ok(metrics.clone())
    }

    /// Get health status
    pub fn get_health_status(&self) -> ContextResult<HealthStatus> {
        let health_checker = self.health_checker.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire health checker lock: {}", e)))?;
        Ok(health_checker.health_status.clone())
    }

    /// Set environment variable
    pub fn set_environment_variable(&self, key: String, value: String) -> ContextResult<()> {
        let mut environment = self.environment.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;
        environment.insert(key, value);
        Ok(())
    }

    /// Get environment variable
    pub fn get_environment_variable(&self, key: &str) -> ContextResult<Option<String>> {
        let environment = self.environment.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;
        Ok(environment.get(key).cloned())
    }

    /// Private helper methods
    fn initialize_environment(&self) -> ContextResult<()> {
        let config = self.config.read().map_err(|e|
            ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;

        let mut environment = self.environment.write().map_err(|e|
            ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;

        // Set working directory
        if let Some(ref working_dir) = config.working_directory {
            environment.insert("PWD".to_string(), working_dir.to_string_lossy().to_string());
        }

        // Override PATH if configured
        if let Some(ref path_override) = config.path_override {
            environment.insert("PATH".to_string(), path_override.clone());
        }

        Ok(())
    }

    fn start_resource_monitoring(&self) -> ContextResult<()> {
        let mut monitor = self.resource_monitor.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire resource monitor lock: {}", e)))?;

        monitor.start_monitoring()
    }

    fn stop_resource_monitoring(&self) -> ContextResult<()> {
        let mut monitor = self.resource_monitor.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire resource monitor lock: {}", e)))?;

        monitor.stop_monitoring()
    }

    fn start_health_checking(&self) -> ContextResult<()> {
        let mut health_checker = self.health_checker.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire health checker lock: {}", e)))?;

        health_checker.start_health_checking()
    }

    fn stop_health_checking(&self) -> ContextResult<()> {
        let mut health_checker = self.health_checker.lock().map_err(|e|
            ContextError::internal(format!("Failed to acquire health checker lock: {}", e)))?;

        health_checker.stop_health_checking()
    }

    fn terminate_all_processes(&self) -> ContextResult<()> {
        let process_ids: Vec<ProcessId> = {
            let processes = self.processes.read().map_err(|e|
                ContextError::internal(format!("Failed to acquire processes lock: {}", e)))?;
            processes.keys().cloned().collect()
        };

        for process_id in process_ids {
            if let Err(e) = self.terminate_process(process_id) {
                // Log error but continue terminating other processes
                eprintln!("Failed to terminate process {}: {}", process_id, e);
            }
        }

        Ok(())
    }
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> Self {
        Self {
            state: MonitorState::Stopped,
            monitor_thread: None,
            current_usage: ResourceUsage::default(),
            history: Vec::new(),
            alerts: Vec::new(),
            config: ResourceMonitorConfig::default(),
        }
    }

    /// Start monitoring
    pub fn start_monitoring(&mut self) -> ContextResult<()> {
        if self.state == MonitorState::Running {
            return Ok(());
        }

        self.state = MonitorState::Running;
        // Note: In a real implementation, we would spawn a monitoring thread here
        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) -> ContextResult<()> {
        self.state = MonitorState::Stopped;
        if let Some(handle) = self.monitor_thread.take() {
            // In a real implementation, we would signal the thread to stop
            // and wait for it to join
        }
        Ok(())
    }

    /// Get current usage
    pub fn get_current_usage(&self) -> ResourceUsage {
        self.current_usage.clone()
    }

    /// Get usage history
    pub fn get_usage_history(&self) -> &[(SystemTime, ResourceUsage)] {
        &self.history
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> &[ResourceAlert] {
        &self.alerts
    }
}

impl RuntimeHealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            health_checks: Vec::new(),
            health_status: HealthStatus::Unknown,
            last_check_time: None,
            health_history: Vec::new(),
            config: HealthCheckConfig::default(),
        }
    }

    /// Add a health check
    pub fn add_health_check(&mut self, health_check: Box<dyn RuntimeHealthCheck>) {
        self.health_checks.push(health_check);
    }

    /// Start health checking
    pub fn start_health_checking(&mut self) -> ContextResult<()> {
        // Note: In a real implementation, we would spawn a health checking thread here
        Ok(())
    }

    /// Stop health checking
    pub fn stop_health_checking(&mut self) -> ContextResult<()> {
        // Note: In a real implementation, we would signal the health checking thread to stop
        Ok(())
    }

    /// Perform health check
    pub fn check_health(&mut self, runtime: &RuntimeContext) -> ContextResult<HealthStatus> {
        let mut overall_status = HealthStatus::Healthy;

        for health_check in &self.health_checks {
            match health_check.check(runtime) {
                Ok(status) => {
                    match status {
                        HealthStatus::Unhealthy => {
                            overall_status = HealthStatus::Unhealthy;
                            break;
                        }
                        HealthStatus::Degraded if overall_status == HealthStatus::Healthy => {
                            overall_status = HealthStatus::Degraded;
                        }
                        HealthStatus::Unknown if overall_status != HealthStatus::Unhealthy => {
                            if overall_status == HealthStatus::Healthy {
                                overall_status = HealthStatus::Unknown;
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => {
                    overall_status = HealthStatus::Unknown;
                    if overall_status != HealthStatus::Unhealthy {
                        overall_status = HealthStatus::Unknown;
                    }
                }
            }
        }

        self.health_status = overall_status.clone();
        self.last_check_time = Some(SystemTime::now());
        self.health_history.push((SystemTime::now(), overall_status.clone()));

        Ok(overall_status)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_context_creation() {
        let config = RuntimeConfig::default();
        let runtime = RuntimeContext::new(
            "test-runtime".to_string(),
            RuntimeType::Native,
            config,
        );

        assert_eq!(runtime.id, "test-runtime");
        assert_eq!(runtime.runtime_type, RuntimeType::Native);
        assert!(matches!(runtime.get_state().unwrap(), RuntimeState::Initializing));
    }

    #[test]
    fn test_runtime_states() {
        assert_eq!(RuntimeState::Running.to_string(), "running");
        assert_eq!(RuntimeState::Stopped.to_string(), "stopped");
    }

    #[test]
    fn test_process_states() {
        assert_eq!(ProcessState::Running.to_string(), "running");
        assert_eq!(ProcessState::Failed.to_string(), "failed");
    }

    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    #[test]
    fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new();
        assert!(matches!(monitor.state, MonitorState::Stopped));
    }

    #[test]
    fn test_health_checker_creation() {
        let checker = RuntimeHealthChecker::new();
        assert!(matches!(checker.health_status, HealthStatus::Unknown));
    }

    #[test]
    fn test_runtime_config_defaults() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_concurrent_processes, 100);
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert!(!config.auto_restart);
    }

    #[test]
    fn test_runtime_type_variants() {
        let native = RuntimeType::Native;
        let docker = RuntimeType::Container(ContainerRuntime::Docker);
        let lambda = RuntimeType::Serverless(ServerlessRuntime::AwsLambda);

        assert_eq!(native, RuntimeType::Native);
        assert_ne!(native, docker);
        assert_ne!(docker, lambda);
    }
}