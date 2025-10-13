//! Runtime Context Module
//!
//! Provides comprehensive runtime environment management for execution contexts,
//! including process context, thread management, memory tracking, and runtime
//! configuration management.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock, Mutex},
    time::{Duration, Instant, SystemTime},
    thread::{ThreadId, current},
    process::{self, Command},
    env::{self, VarError},
    path::{Path, PathBuf},
    fmt::{Debug, Display},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextMetadata, ContextError, ContextResult,
    ContextEvent, IsolationLevel, ContextPriority
};

/// Runtime context for execution environment management
#[derive(Debug)]
pub struct RuntimeContext {
    /// Context identifier
    context_id: String,
    /// Runtime configuration
    config: Arc<RwLock<RuntimeConfig>>,
    /// Process information
    process_info: Arc<RwLock<ProcessInfo>>,
    /// Thread management
    thread_manager: Arc<ThreadManager>,
    /// Memory tracker
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Environment variables
    environment: Arc<RwLock<HashMap<String, String>>>,
    /// Runtime state
    state: Arc<RwLock<ContextState>>,
    /// Metadata
    metadata: Arc<RwLock<ContextMetadata>>,
    /// Performance metrics
    metrics: Arc<Mutex<RuntimeMetrics>>,
    /// Execution timing
    timing: Arc<Mutex<ExecutionTiming>>,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// CPU affinity settings
    pub cpu_affinity: Option<CpuAffinity>,
    /// Thread pool size
    pub thread_pool_size: Option<usize>,
    /// Working directory
    pub working_directory: Option<PathBuf>,
    /// Runtime optimization level
    pub optimization_level: OptimizationLevel,
    /// Garbage collection configuration
    pub gc_config: GarbageCollectionConfig,
    /// Runtime debugging options
    pub debug_options: DebugOptions,
    /// Custom runtime settings
    pub custom_settings: HashMap<String, serde_json::Value>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Some(Duration::from_secs(3600)), // 1 hour
            memory_limit: Some(2 * 1024 * 1024 * 1024), // 2GB
            cpu_affinity: None,
            thread_pool_size: Some(num_cpus::get()),
            working_directory: env::current_dir().ok(),
            optimization_level: OptimizationLevel::Balanced,
            gc_config: GarbageCollectionConfig::default(),
            debug_options: DebugOptions::default(),
            custom_settings: HashMap::new(),
        }
    }
}

/// CPU affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAffinity {
    /// Allowed CPU cores
    pub allowed_cores: Vec<usize>,
    /// CPU binding strategy
    pub binding_strategy: CpuBindingStrategy,
}

/// CPU binding strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuBindingStrategy {
    /// No binding
    None,
    /// Strict binding to specified cores
    Strict,
    /// Preferred cores but allow migration
    Preferred,
    /// Round-robin across cores
    RoundRobin,
}

/// Runtime optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization, maximum debugging
    Debug,
    /// Minimal optimization
    Minimal,
    /// Balanced optimization
    Balanced,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionConfig {
    /// Enable automatic garbage collection
    pub enabled: bool,
    /// GC trigger threshold (memory usage percentage)
    pub trigger_threshold: f32,
    /// GC frequency
    pub frequency: Duration,
    /// Maximum GC pause time
    pub max_pause_time: Duration,
    /// GC strategy
    pub strategy: GcStrategy,
}

impl Default for GarbageCollectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_threshold: 80.0,
            frequency: Duration::from_secs(60),
            max_pause_time: Duration::from_millis(100),
            strategy: GcStrategy::Generational,
        }
    }
}

/// Garbage collection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GcStrategy {
    /// Stop-the-world collection
    StopTheWorld,
    /// Incremental collection
    Incremental,
    /// Concurrent collection
    Concurrent,
    /// Generational collection
    Generational,
    /// Low-latency collection
    LowLatency,
}

/// Debug options for runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugOptions {
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
    /// Enable thread monitoring
    pub enable_thread_monitoring: bool,
    /// Debug output level
    pub debug_level: DebugLevel,
    /// Debug output destination
    pub debug_output: DebugOutput,
}

impl Default for DebugOptions {
    fn default() -> Self {
        Self {
            enable_profiling: false,
            enable_memory_tracking: true,
            enable_thread_monitoring: true,
            debug_level: DebugLevel::Info,
            debug_output: DebugOutput::Stderr,
        }
    }
}

/// Debug levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DebugLevel {
    /// No debug output
    None = 0,
    /// Error messages only
    Error = 1,
    /// Warning and error messages
    Warn = 2,
    /// Info, warning, and error messages
    Info = 3,
    /// Debug and above
    Debug = 4,
    /// All messages including trace
    Trace = 5,
}

/// Debug output destinations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugOutput {
    /// Standard error
    Stderr,
    /// Standard output
    Stdout,
    /// File output
    File(PathBuf),
    /// Network output
    Network { host: String, port: u16 },
    /// Multiple outputs
    Multiple(Vec<DebugOutput>),
}

/// Process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Parent process ID
    pub parent_pid: Option<u32>,
    /// Process name
    pub name: String,
    /// Command line arguments
    pub command_line: Vec<String>,
    /// Process start time
    pub start_time: SystemTime,
    /// Current working directory
    pub working_directory: PathBuf,
    /// Process environment variables
    pub environment_vars: HashMap<String, String>,
    /// Process priority
    pub priority: ProcessPriority,
    /// Process state
    pub state: ProcessState,
}

/// Process priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessPriority {
    /// Low priority
    Low = -10,
    /// Below normal priority
    BelowNormal = -5,
    /// Normal priority
    Normal = 0,
    /// Above normal priority
    AboveNormal = 5,
    /// High priority
    High = 10,
    /// Real-time priority
    RealTime = 20,
}

impl Default for ProcessPriority {
    fn default() -> Self {
        ProcessPriority::Normal
    }
}

/// Process states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessState {
    /// Process is running
    Running,
    /// Process is sleeping
    Sleeping,
    /// Process is waiting
    Waiting,
    /// Process is stopped
    Stopped,
    /// Process is zombie
    Zombie,
    /// Process state unknown
    Unknown,
}

/// Thread manager for runtime context
#[derive(Debug)]
pub struct ThreadManager {
    /// Active threads
    threads: Arc<RwLock<HashMap<ThreadId, ThreadInfo>>>,
    /// Thread pool configuration
    pool_config: Arc<RwLock<ThreadPoolConfig>>,
    /// Thread metrics
    metrics: Arc<Mutex<ThreadMetrics>>,
}

/// Thread information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInfo {
    /// Thread ID
    pub thread_id: u64, // Using u64 for serialization
    /// Thread name
    pub name: Option<String>,
    /// Thread state
    pub state: ThreadState,
    /// Thread priority
    pub priority: ThreadPriority,
    /// Thread creation time
    pub created_at: SystemTime,
    /// CPU time used
    pub cpu_time: Duration,
    /// Stack size
    pub stack_size: Option<usize>,
}

/// Thread states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadState {
    /// Thread is running
    Running,
    /// Thread is blocked
    Blocked,
    /// Thread is waiting
    Waiting,
    /// Thread is terminated
    Terminated,
}

/// Thread priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreadPriority {
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 5,
    /// High priority
    High = 10,
}

impl Default for ThreadPriority {
    fn default() -> Self {
        ThreadPriority::Normal
    }
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Minimum number of threads
    pub min_threads: usize,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Thread idle timeout
    pub idle_timeout: Duration,
    /// Thread stack size
    pub stack_size: Option<usize>,
    /// Thread naming pattern
    pub naming_pattern: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            min_threads: 1,
            max_threads: num_cpus::get() * 2,
            idle_timeout: Duration::from_secs(60),
            stack_size: None,
            naming_pattern: "worker-{}".to_string(),
        }
    }
}

/// Thread metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreadMetrics {
    /// Total threads created
    pub total_created: usize,
    /// Currently active threads
    pub active_count: usize,
    /// Peak thread count
    pub peak_count: usize,
    /// Total CPU time
    pub total_cpu_time: Duration,
    /// Context switches
    pub context_switches: u64,
    /// Thread creation failures
    pub creation_failures: usize,
}

/// Memory tracker for runtime context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryTracker {
    /// Total allocated memory
    pub total_allocated: usize,
    /// Currently used memory
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Memory leak detection
    pub potential_leaks: Vec<MemoryLeak>,
    /// Allocation history
    pub allocation_history: Vec<AllocationRecord>,
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    /// Allocation ID
    pub allocation_id: u64,
    /// Size of leaked memory
    pub size: usize,
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    /// Stack trace (if available)
    pub stack_trace: Option<Vec<String>>,
}

/// Memory allocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRecord {
    /// Allocation ID
    pub id: u64,
    /// Size allocated
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: SystemTime,
    /// Allocation type
    pub allocation_type: AllocationType,
}

/// Memory allocation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationType {
    /// Stack allocation
    Stack,
    /// Heap allocation
    Heap,
    /// Static allocation
    Static,
    /// Memory mapped allocation
    MemoryMapped,
}

/// Runtime performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    /// Execution duration
    pub execution_duration: Duration,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// I/O operations count
    pub io_operations: u64,
    /// Network operations count
    pub network_operations: u64,
    /// System calls count
    pub system_calls: u64,
    /// Context switches count
    pub context_switches: u64,
    /// Page faults count
    pub page_faults: u64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, serde_json::Value>,
}

/// Execution timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTiming {
    /// Context creation time
    pub created_at: Instant,
    /// Context start time
    pub started_at: Option<Instant>,
    /// Context end time
    pub ended_at: Option<Instant>,
    /// Pause timestamps
    pub paused_at: Vec<Instant>,
    /// Resume timestamps
    pub resumed_at: Vec<Instant>,
    /// Total pause duration
    pub total_pause_duration: Duration,
    /// Checkpoint timestamps
    pub checkpoints: HashMap<String, Instant>,
}

impl Default for ExecutionTiming {
    fn default() -> Self {
        Self {
            created_at: Instant::now(),
            started_at: None,
            ended_at: None,
            paused_at: Vec::new(),
            resumed_at: Vec::new(),
            total_pause_duration: Duration::from_secs(0),
            checkpoints: HashMap::new(),
        }
    }
}

impl RuntimeContext {
    /// Create a new runtime context
    pub fn new(context_id: String) -> ContextResult<Self> {
        let process_info = Self::gather_process_info()?;

        let context = Self {
            context_id: context_id.clone(),
            config: Arc::new(RwLock::new(RuntimeConfig::default())),
            process_info: Arc::new(RwLock::new(process_info)),
            thread_manager: Arc::new(ThreadManager::new()),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::default())),
            environment: Arc::new(RwLock::new(env::vars().collect())),
            state: Arc::new(RwLock::new(ContextState::Initializing)),
            metadata: Arc::new(RwLock::new(ContextMetadata::default())),
            metrics: Arc::new(Mutex::new(RuntimeMetrics::default())),
            timing: Arc::new(Mutex::new(ExecutionTiming::default())),
        };

        // Update state to active
        *context.state.write().unwrap() = ContextState::Active;

        Ok(context)
    }

    /// Create runtime context with custom configuration
    pub fn with_config(context_id: String, config: RuntimeConfig) -> ContextResult<Self> {
        let mut context = Self::new(context_id)?;
        *context.config.write().unwrap() = config;
        Ok(context)
    }

    /// Gather current process information
    fn gather_process_info() -> ContextResult<ProcessInfo> {
        let pid = process::id();
        let args: Vec<String> = env::args().collect();
        let name = args.first()
            .and_then(|arg| Path::new(arg).file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ProcessInfo {
            pid,
            parent_pid: None, // Platform-specific implementation needed
            name,
            command_line: args,
            start_time: SystemTime::now(),
            working_directory: env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            environment_vars: env::vars().collect(),
            priority: ProcessPriority::default(),
            state: ProcessState::Running,
        })
    }

    /// Update runtime configuration
    pub fn update_config<F>(&self, updater: F) -> ContextResult<()>
    where
        F: FnOnce(&mut RuntimeConfig) -> ContextResult<()>,
    {
        let mut config = self.config.write()
            .map_err(|e| ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;
        updater(&mut *config)
    }

    /// Get current runtime configuration
    pub fn get_config(&self) -> ContextResult<RuntimeConfig> {
        let config = self.config.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;
        Ok(config.clone())
    }

    /// Set environment variable
    pub fn set_environment_variable(&self, key: String, value: String) -> ContextResult<()> {
        let mut env = self.environment.write()
            .map_err(|e| ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;
        env.insert(key.clone(), value.clone());

        // Also set in actual process environment
        env::set_var(&key, &value);
        Ok(())
    }

    /// Get environment variable
    pub fn get_environment_variable(&self, key: &str) -> ContextResult<Option<String>> {
        let env = self.environment.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;
        Ok(env.get(key).cloned())
    }

    /// Remove environment variable
    pub fn remove_environment_variable(&self, key: &str) -> ContextResult<Option<String>> {
        let mut env = self.environment.write()
            .map_err(|e| ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;
        let removed = env.remove(key);

        // Also remove from actual process environment
        env::remove_var(key);
        Ok(removed)
    }

    /// Get all environment variables
    pub fn get_all_environment_variables(&self) -> ContextResult<HashMap<String, String>> {
        let env = self.environment.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire environment lock: {}", e)))?;
        Ok(env.clone())
    }

    /// Start execution timing
    pub fn start_execution(&self) -> ContextResult<()> {
        let mut timing = self.timing.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;
        timing.started_at = Some(Instant::now());
        Ok(())
    }

    /// End execution timing
    pub fn end_execution(&self) -> ContextResult<Duration> {
        let mut timing = self.timing.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;

        let end_time = Instant::now();
        timing.ended_at = Some(end_time);

        let duration = match timing.started_at {
            Some(start) => end_time.duration_since(start) - timing.total_pause_duration,
            None => Duration::from_secs(0),
        };

        // Update metrics
        let mut metrics = self.metrics.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.execution_duration = duration;

        Ok(duration)
    }

    /// Pause execution timing
    pub fn pause_execution(&self) -> ContextResult<()> {
        let mut timing = self.timing.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;
        timing.paused_at.push(Instant::now());
        Ok(())
    }

    /// Resume execution timing
    pub fn resume_execution(&self) -> ContextResult<()> {
        let mut timing = self.timing.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;

        let resume_time = Instant::now();
        timing.resumed_at.push(resume_time);

        // Calculate pause duration
        if let Some(&pause_time) = timing.paused_at.last() {
            timing.total_pause_duration += resume_time.duration_since(pause_time);
        }

        Ok(())
    }

    /// Add checkpoint
    pub fn add_checkpoint(&self, name: String) -> ContextResult<()> {
        let mut timing = self.timing.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;
        timing.checkpoints.insert(name, Instant::now());
        Ok(())
    }

    /// Get execution timing
    pub fn get_execution_timing(&self) -> ContextResult<ExecutionTiming> {
        let timing = self.timing.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;
        Ok(timing.clone())
    }

    /// Get current runtime metrics
    pub fn get_runtime_metrics(&self) -> ContextResult<RuntimeMetrics> {
        let metrics = self.metrics.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        Ok(metrics.clone())
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, allocated: usize, deallocated: usize) -> ContextResult<()> {
        let mut tracker = self.memory_tracker.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire memory tracker lock: {}", e)))?;

        tracker.total_allocated += allocated;
        tracker.current_usage = tracker.current_usage.saturating_add(allocated).saturating_sub(deallocated);
        tracker.peak_usage = tracker.peak_usage.max(tracker.current_usage);

        if allocated > 0 {
            tracker.allocation_count += 1;
        }
        if deallocated > 0 {
            tracker.deallocation_count += 1;
        }

        Ok(())
    }

    /// Get memory usage
    pub fn get_memory_usage(&self) -> ContextResult<MemoryTracker> {
        let tracker = self.memory_tracker.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire memory tracker lock: {}", e)))?;
        Ok(tracker.clone())
    }

    /// Get process information
    pub fn get_process_info(&self) -> ContextResult<ProcessInfo> {
        let info = self.process_info.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire process info lock: {}", e)))?;
        Ok(info.clone())
    }

    /// Get thread manager
    pub fn get_thread_manager(&self) -> &ThreadManager {
        &self.thread_manager
    }

    /// Check resource limits
    pub fn check_resource_limits(&self) -> ContextResult<()> {
        let config = self.config.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire config lock: {}", e)))?;
        let tracker = self.memory_tracker.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire memory tracker lock: {}", e)))?;

        // Check memory limit
        if let Some(limit) = config.memory_limit {
            if tracker.current_usage > limit {
                return Err(ContextError::custom(
                    "memory_limit_exceeded",
                    format!("Memory usage {} exceeds limit {}", tracker.current_usage, limit)
                ));
            }
        }

        // Check execution time limit
        if let Some(limit) = config.max_execution_time {
            let timing = self.timing.lock()
                .map_err(|e| ContextError::internal(format!("Failed to acquire timing lock: {}", e)))?;

            if let Some(start) = timing.started_at {
                let elapsed = start.elapsed() - timing.total_pause_duration;
                if elapsed > limit {
                    return Err(ContextError::custom(
                        "execution_time_exceeded",
                        format!("Execution time {:?} exceeds limit {:?}", elapsed, limit)
                    ));
                }
            }
        }

        Ok(())
    }
}

impl ThreadManager {
    /// Create a new thread manager
    pub fn new() -> Self {
        Self {
            threads: Arc::new(RwLock::new(HashMap::new())),
            pool_config: Arc::new(RwLock::new(ThreadPoolConfig::default())),
            metrics: Arc::new(Mutex::new(ThreadMetrics::default())),
        }
    }

    /// Register a thread
    pub fn register_thread(&self, thread_id: ThreadId, name: Option<String>) -> ContextResult<()> {
        let mut threads = self.threads.write()
            .map_err(|e| ContextError::internal(format!("Failed to acquire threads lock: {}", e)))?;

        let thread_info = ThreadInfo {
            thread_id: unsafe { std::mem::transmute::<ThreadId, u64>(thread_id) },
            name,
            state: ThreadState::Running,
            priority: ThreadPriority::default(),
            created_at: SystemTime::now(),
            cpu_time: Duration::from_secs(0),
            stack_size: None,
        };

        threads.insert(thread_id, thread_info);

        // Update metrics
        let mut metrics = self.metrics.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.total_created += 1;
        metrics.active_count += 1;
        metrics.peak_count = metrics.peak_count.max(metrics.active_count);

        Ok(())
    }

    /// Unregister a thread
    pub fn unregister_thread(&self, thread_id: ThreadId) -> ContextResult<()> {
        let mut threads = self.threads.write()
            .map_err(|e| ContextError::internal(format!("Failed to acquire threads lock: {}", e)))?;

        threads.remove(&thread_id);

        // Update metrics
        let mut metrics = self.metrics.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        metrics.active_count = metrics.active_count.saturating_sub(1);

        Ok(())
    }

    /// Get thread information
    pub fn get_thread_info(&self, thread_id: ThreadId) -> ContextResult<Option<ThreadInfo>> {
        let threads = self.threads.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire threads lock: {}", e)))?;
        Ok(threads.get(&thread_id).cloned())
    }

    /// Get all active threads
    pub fn get_all_threads(&self) -> ContextResult<HashMap<ThreadId, ThreadInfo>> {
        let threads = self.threads.read()
            .map_err(|e| ContextError::internal(format!("Failed to acquire threads lock: {}", e)))?;
        Ok(threads.clone())
    }

    /// Get thread metrics
    pub fn get_thread_metrics(&self) -> ContextResult<ThreadMetrics> {
        let metrics = self.metrics.lock()
            .map_err(|e| ContextError::internal(format!("Failed to acquire metrics lock: {}", e)))?;
        Ok(metrics.clone())
    }
}

impl ExecutionContextTrait for RuntimeContext {
    fn id(&self) -> &str {
        &self.context_id
    }

    fn context_type(&self) -> ContextType {
        ContextType::Runtime
    }

    fn state(&self) -> ContextState {
        *self.state.read().unwrap()
    }

    fn is_active(&self) -> bool {
        matches!(self.state(), ContextState::Active)
    }

    fn metadata(&self) -> &ContextMetadata {
        // This is a simplified implementation
        // In practice, we'd need to handle this more carefully
        unsafe { &*(self.metadata.read().unwrap().as_ref() as *const ContextMetadata) }
    }

    fn validate(&self) -> Result<(), ContextError> {
        self.check_resource_limits()
    }

    fn clone_with_id(&self, new_id: String) -> Result<Box<dyn ExecutionContextTrait>, ContextError> {
        let config = self.get_config()?;
        let new_context = RuntimeContext::with_config(new_id, config)?;
        Ok(Box::new(new_context))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for ThreadManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_context_creation() {
        let context = RuntimeContext::new("test-runtime".to_string()).unwrap();
        assert_eq!(context.id(), "test-runtime");
        assert_eq!(context.context_type(), ContextType::Runtime);
        assert!(context.is_active());
    }

    #[test]
    fn test_environment_variable_management() {
        let context = RuntimeContext::new("test-env".to_string()).unwrap();

        // Set environment variable
        context.set_environment_variable("TEST_VAR".to_string(), "test_value".to_string()).unwrap();

        // Get environment variable
        let value = context.get_environment_variable("TEST_VAR").unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        // Remove environment variable
        let removed = context.remove_environment_variable("TEST_VAR").unwrap();
        assert_eq!(removed, Some("test_value".to_string()));

        // Check it's removed
        let value = context.get_environment_variable("TEST_VAR").unwrap();
        assert_eq!(value, None);
    }

    #[test]
    fn test_execution_timing() {
        let context = RuntimeContext::new("test-timing".to_string()).unwrap();

        // Start execution
        context.start_execution().unwrap();

        // Add checkpoint
        context.add_checkpoint("checkpoint1".to_string()).unwrap();

        // Pause and resume
        context.pause_execution().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        context.resume_execution().unwrap();

        // End execution
        let duration = context.end_execution().unwrap();
        assert!(duration > Duration::from_secs(0));

        // Check timing
        let timing = context.get_execution_timing().unwrap();
        assert!(timing.started_at.is_some());
        assert!(timing.ended_at.is_some());
        assert_eq!(timing.paused_at.len(), 1);
        assert_eq!(timing.resumed_at.len(), 1);
        assert!(timing.checkpoints.contains_key("checkpoint1"));
    }

    #[test]
    fn test_memory_tracking() {
        let context = RuntimeContext::new("test-memory".to_string()).unwrap();

        // Update memory usage
        context.update_memory_usage(1024, 0).unwrap();
        context.update_memory_usage(2048, 512).unwrap();

        // Check memory usage
        let usage = context.get_memory_usage().unwrap();
        assert_eq!(usage.total_allocated, 3072);
        assert_eq!(usage.current_usage, 2560);
        assert_eq!(usage.peak_usage, 2560);
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.deallocation_count, 1);
    }

    #[test]
    fn test_thread_manager() {
        let manager = ThreadManager::new();
        let thread_id = std::thread::current().id();

        // Register thread
        manager.register_thread(thread_id, Some("test-thread".to_string())).unwrap();

        // Get thread info
        let info = manager.get_thread_info(thread_id).unwrap();
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, Some("test-thread".to_string()));

        // Check metrics
        let metrics = manager.get_thread_metrics().unwrap();
        assert_eq!(metrics.total_created, 1);
        assert_eq!(metrics.active_count, 1);

        // Unregister thread
        manager.unregister_thread(thread_id).unwrap();

        // Check metrics after unregister
        let metrics = manager.get_thread_metrics().unwrap();
        assert_eq!(metrics.active_count, 0);
    }

    #[test]
    fn test_runtime_config() {
        let mut config = RuntimeConfig::default();
        config.max_execution_time = Some(Duration::from_secs(120));
        config.memory_limit = Some(1024 * 1024);

        let context = RuntimeContext::with_config("test-config".to_string(), config).unwrap();
        let retrieved_config = context.get_config().unwrap();

        assert_eq!(retrieved_config.max_execution_time, Some(Duration::from_secs(120)));
        assert_eq!(retrieved_config.memory_limit, Some(1024 * 1024));
    }
}