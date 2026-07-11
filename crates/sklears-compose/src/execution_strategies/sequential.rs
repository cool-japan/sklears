//! Sequential (single-threaded, deterministic) execution strategy.

use crate::task_definitions::{ExecutionTask, TaskRequirements, TaskResult, TaskStatus};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use super::core::{
    ExecutionStrategy, HealthStatus, PerformanceSummary, ResourceUtilization, StrategyConfig,
    StrategyHealth, StrategyMetrics, StrategyState,
};

/// Sequential execution strategy for deterministic, single-threaded execution
#[derive(Debug)]
#[allow(dead_code)]
pub struct SequentialExecutionStrategy {
    /// Strategy configuration
    pub(super) config: StrategyConfig,
    /// Current task queue
    pub(super) task_queue: Arc<Mutex<VecDeque<ExecutionTask>>>,
    /// Execution metrics
    pub(super) metrics: Arc<Mutex<StrategyMetrics>>,
    /// Strategy state
    pub(super) state: Arc<RwLock<StrategyState>>,
    /// Profiling enabled
    pub(super) enable_profiling: bool,
    /// Debugging enabled
    pub(super) enable_debugging: bool,
    /// Checkpoint interval
    pub(super) checkpoint_interval: Option<Duration>,
}
impl SequentialExecutionStrategy {
    /// Create a new sequential execution strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(Mutex::new(StrategyMetrics::default())),
            state: Arc::new(RwLock::new(StrategyState::default())),
            enable_profiling: false,
            enable_debugging: false,
            checkpoint_interval: None,
        }
    }
    /// Create a builder for sequential strategy
    #[must_use]
    pub fn builder() -> SequentialStrategyBuilder {
        SequentialStrategyBuilder::new()
    }
}
impl Default for SequentialExecutionStrategy {
    fn default() -> Self {
        Self::new()
    }
}
impl ExecutionStrategy for SequentialExecutionStrategy {
    fn name(&self) -> &'static str {
        "sequential"
    }
    fn description(&self) -> &'static str {
        "Sequential single-threaded execution strategy"
    }
    fn config(&self) -> &StrategyConfig {
        &self.config
    }
    fn configure(
        &mut self,
        config: StrategyConfig,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            self.config = config;
            Ok(())
        })
    }
    fn initialize(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());
            state.initialized = true;
            state.running = true;
            Ok(())
        })
    }
    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>> {
        Box::pin(async move {
            let start_time = SystemTime::now();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let end_time = SystemTime::now();
            let duration = end_time.duration_since(start_time).unwrap_or_default();
            Ok(TaskResult {
                task_id: task.metadata.id.clone(),
                status: TaskStatus::Completed,
                output: None,
                metrics: crate::task_definitions::TaskExecutionMetrics {
                    start_time,
                    end_time: Some(end_time),
                    duration: Some(duration),
                    queue_wait_time: Duration::from_millis(0),
                    scheduling_time: Duration::from_millis(0),
                    setup_time: Duration::from_millis(0),
                    cleanup_time: Duration::from_millis(0),
                    retry_attempts: 0,
                    checkpoint_count: 0,
                    completion_percentage: 100.0,
                    efficiency_score: Some(0.95),
                },
                resource_usage: crate::task_definitions::TaskResourceUsage {
                    cpu_time: 0.1,
                    memory_usage: 80 * 1024 * 1024,
                    peak_memory_usage: 100 * 1024 * 1024,
                    disk_io_operations: 7,
                    network_usage: 3072,
                    gpu_usage: None,
                    gpu_memory_usage: None,
                },
                performance_metrics: crate::task_definitions::TaskPerformanceMetrics {
                    operations_per_second: 10.0,
                    throughput: 10.0,
                    latency: duration,
                    error_rate: 0.0,
                    cache_hit_rate: Some(0.8),
                    efficiency_score: 0.95,
                },
                error: None,
                logs: Vec::new(),
                artifacts: Vec::new(),
                execution_time: Some(duration),
                metadata: std::collections::HashMap::new(),
            })
        })
    }
    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>> {
        Box::pin(async move {
            let mut results = Vec::new();
            for task in tasks {
                let result = self.execute_task(task).await?;
                results.push(result);
            }
            Ok(results)
        })
    }
    fn can_handle(&self, _task: &ExecutionTask) -> bool {
        true
    }
    fn estimate_execution_time(&self, _task: &ExecutionTask) -> Option<Duration> {
        Some(Duration::from_millis(100))
    }
    fn health_status(&self) -> StrategyHealth {
        StrategyHealth {
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            score: 1.0,
            issues: Vec::new(),
            resource_utilization: ResourceUtilization {
                cpu: 50.0,
                memory: 60.0,
                gpu: None,
                network: 20.0,
                storage: 30.0,
            },
            performance_summary: PerformanceSummary {
                tasks_completed: 100,
                tasks_failed: 0,
                avg_execution_time: Duration::from_millis(100),
                throughput: 10.0,
                error_rate: 0.0,
            },
        }
    }
    fn metrics(&self) -> StrategyMetrics {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    fn shutdown(&mut self) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());
            state.running = false;
            state.initialized = false;
            Ok(())
        })
    }
    fn pause(&mut self) -> SklResult<()> {
        let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());
        state.paused = true;
        Ok(())
    }
    fn resume(&mut self) -> SklResult<()> {
        let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());
        state.paused = false;
        Ok(())
    }
    fn scale(
        &mut self,
        _scale_factor: f64,
    ) -> Pin<Box<dyn Future<Output = SklResult<()>> + Send + '_>> {
        Box::pin(async move {
            Err(SklearsError::InvalidOperation(
                "Sequential strategy does not support scaling".to_string(),
            ))
        })
    }
    fn get_resource_requirements(&self, task: &ExecutionTask) -> TaskRequirements {
        task.requirements.clone()
    }
    fn validate_task(&self, task: &ExecutionTask) -> SklResult<()> {
        if task.metadata.name.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Task name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}
/// Builder for sequential execution strategy
pub struct SequentialStrategyBuilder {
    pub(super) enable_profiling: bool,
    pub(super) enable_debugging: bool,
    pub(super) checkpoint_interval: Option<Duration>,
    pub(super) config: StrategyConfig,
}
impl SequentialStrategyBuilder {
    #[must_use]
    /// Creates a new instance.
    pub fn new() -> Self {
        Self {
            enable_profiling: false,
            enable_debugging: false,
            checkpoint_interval: None,
            config: StrategyConfig::default(),
        }
    }
    #[must_use]
    /// Performs enable profiling.
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }
    #[must_use]
    /// Performs enable debugging.
    pub fn enable_debugging(mut self, enable: bool) -> Self {
        self.enable_debugging = enable;
        self
    }
    #[must_use]
    /// Performs checkpoint interval.
    pub fn checkpoint_interval(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }
    #[must_use]
    /// Performs config.
    pub fn config(mut self, config: StrategyConfig) -> Self {
        self.config = config;
        self
    }
    #[must_use]
    /// Builds and returns the result.
    pub fn build(self) -> SequentialExecutionStrategy {
        SequentialExecutionStrategy {
            config: self.config,
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(Mutex::new(StrategyMetrics::default())),
            state: Arc::new(RwLock::new(StrategyState::default())),
            enable_profiling: self.enable_profiling,
            enable_debugging: self.enable_debugging,
            checkpoint_interval: self.checkpoint_interval,
        }
    }
}
impl Default for SequentialStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}
