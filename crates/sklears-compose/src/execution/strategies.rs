//! Execution strategies for the composable execution engine
//!
//! This module provides pluggable execution strategies that can be configured
//! and used by the execution engine for different workload types.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant, SystemTime};

use sklears_core::error::Result as SklResult;

use super::config::{
    CachingStrategy, LoadBalancingAlgorithm, LoadBalancingConfig, OptimizationLevel,
    ParameterValue, PerformanceTuning, PrefetchingStrategy, StrategyConfig,
    StrategyResourceAllocation,
};
use super::tasks::{ExecutionTask, ResourceUsage, TaskResult, TaskStatus, TaskType};

/// Execution strategy trait for pluggable execution behaviors
pub trait ExecutionStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;

    /// Strategy description
    fn description(&self) -> &str;

    /// Execute a single task
    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>>;

    /// Execute multiple tasks
    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>>;

    /// Get strategy configuration
    fn get_config(&self) -> StrategyConfig;

    /// Update strategy configuration
    fn update_config(&mut self, config: StrategyConfig) -> SklResult<()>;

    /// Get strategy metrics
    fn get_metrics(&self) -> StrategyMetrics;

    /// Check strategy health
    fn health_check(&self) -> StrategyHealth;
}

/// Strategy metrics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Total tasks executed
    pub tasks_executed: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average task execution time
    pub average_execution_time: Duration,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Error count
    pub error_count: u64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Last updated timestamp
    pub last_updated: SystemTime,
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            tasks_executed: 0,
            total_execution_time: Duration::ZERO,
            average_execution_time: Duration::ZERO,
            success_rate: 1.0,
            error_count: 0,
            resource_utilization: ResourceUtilization::default(),
            last_updated: SystemTime::now(),
        }
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization percentage (0.0-100.0)
    pub cpu_percent: f64,
    /// Memory utilization percentage (0.0-100.0)
    pub memory_percent: f64,
    /// I/O utilization percentage (0.0-100.0)
    pub io_percent: f64,
    /// Network utilization percentage (0.0-100.0)
    pub network_percent: f64,
    /// Queue utilization percentage (0.0-100.0)
    pub queue_percent: f64,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            io_percent: 0.0,
            network_percent: 0.0,
            queue_percent: 0.0,
        }
    }
}

/// Strategy health status
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyHealth {
    /// Strategy is healthy and operating normally
    Healthy,
    /// Strategy is degraded but still operational
    Degraded { reason: String },
    /// Strategy is unhealthy and should not be used
    Unhealthy { reason: String },
    /// Strategy health is unknown
    Unknown,
}

/// Sequential execution strategy - executes tasks one by one
pub struct SequentialStrategy {
    config: StrategyConfig,
    metrics: StrategyMetrics,
    start_time: Instant,
}

impl SequentialStrategy {
    #[must_use]
    pub fn new() -> Self {
        let config = StrategyConfig {
            name: "sequential".to_string(),
            parameters: HashMap::new(),
            resource_allocation: StrategyResourceAllocation {
                cpu_cores: 1.0,
                memory_bytes: 100_000_000, // 100MB
                priority: 1,
            },
            performance_tuning: PerformanceTuning {
                optimization_level: OptimizationLevel::Low,
                prefetching: PrefetchingStrategy::None,
                caching: CachingStrategy::None,
                load_balancing: LoadBalancingConfig {
                    enabled: false,
                    algorithm: LoadBalancingAlgorithm::RoundRobin,
                    rebalance_threshold: 0.8,
                    min_load_difference: 0.1,
                },
            },
        };

        Self {
            config,
            metrics: StrategyMetrics::default(),
            start_time: Instant::now(),
        }
    }

    async fn execute_task_impl(&self, task: ExecutionTask) -> SklResult<TaskResult> {
        let start_time = Instant::now();

        // Simple task execution simulation
        match task.task_type {
            TaskType::Computation => {
                // Simulate computation work
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Completed,
                    execution_time: start_time.elapsed(),
                    resource_usage: ResourceUsage::default(),
                    output: None,
                    error: None,
                })
            }
            TaskType::IoOperation => {
                // Simulate I/O work
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Completed,
                    execution_time: start_time.elapsed(),
                    resource_usage: ResourceUsage::default(),
                    output: None,
                    error: None,
                })
            }
            TaskType::NetworkOperation => {
                // Simulate network work
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Completed,
                    execution_time: start_time.elapsed(),
                    resource_usage: ResourceUsage::default(),
                    output: None,
                    error: None,
                })
            }
            TaskType::Custom => {
                // Custom task handling
                tokio::time::sleep(Duration::from_millis(25)).await;
                Ok(TaskResult {
                    task_id: task.id,
                    status: TaskStatus::Completed,
                    execution_time: start_time.elapsed(),
                    resource_usage: ResourceUsage::default(),
                    output: None,
                    error: None,
                })
            }
        }
    }
}

impl ExecutionStrategy for SequentialStrategy {
    fn name(&self) -> &'static str {
        "sequential"
    }

    fn description(&self) -> &'static str {
        "Sequential execution strategy - executes tasks one by one"
    }

    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>> {
        Box::pin(self.execute_task_impl(task))
    }

    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>> {
        Box::pin(async move {
            let mut results = Vec::with_capacity(tasks.len());

            for task in tasks {
                let result = self.execute_task_impl(task).await?;
                results.push(result);
            }

            Ok(results)
        })
    }

    fn get_config(&self) -> StrategyConfig {
        self.config.clone()
    }

    fn update_config(&mut self, config: StrategyConfig) -> SklResult<()> {
        self.config = config;
        Ok(())
    }

    fn get_metrics(&self) -> StrategyMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> StrategyHealth {
        let uptime = self.start_time.elapsed();

        if uptime > Duration::from_secs(24 * 3600) && self.metrics.success_rate > 0.95 {
            StrategyHealth::Healthy
        } else if self.metrics.success_rate > 0.8 {
            StrategyHealth::Degraded {
                reason: "Lower than optimal success rate".to_string(),
            }
        } else {
            StrategyHealth::Unhealthy {
                reason: "Poor success rate".to_string(),
            }
        }
    }
}

/// Parallel execution strategy - executes tasks concurrently
pub struct ParallelStrategy {
    config: StrategyConfig,
    metrics: StrategyMetrics,
    max_concurrent_tasks: usize,
    start_time: Instant,
}

impl ParallelStrategy {
    #[must_use]
    pub fn new(max_concurrent_tasks: usize) -> Self {
        let config = StrategyConfig {
            name: "parallel".to_string(),
            parameters: HashMap::from([(
                "max_concurrent_tasks".to_string(),
                ParameterValue::Integer(max_concurrent_tasks as i64),
            )]),
            resource_allocation: StrategyResourceAllocation {
                cpu_cores: max_concurrent_tasks as f64,
                memory_bytes: max_concurrent_tasks as u64 * 100_000_000, // 100MB per task
                priority: 2,
            },
            performance_tuning: PerformanceTuning {
                optimization_level: OptimizationLevel::High,
                prefetching: PrefetchingStrategy::Adaptive,
                caching: CachingStrategy::LRU,
                load_balancing: LoadBalancingConfig {
                    enabled: true,
                    algorithm: LoadBalancingAlgorithm::LeastLoaded,
                    rebalance_threshold: 0.7,
                    min_load_difference: 0.2,
                },
            },
        };

        Self {
            config,
            metrics: StrategyMetrics::default(),
            max_concurrent_tasks,
            start_time: Instant::now(),
        }
    }

    async fn execute_task_impl(&self, task: ExecutionTask) -> SklResult<TaskResult> {
        let start_time = Instant::now();

        match task.task_type {
            TaskType::Computation => {
                tokio::time::sleep(Duration::from_millis(5)).await; // Faster with parallelism
            }
            TaskType::IoOperation => {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            TaskType::NetworkOperation => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            TaskType::Custom => {
                tokio::time::sleep(Duration::from_millis(12)).await;
            }
        }

        Ok(TaskResult {
            task_id: task.id,
            status: TaskStatus::Completed,
            execution_time: start_time.elapsed(),
            resource_usage: ResourceUsage::default(),
            output: None,
            error: None,
        })
    }
}

impl ExecutionStrategy for ParallelStrategy {
    fn name(&self) -> &'static str {
        "parallel"
    }

    fn description(&self) -> &'static str {
        "Parallel execution strategy - executes tasks concurrently"
    }

    fn execute_task(
        &self,
        task: ExecutionTask,
    ) -> Pin<Box<dyn Future<Output = SklResult<TaskResult>> + Send + '_>> {
        Box::pin(self.execute_task_impl(task))
    }

    fn execute_batch(
        &self,
        tasks: Vec<ExecutionTask>,
    ) -> Pin<Box<dyn Future<Output = SklResult<Vec<TaskResult>>> + Send + '_>> {
        Box::pin(async move {
            use futures::stream::{self, StreamExt};

            let results: Result<Vec<_>, _> = stream::iter(tasks)
                .map(|task| self.execute_task_impl(task))
                .buffer_unordered(self.max_concurrent_tasks)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect();

            results
        })
    }

    fn get_config(&self) -> StrategyConfig {
        self.config.clone()
    }

    fn update_config(&mut self, config: StrategyConfig) -> SklResult<()> {
        // Update max_concurrent_tasks if provided in parameters
        if let Some(ParameterValue::Integer(max_tasks)) =
            config.parameters.get("max_concurrent_tasks")
        {
            self.max_concurrent_tasks = *max_tasks as usize;
        }

        self.config = config;
        Ok(())
    }

    fn get_metrics(&self) -> StrategyMetrics {
        self.metrics.clone()
    }

    fn health_check(&self) -> StrategyHealth {
        if self.metrics.resource_utilization.cpu_percent > 90.0 {
            StrategyHealth::Degraded {
                reason: "High CPU utilization".to_string(),
            }
        } else if self.metrics.success_rate > 0.95 {
            StrategyHealth::Healthy
        } else {
            StrategyHealth::Unhealthy {
                reason: "Poor success rate".to_string(),
            }
        }
    }
}

impl Default for SequentialStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::new(4) // Default to 4 concurrent tasks
    }
}
