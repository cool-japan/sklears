//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::task_definitions::TaskStatus;
use std::time::{Duration, SystemTime};

use super::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_2 {
    use super::*;
    #[test]
    fn test_strategy_config() {
        let config = StrategyConfig::default();
        assert_eq!(config.name, "default_strategy");
        assert_eq!(config.max_concurrent_tasks, 10);
        assert_eq!(config.priority, StrategyPriority::Normal);
    }
    #[test]
    fn test_sequential_strategy_creation() {
        let strategy = SequentialExecutionStrategy::new();
        assert_eq!(strategy.name(), "sequential");
        assert_eq!(
            strategy.description(),
            "Sequential single-threaded execution strategy"
        );
    }
    #[test]
    fn test_sequential_strategy_builder() {
        let strategy = SequentialExecutionStrategy::builder()
            .enable_profiling(true)
            .enable_debugging(true)
            .checkpoint_interval(Duration::from_secs(60))
            .build();
        assert!(strategy.enable_profiling);
        assert!(strategy.enable_debugging);
        assert_eq!(strategy.checkpoint_interval, Some(Duration::from_secs(60)));
    }
    #[test]
    fn test_batch_strategy_builder() {
        let strategy = BatchExecutionStrategy::builder()
            .batch_size(50)
            .max_batch_size(500)
            .parallel_batches(4)
            .enable_adaptive_batching(true)
            .build();
        assert_eq!(strategy.batch_size, 50);
        assert_eq!(strategy.max_batch_size, 500);
        assert_eq!(strategy.parallel_batches, 4);
        assert!(strategy.adaptive_batching);
    }
    #[test]
    fn test_strategy_factory() {
        let config = StrategyConfig::default();
        let result = StrategyFactory::create_strategy(StrategyType::Sequential, config);
        assert!(result.is_ok());
        let available = StrategyFactory::available_strategies();
        assert!(!available.is_empty());
        assert!(available.contains(&StrategyType::Sequential));
    }
    #[test]
    fn test_strategy_registry() {
        let mut registry = StrategyRegistry::new();
        let strategy = SequentialExecutionStrategy::new();
        let result = registry.register("seq".to_string(), Box::new(strategy));
        assert!(result.is_ok());
        assert!(registry.get("seq").is_some());
        assert_eq!(registry.list().len(), 1);
        let result = registry.set_default("seq".to_string());
        assert!(result.is_ok());
        assert_eq!(registry.get_default(), Some(&"seq".to_string()));
    }
    #[test]
    fn test_strategy_health() {
        let health = StrategyHealth {
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            score: 0.95,
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
                tasks_failed: 2,
                avg_execution_time: Duration::from_millis(150),
                throughput: 50.0,
                error_rate: 0.02,
            },
        };
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.score, 0.95);
        assert_eq!(health.performance_summary.error_rate, 0.02);
    }
    #[tokio::test]
    #[cfg_attr(
        miri,
        ignore = "constructs a tokio runtime with an IO driver; kqueue is unsupported by Miri on macOS"
    )]
    async fn test_sequential_strategy_execution() {
        let strategy = SequentialExecutionStrategy::new();
        let task = crate::task_definitions::ExecutionTask::builder()
            .name("test_task")
            .task_type(crate::task_definitions::TaskType::Preprocess)
            .build();
        let result = strategy.execute_task(task).await;
        assert!(result.is_ok());
        let task_result = result.expect("operation should succeed");
        assert_eq!(task_result.status, TaskStatus::Completed);
        assert!(task_result.metrics.duration.is_some());
    }
}
