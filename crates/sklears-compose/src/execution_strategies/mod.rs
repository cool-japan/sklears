//! Execution Strategies for Composable Execution Engine
//!
//! This module provides a comprehensive collection of execution strategies that can be
//! dynamically selected and configured based on workload characteristics, resource
//! availability, and performance requirements. Each strategy implements the
//! `ExecutionStrategy` trait and provides specialized optimization for different
//! execution patterns and environments.
//!
//! # Strategy Architecture
//!
//! The execution strategy system is built around pluggable strategy implementations:
//!
//! ```text
//! ExecutionStrategy (trait)
//! ├── SequentialExecutionStrategy    // Single-threaded, deterministic execution
//! ├── BatchExecutionStrategy         // High-throughput batch processing
//! ├── StreamingExecutionStrategy     // Real-time streaming and low-latency
//! ├── GpuExecutionStrategy          // GPU-accelerated computation
//! ├── DistributedExecutionStrategy   // Multi-node distributed execution
//! └── EventDrivenExecutionStrategy   // Reactive event-based execution
//! ```
//!
//! # Strategy Selection Guide
//!
//! ## `SequentialExecutionStrategy`
//! **Best for**: Development, debugging, deterministic workflows
//! - Single-threaded execution
//! - Predictable resource usage
//! - Easy debugging and profiling
//! - Deterministic results
//!
//! ## `BatchExecutionStrategy`
//! **Best for**: ETL pipelines, batch ML training, bulk data processing
//! - High-throughput processing
//! - Optimal resource utilization
//! - Batch size optimization
//! - Parallel task execution
//!
//! ## `StreamingExecutionStrategy`
//! **Best for**: Real-time inference, live data processing, low-latency requirements
//! - Continuous data processing
//! - Low-latency guarantees
//! - Backpressure handling
//! - Stream buffering
//!
//! ## `GpuExecutionStrategy`
//! **Best for**: Deep learning, matrix operations, parallel computations
//! - GPU acceleration
//! - CUDA/ROCm optimization
//! - Memory management
//! - Multi-GPU support
//!
//! ## `DistributedExecutionStrategy`
//! **Best for**: Large-scale processing, cluster computing, fault tolerance
//! - Multi-node execution
//! - Automatic load balancing
//! - Fault tolerance
//! - Dynamic scaling
//!
//! ## `EventDrivenExecutionStrategy`
//! **Best for**: Microservices, reactive systems, event processing
//! - Asynchronous execution
//! - Event-based triggers
//! - Resource efficiency
//! - Scalable architecture
//!
//! # Usage Examples
//!
//! ## Sequential Strategy for Development
//! ```rust,ignore
//! use sklears_compose::execution_strategies::*;
//!
//! let strategy = SequentialExecutionStrategy::builder()
//!     .enable_profiling(true)
//!     .enable_debugging(true)
//!     .checkpoint_interval(Duration::from_secs(60))
//!     .build();
//!
//! let config = StrategyConfig {
//!     max_concurrent_tasks: 1,
//!     timeout: Some(Duration::from_secs(3600)),
//!     enable_metrics: true,
//!     ..Default::default()
//! };
//!
//! strategy.configure(config).await?;
//! ```
//!
//! ## Batch Strategy for High-Throughput Processing
//! ```rust,ignore
//! let batch_strategy = BatchExecutionStrategy::builder()
//!     .batch_size(100)
//!     .max_batch_size(1000)
//!     .batch_timeout(Duration::from_secs(30))
//!     .parallel_batches(4)
//!     .enable_adaptive_batching(true)
//!     .build();
//!
//! let config = StrategyConfig {
//!     max_concurrent_tasks: 400, // 4 batches * 100 tasks
//!     resource_constraints: ResourceConstraints {
//!         max_cpu_cores: Some(16),
//!         max_memory: Some(64 * 1024 * 1024 * 1024), // 64GB
//!         ..Default::default()
//!     },
//!     performance_goals: PerformanceGoals {
//!         target_throughput: Some(10000.0), // 10k tasks/sec
//!         target_utilization: Some(90.0),
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## Streaming Strategy for Real-Time Processing
//! ```rust,ignore
//! let streaming_strategy = StreamingExecutionStrategy::builder()
//!     .buffer_size(1000)
//!     .max_latency(Duration::from_millis(10))
//!     .backpressure_strategy(BackpressureStrategy::DropOldest)
//!     .enable_flow_control(true)
//!     .watermark_interval(Duration::from_millis(100))
//!     .build();
//!
//! let config = StrategyConfig {
//!     performance_goals: PerformanceGoals {
//!         target_latency: Some(5.0), // 5ms target
//!         target_throughput: Some(100000.0), // 100k/sec
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## GPU Strategy for Hardware Acceleration
//! ```rust,ignore
//! let gpu_strategy = GpuExecutionStrategy::builder()
//!     .devices(vec!["cuda:0".to_string(), "cuda:1".to_string()])
//!     .memory_pool_size(8 * 1024 * 1024 * 1024) // 8GB pool
//!     .enable_memory_optimization(true)
//!     .enable_mixed_precision(true)
//!     .compute_stream_count(4)
//!     .build();
//!
//! let config = StrategyConfig {
//!     resource_constraints: ResourceConstraints {
//!         gpu_devices: Some(2),
//!         gpu_memory: Some(16 * 1024 * 1024 * 1024), // 16GB total
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```
//!
//! ## Distributed Strategy for Cluster Computing
//! ```rust,ignore
//! let distributed_strategy = DistributedExecutionStrategy::builder()
//!     .nodes(vec![
//!         "worker1:8080".to_string(),
//!         "worker2:8080".to_string(),
//!         "worker3:8080".to_string(),
//!     ])
//!     .replication_factor(2)
//!     .enable_auto_scaling(true)
//!     .load_balancing_strategy(LoadBalancingStrategy::RoundRobin)
//!     .enable_fault_tolerance(true)
//!     .build();
//!
//! let config = StrategyConfig {
//!     fault_tolerance: FaultToleranceConfig {
//!         enable_retry: true,
//!         max_retries: 3,
//!         enable_failover: true,
//!         enable_circuit_breaker: true,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```

pub mod batch;
pub mod core;
pub mod distributed;
pub mod event_driven;
pub mod gpu;
pub mod registry;
pub mod sequential;
pub mod streaming;

// Re-export all types
pub use batch::*;
pub use core::*;
pub use distributed::*;
pub use event_driven::*;
pub use gpu::*;
pub use registry::*;
pub use sequential::*;
pub use streaming::*;

#[cfg(test)]
mod tests;
