//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashMap;
use std::time::Instant;

use super::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_2 {
    use super::*;
    /// Test-only fixture device, clearly labeled as such. Unlike the
    /// pre-migration `init_devices()`, this is never returned from any
    /// production code path -- it exists purely so unit tests that exercise
    /// the allocation/kernel-bookkeeping/throughput-estimation *logic* (none
    /// of which touches real hardware regardless of whether the `id` came
    /// from `init_devices()` or here) do not depend on this test machine
    /// actually having a CUDA GPU.
    fn test_fixture_device(id: u32) -> GpuDevice {
        GpuDevice {
            id,
            name: "test-fixture-device".to_string(),
            memory_total: 10_737_418_240,
            memory_available: 9_663_676_416,
            compute_capability: (0, 0),
            multiprocessor_count: 16,
            clock_rate_mhz: 1_000,
            memory_bandwidth_bytes_per_sec: 100_000_000_000,
        }
    }
    #[test]
    fn test_gpu_utils_creation() {
        let utils = GpuUtils::new();
        assert!(utils.devices.is_empty());
        assert!(utils
            .allocations
            .read()
            .expect("operation should succeed")
            .is_empty());
    }
    /// `init_devices()` must never panic and must never fabricate a device.
    /// On this crate's dev/CI machines (macOS, no NVIDIA GPU, and/or built
    /// without the `gpu` feature) an empty list is the expected, correct
    /// result -- we deliberately do not assert `is_empty()` here so this
    /// test also passes unchanged on a real `--features gpu` GPU runner.
    #[test]
    fn test_device_initialization() {
        let mut utils = GpuUtils::new();
        assert!(utils.init_devices().is_ok());
        for device in &utils.devices {
            assert!(!device.name.is_empty());
            assert!(device.memory_total > 0);
        }
    }
    #[test]
    fn test_device_selection() {
        let mut utils = GpuUtils::new();
        utils.devices.push(test_fixture_device(0));
        utils.devices.push(test_fixture_device(1));
        let best_device = utils.get_best_device();
        assert!(best_device.is_some());
    }
    #[test]
    fn test_memory_allocation() {
        let mut utils = GpuUtils::new();
        utils.devices.push(test_fixture_device(0));
        let ptr = utils
            .allocate_memory(1024, 0, "test")
            .expect("operation should succeed");
        assert!(ptr > 0);
        assert!(utils.free_memory(ptr).is_ok());
    }
    #[test]
    fn test_kernel_execution() {
        let mut utils = GpuUtils::new();
        utils.devices.push(test_fixture_device(0));
        let kernel_info = GpuKernelInfo {
            name: "test_kernel".to_string(),
            device_id: 0,
            grid_size: (1, 1, 1),
            block_size: (256, 1, 1),
            shared_memory: 0,
            parameters: HashMap::new(),
        };
        let execution = utils
            .execute_kernel(&kernel_info)
            .expect("operation should succeed");
        assert_eq!(execution.kernel_name, "test_kernel");
        assert!(execution.execution_time > 0.0);
    }
    #[test]
    fn test_array_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = GpuArrayOps::add_arrays(&a, &b, 0).expect("operation should succeed");
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        let result = GpuArrayOps::multiply_arrays(&a, &b, 0).expect("operation should succeed");
        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }
    #[test]
    fn test_matrix_multiplication() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result =
            GpuArrayOps::matrix_multiply(&a, &b, 2, 2, 2, 0).expect("operation should succeed");
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }
    #[test]
    fn test_activation_functions() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let result = GpuArrayOps::apply_activation(&input, ActivationFunction::ReLU, 0)
            .expect("operation should succeed");
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
        let result = GpuArrayOps::apply_activation(&input, ActivationFunction::Sigmoid, 0)
            .expect("operation should succeed");
        assert!(result.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }
    #[test]
    fn test_reduction_operations() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = GpuArrayOps::reduce_sum(&input, 0).expect("operation should succeed");
        assert_eq!(sum, 15.0);
        let max = GpuArrayOps::reduce_max(&input, 0).expect("operation should succeed");
        assert_eq!(max, 5.0);
    }
    #[test]
    fn test_gpu_profiler() {
        let mut profiler = GpuProfiler::new();
        profiler.record_kernel_time("test_kernel", 1.5);
        profiler.record_kernel_time("test_kernel", 2.0);
        profiler.record_memory_transfer(1024, "host_to_device");
        let stats = profiler.get_kernel_stats();
        assert!(stats.contains_key("test_kernel"));
        assert_eq!(stats["test_kernel"].count, 2);
        assert_eq!(stats["test_kernel"].avg_time, 1.75);
        let mem_stats = profiler.get_memory_transfer_stats();
        assert_eq!(mem_stats.total_transfers, 1);
        assert_eq!(mem_stats.total_bytes, 1024);
    }
    #[test]
    fn test_throughput_estimation() {
        let mut utils = GpuUtils::new();
        utils.devices.push(test_fixture_device(0));
        let throughput = utils.estimate_throughput(0, 1000, "add");
        assert!(throughput > 0.0);
        let should_use = utils.should_use_gpu(1000, "add");
        assert!(should_use);
        let should_not_use = utils.should_use_gpu(100, "add");
        assert!(!should_not_use);
    }
    #[test]
    fn test_memory_stats() {
        let mut utils = GpuUtils::new();
        utils.devices.push(test_fixture_device(0));
        let _ptr = utils
            .allocate_memory(1024, 0, "test")
            .expect("operation should succeed");
        let stats = utils.get_memory_stats();
        assert!(stats.contains_key(&0));
        assert_eq!(stats[&0].allocated_memory, 1024);
        assert_eq!(stats[&0].num_allocations, 1);
    }
    #[test]
    fn test_error_handling() {
        let utils = GpuUtils::new();
        let result = utils.allocate_memory(1024, 999, "test");
        assert!(matches!(result, Err(GpuError::DeviceNotFound)));
        let result = utils.free_memory(0);
        assert!(matches!(result, Err(GpuError::InvalidPointer)));
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let result = GpuArrayOps::add_arrays(&a, &b, 0);
        assert!(matches!(result, Err(GpuError::ShapeMismatch)));
    }
    #[test]
    fn test_multi_gpu_coordinator() {
        let mut coordinator = MultiGpuCoordinator::new();
        let result = coordinator.init_all_gpus();
        assert!(result.is_ok() || matches!(result, Err(GpuError::InitializationFailed(_))));
        let workload = DistributedWorkload {
            total_elements: 10_000,
            operation_type: "matrix_multiply".to_string(),
            memory_requirement: 1024 * 1024,
            computation_complexity: 1.0,
        };
        let assignments = coordinator.get_optimal_assignment(&workload);
        assert!(!assignments.is_empty() || coordinator.gpus.is_empty());
    }
    #[test]
    fn test_distributed_operation() {
        let mut coordinator = MultiGpuCoordinator::new();
        let init_result = coordinator.init_all_gpus();
        let operation = DistributedOperation {
            kernel_name: "test_kernel".to_string(),
            workload: DistributedWorkload {
                total_elements: 1000,
                operation_type: "add".to_string(),
                memory_requirement: 4000,
                computation_complexity: 0.5,
            },
        };
        if init_result.is_ok() && !coordinator.gpus.is_empty() {
            let result = coordinator.execute_distributed(&operation);
            if let Ok(dist_result) = result {
                assert!(!dist_result.executions.is_empty());
                assert!(dist_result.total_time >= 0.0);
            } else {
                assert!(!coordinator.gpus.is_empty());
            }
        } else {
            assert!(coordinator.gpus.is_empty());
        }
    }
    #[test]
    fn test_cluster_memory_stats() {
        let mut coordinator = MultiGpuCoordinator::new();
        let _ = coordinator.init_all_gpus();
        let stats = coordinator.get_cluster_memory_stats();
        assert_eq!(stats.num_devices, coordinator.gpus.len());
        assert_eq!(stats.total_free, stats.total_memory - stats.total_allocated);
    }
    #[test]
    fn test_gpu_memory_pool() {
        let mut pool = GpuMemoryPool::new(AllocationStrategy::FirstFit);
        let ptr1 = pool.allocate(1024, 0);
        assert!(ptr1.is_ok());
        let ptr2 = pool.allocate(2048, 0);
        assert!(ptr2.is_ok());
        let free_result = pool.free(ptr1.expect("operation should succeed"), 0);
        assert!(free_result.is_ok());
        let defrag_result = pool.defragment(0);
        assert!(defrag_result.is_ok());
        let defrag = defrag_result.expect("operation should succeed");
        assert!(defrag.fragmentation_after <= defrag.fragmentation_before);
    }
    #[test]
    fn test_memory_pool_strategies() {
        let strategies = vec![
            AllocationStrategy::FirstFit,
            AllocationStrategy::BestFit,
            AllocationStrategy::WorstFit,
            AllocationStrategy::BuddySystem,
        ];
        for strategy in strategies {
            let mut pool = GpuMemoryPool::new(strategy);
            let ptr = pool.allocate(1024, 0);
            assert!(ptr.is_ok());
        }
    }
    #[test]
    fn test_async_gpu_operations() {
        let mut async_ops = AsyncGpuOps::new();
        let stream_id = async_ops.create_stream(0);
        assert!(stream_id.is_ok());
        let kernel_info = GpuKernelInfo {
            name: "async_test".to_string(),
            device_id: 0,
            grid_size: (1, 1, 1),
            block_size: (256, 1, 1),
            shared_memory: 0,
            parameters: HashMap::new(),
        };
        let handle = async_ops
            .launch_kernel_async(&kernel_info, stream_id.expect("operation should succeed"));
        assert!(handle.is_ok());
        let operation_handle = handle.expect("operation should succeed");
        let _is_complete_before = async_ops.is_complete(&operation_handle);
        let execution = async_ops.wait_for_completion(&operation_handle);
        assert!(execution.is_ok());
        let is_complete_after = async_ops.is_complete(&operation_handle);
        assert!(is_complete_after);
    }
    #[test]
    fn test_gpu_optimization_advisor() {
        let mut advisor = GpuOptimizationAdvisor::new();
        let execution = GpuKernelExecution {
            kernel_name: "test_kernel".to_string(),
            device_id: 0,
            grid_size: (10, 1, 1),
            block_size: (32, 1, 1),
            shared_memory: 0,
            execution_time: 5.0,
            parameters: HashMap::new(),
        };
        let recommendations = advisor.analyze_performance("test_kernel", &execution, 1000);
        assert!(!recommendations.is_empty());
        let has_grid_size_recommendation = recommendations
            .iter()
            .any(|r| r.rule_name.contains("Grid Size"));
        assert!(has_grid_size_recommendation);
    }
    #[test]
    fn test_load_balancer() {
        let balancer = LoadBalancer::new();
        let mut gpus = HashMap::new();
        let mut gpu1 = GpuUtils::new();
        let mut gpu2 = GpuUtils::new();
        let _ = gpu1.init_devices();
        let _ = gpu2.init_devices();
        gpus.insert(0, gpu1);
        gpus.insert(1, gpu2);
        let workload = DistributedWorkload {
            total_elements: 10_000,
            operation_type: "matrix_multiply".to_string(),
            memory_requirement: 1024 * 1024,
            computation_complexity: 1.0,
        };
        let assignments = balancer.assign_workload(&workload, &gpus);
        assert_eq!(assignments.len(), gpus.len());
        let total_elements: u32 = assignments
            .iter()
            .map(|a| a.grid_size.0 * a.block_size.0)
            .sum();
        assert!(total_elements > 0);
    }
    #[test]
    fn test_stream_priorities() {
        let mut async_ops = AsyncGpuOps::new();
        let _stream_id = async_ops
            .create_stream(0)
            .expect("operation should succeed");
        let streams = async_ops.streams.get(&0).expect("operation should succeed");
        assert_eq!(streams.len(), 1);
        assert!(matches!(streams[0].priority, StreamPriority::Normal));
    }
    #[test]
    fn test_memory_block_operations() {
        let block1 = MemoryBlock {
            ptr: 1000,
            size: 1024,
            is_allocated: false,
            allocation_time: None,
        };
        let block2 = MemoryBlock {
            ptr: 2024,
            size: 2048,
            is_allocated: true,
            allocation_time: Some(Instant::now()),
        };
        assert!(!block1.is_allocated);
        assert!(block2.is_allocated);
        assert!(block1.allocation_time.is_none());
        assert!(block2.allocation_time.is_some());
    }
    #[test]
    fn test_distributed_workload() {
        let workload = DistributedWorkload {
            total_elements: 1_000_000,
            operation_type: "fft".to_string(),
            memory_requirement: 8 * 1_000_000,
            computation_complexity: 2.5,
        };
        assert_eq!(workload.total_elements, 1_000_000);
        assert_eq!(workload.operation_type, "fft");
        assert!(workload.computation_complexity > 1.0);
    }
    #[test]
    fn test_communication_topology() {
        let ring_topology = CommunicationTopology::Ring;
        let tree_topology = CommunicationTopology::Tree;
        let all_to_all_topology = CommunicationTopology::AllToAll;
        let custom_topology =
            CommunicationTopology::Custom(vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2]]);
        match ring_topology {
            CommunicationTopology::Ring => {}
            _ => panic!(),
        }
        match tree_topology {
            CommunicationTopology::Tree => {}
            _ => panic!(),
        }
        match all_to_all_topology {
            CommunicationTopology::AllToAll => {}
            _ => panic!(),
        }
        match custom_topology {
            CommunicationTopology::Custom(_) => {}
            _ => panic!(),
        }
    }
    #[test]
    fn test_synchronization_barrier() {
        let barrier = SynchronizationBarrier {
            id: 1,
            participating_gpus: vec![0, 1, 2, 3],
            barrier_type: BarrierType::Global,
        };
        assert_eq!(barrier.id, 1);
        assert_eq!(barrier.participating_gpus.len(), 4);
        assert!(matches!(barrier.barrier_type, BarrierType::Global));
    }
    #[test]
    fn test_optimization_recommendation_priorities() {
        let low_priority = RecommendationPriority::Low;
        let medium_priority = RecommendationPriority::Medium;
        let high_priority = RecommendationPriority::High;
        let critical_priority = RecommendationPriority::Critical;
        match low_priority {
            RecommendationPriority::Low => {}
            _ => panic!(),
        }
        match medium_priority {
            RecommendationPriority::Medium => {}
            _ => panic!(),
        }
        match high_priority {
            RecommendationPriority::High => {}
            _ => panic!(),
        }
        match critical_priority {
            RecommendationPriority::Critical => {}
            _ => panic!(),
        }
    }
    #[test]
    fn test_performance_metric_calculations() {
        let metric = PerformanceMetric {
            execution_time: 10.0,
            throughput: 1000.0,
            memory_bandwidth: 0.8,
            occupancy: 0.75,
        };
        assert!(metric.execution_time > 0.0);
        assert!(metric.throughput > 0.0);
        assert!(metric.memory_bandwidth <= 1.0);
        assert!(metric.occupancy <= 1.0);
    }
    #[test]
    fn test_operation_status_transitions() {
        let mut operation = AsyncOperation {
            id: 0,
            kernel_info: GpuKernelInfo {
                name: "test".to_string(),
                device_id: 0,
                grid_size: (1, 1, 1),
                block_size: (1, 1, 1),
                shared_memory: 0,
                parameters: HashMap::new(),
            },
            stream_id: 0,
            start_time: Instant::now(),
            status: OperationStatus::Pending,
        };
        assert!(matches!(operation.status, OperationStatus::Pending));
        operation.status = OperationStatus::Running;
        assert!(matches!(operation.status, OperationStatus::Running));
        operation.status = OperationStatus::Completed;
        assert!(matches!(operation.status, OperationStatus::Completed));
    }
}
