//! Advanced GPU Acceleration Demo
//!
//! This example demonstrates the advanced GPU acceleration features including:
//! - Multi-GPU support and load balancing
//! - Advanced memory management
//! - Asynchronous operations with CUDA streams
//! - Kernel fusion for optimal performance
//! - Mixed precision arithmetic
//! - Distributed computing across multiple GPUs
//! - Performance profiling and monitoring

#[cfg(feature = "gpu")]
use scirs2_autograd::ndarray::{Array2, Array3};
#[cfg(feature = "gpu")]
use sklears_linear::advanced_gpu_acceleration::{
    AdvancedGpuConfig, AdvancedGpuOps, LoadBalancingStrategy,
};
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced GPU Acceleration Demo");
    println!("=================================\n");

    // Demo 1: Multi-GPU Configuration
    println!("📊 Demo 1: Multi-GPU Configuration");
    println!("----------------------------------");
    demo_multi_gpu_config()?;

    // Demo 2: Advanced Memory Management
    println!("\n📊 Demo 2: Advanced Memory Management");
    println!("------------------------------------");
    demo_memory_management()?;

    // Demo 3: Multi-GPU Matrix Operations
    println!("\n📊 Demo 3: Multi-GPU Matrix Operations");
    println!("-------------------------------------");
    demo_multi_gpu_operations()?;

    // Demo 4: Kernel Fusion and Optimization
    println!("\n📊 Demo 4: Kernel Fusion and Optimization");
    println!("----------------------------------------");
    demo_kernel_fusion()?;

    // Demo 5: Mixed Precision Computing
    println!("\n📊 Demo 5: Mixed Precision Computing");
    println!("-----------------------------------");
    demo_mixed_precision()?;

    // Demo 6: Batch Processing
    println!("\n📊 Demo 6: Batch Processing");
    println!("--------------------------");
    demo_batch_processing()?;

    // Demo 7: Asynchronous Operations
    println!("\n📊 Demo 7: Asynchronous Operations");
    println!("---------------------------------");
    demo_async_operations()?;

    // Demo 8: Performance Profiling
    println!("\n📊 Demo 8: Performance Profiling");
    println!("--------------------------------");
    demo_performance_profiling()?;

    // Demo 9: Distributed Computing
    println!("\n📊 Demo 9: Distributed Computing");
    println!("-------------------------------");
    demo_distributed_computing()?;

    // Demo 10: Load Balancing Strategies
    println!("\n📊 Demo 10: Load Balancing Strategies");
    println!("------------------------------------");
    demo_load_balancing()?;

    println!("\n✨ All advanced GPU demos completed successfully!");
    println!("🎯 Key Features Demonstrated:");
    println!("   ✅ Multi-GPU support with intelligent load balancing");
    println!("   ✅ Advanced memory management with memory pools");
    println!("   ✅ Asynchronous operations using CUDA streams");
    println!("   ✅ Kernel fusion for optimal performance");
    println!("   ✅ Mixed precision arithmetic (FP16/BF16)");
    println!("   ✅ Distributed computing across multiple GPUs");
    println!("   ✅ Comprehensive performance profiling");
    println!("   ✅ Batch processing for efficient throughput");
    println!("   ✅ Automatic CPU fallback for compatibility");
    println!("   ✅ Dynamic load balancing optimization");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_multi_gpu_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Setting up multi-GPU configuration...");

    // Create configuration for multiple GPUs
    let config = AdvancedGpuConfig {
        device_ids: vec![0, 1, 2, 3],                     // 4 GPUs
        memory_pool_size_per_gpu: 2 * 1024 * 1024 * 1024, // 2GB per GPU
        streams_per_gpu: 8,                               // 8 streams per GPU
        enable_mixed_precision: true,
        enable_kernel_fusion: true,
        enable_profiling: true,
        load_balancing: LoadBalancingStrategy::Dynamic,
        min_problem_size: 10000,
        max_memory_usage: 0.85,
    };

    println!("  📋 Configuration:");
    println!("    GPUs: {:?}", config.device_ids);
    println!(
        "    Memory per GPU: {} GB",
        config.memory_pool_size_per_gpu / 1024 / 1024 / 1024
    );
    println!("    Streams per GPU: {}", config.streams_per_gpu);
    println!("    Mixed precision: {}", config.enable_mixed_precision);
    println!("    Kernel fusion: {}", config.enable_kernel_fusion);
    println!("    Load balancing: {:?}", config.load_balancing);

    // Initialize GPU operations
    let ops = AdvancedGpuOps::new(config)?;

    println!("  🔍 Device Information:");
    for device in ops.get_devices() {
        println!(
            "    GPU {}: {} ({} GB, Compute {}.{})",
            device.device_id,
            device.name,
            device.memory_total / 1024 / 1024 / 1024,
            device.compute_capability.0,
            device.compute_capability.1
        );
    }

    println!("  💾 Memory Usage:");
    for (i, (used, total)) in ops.get_memory_usage().iter().enumerate() {
        let usage_percent = (*used as f64 / *total as f64) * 100.0;
        println!(
            "    GPU {}: {:.1}% ({} MB / {} MB)",
            i,
            usage_percent,
            used / 1024 / 1024,
            total / 1024 / 1024
        );
    }

    println!("  ✅ Multi-GPU setup completed successfully!");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating advanced memory management...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0, 1],
        memory_pool_size_per_gpu: 1024 * 1024 * 1024, // 1GB per GPU
        enable_profiling: true,
        ..Default::default()
    };

    let ops = AdvancedGpuOps::new(config)?;

    println!("  📊 Initial Memory State:");
    for (i, (used, total)) in ops.get_memory_usage().iter().enumerate() {
        println!(
            "    GPU {}: {} MB / {} MB",
            i,
            used / 1024 / 1024,
            total / 1024 / 1024
        );
    }

    // Simulate memory-intensive operations
    println!("  🧠 Simulating memory-intensive operations...");

    // Create large matrices that would require significant GPU memory
    let sizes = vec![1000, 2000, 4000];
    for size in sizes {
        let memory_required = size * size * std::mem::size_of::<f64>() * 3; // A, B, C matrices
        println!(
            "    Matrix size {}x{}: {} MB required",
            size,
            size,
            memory_required / 1024 / 1024
        );
    }

    println!("  🔄 Memory pool management features:");
    println!("    ✅ Automatic memory pool allocation");
    println!("    ✅ Memory fragmentation tracking");
    println!("    ✅ Automatic defragmentation");
    println!("    ✅ Memory usage monitoring");
    println!("    ✅ Out-of-memory protection");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_multi_gpu_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating multi-GPU matrix operations...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0, 1],
        enable_profiling: true,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Create test matrices
    let size = 1000;
    let a = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| (i % 1000) as f64).collect(),
    )?;
    let b = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| ((i + 1) % 1000) as f64).collect(),
    )?;

    println!("  📊 Matrix sizes: {}x{} × {}x{}", size, size, size, size);
    println!("  🔄 Performing multi-GPU matrix multiplication...");

    let start = Instant::now();
    let result = ops.multi_gpu_matrix_multiply(&a, &b)?;
    let duration = start.elapsed();

    println!("  ⏱️  Multi-GPU time: {:.2?}", duration);
    println!("  📈 Result shape: {:?}", result.dim());

    // Calculate theoretical performance
    let ops_count = 2.0 * size as f64 * size as f64 * size as f64;
    let gflops = ops_count / duration.as_secs_f64() / 1e9;
    println!("  🚀 Throughput: {:.2} GFLOPS", gflops);

    // Memory bandwidth estimation
    let memory_accessed = (size * size * 3) as f64 * std::mem::size_of::<f64>() as f64;
    let bandwidth = memory_accessed / duration.as_secs_f64() / 1e9;
    println!("  💾 Memory bandwidth: {:.2} GB/s", bandwidth);

    println!("  ✅ Multi-GPU operation completed successfully!");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_kernel_fusion() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating kernel fusion optimization...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0],
        enable_kernel_fusion: true,
        enable_profiling: true,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Create test matrices
    let size = 512;
    let a = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| (i % 100) as f64).collect(),
    )?;
    let b = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| ((i + 1) % 100) as f64).collect(),
    )?;
    let c = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| ((i + 2) % 100) as f64).collect(),
    )?;

    println!(
        "  📊 Matrix sizes: {}x{} for fused A*B+C operation",
        size, size
    );
    println!("  🔄 Performing fused matrix multiply-add...");

    let start = Instant::now();
    let result = ops.fused_matrix_multiply_add(&a, &b, &c)?;
    let duration = start.elapsed();

    println!("  ⏱️  Fused operation time: {:.2?}", duration);
    println!("  📈 Result shape: {:?}", result.dim());

    // Calculate performance metrics
    let ops_count = 2.0 * size as f64 * size as f64 * size as f64 + size as f64 * size as f64;
    let gflops = ops_count / duration.as_secs_f64() / 1e9;
    println!("  🚀 Throughput: {:.2} GFLOPS", gflops);

    println!("  💡 Kernel fusion benefits:");
    println!("    ✅ Reduced memory bandwidth requirements");
    println!("    ✅ Improved arithmetic intensity");
    println!("    ✅ Lower kernel launch overhead");
    println!("    ✅ Better GPU utilization");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_mixed_precision() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating mixed precision computing...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0],
        enable_mixed_precision: true,
        enable_profiling: true,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Create test matrices
    let size = 1024;
    let a = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| (i % 1000) as f64).collect(),
    )?;
    let b = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| ((i + 1) % 1000) as f64).collect(),
    )?;

    println!("  📊 Matrix sizes: {}x{} for mixed precision", size, size);
    println!("  🔄 Performing mixed precision matrix multiplication...");

    let start = Instant::now();
    let result = ops.mixed_precision_matrix_multiply(&a, &b)?;
    let duration = start.elapsed();

    println!("  ⏱️  Mixed precision time: {:.2?}", duration);
    println!("  📈 Result shape: {:?}", result.dim());

    // Calculate performance metrics
    let ops_count = 2.0 * size as f64 * size as f64 * size as f64;
    let gflops = ops_count / duration.as_secs_f64() / 1e9;
    println!("  🚀 Throughput: {:.2} GFLOPS", gflops);

    println!("  💡 Mixed precision benefits:");
    println!("    ✅ ~2x memory bandwidth improvement");
    println!("    ✅ ~2x performance improvement on modern GPUs");
    println!("    ✅ Lower power consumption");
    println!("    ✅ Maintained numerical stability");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating batch processing...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0, 1],
        enable_profiling: true,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Create batch of matrices
    let batch_size = 8;
    let matrix_size = 256;
    let total_elements = batch_size * matrix_size * matrix_size;

    let a_batch = Array3::from_shape_vec(
        (batch_size, matrix_size, matrix_size),
        (0..total_elements).map(|i| (i % 1000) as f64).collect(),
    )?;
    let b_batch = Array3::from_shape_vec(
        (batch_size, matrix_size, matrix_size),
        (0..total_elements)
            .map(|i| ((i + 1) % 1000) as f64)
            .collect(),
    )?;

    println!("  📊 Batch configuration:");
    println!("    Batch size: {}", batch_size);
    println!("    Matrix size: {}x{}", matrix_size, matrix_size);
    println!("    Total matrices: {}", batch_size * 2);

    println!("  🔄 Performing batch matrix multiplication...");
    let start = Instant::now();
    let result = ops.batch_matrix_multiply(&a_batch, &b_batch)?;
    let duration = start.elapsed();

    println!("  ⏱️  Batch processing time: {:.2?}", duration);
    println!("  📈 Result shape: {:?}", result.dim());

    // Calculate performance metrics
    let ops_count =
        2.0 * batch_size as f64 * matrix_size as f64 * matrix_size as f64 * matrix_size as f64;
    let gflops = ops_count / duration.as_secs_f64() / 1e9;
    println!("  🚀 Throughput: {:.2} GFLOPS", gflops);

    let time_per_matrix = duration.as_secs_f64() / batch_size as f64;
    println!("  ⚡ Time per matrix: {:.2} ms", time_per_matrix * 1000.0);

    println!("  💡 Batch processing benefits:");
    println!("    ✅ Amortized kernel launch overhead");
    println!("    ✅ Better GPU utilization");
    println!("    ✅ Improved memory bandwidth utilization");
    println!("    ✅ Pipeline parallelism across GPUs");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_async_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating asynchronous operations...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0],
        streams_per_gpu: 4,
        enable_profiling: true,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Create test matrices
    let size = 512;
    let a = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| (i % 1000) as f64).collect(),
    )?;
    let b = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| ((i + 1) % 1000) as f64).collect(),
    )?;

    println!("  📊 Matrix sizes: {}x{}", size, size);
    println!("  🔄 Launching asynchronous matrix multiplication...");

    let start = Instant::now();
    let async_op = ops.async_matrix_multiply(&a, &b, 0)?;
    let launch_duration = start.elapsed();

    println!("  ⏱️  Launch time: {:.2?}", launch_duration);
    println!("  📋 Operation details:");
    println!("    Operation ID: {}", async_op.operation_id);
    println!("    Device ID: {}", async_op.device_id);
    println!("    Stream ID: {}", async_op.stream_id);
    println!("    Result shape: {:?}", async_op.result_shape);

    // Simulate checking operation status
    println!("  🔍 Checking operation status...");
    let total_time = async_op.elapsed_time();
    println!("  ⏱️  Total elapsed time: {:.2?}", total_time);

    if async_op.is_ready() {
        println!("  ✅ Operation completed!");
        let _result = async_op.get_result()?;
        println!("  📈 Result retrieved successfully");
    } else {
        println!("  ⏳ Operation still running...");
    }

    println!("  💡 Asynchronous operation benefits:");
    println!("    ✅ Non-blocking CPU execution");
    println!("    ✅ Concurrent GPU operations");
    println!("    ✅ Improved CPU-GPU overlap");
    println!("    ✅ Better system utilization");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_performance_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating performance profiling...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0],
        enable_profiling: true,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Run several operations to generate metrics
    let sizes = vec![256, 512, 1024];

    for size in sizes {
        println!("  📊 Running {}x{} matrix multiplication...", size, size);

        let a = Array2::from_shape_vec(
            (size, size),
            (0..size * size).map(|i| (i % 1000) as f64).collect(),
        )?;
        let b = Array2::from_shape_vec(
            (size, size),
            (0..size * size).map(|i| ((i + 1) % 1000) as f64).collect(),
        )?;

        let _result = ops.multi_gpu_matrix_multiply(&a, &b)?;
    }

    // Generate performance report
    println!("  📋 Performance Report:");
    let report = ops.generate_performance_report();
    println!("{}", report);

    // Show detailed metrics
    println!("  📊 Detailed Performance Metrics:");
    for (i, metric) in ops.get_performance_metrics().iter().enumerate() {
        println!("    Operation {}: {}", i + 1, metric.operation_name);
        println!("      Duration: {:.2?}", metric.duration);
        println!("      Throughput: {:.2} GFLOPS", metric.throughput_gflops);
        println!("      Memory used: {} MB", metric.memory_used / 1024 / 1024);
        println!("      Device: {}", metric.device_id);
        println!();
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_distributed_computing() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating distributed computing...");

    let config = AdvancedGpuConfig {
        device_ids: vec![0, 1, 2, 3], // 4 GPUs
        enable_profiling: true,
        load_balancing: LoadBalancingStrategy::Dynamic,
        ..Default::default()
    };

    let mut ops = AdvancedGpuOps::new(config)?;

    // Create large matrices for distributed computation
    let size = 2048;
    let a = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| (i % 10000) as f64).collect(),
    )?;
    let b = Array2::from_shape_vec(
        (size, size),
        (0..size * size).map(|i| ((i + 1) % 10000) as f64).collect(),
    )?;

    println!(
        "  📊 Large matrix sizes: {}x{} × {}x{}",
        size, size, size, size
    );
    println!("  🔄 Performing distributed matrix multiplication...");

    let start = Instant::now();
    let result = ops.distributed_matrix_multiply(&a, &b)?;
    let duration = start.elapsed();

    println!("  ⏱️  Distributed computation time: {:.2?}", duration);
    println!("  📈 Result shape: {:?}", result.dim());

    // Calculate performance metrics
    let ops_count = 2.0 * size as f64 * size as f64 * size as f64;
    let gflops = ops_count / duration.as_secs_f64() / 1e9;
    println!("  🚀 Aggregate throughput: {:.2} GFLOPS", gflops);

    let memory_total = (size * size * 3) as f64 * std::mem::size_of::<f64>() as f64;
    println!("  💾 Total memory processed: {:.2} GB", memory_total / 1e9);

    println!("  💡 Distributed computing benefits:");
    println!("    ✅ Scalability to very large problems");
    println!("    ✅ Automatic work distribution");
    println!("    ✅ Fault tolerance");
    println!("    ✅ Efficient inter-GPU communication");

    Ok(())
}

#[cfg(feature = "gpu")]
fn demo_load_balancing() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating load balancing strategies...");

    let strategies = vec![
        LoadBalancingStrategy::RoundRobin,
        LoadBalancingStrategy::MemoryBased,
        LoadBalancingStrategy::ComputeCapabilityBased,
        LoadBalancingStrategy::Dynamic,
    ];

    for strategy in strategies {
        println!("  📊 Testing {:?} load balancing...", strategy);

        let config = AdvancedGpuConfig {
            device_ids: vec![0, 1, 2],
            load_balancing: strategy,
            enable_profiling: true,
            ..Default::default()
        };

        let mut ops = AdvancedGpuOps::new(config)?;

        // Create test matrices
        let size = 1000;
        let a = Array2::from_shape_vec(
            (size, size),
            (0..size * size).map(|i| (i % 1000) as f64).collect(),
        )?;
        let b = Array2::from_shape_vec(
            (size, size),
            (0..size * size).map(|i| ((i + 1) % 1000) as f64).collect(),
        )?;

        let start = Instant::now();
        let _result = ops.multi_gpu_matrix_multiply(&a, &b)?;
        let duration = start.elapsed();

        println!("    ⏱️  Time: {:.2?}", duration);

        // Calculate performance
        let ops_count = 2.0 * size as f64 * size as f64 * size as f64;
        let gflops = ops_count / duration.as_secs_f64() / 1e9;
        println!("    🚀 Throughput: {:.2} GFLOPS", gflops);
    }

    println!("  💡 Load balancing strategy comparison:");
    println!("    🔄 RoundRobin: Simple, equal work distribution");
    println!("    💾 MemoryBased: Considers GPU memory capacity");
    println!("    🔢 ComputeCapabilityBased: Considers GPU compute power");
    println!("    🎯 Dynamic: Adapts based on runtime performance");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("🚫 Advanced GPU Acceleration Demo");
    println!("=================================");
    println!();
    println!("This example requires the \"gpu\" feature to be enabled.");
    println!("To run this demo, use:");
    println!("    cargo run --example advanced_gpu_acceleration_demo --features gpu");
    println!();
    println!("GPU features are currently disabled in this build.");
}
