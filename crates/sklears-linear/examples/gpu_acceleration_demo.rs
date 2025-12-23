//! GPU Acceleration Demo for Linear Models
//!
//! This example demonstrates how to use GPU acceleration with linear models
//! in the sklears library. It shows the performance benefits of GPU acceleration
//! for large-scale linear algebra operations.

use scirs2_autograd::ndarray::{Array1, Array2};
use sklears_core::traits::Fit;
use sklears_linear::LinearRegression;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU Acceleration Demo for Linear Models");
    println!("==========================================\n");

    // Create sample data
    let n_samples = 10000;
    let n_features = 1000;

    println!("📊 Dataset Info:");
    println!("  - Samples: {}", n_samples);
    println!("  - Features: {}", n_features);
    println!("  - Problem size: {} elements\n", n_samples * n_features);

    // Generate synthetic data
    let mut data = Vec::new();
    for i in 0..n_samples {
        for j in 0..n_features {
            data.push((i as f64 + j as f64) * 0.01);
        }
    }
    let x = Array2::from_shape_vec((n_samples, n_features), data)?;

    // Generate target values
    let y = Array1::from_vec((0..n_samples).map(|i| (i as f64) * 0.1 + 1.0).collect());

    // Test without GPU acceleration
    println!("🖥️  Testing CPU-only implementation...");
    let start = Instant::now();
    let cpu_model = LinearRegression::new().fit_intercept(true);

    #[cfg(feature = "gpu")]
    let cpu_model = cpu_model.use_gpu(false);

    let cpu_fitted = cpu_model.fit(&x, &y)?;
    let cpu_time = start.elapsed();

    println!("   ✅ CPU training completed in {:.2?}", cpu_time);
    println!("   📈 Coefficients shape: {:?}", cpu_fitted.coef().shape());

    // Test with GPU acceleration (if available)
    #[cfg(feature = "gpu")]
    {
        println!("\n🎮 Testing GPU-accelerated implementation...");
        let start = Instant::now();
        let gpu_model = LinearRegression::new()
            .fit_intercept(true)
            .use_gpu(true)
            .gpu_min_size(1000); // Use GPU for problems with 1000+ elements

        let gpu_fitted = gpu_model.fit(&x, &y)?;
        let gpu_time = start.elapsed();

        println!("   ✅ GPU training completed in {:.2?}", gpu_time);
        println!("   📈 Coefficients shape: {:?}", gpu_fitted.coef().shape());

        // Compare performance
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("\n🏆 Performance Comparison:");
        println!("   CPU Time: {:.2?}", cpu_time);
        println!("   GPU Time: {:.2?}", gpu_time);
        if speedup > 1.0 {
            println!("   🚀 GPU Speedup: {:.2}x faster!", speedup);
        } else {
            println!(
                "   ⚠️  GPU Overhead: {:.2}x slower (expected for CPU fallback)",
                1.0 / speedup
            );
        }

        // Compare accuracy
        let cpu_coef = cpu_fitted.coef();
        let gpu_coef = gpu_fitted.coef();
        let max_diff = cpu_coef
            .iter()
            .zip(gpu_coef.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("   📊 Maximum coefficient difference: {:.2e}", max_diff);
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n⚠️  GPU acceleration is not enabled.");
        println!("   To enable GPU acceleration, compile with: --features gpu");
    }

    // Demonstrate GPU memory management
    #[cfg(feature = "gpu")]
    {
        use sklears_linear::gpu_acceleration::{GpuLinearOps, GpuMemoryPool};

        println!("\n🧠 GPU Memory Management Demo:");

        // Create a memory pool
        let mut pool = GpuMemoryPool::new(1024 * 1024); // 1MB pool
        println!(
            "   📦 Created memory pool: {} bytes",
            pool.available_memory()
        );

        // Allocate some memory
        pool.allocate(512 * 1024)?;
        println!(
            "   📤 Allocated 512KB, remaining: {} bytes",
            pool.available_memory()
        );

        // Test GPU operations
        let gpu_ops = GpuLinearOps::default()?;
        println!("   🔍 GPU available: {}", gpu_ops.is_gpu_available());

        let stats = gpu_ops.get_performance_stats();
        println!(
            "   📈 Performance stats: {} total operations",
            stats.total_operations
        );
    }

    println!("\n🎯 Key Features Demonstrated:");
    println!("   ✅ Automatic GPU/CPU selection based on problem size");
    println!("   ✅ Seamless fallback to CPU when GPU unavailable");
    println!("   ✅ Memory-efficient GPU operations");
    println!("   ✅ Easy-to-use builder pattern for GPU configuration");
    println!("   ✅ Performance monitoring and statistics");

    println!("\n✨ Demo completed successfully!");

    Ok(())
}
