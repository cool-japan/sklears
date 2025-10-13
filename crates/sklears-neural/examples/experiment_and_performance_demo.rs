//! Comprehensive demonstration of experiment tracking and performance testing features
//!
//! This example shows how to use the neural network crate's experiment tracking
//! and performance testing capabilities to systematically optimize and monitor
//! machine learning experiments.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Random, Rng};

use sklears_neural::{
    // Memory leak detection
    memory_leak_tests::MemoryLeakDetector,
    // Performance testing
    BenchmarkSuite,
    Experiment,
    // Experiment tracking
    ExperimentTracker,
    InMemoryBackend,
    PerformanceProfiler,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Neural Network Experiment Tracking & Performance Testing Demo");
    println!("==============================================================\n");

    // Part 1: Experiment Tracking Demo
    println!("📊 Part 1: Experiment Tracking");
    println!("-------------------------------");

    experiment_tracking_demo()?;

    println!("\n");

    // Part 2: Performance Testing Demo
    println!("⚡ Part 2: Performance Testing & Benchmarking");
    println!("--------------------------------------------");

    performance_testing_demo()?;

    println!("\n");

    // Part 3: Combined Demo - Experiment with Performance Monitoring
    println!("🚀 Part 3: Combined Experiment & Performance Monitoring");
    println!("------------------------------------------------------");

    combined_demo()?;

    Ok(())
}

fn experiment_tracking_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Create experiment tracker with in-memory backend
    let mut tracker = ExperimentTracker::new(InMemoryBackend::new());

    // Create and start experiment
    let exp_id = tracker.create_experiment("MLP_Hyperparameter_Optimization".to_string())?;
    tracker.start_experiment(exp_id.clone())?;

    println!("🔬 Started experiment: {}", exp_id);

    // Log hyperparameters
    tracker.log_hyperparameter("learning_rate".to_string(), 0.01)?;
    tracker.log_hyperparameter("hidden_layers".to_string(), vec![64.0, 32.0])?;
    tracker.log_hyperparameter("batch_size".to_string(), 32i64)?;
    tracker.log_hyperparameter("activation".to_string(), "relu".to_string())?;
    tracker.log_hyperparameter("max_iter".to_string(), 1000i64)?;

    println!("✅ Logged hyperparameters");

    // Simulate training with metrics logging
    simulate_training_with_metrics(&mut tracker)?;

    // Complete experiment
    tracker.complete_experiment()?;
    println!("🏁 Experiment completed");

    // Retrieve and display experiment results
    let experiment = tracker.get_experiment(&exp_id)?;
    display_experiment_results(&experiment);

    Ok(())
}

fn simulate_training_with_metrics(
    tracker: &mut ExperimentTracker<InMemoryBackend>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏃 Simulating training with metric logging...");

    // Simulate 10 epochs of training
    let mut rng = Random::seed(42);
    for epoch in 1..=10 {
        // Simulate decreasing loss with some noise
        let loss = 1.0 * (-0.1 * epoch as f64).exp() + rng.gen::<f64>() * 0.1;

        // Simulate increasing accuracy
        let accuracy = 0.5 + 0.4 * (1.0 - (-0.1 * epoch as f64).exp()) + rng.gen::<f64>() * 0.05;

        // Log metrics
        tracker.log_metric("train_loss".to_string(), loss, Some(epoch))?;
        tracker.log_metric("train_accuracy".to_string(), accuracy, Some(epoch))?;

        if epoch % 5 == 0 {
            println!(
                "  Epoch {}: Loss = {:.4}, Accuracy = {:.4}",
                epoch, loss, accuracy
            );
        }
    }

    // Log final results (would be done through future API - for now just log metrics)
    tracker.log_metric("final_loss".to_string(), 0.15, None)?;
    tracker.log_metric("final_accuracy".to_string(), 0.92, None)?;
    tracker.log_metric("training_time_minutes".to_string(), 2.5, None)?;

    Ok(())
}

fn display_experiment_results(experiment: &Experiment) {
    println!("\n📈 Experiment Results Summary:");
    println!("  Name: {}", experiment.name);
    println!("  Status: {:?}", experiment.status);

    if let Some(duration) = experiment.duration_seconds() {
        println!("  Duration: {} seconds", duration);
    }

    println!("  Hyperparameters:");
    for (key, value) in &experiment.hyperparameters {
        println!("    {}: {:?}", key, value);
    }

    println!("  Final Results:");
    for (key, value) in &experiment.results {
        println!("    {}: {:.4}", key, value);
    }

    println!("  Metrics collected: {}", experiment.metrics.len());
}

fn performance_testing_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Running performance benchmarks...");

    // Create and run benchmark suite
    let mut suite = BenchmarkSuite::new();

    // Add matrix multiplication benchmarks
    add_matrix_benchmarks(&mut suite);

    // Add neural network benchmarks
    add_neural_network_benchmarks(&mut suite);

    // Set some baseline metrics (normally loaded from previous runs)
    set_baseline_metrics(&mut suite);

    // Run all benchmarks
    suite.run_all()?;

    // Generate and display report
    let report = suite.generate_report();
    println!("{}", report);

    // Memory leak detection demo
    memory_leak_detection_demo()?;

    Ok(())
}

fn add_matrix_benchmarks(suite: &mut BenchmarkSuite) {
    use sklears_neural::performance_testing::utils;

    // Small matrix multiplication
    suite.add_benchmark("matrix_multiply_100x100".to_string(), || {
        let a = Array2::ones((100, 100));
        let b = Array2::ones((100, 100));
        utils::benchmark_matrix_multiply(&a, &b, 100)
    });

    // Large matrix multiplication
    suite.add_benchmark("matrix_multiply_500x500".to_string(), || {
        let a = Array2::ones((500, 500));
        let b = Array2::ones((500, 500));
        utils::benchmark_matrix_multiply(&a, &b, 10)
    });
}

fn add_neural_network_benchmarks(suite: &mut BenchmarkSuite) {
    // Neural network forward pass benchmark
    suite.add_benchmark("mlp_forward_pass".to_string(), || {
        let mut profiler = PerformanceProfiler::new();
        profiler.start();

        // Create sample data
        let x = Array2::ones((1000, 10));

        // Simulate forward pass computation
        for i in 0..100 {
            let _ = x.mapv(|v: f64| v.tanh()); // Simulate activation function
            profiler.record_operation();
            profiler.record_samples(x.nrows() as u64);
        }

        profiler.stop()
    });
}

fn set_baseline_metrics(suite: &mut BenchmarkSuite) {
    use sklears_neural::performance_testing::{MemoryStats, PerformanceMetrics};
    use std::time::Duration;

    // Set baselines (these would typically be loaded from storage)
    let baseline_memory = MemoryStats::new();

    suite.set_baseline(
        "matrix_multiply_100x100".to_string(),
        PerformanceMetrics::new(Duration::from_millis(50), baseline_memory.clone())
            .with_ops_per_second(2000.0),
    );

    suite.set_baseline(
        "matrix_multiply_500x500".to_string(),
        PerformanceMetrics::new(Duration::from_millis(500), baseline_memory.clone())
            .with_ops_per_second(20.0),
    );

    suite.set_baseline(
        "mlp_forward_pass".to_string(),
        PerformanceMetrics::new(Duration::from_millis(100), baseline_memory)
            .with_ops_per_second(1000.0)
            .with_samples_per_second(100000.0),
    );
}

fn memory_leak_detection_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Memory leak detection demo...");

    let mut detector = MemoryLeakDetector::new().with_threshold(1_000_000); // 1MB threshold

    // Take initial snapshot
    detector.snapshot_with_label(Some("Initial".to_string()));

    // Simulate some memory-intensive operations
    for i in 0..5 {
        // Simulate memory allocation by creating large arrays
        let _large_array = Array2::<f64>::zeros((1000, 1000)); // 8MB array

        detector.snapshot_with_label(Some(format!("After allocation {}", i + 1)));

        // Small delay to allow memory tracking
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Get statistics and check for leaks
    let stats = detector.get_statistics();
    let leaks = detector.detect_leaks();

    println!("Memory Statistics:");
    println!("  Total snapshots: {}", stats.total_snapshots);
    println!(
        "  Average virtual memory: {:.2} MB",
        stats.virtual_memory_avg / 1_000_000.0
    );
    println!(
        "  Average resident memory: {:.2} MB",
        stats.resident_memory_avg / 1_000_000.0
    );
    println!("  Potential leaks detected: {}", stats.potential_leaks);

    if !leaks.is_empty() {
        println!("⚠️  Detected {} potential memory leaks:", leaks.len());
        for (idx, diff) in leaks {
            println!(
                "  Snapshot {}: +{:.2} MB virtual, +{:.2} MB resident",
                idx,
                diff.virtual_memory_delta as f64 / 1_000_000.0,
                diff.resident_memory_delta as f64 / 1_000_000.0
            );
        }
    } else {
        println!("✅ No memory leaks detected");
    }

    Ok(())
}

fn combined_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Running ML experiment with performance monitoring...");

    // Create experiment tracker
    let mut tracker = ExperimentTracker::new(InMemoryBackend::new());
    let exp_id = tracker.create_experiment("Performance_Optimized_MLP".to_string())?;
    tracker.start_experiment(exp_id.clone())?;

    // Create performance profiler
    let mut profiler = PerformanceProfiler::new();

    // Generate sample data
    let x = generate_sample_data(1000, 20);
    let y = generate_sample_labels(1000, 3);

    // Log hyperparameters
    tracker.log_hyperparameter("hidden_layer_sizes".to_string(), vec![50.0, 30.0])?;
    tracker.log_hyperparameter("activation".to_string(), "relu".to_string())?;
    tracker.log_hyperparameter("max_iter".to_string(), 100i64)?;
    tracker.log_hyperparameter("learning_rate_init".to_string(), 0.001)?;

    println!("📊 Hyperparameters logged");

    // Start performance monitoring
    profiler.start();

    // Train model (simulation)
    println!("🏋️ Training model with performance monitoring...");

    // Simulate model training with performance tracking
    simulate_model_training(&mut profiler, &x, &y)?;

    // Stop profiling and get metrics
    let performance_metrics = profiler.stop()?;

    // Log performance metrics as experiment metrics
    tracker.log_metric(
        "training_time_seconds".to_string(),
        performance_metrics.execution_time.as_secs_f64(),
        None,
    )?;
    tracker.log_metric(
        "operations_per_second".to_string(),
        performance_metrics.ops_per_second,
        None,
    )?;
    tracker.log_metric(
        "samples_per_second".to_string(),
        performance_metrics.samples_per_second,
        None,
    )?;
    tracker.log_metric(
        "peak_memory_mb".to_string(),
        performance_metrics.memory_stats.peak_usage_bytes as f64 / (1024.0 * 1024.0),
        None,
    )?;

    // Complete experiment
    tracker.complete_experiment()?;

    // Display results
    let final_experiment = tracker.get_experiment(&exp_id)?;
    println!("\n🏆 Combined Experiment Results:");
    display_experiment_results(&final_experiment);

    println!("\n⚡ Performance Metrics:");
    println!(
        "  Execution time: {:.2} seconds",
        performance_metrics.execution_time.as_secs_f64()
    );
    println!(
        "  Operations/second: {:.2}",
        performance_metrics.ops_per_second
    );
    println!(
        "  Samples/second: {:.2}",
        performance_metrics.samples_per_second
    );
    println!(
        "  Peak memory usage: {:.2} MB",
        performance_metrics.memory_stats.peak_usage_bytes as f64 / (1024.0 * 1024.0)
    );

    Ok(())
}

fn generate_sample_data(samples: usize, features: usize) -> Array2<f64> {
    let mut rng = Random::seed(42);
    Array2::from_shape_fn((samples, features), |_| rng.gen::<f64>())
}

fn generate_sample_labels(samples: usize, classes: usize) -> Vec<usize> {
    let mut rng = Random::seed(42);
    (0..samples).map(|_| rng.gen_range(0..classes)).collect()
}

fn simulate_model_training(
    profiler: &mut PerformanceProfiler,
    x: &Array2<f64>,
    y: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let (n_samples, n_features) = x.dim();

    println!(
        "  Training on {} samples with {} features",
        n_samples, n_features
    );

    // Simulate training iterations
    for epoch in 1..=20 {
        // Simulate forward pass
        profiler.record_allocation(1024 * n_samples as u64); // Simulate memory allocation

        let activations = x.mapv(|v| v.max(0.0)); // ReLU simulation
        profiler.record_operation();
        profiler.record_samples(n_samples as u64);

        // Simulate backward pass
        let _gradients = activations.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        profiler.record_operation();

        // Simulate weight updates
        profiler.record_operation();

        profiler.record_deallocation(1024 * n_samples as u64); // Simulate memory deallocation

        if epoch % 5 == 0 {
            println!("  Completed epoch {}/20", epoch);
        }
    }

    println!("✅ Model training simulation completed");

    Ok(())
}
