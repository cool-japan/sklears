//! Performance Profiling Examples
//!
//! This example demonstrates comprehensive performance profiling for machine learning
//! pipelines, including execution timing, resource monitoring, bottleneck detection,
//! and optimization recommendations.
//!
//! Run with: cargo run --example performance_profiling

use ndarray::{array, Array1, Array2};
use sklears_compose::{
    mock::{MockPredictor, MockTransformer},
    performance_profiler::{
        BottleneckType, ImplementationDifficulty, OptimizationCategory, OptimizationPriority,
        PerformanceProfiler, ProfilerConfig,
    },
    Pipeline, PipelineBuilder,
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict},
};
use std::{collections::HashMap, thread, time::Duration};

/// Generate sample dataset for profiling
fn generate_sample_dataset(size: usize) -> (Array2<f64>, Array1<f64>) {
    let X = Array2::from_shape_fn((size, 10), |(i, j)| (i + j) as f64 * 0.1);
    let y = Array1::from_shape_fn(size, |i| (i as f64).sin() * 10.0 + 5.0);
    (X, y)
}

/// Simulate a slow operation for profiling demonstration
fn simulate_heavy_computation(duration_ms: u64) {
    thread::sleep(Duration::from_millis(duration_ms));
}

/// Demonstrate basic pipeline profiling
fn demo_basic_profiling() -> SklResult<()> {
    println!("📊 Basic Pipeline Profiling Demo");
    println!("{}", "=".repeat(50));

    // Configure profiler
    let profiler_config = ProfilerConfig {
        enable_timing: true,
        enable_memory_tracking: true,
        enable_cpu_monitoring: true,
        sample_interval_ms: 50,
        max_sessions: 10,
        enable_bottleneck_detection: true,
        enable_optimization_hints: true,
        ..Default::default()
    };

    let profiler = PerformanceProfiler::new(profiler_config);
    let (X, y) = generate_sample_dataset(100);

    println!("🚀 Starting pipeline profiling session...");

    // Start profiling session
    let session_id = profiler.start_session("basic_ml_pipeline");

    // Stage 1: Data preprocessing
    println!("📋 Stage 1: Data preprocessing");
    profiler.start_stage(&session_id, "preprocessing", "transformer")?;
    profiler.record_data_shapes(
        &session_id,
        "preprocessing",
        Some((X.nrows(), X.ncols())),
        Some((X.nrows(), X.ncols())),
    )?;

    let mut stage_params = HashMap::new();
    stage_params.insert("scaler_type".to_string(), "standard".to_string());
    stage_params.insert("handle_missing".to_string(), "impute".to_string());
    profiler.record_stage_parameters(&session_id, "preprocessing", stage_params)?;

    simulate_heavy_computation(150); // Simulate preprocessing time
    let preprocessing_time = profiler.end_stage(&session_id, "preprocessing")?;
    println!(
        "   ⏱️ Completed in: {:.2}ms",
        preprocessing_time.as_millis()
    );

    // Stage 2: Feature engineering
    println!("📋 Stage 2: Feature engineering");
    profiler.start_stage(&session_id, "feature_engineering", "transformer")?;
    profiler.record_data_shapes(
        &session_id,
        "feature_engineering",
        Some((X.nrows(), X.ncols())),
        Some((X.nrows(), X.ncols() * 2)),
    )?;

    let mut fe_params = HashMap::new();
    fe_params.insert("polynomial_degree".to_string(), "2".to_string());
    fe_params.insert("interaction_only".to_string(), "false".to_string());
    profiler.record_stage_parameters(&session_id, "feature_engineering", fe_params)?;

    simulate_heavy_computation(300); // Simulate feature engineering time
    let fe_time = profiler.end_stage(&session_id, "feature_engineering")?;
    println!("   ⏱️ Completed in: {:.2}ms", fe_time.as_millis());

    // Stage 3: Model training
    println!("📋 Stage 3: Model training");
    profiler.start_stage(&session_id, "model_training", "estimator")?;
    profiler.record_data_shapes(
        &session_id,
        "model_training",
        Some((X.nrows(), X.ncols() * 2)),
        None,
    )?;

    let mut model_params = HashMap::new();
    model_params.insert("algorithm".to_string(), "gradient_boosting".to_string());
    model_params.insert("n_estimators".to_string(), "100".to_string());
    model_params.insert("learning_rate".to_string(), "0.1".to_string());
    profiler.record_stage_parameters(&session_id, "model_training", model_params)?;

    simulate_heavy_computation(800); // Simulate training time
    let training_time = profiler.end_stage(&session_id, "model_training")?;
    println!("   ⏱️ Completed in: {:.2}ms", training_time.as_millis());

    // Stage 4: Model evaluation
    println!("📋 Stage 4: Model evaluation");
    profiler.start_stage(&session_id, "evaluation", "evaluator")?;
    profiler.record_data_shapes(
        &session_id,
        "evaluation",
        Some((X.nrows(), X.ncols() * 2)),
        Some((X.nrows(), 1)),
    )?;

    simulate_heavy_computation(100); // Simulate evaluation time
    let eval_time = profiler.end_stage(&session_id, "evaluation")?;
    println!("   ⏱️ Completed in: {:.2}ms", eval_time.as_millis());

    // End profiling session and get results
    println!("\n🏁 Ending profiling session...");
    let completed_session = profiler.end_session(&session_id)?;

    // Display results
    println!("\n📈 Profiling Results:");
    println!(
        "   • Total execution time: {:.2}ms",
        completed_session
            .overall_metrics
            .total_execution_time
            .as_millis()
    );
    println!(
        "   • Peak memory usage: {:.1}MB",
        completed_session.overall_metrics.peak_memory_usage_mb
    );
    println!(
        "   • Average CPU usage: {:.1}%",
        completed_session.overall_metrics.average_cpu_usage
    );
    println!(
        "   • Pipeline stages: {}",
        completed_session.overall_metrics.pipeline_stages
    );
    println!(
        "   • Parallel efficiency: {:.1}%",
        completed_session.overall_metrics.parallel_efficiency * 100.0
    );

    // Show stage breakdown
    println!("\n📊 Stage Performance Breakdown:");
    for (stage_name, stage) in &completed_session.stages {
        let percentage = (stage.execution_time.as_millis() as f64
            / completed_session
                .overall_metrics
                .total_execution_time
                .as_millis() as f64)
            * 100.0;
        println!(
            "   • {}: {:.2}ms ({:.1}%)",
            stage_name,
            stage.execution_time.as_millis(),
            percentage
        );

        if let Some(input_shape) = stage.input_shape {
            println!("     Input shape: {} × {}", input_shape.0, input_shape.1);
        }
        if let Some(output_shape) = stage.output_shape {
            println!("     Output shape: {} × {}", output_shape.0, output_shape.1);
        }
    }

    // Show bottlenecks if detected
    if !completed_session.bottlenecks.is_empty() {
        println!("\n⚠️ Detected Bottlenecks:");
        for bottleneck in &completed_session.bottlenecks {
            println!(
                "   • {}: {} (Severity: {:?})",
                bottleneck.affected_stage, bottleneck.description, bottleneck.severity
            );
            println!("     Impact factor: {:.2}", bottleneck.impact_factor);
        }
    }

    // Show optimization hints
    if !completed_session.optimization_hints.is_empty() {
        println!("\n💡 Optimization Recommendations:");
        for hint in &completed_session.optimization_hints.iter().take(3) {
            println!("   • {} (Priority: {:?})", hint.title, hint.priority);
            println!("     {}", hint.description);
            println!(
                "     Expected improvement: {:.0}%",
                hint.expected_improvement * 100.0
            );
            if !hint.code_examples.is_empty() {
                println!("     Example: {}", hint.code_examples[0]);
            }
            println!();
        }
    }

    Ok(())
}

/// Demonstrate comparative profiling of different configurations
fn demo_comparative_profiling() -> SklResult<()> {
    println!("\n🔬 Comparative Profiling Demo");
    println!("{}", "=".repeat(50));

    let profiler = PerformanceProfiler::default();
    let (X, y) = generate_sample_dataset(200);

    let configurations = vec![
        ("fast_config", vec![100, 50, 200]), // preprocessing, feature_eng, training times
        ("balanced_config", vec![150, 200, 400]),
        ("thorough_config", vec![300, 500, 800]),
    ];

    let mut session_results = Vec::new();

    for (config_name, timing) in configurations {
        println!("\n📋 Testing configuration: {}", config_name);

        let session_id = profiler.start_session(&format!("pipeline_{}", config_name));

        // Run pipeline with specific timing
        profiler.start_stage(&session_id, "preprocessing", "transformer")?;
        simulate_heavy_computation(timing[0]);
        profiler.end_stage(&session_id, "preprocessing")?;

        profiler.start_stage(&session_id, "feature_engineering", "transformer")?;
        simulate_heavy_computation(timing[1]);
        profiler.end_stage(&session_id, "feature_engineering")?;

        profiler.start_stage(&session_id, "training", "estimator")?;
        simulate_heavy_computation(timing[2]);
        profiler.end_stage(&session_id, "training")?;

        let session_result = profiler.end_session(&session_id)?;
        let total_time = session_result
            .overall_metrics
            .total_execution_time
            .as_millis();
        println!("   ⏱️ Total time: {:.2}ms", total_time);

        session_results.push((config_name, session_result));
    }

    // Generate comparative report
    println!("\n📈 Comparative Analysis:");
    let report = profiler.generate_report(None);

    println!("   📊 Summary Statistics:");
    println!("     • Sessions analyzed: {}", report.sessions_analyzed);
    println!(
        "     • Average execution time: {:.2}ms",
        report.summary_metrics.average_execution_time.as_millis()
    );
    println!(
        "     • Fastest execution: {:.2}ms",
        report.summary_metrics.fastest_execution_time.as_millis()
    );
    println!(
        "     • Slowest execution: {:.2}ms",
        report.summary_metrics.slowest_execution_time.as_millis()
    );
    println!(
        "     • Performance variance: {:.2}",
        report.comparative_analysis.performance_variance
    );
    println!(
        "     • Consistency score: {:.2}",
        report.comparative_analysis.consistency_score
    );

    // Best and worst performing configurations
    println!("\n🏆 Performance Comparison:");
    println!(
        "   • Best performing: {}",
        report.comparative_analysis.best_performing_session
    );
    println!(
        "   • Worst performing: {}",
        report.comparative_analysis.worst_performing_session
    );

    // Trend analysis
    println!("\n📈 Trend Analysis:");
    println!(
        "   • Performance trend: {:?}",
        report.trend_analysis.performance_trend
    );
    println!(
        "   • Memory usage trend: {:?}",
        report.trend_analysis.memory_usage_trend
    );

    Ok(())
}

/// Demonstrate real-time profiling with monitoring
fn demo_realtime_profiling() -> SklResult<()> {
    println!("\n⏱️ Real-time Profiling Demo");
    println!("{}", "=".repeat(50));

    // Configure profiler for high-frequency sampling
    let profiler_config = ProfilerConfig {
        enable_timing: true,
        enable_memory_tracking: true,
        enable_cpu_monitoring: true,
        sample_interval_ms: 25, // High frequency sampling
        enable_bottleneck_detection: true,
        enable_optimization_hints: true,
        ..Default::default()
    };

    let profiler = PerformanceProfiler::new(profiler_config);
    let (X, y) = generate_sample_dataset(150);

    println!("🚀 Starting real-time profiling with high-frequency monitoring...");

    let session_id = profiler.start_session("realtime_pipeline");

    // Simulate a pipeline with varying computational load
    let stages = vec![
        ("data_loading", "loader", 100),
        ("validation", "validator", 80),
        ("preprocessing", "transformer", 250),
        ("feature_selection", "selector", 150),
        ("hyperparameter_tuning", "optimizer", 600),
        ("model_training", "estimator", 400),
        ("cross_validation", "evaluator", 300),
        ("final_evaluation", "evaluator", 120),
    ];

    for (stage_name, component_type, duration_ms) in stages {
        println!("📋 Executing: {} ({} ms)", stage_name, duration_ms);

        profiler.start_stage(&session_id, stage_name, component_type)?;

        // Simulate variable load during execution
        let chunk_size = duration_ms / 5;
        for i in 0..5 {
            let chunk_duration = chunk_size + (i * 10); // Increasing load
            simulate_heavy_computation(chunk_duration as u64);

            // Print progress
            print!(".");
            if i == 4 {
                println!();
            }
        }

        let stage_time = profiler.end_stage(&session_id, stage_name)?;
        println!("   ✅ Completed in: {:.2}ms", stage_time.as_millis());
    }

    let completed_session = profiler.end_session(&session_id)?;

    // Display detailed real-time monitoring results
    println!("\n📊 Real-time Monitoring Results:");
    println!(
        "   • Total execution time: {:.2}ms",
        completed_session
            .overall_metrics
            .total_execution_time
            .as_millis()
    );
    println!(
        "   • Peak memory usage: {:.1}MB",
        completed_session.overall_metrics.peak_memory_usage_mb
    );
    println!(
        "   • Data processed: {:.2}MB",
        completed_session.overall_metrics.total_data_processed_mb
    );
    println!(
        "   • Throughput: {:.1} samples/sec",
        completed_session
            .overall_metrics
            .throughput_samples_per_second
    );

    // Show resource utilization details
    println!("\n🔋 Resource Utilization Analysis:");
    for (stage_name, stage) in &completed_session.stages {
        println!(
            "   • {}: {} memory samples, {} CPU samples",
            stage_name,
            stage.memory_samples.len(),
            stage.cpu_samples.len()
        );

        if !stage.memory_samples.is_empty() {
            let avg_memory = stage
                .memory_samples
                .iter()
                .map(|s| s.heap_usage_mb)
                .sum::<f64>()
                / stage.memory_samples.len() as f64;
            println!("     Average memory: {:.1}MB", avg_memory);
        }

        if !stage.cpu_samples.is_empty() {
            let avg_cpu = stage
                .cpu_samples
                .iter()
                .map(|s| s.overall_usage)
                .sum::<f64>()
                / stage.cpu_samples.len() as f64;
            println!("     Average CPU: {:.1}%", avg_cpu);
        }
    }

    // Bottleneck analysis
    if !completed_session.bottlenecks.is_empty() {
        println!("\n⚠️ Real-time Bottleneck Detection:");
        for bottleneck in &completed_session.bottlenecks {
            println!("   • Type: {:?}", bottleneck.bottleneck_type);
            println!("     Stage: {}", bottleneck.affected_stage);
            println!("     Description: {}", bottleneck.description);
            println!(
                "     Resource utilization: {:.1}%",
                bottleneck.metrics.resource_utilization * 100.0
            );
            println!(
                "     Improvement potential: {:.0}%",
                bottleneck.metrics.improvement_potential * 100.0
            );
            println!();
        }
    }

    Ok(())
}

/// Demonstrate profiling-guided optimization
fn demo_optimization_guided_by_profiling() -> SklResult<()> {
    println!("\n🎯 Profiling-Guided Optimization Demo");
    println!("{}", "=".repeat(50));

    let profiler = PerformanceProfiler::default();
    let (X, y) = generate_sample_dataset(100);

    // Profile baseline configuration
    println!("📋 Step 1: Profiling baseline configuration");
    let baseline_id = profiler.start_session("baseline_pipeline");

    profiler.start_stage(&baseline_id, "preprocessing", "transformer")?;
    simulate_heavy_computation(200);
    profiler.end_stage(&baseline_id, "preprocessing")?;

    profiler.start_stage(&baseline_id, "slow_algorithm", "estimator")?;
    simulate_heavy_computation(800); // Intentionally slow
    profiler.end_stage(&baseline_id, "slow_algorithm")?;

    let baseline_session = profiler.end_session(&baseline_id)?;
    let baseline_time = baseline_session
        .overall_metrics
        .total_execution_time
        .as_millis();
    println!("   ⏱️ Baseline time: {:.2}ms", baseline_time);

    // Apply optimizations based on profiling results
    println!("\n📋 Step 2: Applying optimization recommendations");

    // Find bottlenecks and apply fixes
    for bottleneck in &baseline_session.bottlenecks {
        match bottleneck.bottleneck_type {
            BottleneckType::ComputationalBottleneck => {
                println!(
                    "   🔧 Applying computational optimization for: {}",
                    bottleneck.affected_stage
                );
            }
            BottleneckType::MemoryConstraint => {
                println!(
                    "   🔧 Applying memory optimization for: {}",
                    bottleneck.affected_stage
                );
            }
            _ => {}
        }
    }

    // Profile optimized configuration
    println!("\n📋 Step 3: Profiling optimized configuration");
    let optimized_id = profiler.start_session("optimized_pipeline");

    profiler.start_stage(&optimized_id, "preprocessing", "transformer")?;
    simulate_heavy_computation(100); // Optimized preprocessing
    profiler.end_stage(&optimized_id, "preprocessing")?;

    profiler.start_stage(&optimized_id, "fast_algorithm", "estimator")?;
    simulate_heavy_computation(300); // Optimized algorithm
    profiler.end_stage(&optimized_id, "fast_algorithm")?;

    let optimized_session = profiler.end_session(&optimized_id)?;
    let optimized_time = optimized_session
        .overall_metrics
        .total_execution_time
        .as_millis();
    println!("   ⏱️ Optimized time: {:.2}ms", optimized_time);

    // Calculate improvement
    let improvement =
        ((baseline_time as f64 - optimized_time as f64) / baseline_time as f64) * 100.0;
    println!("\n📈 Optimization Results:");
    println!("   • Baseline execution time: {:.2}ms", baseline_time);
    println!("   • Optimized execution time: {:.2}ms", optimized_time);
    println!("   • Performance improvement: {:.1}%", improvement);
    println!(
        "   • Speedup factor: {:.2}x",
        baseline_time as f64 / optimized_time as f64
    );

    // Generate optimization report
    let optimization_report = profiler.generate_report(None);
    println!("\n💡 Key Optimization Insights:");
    for hint in optimization_report
        .optimization_recommendations
        .iter()
        .take(2)
    {
        println!("   • {} (Category: {:?})", hint.title, hint.category);
        println!(
            "     Expected improvement: {:.0}%",
            hint.expected_improvement * 100.0
        );
        println!(
            "     Implementation difficulty: {:?}",
            hint.implementation_difficulty
        );
    }

    Ok(())
}

fn main() -> SklResult<()> {
    println!("🚀 sklears-compose Performance Profiling Examples");
    println!("{}", "=".repeat(60));
    println!("This example demonstrates comprehensive performance profiling:");
    println!("• Detailed execution timing and resource monitoring");
    println!("• Bottleneck detection and analysis");
    println!("• Optimization recommendations");
    println!("• Comparative performance analysis");
    println!("• Real-time profiling with high-frequency sampling");

    // Run all profiling demonstrations
    demo_basic_profiling()?;
    demo_comparative_profiling()?;
    demo_realtime_profiling()?;
    demo_optimization_guided_by_profiling()?;

    println!("\n🎉 All profiling examples completed successfully!");

    println!("\n💡 Performance Profiling Benefits:");
    println!("• Identifies performance bottlenecks automatically");
    println!("• Provides actionable optimization recommendations");
    println!("• Tracks resource usage in real-time");
    println!("• Enables comparative analysis of configurations");
    println!("• Supports profiling-guided optimization workflows");

    println!("\n🔧 Profiling Best Practices:");
    println!("• Profile representative workloads and datasets");
    println!("• Use appropriate sampling intervals for your use case");
    println!("• Focus on high-impact optimizations first");
    println!("• Validate optimizations with A/B testing");
    println!("• Monitor performance regressions continuously");

    println!("\n📚 Integration with Other Tools:");
    println!("• Combine with SIMD benchmarks: cargo bench simd_benchmarks");
    println!("• Use enhanced error handling for debugging issues");
    println!("• Leverage AutoML for automated optimization");
    println!("• Export profiling data for external analysis");

    Ok(())
}
