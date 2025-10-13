//! Interactive Developer Experience Demo
//!
//! This example demonstrates the enhanced developer experience features including:
//! - Improved error messages with actionable suggestions
//! - Pipeline debugging and inspection tools
//! - Interactive debugging sessions
//! - Real dataset examples with comprehensive error handling
//!
//! Run with: cargo run --example interactive_developer_experience

use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_compose::{
    enhanced_errors::{
        DataShape, ErrorContext, ImpactLevel, PerformanceMetrics, PerformanceWarningType,
        PipelineError,
    },
    DeveloperPipelineDebugger, ErrorMessageEnhancer, TraceEntry,
};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;

/// Generate a realistic ML dataset for demonstrations
fn generate_real_world_dataset() -> (Array2<f64>, Array1<f64>) {
    // Simulate a housing price prediction dataset
    let X = array![
        [2104.0, 5.0, 1.0, 45.0], // sqft, bedrooms, floors, age
        [1416.0, 3.0, 2.0, 40.0],
        [1534.0, 3.0, 2.0, 30.0],
        [852.0, 2.0, 1.0, 36.0],
        [1940.0, 4.0, 1.0, 13.0],
        [1352.0, 3.0, 1.0, 25.0],
        [1494.0, 3.0, 1.0, 10.0],
        [2000.0, 4.0, 1.0, 15.0],
        [1652.0, 3.0, 2.0, 20.0],
        [1875.0, 4.0, 2.0, 12.0],
    ];

    // Housing prices (in thousands)
    let y = array![460.0, 232.0, 315.0, 178.0, 240.0, 347.0, 329.0, 369.0, 314.0, 405.0];

    (X, y)
}

/// Demonstrate enhanced error messages with real-world scenarios
fn demo_enhanced_error_messages() -> SklResult<()> {
    println!("🎯 Enhanced Error Messages Demo");
    println!("{}", "=".repeat(60));

    // Scenario 1: Configuration Error
    println!("\n📋 Scenario 1: Configuration Error with Enhanced Messages");
    let config_error = PipelineError::ConfigurationError {
        message: "Invalid learning rate: 2.5. Must be between 0.0001 and 1.0".to_string(),
        suggestions: vec!["Use learning rates between 0.01 and 0.3 for most problems".to_string()],
        context: ErrorContext {
            pipeline_stage: "model_training".to_string(),
            component_name: "gradient_boosting_regressor".to_string(),
            input_shape: Some((1000, 20)),
            parameters: {
                let mut params = HashMap::new();
                params.insert("learning_rate".to_string(), "2.5".to_string());
                params.insert("n_estimators".to_string(), "100".to_string());
                params
            },
            stack_trace: vec![
                "Pipeline::fit()".to_string(),
                "GradientBoostingRegressor::fit()".to_string(),
                "validate_hyperparameters()".to_string(),
            ],
        },
    };

    let enhanced_error = ErrorMessageEnhancer::enhance_error(config_error);
    println!("{}", enhanced_error);

    // Scenario 2: Data Compatibility Error
    println!("\n📋 Scenario 2: Data Shape Mismatch with Debugging Tips");
    let shape_error = PipelineError::DataCompatibilityError {
        expected: DataShape {
            samples: 1000,
            features: 20,
            data_type: "f64".to_string(),
            missing_values: false,
        },
        actual: DataShape {
            samples: 1000,
            features: 15,
            data_type: "f64".to_string(),
            missing_values: true,
        },
        stage: "feature_selection".to_string(),
        suggestions: vec![
            "Check if feature selection removed too many features".to_string(),
            "Verify preprocessing pipeline maintains expected feature count".to_string(),
        ],
    };

    let enhanced_shape_error = ErrorMessageEnhancer::enhance_error(shape_error);
    println!("{}", enhanced_shape_error);

    // Scenario 3: Performance Warning
    println!("\n📋 Scenario 3: Performance Warning with Optimization Suggestions");
    let perf_warning = PipelineError::PerformanceWarning {
        warning_type: PerformanceWarningType::MemoryUsage,
        impact_level: ImpactLevel::High,
        suggestions: vec![
            "Consider using batch processing to reduce memory footprint".to_string(),
            "Enable streaming mode for large datasets".to_string(),
        ],
        metrics: Some(PerformanceMetrics {
            execution_time_ms: 45000.0, // 45 seconds
            memory_usage_mb: 8192.0,    // 8GB
            cpu_utilization: 95.0,
            cache_hit_ratio: 0.15, // Poor cache performance
        }),
    };

    let enhanced_perf_error = ErrorMessageEnhancer::enhance_error(perf_warning);
    println!("{}", enhanced_perf_error);

    Ok(())
}

/// Demonstrate interactive debugging session
fn demo_interactive_debugging() -> SklResult<()> {
    println!("\n🐛 Interactive Debugging Session Demo");
    println!("{}", "=".repeat(60));

    let (X, y) = generate_real_world_dataset();
    let mut debugger = DeveloperPipelineDebugger::new();

    // Start debugging session
    debugger.start_debug_session()?;

    // Set up some breakpoints
    let bp1 = debugger.add_breakpoint("data_preprocessing".to_string(), None);
    let bp2 = debugger.add_breakpoint(
        "model_training".to_string(),
        Some("input_shape[1] != 4".to_string()),
    );

    // Add some watch expressions
    debugger.add_watch("data_shape".to_string(), "X.shape()".to_string());
    debugger.add_watch(
        "memory_usage".to_string(),
        "current_memory_mb()".to_string(),
    );
    debugger.add_watch("training_loss".to_string(), "model.loss()".to_string());

    println!("\n📊 Debug Session Setup:");
    println!("   • Session ID: {}", debugger.session_id);
    println!("   • Breakpoints: {} active", debugger.breakpoints.len());
    println!(
        "   • Watch expressions: {} active",
        debugger.watch_expressions.len()
    );

    // Simulate some execution with trace recording
    let trace_entries = vec![
        TraceEntry {
            timestamp: 1630000000000,
            component: "data_preprocessing".to_string(),
            operation: "standardize_features".to_string(),
            duration_ms: 15.3,
            input_shape: Some((10, 4)),
            output_shape: Some((10, 4)),
            memory_before_mb: 45.2,
            memory_after_mb: 47.1,
            notes: vec!["Applied StandardScaler".to_string()],
        },
        TraceEntry {
            timestamp: 1630000000015,
            component: "feature_engineering".to_string(),
            operation: "polynomial_features".to_string(),
            duration_ms: 8.7,
            input_shape: Some((10, 4)),
            output_shape: Some((10, 14)), // Polynomial expansion
            memory_before_mb: 47.1,
            memory_after_mb: 52.3,
            notes: vec!["Generated polynomial features up to degree 2".to_string()],
        },
        TraceEntry {
            timestamp: 1630000000024,
            component: "model_training".to_string(),
            operation: "fit_linear_regression".to_string(),
            duration_ms: 156.8,
            input_shape: Some((10, 14)),
            output_shape: None,
            memory_before_mb: 52.3,
            memory_after_mb: 58.9,
            notes: vec![
                "Training completed".to_string(),
                "R² score: 0.847".to_string(),
                "RMSE: 45.2".to_string(),
            ],
        },
    ];

    for entry in trace_entries {
        debugger.record_trace(entry);
    }

    // Display debugging summary
    let summary = debugger.get_debug_summary();
    println!("\n📈 Debugging Summary:");
    println!("   • Total trace entries: {}", summary.total_trace_entries);
    println!(
        "   • Total execution time: {:.1}ms",
        summary.total_execution_time_ms
    );
    println!(
        "   • Peak memory usage: {:.1}MB",
        summary.peak_memory_usage_mb
    );
    println!("   • Breakpoints hit: 0"); // None hit in this simulation

    // Show detailed trace
    println!("\n🔍 Execution Trace:");
    for (i, entry) in debugger.execution_trace.iter().enumerate() {
        println!(
            "   {}. {} → {} ({:.1}ms)",
            i + 1,
            entry.component,
            entry.operation,
            entry.duration_ms
        );
        if let (Some(input), Some(output)) = (entry.input_shape, entry.output_shape) {
            println!(
                "      Shape: ({}, {}) → ({}, {})",
                input.0, input.1, output.0, output.1
            );
        }
        println!(
            "      Memory: {:.1}MB → {:.1}MB",
            entry.memory_before_mb, entry.memory_after_mb
        );
        for note in &entry.notes {
            println!("      Note: {}", note);
        }
    }

    Ok(())
}

/// Demonstrate real-world pipeline debugging scenario
fn demo_realistic_debugging_scenario() -> SklResult<()> {
    println!("\n🏠 Real-World Scenario: Housing Price Prediction Pipeline");
    println!("{}", "=".repeat(60));

    let (X, y) = generate_real_world_dataset();

    println!("📊 Dataset Overview:");
    println!("   • Features: {} (sqft, bedrooms, floors, age)", X.ncols());
    println!("   • Samples: {}", X.nrows());
    println!("   • Target: House prices in thousands of dollars");

    // Show some sample data
    println!("\n📋 Sample Data:");
    println!("   Features:     [sqft, bed, floors, age] → Price");
    for i in 0..5.min(X.nrows()) {
        println!(
            "   Sample {}: [{:6.0}, {:3.0}, {:3.0}, {:3.0}] → ${:.0}k",
            i + 1,
            X[[i, 0]],
            X[[i, 1]],
            X[[i, 2]],
            X[[i, 3]],
            y[i]
        );
    }

    // Create a pipeline with potential issues for debugging
    println!("\n🔧 Building ML Pipeline with Debug Monitoring:");

    let mut debugger = DeveloperPipelineDebugger::new();
    debugger.start_debug_session()?;

    // Add comprehensive breakpoints
    debugger.add_breakpoint(
        "data_validation".to_string(),
        Some("missing_values > 0".to_string()),
    );
    debugger.add_breakpoint("feature_scaling".to_string(), None);
    debugger.add_breakpoint(
        "model_training".to_string(),
        Some("training_time > 1000ms".to_string()),
    );

    // Add monitoring expressions
    debugger.add_watch(
        "feature_correlation".to_string(),
        "corr_matrix.max()".to_string(),
    );
    debugger.add_watch(
        "training_convergence".to_string(),
        "loss_history[-1]".to_string(),
    );
    debugger.add_watch(
        "memory_efficiency".to_string(),
        "memory_per_sample".to_string(),
    );

    // Simulate pipeline execution with realistic issues
    let realistic_trace = vec![
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64,
            component: "data_validation".to_string(),
            operation: "check_missing_values".to_string(),
            duration_ms: 2.1,
            input_shape: Some((10, 4)),
            output_shape: Some((10, 4)),
            memory_before_mb: 15.3,
            memory_after_mb: 15.3,
            notes: vec!["No missing values detected".to_string()],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 2,
            component: "feature_scaling".to_string(),
            operation: "standard_scaler_fit_transform".to_string(),
            duration_ms: 12.7,
            input_shape: Some((10, 4)),
            output_shape: Some((10, 4)),
            memory_before_mb: 15.3,
            memory_after_mb: 18.9,
            notes: vec![
                "Features standardized".to_string(),
                "Mean: [1652.4, 3.4, 1.4, 24.6]".to_string(),
                "Std: [440.1, 0.7, 0.5, 12.8]".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 15,
            component: "model_training".to_string(),
            operation: "linear_regression_fit".to_string(),
            duration_ms: 89.4,
            input_shape: Some((10, 4)),
            output_shape: None,
            memory_before_mb: 18.9,
            memory_after_mb: 22.1,
            notes: vec![
                "Training completed successfully".to_string(),
                "R² score: 0.823".to_string(),
                "RMSE: 51.7".to_string(),
                "Feature importance: [0.68, 0.15, 0.12, 0.05]".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 105,
            component: "model_evaluation".to_string(),
            operation: "cross_validation".to_string(),
            duration_ms: 234.6,
            input_shape: Some((10, 4)),
            output_shape: None,
            memory_before_mb: 22.1,
            memory_after_mb: 28.7,
            notes: vec![
                "5-fold cross-validation completed".to_string(),
                "CV R² mean: 0.756 ± 0.142".to_string(),
                "CV RMSE mean: 62.3 ± 18.9".to_string(),
                "Potential overfitting detected".to_string(),
            ],
        },
    ];

    for entry in realistic_trace {
        debugger.record_trace(entry);
    }

    // Analyze results and provide insights
    let summary = debugger.get_debug_summary();
    println!("\n📈 Pipeline Analysis Results:");
    println!(
        "   • Total execution time: {:.1}ms",
        summary.total_execution_time_ms
    );
    println!(
        "   • Peak memory usage: {:.1}MB",
        summary.peak_memory_usage_mb
    );
    println!(
        "   • Memory efficiency: {:.2}MB per sample",
        summary.peak_memory_usage_mb / 10.0
    );

    // Show performance insights
    println!("\n💡 Performance Insights:");
    if summary.total_execution_time_ms > 200.0 {
        println!("   ⚠️  Execution time is high for this dataset size");
        println!("      Consider: Feature selection, simpler models, or caching");
    }

    if summary.peak_memory_usage_mb > 25.0 {
        println!("   ⚠️  Memory usage is high for this dataset size");
        println!("      Consider: Streaming processing or batch operations");
    }

    // Show detailed performance trace
    println!("\n🔍 Detailed Execution Trace:");
    for (i, entry) in debugger.execution_trace.iter().enumerate() {
        let efficiency = if entry.duration_ms > 0.0 {
            entry
                .input_shape
                .map_or(0.0, |(samples, _)| samples as f64 / entry.duration_ms)
        } else {
            0.0
        };

        println!(
            "   {}. {} ({:.1}ms, {:.0} samples/ms)",
            i + 1,
            entry.operation,
            entry.duration_ms,
            efficiency
        );

        for note in &entry.notes {
            if note.contains("overfitting") {
                println!("      🔴 {}", note);
            } else if note.contains("score") || note.contains("RMSE") {
                println!("      📊 {}", note);
            } else {
                println!("      ℹ️  {}", note);
            }
        }
    }

    // Provide actionable recommendations
    println!("\n🎯 Actionable Recommendations:");
    println!("   1. 🟡 Model Complexity: CV score variance is high (±0.142)");
    println!("      → Consider regularization (Ridge/Lasso) or feature selection");
    println!("   2. 🟢 Feature Importance: Square footage dominates (68%)");
    println!("      → Feature engineering looks good, consider polynomial features");
    println!("   3. 🟡 Data Size: Only 10 samples for 4 features");
    println!("      → Collect more data or use simpler models to avoid overfitting");

    println!("\n💻 Next Steps for Development:");
    println!("   • Try regularized models: Ridge(alpha=0.1) or Lasso(alpha=0.01)");
    println!("   • Add cross-validation to hyperparameter tuning");
    println!("   • Implement feature selection to reduce overfitting");
    println!("   • Consider ensemble methods for better generalization");

    Ok(())
}

/// Main demonstration function
fn main() -> SklResult<()> {
    println!("🚀 sklears-compose Interactive Developer Experience Demo");
    println!("{}", "=".repeat(80));
    println!("This demo showcases enhanced developer experience features:");
    println!("• Enhanced error messages with actionable suggestions");
    println!("• Interactive debugging and pipeline inspection");
    println!("• Real-world scenarios with comprehensive analysis");
    println!("• Performance monitoring and optimization tips");

    // Run all demonstrations
    demo_enhanced_error_messages()?;
    demo_interactive_debugging()?;
    demo_realistic_debugging_scenario()?;

    println!("\n🎉 Developer Experience Demo Completed!");
    println!("\n🔧 Available Developer Tools:");
    println!("• ErrorMessageEnhancer::enhance_error() - Get detailed error explanations");
    println!("• DeveloperPipelineDebugger::new() - Start interactive debugging sessions");
    println!("• Add breakpoints with debugger.add_breakpoint()");
    println!("• Monitor values with debugger.add_watch()");
    println!("• Get performance insights with debugger.get_debug_summary()");

    println!("\n📚 Additional Resources:");
    println!("• Documentation: https://docs.sklears.com/developer-experience");
    println!("• Debugging Guide: https://docs.sklears.com/debugging-pipelines");
    println!("• Performance Tips: https://docs.sklears.com/performance-optimization");
    println!("• Best Practices: https://docs.sklears.com/best-practices");

    println!("\n🎯 Try These Next:");
    println!("• cargo run --example comprehensive_ml_pipeline");
    println!("• cargo run --example performance_profiling");
    println!("• cargo run --example automl_integration");

    Ok(())
}
