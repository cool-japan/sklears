//! Enhanced Error Handling Examples
//!
//! This example demonstrates the enhanced error handling system with actionable
//! suggestions, detailed context, and debugging information for common pipeline issues.
//!
//! Run with: cargo run --example enhanced_error_handling

use ndarray::{array, Array1, Array2};
use sklears_compose::{
    enhanced_errors::{
        DataShape, EnhancedErrorBuilder, ErrorContext, ImpactLevel, PerformanceMetrics,
        PerformanceWarningType, PipelineError, ResourceType, StructureErrorType, TypeViolationType,
    },
    mock::{MockPredictor, MockTransformer},
    Pipeline, PipelineBuilder,
};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;

/// Demonstrate configuration error handling
fn demo_configuration_errors() {
    println!("🔧 Configuration Error Handling Demo");
    println!("{}", "=".repeat(50));

    // Simulate common configuration errors
    let errors = vec![
        "Invalid parameter 'learning_rate': must be between 0.0001 and 1.0",
        "Missing required parameter 'n_estimators'",
        "Incompatible parameter combination: cannot use 'auto_scaling' with 'manual_features'",
    ];

    for (i, error_msg) in errors.iter().enumerate() {
        println!("\n📋 Example {} - Configuration Error:", i + 1);
        let error = PipelineError::configuration(error_msg);
        println!("{}", error);
    }

    // Demonstrate enhanced error builder
    println!("\n🔧 Enhanced Error Builder Example:");
    let context = ErrorContext {
        pipeline_stage: "model_training".to_string(),
        component_name: "gradient_boosting".to_string(),
        input_shape: Some((1000, 20)),
        parameters: {
            let mut params = HashMap::new();
            params.insert("learning_rate".to_string(), "2.5".to_string());
            params.insert("max_depth".to_string(), "50".to_string());
            params
        },
        stack_trace: vec![
            "Pipeline::fit()".to_string(),
            "GradientBoostingRegressor::fit()".to_string(),
            "validate_parameters()".to_string(),
        ],
    };

    let enhanced_error = EnhancedErrorBuilder::new()
        .configuration_error("Learning rate parameter out of valid range")
        .context(context)
        .suggestion("Try using learning rates between 0.01 and 0.3 for gradient boosting")
        .suggestion("Consider using AutoML to find optimal hyperparameters")
        .build();

    println!("{}", enhanced_error);
}

/// Demonstrate data compatibility error handling
fn demo_data_compatibility_errors() {
    println!("\n📊 Data Compatibility Error Handling Demo");
    println!("{}", "=".repeat(50));

    // Feature count mismatch
    let expected_shape = DataShape {
        samples: 100,
        features: 10,
        data_type: "float64".to_string(),
        missing_values: false,
    };

    let actual_shape = DataShape {
        samples: 100,
        features: 8,
        data_type: "float64".to_string(),
        missing_values: true,
    };

    println!("\n📋 Example 1 - Feature Count Mismatch:");
    let compatibility_error = PipelineError::data_compatibility(
        expected_shape.clone(),
        actual_shape,
        "feature_selection_stage",
    );
    println!("{}", compatibility_error);

    // Data type mismatch
    let type_mismatch_actual = DataShape {
        samples: 100,
        features: 10,
        data_type: "int32".to_string(),
        missing_values: false,
    };

    println!("\n📋 Example 2 - Data Type Mismatch:");
    let type_error = PipelineError::data_compatibility(
        expected_shape,
        type_mismatch_actual,
        "normalization_stage",
    );
    println!("{}", type_error);

    // Sample size mismatch
    let sample_mismatch = DataShape {
        samples: 50,
        features: 10,
        data_type: "float64".to_string(),
        missing_values: false,
    };

    let expected_samples = DataShape {
        samples: 100,
        features: 10,
        data_type: "float64".to_string(),
        missing_values: false,
    };

    println!("\n📋 Example 3 - Sample Count Mismatch:");
    let sample_error =
        PipelineError::data_compatibility(expected_samples, sample_mismatch, "cross_validation");
    println!("{}", sample_error);
}

/// Demonstrate pipeline structure errors
fn demo_structure_errors() {
    println!("\n🏗️ Pipeline Structure Error Handling Demo");
    println!("{}", "=".repeat(50));

    // Cyclic dependency error
    println!("\n📋 Example 1 - Cyclic Dependency:");
    let cyclic_error = EnhancedErrorBuilder::new()
        .structure_error(
            StructureErrorType::CyclicDependency,
            vec![
                "feature_extractor".to_string(),
                "feature_selector".to_string(),
            ],
        )
        .build();
    println!("{}", cyclic_error);

    // Missing component error
    println!("\n📋 Example 2 - Missing Component:");
    let missing_error = EnhancedErrorBuilder::new()
        .structure_error(
            StructureErrorType::MissingComponent,
            vec!["data_preprocessor".to_string()],
        )
        .build();
    println!("{}", missing_error);

    // Invalid connection error
    println!("\n📋 Example 3 - Invalid Connection:");
    let connection_error = EnhancedErrorBuilder::new()
        .structure_error(
            StructureErrorType::InvalidConnection,
            vec![
                "text_vectorizer".to_string(),
                "image_classifier".to_string(),
            ],
        )
        .build();
    println!("{}", connection_error);
}

/// Demonstrate performance warnings
fn demo_performance_warnings() {
    println!("\n⚡ Performance Warning Handling Demo");
    println!("{}", "=".repeat(50));

    // Memory usage warning
    let memory_metrics = PerformanceMetrics {
        execution_time_ms: 5240.0,
        memory_usage_mb: 1536.0,
        cpu_utilization: 85.2,
        cache_hit_ratio: 0.45,
    };

    println!("\n📋 Example 1 - High Memory Usage:");
    let memory_warning = PipelineError::performance_warning(
        PerformanceWarningType::MemoryUsage,
        ImpactLevel::High,
        Some(memory_metrics),
    );
    println!("{}", memory_warning);

    // Cache inefficiency warning
    let cache_metrics = PerformanceMetrics {
        execution_time_ms: 8750.0,
        memory_usage_mb: 512.0,
        cpu_utilization: 45.1,
        cache_hit_ratio: 0.12,
    };

    println!("\n📋 Example 2 - Cache Inefficiency:");
    let cache_warning = PipelineError::performance_warning(
        PerformanceWarningType::CacheInefficiency,
        ImpactLevel::Medium,
        Some(cache_metrics),
    );
    println!("{}", cache_warning);

    // Computational complexity warning
    println!("\n📋 Example 3 - High Computational Complexity:");
    let complexity_warning = PipelineError::performance_warning(
        PerformanceWarningType::ComputationalComplexity,
        ImpactLevel::Critical,
        None,
    );
    println!("{}", complexity_warning);
}

/// Demonstrate resource constraint errors
fn demo_resource_errors() {
    println!("\n🔋 Resource Constraint Error Handling Demo");
    println!("{}", "=".repeat(50));

    // Memory constraint violation
    println!("\n📋 Example 1 - Memory Constraint:");
    let memory_error = EnhancedErrorBuilder::new()
        .resource_error(
            ResourceType::Memory,
            8192.0, // 8GB limit
            8756.3, // 8.7GB current
            "neural_network_trainer",
        )
        .build();
    println!("{}", memory_error);

    // GPU memory constraint
    println!("\n📋 Example 2 - GPU Memory Constraint:");
    let gpu_error = EnhancedErrorBuilder::new()
        .resource_error(
            ResourceType::GPU,
            11264.0, // 11GB GPU memory
            10890.5, // Current usage
            "cnn_model",
        )
        .build();
    println!("{}", gpu_error);

    // CPU constraint
    println!("\n📋 Example 3 - CPU Constraint:");
    let cpu_error = EnhancedErrorBuilder::new()
        .resource_error(
            ResourceType::CPU,
            100.0, // 100% CPU
            98.7,  // Current usage
            "parallel_ensemble",
        )
        .build();
    println!("{}", cpu_error);
}

/// Demonstrate type safety errors
fn demo_type_safety_errors() {
    println!("\n🛡️ Type Safety Error Handling Demo");
    println!("{}", "=".repeat(50));

    // Input type incompatibility
    println!("\n📋 Example 1 - Incompatible Input Type:");
    let input_type_error = EnhancedErrorBuilder::new()
        .type_safety_error(
            TypeViolationType::IncompatibleInputType,
            "Array2<f64>",
            "Array1<i32>",
            "matrix_multiplication",
        )
        .build();
    println!("{}", input_type_error);

    // Parameter type mismatch
    println!("\n📋 Example 2 - Invalid Parameter Type:");
    let param_error = EnhancedErrorBuilder::new()
        .type_safety_error(
            TypeViolationType::InvalidParameterType,
            "f64",
            "String",
            "hyperparameter_setting",
        )
        .build();
    println!("{}", param_error);

    // Unsupported transformation
    println!("\n📋 Example 3 - Unsupported Transformation:");
    let transform_error = EnhancedErrorBuilder::new()
        .type_safety_error(
            TypeViolationType::UnsupportedTransformation,
            "numerical_data",
            "categorical_text",
            "standard_scaler",
        )
        .build();
    println!("{}", transform_error);
}

/// Demonstrate real-world pipeline error scenario
fn demo_realistic_pipeline_error() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Realistic Pipeline Error Scenario");
    println!("{}", "=".repeat(50));

    // Simulate a complex pipeline with potential errors
    let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let y = array![1.0, 2.0, 3.0];

    // Try to create a pipeline that will encounter various errors
    println!("\n📋 Scenario: Building a complex pipeline with error detection");

    // 1. Data shape validation
    let expected_features = 5;
    let actual_features = X.ncols();

    if actual_features != expected_features {
        let expected_shape = DataShape {
            samples: X.nrows(),
            features: expected_features,
            data_type: "f64".to_string(),
            missing_values: false,
        };

        let actual_shape = DataShape {
            samples: X.nrows(),
            features: actual_features,
            data_type: "f64".to_string(),
            missing_values: false,
        };

        let shape_error = PipelineError::data_compatibility(
            expected_shape,
            actual_shape,
            "pipeline_initialization",
        );
        println!("{}", shape_error);
    }

    // 2. Performance monitoring simulation
    let performance_metrics = PerformanceMetrics {
        execution_time_ms: 15000.0, // 15 seconds - quite slow
        memory_usage_mb: 2048.0,    // 2GB memory usage
        cpu_utilization: 95.0,      // High CPU usage
        cache_hit_ratio: 0.25,      // Poor cache performance
    };

    let perf_warning = PipelineError::performance_warning(
        PerformanceWarningType::ComputationalComplexity,
        ImpactLevel::High,
        Some(performance_metrics),
    );
    println!("{}", perf_warning);

    // 3. Resource constraint simulation
    let resource_warning = EnhancedErrorBuilder::new()
        .resource_error(
            ResourceType::Memory,
            4096.0, // 4GB limit
            3890.2, // Near limit
            "feature_engineering_stage",
        )
        .suggestion("Consider using incremental learning to reduce memory usage")
        .build();
    println!("{}", resource_warning);

    println!("\n✅ Error handling demonstration completed successfully!");
    println!("💡 The enhanced error system helps identify and resolve pipeline issues quickly.");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 sklears-compose Enhanced Error Handling Examples");
    println!("=".repeat(60));
    println!("This example demonstrates comprehensive error handling:");
    println!("• Configuration errors with parameter suggestions");
    println!("• Data compatibility issues with shape analysis");
    println!("• Pipeline structure problems with fix recommendations");
    println!("• Performance warnings with optimization tips");
    println!("• Resource constraint violations with scaling advice");
    println!("• Type safety errors with conversion suggestions");

    // Run all error handling demonstrations
    demo_configuration_errors();
    demo_data_compatibility_errors();
    demo_structure_errors();
    demo_performance_warnings();
    demo_resource_errors();
    demo_type_safety_errors();
    demo_realistic_pipeline_error()?;

    println!("\n🎉 All error handling examples completed!");

    println!("\n💡 Enhanced Error Handling Benefits:");
    println!("• Provides actionable suggestions for fixing issues");
    println!("• Includes detailed context for debugging");
    println!("• Categorizes errors by type and impact level");
    println!("• Offers performance optimization recommendations");
    println!("• Helps prevent common pipeline composition mistakes");

    println!("\n🔧 Usage Tips:");
    println!("• Use EnhancedErrorBuilder for custom errors");
    println!("• Check error suggestions before asking for help");
    println!("• Monitor performance warnings proactively");
    println!("• Validate pipeline structure before execution");
    println!("• Use type safety features to catch errors early");

    println!("\n📚 Next Steps:");
    println!("• Try: cargo run --example performance_profiling");
    println!("• Try: cargo run --example pipeline_debugging");
    println!("• Try: cargo bench simd_benchmarks");

    Ok(())
}
