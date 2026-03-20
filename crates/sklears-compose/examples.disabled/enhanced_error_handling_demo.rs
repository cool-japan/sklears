//! Enhanced Error Handling Demo
//!
//! This example demonstrates the comprehensive error handling system in sklears-compose,
//! showing how detailed error messages with actionable suggestions help developers
//! quickly identify and fix common pipeline issues.
//!
//! Run with: cargo run --example enhanced_error_handling_demo

use scirs2_autograd::ndarray::{array, Array1, Array2};
use sklears_compose::{
    enhanced_errors::{
        DataShape, EnhancedErrorBuilder, ErrorContext, ImpactLevel, PerformanceMetrics,
        PerformanceWarningType, PipelineError, ResourceType, StructureErrorType, TypeViolationType,
    },
    mock::{MockPredictor, MockTransformer},
    FeatureUnion, Pipeline, PipelineBuilder,
};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Predict, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Helper function to simulate shape mismatch for demo purposes
fn simulate_shape_mismatch_error() -> PipelineError {
    let expected = DataShape {
        samples: 100,
        features: 10,
        data_type: "float64".to_string(),
        missing_values: false,
    };

    let actual = DataShape {
        samples: 100,
        features: 5,
        data_type: "float64".to_string(),
        missing_values: false,
    };

    PipelineError::data_compatibility(expected, actual, "problematic_transformer")
}

/// Demonstration functions for different error types

/// Demonstrate configuration errors with detailed suggestions
fn demo_configuration_errors() -> SklResult<()> {
    println!("\n🔧 Configuration Error Demo");
    println!("{}", "=".repeat(50));

    // Simulate a configuration error
    let error = EnhancedErrorBuilder::new()
        .configuration_error("Invalid learning rate parameter: -0.5")
        .suggestion("Learning rates must be positive values")
        .suggestion("Try values between 0.001 and 0.1 for most algorithms")
        .build();

    println!("{}", error);

    // Another configuration error example
    let mut context = ErrorContext::default();
    context.component_name = "LinearRegression".to_string();
    context.pipeline_stage = "model_fitting".to_string();
    context
        .parameters
        .insert("learning_rate".to_string(), "-0.5".to_string());
    context
        .parameters
        .insert("max_iter".to_string(), "1000".to_string());

    let error2 = EnhancedErrorBuilder::new()
        .configuration_error("Missing required parameter: regularization_strength")
        .context(context)
        .build();

    println!("\n{}", error2);

    Ok(())
}

/// Demonstrate data compatibility errors
fn demo_data_compatibility_errors() -> SklResult<()> {
    println!("\n📊 Data Compatibility Error Demo");
    println!("{}", "=".repeat(50));

    // Simulate a data compatibility error without creating an actual pipeline
    println!("Simulating a shape mismatch error that would occur in a pipeline:");

    let enhanced_error = simulate_shape_mismatch_error();
    println!("\n📋 Enhanced Error Details:");
    println!("{}", enhanced_error);

    // Show another example
    let expected = DataShape {
        samples: 1000,
        features: 20,
        data_type: "float32".to_string(),
        missing_values: true,
    };

    let actual = DataShape {
        samples: 1000,
        features: 18,
        data_type: "float64".to_string(),
        missing_values: false,
    };

    let error2 = PipelineError::data_compatibility(expected, actual, "data_preprocessing");
    println!("\n📋 Another Data Compatibility Error:");
    println!("{}", error2);

    Ok(())
}

/// Demonstrate structure errors
fn demo_structure_errors() -> SklResult<()> {
    println!("\n🏗️ Structure Error Demo");
    println!("{}", "=".repeat(50));

    // Simulate cyclic dependency error
    let error = EnhancedErrorBuilder::new()
        .structure_error(
            StructureErrorType::CyclicDependency,
            vec![
                "preprocessor_a".to_string(),
                "preprocessor_b".to_string(),
                "preprocessor_a".to_string(),
            ],
        )
        .build();

    println!("{}", error);

    // Simulate missing component error
    let error2 = EnhancedErrorBuilder::new()
        .structure_error(
            StructureErrorType::MissingComponent,
            vec!["feature_selector".to_string()],
        )
        .build();

    println!("\n{}", error2);

    Ok(())
}

/// Demonstrate performance warnings
fn demo_performance_warnings() -> SklResult<()> {
    println!("\n⚡ Performance Warning Demo");
    println!("{}", "=".repeat(50));

    // High memory usage warning
    let metrics = PerformanceMetrics {
        execution_time_ms: 5000.0,
        memory_usage_mb: 2048.0,
        cpu_utilization: 95.0,
        cache_hit_ratio: 0.3,
    };

    let warning = EnhancedErrorBuilder::new()
        .performance_warning(
            PerformanceWarningType::MemoryUsage,
            ImpactLevel::High,
            Some(metrics.clone()),
        )
        .build();

    println!("{}", warning);

    // Cache inefficiency warning
    let warning2 = EnhancedErrorBuilder::new()
        .performance_warning(
            PerformanceWarningType::CacheInefficiency,
            ImpactLevel::Medium,
            Some(metrics),
        )
        .build();

    println!("\n{}", warning2);

    // Critical computational complexity warning
    let warning3 = EnhancedErrorBuilder::new()
        .performance_warning(
            PerformanceWarningType::ComputationalComplexity,
            ImpactLevel::Critical,
            None,
        )
        .build();

    println!("\n{}", warning3);

    Ok(())
}

/// Demonstrate resource constraint errors
fn demo_resource_errors() -> SklResult<()> {
    println!("\n🔋 Resource Error Demo");
    println!("{}", "=".repeat(50));

    // Memory constraint violation
    let error = EnhancedErrorBuilder::new()
        .resource_error(
            ResourceType::Memory,
            8192.0, // 8GB limit
            7800.0, // 7.8GB current usage
            "large_dataset_processor",
        )
        .build();

    println!("{}", error);

    // GPU memory constraint
    let error2 = EnhancedErrorBuilder::new()
        .resource_error(
            ResourceType::GPU,
            11000.0, // 11GB GPU memory
            10500.0, // 10.5GB used
            "neural_network_trainer",
        )
        .build();

    println!("\n{}", error2);

    Ok(())
}

/// Demonstrate type safety errors
fn demo_type_safety_errors() -> SklResult<()> {
    println!("\n🛡️ Type Safety Error Demo");
    println!("{}", "=".repeat(50));

    // Input type mismatch
    let error = EnhancedErrorBuilder::new()
        .type_safety_error(
            TypeViolationType::IncompatibleInputType,
            "Array2<f64>",
            "Array1<i32>",
            "neural_network_input",
        )
        .build();

    println!("{}", error);

    // Parameter type error
    let error2 = EnhancedErrorBuilder::new()
        .type_safety_error(
            TypeViolationType::InvalidParameterType,
            "f64",
            "String",
            "learning_rate_parameter",
        )
        .build();

    println!("\n{}", error2);

    Ok(())
}

/// Demonstrate error recovery strategies
fn demo_error_recovery() -> SklResult<()> {
    println!("\n🔄 Error Recovery Demo");
    println!("{}", "=".repeat(50));

    // Demonstrate conceptual recovery strategies
    println!("Common ML pipeline error recovery strategies:");

    println!("\n1. 🔧 Shape Mismatch Recovery:");
    println!("   Problem: Expected 10 features, got 8");
    println!("   Solutions:");
    println!("   • Add feature padding with zeros");
    println!("   • Use feature selection to reduce expected features");
    println!("   • Apply dimensionality reduction");

    println!("\n2. 🔄 Type Mismatch Recovery:");
    println!("   Problem: Expected float64, got int32");
    println!("   Solutions:");
    println!("   • Add type conversion transformer");
    println!("   • Use .mapv(|x| x as f64) for numeric conversion");
    println!("   • Handle categorical data with proper encoding");

    println!("\n3. 📊 Missing Data Recovery:");
    println!("   Problem: Model doesn't handle NaN values");
    println!("   Solutions:");
    println!("   • Use SimpleImputer with mean/median/mode strategy");
    println!("   • Apply forward/backward fill for time series");
    println!("   • Remove rows/columns with excessive missing data");

    println!("\n4. 🏗️ Pipeline Structure Recovery:");
    println!("   Problem: Incompatible pipeline stages");
    println!("   Solutions:");
    println!("   • Add adapter transformers between stages");
    println!("   • Reorder pipeline components");
    println!("   • Use ColumnTransformer for heterogeneous data");

    println!("\n5. ⚡ Performance Recovery:");
    println!("   Problem: Pipeline too slow or memory intensive");
    println!("   Solutions:");
    println!("   • Enable chunked/streaming processing");
    println!("   • Use dimensionality reduction");
    println!("   • Apply feature selection");
    println!("   • Switch to approximate algorithms");

    Ok(())
}

/// Generate test data for demonstrations
fn generate_test_data(n_samples: usize, n_features: usize) -> (Array2<Float>, Array1<Float>) {
    let mut X = Array2::<Float>::zeros((n_samples, n_features));
    let mut y = Array1::<Float>::zeros(n_samples);

    // Fill with some dummy data
    for i in 0..n_samples {
        for j in 0..n_features {
            X[[i, j]] = (i + j) as Float * 0.1;
        }
        y[i] = (i as Float * 0.5) + 10.0;
    }

    (X, y)
}

/// Demonstrate error handling best practices
fn demo_best_practices() -> SklResult<()> {
    println!("\n📚 Error Handling Best Practices");
    println!("{}", "=".repeat(50));

    println!("1. 🎯 Always provide context:");
    println!("   - Include component names, pipeline stages, and data characteristics");
    println!("   - Add parameter values and configuration details");

    println!("\n2. 💡 Make suggestions actionable:");
    println!("   - Provide specific parameter ranges and values");
    println!("   - Include code examples where possible");
    println!("   - Explain the reasoning behind suggestions");

    println!("\n3. ⚠️  Use appropriate severity levels:");
    println!("   - Critical: System failures requiring immediate action");
    println!("   - High: Errors preventing execution");
    println!("   - Medium: Issues that may cause problems");
    println!("   - Low: Performance warnings and suggestions");

    println!("\n4. 🔍 Include debugging information:");
    println!("   - Data shapes, types, and value ranges");
    println!("   - Performance metrics and resource usage");
    println!("   - Stack traces and component flow");

    println!("\n5. 🔄 Provide recovery strategies:");
    println!("   - Suggest alternative configurations");
    println!("   - Recommend data preprocessing steps");
    println!("   - Offer fallback options");

    println!("\n6. 📖 Link to documentation:");
    println!("   - Include relevant documentation URLs");
    println!("   - Reference examples and tutorials");
    println!("   - Point to troubleshooting guides");

    Ok(())
}

/// Main demonstration function
fn main() -> SklResult<()> {
    println!("🚀 Enhanced Error Handling Demo");
    println!("{}", "=".repeat(60));
    println!("This demo showcases sklears-compose's comprehensive error handling system");
    println!("with detailed error messages, contextual information, and actionable suggestions.");

    // Run all demonstrations
    demo_configuration_errors()?;
    demo_data_compatibility_errors()?;
    demo_structure_errors()?;
    demo_performance_warnings()?;
    demo_resource_errors()?;
    demo_type_safety_errors()?;
    demo_error_recovery()?;
    demo_best_practices()?;

    println!("\n🎉 Demo completed! These enhanced errors help developers:");
    println!("   • Quickly identify the root cause of issues");
    println!("   • Understand the context and impact of problems");
    println!("   • Get specific, actionable suggestions for fixes");
    println!("   • Learn best practices for avoiding similar issues");
    println!("   • Implement effective recovery strategies");

    Ok(())
}
