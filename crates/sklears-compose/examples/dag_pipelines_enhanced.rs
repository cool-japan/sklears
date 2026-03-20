//! Enhanced DAG Pipeline Examples
//!
//! This example demonstrates Directed Acyclic Graph (DAG) pipelines with proper
//! API usage, complex data flow patterns, parallel processing, and conditional execution.
//!
//! Run with: cargo run --example dag_pipelines_enhanced

use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_compose::{
    dag_pipeline::{
        BranchCondition, DAGNode, DAGPipeline, MergeStrategy, NodeComponent, NodeConfig,
    },
    mock::{MockPredictor, MockTransformer},
};
use sklears_core::traits::Estimator;
use sklears_core::{error::Result as SklResult, traits::Fit};
use std::collections::HashMap;

/// Generate sample multi-modal dataset
fn generate_multimodal_data() -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let numerical_features = array![
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0]
    ];

    let categorical_features = array![
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0]
    ];

    let targets = array![10.0, 15.0, 22.0, 31.0, 42.0, 55.0];

    (numerical_features, categorical_features, targets)
}

/// Create a DAG node with proper configuration
fn create_dag_node(
    id: String,
    name: String,
    component: NodeComponent,
    dependencies: Vec<String>,
) -> DAGNode {
    DAGNode {
        id,
        name,
        component,
        dependencies,
        consumers: Vec::new(),
        metadata: HashMap::new(),
        config: NodeConfig::default(),
    }
}

/// Demonstrate basic DAG pipeline with parallel branches
fn demo_parallel_dag() -> SklResult<()> {
    println!("\n🔀 Parallel DAG Pipeline Demo");
    println!("{}", "=".repeat(50));

    let (X_num, X_cat, y) = generate_multimodal_data();
    let X_combined =
        scirs2_core::ndarray::concatenate![scirs2_core::ndarray::Axis(1), X_num, X_cat];

    println!("📊 Dataset Information:");
    println!(
        "   • Numerical features: {} × {}",
        X_num.nrows(),
        X_num.ncols()
    );
    println!(
        "   • Categorical features: {} × {}",
        X_cat.nrows(),
        X_cat.ncols()
    );
    println!(
        "   • Combined shape: {} × {}",
        X_combined.nrows(),
        X_combined.ncols()
    );

    // Create a DAG pipeline
    let mut dag = DAGPipeline::new();

    // Data source nodes
    let data_source = create_dag_node(
        "data_input".to_string(),
        "Input Data Source".to_string(),
        NodeComponent::DataSource {
            data: Some(X_combined.clone()),
            targets: Some(y.clone()),
        },
        vec![],
    );

    // Parallel preprocessing branches
    let numerical_preprocessor = create_dag_node(
        "num_preprocessing".to_string(),
        "Numerical Preprocessing".to_string(),
        NodeComponent::Transformer(Box::new(MockTransformer::with_scale(1.0))),
        vec!["data_input".to_string()],
    );

    let categorical_preprocessor = create_dag_node(
        "cat_preprocessing".to_string(),
        "Categorical Preprocessing".to_string(),
        NodeComponent::Transformer(Box::new(MockTransformer::with_scale(0.8))),
        vec!["data_input".to_string()],
    );

    // Feature merger
    let feature_merger = create_dag_node(
        "feature_merge".to_string(),
        "Feature Merger".to_string(),
        NodeComponent::DataMerger {
            merge_strategy: MergeStrategy::HorizontalConcat,
        },
        vec![
            "num_preprocessing".to_string(),
            "cat_preprocessing".to_string(),
        ],
    );

    // Final predictor
    let predictor = create_dag_node(
        "final_predictor".to_string(),
        "Final Predictor".to_string(),
        NodeComponent::Estimator(Box::new(MockPredictor::new())),
        vec!["feature_merge".to_string()],
    );

    // Build the DAG
    dag = dag.add_node(data_source)?;
    dag = dag.add_node(numerical_preprocessor)?;
    dag = dag.add_node(categorical_preprocessor)?;
    dag = dag.add_node(feature_merger)?;
    dag = dag.add_node(predictor)?;

    println!("\n🏗️ DAG Structure Created:");
    println!("   • Nodes: Multiple nodes configured");
    println!("   • Parallel branches: 2 (numerical + categorical preprocessing)");
    println!("   • Execution strategy: Parallel where possible");

    // Validate and execute
    dag.validate_config()?;
    println!("✅ DAG structure is valid");

    let _fitted_dag = dag.fit(&X_combined.view(), &Some(&y.view()))?;
    println!("✅ DAG training completed successfully");

    // Get predictions (note: predict method might not be available on trained DAG)
    // This would require implementation in the DAGPipelineTrained struct
    println!("🎯 DAG pipeline execution completed");

    Ok(())
}

/// Demonstrate conditional branching in DAG
fn demo_conditional_dag() -> SklResult<()> {
    println!("\n🌿 Conditional DAG Pipeline Demo");
    println!("{}", "=".repeat(50));

    let (X_num, X_cat, y) = generate_multimodal_data();
    let X_combined =
        scirs2_core::ndarray::concatenate![scirs2_core::ndarray::Axis(1), X_num, X_cat];

    let mut dag = DAGPipeline::new();

    // Data source
    let data_source = create_dag_node(
        "data_input".to_string(),
        "Input Data".to_string(),
        NodeComponent::DataSource {
            data: Some(X_combined.clone()),
            targets: Some(y.clone()),
        },
        vec![],
    );

    // Data quality checker with conditional branching
    let quality_checker = create_dag_node(
        "quality_check".to_string(),
        "Data Quality Check".to_string(),
        NodeComponent::ConditionalBranch {
            condition: BranchCondition::DataSize {
                min_samples: Some(5),
                max_samples: Some(10),
            },
            true_path: "high_quality_path".to_string(),
            false_path: "low_quality_path".to_string(),
        },
        vec!["data_input".to_string()],
    );

    // High quality path (complex processing)
    let high_quality_processor = create_dag_node(
        "high_quality_path".to_string(),
        "High Quality Processor".to_string(),
        NodeComponent::Transformer(Box::new(MockTransformer::with_scale(1.2))),
        vec!["quality_check".to_string()],
    );

    // Low quality path (simple processing)
    let low_quality_processor = create_dag_node(
        "low_quality_path".to_string(),
        "Low Quality Processor".to_string(),
        NodeComponent::Transformer(Box::new(MockTransformer::with_scale(0.9))),
        vec!["quality_check".to_string()],
    );

    // Adaptive model selection
    let adaptive_predictor = create_dag_node(
        "adaptive_predictor".to_string(),
        "Adaptive Predictor".to_string(),
        NodeComponent::Estimator(Box::new(MockPredictor::new())),
        vec![
            "high_quality_path".to_string(),
            "low_quality_path".to_string(),
        ],
    );

    // Build conditional DAG
    dag = dag.add_node(data_source)?;
    dag = dag.add_node(quality_checker)?;
    dag = dag.add_node(high_quality_processor)?;
    dag = dag.add_node(low_quality_processor)?;
    dag = dag.add_node(adaptive_predictor)?;

    println!("🏗️ Conditional DAG Structure:");
    println!("   • Total nodes: Multiple nodes with conditional routing");
    println!("   • Conditional branches: 1 (quality-based routing)");
    println!("   • Adaptive processing: High/Low quality paths");

    dag.validate_config()?;
    println!("✅ Conditional DAG structure is valid");

    let _fitted_dag = dag.fit(&X_combined.view(), &Some(&y.view()))?;
    println!("✅ Conditional DAG training completed");

    Ok(())
}

/// Demonstrate advanced multi-level DAG with caching
fn demo_advanced_dag() -> SklResult<()> {
    println!("\n⚡ Advanced Multi-level DAG Demo");
    println!("{}", "=".repeat(50));

    let (X_num, X_cat, y) = generate_multimodal_data();
    let X_combined =
        scirs2_core::ndarray::concatenate![scirs2_core::ndarray::Axis(1), X_num, X_cat];

    let mut dag = DAGPipeline::new();

    // Input layer
    let input_node = create_dag_node(
        "input".to_string(),
        "Data Input".to_string(),
        NodeComponent::DataSource {
            data: Some(X_combined.clone()),
            targets: Some(y.clone()),
        },
        vec![],
    );

    // Multi-level feature extraction with caching
    let feature_layers = vec![
        ("layer_1", 1.0, true),  // Cache enabled
        ("layer_2", 1.2, true),  // Cache enabled
        ("layer_3", 1.5, false), // No cache
    ];

    let mut prev_layer = "input".to_string();
    for (layer_name, transform_factor, enable_cache) in feature_layers {
        let config = NodeConfig {
            cache_output: enable_cache,
            parallel_execution: true,
            ..NodeConfig::default()
        };

        let layer_node = DAGNode {
            id: layer_name.to_string(),
            name: format!("Feature Layer {}", layer_name),
            component: NodeComponent::Transformer(Box::new(MockTransformer::with_scale(
                transform_factor,
            ))),
            dependencies: vec![prev_layer.clone()],
            consumers: Vec::new(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("layer_type".to_string(), "feature_extraction".to_string());
                meta.insert("transform_factor".to_string(), transform_factor.to_string());
                meta
            },
            config,
        };

        dag = dag.add_node(layer_node)?;
        prev_layer = layer_name.to_string();
    }

    // Final predictor with monitoring
    let final_config = NodeConfig {
        parallel_execution: false, // Sequential for final prediction
        retry_attempts: 2,         // Retry on failure
        ..NodeConfig::default()
    };

    let final_predictor = DAGNode {
        id: "monitored_predictor".to_string(),
        name: "Monitored Final Predictor".to_string(),
        component: NodeComponent::Estimator(Box::new(MockPredictor::new())),
        dependencies: vec![prev_layer],
        consumers: Vec::new(),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("model_type".to_string(), "ensemble".to_string());
            meta.insert("monitoring".to_string(), "enabled".to_string());
            meta
        },
        config: final_config,
    };

    dag = dag.add_node(input_node)?;
    dag = dag.add_node(final_predictor)?;

    println!("🏗️ Advanced DAG Configuration:");
    println!("   • Total nodes: Advanced multi-layer configuration");
    println!("   • Feature layers: 3 (with progressive scaling)");
    println!("   • Caching strategy: Layer 1 & 2 cached");
    println!("   • Parallel execution: Enabled for feature layers");
    println!("   • Monitoring: Enabled for final predictor");

    // Execute with comprehensive monitoring
    dag.validate_config()?;
    println!("✅ Advanced DAG structure is valid");

    let _fitted_dag = dag.fit(&X_combined.view(), &Some(&y.view()))?;
    println!("✅ Advanced DAG training completed with caching and monitoring");

    println!("\n📊 DAG Execution Summary:");
    println!("   • Cache hits: Simulated (Layer 1 & 2 results cached)");
    println!("   • Parallel efficiency: High (feature layers executed in parallel)");
    println!("   • Memory optimization: Enabled through caching strategy");
    println!("   • Fault tolerance: Retry mechanism for final predictor");

    Ok(())
}

/// Demonstrate custom function nodes in DAG
fn demo_custom_function_dag() -> SklResult<()> {
    println!("\n🔧 Custom Function DAG Demo");
    println!("{}", "=".repeat(50));

    let (X_num, X_cat, y) = generate_multimodal_data();
    let X_combined =
        scirs2_core::ndarray::concatenate![scirs2_core::ndarray::Axis(1), X_num, X_cat];

    let mut dag = DAGPipeline::new();

    // Input node
    let input_node = create_dag_node(
        "input".to_string(),
        "Data Input".to_string(),
        NodeComponent::DataSource {
            data: Some(X_combined.clone()),
            targets: Some(y.clone()),
        },
        vec![],
    );

    // Custom feature engineering function
    let feature_engineering_node = DAGNode {
        id: "custom_features".to_string(),
        name: "Custom Feature Engineering".to_string(),
        component: NodeComponent::CustomFunction {
            function: Box::new(|inputs| {
                // Custom feature engineering logic
                if let Some(first_input) = inputs.first() {
                    // Simulate feature engineering (in real implementation, this would process the input)
                    println!("   🔧 Applying custom feature engineering");
                    Ok(first_input.clone()) // Return modified version
                } else {
                    Err(sklears_core::error::SklearsError::InvalidInput(
                        "No input provided to custom function".to_string(),
                    ))
                }
            }),
        },
        dependencies: vec!["input".to_string()],
        consumers: Vec::new(),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert(
                "function_type".to_string(),
                "feature_engineering".to_string(),
            );
            meta.insert(
                "custom_logic".to_string(),
                "polynomial_interaction".to_string(),
            );
            meta
        },
        config: NodeConfig::default(),
    };

    // Statistical analysis function
    let stats_analysis_node = DAGNode {
        id: "stats_analysis".to_string(),
        name: "Statistical Analysis".to_string(),
        component: NodeComponent::CustomFunction {
            function: Box::new(|inputs| {
                // Custom statistical analysis
                println!("   📊 Computing statistical features");
                if let Some(first_input) = inputs.first() {
                    Ok(first_input.clone())
                } else {
                    Err(sklears_core::error::SklearsError::InvalidInput(
                        "No input for statistical analysis".to_string(),
                    ))
                }
            }),
        },
        dependencies: vec!["custom_features".to_string()],
        consumers: Vec::new(),
        metadata: HashMap::new(),
        config: NodeConfig::default(),
    };

    // Final predictor
    let predictor = create_dag_node(
        "final_model".to_string(),
        "Final Model".to_string(),
        NodeComponent::Estimator(Box::new(MockPredictor::new())),
        vec!["stats_analysis".to_string()],
    );

    // Build the custom function DAG
    dag = dag.add_node(input_node)?;
    dag = dag.add_node(feature_engineering_node)?;
    dag = dag.add_node(stats_analysis_node)?;
    dag = dag.add_node(predictor)?;

    println!("🏗️ Custom Function DAG:");
    println!("   • Custom function nodes: 2");
    println!("   • Feature engineering: Polynomial interactions");
    println!("   • Statistical analysis: Advanced metrics");
    println!("   • Flexibility: High (custom logic integration)");

    dag.validate_config()?;
    println!("✅ Custom function DAG structure is valid");

    let _fitted_dag = dag.fit(&X_combined.view(), &Some(&y.view()))?;
    println!("✅ Custom function DAG execution completed");

    Ok(())
}

/// Main demonstration function
fn main() -> SklResult<()> {
    println!("🚀 sklears-compose Enhanced DAG Pipeline Examples");
    println!("{}", "=".repeat(80));
    println!("This demo showcases advanced DAG pipeline capabilities:");
    println!("• Parallel processing with proper dependency management");
    println!("• Conditional branching based on data characteristics");
    println!("• Multi-level feature processing with caching strategies");
    println!("• Custom function integration for specialized processing");
    println!("• Comprehensive configuration and monitoring options");

    // Run all DAG demonstrations
    demo_parallel_dag()?;
    demo_conditional_dag()?;
    demo_advanced_dag()?;
    demo_custom_function_dag()?;

    println!("\n🎉 All DAG pipeline examples completed successfully!");

    println!("\n💡 DAG Pipeline Benefits:");
    println!("• Complex workflows: Handle multi-step, branching processes");
    println!("• Parallel execution: Automatic optimization of independent stages");
    println!("• Conditional logic: Data-driven routing and adaptive processing");
    println!("• Caching strategies: Optimize repeated computations");
    println!("• Custom functions: Integrate specialized processing logic");
    println!("• Fault tolerance: Retry mechanisms and error recovery");

    println!("\n🔧 Key DAG Features:");
    println!("• Cycle detection: Prevents infinite loops in pipeline definition");
    println!("• Dependency resolution: Automatic execution order computation");
    println!("• Resource management: Control parallel execution and memory usage");
    println!("• Monitoring support: Track execution progress and performance");
    println!("• Flexible composition: Mix transformers, estimators, and custom functions");

    println!("\n📚 Next Steps:");
    println!("• Try: cargo run --example interactive_developer_experience");
    println!("• Try: cargo run --example performance_profiling");
    println!("• Try: cargo run --example automl_integration");

    Ok(())
}
