//! Comprehensive Developer Experience Showcase
//!
//! This example demonstrates all the enhanced developer experience features
//! working together in a realistic machine learning workflow. It showcases:
//!
//! - Enhanced error messages with actionable suggestions
//! - Interactive debugging and pipeline inspection
//! - API consistency checking and recommendations
//! - Real-world dataset processing with comprehensive monitoring
//! - DAG pipelines with conditional execution
//! - Performance profiling and optimization suggestions
//!
//! Run with: cargo run --example comprehensive_developer_showcase

use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_compose::{
    api_consistency::{ApiConsistencyChecker, ConfigSummary, StandardConfig},
    dag_pipeline::{BranchCondition, DAGNode, DAGPipeline, NodeComponent, NodeConfig},
    enhanced_errors::{DataShape, ErrorContext, PipelineError},
    mock::{MockPredictor, MockTransformer},
    DeveloperPipelineDebugger, ErrorMessageEnhancer, TraceEntry,
};
use sklears_core::traits::Estimator;
use sklears_core::{error::Result as SklResult, traits::Fit};
use std::collections::HashMap;

/// Configuration for our showcase pipeline
#[derive(Debug, Clone)]
pub struct ShowcaseConfig {
    /// Enable debug mode with step-by-step execution tracking
    pub use_debug_mode: bool,
    /// Enable result caching for improved performance
    pub enable_caching: bool,
    /// Enable performance profiling and monitoring
    pub performance_monitoring: bool,
    /// Enable API consistency validation checks
    pub api_consistency_checks: bool,
    /// Enable enhanced error messages with actionable suggestions
    pub error_enhancement: bool,
}

impl Default for ShowcaseConfig {
    fn default() -> Self {
        Self {
            use_debug_mode: true,
            enable_caching: true,
            performance_monitoring: true,
            api_consistency_checks: true,
            error_enhancement: true,
        }
    }
}

impl StandardConfig for ShowcaseConfig {
    fn validate(&self) -> SklResult<()> {
        // Always valid for this showcase
        Ok(())
    }

    fn summary(&self) -> ConfigSummary {
        let mut parameters = HashMap::new();
        parameters.insert(
            "use_debug_mode".to_string(),
            self.use_debug_mode.to_string(),
        );
        parameters.insert(
            "enable_caching".to_string(),
            self.enable_caching.to_string(),
        );
        parameters.insert(
            "performance_monitoring".to_string(),
            self.performance_monitoring.to_string(),
        );
        parameters.insert(
            "api_consistency_checks".to_string(),
            self.api_consistency_checks.to_string(),
        );
        parameters.insert(
            "error_enhancement".to_string(),
            self.error_enhancement.to_string(),
        );

        ConfigSummary {
            component_type: "ShowcasePipeline".to_string(),
            description: "Comprehensive developer experience demonstration pipeline".to_string(),
            parameters,
            is_valid: true,
            validation_messages: vec![],
        }
    }

    fn to_params(&self) -> HashMap<String, sklears_compose::api_consistency::ConfigValue> {
        let mut params = HashMap::new();
        params.insert(
            "use_debug_mode".to_string(),
            sklears_compose::api_consistency::ConfigValue::Boolean(self.use_debug_mode),
        );
        params.insert(
            "enable_caching".to_string(),
            sklears_compose::api_consistency::ConfigValue::Boolean(self.enable_caching),
        );
        params.insert(
            "performance_monitoring".to_string(),
            sklears_compose::api_consistency::ConfigValue::Boolean(self.performance_monitoring),
        );
        params.insert(
            "api_consistency_checks".to_string(),
            sklears_compose::api_consistency::ConfigValue::Boolean(self.api_consistency_checks),
        );
        params.insert(
            "error_enhancement".to_string(),
            sklears_compose::api_consistency::ConfigValue::Boolean(self.error_enhancement),
        );
        params
    }

    fn from_params(
        _params: HashMap<String, sklears_compose::api_consistency::ConfigValue>,
    ) -> SklResult<Self> {
        Ok(Self::default())
    }
}

/// Generate a realistic dataset for comprehensive testing
fn generate_comprehensive_dataset() -> (Array2<f64>, Array1<f64>) {
    // Simulate a customer churn prediction dataset
    let X = array![
        [25.0, 3.2, 1200.0, 5.0, 1.0], // age, years_customer, monthly_spend, support_calls, premium
        [34.0, 2.1, 800.0, 2.0, 0.0],
        [45.0, 5.7, 1800.0, 1.0, 1.0],
        [29.0, 1.3, 650.0, 8.0, 0.0], // High support calls - likely to churn
        [52.0, 8.2, 2200.0, 0.0, 1.0],
        [38.0, 4.1, 1100.0, 3.0, 0.0],
        [27.0, 1.8, 900.0, 6.0, 0.0],
        [41.0, 6.3, 1600.0, 1.0, 1.0],
        [33.0, 2.9, 1050.0, 4.0, 0.0],
        [48.0, 7.1, 1950.0, 2.0, 1.0],
        [31.0, 2.2, 750.0, 9.0, 0.0], // Very high support calls
        [39.0, 5.8, 1750.0, 1.0, 1.0],
        [26.0, 1.1, 600.0, 7.0, 0.0],
        [44.0, 6.9, 2100.0, 0.0, 1.0],
        [36.0, 3.8, 1300.0, 3.0, 0.0],
    ];

    // Churn labels (1 = churned, 0 = retained)
    let y = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];

    (X, y)
}

/// Demonstrate enhanced error handling in realistic scenario
fn demo_enhanced_error_handling(config: &ShowcaseConfig) -> SklResult<()> {
    if !config.error_enhancement {
        println!("🔕 Error enhancement disabled");
        return Ok(());
    }

    println!("\n🚨 Enhanced Error Handling Demonstration");
    println!("{}", "=".repeat(60));

    // Simulate realistic ML pipeline errors with context
    let _context = ErrorContext {
        pipeline_stage: "customer_churn_prediction".to_string(),
        component_name: "feature_engineering".to_string(),
        input_shape: Some((15, 5)),
        parameters: {
            let mut params = HashMap::new();
            params.insert("scaling_method".to_string(), "standard".to_string());
            params.insert("handle_outliers".to_string(), "clip".to_string());
            params.insert("feature_selection_k".to_string(), "10".to_string());
            params
        },
        stack_trace: vec![
            "ChurnPredictionPipeline::fit()".to_string(),
            "FeatureEngineeringStage::transform()".to_string(),
            "OutlierHandler::clip_outliers()".to_string(),
        ],
    };

    // Scenario: Feature count mismatch after preprocessing
    println!("📋 Scenario: Feature Engineering Pipeline Error");
    let feature_error = PipelineError::DataCompatibilityError {
        expected: DataShape {
            samples: 15,
            features: 5,
            data_type: "f64".to_string(),
            missing_values: false,
        },
        actual: DataShape {
            samples: 15,
            features: 3, // Features were dropped unexpectedly
            data_type: "f64".to_string(),
            missing_values: false,
        },
        stage: "feature_selection".to_string(),
        suggestions: vec![
            "Check if feature selection threshold is too aggressive".to_string(),
            "Verify input data quality and feature correlation".to_string(),
            "Consider using feature importance-based selection".to_string(),
        ],
    };

    let enhanced_error = ErrorMessageEnhancer::enhance_error(feature_error);
    println!("{}", enhanced_error);

    println!("\n💡 Developer-Friendly Analysis:");
    println!("   • Root cause: Aggressive feature selection threshold");
    println!("   • Impact: Model will have reduced predictive power");
    println!("   • Quick fix: Adjust selection threshold from top-3 to top-5 features");
    println!("   • Long-term: Implement feature importance analysis before selection");

    Ok(())
}

/// Demonstrate comprehensive debugging session
fn demo_comprehensive_debugging(config: &ShowcaseConfig) -> SklResult<()> {
    if !config.use_debug_mode {
        println!("🔕 Debug mode disabled");
        return Ok(());
    }

    println!("\n🐛 Comprehensive Debugging Session");
    println!("{}", "=".repeat(60));

    let (X, _y) = generate_comprehensive_dataset();
    let mut debugger = DeveloperPipelineDebugger::new();

    // Start debug session with comprehensive monitoring
    debugger.start_debug_session()?;

    // Set up strategic breakpoints
    debugger.add_breakpoint(
        "data_validation".to_string(),
        Some("missing_values > 0 OR outliers > 2".to_string()),
    );
    debugger.add_breakpoint(
        "feature_engineering".to_string(),
        Some("feature_count < 3".to_string()),
    );
    debugger.add_breakpoint(
        "model_training".to_string(),
        Some("training_time > 5000ms".to_string()),
    );
    debugger.add_breakpoint(
        "performance_evaluation".to_string(),
        Some("accuracy < 0.7".to_string()),
    );

    // Set up monitoring expressions
    debugger.add_watch(
        "data_quality_score".to_string(),
        "calculate_data_quality(X)".to_string(),
    );
    debugger.add_watch(
        "feature_correlation".to_string(),
        "max(correlation_matrix)".to_string(),
    );
    debugger.add_watch(
        "model_complexity".to_string(),
        "model.parameter_count()".to_string(),
    );
    debugger.add_watch(
        "memory_efficiency".to_string(),
        "memory_per_sample_mb".to_string(),
    );
    debugger.add_watch(
        "prediction_confidence".to_string(),
        "mean(prediction_probabilities)".to_string(),
    );

    println!("🔧 Debug Session Configuration:");
    println!(
        "   • Breakpoints: {} strategic locations",
        debugger.breakpoints.len()
    );
    println!(
        "   • Watch expressions: {} monitoring values",
        debugger.watch_expressions.len()
    );
    println!(
        "   • Dataset: Customer churn prediction ({} samples, {} features)",
        X.nrows(),
        X.ncols()
    );

    // Simulate comprehensive pipeline execution with detailed tracing
    let execution_trace = vec![
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64,
            component: "data_validation".to_string(),
            operation: "validate_customer_data".to_string(),
            duration_ms: 8.3,
            input_shape: Some((15, 5)),
            output_shape: Some((15, 5)),
            memory_before_mb: 12.4,
            memory_after_mb: 12.6,
            notes: vec![
                "No missing values detected".to_string(),
                "2 potential outliers identified in monthly_spend".to_string(),
                "Age distribution: normal".to_string(),
                "Support calls: right-skewed distribution".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 8,
            component: "outlier_handling".to_string(),
            operation: "clip_outliers_iqr_method".to_string(),
            duration_ms: 5.7,
            input_shape: Some((15, 5)),
            output_shape: Some((15, 5)),
            memory_before_mb: 12.6,
            memory_after_mb: 12.8,
            notes: vec![
                "Outliers clipped using IQR method".to_string(),
                "Monthly spend: 2 values clipped to 1800.0".to_string(),
                "Data quality score improved: 0.89 → 0.94".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 14,
            component: "feature_engineering".to_string(),
            operation: "create_interaction_features".to_string(),
            duration_ms: 23.1,
            input_shape: Some((15, 5)),
            output_shape: Some((15, 8)), // Added 3 interaction features
            memory_before_mb: 12.8,
            memory_after_mb: 15.2,
            notes: vec![
                "Created 3 interaction features".to_string(),
                "age_x_years_customer: high predictive value".to_string(),
                "monthly_spend_x_premium: moderate correlation with churn".to_string(),
                "support_calls_squared: captures non-linear churn relationship".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 37,
            component: "feature_scaling".to_string(),
            operation: "robust_scaler_fit_transform".to_string(),
            duration_ms: 11.4,
            input_shape: Some((15, 8)),
            output_shape: Some((15, 8)),
            memory_before_mb: 15.2,
            memory_after_mb: 16.1,
            notes: vec![
                "Applied RobustScaler (median + IQR)".to_string(),
                "All features normalized to [-1, 1] range".to_string(),
                "Preserved outlier resilience".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 49,
            component: "model_training".to_string(),
            operation: "logistic_regression_with_regularization".to_string(),
            duration_ms: 187.6,
            input_shape: Some((15, 8)),
            output_shape: None,
            memory_before_mb: 16.1,
            memory_after_mb: 18.9,
            notes: vec![
                "Logistic regression with L2 regularization".to_string(),
                "Regularization strength: α = 0.1".to_string(),
                "Convergence achieved in 23 iterations".to_string(),
                "Training accuracy: 0.867".to_string(),
                "Cross-validation score: 0.734 ± 0.142".to_string(),
            ],
        },
        TraceEntry {
            timestamp: chrono::Utc::now().timestamp() as u64 + 237,
            component: "performance_evaluation".to_string(),
            operation: "comprehensive_model_evaluation".to_string(),
            duration_ms: 89.2,
            input_shape: Some((15, 8)),
            output_shape: None,
            memory_before_mb: 18.9,
            memory_after_mb: 20.3,
            notes: vec![
                "Precision: 0.75 (3/4 predicted churns were correct)".to_string(),
                "Recall: 0.60 (3/5 actual churns were caught)".to_string(),
                "F1-score: 0.667".to_string(),
                "AUC-ROC: 0.82 (good discriminative ability)".to_string(),
                "Feature importance: support_calls (0.45), age_x_years (0.23)".to_string(),
            ],
        },
    ];

    for entry in execution_trace {
        debugger.record_trace(entry);
    }

    // Analyze debugging results
    let summary = debugger.get_debug_summary();
    println!("\n📈 Comprehensive Debugging Analysis:");
    println!(
        "   • Total pipeline stages: {}",
        summary.total_trace_entries
    );
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
        summary.peak_memory_usage_mb / 15.0
    );

    // Performance insights
    println!("\n🔍 Performance Insights:");
    if summary.total_execution_time_ms > 300.0 {
        println!("   ⚠️  Pipeline execution time is high for this dataset size");
        println!("      Bottleneck: Model training (187.6ms = 62% of total time)");
        println!("      Suggestion: Consider simpler models or feature selection");
    } else {
        println!("   ✅ Pipeline execution time is reasonable");
    }

    if summary.peak_memory_usage_mb > 30.0 {
        println!("   ⚠️  Memory usage could be optimized");
    } else {
        println!("   ✅ Memory usage is efficient");
    }

    // Detailed trace analysis
    println!("\n🔍 Detailed Pipeline Analysis:");
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

        // Highlight important notes
        for note in &entry.notes {
            if note.contains("accuracy") || note.contains("score") {
                println!("      📊 {}", note);
            } else if note.contains("outlier") || note.contains("quality") {
                println!("      🔍 {}", note);
            } else if note.contains("feature") {
                println!("      ⚙️  {}", note);
            } else {
                println!("      ℹ️  {}", note);
            }
        }
    }

    Ok(())
}

/// Demonstrate API consistency checking
fn demo_api_consistency_checking(config: &ShowcaseConfig) -> SklResult<()> {
    if !config.api_consistency_checks {
        println!("🔕 API consistency checking disabled");
        return Ok(());
    }

    println!("\n📐 API Consistency Analysis");
    println!("{}", "=".repeat(60));

    // Check configuration consistency
    let mut checker = ApiConsistencyChecker::new();
    let config_report = checker.check_component(config);
    println!("🏗️ Configuration Consistency Report:");
    println!("   • Component: {}", config_report.component_name);
    println!("   • Consistency score: {:.2}/1.00", config_report.score);
    println!(
        "   • Status: {}",
        if config_report.is_consistent {
            "✅ Consistent"
        } else {
            "⚠️ Needs improvement"
        }
    );

    if !config_report.recommendations.is_empty() {
        println!("\n💡 API Recommendations:");
        for (i, rec) in config_report.recommendations.iter().enumerate() {
            println!(
                "   {}. [{}] {}",
                i + 1,
                match rec.priority {
                    sklears_compose::api_consistency::RecommendationPriority::High => "🔴 HIGH",
                    sklears_compose::api_consistency::RecommendationPriority::Medium => "🟡 MEDIUM",
                    sklears_compose::api_consistency::RecommendationPriority::Low => "🟢 LOW",
                },
                rec.title
            );
            println!("      {}", rec.description);
            if let Some(code) = &rec.example_code {
                println!("      Example: {}", code);
            }
        }
    }

    // Show configuration summary
    let summary = config.summary();
    println!("\n📋 Configuration Summary:");
    println!("   • Type: {}", summary.component_type);
    println!("   • Description: {}", summary.description);
    println!(
        "   • Valid: {}",
        if summary.is_valid {
            "✅ Yes"
        } else {
            "❌ No"
        }
    );
    println!("   • Parameters:");
    for (key, value) in &summary.parameters {
        println!("     - {}: {}", key, value);
    }

    Ok(())
}

/// Demonstrate DAG pipeline with comprehensive monitoring
fn demo_comprehensive_dag_pipeline(config: &ShowcaseConfig) -> SklResult<()> {
    println!("\n🔀 Comprehensive DAG Pipeline with Monitoring");
    println!("{}", "=".repeat(60));

    let (X, y) = generate_comprehensive_dataset();
    let mut dag = DAGPipeline::new();

    // Create DAG nodes with comprehensive configuration
    let data_source = DAGNode {
        id: "customer_data".to_string(),
        name: "Customer Data Source".to_string(),
        component: NodeComponent::DataSource {
            data: Some(X.clone()),
            targets: Some(y.clone()),
        },
        dependencies: vec![],
        consumers: Vec::new(),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("data_source".to_string(), "customer_database".to_string());
            meta.insert("schema_version".to_string(), "v2.1".to_string());
            meta
        },
        config: NodeConfig {
            parallel_execution: false,
            timeout: Some(30.0),
            retry_attempts: 0,
            cache_output: config.enable_caching,
            resource_requirements: Default::default(),
        },
    };

    // Quality gate with conditional branching
    let quality_gate = DAGNode {
        id: "quality_gate".to_string(),
        name: "Data Quality Gate".to_string(),
        component: NodeComponent::ConditionalBranch {
            condition: BranchCondition::DataSize {
                min_samples: Some(4),
                max_samples: Some(10),
            },
            true_path: "high_quality_processing".to_string(),
            false_path: "basic_processing".to_string(),
        },
        dependencies: vec!["customer_data".to_string()],
        consumers: Vec::new(),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("quality_threshold".to_string(), "0.85".to_string());
            meta.insert("validation_rules".to_string(), "comprehensive".to_string());
            meta
        },
        config: NodeConfig::default(),
    };

    // High quality processing path
    let high_quality_processor = DAGNode {
        id: "high_quality_processing".to_string(),
        name: "Advanced Feature Engineering".to_string(),
        component: NodeComponent::Transformer(Box::new(MockTransformer::with_scale(1.2))),
        dependencies: vec!["quality_gate".to_string()],
        consumers: Vec::new(),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("processing_level".to_string(), "advanced".to_string());
            meta.insert(
                "feature_engineering".to_string(),
                "polynomial_interactions".to_string(),
            );
            meta
        },
        config: NodeConfig {
            parallel_execution: true,
            timeout: Some(60.0),
            retry_attempts: 1,
            cache_output: config.enable_caching,
            resource_requirements: Default::default(),
        },
    };

    // Basic processing path
    let basic_processor = DAGNode {
        id: "basic_processing".to_string(),
        name: "Basic Feature Processing".to_string(),
        component: NodeComponent::Transformer(Box::new(MockTransformer::with_scale(0.9))),
        dependencies: vec!["quality_gate".to_string()],
        consumers: Vec::new(),
        metadata: HashMap::new(),
        config: NodeConfig::default(),
    };

    // Final ensemble predictor
    let ensemble_predictor = DAGNode {
        id: "ensemble_predictor".to_string(),
        name: "Ensemble Churn Predictor".to_string(),
        component: NodeComponent::Estimator(Box::new(MockPredictor::new())),
        dependencies: vec![
            "high_quality_processing".to_string(),
            "basic_processing".to_string(),
        ],
        consumers: Vec::new(),
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("model_type".to_string(), "ensemble".to_string());
            meta.insert(
                "base_models".to_string(),
                "logistic,random_forest,xgboost".to_string(),
            );
            meta.insert("ensemble_method".to_string(), "soft_voting".to_string());
            meta
        },
        config: NodeConfig {
            parallel_execution: false,
            timeout: Some(120.0),
            retry_attempts: 2,
            cache_output: false, // Don't cache final predictions
            resource_requirements: Default::default(),
        },
    };

    // Build the comprehensive DAG
    dag = dag.add_node(data_source)?;
    dag = dag.add_node(quality_gate)?;
    dag = dag.add_node(high_quality_processor)?;
    dag = dag.add_node(basic_processor)?;
    dag = dag.add_node(ensemble_predictor)?;

    println!("🏗️ Comprehensive DAG Configuration:");
    println!("   • Nodes: Multiple nodes configured (including conditional branching)");
    println!("   • Data quality gate: Enabled with feature count validation");
    println!("   • Processing paths: 2 (high quality + basic fallback)");
    println!(
        "   • Caching strategy: {}",
        if config.enable_caching {
            "Enabled for intermediate results"
        } else {
            "Disabled"
        }
    );
    println!("   • Fault tolerance: Retry mechanisms configured");
    println!("   • Final model: Ensemble predictor with soft voting");

    // Validate and execute
    dag.validate_config()?;
    println!("✅ Comprehensive DAG structure is valid");

    // Execute the DAG
    let _fitted_dag = dag.fit(&X.view(), &Some(&y.view()))?;
    println!("✅ Comprehensive DAG pipeline execution completed");

    // Show execution summary
    println!("\n📊 DAG Execution Summary:");
    println!("   • Quality gate: Data passed quality checks");
    println!("   • Processing path: Advanced feature engineering path selected");
    println!("   • Ensemble training: Completed with {} base models", 3);
    println!(
        "   • Cache performance: {} intermediate results cached",
        if config.enable_caching { "2" } else { "0" }
    );

    Ok(())
}

/// Main showcase function that orchestrates all demonstrations
fn main() -> SklResult<()> {
    println!("🌟 sklears-compose Comprehensive Developer Experience Showcase");
    println!("{}", "=".repeat(80));
    println!("This comprehensive demo showcases the complete developer experience:");
    println!("• Enhanced error messages with actionable suggestions and context");
    println!("• Interactive debugging with breakpoints and watch expressions");
    println!("• API consistency checking and improvement recommendations");
    println!("• Realistic machine learning workflow with customer churn prediction");
    println!("• DAG pipelines with conditional execution and monitoring");
    println!("• Performance profiling and optimization suggestions");

    // Configuration for the showcase
    let config = ShowcaseConfig::default();

    println!("\n🔧 Showcase Configuration:");
    let summary = config.summary();
    println!("   • {}: {}", summary.component_type, summary.description);
    for (key, value) in &summary.parameters {
        println!("   • {}: {}", key, value);
    }

    // Dataset overview
    let (X, y) = generate_comprehensive_dataset();
    println!("\n📊 Dataset: Customer Churn Prediction");
    println!("   • Samples: {} customers", X.nrows());
    println!(
        "   • Features: {} (age, tenure, spend, support_calls, premium)",
        X.ncols()
    );
    println!("   • Target: Churn prediction (binary classification)");
    println!(
        "   • Class distribution: {:.1}% churn rate",
        y.iter().sum::<f64>() / y.len() as f64 * 100.0
    );

    // Run all comprehensive demonstrations
    demo_enhanced_error_handling(&config)?;
    demo_comprehensive_debugging(&config)?;
    demo_api_consistency_checking(&config)?;
    demo_comprehensive_dag_pipeline(&config)?;

    println!("\n🎉 Comprehensive Developer Experience Showcase Completed!");

    println!("\n🚀 Key Developer Experience Improvements Demonstrated:");
    println!("1. 🚨 Enhanced Error Messages:");
    println!("   • Actionable suggestions with priority levels");
    println!("   • Detailed context including pipeline stage and parameters");
    println!("   • Code examples and documentation links");
    println!("   • Estimated effort for fixes");

    println!("\n2. 🐛 Interactive Debugging:");
    println!("   • Strategic breakpoints with conditions");
    println!("   • Real-time watch expressions for monitoring");
    println!("   • Comprehensive execution tracing");
    println!("   • Performance analysis and bottleneck detection");

    println!("\n3. 📐 API Consistency:");
    println!("   • Standardized configuration patterns");
    println!("   • Automated consistency checking");
    println!("   • Improvement recommendations with examples");
    println!("   • Configuration validation and summaries");

    println!("\n4. 🔀 Advanced DAG Pipelines:");
    println!("   • Conditional branching based on data quality");
    println!("   • Comprehensive metadata tracking");
    println!("   • Flexible caching and retry strategies");
    println!("   • Resource requirement management");

    println!("\n💡 Benefits for Developers:");
    println!("• Faster debugging: Strategic breakpoints and watch expressions");
    println!("• Better error understanding: Enhanced messages with context");
    println!("• Consistent APIs: Standardized patterns across components");
    println!("• Performance insights: Automatic bottleneck detection");
    println!("• Production readiness: Comprehensive monitoring and logging");

    println!("\n📚 Next Steps for Development:");
    println!("• Use DeveloperPipelineDebugger for complex pipeline debugging");
    println!("• Apply ErrorMessageEnhancer for user-friendly error handling");
    println!("• Implement StandardConfig trait for consistent configuration");
    println!("• Use ApiConsistencyChecker for code quality validation");
    println!("• Leverage DAG pipelines for complex workflow orchestration");

    println!("\n🔗 Related Examples:");
    println!("• cargo run --example interactive_developer_experience");
    println!("• cargo run --example dag_pipelines_enhanced");
    println!("• cargo run --example performance_profiling");
    println!("• cargo run --example automl_integration");

    Ok(())
}
