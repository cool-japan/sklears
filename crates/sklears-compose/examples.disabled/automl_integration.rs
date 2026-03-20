//! AutoML Integration Examples
//!
//! This example demonstrates automated machine learning capabilities including
//! neural architecture search, hyperparameter optimization, automated feature
//! engineering, and multi-objective optimization.
//!
//! Run with: cargo run --example automl_integration

use ndarray::{array, Array1, Array2};
use sklears_compose::{
    automl::{
        AutoMLConfig, AutoMLOptimizer, NASStrategy, NeuralArchitectureSearch, OptimizationHistory,
        OptimizationMetric, ParameterRange, ParameterValue, SearchStrategy, TrialResult,
        TrialStatus,
    },
    mock::{MockPredictor, MockTransformer},
    Pipeline, PipelineBuilder,
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict},
};
use std::collections::HashMap;

/// Generate dataset for AutoML experiments
fn generate_automl_dataset() -> (Array2<f64>, Array1<f64>) {
    let X = array![
        [1.0, 2.0, 3.0, 0.5, 1.5],
        [2.0, 3.0, 4.0, 1.0, 2.0],
        [3.0, 4.0, 5.0, 1.5, 2.5],
        [4.0, 5.0, 6.0, 2.0, 3.0],
        [5.0, 6.0, 7.0, 2.5, 3.5],
        [6.0, 7.0, 8.0, 3.0, 4.0],
        [7.0, 8.0, 9.0, 3.5, 4.5],
        [8.0, 9.0, 10.0, 4.0, 5.0],
        [9.0, 10.0, 11.0, 4.5, 5.5],
        [10.0, 11.0, 12.0, 5.0, 6.0]
    ];

    let y = array![15.0, 30.0, 50.0, 75.0, 105.0, 140.0, 180.0, 225.0, 275.0, 330.0];
    (X, y)
}

/// Demonstrate hyperparameter optimization
fn demo_hyperparameter_optimization() -> SklResult<()> {
    println!("\n🎛️ Hyperparameter Optimization Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_automl_dataset();

    // Define parameter search space
    let mut search_space = HashMap::new();
    search_space.insert(
        "learning_rate".to_string(),
        ParameterRange::Float {
            min: 0.001,
            max: 0.3,
            log_scale: true,
        },
    );
    search_space.insert(
        "n_estimators".to_string(),
        ParameterRange::Integer { min: 10, max: 200 },
    );
    search_space.insert(
        "max_depth".to_string(),
        ParameterRange::Integer { min: 1, max: 10 },
    );
    search_space.insert(
        "regularization".to_string(),
        ParameterRange::Float {
            min: 0.0,
            max: 1.0,
            log_scale: false,
        },
    );

    // Configure AutoML optimizer
    let automl_config = AutoMLConfig::builder()
        .search_strategy(SearchStrategy::BayesianOptimization { n_trials: 50 })
        .optimization_metric(OptimizationMetric::RMSE)
        .cv_folds(5)
        .time_budget_minutes(Some(10))
        .early_stopping_rounds(Some(10))
        .build();

    let mut automl_optimizer = AutoMLOptimizer::new(automl_config);

    println!("🏗️ Created AutoML optimizer with Bayesian optimization");
    println!("🎯 Optimizing {} hyperparameters", search_space.len());
    println!("⏰ Time budget: 10 minutes with early stopping");

    // Run optimization
    let optimization_result = automl_optimizer.optimize(
        &X.view(),
        &y.view(),
        search_space,
        Box::new(|| Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>),
    )?;

    println!("✅ Optimization completed!");
    println!("🏆 Best RMSE: {:.4}", optimization_result.best_score());
    println!("🔧 Best parameters:");

    for (param_name, param_value) in optimization_result.best_params() {
        match param_value {
            ParameterValue::Float(val) => println!("   • {}: {:.4}", param_name, val),
            ParameterValue::Integer(val) => println!("   • {}: {}", param_name, val),
            ParameterValue::String(val) => println!("   • {}: {}", param_name, val),
            ParameterValue::Boolean(val) => println!("   • {}: {}", param_name, val),
        }
    }

    // Show optimization history
    if let Some(history) = optimization_result.history() {
        println!("📈 Optimization Progress:");
        println!("   • Total trials: {}", history.total_trials());
        println!("   • Successful trials: {}", history.successful_trials());
        println!(
            "   • Best trial found at iteration: {}",
            history.best_trial_iteration()
        );
        println!("   • Convergence achieved: {}", history.converged());
    }

    Ok(())
}

/// Demonstrate neural architecture search
fn demo_neural_architecture_search() -> SklResult<()> {
    println!("\n🧠 Neural Architecture Search Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_automl_dataset();

    // Configure NAS with different search strategies
    let nas_config = sklears_compose::automl::NeuralSearchSpace::builder()
        .layer_types(vec![
            sklears_compose::automl::LayerType::Dense,
            sklears_compose::automl::LayerType::Dropout,
            sklears_compose::automl::LayerType::BatchNorm,
        ])
        .activation_functions(vec![
            sklears_compose::automl::ActivationFunction::ReLU,
            sklears_compose::automl::ActivationFunction::Tanh,
            sklears_compose::automl::ActivationFunction::Sigmoid,
        ])
        .hidden_size_range(8, 128)
        .depth_range(1, 6)
        .dropout_range(0.0, 0.5)
        .build();

    // Create NAS optimizer
    let nas_optimizer = NeuralArchitectureSearch::builder()
        .search_space(nas_config)
        .strategy(NASStrategy::RandomSearch { n_trials: 30 })
        .evaluation_metric(OptimizationMetric::RMSE)
        .early_stopping_patience(5)
        .build();

    println!("🏗️ Created NAS with random search strategy");
    println!("🔍 Searching {} layer types", 3);
    println!("📐 Architecture depth range: 1-6 layers");
    println!("🎯 Hidden size range: 8-128 units");

    // Run architecture search
    let nas_result = nas_optimizer.search(&X.view(), &y.view())?;

    println!("✅ Architecture search completed!");

    if let Some(best_arch) = nas_result.best_architecture() {
        println!("🏆 Best Architecture:");
        println!("   • Layers: {}", best_arch.num_layers());
        println!("   • Total parameters: {}", best_arch.total_parameters());
        println!("   • Architecture score: {:.4}", best_arch.score());
        println!("   • Architecture description:");

        for (i, layer) in best_arch.layers().iter().enumerate() {
            println!("     Layer {}: {:?}", i + 1, layer);
        }
    }

    // Show search statistics
    if let Some(search_stats) = nas_result.search_statistics() {
        println!("📊 Search Statistics:");
        println!(
            "   • Architectures evaluated: {}",
            search_stats.architectures_evaluated()
        );
        println!(
            "   • Average parameters per architecture: {:.0}",
            search_stats.avg_parameters()
        );
        println!(
            "   • Best architecture percentile: {:.1}%",
            search_stats.best_percentile()
        );
        println!(
            "   • Search efficiency: {:.2}",
            search_stats.search_efficiency()
        );
    }

    Ok(())
}

/// Demonstrate automated feature engineering
fn demo_automated_feature_engineering() -> SklResult<()> {
    println!("\n⚙️ Automated Feature Engineering Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_automl_dataset();

    // Configure automated feature engineering
    let feature_config = sklears_compose::automl::FeatureEngineeringConfig::builder()
        .polynomial_features(true)
        .interaction_features(true)
        .statistical_features(true)
        .max_polynomial_degree(3)
        .max_interaction_depth(2)
        .feature_selection_k_best(10)
        .build();

    let auto_feature_engineer =
        sklears_compose::automl::AutomatedFeatureEngineer::new(feature_config);

    println!("🏗️ Configured automated feature engineering");
    println!("🔢 Polynomial features up to degree 3");
    println!("🔗 Interaction features up to depth 2");
    println!("📊 Statistical features enabled");
    println!("🎯 Selecting top 10 features");

    // Generate and select features
    let (engineered_features, feature_info) =
        auto_feature_engineer.fit_transform(&X.view(), &y.view())?;

    println!("✅ Feature engineering completed!");
    println!("📈 Original features: {}", X.ncols());
    println!("✨ Generated features: {}", feature_info.total_generated());
    println!("🎯 Selected features: {}", engineered_features.ncols());
    println!(
        "📊 Feature expansion ratio: {:.2}x",
        engineered_features.ncols() as f64 / X.ncols() as f64
    );

    // Show feature importance scores
    if let Some(importance_scores) = feature_info.importance_scores() {
        println!("🔍 Top 5 Most Important Features:");
        let mut scored_features: Vec<_> = importance_scores.iter().enumerate().collect();
        scored_features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, (feature_idx, score)) in scored_features.iter().take(5).enumerate() {
            println!("   {}. Feature {}: {:.4}", i + 1, feature_idx, score);
        }
    }

    // Display feature types
    println!("🏷️ Feature Types Generated:");
    if let Some(feature_types) = feature_info.feature_types() {
        for (feature_type, count) in feature_types {
            println!("   • {}: {} features", feature_type, count);
        }
    }

    Ok(())
}

/// Demonstrate multi-objective optimization
fn demo_multi_objective_optimization() -> SklResult<()> {
    println!("\n🎯 Multi-Objective Optimization Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_automl_dataset();

    // Configure multi-objective optimization
    let objectives = vec![
        OptimizationMetric::RMSE,         // Minimize prediction error
        OptimizationMetric::ModelSize,    // Minimize model complexity
        OptimizationMetric::TrainingTime, // Minimize training time
    ];

    let multi_obj_config = AutoMLConfig::builder()
        .search_strategy(SearchStrategy::GeneticAlgorithm {
            population_size: 50,
            n_generations: 20,
        })
        .multi_objective_metrics(objectives.clone())
        .cv_folds(3)
        .build();

    let mut multi_obj_optimizer = AutoMLOptimizer::new(multi_obj_config);

    println!("🏗️ Created multi-objective optimizer");
    println!("🎯 Optimizing {} objectives:", objectives.len());
    for (i, objective) in objectives.iter().enumerate() {
        println!("   {}. {:?}", i + 1, objective);
    }
    println!("🧬 Using genetic algorithm with population size 50");

    // Define parameter space for multi-objective optimization
    let mut multi_obj_space = HashMap::new();
    multi_obj_space.insert(
        "model_complexity".to_string(),
        ParameterRange::Integer { min: 1, max: 5 },
    );
    multi_obj_space.insert(
        "regularization_strength".to_string(),
        ParameterRange::Float {
            min: 0.01,
            max: 10.0,
            log_scale: true,
        },
    );

    // Run multi-objective optimization
    let pareto_results = multi_obj_optimizer.optimize_multi_objective(
        &X.view(),
        &y.view(),
        multi_obj_space,
        Box::new(|| Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>),
    )?;

    println!("✅ Multi-objective optimization completed!");
    println!(
        "🏆 Pareto optimal solutions found: {}",
        pareto_results.pareto_front().len()
    );

    // Display Pareto front
    println!("📊 Pareto Front (Top 3 Solutions):");
    for (i, solution) in pareto_results.pareto_front().iter().take(3).enumerate() {
        println!("   Solution {}:", i + 1);
        println!("     • RMSE: {:.4}", solution.objectives()[0]);
        println!("     • Model Size: {:.2}KB", solution.objectives()[1]);
        println!("     • Training Time: {:.2}s", solution.objectives()[2]);
    }

    // Show optimization statistics
    if let Some(multi_obj_stats) = pareto_results.optimization_stats() {
        println!("📈 Multi-Objective Statistics:");
        println!(
            "   • Total evaluations: {}",
            multi_obj_stats.total_evaluations()
        );
        println!(
            "   • Pareto front coverage: {:.1}%",
            multi_obj_stats.pareto_coverage() * 100.0
        );
        println!(
            "   • Hypervolume indicator: {:.4}",
            multi_obj_stats.hypervolume()
        );
        println!(
            "   • Solution diversity: {:.3}",
            multi_obj_stats.diversity_metric()
        );
    }

    Ok(())
}

/// Demonstrate automated pipeline optimization
fn demo_automated_pipeline_optimization() -> SklResult<()> {
    println!("\n🏭 Automated Pipeline Optimization Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_automl_dataset();

    // Define pipeline template with tunable components
    let pipeline_template = sklears_compose::automl::PipelineTemplate::builder()
        .preprocessing_options(vec![
            sklears_compose::automl::PreprocessingOption::StandardScaler,
            sklears_compose::automl::PreprocessingOption::MinMaxScaler,
            sklears_compose::automl::PreprocessingOption::RobustScaler,
        ])
        .feature_selection_options(vec![
            sklears_compose::automl::FeatureSelectionOption::SelectKBest(10),
            sklears_compose::automl::FeatureSelectionOption::SelectPercentile(50),
            sklears_compose::automl::FeatureSelectionOption::VarianceThreshold(0.01),
        ])
        .model_options(vec![
            sklears_compose::automl::ModelOption::LinearRegression,
            sklears_compose::automl::ModelOption::RandomForest,
            sklears_compose::automl::ModelOption::GradientBoosting,
        ])
        .build();

    // Create pipeline optimizer
    let pipeline_optimizer = sklears_compose::automl::AutomatedPipelineOptimizer::builder()
        .template(pipeline_template)
        .search_strategy(SearchStrategy::RandomSearch { n_trials: 25 })
        .evaluation_metric(OptimizationMetric::RMSE)
        .cv_folds(5)
        .build();

    println!("🏗️ Created automated pipeline optimizer");
    println!("🔧 Preprocessing options: 3");
    println!("🎯 Feature selection options: 3");
    println!("🤖 Model options: 3");
    println!("🔍 Total pipeline configurations: 27");

    // Optimize pipeline structure and hyperparameters
    let pipeline_result = pipeline_optimizer.optimize(&X.view(), &y.view())?;

    println!("✅ Pipeline optimization completed!");

    if let Some(best_pipeline) = pipeline_result.best_pipeline() {
        println!("🏆 Best Pipeline Configuration:");
        println!("   • Preprocessor: {:?}", best_pipeline.preprocessor());
        println!(
            "   • Feature Selector: {:?}",
            best_pipeline.feature_selector()
        );
        println!("   • Model: {:?}", best_pipeline.model());
        println!("   • Pipeline Score: {:.4}", best_pipeline.score());
        println!(
            "   • Cross-validation std: {:.4}",
            best_pipeline.score_std()
        );
    }

    // Show pipeline search statistics
    if let Some(search_stats) = pipeline_result.search_stats() {
        println!("📊 Pipeline Search Statistics:");
        println!(
            "   • Configurations evaluated: {}",
            search_stats.configurations_evaluated()
        );
        println!(
            "   • Best configuration found at trial: {}",
            search_stats.best_trial_number()
        );
        println!(
            "   • Average score improvement: {:.1}%",
            search_stats.avg_improvement() * 100.0
        );
    }

    Ok(())
}

/// Demonstrate advanced AutoML features
fn demo_advanced_automl() -> SklResult<()> {
    println!("\n⚡ Advanced AutoML Features Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_automl_dataset();

    // Configure advanced AutoML with ensemble and meta-learning
    let advanced_config = AutoMLConfig::builder()
        .search_strategy(SearchStrategy::HybridOptimization {
            initial_random_trials: 10,
            bayesian_trials: 30,
            genetic_refinement_generations: 5,
        })
        .optimization_metric(OptimizationMetric::RMSE)
        .enable_ensemble_selection(true)
        .enable_meta_learning(true)
        .warm_start_from_metalearning(true)
        .auto_feature_engineering(true)
        .cv_folds(5)
        .build();

    let mut advanced_optimizer = AutoMLOptimizer::new(advanced_config);

    println!("🏗️ Created advanced AutoML system");
    println!("🧠 Meta-learning enabled for warm start");
    println!("⚡ Hybrid optimization strategy");
    println!("🎭 Automatic ensemble selection");
    println!("⚙️ Automated feature engineering");

    // Define comprehensive search space
    let mut advanced_space = HashMap::new();

    // Model selection parameters
    advanced_space.insert(
        "model_family".to_string(),
        ParameterRange::Categorical(vec![
            "linear".to_string(),
            "tree".to_string(),
            "ensemble".to_string(),
            "neural".to_string(),
        ]),
    );

    // Run advanced optimization
    let advanced_result =
        advanced_optimizer.optimize_advanced(&X.view(), &y.view(), advanced_space)?;

    println!("✅ Advanced AutoML completed!");

    if let Some(final_model) = advanced_result.final_model() {
        println!("🏆 Final Model:");
        println!("   • Type: {:?}", final_model.model_type());
        println!(
            "   • Performance: {:.4} RMSE",
            final_model.performance_score()
        );
        println!("   • Is ensemble: {}", final_model.is_ensemble());
        println!("   • Features used: {}", final_model.n_features());

        if final_model.is_ensemble() {
            println!("   • Ensemble size: {}", final_model.ensemble_size());
            println!(
                "   • Ensemble diversity: {:.3}",
                final_model.ensemble_diversity()
            );
        }
    }

    // Show comprehensive statistics
    println!("📈 Comprehensive AutoML Statistics:");
    println!(
        "   • Total optimization time: {:.1}s",
        advanced_result.total_time_seconds()
    );
    println!(
        "   • Models evaluated: {}",
        advanced_result.models_evaluated()
    );
    println!(
        "   • Feature engineering iterations: {}",
        advanced_result.feature_engineering_iterations()
    );
    println!(
        "   • Meta-learning speedup: {:.1}x",
        advanced_result.metalearning_speedup()
    );
    println!(
        "   • Final model complexity: {:.2}",
        advanced_result.model_complexity()
    );

    Ok(())
}

fn main() -> SklResult<()> {
    println!("🚀 sklears-compose AutoML Integration Examples");
    println!("=".repeat(60));
    println!("This example demonstrates automated machine learning:");
    println!("• Hyperparameter optimization");
    println!("• Neural architecture search");
    println!("• Automated feature engineering");
    println!("• Multi-objective optimization");
    println!("• Automated pipeline optimization");

    // Run all AutoML demonstrations
    demo_hyperparameter_optimization()?;
    demo_neural_architecture_search()?;
    demo_automated_feature_engineering()?;
    demo_multi_objective_optimization()?;
    demo_automated_pipeline_optimization()?;
    demo_advanced_automl()?;

    println!("\n🎉 All AutoML examples completed successfully!");

    println!("\n💡 AutoML Best Practices:");
    println!("• Use Bayesian optimization for small search spaces");
    println!("• Apply genetic algorithms for multi-objective problems");
    println!("• Enable meta-learning for faster convergence");
    println!("• Consider ensemble methods for better performance");
    println!("• Balance automation with domain expertise");

    println!("\n🔬 Key Benefits:");
    println!("• Reduces manual hyperparameter tuning");
    println!("• Discovers optimal neural architectures");
    println!("• Automates feature engineering processes");
    println!("• Handles multi-objective trade-offs");
    println!("• Scales to complex pipeline optimization");

    println!("\n📚 Next Steps:");
    println!("• Try: cargo run --example time_series_pipelines");
    println!("• Try: cargo run --example computer_vision_pipelines");
    println!("• Try: cargo run --example streaming_pipelines");

    Ok(())
}
