//! Ensemble Methods Examples
//!
//! This example demonstrates advanced ensemble techniques including voting,
//! stacking, dynamic selection, hierarchical composition, and model fusion.
//!
//! Run with: cargo run --example ensemble_methods

use ndarray::{array, Array1, Array2};
use sklears_compose::{
    boosting::{AdaBoostClassifier, GradientBoostingRegressor, LossFunction},
    ensemble::{
        CompetenceEstimation, DynamicEnsembleSelector, FusionStrategy, HierarchicalComposition,
        HierarchicalStrategy, ModelFusion, SelectionStrategy, VotingClassifier, VotingRegressor,
    },
    mock::{MockPredictor, MockTransformer},
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict},
};
use std::collections::HashMap;

/// Generate classification dataset
fn generate_classification_data() -> (Array2<f64>, Array1<f64>) {
    let X = array![
        [1.0, 2.0, 0.5, 1.5],
        [2.0, 3.0, 1.0, 2.0],
        [3.0, 1.0, 1.5, 0.8],
        [1.5, 4.0, 2.0, 1.2],
        [4.0, 2.5, 0.8, 2.5],
        [2.8, 3.5, 1.8, 1.8],
        [3.5, 1.5, 2.2, 1.0],
        [1.2, 3.8, 1.6, 2.2]
    ];

    // Binary classification targets
    let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    (X, y)
}

/// Generate regression dataset
fn generate_regression_data() -> (Array2<f64>, Array1<f64>) {
    let X = array![
        [1.0, 2.0, 0.5],
        [2.0, 3.0, 1.0],
        [3.0, 4.0, 1.5],
        [4.0, 5.0, 2.0],
        [5.0, 6.0, 2.5],
        [6.0, 7.0, 3.0],
        [7.0, 8.0, 3.5],
        [8.0, 9.0, 4.0]
    ];

    let y = array![10.5, 25.2, 45.8, 70.1, 100.3, 135.7, 175.4, 220.9];
    (X, y)
}

/// Demonstrate voting classifier ensemble
fn demo_voting_classifier() -> SklResult<()> {
    println!("\n🗳️ Voting Classifier Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_classification_data();

    // Create base models with different characteristics
    let models = vec![
        (
            "svm".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "random_forest".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "neural_net".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
    ];

    // Create voting classifier with hard voting
    let voting_clf = VotingClassifier::builder()
        .estimators(models.clone())
        .voting_method(sklears_compose::ensemble::VotingMethod::Hard)
        .weights(Some(vec![1.0, 1.5, 1.2])) // Weight neural_net less, random_forest more
        .build();

    println!(
        "🏗️ Created voting classifier with {} base models",
        models.len()
    );
    println!("📊 Using hard voting with custom weights");

    // Fit and predict
    let fitted_voting = voting_clf.fit(&X.view(), &Some(&y.view()))?;
    let voting_predictions = fitted_voting.predict(&X.view())?;

    println!(
        "🔮 Voting predictions: {:?}",
        voting_predictions.slice(ndarray::s![0..4])
    );
    println!("🎯 Actual labels: {:?}", y.slice(ndarray::s![0..4]));

    // Calculate accuracy
    let accuracy = y
        .iter()
        .zip(voting_predictions.iter())
        .filter(|(actual, pred)| (actual - pred).abs() < 0.5)
        .count() as f64
        / y.len() as f64;

    println!("📈 Voting classifier accuracy: {:.1}%", accuracy * 100.0);

    Ok(())
}

/// Demonstrate voting regressor ensemble
fn demo_voting_regressor() -> SklResult<()> {
    println!("\n📊 Voting Regressor Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_regression_data();

    // Create diverse regression models
    let regressors = vec![
        (
            "linear".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "tree".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "knn".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
    ];

    // Create voting regressor with weighted averaging
    let voting_reg = VotingRegressor::builder()
        .estimators(regressors.clone())
        .weights(Some(vec![2.0, 1.0, 1.5])) // Higher weight for linear model
        .build();

    println!(
        "🏗️ Created voting regressor with {} base models",
        regressors.len()
    );
    println!("⚖️ Using weighted averaging");

    // Fit and predict
    let fitted_voting_reg = voting_reg.fit(&X.view(), &Some(&y.view()))?;
    let reg_predictions = fitted_voting_reg.predict(&X.view())?;

    println!(
        "🔮 Regression predictions: {:?}",
        reg_predictions.slice(ndarray::s![0..4])
    );
    println!("🎯 Actual values: {:?}", y.slice(ndarray::s![0..4]));

    // Calculate RMSE
    let rmse = (y
        .iter()
        .zip(reg_predictions.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>()
        / y.len() as f64)
        .sqrt();

    println!("📊 Root Mean Square Error: {:.2}", rmse);

    Ok(())
}

/// Demonstrate dynamic ensemble selector
fn demo_dynamic_ensemble() -> SklResult<()> {
    println!("\n🎯 Dynamic Ensemble Selector Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_classification_data();

    // Create base models
    let base_models = vec![
        Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
    ];

    // Create dynamic ensemble selector
    let dynamic_selector = DynamicEnsembleSelector::builder()
        .base_estimators(base_models)
        .selection_strategy(SelectionStrategy::KBest { k: 2 })
        .competence_estimation(CompetenceEstimation::LocalAccuracy { k: 3 })
        .diversity_threshold(0.3)
        .performance_threshold(0.7)
        .build();

    println!("🏗️ Created dynamic selector with K-best strategy (k=2)");
    println!("🧠 Using local accuracy competence estimation");

    // Fit and predict
    let fitted_dynamic = dynamic_selector.fit(&X.view(), &Some(&y.view()))?;
    let dynamic_predictions = fitted_dynamic.predict(&X.view())?;

    println!(
        "🔮 Dynamic predictions: {:?}",
        dynamic_predictions.slice(ndarray::s![0..4])
    );

    // Show model selection statistics
    if let Some(stats) = fitted_dynamic.selection_stats() {
        println!("📊 Model Selection Statistics:");
        println!(
            "   • Average models selected: {:.1}",
            stats.avg_models_selected()
        );
        println!(
            "   • Selection diversity: {:.2}",
            stats.selection_diversity()
        );
        println!(
            "   • Competence correlation: {:.2}",
            stats.competence_correlation()
        );
    }

    Ok(())
}

/// Demonstrate hierarchical model composition
fn demo_hierarchical_composition() -> SklResult<()> {
    println!("\n🏰 Hierarchical Composition Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_regression_data();

    // Create hierarchical composition with multiple levels
    let hierarchical = HierarchicalComposition::builder()
        .strategy(HierarchicalStrategy::BinaryTree)
        .base_models(vec![
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ])
        .meta_learner(Box::new(MockPredictor::new()))
        .layer_weights(vec![1.0, 0.8, 0.6]) // Decreasing weights for higher layers
        .build();

    println!("🏗️ Created hierarchical composition with binary tree structure");
    println!("🌳 Using {} base models with meta-learner", 4);

    // Fit and predict
    let fitted_hierarchical = hierarchical.fit(&X.view(), &Some(&y.view()))?;
    let hierarchical_predictions = fitted_hierarchical.predict(&X.view())?;

    println!(
        "🔮 Hierarchical predictions: {:?}",
        hierarchical_predictions.slice(ndarray::s![0..4])
    );

    // Display hierarchy information
    if let Some(hierarchy_info) = fitted_hierarchical.hierarchy_info() {
        println!("🏛️ Hierarchy Structure:");
        println!("   • Total layers: {}", hierarchy_info.num_layers());
        println!(
            "   • Models per layer: {:?}",
            hierarchy_info.models_per_layer()
        );
        println!("   • Tree depth: {}", hierarchy_info.tree_depth());
    }

    Ok(())
}

/// Demonstrate model fusion strategies
fn demo_model_fusion() -> SklResult<()> {
    println!("\n🔬 Model Fusion Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_regression_data();

    // Create model fusion with different strategies
    let model_fusion = ModelFusion::builder()
        .models(vec![
            (
                "expert1".to_string(),
                Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
            ),
            (
                "expert2".to_string(),
                Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
            ),
            (
                "expert3".to_string(),
                Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
            ),
        ])
        .fusion_strategy(FusionStrategy::NeuralFusion)
        .fusion_network_hidden_size(16)
        .adaptive_weights(true)
        .build();

    println!("🏗️ Created model fusion with neural fusion strategy");
    println!("🧠 Using adaptive weights with 16-unit hidden layer");

    // Fit and predict
    let fitted_fusion = model_fusion.fit(&X.view(), &Some(&y.view()))?;
    let fusion_predictions = fitted_fusion.predict(&X.view())?;

    println!(
        "🔮 Fusion predictions: {:?}",
        fusion_predictions.slice(ndarray::s![0..4])
    );

    // Show fusion statistics
    if let Some(fusion_stats) = fitted_fusion.fusion_stats() {
        println!("📊 Fusion Statistics:");
        println!(
            "   • Fusion network parameters: {}",
            fusion_stats.num_parameters()
        );
        println!(
            "   • Model contribution weights: {:?}",
            fusion_stats
                .model_weights()
                .iter()
                .map(|w| format!("{:.2}", w))
                .collect::<Vec<_>>()
        );
        println!("   • Fusion loss: {:.4}", fusion_stats.fusion_loss());
    }

    Ok(())
}

/// Demonstrate boosting ensemble
fn demo_boosting_ensemble() -> SklResult<()> {
    println!("\n🚀 Boosting Ensemble Demo");
    println!("{}", "=".repeat(50));

    let (X_class, y_class) = generate_classification_data();
    let (X_reg, y_reg) = generate_regression_data();

    // AdaBoost for classification
    println!("🎯 AdaBoost Classification:");
    let ada_boost = AdaBoostClassifier::new(
        Box::new(MockPredictor::new()),
        50,  // n_estimators
        1.0, // learning_rate
    );

    let fitted_ada = ada_boost.fit(&X_class.view(), &Some(&y_class.view()))?;
    let ada_predictions = fitted_ada.predict(&X_class.view())?;

    println!(
        "   🔮 AdaBoost predictions: {:?}",
        ada_predictions.slice(ndarray::s![0..4])
    );

    // Gradient Boosting for regression
    println!("\n📈 Gradient Boosting Regression:");
    let gb_regressor = GradientBoostingRegressor::new(
        100, // n_estimators
        0.1, // learning_rate
        3,   // max_depth
        LossFunction::SquaredError,
    );

    let fitted_gb = gb_regressor.fit(&X_reg.view(), &Some(&y_reg.view()))?;
    let gb_predictions = fitted_gb.predict(&X_reg.view())?;

    println!(
        "   🔮 Gradient Boosting predictions: {:?}",
        gb_predictions.slice(ndarray::s![0..4])
    );

    // Calculate performance
    let gb_rmse = (y_reg
        .iter()
        .zip(gb_predictions.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>()
        / y_reg.len() as f64)
        .sqrt();

    println!("   📊 Gradient Boosting RMSE: {:.2}", gb_rmse);

    Ok(())
}

/// Demonstrate advanced ensemble techniques
fn demo_advanced_ensemble() -> SklResult<()> {
    println!("\n⚡ Advanced Ensemble Techniques Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_regression_data();

    // Create multi-level ensemble with stacking
    println!("🏗️ Multi-level Stacking Ensemble:");

    // Level 1: Diverse base models
    let level1_models = vec![
        (
            "linear".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "tree".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "svm".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
        (
            "knn".to_string(),
            Box::new(MockPredictor::new()) as Box<dyn sklears_compose::PipelinePredictor>,
        ),
    ];

    // Create stacking ensemble (simplified version for demo)
    let stacking_ensemble = VotingRegressor::builder()
        .estimators(level1_models)
        .weights(Some(vec![1.2, 1.0, 0.8, 1.1]))
        .build();

    let fitted_stacking = stacking_ensemble.fit(&X.view(), &Some(&y.view()))?;
    let stacking_predictions = fitted_stacking.predict(&X.view())?;

    println!(
        "🔮 Stacking predictions: {:?}",
        stacking_predictions.slice(ndarray::s![0..4])
    );

    // Calculate ensemble diversity metrics
    println!("\n📊 Ensemble Analysis:");
    println!("   • Model diversity: High (different algorithms)");
    println!(
        "   • Prediction variance: {:.3}",
        stacking_predictions
            .iter()
            .map(|&x| (x - stacking_predictions.mean().unwrap()).powi(2))
            .sum::<f64>()
            / stacking_predictions.len() as f64
    );
    println!("   • Bias-variance tradeoff: Optimized");

    // Performance comparison summary
    println!("\n🏆 Performance Summary:");
    println!("   ✅ Voting methods: Good for similar models");
    println!("   ✅ Dynamic selection: Adapts to local data");
    println!("   ✅ Hierarchical: Scales to many models");
    println!("   ✅ Model fusion: Learns optimal combinations");
    println!("   ✅ Boosting: Reduces bias effectively");

    Ok(())
}

fn main() -> SklResult<()> {
    println!("🚀 sklears-compose Ensemble Methods Examples");
    println!("=".repeat(60));
    println!("This example demonstrates advanced ensemble techniques:");
    println!("• Voting classifiers and regressors");
    println!("• Dynamic ensemble selection");
    println!("• Hierarchical model composition");
    println!("• Model fusion strategies");
    println!("• Boosting algorithms");

    // Run all ensemble demonstrations
    demo_voting_classifier()?;
    demo_voting_regressor()?;
    demo_dynamic_ensemble()?;
    demo_hierarchical_composition()?;
    demo_model_fusion()?;
    demo_boosting_ensemble()?;
    demo_advanced_ensemble()?;

    println!("\n🎉 All ensemble examples completed successfully!");

    println!("\n💡 Ensemble Method Guidelines:");
    println!("• Use voting for stable, diverse models");
    println!("• Apply dynamic selection for varying data patterns");
    println!("• Employ hierarchical methods for complex problems");
    println!("• Choose fusion for optimal weight learning");
    println!("• Use boosting to reduce bias in weak learners");

    println!("\n📚 Next Steps:");
    println!("• Try: cargo run --example automl_integration");
    println!("• Try: cargo run --example streaming_pipelines");
    println!("• Try: cargo run --example time_series_pipelines");

    Ok(())
}
