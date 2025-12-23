//! Ensemble Methods Example
//!
//! This example demonstrates ensemble learning techniques including:
//! - Voting Classifier (combining multiple models)
//! - Voting Regressor (averaging predictions)
//! - Dynamic Ensemble Selection
//!
//! Run with: cargo run --example ensemble_methods

use scirs2_core::ndarray::array;
use sklears_compose::{
    ensemble::{
        dynamic_selection::{DynamicEnsembleSelector, SelectionStrategy},
        voting::{VotingClassifier, VotingRegressor},
    },
    mock::MockPredictor,
};
use sklears_core::{error::Result as SklResult, traits::Fit};

fn main() -> SklResult<()> {
    println!("🎯 Ensemble Learning Methods");
    println!("============================\n");

    // Create sample training data
    let X_train = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ];

    let y_train = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

    println!(
        "📊 Training data: {} samples, {} features",
        X_train.nrows(),
        X_train.ncols()
    );
    println!();

    // =========================================================================
    // Example 1: Voting Classifier
    // =========================================================================
    println!("🗳️  Example 1: Voting Classifier");
    println!("==================================");
    println!("Combines predictions from multiple models using voting");
    println!();

    // Create base estimators
    let estimator1 = Box::new(MockPredictor::new());
    let estimator2 = Box::new(MockPredictor::new());
    let estimator3 = Box::new(MockPredictor::new());

    // Create voting classifier with hard voting
    let voting_clf = VotingClassifier::builder()
        .estimator("model1", estimator1)
        .estimator("model2", estimator2)
        .estimator("model3", estimator3)
        .voting("hard") // "hard" for majority vote, "soft" for probability averaging
        .build();

    println!("Created VotingClassifier with:");
    println!("  • 3 base estimators");
    println!("  • Hard voting strategy (majority vote)");
    println!();

    let y_view = y_train.view();
    let _trained_voting = voting_clf.fit(&X_train.view(), &Some(&y_view))?;
    println!("✅ Voting classifier trained successfully");
    println!();

    // =========================================================================
    // Example 2: Voting Regressor
    // =========================================================================
    println!("📊 Example 2: Voting Regressor");
    println!("===============================");
    println!("Combines regressor predictions by averaging");
    println!();

    // Create voting regressor
    let reg1 = Box::new(MockPredictor::new());
    let reg2 = Box::new(MockPredictor::new());
    let reg3 = Box::new(MockPredictor::new());

    let voting_reg = VotingRegressor::builder()
        .estimator("regressor1", reg1)
        .estimator("regressor2", reg2)
        .estimator("regressor3", reg3)
        .weights(vec![2.0, 1.0, 1.0]) // Weight first regressor 2x
        .build();

    println!("Created VotingRegressor with:");
    println!("  • 3 base regressors");
    println!("  • Weighted averaging (weights: [2.0, 1.0, 1.0])");
    println!();

    let _trained_voting_reg = voting_reg.fit(&X_train.view(), &Some(&y_view))?;
    println!("✅ Voting regressor trained successfully");
    println!();

    // =========================================================================
    // Example 3: Dynamic Ensemble Selection
    // =========================================================================
    println!("⚡ Example 3: Dynamic Ensemble Selection");
    println!("=========================================");
    println!("Dynamically selects best model(s) for each prediction");
    println!();

    // Create dynamic selector
    let model1 = Box::new(MockPredictor::new());
    let model2 = Box::new(MockPredictor::new());
    let model3 = Box::new(MockPredictor::new());

    let dynamic_selector = DynamicEnsembleSelector::builder()
        .estimator("fast_model", model1)
        .estimator("accurate_model", model2)
        .estimator("robust_model", model3)
        .selection_strategy(SelectionStrategy::KBest { k: 2 })
        .build();

    println!("Created DynamicEnsembleSelector with:");
    println!("  • 3 candidate models");
    println!("  • K-Best selection (k=2)");
    println!("  • Selects top 2 models per sample");
    println!();

    let _trained_dynamic = dynamic_selector.fit(&X_train.view(), &Some(&y_view))?;
    println!("✅ Dynamic selector trained successfully");
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("🎉 All Ensemble Methods Working!");
    println!("=================================\n");

    println!("✨ Summary:");
    println!("  1. Voting Classifier - Combines class predictions");
    println!("     • Hard voting: majority class");
    println!("     • Soft voting: average probabilities");
    println!("     • Optional weights for each estimator");
    println!();
    println!("  2. Voting Regressor - Combines regression predictions");
    println!("     • Weighted or simple averaging");
    println!("     • Reduces variance through averaging");
    println!();
    println!("  3. Dynamic Selection - Adaptive model choice");
    println!("     • K-Best: Select top K models");
    println!("     • Threshold: Models above performance threshold");
    println!("     • Local Competence: Best for each region");
    println!();

    println!("📚 When to use each:");
    println!("  • Voting: Simple, robust, works well with diverse models");
    println!("  • Weighted voting: When you know some models are better");
    println!("  • Dynamic: When model performance varies across data regions");
    println!();

    println!("💡 Pro Tips:");
    println!("  • Use diverse base models for better ensemble performance");
    println!("  • Cross-validate to avoid overfitting in stacking");
    println!("  • Monitor individual model performance in dynamic selection");

    Ok(())
}
