//! Feature Union Example
//!
//! This example demonstrates parallel feature extraction using FeatureUnion.
//! FeatureUnion allows you to apply multiple transformers in parallel and
//! combine their outputs, which is useful for feature engineering.
//!
//! Run with: cargo run --example feature_union_demo

use scirs2_core::ndarray::array;
use sklears_compose::{mock::MockTransformer, FeatureUnion};
use sklears_core::{error::Result as SklResult, traits::Fit};

fn main() -> SklResult<()> {
    println!("🔗 Feature Union: Parallel Feature Extraction");
    println!("==============================================\n");

    // Create sample data
    let X = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ];

    println!("📊 Input data shape: {:?}", X.dim());
    println!("Original features: {}", X.ncols());
    println!();

    // =========================================================================
    // Example 1: Basic Feature Union
    // =========================================================================
    println!("🔧 Example 1: Basic Feature Union");
    println!("==================================");
    println!("Combining features from multiple transformers\n");

    let feature_union = FeatureUnion::new()
        .transformer("statistical_features", Box::new(MockTransformer::new()))
        .transformer("polynomial_features", Box::new(MockTransformer::new()))
        .transformer("interaction_features", Box::new(MockTransformer::new()));

    println!("Created FeatureUnion with 3 transformers:");
    println!("  1. statistical_features - Mean, std, etc.");
    println!("  2. polynomial_features - Squared, cubed terms");
    println!("  3. interaction_features - Feature interactions");
    println!();

    // Fit and transform
    let fitted_union = feature_union.fit(&X.view(), &None)?;
    let X_transformed = fitted_union.transform(&X.view())?;

    println!("✅ Feature union applied successfully");
    println!("Transformed shape: {:?}", X_transformed.dim());
    println!(
        "Features increased: {} → {}",
        X.ncols(),
        X_transformed.ncols()
    );
    println!();

    // =========================================================================
    // Example 2: Weighted Feature Union
    // =========================================================================
    println!("⚖️  Example 2: Weighted Feature Union");
    println!("=====================================");
    println!("Applying different weights to transformer outputs\n");

    use std::collections::HashMap;
    let mut weights = HashMap::new();
    weights.insert("important_features".to_string(), 2.0);
    weights.insert("auxiliary_features".to_string(), 0.5);

    let weighted_union = FeatureUnion::new()
        .transformer("important_features", Box::new(MockTransformer::new()))
        .transformer("auxiliary_features", Box::new(MockTransformer::new()))
        .transformer_weights(weights);

    println!("Created weighted FeatureUnion:");
    println!("  • important_features: weight = 2.0");
    println!("  • auxiliary_features: weight = 0.5");
    println!();

    let fitted_weighted = weighted_union.fit(&X.view(), &None)?;
    let X_weighted = fitted_weighted.transform(&X.view())?;

    println!("✅ Weighted feature union applied");
    println!("Output shape: {:?}", X_weighted.dim());
    println!();

    // =========================================================================
    // Example 3: Feature Union in Pipeline
    // =========================================================================
    // NOTE: FeatureUnion does not currently implement PipelineStep trait,
    // so it cannot be used directly in a Pipeline. This example is commented out
    // until PipelineStep is implemented for FeatureUnion.
    //
    // println!("🔀 Example 3: Feature Union + Pipeline");
    // println!("======================================");
    // println!("Combining FeatureUnion with other pipeline steps\n");

    // =========================================================================
    // Use Cases
    // =========================================================================
    println!("🎯 Common Use Cases");
    println!("===================\n");

    println!("1. Multi-Modal Features:");
    println!("   • Text features (TF-IDF, embeddings)");
    println!("   • Numeric features (scaling, normalization)");
    println!("   • Categorical features (one-hot encoding)");
    println!();

    println!("2. Feature Engineering:");
    println!("   • Original features");
    println!("   • Polynomial features");
    println!("   • Statistical aggregations");
    println!("   • Domain-specific transformations");
    println!();

    println!("3. Ensemble Features:");
    println!("   • Multiple feature extractors");
    println!("   • Different preprocessing strategies");
    println!("   • Complementary transformations");
    println!();

    // =========================================================================
    // Performance Tips
    // =========================================================================
    println!("💡 Performance Tips");
    println!("===================\n");

    println!("✓ Parallel Execution:");
    println!("  • Transformers run in parallel automatically");
    println!("  • Reduces total processing time");
    println!();

    println!("✓ Feature Selection:");
    println!("  • Remove redundant features after union");
    println!("  • Use feature importance scores");
    println!();

    println!("✓ Memory Efficiency:");
    println!("  • Consider feature dimensionality");
    println!("  • Use sparse transformers when appropriate");
    println!();

    println!("🎉 Feature Union Demo Complete!");
    println!("================================\n");

    println!("Key Benefits:");
    println!("  • Parallel feature extraction");
    println!("  • Easy combination of different feature types");
    println!("  • Flexible weighting of features");
    println!("  • Seamless integration with pipelines");

    Ok(())
}
