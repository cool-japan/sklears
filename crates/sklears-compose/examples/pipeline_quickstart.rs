//! Pipeline Quickstart Example for sklears-compose
//!
//! This example demonstrates the most common use case: creating a simple
//! machine learning pipeline with preprocessing and prediction.
//!
//! Run with: cargo run --example pipeline_quickstart

use scirs2_core::ndarray::array;
use sklears_compose::{mock::MockTransformer, Pipeline};
use sklears_core::{error::Result as SklResult, traits::Fit};

fn main() -> SklResult<()> {
    println!("🚀 Sklears-Compose Quickstart");
    println!("============================\n");

    // 1. Create sample training data
    println!("📊 Step 1: Creating sample data");
    println!("-------------------------------");

    let X_train = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ];

    let y_train = array![10.0, 20.0, 30.0, 40.0];

    println!("Training data shape: {:?}", X_train.dim());
    println!("Training labels shape: {:?}", y_train.dim());
    println!();

    // 2. Create a pipeline
    println!("🔧 Step 2: Building a pipeline");
    println!("------------------------------");

    let pipeline = Pipeline::builder()
        .step("scaler", Box::new(MockTransformer::new()))
        .step("model", Box::new(MockTransformer::new()))
        .build();

    println!("✅ Pipeline created with 2 steps:");
    println!("   1. scaler - Feature scaling");
    println!("   2. model - Prediction model");
    println!();

    // 3. Train the pipeline
    println!("🎓 Step 3: Training the pipeline");
    println!("--------------------------------");

    let y_view = y_train.view();
    let trained_pipeline = pipeline.fit(&X_train.view(), &Some(&y_view))?;

    println!("✅ Pipeline trained successfully");
    println!();

    // 4. Make predictions
    println!("🔮 Step 4: Making predictions");
    println!("-----------------------------");

    let X_test = array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]];

    let predictions = trained_pipeline.predict(&X_test.view())?;

    println!("Test data shape: {:?}", X_test.dim());
    println!("Predictions shape: {:?}", predictions.dim());
    println!("Predictions: {:?}", predictions);
    println!();

    // 5. Transform data through the pipeline
    println!("🔄 Step 5: Transforming data");
    println!("---------------------------");

    let transformed = trained_pipeline.transform(&X_test.view())?;

    println!("Transformed data shape: {:?}", transformed.dim());
    println!();

    println!("🎉 Quickstart Complete!");
    println!("======================");
    println!("\n✨ Key Takeaways:");
    println!("  • Pipelines chain multiple transformers and estimators");
    println!("  • Builder pattern provides type-safe construction");
    println!("  • fit() trains all components sequentially");
    println!("  • predict() runs the full pipeline for inference");
    println!("  • transform() applies transformations without prediction");
    println!("\n📚 Next Steps:");
    println!("  • See advanced_features_showcase.rs for DAG pipelines");
    println!("  • See ensemble examples for voting and stacking");
    println!("  • See domain_specific_pipelines.rs for computer vision & NLP");

    Ok(())
}
