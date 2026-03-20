//! Basic Pipeline Composition Examples
//!
//! This example demonstrates fundamental pipeline composition patterns in sklears-compose,
//! including sequential pipelines, feature unions, and column transformers.
//!
//! Run with: cargo run --example basic_pipeline_composition

use ndarray::{array, Array1, Array2};
use sklears_compose::{
    column_transformer::ColumnTransformer,
    mock::{MockPredictor, MockTransformer},
    FeatureUnion, Pipeline, PipelineBuilder, TransformedTargetRegressor,
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict, Transform},
};

/// Generate sample dataset for demonstration
fn generate_sample_data() -> (Array2<f64>, Array1<f64>) {
    let X = array![
        [1.0, 2.0, 10.0, 0.5],
        [2.0, 3.0, 20.0, 1.0],
        [3.0, 4.0, 30.0, 1.5],
        [4.0, 5.0, 40.0, 2.0],
        [5.0, 6.0, 50.0, 2.5],
        [6.0, 7.0, 60.0, 3.0],
        [7.0, 8.0, 70.0, 3.5],
        [8.0, 9.0, 80.0, 4.0]
    ];

    let y = array![10.0, 25.0, 45.0, 70.0, 100.0, 135.0, 175.0, 220.0];
    (X, y)
}

/// Demonstrate basic sequential pipeline
fn demo_sequential_pipeline() -> SklResult<()> {
    println!("\n🔧 Sequential Pipeline Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_sample_data();

    // Create a sequential pipeline: scaler -> predictor
    let pipeline = Pipeline::builder()
        .step("scaler", Box::new(MockTransformer::new()))
        .step(
            "feature_engineer",
            Box::new(MockTransformer::with_scale(2.0)),
        )
        .estimator(Box::new(MockPredictor::new()))
        .build();

    println!("📊 Input data shape: {:?}", X.dim());
    println!("🎯 Target data shape: {:?}", y.dim());

    // Fit the pipeline
    let fitted_pipeline = pipeline.fit(&X.view(), &Some(&y.view()))?;
    println!("✅ Pipeline fitted successfully");

    // Make predictions
    let predictions = fitted_pipeline.predict(&X.view())?;
    println!("🔮 Predictions: {:?}", predictions.slice(ndarray::s![0..3]));
    println!("📈 Original targets: {:?}", y.slice(ndarray::s![0..3]));

    Ok(())
}

/// Demonstrate feature union for parallel feature extraction
fn demo_feature_union() -> SklResult<()> {
    println!("\n🔀 Feature Union Demo");
    println!("{}", "=".repeat(50));

    let (X, _y) = generate_sample_data();

    // Create feature union with multiple transformers
    let mut feature_weights = std::collections::HashMap::new();
    feature_weights.insert("scaler".to_string(), 1.0);
    feature_weights.insert("polynomial".to_string(), 0.8);
    feature_weights.insert("normalizer".to_string(), 1.2);

    let feature_union = FeatureUnion::new()
        .transformer("scaler", Box::new(MockTransformer::new()))
        .transformer("polynomial", Box::new(MockTransformer::with_scale(2.0)))
        .transformer("normalizer", Box::new(MockTransformer::with_scale(0.5)))
        .transformer_weights(feature_weights)
        .n_jobs(Some(4));

    println!("📊 Input features: {}", X.ncols());

    // Fit and transform
    let fitted_union = feature_union.fit(&X.view(), &None)?;
    let transformed = fitted_union.transform(&X.view())?;

    println!("🔧 Original feature count: {}", X.ncols());
    println!("✨ Transformed feature count: {}", transformed.ncols());
    println!(
        "📈 Feature expansion ratio: {:.2}x",
        transformed.ncols() as f64 / X.ncols() as f64
    );

    // Display some transformed features
    println!("🔍 Sample transformed features:");
    for i in 0..std::cmp::min(3, transformed.nrows()) {
        println!(
            "   Sample {}: {:?}",
            i,
            transformed
                .row(i)
                .slice(ndarray::s![0..std::cmp::min(6, transformed.ncols())])
        );
    }

    Ok(())
}

/// Demonstrate transformed target regressor
fn demo_transformed_target() -> SklResult<()> {
    println!("\n🎯 Transformed Target Regressor Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_sample_data();

    // Create transformed target regressor with log transformation
    let transformer = TransformedTargetRegressor::new(Box::new(MockPredictor::new()))
        .func(|target| target.mapv(|x| (x + 1.0).ln()))
        .inverse_func(|pred| pred.mapv(|x| x.exp() - 1.0))
        .check_inverse(true);

    println!(
        "📊 Original target range: [{:.2}, {:.2}]",
        y.fold(f64::INFINITY, |acc, &x| acc.min(x)),
        y.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    );

    // Apply transformation
    let log_y = y.mapv(|x| (x + 1.0).ln());
    println!(
        "📈 Log-transformed range: [{:.2}, {:.2}]",
        log_y.fold(f64::INFINITY, |acc, &x| acc.min(x)),
        log_y.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    );

    // Fit and predict
    let fitted_transformer = transformer.fit(&X.view(), &Some(&y.view()))?;
    let predictions = fitted_transformer.predict(&X.view())?;

    println!(
        "🔮 Sample predictions: {:?}",
        predictions.slice(ndarray::s![0..3])
    );
    println!("🎯 Sample targets: {:?}", y.slice(ndarray::s![0..3]));

    Ok(())
}

/// Demonstrate complex pipeline with multiple stages
fn demo_complex_pipeline() -> SklResult<()> {
    println!("\n🏗️ Complex Pipeline Demo");
    println!("{}", "=".repeat(50));

    let (X, y) = generate_sample_data();

    // Create feature union for preprocessing
    let preprocessing = FeatureUnion::new()
        .transformer("scaler", Box::new(MockTransformer::new()))
        .transformer("poly", Box::new(MockTransformer::with_scale(1.5)));

    // Create complex pipeline: preprocessing -> selection -> prediction
    let complex_pipeline = Pipeline::builder()
        .step("preprocessing", Box::new(preprocessing))
        .step(
            "feature_selector",
            Box::new(MockTransformer::with_scale(0.8)),
        )
        .estimator(Box::new(MockPredictor::new()))
        .build();

    println!("📊 Building complex pipeline with {} steps", 2);

    // Fit the entire pipeline
    let fitted_complex = complex_pipeline.fit(&X.view(), &Some(&y.view()))?;
    let complex_predictions = fitted_complex.predict(&X.view())?;

    println!("✅ Complex pipeline fitted and executed successfully");
    println!(
        "🔮 Complex predictions: {:?}",
        complex_predictions.slice(ndarray::s![0..3])
    );

    // Calculate simple performance metrics
    let mse = y
        .iter()
        .zip(complex_predictions.iter())
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>()
        / y.len() as f64;

    println!("📊 Mean Squared Error: {:.4}", mse);

    Ok(())
}

/// Interactive demo runner
fn main() -> SklResult<()> {
    println!("🚀 sklears-compose Basic Pipeline Composition Examples");
    println!("{}", "=".repeat(60));
    println!("This example demonstrates fundamental pipeline patterns:");
    println!("• Sequential pipelines");
    println!("• Feature unions");
    println!("• Transformed target regression");
    println!("• Complex multi-stage pipelines");

    // Run all demonstrations
    demo_sequential_pipeline()?;
    demo_feature_union()?;
    demo_transformed_target()?;
    demo_complex_pipeline()?;

    println!("\n🎉 All examples completed successfully!");
    println!("\n💡 Key Takeaways:");
    println!("• Pipelines enable sequential data processing");
    println!("• Feature unions allow parallel feature extraction");
    println!("• Target transformation can improve model performance");
    println!("• Complex pipelines can combine multiple techniques");

    println!("\n📚 Next Steps:");
    println!("• Try: cargo run --example dag_pipelines");
    println!("• Try: cargo run --example ensemble_methods");
    println!("• Try: cargo run --example automl_integration");

    Ok(())
}
