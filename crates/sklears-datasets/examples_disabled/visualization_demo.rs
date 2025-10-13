//! Visualization Demo
//!
//! This example demonstrates how to use the visualization utilities
//! in the sklears-datasets crate to create various plots.
//!
//! This example requires the `visualization` feature to be enabled:
//! cargo run --example visualization_demo --features visualization

#[cfg(not(feature = "visualization"))]
fn main() {
    println!("This example requires the 'visualization' feature to be enabled.");
    println!("Run with: cargo run --example visualization_demo --features visualization");
}

#[cfg(feature = "visualization")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    main_impl()
}

#[cfg(feature = "visualization")]
use sklears_datasets::{
    make_blobs, make_classification, make_regression, plot_2d_classification, plot_2d_regression,
    plot_correlation_matrix, plot_dataset_comparison, plot_feature_distributions,
    plot_quality_metrics, PlotConfig,
};
#[cfg(feature = "visualization")]
use std::collections::HashMap;
#[cfg(feature = "visualization")]
use tempfile::tempdir;

#[cfg(feature = "visualization")]
fn main_impl() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Visualization Demo for SkleaRS Datasets\n");

    // Create a temporary directory for saving plots
    let temp_dir = tempdir()?;
    let temp_path = temp_dir.path();

    // 1. 2D Classification Plot
    println!("📊 Creating 2D classification plot...");
    let (features, targets) = make_classification(200, 2, 3, 0, 3, Some(42))?;

    let config = PlotConfig {
        title: "2D Classification Dataset".to_string(),
        xlabel: "Feature 1".to_string(),
        ylabel: "Feature 2".to_string(),
        width: 800,
        height: 600,
        color_palette: "viridis".to_string(),
        ..Default::default()
    };

    let plot_path = temp_path.join("classification_2d.png");
    plot_2d_classification(&plot_path, &features, &targets, Some(config))?;
    println!("✅ Saved to: {:?}", plot_path);

    // 2. 2D Regression Plot
    println!("\n📈 Creating 2D regression plot...");
    let (reg_features, reg_targets) = make_regression(150, 2, 1, 0.1, Some(123))?;

    let reg_config = PlotConfig {
        title: "2D Regression Dataset".to_string(),
        xlabel: "X1".to_string(),
        ylabel: "X2".to_string(),
        color_palette: "plasma".to_string(),
        ..Default::default()
    };

    let reg_plot_path = temp_path.join("regression_2d.png");
    plot_2d_regression(
        &reg_plot_path,
        &reg_features,
        &reg_targets,
        Some(reg_config),
    )?;
    println!("✅ Saved to: {:?}", reg_plot_path);

    // 3. Feature Distributions Plot
    println!("\n📋 Creating feature distributions plot...");
    let (dist_features, _) = make_blobs(300, 4, 3, 1.5, Some(456))?;
    let feature_names = vec![
        "Temperature".to_string(),
        "Humidity".to_string(),
        "Pressure".to_string(),
        "Wind Speed".to_string(),
    ];

    let dist_config = PlotConfig {
        title: "Feature Distributions".to_string(),
        width: 1000,
        height: 800,
        ..Default::default()
    };

    let dist_plot_path = temp_path.join("feature_distributions.png");
    plot_feature_distributions(
        &dist_plot_path,
        &dist_features,
        Some(&feature_names),
        Some(dist_config),
    )?;
    println!("✅ Saved to: {:?}", dist_plot_path);

    // 4. Correlation Matrix Heatmap
    println!("\n🔥 Creating correlation matrix heatmap...");
    let (corr_features, _) = make_classification(250, 5, 4, 0, 4, Some(789))?;
    let corr_feature_names = vec![
        "Feature A".to_string(),
        "Feature B".to_string(),
        "Feature C".to_string(),
        "Feature D".to_string(),
        "Feature E".to_string(),
    ];

    let corr_config = PlotConfig {
        title: "Feature Correlation Matrix".to_string(),
        width: 700,
        height: 700,
        ..Default::default()
    };

    let corr_plot_path = temp_path.join("correlation_matrix.png");
    plot_correlation_matrix(
        &corr_plot_path,
        &corr_features,
        Some(&corr_feature_names),
        Some(corr_config),
    )?;
    println!("✅ Saved to: {:?}", corr_plot_path);

    // 5. Quality Metrics Bar Chart
    println!("\n📊 Creating quality metrics chart...");
    let mut metrics = HashMap::new();
    metrics.insert("Data Completeness".to_string(), 0.95);
    metrics.insert("Consistency".to_string(), 0.88);
    metrics.insert("Validity".to_string(), 0.92);
    metrics.insert("Accuracy".to_string(), 0.86);
    metrics.insert("Uniqueness".to_string(), 0.94);
    metrics.insert("Timeliness".to_string(), 0.78);

    let quality_config = PlotConfig {
        title: "Dataset Quality Assessment".to_string(),
        xlabel: "Quality Dimensions".to_string(),
        ylabel: "Score (0-1)".to_string(),
        width: 900,
        height: 500,
        ..Default::default()
    };

    let quality_plot_path = temp_path.join("quality_metrics.png");
    plot_quality_metrics(&quality_plot_path, &metrics, Some(quality_config))?;
    println!("✅ Saved to: {:?}", quality_plot_path);

    // 6. Dataset Comparison
    println!("\n🔍 Creating dataset comparison plot...");
    let (dataset1, targets1) = make_blobs(150, 2, 4, 1.0, Some(111))?;
    let (dataset2, targets2) = make_blobs(150, 2, 4, 2.0, Some(222))?;

    let comparison_config = PlotConfig {
        title: "Dataset Comparison".to_string(),
        xlabel: "X-axis".to_string(),
        ylabel: "Y-axis".to_string(),
        width: 1200,
        height: 600,
        color_palette: "inferno".to_string(),
        ..Default::default()
    };

    let comparison_plot_path = temp_path.join("dataset_comparison.png");
    plot_dataset_comparison(
        &comparison_plot_path,
        (&dataset1, &targets1, "Compact Clusters"),
        (&dataset2, &targets2, "Spread Out Clusters"),
        Some(comparison_config),
    )?;
    println!("✅ Saved to: {:?}", comparison_plot_path);

    println!("\n🎉 All visualizations created successfully!");
    println!("📁 Check the temp directory: {:?}", temp_path);

    // Keep the temp directory alive so user can see the files
    println!("\n💡 Tip: Enable the 'visualization' feature to use these functions:");
    println!("   cargo run --example visualization_demo --features visualization");

    Ok(())
}
