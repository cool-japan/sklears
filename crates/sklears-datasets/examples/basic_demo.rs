//! Basic demonstration of sklears-datasets functionality
//!
//! This example shows how to use the basic dataset generation functions
//! available in the current minimal implementation.

use sklears_datasets::{make_blobs, make_classification, make_regression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Basic sklears-datasets Demo");
    println!("==============================\n");

    // Generate blob clusters
    // make_blobs(n_samples, n_features, centers, cluster_std, random_state)
    println!("📊 Generating blob clusters...");
    let (features, targets) = make_blobs(100, 2, 3, 1.0, Some(42))?;
    println!(
        "  - Generated {} samples with {} features",
        features.nrows(),
        features.ncols()
    );
    println!("  - Has targets: true\n");

    // Generate classification dataset
    // make_classification(n_samples, n_features, n_informative, n_redundant, n_classes, random_state)
    println!("🏷️  Generating classification dataset...");
    let (class_features, class_targets) = make_classification(150, 4, 2, 1, 3, Some(42))?;
    println!(
        "  - Generated {} samples with {} features",
        class_features.nrows(),
        class_features.ncols()
    );
    println!("  - Has targets: true\n");

    // Generate regression dataset
    // make_regression(n_samples, n_features, n_informative, noise, random_state)
    println!("📈 Generating regression dataset...");
    let (reg_features, reg_targets) = make_regression(200, 5, 3, 0.1, Some(42))?;
    println!(
        "  - Generated {} samples with {} features",
        reg_features.nrows(),
        reg_features.ncols()
    );
    println!("  - Has targets: true\n");

    // Display some basic statistics
    println!("📋 Basic Statistics:");
    display_stats("Blobs", &features, &targets);
    display_stats("Classification", &class_features, &class_targets);
    display_reg_stats("Regression", &reg_features, &reg_targets);

    println!("✅ Demo completed successfully!");
    Ok(())
}

fn display_stats(
    name: &str,
    features: &scirs2_core::ndarray::Array2<f64>,
    targets: &scirs2_core::ndarray::Array1<i32>,
) {
    println!("  {} Dataset:", name);
    println!("    - Shape: {} × {}", features.nrows(), features.ncols());

    let min_target = *targets.iter().min().unwrap_or(&0);
    let max_target = *targets.iter().max().unwrap_or(&0);
    println!("    - Target range: [{}, {}]", min_target, max_target);
}

fn display_reg_stats(
    name: &str,
    features: &scirs2_core::ndarray::Array2<f64>,
    targets: &scirs2_core::ndarray::Array1<f64>,
) {
    println!("  {} Dataset:", name);
    println!("    - Shape: {} × {}", features.nrows(), features.ncols());

    let min_target = targets.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_target = targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_target = targets.sum() / targets.len() as f64;
    println!("    - Target range: [{:.3}, {:.3}]", min_target, max_target);
    println!("    - Target mean: {:.3}", mean_target);
}
