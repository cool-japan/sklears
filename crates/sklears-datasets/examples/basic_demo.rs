//! Basic demonstration of sklears-datasets functionality
//!
//! This example shows how to use the basic dataset generation functions
//! available in the current minimal implementation.

use sklears_datasets::{make_blobs, make_classification, make_regression, SimpleDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Basic sklears-datasets Demo");
    println!("==============================\n");

    // Generate blob clusters
    println!("📊 Generating blob clusters...");
    let blobs = make_blobs(100, 2, Some(3), Some(1.0), None, Some(42))?;
    println!(
        "  - Generated {} samples with {} features",
        blobs.n_samples(),
        blobs.n_features()
    );
    println!("  - Has targets: {}\n", blobs.targets.is_some());

    // Generate classification dataset
    println!("🏷️  Generating classification dataset...");
    let classification = make_classification(
        150,
        4,
        Some(3),
        Some(1),
        None,
        Some(2),
        None,
        None,
        None,
        None,
        Some(42),
    )?;
    println!(
        "  - Generated {} samples with {} features",
        classification.n_samples(),
        classification.n_features()
    );
    println!("  - Has targets: {}\n", classification.targets.is_some());

    // Generate regression dataset
    println!("📈 Generating regression dataset...");
    let regression = make_regression(200, 5, Some(3), None, Some(0.1), None, None, Some(42))?;
    println!(
        "  - Generated {} samples with {} features",
        regression.n_samples(),
        regression.n_features()
    );
    println!("  - Has targets: {}\n", regression.targets.is_some());

    // Display some basic statistics
    println!("📋 Basic Statistics:");
    display_dataset_stats("Blobs", &blobs);
    display_dataset_stats("Classification", &classification);
    display_dataset_stats("Regression", &regression);

    println!("✅ Demo completed successfully!");
    Ok(())
}

fn display_dataset_stats(name: &str, dataset: &SimpleDataset) {
    println!("  {} Dataset:", name);
    println!(
        "    - Shape: {} × {}",
        dataset.n_samples(),
        dataset.n_features()
    );

    if let Some(ref targets) = dataset.targets {
        let min_target = targets.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_target = targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_target = targets.sum() / targets.len() as f64;
        println!("    - Target range: [{:.3}, {:.3}]", min_target, max_target);
        println!("    - Target mean: {:.3}", mean_target);
    }

    // Feature statistics
    let feature_means: Vec<f64> = (0..dataset.n_features())
        .map(|i| {
            let column = dataset.features.column(i);
            column.sum() / column.len() as f64
        })
        .collect();

    println!(
        "    - Feature means: {:?}",
        feature_means
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );
}
