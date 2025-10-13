//! Example: DBSCAN Clustering
//!
//! This example demonstrates:
//! - DBSCAN for discovering clusters of arbitrary shape
//! - Noise detection
//! - Parameter tuning (eps and min_samples)

use scirs2_core::ndarray::Array2;
use sklears::clustering::{DBSCAN, NOISE};
use sklears::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<()> {
    println!("=== DBSCAN Clustering Example ===\n");

    // 1. Basic DBSCAN on synthetic data
    println!("1. Basic DBSCAN Clustering");
    println!("-------------------------");

    // Create synthetic data with non-spherical clusters
    let data = create_moons_dataset();

    let dbscan = DBSCAN::new().eps(0.3).min_samples(5).fit(&data, &())?;

    let labels = dbscan.labels();
    println!("Number of clusters found: {}", dbscan.n_clusters());
    println!("Number of noise points: {}", dbscan.n_noise_points());
    println!(
        "Number of core samples: {}",
        dbscan.core_sample_indices().len()
    );

    // Count points per cluster
    let mut cluster_counts = std::collections::HashMap::new();
    for &label in labels.iter() {
        *cluster_counts.entry(label).or_insert(0) += 1;
    }

    println!("\nCluster sizes:");
    let mut sorted_labels: Vec<_> = cluster_counts.keys().collect();
    sorted_labels.sort();
    for &label in &sorted_labels {
        if *label == NOISE {
            println!("  Noise: {} points", cluster_counts[label]);
        } else {
            println!("  Cluster {}: {} points", label, cluster_counts[label]);
        }
    }

    // 2. Parameter sensitivity
    println!("\n2. Parameter Sensitivity Analysis");
    println!("---------------------------------");

    let eps_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let min_samples_values = vec![3, 5, 7];

    println!("eps   min_samples   clusters   noise_points");
    println!("---   -----------   --------   ------------");

    for &eps in &eps_values {
        for &min_samples in &min_samples_values {
            let model = DBSCAN::new()
                .eps(eps)
                .min_samples(min_samples)
                .fit(&data, &())?;

            println!(
                "{:.1}   {:11}   {:8}   {:12}",
                eps,
                min_samples,
                model.n_clusters(),
                model.n_noise_points()
            );
        }
    }

    // 3. Handling different cluster densities
    println!("\n3. Clusters with Different Densities");
    println!("------------------------------------");

    let mixed_density_data = create_mixed_density_clusters();

    // Try different eps values
    let eps_trials = vec![0.5, 1.0, 1.5];

    for &eps in &eps_trials {
        let model = DBSCAN::new()
            .eps(eps)
            .min_samples(5)
            .fit(&mixed_density_data, &())?;

        println!("\nWith eps={:.1}:", eps);
        println!("  Clusters found: {}", model.n_clusters());
        println!("  Noise points: {}", model.n_noise_points());
    }

    // 4. Detecting outliers
    println!("\n4. Outlier Detection with DBSCAN");
    println!("---------------------------------");

    // Create data with outliers
    let data_with_outliers = create_data_with_outliers();

    let outlier_detector = DBSCAN::new()
        .eps(0.5)
        .min_samples(5)
        .fit(&data_with_outliers, &())?;

    let outlier_labels = outlier_detector.labels();
    let outlier_indices: Vec<_> = outlier_labels
        .iter()
        .enumerate()
        .filter(|(_, &label)| label == NOISE)
        .map(|(i, _)| i)
        .collect();

    println!(
        "Detected {} outliers out of {} points",
        outlier_indices.len(),
        data_with_outliers.nrows()
    );

    if !outlier_indices.is_empty() {
        println!(
            "Outlier indices: {:?}",
            &outlier_indices[..5.min(outlier_indices.len())]
        );
        if outlier_indices.len() > 5 {
            println!("... and {} more", outlier_indices.len() - 5);
        }
    }

    // 5. Comparison with K-Means
    println!("\n5. DBSCAN vs K-Means on Non-Spherical Clusters");
    println!("----------------------------------------------");

    // Use the moons dataset which has non-spherical clusters
    let comparison_data = create_moons_dataset();

    // DBSCAN
    let dbscan_model = DBSCAN::new()
        .eps(0.3)
        .min_samples(5)
        .fit(&comparison_data, &())?;

    println!("DBSCAN results:");
    println!("  Clusters: {}", dbscan_model.n_clusters());
    println!("  Noise points: {}", dbscan_model.n_noise_points());

    // K-Means (for comparison)
    use sklears::clustering::{KMeans, KMeansConfig};
    let config = KMeansConfig {
        n_clusters: 2,
        random_seed: Some(42),
        ..Default::default()
    };
    let kmeans = KMeans::new(config);
    let y_dummy = scirs2_core::ndarray::Array1::zeros(comparison_data.nrows());
    let kmeans_model = kmeans.fit(&comparison_data, &y_dummy)?;

    println!("\nK-Means results:");
    println!("  Clusters: 2 (specified)");
    // Note: inertia() not available in current KMeansFitted API

    println!("\nNote: DBSCAN can find clusters of arbitrary shape,");
    println!("while K-Means assumes spherical clusters.");

    Ok(())
}

/// Create two half-moon shaped clusters
fn create_moons_dataset() -> Array2<f64> {
    let n_samples = 100;
    let mut data = Array2::zeros((n_samples * 2, 2));

    // First moon
    for i in 0..n_samples {
        let angle = i as f64 * PI / n_samples as f64;
        data[[i, 0]] = angle.cos() + 0.05 * (i as f64 / 10.0).sin();
        data[[i, 1]] = angle.sin() + 0.05 * (i as f64 / 10.0).cos();
    }

    // Second moon (shifted and rotated)
    for i in 0..n_samples {
        let angle = i as f64 * PI / n_samples as f64;
        data[[n_samples + i, 0]] = 1.0 - angle.cos() + 0.05 * (i as f64 / 10.0).sin();
        data[[n_samples + i, 1]] = 0.5 - angle.sin() + 0.05 * (i as f64 / 10.0).cos();
    }

    data
}

/// Create clusters with different densities
fn create_mixed_density_clusters() -> Array2<f64> {
    let mut data = Vec::new();

    // Dense cluster
    for i in 0..50 {
        for j in 0..50 {
            data.push([i as f64 * 0.1, j as f64 * 0.1]);
        }
    }

    // Sparse cluster
    for i in 0..10 {
        for j in 0..10 {
            data.push([10.0 + i as f64 * 0.5, 10.0 + j as f64 * 0.5]);
        }
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Create data with obvious outliers
fn create_data_with_outliers() -> Array2<f64> {
    let mut data = Vec::new();

    // Main cluster
    for i in 0..100 {
        let x = 5.0 + 0.5 * (i as f64 / 10.0).cos();
        let y = 5.0 + 0.5 * (i as f64 / 10.0).sin();
        data.push([x, y]);
    }

    // Outliers
    data.push([0.0, 0.0]);
    data.push([10.0, 0.0]);
    data.push([0.0, 10.0]);
    data.push([10.0, 10.0]);
    data.push([2.5, 7.5]);

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}
