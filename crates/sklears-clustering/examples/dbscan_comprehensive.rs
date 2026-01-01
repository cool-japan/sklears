//! Comprehensive DBSCAN and Density-Based Clustering Examples
//!
//! This example demonstrates density-based clustering algorithms:
//! - DBSCAN (Density-Based Spatial Clustering)
//! - HDBSCAN (Hierarchical DBSCAN)
//! - OPTICS (Ordering Points To Identify Clustering Structure)
//! - Density Peaks Clustering
//!
//! Run with: cargo run --example dbscan_comprehensive

use scirs2_core::ndarray::Array2;
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution};
use sklears_clustering::{ClusterMethod, DensityPeaks, Optics, DBSCAN, HDBSCAN, NOISE};
use sklears_core::prelude::*;

/// Generate data with varying density regions
fn generate_density_varying_data(n_samples: usize) -> Array2<f64> {
    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, 2));

    // Dense cluster 1 (high density)
    let dense1_samples = n_samples / 3;
    for i in 0..dense1_samples {
        let normal = Normal::new(0.0, 0.5).unwrap();
        data[[i, 0]] = 0.0 + normal.sample(&mut rng);
        data[[i, 1]] = 0.0 + normal.sample(&mut rng);
    }

    // Dense cluster 2 (medium density)
    let dense2_samples = n_samples / 3;
    for i in dense1_samples..dense1_samples + dense2_samples {
        let normal = Normal::new(0.0, 1.0).unwrap();
        data[[i, 0]] = 10.0 + normal.sample(&mut rng);
        data[[i, 1]] = 10.0 + normal.sample(&mut rng);
    }

    // Sparse cluster 3 (low density)
    let _sparse_samples = n_samples - dense1_samples - dense2_samples;
    for i in dense1_samples + dense2_samples..n_samples {
        let normal = Normal::new(0.0, 2.0).unwrap();
        data[[i, 0]] = 20.0 + normal.sample(&mut rng);
        data[[i, 1]] = 0.0 + normal.sample(&mut rng);
    }

    data
}

/// Generate data with noise points
fn generate_data_with_noise(n_samples: usize, noise_ratio: f64) -> Array2<f64> {
    let mut rng = thread_rng();
    let n_noise = (n_samples as f64 * noise_ratio) as usize;
    let n_cluster = n_samples - n_noise;

    let mut data = Array2::zeros((n_samples, 2));

    // Cluster data
    for i in 0..n_cluster {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let cluster_id = i % 3;
        let center_x = (cluster_id as f64) * 10.0;
        data[[i, 0]] = center_x + normal.sample(&mut rng);
        data[[i, 1]] = normal.sample(&mut rng);
    }

    // Noise points (uniformly distributed)
    for i in n_cluster..n_samples {
        data[[i, 0]] = rng.random_range(-5.0..35.0);
        data[[i, 1]] = rng.random_range(-10.0..10.0);
    }

    data
}

/// Example 1: Basic DBSCAN clustering
fn example_basic_dbscan() {
    println!("\n=== Example 1: Basic DBSCAN ===");

    let data = generate_data_with_noise(300, 0.1);

    // DBSCAN with eps=1.5 and min_samples=5
    let dbscan = DBSCAN::new().eps(1.5).min_samples(5);

    match dbscan.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();

            // Count clusters and noise points
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();
            let n_noise = labels.iter().filter(|&&l| l == NOISE).count();

            println!("Number of clusters found: {}", n_clusters);
            println!("Number of noise points: {}", n_noise);
            println!(
                "Noise ratio: {:.1}%",
                (n_noise as f64 / labels.len() as f64) * 100.0
            );

            // Show cluster sizes
            let mut cluster_sizes = std::collections::HashMap::new();
            for &label in labels.iter() {
                if label != NOISE {
                    *cluster_sizes.entry(label).or_insert(0) += 1;
                }
            }

            println!("Cluster sizes:");
            let mut sorted_clusters: Vec<_> = cluster_sizes.iter().collect();
            sorted_clusters.sort_by_key(|(k, _)| *k);
            for (cluster_id, size) in sorted_clusters {
                println!("  Cluster {}: {} points", cluster_id, size);
            }
        }
        Err(e) => eprintln!("DBSCAN error: {}", e),
    }
}

/// Example 2: Parameter sensitivity - varying eps
fn example_dbscan_eps_sensitivity() {
    println!("\n=== Example 2: DBSCAN eps Parameter Sensitivity ===");

    let data = generate_data_with_noise(300, 0.1);

    let eps_values = vec![0.5, 1.0, 1.5, 2.0, 3.0];

    println!("Testing different eps values (min_samples=5):");

    for eps in eps_values {
        let dbscan = DBSCAN::new().eps(eps).min_samples(5);

        match dbscan.fit(&data, &()) {
            Ok(fitted) => {
                let labels = fitted.labels();
                let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
                let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();
                let n_noise = labels.iter().filter(|&&l| l == NOISE).count();

                println!(
                    "  eps={:.1}: {} clusters, {} noise points",
                    eps, n_clusters, n_noise
                );
            }
            Err(e) => eprintln!("  eps={:.1}: Error - {}", eps, e),
        }
    }
}

/// Example 3: HDBSCAN for hierarchical density-based clustering
fn example_hdbscan() {
    println!("\n=== Example 3: HDBSCAN - Hierarchical DBSCAN ===");

    let data = generate_density_varying_data(400);

    // HDBSCAN automatically handles varying density
    let hdbscan = HDBSCAN::new()
        .min_cluster_size(15)
        .min_samples(5)
        .cluster_selection_epsilon(0.0);

    println!("Running HDBSCAN with min_cluster_size=15...");

    match hdbscan.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();
            let n_noise = labels.iter().filter(|&&l| l == NOISE).count();

            println!("HDBSCAN Results:");
            println!("  Clusters found: {}", n_clusters);
            println!("  Noise points: {}", n_noise);

            // Cluster distribution
            let mut cluster_sizes = std::collections::HashMap::new();
            for &label in labels.iter() {
                if label != NOISE {
                    *cluster_sizes.entry(label).or_insert(0) += 1;
                }
            }

            println!("  Cluster distribution:");
            let mut sorted_clusters: Vec<_> = cluster_sizes.iter().collect();
            sorted_clusters.sort_by_key(|(k, _)| *k);
            for (cluster_id, size) in sorted_clusters {
                println!("    Cluster {}: {} points", cluster_id, size);
            }
        }
        Err(e) => eprintln!("HDBSCAN error: {}", e),
    }
}

/// Example 4: OPTICS for ordering and cluster structure
fn example_optics() {
    println!("\n=== Example 4: OPTICS - Cluster Structure Analysis ===");

    let data = generate_density_varying_data(300);

    // OPTICS produces a reachability plot for cluster structure analysis
    let optics = Optics::new()
        .min_samples(5)
        .max_eps(5.0)
        .cluster_method(ClusterMethod::Threshold(0.5));

    println!("Running OPTICS with Xi cluster extraction (xi=0.05)...");

    match optics.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();

            println!("OPTICS Results:");
            println!("  Clusters extracted: {}", n_clusters);

            // Show cluster hierarchy information
            let mut cluster_sizes = std::collections::HashMap::new();
            for &label in labels.iter() {
                if label != NOISE {
                    *cluster_sizes.entry(label).or_insert(0) += 1;
                }
            }

            if !cluster_sizes.is_empty() {
                println!("  Cluster sizes:");
                let mut sorted_clusters: Vec<_> = cluster_sizes.iter().collect();
                sorted_clusters.sort_by_key(|(k, _)| *k);
                for (cluster_id, size) in sorted_clusters {
                    println!("    Cluster {}: {} points", cluster_id, size);
                }
            }
        }
        Err(e) => eprintln!("OPTICS error: {}", e),
    }
}

/// Example 5: Density Peaks clustering
fn example_density_peaks() {
    println!("\n=== Example 5: Density Peaks Clustering ===");

    let data = generate_density_varying_data(300);

    // Density Peaks automatically finds cluster centers based on density
    let density_peaks = DensityPeaks::new()
        .cutoff_distance(2.0)
        .cutoff_percentile(2.0);

    println!("Finding density peaks for automatic cluster center detection...");

    match density_peaks.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();

            println!("Density Peaks Results:");
            println!("  Cluster centers detected: {}", n_clusters);

            // Analyze cluster distribution
            let mut cluster_sizes = std::collections::HashMap::new();
            for &label in labels.iter() {
                if label != NOISE {
                    *cluster_sizes.entry(label).or_insert(0) += 1;
                }
            }

            println!("  Cluster distribution:");
            let mut sorted_clusters: Vec<_> = cluster_sizes.iter().collect();
            sorted_clusters.sort_by_key(|(k, _)| *k);
            for (cluster_id, size) in sorted_clusters {
                println!("    Cluster {}: {} points", cluster_id, size);
            }
        }
        Err(e) => eprintln!("Density Peaks error: {}", e),
    }
}

/// Example 6: Comparing density-based methods
fn example_algorithm_comparison() {
    println!("\n=== Example 6: Density-Based Algorithm Comparison ===");

    let data = generate_density_varying_data(250);

    println!("Comparing DBSCAN, HDBSCAN, and OPTICS on varying-density data:");

    // DBSCAN
    let dbscan = DBSCAN::new().eps(1.5).min_samples(5);
    match dbscan.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();
            println!("  DBSCAN: {} clusters", n_clusters);
        }
        Err(e) => eprintln!("  DBSCAN error: {}", e),
    }

    // HDBSCAN
    let hdbscan = HDBSCAN::new().min_cluster_size(15).min_samples(5);
    match hdbscan.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();
            println!(
                "  HDBSCAN: {} clusters (handles varying density)",
                n_clusters
            );
        }
        Err(e) => eprintln!("  HDBSCAN error: {}", e),
    }

    // OPTICS
    let optics = Optics::new()
        .min_samples(5)
        .max_eps(5.0)
        .cluster_method(ClusterMethod::Threshold(0.5));
    match optics.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
            let n_clusters = unique_labels.iter().filter(|&&l| l != NOISE).count();
            println!(
                "  OPTICS: {} clusters (with reachability analysis)",
                n_clusters
            );
        }
        Err(e) => eprintln!("  OPTICS error: {}", e),
    }
}

fn main() {
    println!("=================================================");
    println!("Density-Based Clustering - Comprehensive Guide");
    println!("=================================================");

    example_basic_dbscan();
    example_dbscan_eps_sensitivity();
    example_hdbscan();
    example_optics();
    example_density_peaks();
    example_algorithm_comparison();

    println!("\n=================================================");
    println!("All examples completed!");
    println!("=================================================");
}
