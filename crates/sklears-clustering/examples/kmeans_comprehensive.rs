//! Comprehensive K-Means Clustering Examples
//!
//! This example demonstrates all major K-Means variants and their usage:
//! - Standard K-Means with different initialization methods
//! - Mini-Batch K-Means for large datasets
//! - X-Means for automatic cluster number selection
//! - G-Means for Gaussian cluster detection
//!
//! Run with: cargo run --example kmeans_comprehensive

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution, Rng};
use sklears_clustering::{
    GMeans, GMeansConfig, InformationCriterion, KMeans, KMeansConfig, KMeansInit, MiniBatchKMeans,
    MiniBatchKMeansConfig, XMeans, XMeansConfig,
};
use sklears_core::prelude::*;

/// Generate synthetic clustered data with well-separated Gaussian blobs
fn generate_clustered_data(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    cluster_std: f64,
) -> Array2<f64> {
    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    let samples_per_cluster = n_samples / n_clusters;

    for cluster_id in 0..n_clusters {
        let start_idx = cluster_id * samples_per_cluster;
        let end_idx = if cluster_id == n_clusters - 1 {
            n_samples
        } else {
            (cluster_id + 1) * samples_per_cluster
        };

        // Random cluster center in [-20, 20] range
        let center: Vec<f64> = (0..n_features)
            .map(|_| rng.random_range(-20.0..20.0))
            .collect();

        let normal = Normal::new(0.0, cluster_std).unwrap();

        for i in start_idx..end_idx {
            for j in 0..n_features {
                data[[i, j]] = center[j] + normal.sample(&mut rng);
            }
        }
    }

    data
}

/// Example 1: Basic K-Means with default settings
fn example_basic_kmeans() {
    println!("\n=== Example 1: Basic K-Means ===");

    // Generate sample data: 300 samples, 2 features, 3 clusters
    let data = generate_clustered_data(300, 2, 3, 1.0);

    // Create K-Means with explicit configuration
    let config = KMeansConfig {
        n_clusters: 3,
        max_iter: 300,
        init: KMeansInit::KMeansPlusPlus,
        tolerance: 1e-4,
        random_seed: Some(42),
    };
    let kmeans = KMeans::new(config);

    // Fit the model (K-Means doesn't use labels for unsupervised learning)
    let dummy_labels = Array1::zeros(data.nrows());
    match kmeans.fit(&data, &dummy_labels) {
        Ok(fitted) => {
            println!("K-Means converged in {} iterations", fitted.n_iterations);
            println!("Final inertia: {:.2}", fitted.inertia);

            // Get cluster assignments
            match fitted.predict(&data) {
                Ok(labels) => {
                    let labels_preview: Vec<_> = labels.iter().take(20).collect();
                    println!("Predicted labels (first 20): {:?}", labels_preview);

                    // Count samples per cluster
                    let mut cluster_counts = vec![0; 3];
                    for &label in labels.iter() {
                        if label >= 0 && (label as usize) < 3 {
                            cluster_counts[label as usize] += 1;
                        }
                    }

                    println!("Samples per cluster:");
                    for (cluster_id, count) in cluster_counts.iter().enumerate() {
                        println!("  Cluster {}: {} samples", cluster_id, count);
                    }
                }
                Err(e) => eprintln!("Prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("Fitting error: {}", e),
    }
}

/// Example 2: Comparing K-Means initialization methods
fn example_initialization_comparison() {
    println!("\n=== Example 2: K-Means Initialization Methods ===");

    let data = generate_clustered_data(500, 5, 4, 1.5);
    let dummy_labels = Array1::zeros(data.nrows());

    // Test different initialization methods
    let init_methods = vec![
        ("Random", KMeansInit::Random),
        ("K-Means++", KMeansInit::KMeansPlusPlus),
    ];

    for (name, init_method) in init_methods {
        let config = KMeansConfig {
            n_clusters: 4,
            max_iter: 100,
            init: init_method,
            tolerance: 1e-4,
            random_seed: Some(42),
        };
        let kmeans = KMeans::new(config);

        match kmeans.fit(&data, &dummy_labels) {
            Ok(fitted) => {
                println!("{} initialization:", name);
                println!("  - Converged in {} iterations", fitted.n_iterations);
                println!("  - Final inertia: {:.2}", fitted.inertia);
            }
            Err(e) => eprintln!("{} initialization error: {}", name, e),
        }
    }
}

/// Example 3: Mini-Batch K-Means for large datasets
fn example_minibatch_kmeans() {
    println!("\n=== Example 3: Mini-Batch K-Means for Large Datasets ===");

    // Generate larger dataset: 10,000 samples
    let data = generate_clustered_data(10000, 10, 5, 2.0);

    // Mini-Batch K-Means processes data in batches for memory efficiency
    let config = MiniBatchKMeansConfig {
        n_clusters: 5,
        batch_size: 100,
        max_iter: 100,
        random_seed: Some(42),
    };
    let mb_kmeans = MiniBatchKMeans::new(config);

    let dummy_labels = Array1::zeros(data.nrows());

    println!(
        "Processing {} samples with batch size {}",
        data.nrows(),
        100
    );

    match mb_kmeans.fit(&data, &dummy_labels) {
        Ok(fitted) => {
            println!("Mini-Batch K-Means successfully fitted");
            println!("Converged in {} iterations", fitted.n_iterations);
            println!("Final inertia: {:.2}", fitted.inertia);

            match fitted.predict(&data) {
                Ok(labels) => {
                    // Count samples per cluster
                    let mut cluster_counts = vec![0; 5];
                    for &label in labels.iter() {
                        if label >= 0 && (label as usize) < 5 {
                            cluster_counts[label as usize] += 1;
                        }
                    }

                    println!("Samples per cluster:");
                    for (cluster_id, count) in cluster_counts.iter().enumerate() {
                        println!("  Cluster {}: {} samples", cluster_id, count);
                    }
                }
                Err(e) => eprintln!("Prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("Fitting error: {}", e),
    }
}

/// Example 4: X-Means for automatic cluster number selection
fn example_xmeans() {
    println!("\n=== Example 4: X-Means - Automatic Cluster Selection ===");

    // Generate data with unknown number of clusters (true: 4 clusters)
    let data = generate_clustered_data(600, 3, 4, 1.2);

    // X-Means automatically determines the number of clusters
    let config = XMeansConfig {
        k_min: 2,
        k_max: 10,
        max_iter: 100,
        criterion: InformationCriterion::BIC,
        random_seed: Some(42),
    };
    let xmeans = XMeans::new(config);

    let dummy_labels = Array1::zeros(data.nrows());

    println!("Searching for optimal number of clusters between 2 and 10...");

    match xmeans.fit(&data, &dummy_labels) {
        Ok(fitted) => {
            println!("X-Means completed successfully");
            println!("Converged in {} iterations", fitted.n_iterations);

            match fitted.predict(&data) {
                Ok(labels) => {
                    let max_label = labels.iter().max().copied().unwrap_or(0);
                    let n_clusters = (max_label + 1) as usize;
                    println!("Number of clusters found: {}", n_clusters);

                    // Show cluster distribution
                    let mut cluster_counts = vec![0; n_clusters];
                    for &label in labels.iter() {
                        if label >= 0 && (label as usize) < n_clusters {
                            cluster_counts[label as usize] += 1;
                        }
                    }

                    println!("Cluster distribution:");
                    for (cluster_id, count) in cluster_counts.iter().enumerate() {
                        if *count > 0 {
                            println!(
                                "  Cluster {}: {} samples ({:.1}%)",
                                cluster_id,
                                count,
                                (*count as f64 / labels.len() as f64) * 100.0
                            );
                        }
                    }
                }
                Err(e) => eprintln!("Prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("X-Means error: {}", e),
    }
}

/// Example 5: G-Means for Gaussian cluster detection
fn example_gmeans() {
    println!("\n=== Example 5: G-Means - Gaussian Cluster Detection ===");

    // Generate Gaussian-distributed clusters
    let data = generate_clustered_data(800, 4, 5, 1.0);

    // G-Means uses statistical tests to find Gaussian clusters
    let config = GMeansConfig {
        k_min: 2,
        k_max: 10,
        max_iter: 100,
        alpha: 0.01, // Significance level for statistical test
        random_seed: Some(42),
    };
    let gmeans = GMeans::new(config);

    let dummy_labels = Array1::zeros(data.nrows());

    println!("Detecting Gaussian clusters with Anderson-Darling test...");

    match gmeans.fit(&data, &dummy_labels) {
        Ok(fitted) => {
            println!("G-Means completed successfully");

            match fitted.predict(&data) {
                Ok(labels) => {
                    let max_label = labels.iter().max().copied().unwrap_or(0);
                    let n_clusters = (max_label + 1) as usize;
                    println!("Number of Gaussian clusters detected: {}", n_clusters);

                    // Analyze cluster sizes
                    let mut cluster_sizes = vec![0; n_clusters];
                    for &label in labels.iter() {
                        if label >= 0 && (label as usize) < n_clusters {
                            cluster_sizes[label as usize] += 1;
                        }
                    }

                    println!("Cluster sizes:");
                    for (cluster_id, size) in cluster_sizes.iter().enumerate() {
                        if *size > 0 {
                            println!("  Cluster {}: {} samples", cluster_id, size);
                        }
                    }
                }
                Err(e) => eprintln!("Prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("G-Means error: {}", e),
    }
}

/// Example 6: Parameter sensitivity analysis
fn example_parameter_sensitivity() {
    println!("\n=== Example 6: Parameter Sensitivity Analysis ===");

    let data = generate_clustered_data(400, 3, 3, 1.0);
    let dummy_labels = Array1::zeros(data.nrows());

    // Test different tolerance values
    let tolerances = vec![1e-2, 1e-3, 1e-4, 1e-5];

    println!("Testing convergence with different tolerance values:");

    for tol in tolerances {
        let config = KMeansConfig {
            n_clusters: 3,
            max_iter: 100,
            init: KMeansInit::KMeansPlusPlus,
            tolerance: tol,
            random_seed: Some(42),
        };
        let kmeans = KMeans::new(config);

        match kmeans.fit(&data, &dummy_labels) {
            Ok(fitted) => {
                println!(
                    "  Tolerance {:.0e}: Converged in {} iterations",
                    tol, fitted.n_iterations
                );
            }
            Err(e) => eprintln!("  Tolerance {:.0e}: Error - {}", tol, e),
        }
    }
}

fn main() {
    println!("========================================");
    println!("K-Means Clustering - Comprehensive Guide");
    println!("========================================");

    example_basic_kmeans();
    example_initialization_comparison();
    example_minibatch_kmeans();
    example_xmeans();
    example_gmeans();
    example_parameter_sensitivity();

    println!("\n========================================");
    println!("All examples completed!");
    println!("========================================");
}
