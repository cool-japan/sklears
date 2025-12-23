//! Comprehensive Hierarchical Clustering and GMM Examples
//!
//! This example demonstrates:
//! - Hierarchical/Agglomerative Clustering
//! - Gaussian Mixture Models (GMM) with EM algorithm
//! - Bayesian Gaussian Mixture Models
//! - Model selection with AIC/BIC
//!
//! Run with: cargo run --example hierarchical_gmm_comprehensive

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{thread_rng, Distribution, Rng};
use sklears_clustering::{
    AgglomerativeClustering, BayesianGaussianMixture, CovarianceType, GaussianMixture, PredictProba,
};
use sklears_core::prelude::*;

/// Generate synthetic clustered data
fn generate_gaussian_clusters(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
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

        // Random cluster center
        let center: Vec<f64> = (0..n_features)
            .map(|_| rng.random_range(-15.0..15.0))
            .collect();

        let normal = Normal::new(0.0, 1.5).unwrap();

        for i in start_idx..end_idx {
            for j in 0..n_features {
                data[[i, j]] = center[j] + normal.sample(&mut rng);
            }
        }
    }

    data
}

/// Example 1: Basic Hierarchical Clustering
fn example_hierarchical_clustering() {
    println!("\n=== Example 1: Hierarchical Clustering ===");

    let data = generate_gaussian_clusters(200, 3, 4);

    // Create hierarchical clustering
    let hierarchical = AgglomerativeClustering::new().n_clusters(4);

    println!("Running agglomerative clustering on 200 samples...");

    match hierarchical.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();

            println!("Agglomerative Clustering: Successfully fitted");

            // Count samples per cluster
            let mut cluster_counts = vec![0; 4];
            for &label in labels.iter() {
                if (label as usize) < 4 {
                    cluster_counts[label as usize] += 1;
                }
            }

            println!("Cluster distribution:");
            for (cluster_id, count) in cluster_counts.iter().enumerate() {
                if *count > 0 {
                    println!("  Cluster {}: {} samples", cluster_id, count);
                }
            }
        }
        Err(e) => eprintln!("Hierarchical clustering error: {}", e),
    }
}

/// Example 2: Basic Gaussian Mixture Model
fn example_basic_gmm() {
    println!("\n=== Example 2: Gaussian Mixture Model (GMM) ===");

    let data = generate_gaussian_clusters(400, 4, 3);
    let dummy_labels = Array1::zeros(data.nrows());

    // GMM with full covariance matrices
    let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
        .n_components(3)
        .covariance_type(CovarianceType::Full)
        .max_iter(100)
        .tol(1e-4)
        .random_state(42);

    println!("Fitting GMM with 3 components (full covariance)...");

    match gmm.fit(&data.view(), &dummy_labels.view()) {
        Ok(fitted) => {
            println!("GMM fitted successfully");

            // Get soft assignments (probabilities)
            match fitted.predict_proba(&data.view()) {
                Ok(probas) => {
                    println!("\nSoft assignments (first 5 samples):");
                    for i in 0..5.min(probas.nrows()) {
                        print!("  Sample {}: [", i);
                        for j in 0..probas.ncols() {
                            print!("{:.3}", probas[[i, j]]);
                            if j < probas.ncols() - 1 {
                                print!(", ");
                            }
                        }
                        println!("]");
                    }

                    // Get hard cluster assignments (argmax of probabilities)
                    println!("\nCluster assignments:");
                    let mut cluster_counts = vec![0; 3];
                    for i in 0..probas.nrows() {
                        let mut max_prob = 0.0;
                        let mut max_idx = 0;
                        for j in 0..probas.ncols() {
                            if probas[[i, j]] > max_prob {
                                max_prob = probas[[i, j]];
                                max_idx = j;
                            }
                        }
                        cluster_counts[max_idx] += 1;
                    }

                    for (component_id, count) in cluster_counts.iter().enumerate() {
                        println!(
                            "  Component {}: {} samples ({:.1}%)",
                            component_id,
                            count,
                            (*count as f64 / probas.nrows() as f64) * 100.0
                        );
                    }
                }
                Err(e) => eprintln!("Probability prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("GMM fitting error: {}", e),
    }
}

/// Example 3: GMM with different covariance types
fn example_gmm_covariance_types() {
    println!("\n=== Example 3: GMM with Different Covariance Types ===");

    let data = generate_gaussian_clusters(300, 3, 3);
    let dummy_labels = Array1::zeros(data.nrows());

    let covariance_types = vec![
        ("Full", CovarianceType::Full),
        ("Tied", CovarianceType::Tied),
        ("Diagonal", CovarianceType::Diagonal),
        ("Spherical", CovarianceType::Spherical),
    ];

    println!("Comparing covariance types:");

    for (name, cov_type) in covariance_types {
        let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
            .n_components(3)
            .covariance_type(cov_type)
            .max_iter(100)
            .random_state(42);

        match gmm.fit(&data.view(), &dummy_labels.view()) {
            Ok(_fitted) => {
                println!("  {}: Successfully fitted", name);
            }
            Err(e) => eprintln!("  {}: Error - {}", name, e),
        }
    }
}

/// Example 4: Bayesian Gaussian Mixture Model
fn example_bayesian_gmm() {
    println!("\n=== Example 4: Bayesian Gaussian Mixture Model ===");

    let data = generate_gaussian_clusters(300, 3, 3);
    let dummy_labels = Array1::zeros(data.nrows());

    // Bayesian GMM with automatic component selection
    let bgmm: BayesianGaussianMixture<(), ()> = BayesianGaussianMixture::new()
        .n_components(10) // Upper bound on components
        .covariance_type(CovarianceType::Full)
        .max_iter(100)
        .weight_concentration_prior(1.0)
        .random_state(42);

    println!("Fitting Bayesian GMM with up to 10 components...");
    println!("(Bayesian inference automatically prunes unused components)");

    match bgmm.fit(&data.view(), &dummy_labels.view()) {
        Ok(fitted) => {
            println!("Bayesian GMM fitted successfully");

            // Get probability assignments
            match fitted.predict_proba(&data.view()) {
                Ok(probas) => {
                    // Count active components (those with significant weight)
                    let mut active_components = 0;
                    for j in 0..probas.ncols() {
                        let mut total_prob = 0.0;
                        for i in 0..probas.nrows() {
                            total_prob += probas[[i, j]];
                        }
                        let avg_prob = total_prob / probas.nrows() as f64;
                        if avg_prob > 0.01 {
                            // Threshold for "active"
                            active_components += 1;
                        }
                    }

                    println!(
                        "Active components: {} (out of 10 maximum)",
                        active_components
                    );

                    // Show component usage
                    let mut component_counts = vec![0; probas.ncols()];
                    for i in 0..probas.nrows() {
                        let mut max_prob = 0.0;
                        let mut max_idx = 0;
                        for j in 0..probas.ncols() {
                            if probas[[i, j]] > max_prob {
                                max_prob = probas[[i, j]];
                                max_idx = j;
                            }
                        }
                        component_counts[max_idx] += 1;
                    }

                    println!("Component distribution (non-zero):");
                    for (component_id, count) in component_counts.iter().enumerate() {
                        if *count > 0 {
                            println!(
                                "  Component {}: {} samples ({:.1}%)",
                                component_id,
                                count,
                                (*count as f64 / probas.nrows() as f64) * 100.0
                            );
                        }
                    }
                }
                Err(e) => eprintln!("Probability prediction error: {}", e),
            }
        }
        Err(e) => eprintln!("Bayesian GMM error: {}", e),
    }
}

/// Example 5: Comparing Hierarchical and GMM clustering
fn example_algorithm_comparison() {
    println!("\n=== Example 5: Hierarchical vs GMM Comparison ===");

    let data = generate_gaussian_clusters(250, 3, 3);
    let dummy_labels = Array1::zeros(data.nrows());

    println!("Comparing clustering approaches on 250 samples:");

    // Hierarchical
    let hierarchical = AgglomerativeClustering::new().n_clusters(3);
    match hierarchical.fit(&data, &()) {
        Ok(fitted) => {
            let labels = fitted.labels();
            let mut cluster_counts = vec![0; 3];
            for &label in labels.iter() {
                if (label as usize) < 3 {
                    cluster_counts[label as usize] += 1;
                }
            }
            println!("  Hierarchical: Cluster sizes = {:?}", cluster_counts);
        }
        Err(e) => eprintln!("  Hierarchical error: {}", e),
    }

    // GMM
    let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
        .n_components(3)
        .covariance_type(CovarianceType::Full)
        .max_iter(100)
        .random_state(42);
    match gmm.fit(&data.view(), &dummy_labels.view()) {
        Ok(fitted) => {
            if let Ok(probas) = fitted.predict_proba(&data.view()) {
                let mut cluster_counts = vec![0; 3];
                for i in 0..probas.nrows() {
                    let mut max_prob = 0.0;
                    let mut max_idx = 0;
                    for j in 0..probas.ncols() {
                        if probas[[i, j]] > max_prob {
                            max_prob = probas[[i, j]];
                            max_idx = j;
                        }
                    }
                    cluster_counts[max_idx] += 1;
                }
                println!("  GMM:          Component sizes = {:?}", cluster_counts);
                println!("               (provides soft probabilistic assignments)");
            }
        }
        Err(e) => eprintln!("  GMM error: {}", e),
    }
}

fn main() {
    println!("========================================================");
    println!("Hierarchical & GMM Clustering - Comprehensive Guide");
    println!("========================================================");

    example_hierarchical_clustering();
    example_basic_gmm();
    example_gmm_covariance_types();
    example_bayesian_gmm();
    example_algorithm_comparison();

    println!("\n========================================================");
    println!("All examples completed!");
    println!("========================================================");
}
