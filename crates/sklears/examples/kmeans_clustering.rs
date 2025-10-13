//! Example: K-Means Clustering
//!
//! This example demonstrates:
//! - Basic K-means clustering
//! - K-means++ initialization
//! - Multiple runs with different initializations
//! - Cluster visualization using distances

use scirs2_core::ndarray::{array, s, Array2, Axis};
use sklears::clustering::{KMeans, KMeansConfig, KMeansInit};
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("=== K-Means Clustering Example ===\n");

    // 1. Basic K-Means on synthetic data
    println!("1. Basic K-Means Clustering");
    println!("--------------------------");

    // Create synthetic data with 3 well-separated clusters
    let cluster1 = generate_cluster(30, [2.0, 2.0], 0.5);
    let cluster2 = generate_cluster(30, [8.0, 2.0], 0.5);
    let cluster3 = generate_cluster(30, [5.0, 8.0], 0.5);

    // Combine all clusters
    let mut data = Array2::zeros((90, 2));
    data.slice_mut(s![0..30, ..]).assign(&cluster1);
    data.slice_mut(s![30..60, ..]).assign(&cluster2);
    data.slice_mut(s![60..90, ..]).assign(&cluster3);

    // Fit K-Means with K-means++ initialization
    let config = KMeansConfig {
        n_clusters: 3,
        init: KMeansInit::KMeansPlusPlus,
        random_seed: Some(42),
        ..Default::default()
    };
    let kmeans_model = KMeans::new(config);
    let y_dummy = scirs2_core::ndarray::Array1::zeros(data.nrows());
    let kmeans = kmeans_model.fit(&data, &y_dummy)?;

    // Note: KMeansFitted API currently doesn't expose n_iter(), inertia(), cluster_centers()
    println!("K-Means model fitted successfully");

    // 2. Compare initialization methods
    println!("\n2. Comparing Initialization Methods");
    println!("-----------------------------------");

    // K-means++ initialization
    let config_pp = KMeansConfig {
        n_clusters: 3,
        init: KMeansInit::KMeansPlusPlus,
        random_seed: Some(42),
        ..Default::default()
    };
    let kmeans_pp_model = KMeans::new(config_pp);
    let _kmeans_pp = kmeans_pp_model.fit(&data, &y_dummy)?;

    println!("K-means++:");
    println!("  Model fitted successfully");

    // Random initialization
    let config_random = KMeansConfig {
        n_clusters: 3,
        init: KMeansInit::Random,
        random_seed: Some(42),
        ..Default::default()
    };
    let kmeans_random_model = KMeans::new(config_random);
    let _kmeans_random = kmeans_random_model.fit(&data, &y_dummy)?;

    println!("\nRandom initialization:");
    println!("  Model fitted successfully");

    // 3. Elbow method for choosing K
    println!("\n3. Elbow Method for Choosing K");
    println!("------------------------------");

    let k_values = vec![1, 2, 3, 4, 5, 6, 7, 8];
    println!("K    Inertia");
    println!("---  -------");

    for &k in &k_values {
        let config_elbow = KMeansConfig {
            n_clusters: k,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans_elbow = KMeans::new(config_elbow);
        let _model = kmeans_elbow.fit(&data, &y_dummy)?;
        println!("{:2}   Fitted", k);
    }

    println!("\nNote: Look for the 'elbow' where inertia decreases more slowly");

    // 4. Predict on new data
    println!("\n4. Predicting Cluster Membership");
    println!("---------------------------------");

    let new_points = array![
        [2.0, 2.5], // Near cluster 1
        [8.5, 2.0], // Near cluster 2
        [5.0, 7.5], // Near cluster 3
        [5.0, 5.0], // Between clusters
    ];

    let predictions = kmeans.predict(&new_points)?;

    println!("New point predictions:");
    for (i, &label) in predictions.iter().enumerate() {
        println!(
            "  Point [{:.1}, {:.1}] -> Cluster {}",
            new_points[[i, 0]],
            new_points[[i, 1]],
            label
        );
    }

    // 5. Transform to distance space
    println!("\n5. Distance to Cluster Centers");
    println!("------------------------------");

    // Note: transform() method not available in current KMeansFitted API
    println!("Distance calculation not available in current API");

    // 6. Mini-batch K-Means (simulated with subset)
    println!("\n6. Mini-batch K-Means Simulation");
    println!("--------------------------------");

    // For very large datasets, you might want to fit on a subset
    let n_subset = 30;
    let subset_indices: Vec<usize> = (0..90).step_by(3).collect();
    let data_subset = data.select(Axis(0), &subset_indices);

    let config_mini = KMeansConfig {
        n_clusters: 3,
        max_iter: 50, // Fewer iterations for mini-batch
        random_seed: Some(42),
        ..Default::default()
    };
    let mini_kmeans_model = KMeans::new(config_mini);
    let y_subset_dummy = scirs2_core::ndarray::Array1::zeros(data_subset.nrows());
    let mini_kmeans = mini_kmeans_model.fit(&data_subset, &y_subset_dummy)?;

    // Predict on full dataset
    let full_predictions = mini_kmeans.predict(&data)?;

    println!("Mini-batch K-Means trained on {} samples", n_subset);
    println!("Applied to full dataset of {} samples", data.nrows());

    // Count cluster sizes
    let mut cluster_counts = vec![0; 3];
    for &label in full_predictions.iter() {
        cluster_counts[label as usize] += 1;
    }

    println!("Cluster sizes: {:?}", cluster_counts);

    Ok(())
}

/// Generate a cluster of points around a center with some noise
fn generate_cluster(n_points: usize, center: [f64; 2], std_dev: f64) -> Array2<f64> {
    use scirs2_core::essentials::Normal;
    use scirs2_core::random::{rngs::StdRng, Distribution, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, std_dev).unwrap();

    Array2::from_shape_fn((n_points, 2), |(_, j)| center[j] + normal.sample(&mut rng))
}
