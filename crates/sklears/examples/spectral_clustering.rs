//! Example: Spectral Clustering
//!
//! This example demonstrates:
//! - Spectral clustering for non-convex clusters
//! - Different affinity matrix construction methods
//! - Comparison with K-means on challenging datasets

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears::clustering::{Affinity, KMeans, KMeansConfig, SpectralClustering};
use sklears::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<()> {
    println!("=== Spectral Clustering Example ===\n");

    // 1. Basic Spectral Clustering
    println!("1. Basic Spectral Clustering");
    println!("---------------------------");

    // Create concentric circles dataset
    let circles_data = create_concentric_circles();

    let model: SpectralClustering = SpectralClustering::new()
        .n_clusters(2)
        .affinity(Affinity::RBF)
        .gamma(0.1)
        .fit(&circles_data.view(), &Array1::zeros(0).view())?;

    println!("Number of samples: {}", circles_data.nrows());
    println!(
        "Affinity matrix shape: {:?}",
        model.affinity_matrix().shape()
    );

    // Count points in each cluster
    let mut cluster_counts = vec![0; 2];
    for &label in model.labels().iter() {
        cluster_counts[label] += 1;
    }
    println!("Cluster sizes: {:?}", cluster_counts);

    // 2. Comparing Affinity Types
    println!("\n2. Different Affinity Construction Methods");
    println!("-----------------------------------------");

    let affinity_types = vec![
        (Affinity::RBF, "RBF (Gaussian)"),
        (Affinity::NearestNeighbors, "Nearest Neighbors"),
    ];

    for (affinity, name) in affinity_types {
        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(affinity)
            .n_neighbors(10)
            .gamma(0.1)
            .fit(&circles_data.view(), &Array1::zeros(0).view())?;

        // Check clustering quality by seeing if inner/outer circles are separated
        let labels = model.labels();
        let inner_label = labels[0]; // First point is from inner circle
        let mut correct = 0;

        for (i, &label) in labels.iter().enumerate() {
            if i < circles_data.nrows() / 2 {
                // Inner circle points
                if label == inner_label {
                    correct += 1;
                }
            } else {
                // Outer circle points
                if label != inner_label {
                    correct += 1;
                }
            }
        }

        let accuracy = 100.0 * correct as f64 / labels.len() as f64;
        println!("{:20} accuracy: {:.1}%", name, accuracy);
    }

    // 3. Parameter sensitivity
    println!("\n3. Gamma Parameter Sensitivity (RBF Kernel)");
    println!("-------------------------------------------");

    let gamma_values = vec![0.01, 0.1, 1.0, 10.0];

    println!("Gamma   Cluster Balance");
    println!("------  ---------------");

    for &gamma in &gamma_values {
        let model: SpectralClustering = SpectralClustering::new()
            .n_clusters(2)
            .affinity(Affinity::RBF)
            .gamma(gamma)
            .fit(&circles_data.view(), &Array1::zeros(0).view())?;

        let mut counts = [0; 2];
        for &label in model.labels().iter() {
            counts[label] += 1;
        }

        println!("{:6.2}  [{:3}, {:3}]", gamma, counts[0], counts[1]);
    }

    // 4. Spectral vs K-Means on non-convex data
    println!("\n4. Spectral vs K-Means on Non-Convex Clusters");
    println!("---------------------------------------------");

    // Create two moons dataset
    let moons_data = create_two_moons();

    // Spectral clustering
    let spectral_model: SpectralClustering = SpectralClustering::new()
        .n_clusters(2)
        .affinity(Affinity::RBF)
        .gamma(1.0)
        .fit(&moons_data.view(), &Array1::zeros(0).view())?;

    // K-Means clustering
    let kmeans_config = KMeansConfig {
        n_clusters: 2,
        random_seed: Some(42),
        ..Default::default()
    };
    let kmeans = KMeans::new(kmeans_config);
    let y_dummy = Array1::zeros(moons_data.nrows());
    let kmeans_model = kmeans.fit(&moons_data, &y_dummy)?;

    // Evaluate clustering quality
    let spectral_score = evaluate_clustering(&moons_data, spectral_model.labels());
    let kmeans_labels = Array1::from_vec(kmeans_model.labels.iter().map(|&x| x as usize).collect());
    let kmeans_score = evaluate_clustering(&moons_data, &kmeans_labels);

    println!("Spectral clustering score: {:.3}", spectral_score);
    println!("K-Means clustering score:  {:.3}", kmeans_score);
    println!("\nNote: Higher scores indicate better separation of the two moons");

    // 5. Multiple clusters with complex shapes
    println!("\n5. Multiple Non-Convex Clusters");
    println!("-------------------------------");

    let spiral_data = create_spiral_dataset();

    let model: SpectralClustering = SpectralClustering::new()
        .n_clusters(3)
        .affinity(Affinity::NearestNeighbors)
        .n_neighbors(5)
        .fit(&spiral_data.view(), &Array1::zeros(0).view())?;

    let mut cluster_sizes = vec![0; 3];
    for &label in model.labels().iter() {
        cluster_sizes[label] += 1;
    }

    println!("Found 3 spiral arms with sizes: {:?}", cluster_sizes);

    // 6. Using precomputed affinity (similarity) matrix
    println!("\n6. Custom Affinity Matrix");
    println!("-------------------------");

    // Create a small dataset
    let small_data = array![[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0],];

    // Compute custom affinity matrix (e.g., based on connectivity)
    let n = small_data.nrows();
    let mut affinity = Array2::zeros((n, n));

    // Simple example: connect points if distance < 1.0
    for i in 0..n {
        for j in 0..n {
            let dx: f64 = small_data[[i, 0]] - small_data[[j, 0]];
            let dy: f64 = small_data[[i, 1]] - small_data[[j, 1]];
            let dist = (dx * dx + dy * dy).sqrt();
            affinity[[i, j]] = if dist < 1.0 { 1.0 } else { 0.0 };
        }
    }

    // Note: In the actual implementation, you would pass the precomputed affinity
    // For this example, we'll use regular spectral clustering
    let model: SpectralClustering = SpectralClustering::new()
        .n_clusters(2)
        .affinity(Affinity::RBF)
        .gamma(10.0) // High gamma for sharp cutoff
        .fit(&small_data.view(), &Array1::zeros(0).view())?;

    println!("Clustering result: {:?}", model.labels());
    println!("Expected: points 0,1 in one cluster and 2,3 in another");

    Ok(())
}

/// Create concentric circles dataset
fn create_concentric_circles() -> Array2<f64> {
    let mut data = Vec::new();
    let n_points = 100;
    let mut rng = thread_rng();

    // Inner circle
    for i in 0..n_points {
        let angle = 2.0 * PI * i as f64 / n_points as f64;
        let x = 0.3 * angle.cos() + 0.05 * (rng.random::<f64>() - 0.5);
        let y = 0.3 * angle.sin() + 0.05 * (rng.random::<f64>() - 0.5);
        data.push([x, y]);
    }

    // Outer circle
    for i in 0..n_points {
        let angle = 2.0 * PI * i as f64 / n_points as f64;
        let x = angle.cos() + 0.05 * (rng.random::<f64>() - 0.5);
        let y = angle.sin() + 0.05 * (rng.random::<f64>() - 0.5);
        data.push([x, y]);
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Create two moons dataset
fn create_two_moons() -> Array2<f64> {
    let mut data = Vec::new();
    let n_points = 100;
    let mut rng = thread_rng();

    // First moon
    for i in 0..n_points {
        let angle = PI * i as f64 / n_points as f64;
        let x = angle.cos() + 0.05 * (rng.random::<f64>() - 0.5);
        let y = angle.sin() + 0.05 * (rng.random::<f64>() - 0.5);
        data.push([x, y]);
    }

    // Second moon (shifted and flipped)
    for i in 0..n_points {
        let angle = PI * i as f64 / n_points as f64;
        let x = 1.0 - angle.cos() + 0.05 * (rng.random::<f64>() - 0.5);
        let y = 0.5 - angle.sin() + 0.05 * (rng.random::<f64>() - 0.5);
        data.push([x, y]);
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Create spiral dataset with 3 arms
fn create_spiral_dataset() -> Array2<f64> {
    let mut data = Vec::new();
    let n_points = 100;
    let mut rng = thread_rng();

    for arm in 0..3 {
        let angle_offset = arm as f64 * 2.0 * PI / 3.0;

        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            let angle = 4.0 * PI * t + angle_offset;
            let radius = 0.1 + 0.8 * t;

            let x = radius * angle.cos() + 0.03 * (rng.random::<f64>() - 0.5);
            let y = radius * angle.sin() + 0.03 * (rng.random::<f64>() - 0.5);
            data.push([x, y]);
        }
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Simple clustering quality evaluation
fn evaluate_clustering(data: &Array2<f64>, labels: &Array1<usize>) -> f64 {
    // For two moons: points in first half should have same label,
    // points in second half should have different label
    let n = data.nrows();
    let first_half_label = labels[0];
    let mut correct = 0;

    for i in 0..n / 2 {
        if labels[i] == first_half_label {
            correct += 1;
        }
    }

    for i in n / 2..n {
        if labels[i] != first_half_label {
            correct += 1;
        }
    }

    correct as f64 / n as f64
}
