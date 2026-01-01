//! Example: Mean Shift Clustering
//!
//! This example demonstrates:
//! - Mean shift clustering without specifying number of clusters
//! - Automatic bandwidth estimation
//! - Bin seeding for faster computation

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears::clustering::MeanShift;
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("=== Mean Shift Clustering Example ===\n");

    // 1. Basic Mean Shift clustering
    println!("1. Basic Mean Shift Clustering");
    println!("-----------------------------");

    // Create synthetic data with natural clusters
    let data = create_blob_dataset();

    // Mean shift with automatic bandwidth estimation
    let model: MeanShift = MeanShift::new().fit(&data.view(), &Array1::zeros(0).view())?;

    println!("Number of clusters found: {}", model.n_clusters());
    println!("Bandwidth used: {:.3}", model.bandwidth_used());

    // Analyze cluster centers
    let centers = model.cluster_centers()?;
    println!("\nCluster centers:");
    for (i, center) in centers.outer_iter().enumerate() {
        println!("  Cluster {}: [{:.2}, {:.2}]", i, center[0], center[1]);
    }

    // 2. Manual bandwidth selection
    println!("\n2. Effect of Bandwidth Parameter");
    println!("--------------------------------");

    let bandwidths = vec![0.5, 1.0, 2.0, 5.0];

    println!("Bandwidth  Clusters");
    println!("---------  --------");

    for &bandwidth in &bandwidths {
        let model: MeanShift = MeanShift::new()
            .bandwidth(bandwidth)
            .fit(&data.view(), &Array1::zeros(0).view())?;

        println!("{:9.1}  {:8}", bandwidth, model.n_clusters());
    }

    // 3. Bin seeding for performance
    println!("\n3. Bin Seeding for Faster Computation");
    println!("-------------------------------------");

    // Create larger dataset
    let large_data = create_large_dataset();

    // Without bin seeding
    let start = std::time::Instant::now();
    let model_no_bin: MeanShift = MeanShift::new()
        .bandwidth(1.0)
        .bin_seeding(false)
        .fit(&large_data.view(), &Array1::zeros(0).view())?;
    let time_no_bin = start.elapsed();

    // With bin seeding
    let start = std::time::Instant::now();
    let model_bin: MeanShift = MeanShift::new()
        .bandwidth(1.0)
        .bin_seeding(true)
        .min_bin_freq(5)
        .fit(&large_data.view(), &Array1::zeros(0).view())?;
    let time_bin = start.elapsed();

    println!(
        "Without bin seeding: {} clusters in {:.2}s",
        model_no_bin.n_clusters(),
        time_no_bin.as_secs_f64()
    );
    println!(
        "With bin seeding:    {} clusters in {:.2}s",
        model_bin.n_clusters(),
        time_bin.as_secs_f64()
    );

    // 4. Finding modes in multimodal data
    println!("\n4. Finding Modes in Multimodal Data");
    println!("-----------------------------------");

    // Create multimodal data (mixture of Gaussians)
    let multimodal_data = create_multimodal_data();

    let model: MeanShift = MeanShift::new()
        .cluster_all(true)
        .fit(&multimodal_data.view(), &Array1::zeros(0).view())?;

    println!("Found {} modes in the data", model.n_clusters());

    // Analyze mode strengths (cluster sizes)
    let labels = model.labels();
    let mut cluster_sizes = vec![0; model.n_clusters()];
    for &label in labels.iter() {
        cluster_sizes[label] += 1;
    }

    println!("\nMode strengths (cluster sizes):");
    for (i, &size) in cluster_sizes.iter().enumerate() {
        let percentage = 100.0 * size as f64 / labels.len() as f64;
        println!("  Mode {}: {} points ({:.1}%)", i, size, percentage);
    }

    // 5. Predict new points
    println!("\n5. Predicting Cluster Membership");
    println!("--------------------------------");

    let new_points = array![
        [2.5, 2.5],   // Near first cluster
        [7.5, 2.5],   // Near second cluster
        [5.0, 5.0],   // Between clusters
        [10.0, 10.0], // Far from all clusters
    ];

    let predictions = model.predict(&new_points.view())?;

    println!("New point predictions:");
    for (i, &label) in predictions.iter().enumerate() {
        let center = model.cluster_centers()?.row(label);
        println!(
            "  Point [{:.1}, {:.1}] -> Cluster {} (center: [{:.2}, {:.2}])",
            new_points[[i, 0]],
            new_points[[i, 1]],
            label,
            center[0],
            center[1]
        );
    }

    Ok(())
}

/// Create a dataset with natural blob-like clusters
fn create_blob_dataset() -> Array2<f64> {
    let mut data = Vec::new();
    let mut rng = thread_rng();

    // Cluster 1: centered at (2, 2)
    for _ in 0..100 {
        let x = 2.0 + 0.5 * rng.random::<f64>() - 0.25;
        let y = 2.0 + 0.5 * rng.random::<f64>() - 0.25;
        data.push([x, y]);
    }

    // Cluster 2: centered at (8, 2)
    for _ in 0..100 {
        let x = 8.0 + 0.5 * rng.random::<f64>() - 0.25;
        let y = 2.0 + 0.5 * rng.random::<f64>() - 0.25;
        data.push([x, y]);
    }

    // Cluster 3: centered at (5, 7)
    for _ in 0..100 {
        let x = 5.0 + 0.5 * rng.random::<f64>() - 0.25;
        let y = 7.0 + 0.5 * rng.random::<f64>() - 0.25;
        data.push([x, y]);
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Create a larger dataset for performance testing
fn create_large_dataset() -> Array2<f64> {
    let mut data = Vec::new();
    let mut rng = thread_rng();

    // Create 5 clusters with 200 points each
    let centers = vec![[0.0, 0.0], [5.0, 0.0], [2.5, 4.0], [0.0, 5.0], [5.0, 5.0]];

    for center in centers {
        for _ in 0..200 {
            let x = center[0] + rng.random::<f64>() - 0.5;
            let y = center[1] + rng.random::<f64>() - 0.5;
            data.push([x, y]);
        }
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Create multimodal data with overlapping distributions
fn create_multimodal_data() -> Array2<f64> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Vec::new();

    // Mode 1: Strong mode at (0, 0)
    for _ in 0..150 {
        let x = 0.3 * rng.random::<f64>() - 0.15;
        let y = 0.3 * rng.random::<f64>() - 0.15;
        data.push([x, y]);
    }

    // Mode 2: Weaker mode at (3, 0)
    for _ in 0..75 {
        let x = 3.0 + 0.5 * rng.random::<f64>() - 0.25;
        let y = 0.5 * rng.random::<f64>() - 0.25;
        data.push([x, y]);
    }

    // Mode 3: Diffuse mode at (1.5, 2)
    for _ in 0..50 {
        let x = 1.5 + 1.0 * rng.random::<f64>() - 0.5;
        let y = 2.0 + 1.0 * rng.random::<f64>() - 0.5;
        data.push([x, y]);
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}
