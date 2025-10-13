//! Example: Hierarchical Clustering
//!
//! This example demonstrates:
//! - Agglomerative clustering with different linkage methods
//! - Using distance thresholds vs number of clusters
//! - Analyzing the linkage matrix (dendrogram)

use scirs2_core::ndarray::array;
use sklears::clustering::{AgglomerativeClustering, LinkageMethod, Metric};
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("=== Hierarchical Clustering Example ===\n");

    // 1. Basic Hierarchical Clustering
    println!("1. Basic Agglomerative Clustering");
    println!("--------------------------------");

    // Create synthetic hierarchical data
    let data = array![
        // Group 1 - tight cluster
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        // Group 2 - another tight cluster
        [3.0, 0.0],
        [3.1, 0.1],
        [3.2, 0.0],
        // Group 3 - looser cluster
        [6.0, 0.0],
        [6.5, 0.5],
        [7.0, 0.0],
    ];

    let model = AgglomerativeClustering::new()
        .n_clusters(3)
        .linkage(LinkageMethod::Ward)
        .fit(&data, &())?;

    println!("Cluster labels: {:?}", model.labels());
    println!("Number of clusters: {}", model.n_clusters());

    // 2. Different Linkage Methods
    println!("\n2. Comparing Linkage Methods");
    println!("----------------------------");

    let linkage_methods = vec![
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
    ];

    for method in linkage_methods {
        let model = AgglomerativeClustering::new()
            .n_clusters(3)
            .linkage(method)
            .fit(&data, &())?;

        println!("{:?} linkage: {:?}", method, model.labels());
    }

    // 3. Distance Threshold
    println!("\n3. Using Distance Threshold");
    println!("---------------------------");

    // Create data with clear hierarchical structure
    let hierarchical_data = array![
        // Subcluster 1.1
        [0.0, 0.0],
        [0.1, 0.0],
        // Subcluster 1.2
        [1.0, 0.0],
        [1.1, 0.0],
        // Subcluster 2.1
        [10.0, 0.0],
        [10.1, 0.0],
        // Subcluster 2.2
        [11.0, 0.0],
        [11.1, 0.0],
    ];

    let thresholds = vec![0.5, 1.5, 5.0];

    for threshold in thresholds {
        let model = AgglomerativeClustering::new()
            .distance_threshold(threshold)
            .linkage(LinkageMethod::Single)
            .fit(&hierarchical_data, &())?;

        println!(
            "Threshold {:.1}: {} clusters, labels: {:?}",
            threshold,
            model.n_clusters(),
            model.labels()
        );
    }

    // 4. Different Metrics
    println!("\n4. Different Distance Metrics");
    println!("-----------------------------");

    // Create 2D data where different metrics might give different results
    let metric_data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
    ];

    let metrics = vec![Metric::Euclidean, Metric::Manhattan, Metric::Chebyshev];

    for metric in metrics {
        let model = AgglomerativeClustering::new()
            .n_clusters(2)
            .linkage(LinkageMethod::Average)
            .metric(metric)
            .fit(&metric_data, &())?;

        println!("{:?} metric: {:?}", metric, model.labels());
    }

    // 5. Analyzing the Linkage Matrix
    println!("\n5. Linkage Matrix Analysis");
    println!("--------------------------");

    let analysis_data = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0],];

    let model = AgglomerativeClustering::new()
        .n_clusters(2)
        .linkage(LinkageMethod::Single)
        .fit(&analysis_data, &())?;

    let linkage_matrix = model.linkage_matrix();
    println!("Linkage matrix shape: {:?}", linkage_matrix.shape());
    println!("\nMerge sequence:");
    println!("Step | Cluster 1 | Cluster 2 | Distance | New Size");
    println!("-----|-----------|-----------|----------|----------");

    for (i, row) in linkage_matrix.outer_iter().enumerate() {
        println!(
            "{:4} | {:9.0} | {:9.0} | {:8.3} | {:8.0}",
            i + 1,
            row[0],
            row[1],
            row[2],
            row[3]
        );
    }

    // 6. Real-world Example: Customer Segmentation
    println!("\n6. Customer Segmentation Example");
    println!("--------------------------------");

    // Simulated customer data: [spending, frequency]
    let customers = array![
        // High value, high frequency
        [1000.0, 50.0],
        [1200.0, 48.0],
        [900.0, 52.0],
        // High value, low frequency
        [1500.0, 10.0],
        [1800.0, 8.0],
        [1600.0, 12.0],
        // Low value, high frequency
        [100.0, 45.0],
        [150.0, 50.0],
        [120.0, 48.0],
        // Low value, low frequency
        [50.0, 5.0],
        [80.0, 8.0],
        [60.0, 6.0],
    ];

    // Normalize the data first (important for clustering)
    let spending_mean = customers.column(0).mean().unwrap();
    let spending_std = customers.column(0).std(0.0);
    let frequency_mean = customers.column(1).mean().unwrap();
    let frequency_std = customers.column(1).std(0.0);

    let mut normalized = customers.clone();
    for i in 0..customers.nrows() {
        normalized[[i, 0]] = (customers[[i, 0]] - spending_mean) / spending_std;
        normalized[[i, 1]] = (customers[[i, 1]] - frequency_mean) / frequency_std;
    }

    let customer_model = AgglomerativeClustering::new()
        .n_clusters(4)
        .linkage(LinkageMethod::Ward)
        .fit(&normalized, &())?;

    println!("Customer segments: {:?}", customer_model.labels());

    // Analyze segments
    let mut segment_stats = vec![vec![]; 4];
    for (i, &label) in customer_model.labels().iter().enumerate() {
        segment_stats[label].push(i);
    }

    for (segment, customer_indices) in segment_stats.iter().enumerate() {
        if !customer_indices.is_empty() {
            let avg_spending: f64 = customer_indices
                .iter()
                .map(|&i| customers[[i, 0]])
                .sum::<f64>()
                / customer_indices.len() as f64;
            let avg_frequency: f64 = customer_indices
                .iter()
                .map(|&i| customers[[i, 1]])
                .sum::<f64>()
                / customer_indices.len() as f64;

            println!(
                "Segment {}: {} customers, avg spending: ${:.0}, avg frequency: {:.0}",
                segment,
                customer_indices.len(),
                avg_spending,
                avg_frequency
            );
        }
    }

    Ok(())
}
