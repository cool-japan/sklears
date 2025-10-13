//! Simple performance showcase
//!
//! This example demonstrates basic performance comparisons between
//! different manifold learning algorithms.

use scirs2_core::ndarray::Array2;
use sklears_core::traits::{Fit, Transform};
use sklears_manifold::{benchmark_datasets::BenchmarkDatasets, Isomap, TSNE, UMAP};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Simple Performance Showcase");
    println!("{}", "=".repeat(50));

    // Generate test data
    let n_samples = 1000;
    let (data, _) = BenchmarkDatasets::swiss_roll(n_samples, 0.1, 42);
    println!(
        "Generated Swiss Roll dataset: {} samples in 3D",
        data.nrows()
    );

    // Test different algorithms
    let algorithms = vec![("t-SNE", "tsne"), ("UMAP", "umap"), ("Isomap", "isomap")];

    println!("\n{}", "-".repeat(40));
    println!("Algorithm Performance Comparison");
    println!("{}", "-".repeat(40));

    for (name, method) in &algorithms {
        println!("\nTesting {}...", name);
        let start = Instant::now();

        let embedding = match *method {
            "tsne" => {
                let tsne = TSNE::new().n_components(2).perplexity(30.0).n_iter(250);
                let fitted = tsne.fit(&data.view(), &()).unwrap();
                fitted.transform(&data.view()).unwrap()
            }
            "umap" => {
                let umap = UMAP::new().n_components(2).n_neighbors(15).min_dist(0.1);
                let fitted = umap.fit(&data.view(), &()).unwrap();
                fitted.transform(&data.view()).unwrap()
            }
            "isomap" => {
                let isomap = Isomap::new().n_components(2).n_neighbors(10);
                let fitted = isomap.fit(&data.view(), &()).unwrap();
                fitted.transform(&data.view()).unwrap()
            }
            _ => unreachable!(),
        };

        let elapsed = start.elapsed();

        // Calculate a simple quality metric (preservation of local distances)
        let quality = calculate_local_preservation(&data, &embedding, 5);

        println!("  Time: {:.3}ms", elapsed.as_millis());
        println!("  Output shape: {:?}", embedding.shape());
        println!("  Local preservation (k=5): {:.3}", quality);
    }

    // Test scalability
    println!("\n{}", "-".repeat(40));
    println!("Scalability Test");
    println!("{}", "-".repeat(40));

    let sizes = vec![500, 1000, 1500];

    for &size in &sizes {
        println!("\nTesting with {} samples:", size);
        let (test_data, _) = BenchmarkDatasets::swiss_roll(size, 0.1, 42);

        // Quick t-SNE test
        let start = Instant::now();
        let tsne = TSNE::new().n_components(2).perplexity(20.0).n_iter(100);
        let fitted = tsne.fit(&test_data.view(), &()).unwrap();
        let _embedding = fitted.transform(&test_data.view()).unwrap();
        let elapsed = start.elapsed();

        println!(
            "  t-SNE: {:.3}ms ({:.1} samples/sec)",
            elapsed.as_millis(),
            size as f64 / elapsed.as_secs_f64()
        );
    }

    println!("\n✅ Performance showcase completed!");
    println!("Key insights:");
    println!("  • t-SNE: Good for visualization, slower for large datasets");
    println!("  • UMAP: Fast and preserves both local and global structure");
    println!("  • Isomap: Good for nonlinear manifolds, moderate speed");

    Ok(())
}

/// Calculate how well local neighborhoods are preserved
fn calculate_local_preservation(original: &Array2<f64>, embedding: &Array2<f64>, k: usize) -> f64 {
    let n = original.nrows();
    let mut total_preservation = 0.0;

    for i in 0..n {
        // Find k nearest neighbors in original space
        let mut orig_distances: Vec<(f64, usize)> = Vec::new();
        for j in 0..n {
            if i != j {
                let mut dist_sq: f64 = 0.0;
                for d in 0..original.ncols() {
                    let diff = original[[i, d]] - original[[j, d]];
                    dist_sq += diff * diff;
                }
                orig_distances.push((dist_sq.sqrt(), j));
            }
        }
        orig_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let orig_neighbors: Vec<usize> =
            orig_distances.iter().take(k).map(|(_, idx)| *idx).collect();

        // Find k nearest neighbors in embedding space
        let mut emb_distances: Vec<(f64, usize)> = Vec::new();
        for j in 0..n {
            if i != j {
                let mut dist_sq: f64 = 0.0;
                for d in 0..embedding.ncols() {
                    let diff = embedding[[i, d]] - embedding[[j, d]];
                    dist_sq += diff * diff;
                }
                emb_distances.push((dist_sq.sqrt(), j));
            }
        }
        emb_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let emb_neighbors: Vec<usize> = emb_distances.iter().take(k).map(|(_, idx)| *idx).collect();

        // Calculate overlap
        let intersection = orig_neighbors
            .iter()
            .filter(|&x| emb_neighbors.contains(x))
            .count();

        total_preservation += intersection as f64 / k as f64;
    }

    total_preservation / n as f64
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_preservation_calculation() {
        // Create simple test case
        let original = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.0, 1.0, // Point 2
                1.0, 1.0, // Point 3
            ],
        )
        .unwrap();

        // Perfect preservation (same layout)
        let preservation = calculate_local_preservation(&original, &original, 2);
        assert!((preservation - 1.0).abs() < 1e-6);

        // Random embedding (should have lower preservation)
        let random_embedding =
            Array2::from_shape_vec((4, 2), vec![3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.0]).unwrap();

        let random_preservation = calculate_local_preservation(&original, &random_embedding, 2);
        assert!(random_preservation < 1.0);
    }
}
