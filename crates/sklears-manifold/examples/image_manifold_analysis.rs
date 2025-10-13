#![allow(deprecated)]

//! Real-world case study: Image-like data manifold analysis
//!
//! This example demonstrates how to use manifold learning techniques
//! to analyze high-dimensional image-like data, simulating a common
//! computer vision workflow.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::traits::{Fit, Transform};
use sklears_manifold::{benchmark_datasets::BenchmarkDatasets, Isomap, TSNE, UMAP};
use std::f64::consts::PI;

/// Generate synthetic image patches with known structure
fn generate_image_patches(
    n_patches: usize,
    patch_size: usize,
    random_state: u64,
) -> (Array2<f64>, Array1<usize>) {
    let mut rng = StdRng::seed_from_u64(random_state);
    let n_pixels = patch_size * patch_size;
    let mut patches = Array2::zeros((n_patches, n_pixels));
    let mut labels = Array1::zeros(n_patches);

    for i in 0..n_patches {
        let patch_type = i % 4; // 4 different patch types
        labels[i] = patch_type;

        match patch_type {
            0 => {
                // Horizontal stripes
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        let value = if (y / 2) % 2 == 0 { 1.0 } else { 0.0 };
                        patches[[i, y * patch_size + x]] = value + rng.gen::<f64>() * 0.2 - 0.1;
                    }
                }
            }
            1 => {
                // Vertical stripes
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        let value = if (x / 2) % 2 == 0 { 1.0 } else { 0.0 };
                        patches[[i, y * patch_size + x]] = value + rng.gen::<f64>() * 0.2 - 0.1;
                    }
                }
            }
            2 => {
                // Diagonal pattern
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        let value = if (x + y) % 3 == 0 { 1.0 } else { 0.0 };
                        patches[[i, y * patch_size + x]] = value + rng.gen::<f64>() * 0.2 - 0.1;
                    }
                }
            }
            3 => {
                // Circular pattern
                let center = patch_size as f64 / 2.0;
                for y in 0..patch_size {
                    for x in 0..patch_size {
                        let dx = x as f64 - center;
                        let dy = y as f64 - center;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let value = if distance < center * 0.7 { 1.0 } else { 0.0 };
                        patches[[i, y * patch_size + x]] = value + rng.gen::<f64>() * 0.2 - 0.1;
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    (patches, labels)
}

/// Generate time series data with different patterns
fn generate_time_series_data(
    n_series: usize,
    series_length: usize,
    random_state: u64,
) -> (Array2<f64>, Array1<usize>) {
    let mut rng = StdRng::seed_from_u64(random_state);
    let mut data = Array2::zeros((n_series, series_length));
    let mut labels = Array1::zeros(n_series);

    for i in 0..n_series {
        let pattern_type = i % 3; // 3 different time series patterns
        labels[i] = pattern_type;

        match pattern_type {
            0 => {
                // Sinusoidal pattern
                let frequency = rng.gen::<f64>() * (2.0 - (0.5)) + (0.5);
                let phase = rng.gen::<f64>() * (2.0 * PI - (0.0)) + (0.0);
                for t in 0..series_length {
                    let value =
                        (frequency * t as f64 * 2.0 * PI / series_length as f64 + phase).sin();
                    data[[i, t]] = value + rng.gen::<f64>() * 0.2 - 0.1;
                }
            }
            1 => {
                // Linear trend with noise
                let slope = rng.gen::<f64>() * (1.0 - (-1.0)) + (-1.0);
                let intercept = rng.gen::<f64>() * (0.5 - (-0.5)) + (-0.5);
                for t in 0..series_length {
                    let value = slope * t as f64 / series_length as f64 + intercept;
                    data[[i, t]] = value + rng.gen::<f64>() * 0.4 - 0.2;
                }
            }
            2 => {
                // Exponential decay
                let decay_rate = rng.gen::<f64>() * (0.5 - (0.1)) + (0.1);
                let amplitude = rng.gen::<f64>() * (2.0 - (0.5)) + (0.5);
                for t in 0..series_length {
                    let value = amplitude * (-decay_rate * t as f64).exp();
                    data[[i, t]] = value + rng.gen::<f64>() * 0.2 - 0.1;
                }
            }
            _ => unreachable!(),
        }
    }

    (data, labels)
}

/// Simple stress calculation for embedding quality
fn calculate_stress(original: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
    let n = original.nrows();
    let mut stress = 0.0;

    for i in 0..n {
        for j in i + 1..n {
            // Original distance
            let mut orig_dist_sq = 0.0;
            for k in 0..original.ncols() {
                let diff = original[[i, k]] - original[[j, k]];
                orig_dist_sq += diff * diff;
            }
            let orig_dist = orig_dist_sq.sqrt();

            // Embedding distance
            let mut emb_dist_sq = 0.0;
            for k in 0..embedding.ncols() {
                let diff = embedding[[i, k]] - embedding[[j, k]];
                emb_dist_sq += diff * diff;
            }
            let emb_dist = emb_dist_sq.sqrt();

            let diff = orig_dist - emb_dist;
            stress += diff * diff;
        }
    }

    stress.sqrt() / (n * (n - 1) / 2) as f64
}

/// Evaluate clustering quality of embedding
fn evaluate_embedding_quality(embedding: &Array2<f64>, true_labels: &Array1<usize>) -> f64 {
    let n_samples = embedding.nrows();
    let mut correct_neighbors = 0;
    let k = 5; // Check 5 nearest neighbors

    for i in 0..n_samples {
        let mut distances: Vec<(f64, usize)> = Vec::new();

        // Calculate distances to all other points
        for j in 0..n_samples {
            if i != j {
                let mut dist_sq = 0.0;
                for d in 0..embedding.ncols() {
                    let diff = embedding[[i, d]] - embedding[[j, d]];
                    dist_sq += diff * diff;
                }
                distances.push((dist_sq.sqrt(), j));
            }
        }

        // Sort by distance and check k nearest neighbors
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for neighbor_idx in 0..k.min(distances.len()) {
            let neighbor_label = true_labels[distances[neighbor_idx].1];
            if neighbor_label == true_labels[i] {
                correct_neighbors += 1;
            }
        }
    }

    correct_neighbors as f64 / (n_samples * k) as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Real-world Manifold Learning Case Studies");
    println!("{}", "=".repeat(50));

    // Case Study 1: Image patch analysis
    println!("\n📸 Case Study 1: Image Patch Analysis");
    println!("{}", "-".repeat(40));

    let (image_patches, patch_labels) = generate_image_patches(200, 8, 42);
    println!(
        "Generated {} image patches of size 8x8 ({} features)",
        image_patches.nrows(),
        image_patches.ncols()
    );

    // Apply different manifold learning techniques
    let algorithms = vec![("t-SNE", "tsne"), ("UMAP", "umap"), ("Isomap", "isomap")];

    for (name, method) in &algorithms {
        println!("\nApplying {} to image patches...", name);

        let embedding = match *method {
            "tsne" => {
                let tsne = TSNE::new().n_components(2).perplexity(30.0).n_iter(300);
                let fitted = tsne.fit(&image_patches.view(), &()).unwrap();
                fitted.transform(&image_patches.view()).unwrap()
            }
            "umap" => {
                let umap = UMAP::new().n_components(2).n_neighbors(15).min_dist(0.1);
                let fitted = umap.fit(&image_patches.view(), &()).unwrap();
                fitted.transform(&image_patches.view()).unwrap()
            }
            "isomap" => {
                let isomap = Isomap::new().n_components(2).n_neighbors(10);
                let fitted = isomap.fit(&image_patches.view(), &()).unwrap();
                fitted.transform(&image_patches.view()).unwrap()
            }
            _ => unreachable!(),
        };

        let quality = evaluate_embedding_quality(&embedding, &patch_labels);
        println!(
            "{} embedding quality (neighbor preservation): {:.3}",
            name, quality
        );
    }

    // Case Study 2: Time series pattern discovery
    println!("\n⏰ Case Study 2: Time Series Pattern Discovery");
    println!("{}", "-".repeat(40));

    let (time_series, ts_labels) = generate_time_series_data(150, 50, 123);
    println!(
        "Generated {} time series of length {} ({} features)",
        time_series.nrows(),
        time_series.ncols(),
        time_series.ncols()
    );

    for (name, method) in &algorithms {
        println!("\nApplying {} to time series data...", name);

        let embedding = match *method {
            "tsne" => {
                let tsne = TSNE::new().n_components(2).perplexity(20.0).n_iter(500);
                let fitted = tsne.fit(&time_series.view(), &()).unwrap();
                fitted.transform(&time_series.view()).unwrap()
            }
            "umap" => {
                let umap = UMAP::new().n_components(2).n_neighbors(10).min_dist(0.05);
                let fitted = umap.fit(&time_series.view(), &()).unwrap();
                fitted.transform(&time_series.view()).unwrap()
            }
            "isomap" => {
                let isomap = Isomap::new().n_components(2).n_neighbors(8);
                let fitted = isomap.fit(&time_series.view(), &()).unwrap();
                fitted.transform(&time_series.view()).unwrap()
            }
            _ => unreachable!(),
        };

        let quality = evaluate_embedding_quality(&embedding, &ts_labels);
        println!(
            "{} embedding quality (neighbor preservation): {:.3}",
            name, quality
        );
    }

    // Case Study 3: High-dimensional benchmark comparison
    println!("\n🏔️  Case Study 3: Swiss Roll with Multiple Algorithms");
    println!("{}", "-".repeat(40));

    let (swiss_data, _swiss_colors) = BenchmarkDatasets::swiss_roll(300, 0.1, 789);
    println!(
        "Generated Swiss Roll dataset: {} samples in 3D",
        swiss_data.nrows()
    );

    // Compare different algorithms on Swiss Roll
    let algorithms = vec![
        ("t-SNE (Fast)", "tsne_fast"),
        ("t-SNE (Quality)", "tsne_quality"),
        ("UMAP", "umap"),
    ];

    for (preset_name, method) in &algorithms {
        println!("\nUsing configuration: {}", preset_name);

        let embedding = match *method {
            "tsne_fast" => {
                let tsne = TSNE::new().n_components(2).perplexity(20.0).n_iter(250);
                let fitted = tsne.fit(&swiss_data.view(), &()).unwrap();
                fitted.transform(&swiss_data.view()).unwrap()
            }
            "tsne_quality" => {
                let tsne = TSNE::new().n_components(2).perplexity(50.0).n_iter(1000);
                let fitted = tsne.fit(&swiss_data.view(), &()).unwrap();
                fitted.transform(&swiss_data.view()).unwrap()
            }
            "umap" => {
                let umap = UMAP::new().n_components(2).n_neighbors(15).min_dist(0.1);
                let fitted = umap.fit(&swiss_data.view(), &()).unwrap();
                fitted.transform(&swiss_data.view()).unwrap()
            }
            _ => unreachable!(),
        };

        // Calculate a simple quality metric
        let stress = calculate_stress(&swiss_data, &embedding);
        println!("{} - Normalized stress: {:.4}", preset_name, stress);
    }

    println!("\n✅ Case studies completed successfully!");
    println!("These examples demonstrate how manifold learning can be applied to:");
    println!("  • Image patch analysis for computer vision");
    println!("  • Time series pattern discovery");
    println!("  • Benchmark dataset evaluation");

    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_patch_generation() {
        let (patches, labels) = generate_image_patches(40, 4, 42);
        assert_eq!(patches.shape(), &[40, 16]); // 4x4 = 16 pixels
        assert_eq!(labels.len(), 40);

        // Check that we have all 4 pattern types
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 4);
    }

    #[test]
    fn test_time_series_generation() {
        let (series, labels) = generate_time_series_data(30, 20, 123);
        assert_eq!(series.shape(), &[30, 20]);
        assert_eq!(labels.len(), 30);

        // Check that we have all 3 pattern types
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 3);
    }

    #[test]
    fn test_embedding_quality_evaluation() {
        // Create simple test case where points of same label are close
        let embedding = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // label 0
                0.1, 0.1, // label 0
                1.0, 1.0, // label 1
                1.1, 1.1, // label 1
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let quality = evaluate_embedding_quality(&embedding, &labels);
        assert!(quality > 0.5); // Should have good quality
    }
}
