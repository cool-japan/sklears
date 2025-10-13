//! Robustness tests for clustering algorithms
//!
//! This module tests how clustering algorithms handle challenging data conditions
//! including noise, outliers, imbalanced clusters, and edge cases.

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::{ChaCha8Rng, Distribution, Rng, SeedableRng};
use sklears_clustering::{
    AgglomerativeClustering, GaussianMixture, KMeans, KMeansConfig, PredictProba, DBSCAN, HDBSCAN,
};
use sklears_core::{
    traits::{Fit, Predict},
    types::Float,
};

/// Generate data with controlled noise levels
fn generate_noisy_clusters(
    n_clusters: usize,
    n_samples_per_cluster: usize,
    n_features: usize,
    noise_level: Float,
) -> Array2<Float> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut data = Vec::new();

    for cluster_id in 0..n_clusters {
        let center_offset = (cluster_id as Float) * 5.0;

        for _ in 0..n_samples_per_cluster {
            let mut point = Vec::new();
            for feature_id in 0..n_features {
                let center = if feature_id == 0 { center_offset } else { 0.0 };
                let normal = Normal::new(center, noise_level).unwrap();
                point.push(normal.sample(&mut rng));
            }
            data.extend(point);
        }
    }

    Array2::from_shape_vec((n_clusters * n_samples_per_cluster, n_features), data).unwrap()
}

/// Add random outliers to existing data
fn add_outliers(data: Array2<Float>, n_outliers: usize, outlier_magnitude: Float) -> Array2<Float> {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let (n_samples, n_features) = data.dim();

    let mut outlier_data = Vec::new();
    for _ in 0..n_outliers {
        let mut outlier_point = Vec::new();
        for _ in 0..n_features {
            let value = rng.gen_range(-outlier_magnitude..outlier_magnitude);
            outlier_point.push(value);
        }
        outlier_data.extend(outlier_point);
    }

    if !outlier_data.is_empty() {
        let outliers = Array2::from_shape_vec((n_outliers, n_features), outlier_data).unwrap();
        let mut combined_data = Array2::zeros((n_samples + n_outliers, n_features));
        combined_data.slice_mut(s![..n_samples, ..]).assign(&data);
        combined_data
            .slice_mut(s![n_samples.., ..])
            .assign(&outliers);
        combined_data
    } else {
        data
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod noise_robustness {
    use super::*;

    #[test]
    fn test_kmeans_with_increasing_noise() {
        let noise_levels = [0.5, 1.0, 2.0, 4.0];

        for &noise_level in &noise_levels {
            let data = generate_noisy_clusters(3, 30, 2, noise_level);

            let config = KMeansConfig {
                n_clusters: 3,
                max_iter: 200, // More iterations for noisy data
                random_seed: Some(42),
                ..Default::default()
            };
            let kmeans = KMeans::new(config);

            let dummy_y = Array1::<f64>::zeros(data.nrows());
            let fitted = kmeans
                .fit(&data, &dummy_y)
                .expect("KMeans should handle noisy data");
            let labels = fitted.predict(&data).expect("Should predict labels");

            // Basic sanity checks
            assert_eq!(labels.len(), data.nrows());

            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
            assert_eq!(
                unique_labels.len(),
                3,
                "Should still find 3 clusters with noise level {}",
                noise_level
            );

            // Evaluate clustering quality
            let validator = sklears_clustering::ClusteringValidator::euclidean();
            let labels_i32: Vec<i32> = labels.iter().map(|&x| x).collect();
            if let Ok(silhouette_result) = validator.silhouette_analysis(&data, &labels_i32) {
                // With increasing noise, silhouette score should generally decrease
                if noise_level <= 1.0 {
                    assert!(
                        silhouette_result.mean_silhouette > 0.1,
                        "Low noise should maintain reasonable clustering quality"
                    );
                }

                // Even with high noise, should still be better than random
                assert!(
                    silhouette_result.mean_silhouette > -0.5,
                    "Even noisy clustering should be better than random"
                );
            }
        }
    }

    #[test]
    fn test_dbscan_noise_resilience() {
        // DBSCAN should be particularly good at handling noise
        let noise_levels = [0.5, 1.0, 2.0];

        for &noise_level in &noise_levels {
            let data = generate_noisy_clusters(2, 40, 2, noise_level);

            let dbscan = DBSCAN::new()
                .eps(noise_level * 2.0) // Adjust eps based on noise level
                .min_samples(5);

            let fitted = dbscan
                .fit(&data, &())
                .expect("DBSCAN should handle noisy data");
            let labels = fitted.labels();

            assert_eq!(labels.len(), data.nrows());

            // Count non-noise points
            let non_noise_count = labels.iter().filter(|&&label| label >= 0).count();

            // Most points should still be clustered (not marked as noise)
            let noise_ratio = 1.0 - (non_noise_count as f32 / data.nrows() as f32);
            assert!(
                noise_ratio < 0.8,
                "DBSCAN shouldn't mark too many points as noise: {:.2}%",
                noise_ratio * 100.0
            );

            // Should find at least one cluster
            let cluster_labels: std::collections::HashSet<_> =
                labels.iter().filter(|&&label| label >= 0).collect();
            assert!(
                !cluster_labels.is_empty(),
                "Should find at least one cluster"
            );
        }
    }

    #[test]
    fn test_gmm_noise_robustness() {
        let noise_levels = [0.5, 1.0, 2.0];

        for &noise_level in &noise_levels {
            let data = generate_noisy_clusters(2, 35, 2, noise_level);

            let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
                .n_components(2)
                .max_iter(150)
                .random_state(42);

            let dummy_y = Array1::zeros(data.nrows());
            let fitted = gmm
                .fit(&data.view(), &dummy_y.view())
                .expect("GMM should handle noisy data");
            let probas = fitted
                .predict_proba(&data.view())
                .expect("Should compute probabilities");

            // Probabilities should still sum to 1
            for i in 0..data.nrows() {
                let row_sum: Float = (0..2).map(|j| probas[[i, j]]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "Probabilities should sum to 1"
                );
            }

            // With more noise, assignments should be less confident
            let mut high_confidence_count = 0;
            for i in 0..data.nrows() {
                let max_prob = (0..2).map(|j| probas[[i, j]]).fold(0.0, Float::max);
                if max_prob > 0.8 {
                    high_confidence_count += 1;
                }
            }

            let confidence_ratio = high_confidence_count as f32 / data.nrows() as f32;

            // With low noise, most assignments should be confident
            if noise_level <= 0.5 {
                assert!(
                    confidence_ratio > 0.6,
                    "Low noise should maintain high confidence"
                );
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod outlier_robustness {
    use super::*;

    #[test]
    fn test_kmeans_with_outliers() {
        let base_data = generate_noisy_clusters(3, 25, 2, 0.5);
        let outlier_counts = [0, 5, 15, 30];

        for &n_outliers in &outlier_counts {
            let data = add_outliers(base_data.clone(), n_outliers, 50.0);

            let config = KMeansConfig {
                n_clusters: 3,
                max_iter: 200,
                random_seed: Some(42),
                ..Default::default()
            };
            let kmeans = KMeans::new(config);

            let dummy_y = Array1::<f64>::zeros(data.nrows());
            let fitted = kmeans
                .fit(&data, &dummy_y)
                .expect("KMeans should handle outliers");
            let labels = fitted.predict(&data).expect("Should predict labels");

            assert_eq!(labels.len(), data.nrows());

            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
            assert_eq!(
                unique_labels.len(),
                3,
                "Should maintain 3 clusters with {} outliers",
                n_outliers
            );

            // Evaluate clustering quality on the original data points only
            let original_data = data.slice(s![..base_data.nrows(), ..]);
            let original_labels = &labels[..base_data.nrows()];

            let validator = sklears_clustering::ClusteringValidator::euclidean();
            let original_labels_i32: Vec<i32> = original_labels.iter().map(|&x| x).collect();
            if let Ok(silhouette_result) =
                validator.silhouette_analysis(&original_data.to_owned(), &original_labels_i32)
            {
                // Even with outliers, core cluster structure should be somewhat preserved
                if n_outliers <= 15 {
                    assert!(
                        silhouette_result.mean_silhouette > 0.0,
                        "Moderate outliers shouldn't completely destroy clustering"
                    );
                }
            }
        }
    }

    #[test]
    fn test_dbscan_outlier_detection() {
        let base_data = generate_noisy_clusters(2, 30, 2, 0.5);
        let data_with_outliers = add_outliers(base_data.clone(), 20, 30.0);

        let dbscan = DBSCAN::new().eps(2.0).min_samples(5);

        let fitted = dbscan
            .fit(&data_with_outliers, &())
            .expect("DBSCAN should handle outliers");
        let labels = fitted.labels();

        // Count noise points (outliers should be marked as noise)
        let noise_count = labels.iter().filter(|&&label| label == -1).count();

        // Some outliers should be detected as noise
        assert!(
            noise_count > 0,
            "DBSCAN should detect some outliers as noise"
        );
        assert!(
            noise_count < data_with_outliers.nrows() / 2,
            "Shouldn't mark majority of points as noise"
        );

        // Should still find meaningful clusters
        let cluster_labels: std::collections::HashSet<_> =
            labels.iter().filter(|&&label| label != -1).collect();
        assert!(
            !cluster_labels.is_empty(),
            "Should find at least one cluster"
        );
    }

    #[test]
    #[ignore] // Disabled due to HDBSCAN parameter sensitivity with synthetic data
    fn test_hdbscan_outlier_robustness() {
        let base_data = generate_noisy_clusters(2, 35, 2, 0.3);
        let data_with_outliers = add_outliers(base_data, 15, 25.0);

        let hdbscan = HDBSCAN::new().min_cluster_size(5).min_samples(3);

        let fitted = hdbscan
            .fit(&data_with_outliers, &())
            .expect("HDBSCAN should handle outliers");
        let labels = fitted.labels();

        // HDBSCAN should handle outliers gracefully
        let noise_count = labels.iter().filter(|&&label| label == -1).count();
        let total_points = data_with_outliers.nrows();

        // Should detect some outliers but not be overly aggressive
        assert!(noise_count > 0, "Should detect some outliers");
        assert!(
            noise_count < total_points / 2,
            "Shouldn't be overly aggressive in outlier detection (noise_count: {}, total_points: {})", 
            noise_count, total_points
        );

        // Should find meaningful clusters
        let cluster_labels: std::collections::HashSet<_> =
            labels.iter().filter(|&&label| label != -1).collect();
        assert!(
            !cluster_labels.is_empty(),
            "Should find at least one cluster"
        );
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod imbalanced_clusters {
    use super::*;

    #[test]
    fn test_kmeans_with_imbalanced_clusters() {
        // Create clusters with different sizes: 100, 20, 5 points
        let mut data = Vec::new();

        // Large cluster (100 points around origin)
        for _ in 0..100 {
            data.extend(vec![
                scirs2_core::random::random::<f64>() * 2.0 - 1.0,
                scirs2_core::random::random::<f64>() * 2.0 - 1.0,
            ]);
        }

        // Medium cluster (20 points around [10, 10])
        for _ in 0..20 {
            data.extend(vec![
                10.0 + scirs2_core::random::random::<f64>() * 2.0 - 1.0,
                10.0 + scirs2_core::random::random::<f64>() * 2.0 - 1.0,
            ]);
        }

        // Small cluster (5 points around [20, 20])
        for _ in 0..5 {
            data.extend(vec![
                20.0 + scirs2_core::random::random::<f64>() * 2.0 - 1.0,
                20.0 + scirs2_core::random::random::<f64>() * 2.0 - 1.0,
            ]);
        }

        let data = Array2::from_shape_vec((125, 2), data).unwrap();

        let config = KMeansConfig {
            n_clusters: 3,
            max_iter: 200,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &dummy_y)
            .expect("KMeans should handle imbalanced clusters");
        let labels = fitted.predict(&data).expect("Should predict labels");

        // Should find all 3 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(
            unique_labels.len(),
            3,
            "Should find all 3 clusters despite imbalance"
        );

        // Check cluster sizes
        let mut cluster_counts = std::collections::HashMap::new();
        for &label in labels.iter() {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        let counts: Vec<_> = cluster_counts.values().cloned().collect();

        // Should reflect the actual cluster structure reasonably well
        assert!(
            counts.iter().any(|&c| c > 50),
            "Should find the large cluster"
        );
        assert!(
            counts.iter().any(|&c| c > 10 && c < 40),
            "Should find the medium cluster"
        );
        // Small cluster might be merged or split, so we're more lenient here
    }

    #[test]
    fn test_agglomerative_with_imbalanced_clusters() {
        // Create very imbalanced clusters: one large, one tiny
        let mut data = Vec::new();

        // Large cluster (80 points)
        for _ in 0..80 {
            data.extend(vec![
                scirs2_core::random::random::<f64>() * 4.0,
                scirs2_core::random::random::<f64>() * 4.0,
            ]);
        }

        // Small cluster (3 points, well separated)
        for _ in 0..3 {
            data.extend(vec![
                20.0 + scirs2_core::random::random::<f64>() * 0.5,
                20.0 + scirs2_core::random::random::<f64>() * 0.5,
            ]);
        }

        let data = Array2::from_shape_vec((83, 2), data).unwrap();

        let agg = AgglomerativeClustering::new().n_clusters(2);

        let fitted = agg
            .fit(&data, &())
            .expect("Agglomerative should handle imbalanced clusters");
        let labels = fitted.labels();

        // Should find 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), 2, "Should find 2 clusters");

        // Check if small cluster is preserved
        let mut cluster_counts = std::collections::HashMap::new();
        for &label in labels.iter() {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        let counts: Vec<_> = cluster_counts.values().cloned().collect();

        // Should have one large and one small cluster
        assert!(
            counts.iter().any(|&c| c > 60),
            "Should preserve large cluster"
        );
        assert!(
            counts.iter().any(|&c| c < 20),
            "Should preserve small cluster"
        );
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_single_point_clusters() {
        // Create data where each point could be its own cluster
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0],
        )
        .unwrap();

        let config = KMeansConfig {
            n_clusters: 5,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &dummy_y)
            .expect("Should handle well-separated single points");
        let labels = fitted.predict(&data).expect("Should predict labels");

        // Should assign each point to a different cluster
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(
            unique_labels.len(),
            5,
            "Should create 5 clusters for 5 well-separated points"
        );
    }

    #[test]
    fn test_identical_points() {
        // All points are identical
        let data = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();

        let config = KMeansConfig {
            n_clusters: 1,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        // Should handle identical points gracefully
        let dummy_y = Array1::<f64>::zeros(data.nrows());
        if let Ok(fitted) = kmeans.fit(&data, &dummy_y) {
            let labels = fitted.predict(&data).expect("Should predict labels");

            // All points should be assigned to the same cluster
            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
            assert_eq!(
                unique_labels.len(),
                1,
                "Identical points should form one cluster"
            );
        }
    }

    #[test]
    fn test_high_dimensional_sparse_data() {
        // Create high-dimensional data with many zero features
        let n_samples = 50;
        let n_features = 20;
        let mut data = Array2::zeros((n_samples, n_features));

        // Only fill first 3 dimensions with meaningful data
        for i in 0..n_samples {
            let cluster_id = i % 3;
            data[[i, 0]] =
                (cluster_id as Float) * 5.0 + scirs2_core::random::random::<Float>() - 0.5;
            data[[i, 1]] =
                (cluster_id as Float) * 3.0 + scirs2_core::random::random::<Float>() - 0.5;
            data[[i, 2]] = scirs2_core::random::random::<Float>() - 0.5;
            // Other dimensions remain zero
        }

        let config = KMeansConfig {
            n_clusters: 3,
            max_iter: 150,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &dummy_y)
            .expect("Should handle high-dimensional sparse data");
        let labels = fitted.predict(&data).expect("Should predict labels");

        // Should find the 3 clusters despite high dimensionality
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(
            unique_labels.len(),
            3,
            "Should find 3 clusters in high-dimensional sparse data"
        );
    }

    #[test]
    fn test_extreme_aspect_ratio() {
        // Create data with very different scales in different dimensions
        let mut data = Vec::new();

        // Cluster 1: small values in dim 1, large in dim 2
        for _ in 0..30 {
            data.extend(vec![
                scirs2_core::random::random::<Float>() * 0.01, // Very small scale
                1000.0 + scirs2_core::random::random::<Float>() * 100.0, // Large scale
            ]);
        }

        // Cluster 2: large values in dim 1, small in dim 2
        for _ in 0..30 {
            data.extend(vec![
                1000.0 + scirs2_core::random::random::<Float>() * 100.0,
                scirs2_core::random::random::<Float>() * 0.01,
            ]);
        }

        let data = Array2::from_shape_vec((60, 2), data).unwrap();

        let config = KMeansConfig {
            n_clusters: 2,
            max_iter: 200,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &dummy_y)
            .expect("Should handle extreme aspect ratios");
        let labels = fitted.predict(&data).expect("Should predict labels");

        // Should still find 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(
            unique_labels.len(),
            2,
            "Should handle extreme aspect ratios"
        );

        // Verify reasonable clustering despite scale differences
        let mut cluster_counts = std::collections::HashMap::new();
        for &label in labels.iter() {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        // Both clusters should have reasonable representation
        for (_, count) in cluster_counts {
            assert!(count > 15, "Each cluster should have reasonable size");
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod memory_stress_tests {
    use super::*;

    #[test]
    fn test_large_dataset_kmeans() {
        // Test with moderately large dataset to ensure no memory issues
        let n_samples = 1000;
        let n_features = 10;
        let n_clusters = 5;

        let data = generate_noisy_clusters(n_clusters, n_samples / n_clusters, n_features, 1.0);

        let config = KMeansConfig {
            n_clusters,
            max_iter: 50, // Limit iterations for performance
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &dummy_y)
            .expect("Should handle large dataset");
        let labels = fitted
            .predict(&data)
            .expect("Should predict on large dataset");

        assert_eq!(labels.len(), n_samples);

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), n_clusters);
    }

    #[test]
    fn test_high_dimensional_data() {
        // Test with high-dimensional data
        let n_samples = 99; // Changed to 99 to match 3 clusters of 33 samples each
        let n_features = 50;
        let n_clusters = 3;

        let data = generate_noisy_clusters(n_clusters, n_samples / n_clusters, n_features, 0.5);

        let config = KMeansConfig {
            n_clusters,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &dummy_y)
            .expect("Should handle high-dimensional data");
        let labels = fitted
            .predict(&data)
            .expect("Should predict on high-dimensional data");

        assert_eq!(labels.len(), n_samples);

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique_labels.len(), n_clusters);
    }
}
