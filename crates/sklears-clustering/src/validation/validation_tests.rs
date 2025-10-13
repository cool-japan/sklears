//! Comprehensive Test Suite for Clustering Validation
//!
//! This module provides comprehensive integration tests for all clustering validation
//! components, ensuring that the validation framework works correctly across all
//! metrics and use cases.
//!
//! # Test Categories
//! - Integration tests across all validation modules
//! - Property-based tests for validation metrics
//! - Performance and stress tests
//! - Edge case and error handling tests
//! - Cross-metric consistency tests
//! - Real-world scenario tests

use super::*;
use scirs2_core::ndarray::{array, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{thread_rng, Distribution, Random, Rng, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Generate well-separated clusters for testing
pub fn generate_well_separated_clusters() -> (Array2<f64>, Vec<i32>) {
    let data = array![
        [1.0, 1.0], // Cluster 0
        [1.1, 1.1], // Cluster 0
        [1.2, 0.9], // Cluster 0
        [0.9, 1.0], // Cluster 0
        [5.0, 5.0], // Cluster 1
        [5.1, 5.1], // Cluster 1
        [4.9, 5.0], // Cluster 1
        [5.0, 4.9], // Cluster 1
        [9.0, 1.0], // Cluster 2
        [9.1, 1.1], // Cluster 2
        [8.9, 0.9], // Cluster 2
        [9.0, 0.8], // Cluster 2
    ];
    let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    (data, labels)
}

/// Generate overlapping clusters for testing
pub fn generate_overlapping_clusters() -> (Array2<f64>, Vec<i32>) {
    let data = array![
        [1.0, 1.0], // Cluster 0
        [1.5, 1.5], // Cluster 0
        [2.0, 2.0], // Cluster 0 (overlapping)
        [2.5, 2.5], // Cluster 1 (overlapping)
        [3.0, 3.0], // Cluster 1
        [3.5, 3.5], // Cluster 1
        [2.2, 2.8], // Cluster 0 (in overlap region)
        [2.8, 2.2], // Cluster 1 (in overlap region)
    ];
    let labels = vec![0, 0, 0, 1, 1, 1, 0, 1];
    (data, labels)
}

/// Generate elongated clusters for testing
pub fn generate_elongated_clusters() -> (Array2<f64>, Vec<i32>) {
    let data = array![
        [0.0, 0.0], // Cluster 0 (horizontal line)
        [1.0, 0.1], // Cluster 0
        [2.0, 0.0], // Cluster 0
        [3.0, 0.1], // Cluster 0
        [0.0, 5.0], // Cluster 1 (vertical line)
        [0.1, 6.0], // Cluster 1
        [0.0, 7.0], // Cluster 1
        [0.1, 8.0], // Cluster 1
    ];
    let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];
    (data, labels)
}

/// Generate random clusters for stress testing
pub fn generate_random_clusters(
    n_samples: usize,
    n_clusters: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut data = Array2::<f64>::zeros((n_samples, n_features));
    let mut labels = vec![0; n_samples];

    // Generate cluster centers
    let mut centers = Vec::new();
    for _ in 0..n_clusters {
        let mut center = vec![0.0; n_features];
        for j in 0..n_features {
            center[j] = rng.gen_range(-10.0..10.0);
        }
        centers.push(center);
    }

    // Assign points to clusters and add noise
    for i in 0..n_samples {
        let cluster_id = i % n_clusters;
        labels[i] = cluster_id as i32;

        for j in 0..n_features {
            let noise = <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng);
            data[[i, j]] = centers[cluster_id][j] + noise;
        }
    }

    (data, labels)
}

/// Generate clusters with noise points
pub fn generate_clusters_with_noise() -> (Array2<f64>, Vec<i32>) {
    let data = array![
        [1.0, 1.0],   // Cluster 0
        [1.1, 1.1],   // Cluster 0
        [1.2, 0.9],   // Cluster 0
        [5.0, 5.0],   // Cluster 1
        [5.1, 5.1],   // Cluster 1
        [4.9, 5.0],   // Cluster 1
        [10.0, 10.0], // Noise
        [-5.0, -5.0], // Noise
        [0.0, 10.0],  // Noise
    ];
    let labels = vec![0, 0, 0, 1, 1, 1, -1, -1, -1];
    (data, labels)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test for all validation metrics on well-separated clusters
    #[test]
    fn test_comprehensive_validation_well_separated() {
        let (data, labels) = generate_well_separated_clusters();
        let validator = automated_validation::AutomatedValidator::euclidean();
        let config = automated_validation::AutomatedValidationConfig::fast();

        let result = validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();

        // Well-separated clusters should have good scores
        assert!(result.silhouette_score > 0.5);
        assert!(result.calinski_harabasz_score > 5.0);
        assert!(result.davies_bouldin_score < 2.0);
        assert!(result.coherence_score > 0.5);
        assert!(result.separation_score > 0.0);

        // Overall quality should be at least Fair
        assert!(matches!(
            result.overall_quality,
            automated_validation::ClusterQuality::Excellent
                | automated_validation::ClusterQuality::Good
                | automated_validation::ClusterQuality::Fair
        ));

        assert!(!result.validation_summary.is_empty());
    }

    /// Integration test for overlapping clusters
    #[test]
    fn test_comprehensive_validation_overlapping() {
        let (data, labels) = generate_overlapping_clusters();
        let validator = automated_validation::AutomatedValidator::euclidean();
        let config = automated_validation::AutomatedValidationConfig::fast();

        let result = validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();

        // Overlapping clusters should have lower scores
        assert!(result.silhouette_score >= -1.0 && result.silhouette_score <= 1.0);
        assert!(result.calinski_harabasz_score >= 0.0);
        assert!(result.davies_bouldin_score >= 0.0);
        assert!(result.coherence_score >= 0.0 && result.coherence_score <= 1.0);

        // Should have some recommendations for overlapping clusters
        assert!(!result.recommendations.is_empty());
    }

    /// Test all validation metrics independently
    #[test]
    fn test_individual_validation_metrics() {
        let (data, labels) = generate_well_separated_clusters();

        // Test internal validation
        let internal_validator = ClusteringValidator::euclidean();
        let silhouette = internal_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let ch_index = internal_validator
            .calinski_harabasz_index(&data, &labels)
            .unwrap();
        let db_index = internal_validator
            .davies_bouldin_index(&data, &labels)
            .unwrap();
        let dunn_index = internal_validator.dunn_index(&data, &labels).unwrap();

        assert!(silhouette.mean_silhouette > 0.0);
        assert!(ch_index > 0.0);
        assert!(db_index > 0.0);
        assert!(dunn_index > 0.0);

        // Test coherence and separation
        let cs_analyzer = coherence_separation::CoherenceSeparationAnalyzer::euclidean();
        let coherence = cs_analyzer.cluster_coherence(&data, &labels).unwrap();
        let separation = cs_analyzer.cluster_separation(&data, &labels).unwrap();

        assert!(coherence.overall_coherence > 0.0);
        assert!(separation.avg_centroid_separation > 0.0);
        assert!(separation.gap_ratio > 0.0);

        // Test external validation (need ground truth)
        let external_validator = ClusteringValidator::euclidean();
        let ari = external_validator
            .internal_validator
            .adjusted_rand_index(&labels, &labels)
            .unwrap();
        let nmi = external_validator
            .internal_validator
            .normalized_mutual_information(&labels, &labels)
            .unwrap();

        assert!((ari - 1.0).abs() < 1e-10); // Perfect match with itself
        assert!((nmi - 1.0).abs() < 1e-10); // Perfect match with itself
    }

    /// Test gap statistic for optimal cluster selection
    #[test]
    fn test_gap_statistic_comprehensive() {
        let (data, _) = generate_well_separated_clusters();
        let gap_analyzer = ClusteringValidator::euclidean();

        // Simple clustering function for testing
        let simple_clustering = |data: &Array2<f64>, k: usize| -> Result<Vec<i32>> {
            // Mock clustering that assigns points cyclically to k clusters
            let labels: Vec<i32> = (0..data.nrows()).map(|i| (i % k) as i32).collect();
            Ok(labels)
        };

        let result = gap_analyzer
            .gap_statistic(&data, 1..6, Some(10), simple_clustering)
            .unwrap();

        assert_eq!(result.k_values, vec![1, 2, 3, 4, 5]);
        assert_eq!(result.gap_values.len(), 5);
        assert_eq!(result.gap_std_errors.len(), 5);
        assert!(result.optimal_k >= 1 && result.optimal_k <= 5);

        // Test gap result methods
        assert!(result.gap_for_k(2).is_some());
        assert!(result.gap_for_k(10).is_none());

        let recommendations = result.recommended_k_values(3);
        assert!(recommendations.len() <= 3);
    }

    /// Test stability analysis methods
    #[test]
    fn test_stability_analysis_comprehensive() {
        let (data, _) = generate_well_separated_clusters();
        let stability_analyzer = stability_analysis::StabilityAnalyzer::euclidean();

        // Simple clustering function for testing
        let simple_clustering = |data: &Array2<f64>| -> Result<Vec<i32>> {
            let labels: Vec<i32> = data
                .axis_iter(scirs2_core::ndarray::Axis(0))
                .map(|row| {
                    if row[0] < 3.0 {
                        0
                    } else if row[0] < 7.0 {
                        1
                    } else {
                        2
                    }
                })
                .collect();
            Ok(labels)
        };

        // Test subsample stability
        let subsample_result = stability_analyzer
            .subsample_stability(&data, simple_clustering, 0.7, 5)
            .unwrap();

        assert!(subsample_result.mean_stability >= 0.0 && subsample_result.mean_stability <= 1.0);
        assert!(subsample_result.std_stability >= 0.0);
        assert!(subsample_result.n_successful_trials >= 2);

        // Test consensus stability
        let seeded_clustering =
            |data: &Array2<f64>, _seed: u64| -> Result<Vec<i32>> { simple_clustering(data) };

        let consensus_result = stability_analyzer
            .consensus_stability(seeded_clustering, &data, 5, None)
            .unwrap();

        assert!(consensus_result.mean_stability >= 0.0 && consensus_result.mean_stability <= 1.0);
        assert!(consensus_result.n_successful_runs >= 2);
        assert_eq!(consensus_result.consensus_matrix.nrows(), data.nrows());

        // Test cross-validation stability
        let cv_result = stability_analyzer
            .cross_validation_stability(simple_clustering, &data, 3, 2)
            .unwrap();

        assert!(cv_result.mean_stability >= 0.0 && cv_result.mean_stability <= 1.0);
        assert_eq!(cv_result.k_folds, 3);
        assert_eq!(cv_result.n_repeats, 2);
    }

    /// Test validation with different distance metrics
    #[test]
    fn test_validation_different_metrics() {
        let (data, labels) = generate_well_separated_clusters();
        let config = automated_validation::AutomatedValidationConfig::fast();

        let euclidean_validator = automated_validation::AutomatedValidator::euclidean();
        let manhattan_validator = automated_validation::AutomatedValidator::manhattan();
        let cosine_validator = automated_validation::AutomatedValidator::cosine();

        let euc_result = euclidean_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();
        let man_result = manhattan_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();
        let cos_result = cosine_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();

        // All metrics should produce valid results
        assert!(euc_result.internal_quality_score >= 0.0);
        assert!(man_result.internal_quality_score >= 0.0);
        assert!(cos_result.internal_quality_score >= 0.0);

        // Results should be different for different metrics
        assert_ne!(euc_result.silhouette_score, man_result.silhouette_score);
        assert_ne!(euc_result.silhouette_score, cos_result.silhouette_score);
    }

    /// Test validation with noise points
    #[test]
    fn test_validation_with_noise() {
        let (data, labels) = generate_clusters_with_noise();
        let internal_validator = ClusteringValidator::euclidean();

        let silhouette = internal_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();

        // Should handle noise points (label -1) correctly
        assert_eq!(silhouette.sample_silhouettes.len(), data.nrows());

        // Noise points should have silhouette score of 0
        for (i, &label) in labels.iter().enumerate() {
            if label == -1 {
                assert_eq!(silhouette.sample_silhouettes[i], 0.0);
            }
        }
    }

    /// Property-based test: silhouette scores should be in [-1, 1]
    #[test]
    fn test_silhouette_bounds_property() {
        for seed in 0..10 {
            let (data, labels) = generate_random_clusters(50, 5, 3, seed);
            let internal_validator = ClusteringValidator::euclidean();

            if let Ok(silhouette) = internal_validator.silhouette_analysis(&data, &labels) {
                assert!(silhouette.mean_silhouette >= -1.0 && silhouette.mean_silhouette <= 1.0);
                for &score in &silhouette.sample_silhouettes {
                    assert!(score >= -1.0 && score <= 1.0);
                }
            }
        }
    }

    /// Property-based test: validation metrics should be consistent
    #[test]
    fn test_metric_consistency_property() {
        for seed in 0..5 {
            let (data, labels) = generate_random_clusters(30, 3, 2, seed);
            let internal_validator = ClusteringValidator::euclidean();

            if let (Ok(silhouette), Ok(ch_index), Ok(db_index)) = (
                internal_validator.silhouette_analysis(&data, &labels),
                internal_validator.calinski_harabasz_index(&data, &labels),
                internal_validator.davies_bouldin_index(&data, &labels),
            ) {
                // Basic sanity checks
                assert!(ch_index >= 0.0);
                assert!(db_index >= 0.0);
                assert!(silhouette.mean_silhouette >= -1.0 && silhouette.mean_silhouette <= 1.0);

                // Higher CH index should generally correlate with better clustering
                // (This is a weak property test)
                if ch_index > 10.0 {
                    assert!(silhouette.mean_silhouette > -0.5);
                }
            }
        }
    }

    /// Test validation configuration presets
    #[test]
    fn test_validation_config_presets() {
        let fast = automated_validation::AutomatedValidationConfig::fast();
        assert!(!fast.include_stability);
        assert_eq!(fast.internal_weight, 1.0);

        let comprehensive = automated_validation::AutomatedValidationConfig::comprehensive();
        assert!(comprehensive.include_stability);
        assert_eq!(comprehensive.quality_threshold, 0.6);

        let production = automated_validation::AutomatedValidationConfig::production();
        assert!(production.include_stability);
        assert_eq!(production.quality_threshold, 0.7);

        // All presets should be valid
        assert!(fast.validate().is_ok());
        assert!(comprehensive.validate().is_ok());
        assert!(production.validate().is_ok());
    }

    /// Test error handling across validation modules
    #[test]
    fn test_validation_error_handling() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let wrong_labels = vec![0]; // Wrong length
        let empty_labels: Vec<i32> = vec![];
        let single_cluster_labels = vec![0, 0];
        let noise_only_labels = vec![-1, -1];

        let internal_validator = ClusteringValidator::euclidean();
        let cs_analyzer = coherence_separation::CoherenceSeparationAnalyzer::euclidean();
        let automated_validator = automated_validation::AutomatedValidator::euclidean();
        let config = automated_validation::AutomatedValidationConfig::default();

        // Test mismatched dimensions
        assert!(internal_validator
            .silhouette_analysis(&data, &wrong_labels)
            .is_err());
        assert!(cs_analyzer.cluster_coherence(&data, &wrong_labels).is_err());
        assert!(automated_validator
            .automated_cluster_validation(&data, &wrong_labels, &config)
            .is_err());

        // Test insufficient clusters
        assert!(internal_validator
            .silhouette_analysis(&data, &single_cluster_labels)
            .is_err());
        assert!(cs_analyzer
            .cluster_separation(&data, &single_cluster_labels)
            .is_err());

        // Test noise-only labels
        assert!(internal_validator
            .silhouette_analysis(&data, &noise_only_labels)
            .is_err());
        assert!(cs_analyzer
            .cluster_coherence(&data, &noise_only_labels)
            .is_err());
    }

    /// Performance test for large datasets
    #[test]
    fn test_validation_performance() {
        let (data, labels) = generate_random_clusters(200, 5, 4, 42);
        let internal_validator = ClusteringValidator::euclidean();

        // Should complete within reasonable time
        let start = std::time::Instant::now();
        let silhouette = internal_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let duration = start.elapsed();

        assert!(duration.as_secs() < 5); // Should complete within 5 seconds
        assert_eq!(silhouette.sample_silhouettes.len(), 200);
    }

    /// Test validation with elongated clusters
    #[test]
    fn test_validation_elongated_clusters() {
        let (data, labels) = generate_elongated_clusters();
        let cs_analyzer = coherence_separation::CoherenceSeparationAnalyzer::euclidean();

        let coherence = cs_analyzer.cluster_coherence(&data, &labels).unwrap();
        let separation = cs_analyzer.cluster_separation(&data, &labels).unwrap();

        // Elongated clusters should have lower shape regularity
        assert!(coherence.overall_shape_regularity < 0.9);

        // But should still have reasonable separation
        assert!(separation.avg_centroid_separation > 0.0);
        assert!(separation.gap_ratio > 1.0);
    }

    /// Cross-validation test: results should be stable
    #[test]
    fn test_validation_stability() {
        let (data, labels) = generate_well_separated_clusters();
        let internal_validator = ClusteringValidator::euclidean();

        // Run validation multiple times - results should be consistent
        let mut silhouette_scores = Vec::new();
        let mut ch_scores = Vec::new();

        for _ in 0..5 {
            let silhouette = internal_validator
                .silhouette_analysis(&data, &labels)
                .unwrap();
            let ch_index = internal_validator
                .calinski_harabasz_index(&data, &labels)
                .unwrap();

            silhouette_scores.push(silhouette.mean_silhouette);
            ch_scores.push(ch_index);
        }

        // All results should be identical (deterministic)
        for score in &silhouette_scores[1..] {
            assert!((score - silhouette_scores[0]).abs() < 1e-10);
        }

        for score in &ch_scores[1..] {
            assert!((score - ch_scores[0]).abs() < 1e-10);
        }
    }

    /// Test comprehensive validation pipeline
    #[test]
    fn test_end_to_end_validation_pipeline() {
        let (data, labels) = generate_well_separated_clusters();

        // Step 1: Basic internal validation
        let internal_validator = ClusteringValidator::euclidean();
        let silhouette = internal_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        assert!(silhouette.mean_silhouette > 0.5);

        // Step 2: Coherence and separation analysis
        let cs_analyzer = coherence_separation::CoherenceSeparationAnalyzer::euclidean();
        let coherence = cs_analyzer.cluster_coherence(&data, &labels).unwrap();
        let separation = cs_analyzer.cluster_separation(&data, &labels).unwrap();
        assert!(coherence.overall_coherence > 0.5);
        assert!(separation.gap_ratio > 1.0);

        // Step 3: External validation (with perfect ground truth)
        let external_validator = ClusteringValidator::euclidean();
        let ari = external_validator
            .internal_validator
            .adjusted_rand_index(&labels, &labels)
            .unwrap();
        assert!((ari - 1.0).abs() < 1e-10);

        // Step 4: Automated comprehensive validation
        let automated_validator = automated_validation::AutomatedValidator::euclidean();
        let config = automated_validation::AutomatedValidationConfig::fast();
        let result = automated_validator
            .automated_cluster_validation(&data, &labels, &config)
            .unwrap();

        assert!(matches!(
            result.overall_quality,
            automated_validation::ClusterQuality::Excellent
                | automated_validation::ClusterQuality::Good
                | automated_validation::ClusterQuality::Fair
        ));

        // Step 5: Verify recommendations are reasonable
        if result.overall_quality == automated_validation::ClusterQuality::Poor {
            assert!(!result.recommendations.is_empty());
        }

        // Step 6: Generate detailed report
        let report = result.detailed_report();
        assert!(report.contains("Automated Cluster Validation Report"));
        assert!(report.contains("Overall Quality"));
    }

    /// Test validation types and utilities
    #[test]
    fn test_validation_types() {
        // Test ValidationMetric
        let euclidean = validation_types::ValidationMetric::Euclidean;
        let manhattan = validation_types::ValidationMetric::Manhattan;
        let cosine = validation_types::ValidationMetric::Cosine;
        let minkowski = validation_types::ValidationMetric::Minkowski(2.0);

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!(euclidean.compute_distance(&a, &b) > 0.0);
        assert!(manhattan.compute_distance(&a, &b) > 0.0);
        assert!(cosine.compute_distance(&a, &b) >= 0.0);
        assert!(minkowski.compute_distance(&a, &b) > 0.0);

        // Test metric properties
        assert!(!euclidean.is_high_dimensional_suitable());
        assert!(cosine.is_high_dimensional_suitable());
        assert!(euclidean.requires_scaling());
        assert!(!cosine.requires_scaling());

        // Test ValidationConfig presets
        let fast_config = validation_types::ValidationConfig::fast();
        let comprehensive_config = validation_types::ValidationConfig::comprehensive();
        let high_dim_config = validation_types::ValidationConfig::high_dimensional();

        assert!(!fast_config.compute_confidence_intervals);
        assert!(comprehensive_config.compute_confidence_intervals);
        assert_eq!(
            high_dim_config.metric,
            validation_types::ValidationMetric::Cosine
        );
    }

    /// Integration test with all modules working together
    #[test]
    fn test_full_integration() {
        let (data, labels) = generate_well_separated_clusters();

        // Create validators for all components
        let internal_validator = ClusteringValidator::euclidean();
        let external_validator = ClusteringValidator::euclidean();
        let cs_analyzer = coherence_separation::CoherenceSeparationAnalyzer::euclidean();
        let gap_analyzer = ClusteringValidator::euclidean();

        // Run all validation methods
        let silhouette = internal_validator
            .silhouette_analysis(&data, &labels)
            .unwrap();
        let ch_index = internal_validator
            .calinski_harabasz_index(&data, &labels)
            .unwrap();
        let db_index = internal_validator
            .davies_bouldin_index(&data, &labels)
            .unwrap();

        let ari = external_validator
            .internal_validator
            .adjusted_rand_index(&labels, &labels)
            .unwrap();
        let nmi = external_validator
            .internal_validator
            .normalized_mutual_information(&labels, &labels)
            .unwrap();

        let coherence = cs_analyzer.cluster_coherence(&data, &labels).unwrap();
        let separation = cs_analyzer.cluster_separation(&data, &labels).unwrap();

        // Simple clustering function for gap statistic
        let simple_clustering = |data: &Array2<f64>, k: usize| -> Result<Vec<i32>> {
            let labels: Vec<i32> = (0..data.nrows()).map(|i| (i % k) as i32).collect();
            Ok(labels)
        };

        let gap_result = gap_analyzer
            .gap_statistic(&data, 2..5, Some(5), simple_clustering)
            .unwrap();

        // Verify all results are reasonable
        assert!(silhouette.mean_silhouette > 0.0);
        assert!(ch_index > 0.0);
        assert!(db_index > 0.0);
        assert!((ari - 1.0).abs() < 1e-10); // Perfect self-match
        assert!((nmi - 1.0).abs() < 1e-10); // Perfect self-match
        assert!(coherence.overall_coherence > 0.0);
        assert!(separation.avg_centroid_separation > 0.0);
        assert!(gap_result.optimal_k >= 2 && gap_result.optimal_k < 5);

        // Cross-check consistency
        if silhouette.mean_silhouette > 0.7 {
            assert!(ch_index > 5.0); // High silhouette should correlate with high CH
            assert!(coherence.overall_coherence > 0.5); // And good coherence
        }

        if separation.gap_ratio > 2.0 {
            assert!(separation.overlap_measure < 0.5); // Good separation should mean low overlap
        }
    }
}
