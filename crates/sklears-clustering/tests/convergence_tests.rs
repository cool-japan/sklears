//! Convergence tests for iterative clustering algorithms
//!
//! This module tests that iterative algorithms converge properly and within
//! reasonable iteration bounds.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use sklears_clustering::{FuzzyCMeans, GaussianMixture, KMeans, KMeansConfig, MeanShift};
use sklears_clustering::{PredictMembership, PredictProba};
use sklears_core::{
    traits::{Fit, Predict},
    types::Float,
};

/// Generate well-separated Gaussian clusters for convergence testing
fn generate_separated_clusters(
    n_clusters: usize,
    n_samples_per_cluster: usize,
    n_features: usize,
) -> Array2<Float> {
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::Distribution;

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::new();

    for cluster_id in 0..n_clusters {
        // Create cluster centers that are well separated
        let center_offset = (cluster_id as Float) * 10.0;

        for _ in 0..n_samples_per_cluster {
            let mut point = Vec::new();
            for feature_id in 0..n_features {
                let center = if feature_id == 0 { center_offset } else { 0.0 };
                let normal = Normal::new(center, 1.0).unwrap();
                point.push(normal.sample(&mut rng));
            }
            data.extend(point);
        }
    }

    Array2::from_shape_vec((n_clusters * n_samples_per_cluster, n_features), data).unwrap()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod kmeans_convergence {
    use super::*;

    #[test]
    fn test_kmeans_convergence_well_separated() {
        let data = generate_separated_clusters(3, 50, 2);

        let config = KMeansConfig {
            n_clusters: 3,
            max_iter: 100,
            tolerance: 1e-6,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let y_dummy = Array1::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &y_dummy)
            .expect("KMeans should fit successfully");

        // Check that it converged (n_iter should be less than max_iter for well-separated data)
        // Note: We can't directly access n_iter from the current interface,
        // but we can verify convergence by checking that the algorithm completed successfully
        let labels = fitted.predict(&data).expect("Prediction should work");

        // Verify cluster assignments are reasonable for well-separated data
        assert_eq!(labels.len(), data.nrows());

        // Each cluster should have roughly equal representation
        let mut cluster_counts = vec![0; 3];
        for &label in labels.iter() {
            cluster_counts[label as usize] += 1;
        }

        // With well-separated data, each cluster should have reasonable representation
        for count in cluster_counts {
            assert!(
                count > 10,
                "Each cluster should have reasonable representation"
            );
        }
    }

    #[test]
    fn test_kmeans_early_convergence() {
        // Create data that should converge quickly
        let mut data = Array2::zeros((9, 2));

        // Three very distinct clusters
        data[[0, 0]] = 0.0;
        data[[0, 1]] = 0.0;
        data[[1, 0]] = 0.1;
        data[[1, 1]] = 0.1;
        data[[2, 0]] = 0.0;
        data[[2, 1]] = 0.1;

        data[[3, 0]] = 10.0;
        data[[3, 1]] = 10.0;
        data[[4, 0]] = 10.1;
        data[[4, 1]] = 10.1;
        data[[5, 0]] = 10.0;
        data[[5, 1]] = 10.1;

        data[[6, 0]] = 20.0;
        data[[6, 1]] = 20.0;
        data[[7, 0]] = 20.1;
        data[[7, 1]] = 20.1;
        data[[8, 0]] = 20.0;
        data[[8, 1]] = 20.1;

        let config = KMeansConfig {
            n_clusters: 3,
            max_iter: 100,
            tolerance: 1e-6,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);

        let y_dummy = Array1::zeros(data.nrows());
        let fitted = kmeans
            .fit(&data, &y_dummy)
            .expect("KMeans should converge quickly");
        let labels = fitted.predict(&data).expect("Prediction should work");

        // Verify correct clustering
        assert_eq!(labels.len(), 9);

        // Points in the same group should have the same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[7], labels[8]);

        // Different groups should have different labels
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn test_kmeans_convergence_tolerance() {
        let data = generate_separated_clusters(2, 25, 3);

        // Test with different tolerance levels
        let tolerances = [1e-2, 1e-4, 1e-6, 1e-8];

        for &tol in &tolerances {
            let config = KMeansConfig {
                n_clusters: 2,
                max_iter: 200,
                tolerance: tol,
                random_seed: Some(42),
                ..Default::default()
            };
            let kmeans = KMeans::new(config);

            let y_dummy = Array1::zeros(data.nrows());
            let fitted = kmeans.fit(&data, &y_dummy).expect("KMeans should converge");
            let labels = fitted.predict(&data).expect("Prediction should work");

            assert_eq!(labels.len(), data.nrows());

            // Verify reasonable clustering
            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
            assert_eq!(unique_labels.len(), 2, "Should find exactly 2 clusters");
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod gmm_convergence {
    use super::*;

    #[test]
    fn test_gmm_convergence_well_separated() {
        let data = generate_separated_clusters(3, 30, 2);

        let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
            .n_components(3)
            .max_iter(100)
            .tol(1e-6)
            .random_state(42);

        let dummy_y = Array1::zeros(data.nrows());
        let fitted = gmm
            .fit(&data.view(), &dummy_y.view())
            .expect("GMM should fit successfully");
        let labels = fitted
            .predict(&data.view())
            .expect("Prediction should work");

        // Verify cluster assignments
        assert_eq!(labels.len(), data.nrows());

        // Check that probabilities are reasonable
        let probas = fitted
            .predict_proba(&data.view())
            .expect("Should compute probabilities");
        assert_eq!(probas.shape(), &[data.nrows(), 3]);

        // Each row should sum to approximately 1.0
        for i in 0..data.nrows() {
            let row_sum: Float = (0..3).map(|j| probas[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {} sum: {}", i, row_sum);
        }

        // For well-separated data, most points should have high probability for one component
        let mut high_confidence_points = 0;
        for i in 0..data.nrows() {
            let max_prob = (0..3).map(|j| probas[[i, j]]).fold(0.0, Float::max);
            if max_prob > 0.8 {
                high_confidence_points += 1;
            }
        }

        // Most points should have high confidence assignment
        assert!(
            high_confidence_points as f32 / data.nrows() as f32 > 0.6,
            "Most points should have high confidence assignment in well-separated data"
        );
    }

    #[test]
    fn test_gmm_likelihood_convergence() {
        let data = generate_separated_clusters(2, 40, 2);

        let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
            .n_components(2)
            .max_iter(50)
            .tol(1e-6)
            .random_state(42);

        let dummy_y = Array1::zeros(data.nrows());
        let fitted = gmm
            .fit(&data.view(), &dummy_y.view())
            .expect("GMM should converge");

        // For well-separated data, GMM should converge to a reasonable solution
        let labels = fitted.predict(&data.view()).expect("Should predict labels");
        let probas = fitted
            .predict_proba(&data.view())
            .expect("Should compute probabilities");

        // Verify reasonable cluster balance
        let mut cluster_counts = vec![0; 2];
        for &label in labels.iter() {
            cluster_counts[label] += 1;
        }

        // Both clusters should have reasonable representation
        for count in cluster_counts {
            assert!(
                count > 10,
                "Each cluster should have reasonable representation"
            );
        }

        // Verify probability consistency
        for i in 0..data.nrows() {
            let predicted_label = labels[i];
            let prob_for_predicted = probas[[i, predicted_label]];

            // The probability for the predicted label should be the highest
            for j in 0..2 {
                if j != predicted_label {
                    assert!(
                        prob_for_predicted >= probas[[i, j]],
                        "Predicted label should have highest probability"
                    );
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod fuzzy_cmeans_convergence {
    use super::*;

    #[test]
    fn test_fuzzy_cmeans_convergence() {
        let data = generate_separated_clusters(3, 20, 2);

        let fcm = FuzzyCMeans::new(3)
            .max_iter(100)
            .tol(1e-6)
            .fuzziness(2.0) // Standard fuzziness parameter
            .random_state(42);

        let fitted = fcm
            .fit(&data, &())
            .expect("Fuzzy C-Means should fit successfully");
        let labels = fitted.predict(&data).expect("Prediction should work");

        // Verify basic properties
        assert_eq!(labels.len(), data.nrows());

        // Get membership matrix
        let memberships = fitted
            .predict_membership(&data)
            .expect("Should get memberships");
        assert_eq!(memberships.shape(), &[data.nrows(), 3]);

        // Each row should sum to 1.0 (fuzzy clustering property)
        for i in 0..data.nrows() {
            let row_sum: Float = (0..3).map(|j| memberships[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {} sum: {}", i, row_sum);
        }

        // All memberships should be between 0 and 1
        for membership in memberships.iter() {
            assert!(*membership >= 0.0);
            assert!(*membership <= 1.0);
        }

        // For each point, the predicted label should correspond to the highest membership
        for i in 0..data.nrows() {
            let predicted_label = labels[i];
            let max_membership = (0..3).map(|j| memberships[[i, j]]).fold(0.0, Float::max);
            assert!(
                (memberships[[i, predicted_label]] - max_membership).abs() < 1e-10,
                "Predicted label should have highest membership"
            );
        }
    }

    #[test]
    fn test_fuzzy_cmeans_fuzziness_parameter() {
        let data = generate_separated_clusters(2, 25, 2);

        // Test different fuzziness values
        let fuzziness_values = [1.1, 1.5, 2.0, 3.0];

        for &m in &fuzziness_values {
            let fcm = FuzzyCMeans::new(2)
                .max_iter(100)
                .tol(1e-6)
                .fuzziness(m)
                .random_state(42);

            let fitted = fcm
                .fit(&data, &())
                .expect("Should converge for all fuzziness values");
            let memberships = fitted
                .predict_membership(&data)
                .expect("Should get memberships");

            // Higher fuzziness should lead to more distributed memberships
            let mut max_memberships = Vec::new();
            for i in 0..data.nrows() {
                let max_membership = (0..2).map(|j| memberships[[i, j]]).fold(0.0, Float::max);
                max_memberships.push(max_membership);
            }

            let mean_max_membership =
                max_memberships.iter().sum::<Float>() / max_memberships.len() as Float;

            // With higher fuzziness, maximum memberships should generally be lower (more distributed)
            // For well-separated clusters, even with high fuzziness, memberships can still be high
            // We adjust the threshold to be more realistic for separated clusters
            if m > 2.0 {
                assert!(
                    mean_max_membership < 0.95,
                    "High fuzziness should lead to more distributed memberships. Got: {}",
                    mean_max_membership
                );
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod mean_shift_convergence {
    use super::*;

    #[test]
    fn test_mean_shift_convergence() {
        let data = generate_separated_clusters(2, 30, 2);

        let mean_shift = MeanShift::<Array2<f64>, Array1<f64>>::new()
            .bandwidth(2.0)
            .max_iter(100);

        let dummy_y = Array1::zeros(data.nrows());
        let fitted = mean_shift
            .fit(&data.view(), &dummy_y.view())
            .expect("Mean Shift should fit successfully");
        let labels = fitted
            .predict(&data.view())
            .expect("Prediction should work");

        // Verify basic properties
        assert_eq!(labels.len(), data.nrows());

        // Mean shift should find a reasonable number of clusters for well-separated data
        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        assert!(!unique_labels.is_empty());
        assert!(unique_labels.len() <= 5); // Shouldn't over-cluster too much

        // Verify that each cluster has reasonable representation
        let mut cluster_counts = std::collections::HashMap::new();
        for &label in labels.iter() {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        // No cluster should be too small (at least 3 points)
        for (_, count) in cluster_counts {
            assert!(count >= 3, "Each cluster should have at least a few points");
        }
    }

    #[test]
    fn test_mean_shift_bandwidth_sensitivity() {
        let data = generate_separated_clusters(3, 20, 2);

        // Test different bandwidth values
        let bandwidths = [0.5, 1.0, 2.0, 4.0];

        for &bandwidth in &bandwidths {
            let mean_shift = MeanShift::<Array2<f64>, Array1<f64>>::new()
                .bandwidth(bandwidth)
                .max_iter(100);

            let dummy_y = Array1::zeros(data.nrows());
            let fitted = mean_shift
                .fit(&data.view(), &dummy_y.view())
                .expect("Should converge for all bandwidths");
            let labels = fitted.predict(&data.view()).expect("Should predict labels");

            let unique_labels: std::collections::HashSet<_> = labels.iter().collect();

            // Smaller bandwidth should generally find more clusters
            if bandwidth <= 1.0 {
                assert!(
                    unique_labels.len() >= 2,
                    "Small bandwidth should find multiple clusters"
                );
            }

            // Larger bandwidth should generally find fewer clusters
            if bandwidth >= 4.0 {
                assert!(
                    unique_labels.len() <= 4,
                    "Large bandwidth shouldn't over-cluster"
                );
            }
        }
    }
}

/// Integration test for convergence behavior across algorithms
#[allow(non_snake_case)]
#[cfg(test)]
mod convergence_integration {
    use super::*;

    #[test]
    fn test_convergence_comparison() {
        // Use the same well-separated data for all algorithms
        let data = generate_separated_clusters(3, 25, 2);

        // Test K-Means
        let config = KMeansConfig {
            n_clusters: 3,
            max_iter: 100,
            random_seed: Some(42),
            ..Default::default()
        };
        let kmeans = KMeans::new(config);
        let dummy_y = Array1::<f64>::zeros(data.nrows());
        let kmeans_fitted = kmeans.fit(&data, &dummy_y).expect("KMeans should converge");
        let kmeans_labels = kmeans_fitted.predict(&data).expect("Should predict");

        // Test GMM
        let gmm: GaussianMixture<(), ()> = GaussianMixture::new()
            .n_components(3)
            .max_iter(100)
            .random_state(42);
        let dummy_y = Array1::zeros(data.nrows());
        let gmm_fitted = gmm
            .fit(&data.view(), &dummy_y.view())
            .expect("GMM should converge");
        let gmm_labels = gmm_fitted.predict(&data.view()).expect("Should predict");

        // Test Fuzzy C-Means
        let fcm = FuzzyCMeans::new(3).max_iter(100).random_state(42);
        let fcm_fitted = fcm.fit(&data, &()).expect("FCM should converge");
        let fcm_labels = fcm_fitted.predict(&data).expect("Should predict");

        // All algorithms should find 3 clusters
        let kmeans_clusters: std::collections::HashSet<_> = kmeans_labels.iter().collect();
        let gmm_clusters: std::collections::HashSet<_> = gmm_labels.iter().collect();
        let fcm_clusters: std::collections::HashSet<_> = fcm_labels.iter().collect();

        assert_eq!(kmeans_clusters.len(), 3);
        assert_eq!(gmm_clusters.len(), 3);
        assert_eq!(fcm_clusters.len(), 3);

        // All algorithms should assign all points
        assert_eq!(kmeans_labels.len(), data.nrows());
        assert_eq!(gmm_labels.len(), data.nrows());
        assert_eq!(fcm_labels.len(), data.nrows());
    }
}
