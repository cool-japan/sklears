#![allow(deprecated)]
//! Integration and property-based tests for sklears-python
//!
//! This module contains comprehensive tests to ensure the correctness,
//! robustness, and performance of the Python bindings for sklears.

// Test modules
mod test_clustering;
mod test_linear_regression;
mod test_metrics;

// Common test utilities
pub mod common {
    use scirs2_autograd::ndarray::{Array1, Array2};
    use scirs2_core::random::{thread_rng, Rng};

    /// Generate synthetic regression data for testing
    pub fn generate_regression_data(
        n_samples: usize,
        n_features: usize,
        noise_scale: f64,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = thread_rng();

        let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>());
        let true_coef = Array1::from_shape_fn(n_features, |_| rng.gen::<f64>());
        let noise = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>() * noise_scale);
        let y = x.dot(&true_coef) + noise;

        (x, y, true_coef)
    }

    /// Generate synthetic classification data for testing
    pub fn generate_classification_data(
        n_samples: usize,
        n_classes: usize,
    ) -> (Vec<i32>, Vec<i32>) {
        let mut rng = thread_rng();

        let y_true: Vec<i32> = (0..n_samples)
            .map(|_| (rng.gen::<f64>() * n_classes as f64).floor() as i32)
            .collect();

        // Generate predictions with some accuracy
        let y_pred: Vec<i32> = y_true
            .iter()
            .map(|&true_label| {
                if rng.gen::<f64>() < 0.8 {
                    true_label
                } else {
                    (rng.gen::<f64>() * n_classes as f64).floor() as i32
                }
            })
            .collect();

        (y_true, y_pred)
    }

    /// Generate synthetic clustering data with known clusters
    pub fn generate_clustering_data(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
    ) -> (Array2<f64>, Vec<i32>) {
        let mut rng = thread_rng();

        // Generate cluster centers
        let centers = Array2::from_shape_fn((n_clusters, n_features), |_| rng.gen::<f64>() * 10.0);

        let mut data = Vec::new();
        let mut labels = Vec::new();

        let samples_per_cluster = n_samples / n_clusters;
        let remaining_samples = n_samples % n_clusters;

        for cluster_id in 0..n_clusters {
            let cluster_samples = if cluster_id < remaining_samples {
                samples_per_cluster + 1
            } else {
                samples_per_cluster
            };

            let center = centers.row(cluster_id);

            for _ in 0..cluster_samples {
                // Generate point near cluster center
                let noise = Array1::from_shape_fn(n_features, |_| (rng.gen::<f64>() - 0.5) * 0.5);
                let point = center.to_owned() + noise;
                data.push(point);
                labels.push(cluster_id as i32);
            }
        }

        // Convert to Array2
        let data_array = Array2::from_shape_fn((n_samples, n_features), |(i, j)| data[i][j]);

        (data_array, labels)
    }

    /// Calculate basic statistics for validation
    pub fn calculate_basic_stats(data: &Array1<f64>) -> (f64, f64, f64, f64) {
        let mean = data.mean().unwrap_or(0.0);
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let variance = if data.len() > 1 {
            let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
            sum_sq_diff / (data.len() - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();

        (mean, std_dev, min, max)
    }

    /// Validate array properties for testing
    pub fn validate_array_properties(data: &Array2<f64>) -> bool {
        // Check for finite values
        let all_finite = data.iter().all(|&x| x.is_finite());

        // Check for reasonable dimensions
        let reasonable_dims = data.nrows() > 0 && data.ncols() > 0;

        all_finite && reasonable_dims
    }
}
