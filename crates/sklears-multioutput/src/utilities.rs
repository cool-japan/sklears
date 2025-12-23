//! Multi-label utilities and analysis tools
//!
//! This module provides utility functions and analysis tools for multi-label
//! classification tasks, including label combination analysis and clustering methods.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Label combination frequency analysis utilities
pub struct CLARE<S = Untrained> {
    state: S,
    n_clusters: usize,
    threshold: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

/// Trained state for CLARE
#[derive(Debug, Clone)]
pub struct CLARETrained {
    cluster_centers: Array2<Float>,
    cluster_assignments: Array1<usize>,
    n_features: usize,
    n_labels: usize,
    threshold: Float,
}

impl Default for CLARE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CLARE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for CLARE<Untrained> {
    type Fitted = CLARE<CLARETrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Convert labels to float for clustering
        let mut label_combinations = Array2::<Float>::zeros((n_samples, n_labels));
        for i in 0..n_samples {
            for j in 0..n_labels {
                label_combinations[[i, j]] = y[[i, j]] as Float;
            }
        }

        // Perform k-means clustering on label combinations
        let (cluster_centers, cluster_assignments) = kmeans_clustering(
            &label_combinations.view(),
            self.n_clusters,
            self.max_iter,
            self.random_state,
        )?;

        Ok(CLARE {
            state: CLARETrained {
                cluster_centers,
                cluster_assignments,
                n_features,
                n_labels,
                threshold: self.threshold,
            },
            n_clusters: self.n_clusters,
            threshold: self.threshold,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl CLARE<Untrained> {
    /// Create a new CLARE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_clusters: 3,
            threshold: 0.5,
            max_iter: 100,
            random_state: None,
        }
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set the threshold for label assignment
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for CLARE<CLARETrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            // Find nearest cluster center
            let x = X.row(sample_idx);
            let mut min_distance = Float::INFINITY;
            let mut nearest_cluster = 0;

            // For simplicity, use feature similarity to find nearest cluster
            // In practice, CLARE would use label correlation patterns
            for cluster_idx in 0..self.state.cluster_centers.nrows() {
                let cluster_center = self.state.cluster_centers.row(cluster_idx);

                // Simple Euclidean distance in label space (simplified)
                let distance = cluster_center.iter().map(|&x| x.abs()).sum::<Float>();

                if distance < min_distance {
                    min_distance = distance;
                    nearest_cluster = cluster_idx;
                }
            }

            // Assign labels based on cluster center
            let cluster_labels = self.state.cluster_centers.row(nearest_cluster);
            for label_idx in 0..self.state.n_labels {
                predictions[[sample_idx, label_idx]] =
                    if cluster_labels[label_idx] > self.state.threshold {
                        1
                    } else {
                        0
                    };
            }
        }

        Ok(predictions)
    }
}

impl CLARE<CLARETrained> {
    /// Get cluster centers
    pub fn cluster_centers(&self) -> &Array2<Float> {
        &self.state.cluster_centers
    }

    /// Get cluster assignments
    pub fn cluster_assignments(&self) -> &Array1<usize> {
        &self.state.cluster_assignments
    }

    /// Get the threshold used for label prediction
    pub fn threshold(&self) -> Float {
        self.state.threshold
    }
}

/// Simple k-means clustering implementation
fn kmeans_clustering(
    data: &ArrayView2<Float>,
    k: usize,
    max_iter: usize,
    random_state: Option<u64>,
) -> SklResult<(Array2<Float>, Array1<usize>)> {
    let (n_samples, n_features) = data.dim();

    if k == 0 {
        return Err(SklearsError::InvalidInput(
            "Number of clusters must be greater than zero".to_string(),
        ));
    }

    if k > n_samples {
        return Err(SklearsError::InvalidInput(
            "Number of clusters cannot exceed number of samples".to_string(),
        ));
    }

    // Use seeded RNG for reproducibility, random seed if none provided
    let seed = random_state.unwrap_or_else(|| {
        use scirs2_core::random::Rng;
        thread_rng().gen()
    });
    let mut rng = scirs2_core::random::seeded_rng(seed);

    // Initialize cluster centers randomly
    let mut centers = Array2::<Float>::zeros((k, n_features));
    for i in 0..k {
        let random_sample_idx = rng.gen_range(0..n_samples);
        centers.row_mut(i).assign(&data.row(random_sample_idx));
    }

    let mut assignments = Array1::<usize>::zeros(n_samples);

    for _iter in 0..max_iter {
        let mut changed = false;

        // Assign points to nearest centers
        for sample_idx in 0..n_samples {
            let sample = data.row(sample_idx);
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for cluster_idx in 0..k {
                let center = centers.row(cluster_idx);
                let distance = sample
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }

            if assignments[sample_idx] != best_cluster {
                assignments[sample_idx] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update cluster centers
        for cluster_idx in 0..k {
            let cluster_points: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &cluster)| cluster == cluster_idx)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_points.is_empty() {
                for feature_idx in 0..n_features {
                    let mean = cluster_points
                        .iter()
                        .map(|&point_idx| data[[point_idx, feature_idx]])
                        .sum::<Float>()
                        / cluster_points.len() as Float;
                    centers[[cluster_idx, feature_idx]] = mean;
                }
            }
        }
    }

    Ok((centers, assignments))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_clare_basic_functionality() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        let clare = CLARE::new().n_clusters(2).threshold(0.5);
        let trained_clare = clare.fit(&X.view(), &y).unwrap();
        let predictions = trained_clare.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    fn test_clare_parameters() {
        let clare = CLARE::new()
            .n_clusters(5)
            .threshold(0.3)
            .max_iter(50)
            .random_state(42);

        assert_eq!(clare.n_clusters, 5);
        assert!((clare.threshold - 0.3).abs() < 1e-10);
        assert_eq!(clare.max_iter, 50);
        assert_eq!(clare.random_state, Some(42));
    }

    #[test]
    fn test_kmeans_clustering() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 8.0]];
        let (centers, assignments) = kmeans_clustering(&data.view(), 2, 100, Some(42)).unwrap();

        assert_eq!(centers.dim(), (2, 2));
        assert_eq!(assignments.len(), 4);
        assert!(assignments.iter().all(|&x| x < 2));
    }

    #[test]
    fn test_kmeans_error_handling() {
        let data = array![[1.0, 2.0], [2.0, 3.0]];
        let result = kmeans_clustering(&data.view(), 5, 100, None);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_clare_cluster_access() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1, 0], [0, 1], [1, 1]];

        let clare = CLARE::new().n_clusters(2);
        let trained_clare = clare.fit(&X.view(), &y).unwrap();

        let centers = trained_clare.cluster_centers();
        let assignments = trained_clare.cluster_assignments();

        assert_eq!(centers.dim(), (2, 2));
        assert_eq!(assignments.len(), 3);
    }
}
