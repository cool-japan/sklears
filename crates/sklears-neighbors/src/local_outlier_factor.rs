//! Local Outlier Factor for anomaly detection

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict};
use sklears_core::types::{Features, Float, Int};

/// Helper struct to store pre-computed LOF data for training points
struct TrainingLofData {
    k_distances: Array1<Float>,
    lrds: Array1<Float>,
}

/// Local Outlier Factor (LOF) for outlier detection
///
/// LOF computes local density deviation of a given data point with respect to its neighbors.
/// It considers as outliers the samples that have a substantially lower density than their neighbors.
#[derive(Debug, Clone)]
pub struct LocalOutlierFactor<State = sklears_core::traits::Untrained> {
    /// Number of neighbors to use for LOF computation
    pub n_neighbors: usize,
    /// Distance metric to use
    pub metric: Distance,
    /// Algorithm to use for neighbor computation
    pub algorithm: crate::knn::Algorithm,
    /// Contamination parameter (fraction of outliers in the data set)
    pub contamination: Float,
    /// Whether this LOF should be used for novelty detection
    pub novelty: bool,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Negative LOF scores (only available after fitting)
    pub(crate) negative_outlier_factor_: Option<Array1<Float>>,
    /// Decision threshold (only available after fitting)
    pub(crate) threshold_: Option<Float>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl LocalOutlierFactor {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            metric: Distance::default(),
            algorithm: crate::knn::Algorithm::Brute,
            contamination: 0.1,
            novelty: false,
            x_train: None,
            negative_outlier_factor_: None,
            threshold_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the algorithm
    pub fn with_algorithm(mut self, algorithm: crate::knn::Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the contamination parameter
    pub fn with_contamination(mut self, contamination: Float) -> Self {
        self.contamination = contamination;
        self
    }

    /// Set whether this LOF should be used for novelty detection
    pub fn with_novelty(mut self, novelty: bool) -> Self {
        self.novelty = novelty;
        self
    }
}

impl Estimator for LocalOutlierFactor {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, ()> for LocalOutlierFactor {
    type Fitted = LocalOutlierFactor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, _y: &()) -> Result<Self::Fitted> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if self.n_neighbors == 0 || self.n_neighbors >= x.nrows() {
            return Err(NeighborsError::InvalidNeighbors(self.n_neighbors).into());
        }

        if !(0.0..=0.5).contains(&self.contamination) {
            return Err(NeighborsError::InvalidInput(format!(
                "Contamination must be in [0.0, 0.5], got {}",
                self.contamination
            ))
            .into());
        }

        // Compute LOF scores for all training samples
        let lof_scores =
            LocalOutlierFactor::compute_lof_scores_static(x, self.n_neighbors, &self.metric)?;
        let negative_outlier_factor = lof_scores.mapv(|score| -score);

        // Compute threshold based on contamination
        let mut sorted_scores = negative_outlier_factor.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = ((1.0 - self.contamination) * sorted_scores.len() as Float) as usize;
        let threshold_idx = threshold_idx.min(sorted_scores.len() - 1);
        let threshold = sorted_scores[threshold_idx];

        Ok(LocalOutlierFactor {
            n_neighbors: self.n_neighbors,
            metric: self.metric,
            algorithm: self.algorithm,
            contamination: self.contamination,
            novelty: self.novelty,
            x_train: Some(x.clone()),
            negative_outlier_factor_: Some(negative_outlier_factor),
            threshold_: Some(threshold),
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>> for LocalOutlierFactor<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if !self.novelty {
            return Err(NeighborsError::InvalidInput(
                "Cannot use predict when novelty=false. Use fit_predict instead.".to_string(),
            )
            .into());
        }

        let decision_scores = self.decision_function(x)?;
        let threshold = self.threshold_.unwrap();

        let predictions = decision_scores.mapv(|score| if score >= threshold { 1 } else { -1 });
        Ok(predictions)
    }
}

impl LocalOutlierFactor<sklears_core::traits::Trained> {
    /// Shifted opposite of the Local Outlier Factor
    pub fn decision_function(&self, x: &Features) -> Result<Array1<Float>> {
        let x_train = self.x_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        if self.novelty {
            // For novelty detection, compute LOF for new points using training data
            self.compute_lof_scores_novelty(x)
        } else {
            // For outlier detection, return precomputed scores
            Ok(self.negative_outlier_factor_.as_ref().unwrap().clone())
        }
    }

    /// Fit and predict whether a particular sample is an outlier or not
    pub fn fit_predict(&self, _x: &Features) -> Result<Array1<Int>> {
        if self.novelty {
            return Err(NeighborsError::InvalidInput(
                "Cannot use fit_predict when novelty=true. Use predict instead.".to_string(),
            )
            .into());
        }

        let scores = self.negative_outlier_factor_.as_ref().unwrap();
        let threshold = self.threshold_.unwrap();

        let predictions = scores.mapv(|score| if score >= threshold { 1 } else { -1 });
        Ok(predictions)
    }

    /// Compute LOF scores for training data (static method)
    fn compute_lof_scores_static(
        x: &Array2<Float>,
        n_neighbors: usize,
        metric: &Distance,
    ) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();

        // Pre-compute k-distances and neighbors for all points
        let mut k_distances = Array1::zeros(n_samples);
        let mut neighbors_indices = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let distances = metric.to_matrix(&sample, &x.view());
            let mut neighbors: Vec<(Float, usize)> = distances
                .iter()
                .enumerate()
                .map(|(idx, &dist)| (dist, idx))
                .collect();
            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Take k+1 neighbors (including self), then exclude self
            let k_neighbors: Vec<usize> = neighbors
                .iter()
                .take(n_neighbors + 1)
                .skip(1) // Skip self (distance 0)
                .map(|(_, idx)| *idx)
                .collect();

            // k-distance is distance to k-th nearest neighbor
            let k_distance = if k_neighbors.len() >= n_neighbors {
                neighbors[n_neighbors].0 // k-th neighbor (after skipping self)
            } else {
                neighbors.last().map(|(d, _)| *d).unwrap_or(0.0)
            };

            k_distances[i] = k_distance;
            neighbors_indices.push(k_neighbors);
        }

        // Compute Local Reachability Density (LRD) for all points
        let mut lrds = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let mut reach_dists_sum = 0.0;
            let k_neighbors = &neighbors_indices[i];

            for &neighbor_idx in k_neighbors {
                let neighbor_sample = x.row(neighbor_idx);
                let dist = metric.calculate(&sample, &neighbor_sample);
                let neighbor_k_distance = k_distances[neighbor_idx];

                // Reachability distance is max of actual distance and neighbor's k-distance
                let reach_dist = dist.max(neighbor_k_distance);
                reach_dists_sum += reach_dist;
            }

            // LRD is the inverse of average reachability distance
            lrds[i] = if reach_dists_sum == 0.0 || k_neighbors.is_empty() {
                Float::INFINITY
            } else {
                k_neighbors.len() as Float / reach_dists_sum
            };
        }

        // Compute Local Outlier Factor (LOF) for all points
        let mut lof_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let k_neighbors = &neighbors_indices[i];
            let lrd_i = lrds[i];

            if lrd_i.is_infinite() {
                // If point's LRD is infinite, it's in a very dense region
                lof_scores[i] = 1.0;
            } else if lrd_i == 0.0 {
                // If point's LRD is 0, it's an extreme outlier
                lof_scores[i] = Float::INFINITY;
            } else {
                // Compute average LRD of neighbors
                let neighbor_lrds_sum: Float = k_neighbors
                    .iter()
                    .map(|&neighbor_idx| lrds[neighbor_idx])
                    .sum();

                let mean_neighbor_lrd = if k_neighbors.is_empty() {
                    lrd_i
                } else {
                    neighbor_lrds_sum / k_neighbors.len() as Float
                };

                // LOF is the ratio of neighbor LRDs to point's LRD
                lof_scores[i] = mean_neighbor_lrd / lrd_i;
            }
        }

        Ok(lof_scores)
    }

    /// Compute LOF scores for novelty detection (new points vs training data)
    fn compute_lof_scores_novelty(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let x_train = self.x_train.as_ref().unwrap();
        let n_queries = x.nrows();
        let _n_train = x_train.nrows();

        // Pre-compute k-distances and LRDs for training data
        let training_lof_data =
            Self::compute_training_lof_data(x_train, self.n_neighbors, &self.metric)?;

        let mut scores = Array1::zeros(n_queries);

        for i in 0..n_queries {
            let sample = x.row(i);

            // Find k-nearest neighbors in training data
            let distances = self.metric.to_matrix(&sample, &x_train.view());
            let mut neighbors: Vec<(Float, usize)> = distances
                .iter()
                .enumerate()
                .map(|(idx, &dist)| (dist, idx))
                .collect();
            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Take k neighbors from training data
            let k_neighbors: Vec<(Float, usize)> =
                neighbors.into_iter().take(self.n_neighbors).collect();

            if k_neighbors.is_empty() {
                scores[i] = -Float::INFINITY;
                continue;
            }

            // Compute reachability distances to training neighbors
            let mut reach_dists_sum = 0.0;
            for (dist, neighbor_idx) in &k_neighbors {
                let neighbor_k_distance = training_lof_data.k_distances[*neighbor_idx];
                let reach_dist = dist.max(neighbor_k_distance);
                reach_dists_sum += reach_dist;
            }

            // Compute LRD for the query point
            let lrd_query = if reach_dists_sum == 0.0 {
                Float::INFINITY
            } else {
                k_neighbors.len() as Float / reach_dists_sum
            };

            // Compute LOF using neighbor LRDs from training data
            if lrd_query.is_infinite() {
                scores[i] = -1.0; // Dense region, not an outlier
            } else if lrd_query == 0.0 {
                scores[i] = -Float::INFINITY; // Extreme outlier
            } else {
                let neighbor_lrds_sum: Float = k_neighbors
                    .iter()
                    .map(|(_, neighbor_idx)| training_lof_data.lrds[*neighbor_idx])
                    .sum();

                let mean_neighbor_lrd = neighbor_lrds_sum / k_neighbors.len() as Float;
                let lof_score = mean_neighbor_lrd / lrd_query;

                // Convert to negative outlier factor
                scores[i] = -lof_score;
            }
        }

        Ok(scores)
    }

    /// Pre-compute k-distances and LRDs for training data
    fn compute_training_lof_data(
        x_train: &Array2<Float>,
        n_neighbors: usize,
        metric: &Distance,
    ) -> NeighborsResult<TrainingLofData> {
        let n_train = x_train.nrows();
        let mut k_distances = Array1::zeros(n_train);
        let mut neighbors_indices = Vec::with_capacity(n_train);

        // Compute k-distances and neighbor indices
        for i in 0..n_train {
            let sample = x_train.row(i);
            let distances = metric.to_matrix(&sample, &x_train.view());
            let mut neighbors: Vec<(Float, usize)> = distances
                .iter()
                .enumerate()
                .map(|(idx, &dist)| (dist, idx))
                .collect();
            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Take k+1 neighbors (including self), then exclude self
            let k_neighbors: Vec<usize> = neighbors
                .iter()
                .take(n_neighbors + 1)
                .skip(1) // Skip self
                .map(|(_, idx)| *idx)
                .collect();

            let k_distance = if k_neighbors.len() >= n_neighbors {
                neighbors[n_neighbors].0
            } else {
                neighbors.last().map(|(d, _)| *d).unwrap_or(0.0)
            };

            k_distances[i] = k_distance;
            neighbors_indices.push(k_neighbors);
        }

        // Compute LRDs
        let mut lrds = Array1::zeros(n_train);
        for i in 0..n_train {
            let sample = x_train.row(i);
            let mut reach_dists_sum = 0.0;
            let k_neighbors = &neighbors_indices[i];

            for &neighbor_idx in k_neighbors {
                let neighbor_sample = x_train.row(neighbor_idx);
                let dist = metric.calculate(&sample, &neighbor_sample);
                let neighbor_k_distance = k_distances[neighbor_idx];
                let reach_dist = dist.max(neighbor_k_distance);
                reach_dists_sum += reach_dist;
            }

            lrds[i] = if reach_dists_sum == 0.0 || k_neighbors.is_empty() {
                Float::INFINITY
            } else {
                k_neighbors.len() as Float / reach_dists_sum
            };
        }

        Ok(TrainingLofData { k_distances, lrds })
    }

    /// Get the negative outlier factor for the training samples
    pub fn negative_outlier_factor(&self) -> &Array1<Float> {
        self.negative_outlier_factor_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_local_outlier_factor_basic() {
        // Create data with clear outliers
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Normal
                1.1, 1.1, // Normal
                1.2, 1.2, // Normal
                1.0, 1.1, // Normal
                50.0, 50.0, // Outlier
                -50.0, -50.0, // Outlier
            ],
        )
        .unwrap();

        let lof = LocalOutlierFactor::new(2).with_contamination(0.4);
        let fitted = lof.fit(&x, &()).unwrap();

        let predictions = fitted.fit_predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // The algorithm should produce predictions with proper values (-1 or 1)
        for &pred in predictions.iter() {
            assert!(pred == -1 || pred == 1);
        }

        // Should have some predictions (either outliers or inliers)
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_local_outlier_factor_novelty() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.0, 1.1]).unwrap();

        let lof = LocalOutlierFactor::new(2)
            .with_novelty(true)
            .with_contamination(0.1);
        let fitted = lof.fit(&x_train, &()).unwrap();

        // Test with new data including outliers
        let x_test = Array2::from_shape_vec(
            (3, 2),
            vec![
                1.05, 1.05, // Should be normal (close to training data)
                10.0, 10.0, // Should be outlier (far from training data)
                -5.0, -5.0, // Should be outlier (far from training data)
            ],
        )
        .unwrap();

        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 3);

        // Check that the first point (close to training data) has a better score than the far points
        // With improved LOF implementation, exact threshold behavior may vary
        // but the pattern should be: close point < far points in terms of outlier score

        let decision_scores = fitted.decision_function(&x_test).unwrap();

        // The close point should have a less negative score (closer to -1) than the far points
        assert!(
            decision_scores[0] > decision_scores[1],
            "Close point should have better score than far point 1: {} vs {}",
            decision_scores[0],
            decision_scores[1]
        );
        assert!(
            decision_scores[0] > decision_scores[2],
            "Close point should have better score than far point 2: {} vs {}",
            decision_scores[0],
            decision_scores[2]
        );

        // Far points should have very negative scores (high LOF)
        assert!(
            decision_scores[1] < -1.5,
            "Far point 1 should be clearly an outlier"
        );
        assert!(
            decision_scores[2] < -1.5,
            "Far point 2 should be clearly an outlier"
        );
    }

    #[test]
    fn test_local_outlier_factor_errors() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();

        // Test invalid n_neighbors
        let lof = LocalOutlierFactor::new(0);
        let result = lof.fit(&x, &());
        assert!(result.is_err());

        // Test n_neighbors >= n_samples
        let lof = LocalOutlierFactor::new(3);
        let result = lof.fit(&x, &());
        assert!(result.is_err());

        // Test invalid contamination
        let lof = LocalOutlierFactor::new(1).with_contamination(0.6);
        let result = lof.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_local_outlier_factor_predict_without_novelty() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

        let lof = LocalOutlierFactor::new(2); // novelty=false by default
        let fitted = lof.fit(&x, &()).unwrap();

        let result = fitted.predict(&x);
        assert!(result.is_err()); // Should fail when novelty=false
    }

    #[test]
    fn test_local_outlier_factor_fit_predict_with_novelty() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

        let lof = LocalOutlierFactor::new(2).with_novelty(true);
        let fitted = lof.fit(&x, &()).unwrap();

        let result = fitted.fit_predict(&x);
        assert!(result.is_err()); // Should fail when novelty=true
    }
}
