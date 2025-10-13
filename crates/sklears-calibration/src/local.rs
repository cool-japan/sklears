//! Local calibration methods for spatially-aware probability calibration
//!
//! This module provides calibration methods that consider the local neighborhood
//! around each prediction, allowing for more flexible calibration relationships
//! that vary across different regions of the feature space.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

use crate::CalibrationEstimator;

/// Local calibration using k-nearest neighbors
///
/// This calibrator finds the k nearest neighbors for each prediction point
/// and calibrates based on the local empirical frequencies within that neighborhood.
#[derive(Debug, Clone)]
pub struct LocalKNNCalibrator {
    k: usize,
    training_features: Option<Array2<Float>>,
    training_probabilities: Option<Array1<Float>>,
    training_labels: Option<Array1<i32>>,
    distance_metric: DistanceMetric,
    weighting: LocalWeighting,
}

/// Distance metrics for local calibration
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Mahalanobis distance (requires covariance matrix)
    Mahalanobis,
}

/// Weighting schemes for local calibration
#[derive(Debug, Clone)]
pub enum LocalWeighting {
    /// Uniform weighting (all neighbors have equal weight)
    Uniform,
    /// Distance-based weighting (closer neighbors have higher weight)
    DistanceWeighted,
    /// Gaussian kernel weighting with specified bandwidth
    Gaussian { bandwidth: Float },
}

impl LocalKNNCalibrator {
    /// Create a new local k-NN calibrator
    pub fn new(k: usize) -> Self {
        Self {
            k,
            training_features: None,
            training_probabilities: None,
            training_labels: None,
            distance_metric: DistanceMetric::Euclidean,
            weighting: LocalWeighting::DistanceWeighted,
        }
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the weighting scheme
    pub fn weighting(mut self, weighting: LocalWeighting) -> Self {
        self.weighting = weighting;
        self
    }

    /// Fit the local calibrator with features and probabilities
    pub fn fit_with_features(
        mut self,
        features: &Array2<Float>,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Self> {
        if features.nrows() != probabilities.len() || features.nrows() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Features, probabilities, and target arrays must have the same length".to_string(),
            ));
        }

        if features.nrows() < self.k {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples ({}) must be >= k ({})",
                features.nrows(),
                self.k
            )));
        }

        self.training_features = Some(features.clone());
        self.training_probabilities = Some(probabilities.clone());
        self.training_labels = Some(y_true.clone());

        Ok(self)
    }

    /// Predict calibrated probabilities using local neighborhoods
    pub fn predict_proba_with_features(&self, features: &Array2<Float>) -> Result<Array1<Float>> {
        let training_features =
            self.training_features
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_features".to_string(),
                })?;
        let training_probabilities =
            self.training_probabilities
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_features".to_string(),
                })?;
        let training_labels =
            self.training_labels
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_features".to_string(),
                })?;

        let mut calibrated_probabilities = Array1::zeros(features.nrows());

        for (i, test_point) in features.axis_iter(Axis(0)).enumerate() {
            // Find k nearest neighbors
            let mut distances: Vec<(usize, Float)> = training_features
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(idx, train_point)| {
                    let distance = self.calculate_distance(&test_point, &train_point);
                    (idx, distance)
                })
                .collect();

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(self.k);

            // Calculate weights
            let weights = self.calculate_weights(&distances);

            // Calculate locally calibrated probability
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, (neighbor_idx, _)) in distances.iter().enumerate() {
                let weight = weights[j];
                let true_label = training_labels[*neighbor_idx];
                weighted_sum += weight * true_label as Float;
                weight_sum += weight;
            }

            let calibrated_prob: Float = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.5 // Default to 0.5 if no valid weights
            };

            calibrated_probabilities[i] =
                calibrated_prob.clamp(1e-15 as Float, 1.0 - 1e-15 as Float);
        }

        Ok(calibrated_probabilities)
    }

    fn calculate_distance(&self, point1: &ArrayView1<Float>, point2: &ArrayView1<Float>) -> Float {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                let diff = point1 - point2;
                diff.mapv(|x| x * x).sum().sqrt()
            }
            DistanceMetric::Manhattan => {
                let diff = point1 - point2;
                diff.mapv(|x| x.abs()).sum()
            }
            DistanceMetric::Mahalanobis => {
                // Simplified Mahalanobis distance (assumes identity covariance)
                // In practice, you would compute the actual covariance matrix
                let diff = point1 - point2;
                diff.mapv(|x| x * x).sum().sqrt()
            }
        }
    }

    fn calculate_weights(&self, distances: &[(usize, Float)]) -> Vec<Float> {
        match self.weighting {
            LocalWeighting::Uniform => vec![1.0; distances.len()],
            LocalWeighting::DistanceWeighted => {
                distances
                    .iter()
                    .map(|(_, distance)| {
                        if *distance > 0.0 {
                            1.0 / distance
                        } else {
                            1e6 // Very large weight for exact matches
                        }
                    })
                    .collect()
            }
            LocalWeighting::Gaussian { bandwidth } => distances
                .iter()
                .map(|(_, distance)| (-0.5 * (distance / bandwidth).powi(2)).exp())
                .collect(),
        }
    }
}

impl CalibrationEstimator for LocalKNNCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // For compatibility with CalibrationEstimator trait, we create dummy features
        // based on the probabilities themselves
        let features = probabilities.clone().insert_axis(Axis(1));
        *self = self
            .clone()
            .fit_with_features(&features, probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For compatibility with CalibrationEstimator trait, we create dummy features
        let features = probabilities.clone().insert_axis(Axis(1));
        self.predict_proba_with_features(&features)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Local binning calibrator
///
/// This calibrator creates local bins based on feature similarity and
/// calibrates probabilities within each local region.
#[derive(Debug, Clone)]
pub struct LocalBinningCalibrator {
    n_bins: usize,
    bin_assignments: Option<HashMap<usize, usize>>,
    bin_calibrators: Option<Vec<EmpiricalBinCalibrator>>,
    training_features: Option<Array2<Float>>,
}

#[derive(Debug, Clone)]
struct EmpiricalBinCalibrator {
    bin_prob: Float,
    n_positive: usize,
    n_total: usize,
}

impl EmpiricalBinCalibrator {
    fn new() -> Self {
        Self {
            bin_prob: 0.5,
            n_positive: 0,
            n_total: 0,
        }
    }

    fn add_sample(&mut self, is_positive: bool) {
        self.n_total += 1;
        if is_positive {
            self.n_positive += 1;
        }
        self.update_probability();
    }

    fn update_probability(&mut self) {
        self.bin_prob = if self.n_total > 0 {
            self.n_positive as Float / self.n_total as Float
        } else {
            0.5
        };
    }

    fn predict(&self) -> Float {
        self.bin_prob
    }
}

impl LocalBinningCalibrator {
    /// Create a new local binning calibrator
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            bin_assignments: None,
            bin_calibrators: None,
            training_features: None,
        }
    }

    /// Fit the local binning calibrator with features
    pub fn fit_with_features(
        mut self,
        features: &Array2<Float>,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Self> {
        if features.nrows() != probabilities.len() || features.nrows() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Features, probabilities, and target arrays must have the same length".to_string(),
            ));
        }

        // Simple k-means-like clustering to create local bins
        let bin_assignments = self.create_local_bins(features)?;
        let mut bin_calibrators = vec![EmpiricalBinCalibrator::new(); self.n_bins];

        // Train each bin calibrator
        for (i, &bin_id) in bin_assignments.iter().enumerate() {
            let is_positive = y_true[i] == 1;
            bin_calibrators[bin_id].add_sample(is_positive);
        }

        self.bin_assignments = Some(bin_assignments.into_iter().enumerate().collect());
        self.bin_calibrators = Some(bin_calibrators);
        self.training_features = Some(features.clone());

        Ok(self)
    }

    /// Predict calibrated probabilities using local bins
    pub fn predict_proba_with_features(&self, features: &Array2<Float>) -> Result<Array1<Float>> {
        let training_features =
            self.training_features
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_features".to_string(),
                })?;
        let bin_calibrators =
            self.bin_calibrators
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba_with_features".to_string(),
                })?;

        let mut calibrated_probabilities = Array1::zeros(features.nrows());

        for (i, test_point) in features.axis_iter(Axis(0)).enumerate() {
            // Find the nearest bin
            let bin_id = self.find_nearest_bin(&test_point, training_features)?;
            calibrated_probabilities[i] = bin_calibrators[bin_id].predict();
        }

        Ok(calibrated_probabilities)
    }

    fn create_local_bins(&self, features: &Array2<Float>) -> Result<Vec<usize>> {
        let n_samples = features.nrows();
        let mut bin_assignments = vec![0; n_samples];

        // Simple assignment based on feature quantiles
        if features.ncols() > 0 {
            let first_feature = features.column(0);
            let mut sorted_indices: Vec<usize> = (0..n_samples).collect();
            sorted_indices.sort_by(|&a, &b| {
                first_feature[a]
                    .partial_cmp(&first_feature[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let bin_size = n_samples / self.n_bins;
            for (i, &idx) in sorted_indices.iter().enumerate() {
                let bin_id = (i / bin_size).min(self.n_bins - 1);
                bin_assignments[idx] = bin_id;
            }
        }

        Ok(bin_assignments)
    }

    fn find_nearest_bin(
        &self,
        test_point: &ArrayView1<Float>,
        training_features: &Array2<Float>,
    ) -> Result<usize> {
        // Find the nearest training point and return its bin
        let mut min_distance = Float::INFINITY;
        let mut nearest_bin = 0;

        for (i, train_point) in training_features.axis_iter(Axis(0)).enumerate() {
            let distance = self.euclidean_distance(test_point, &train_point);
            if distance < min_distance {
                min_distance = distance;
                if let Some(bin_assignments) = &self.bin_assignments {
                    if let Some(&bin_id) = bin_assignments.get(&i) {
                        nearest_bin = bin_id;
                    }
                }
            }
        }

        Ok(nearest_bin)
    }

    fn euclidean_distance(&self, point1: &ArrayView1<Float>, point2: &ArrayView1<Float>) -> Float {
        let diff = point1 - point2;
        diff.mapv(|x| x * x).sum().sqrt()
    }
}

impl CalibrationEstimator for LocalBinningCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // For compatibility with CalibrationEstimator trait, we create dummy features
        let features = probabilities.clone().insert_axis(Axis(1));
        *self = self
            .clone()
            .fit_with_features(&features, probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For compatibility with CalibrationEstimator trait, we create dummy features
        let features = probabilities.clone().insert_axis(Axis(1));
        self.predict_proba_with_features(&features)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_local_knn_calibrator() {
        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        let calibrator = LocalKNNCalibrator::new(3)
            .fit_with_features(&features, &probabilities, &y_true)
            .unwrap();

        let test_features = array![[2.5, 3.5], [4.5, 5.5]];
        let calibrated = calibrator
            .predict_proba_with_features(&test_features)
            .unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_local_binning_calibrator() {
        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        let calibrator = LocalBinningCalibrator::new(3)
            .fit_with_features(&features, &probabilities, &y_true)
            .unwrap();

        let test_features = array![[2.5, 3.5], [4.5, 5.5]];
        let calibrated = calibrator
            .predict_proba_with_features(&test_features)
            .unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_local_calibrator_with_trait() {
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        let mut calibrator = LocalKNNCalibrator::new(3);
        calibrator.fit(&probabilities, &y_true).unwrap();

        let test_probabilities = array![0.2, 0.8];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_distance_metrics() {
        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        // Test different distance metrics
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Mahalanobis,
        ] {
            let calibrator = LocalKNNCalibrator::new(2)
                .distance_metric(metric)
                .fit_with_features(&features, &probabilities, &y_true)
                .unwrap();

            let test_features = array![[2.5, 3.5]];
            let calibrated = calibrator
                .predict_proba_with_features(&test_features)
                .unwrap();

            assert_eq!(calibrated.len(), 1);
            assert!(calibrated[0] >= 0.0 && calibrated[0] <= 1.0);
        }
    }

    #[test]
    fn test_weighting_schemes() {
        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        // Test different weighting schemes
        for weighting in [
            LocalWeighting::Uniform,
            LocalWeighting::DistanceWeighted,
            LocalWeighting::Gaussian { bandwidth: 1.0 },
        ] {
            let calibrator = LocalKNNCalibrator::new(2)
                .weighting(weighting)
                .fit_with_features(&features, &probabilities, &y_true)
                .unwrap();

            let test_features = array![[2.5, 3.5]];
            let calibrated = calibrator
                .predict_proba_with_features(&test_features)
                .unwrap();

            assert_eq!(calibrated.len(), 1);
            assert!(calibrated[0] >= 0.0 && calibrated[0] <= 1.0);
        }
    }
}
