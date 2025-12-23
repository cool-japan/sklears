//! Angle-Based Outlier Detection (ABOD)
//!
//! ABOD is particularly effective for outlier detection in high-dimensional spaces
//! where traditional distance-based methods may fail due to the curse of dimensionality.
//! It measures the variance of angles between a point and all pairs of other points.

use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::traits::{Estimator, Fit};
use sklears_core::types::{Features, Float};

/// Angle-Based Outlier Detection algorithm
#[derive(Debug, Clone)]
pub struct AngleBasedOutlierDetection<State = sklears_core::traits::Untrained> {
    /// Number of k-nearest neighbors to consider for angle computation
    pub k: usize,
    /// Contamination rate (fraction of outliers expected)
    pub contamination: Float,
    /// Whether to use fast ABOD (FastABOD) for large datasets
    pub fast_abod: bool,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// ABOD scores for training data (only available after fitting)
    pub(crate) abod_scores: Option<Array1<Float>>,
    /// Threshold for outlier detection (only available after fitting)
    pub(crate) threshold: Option<Float>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl AngleBasedOutlierDetection {
    pub fn new() -> Self {
        Self {
            k: 20,
            contamination: 0.1,
            fast_abod: true,
            x_train: None,
            abod_scores: None,
            threshold: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the number of nearest neighbors
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the contamination rate
    pub fn with_contamination(mut self, contamination: Float) -> Self {
        self.contamination = contamination;
        self
    }

    /// Set whether to use fast ABOD
    pub fn with_fast_abod(mut self, fast_abod: bool) -> Self {
        self.fast_abod = fast_abod;
        self
    }
}

impl Default for AngleBasedOutlierDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AngleBasedOutlierDetection {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, ()> for AngleBasedOutlierDetection {
    type Fitted = AngleBasedOutlierDetection<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, _y: &()) -> sklears_core::error::Result<Self::Fitted> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if self.k >= x.nrows() {
            return Err(NeighborsError::InvalidNeighbors(self.k).into());
        }

        let abod_scores = if self.fast_abod && x.nrows() > 1000 {
            self.compute_fast_abod_scores(x)?
        } else {
            self.compute_abod_scores(x)?
        };

        // Determine threshold based on contamination rate
        let mut sorted_scores = abod_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = (self.contamination * sorted_scores.len() as Float) as usize;
        let threshold = if threshold_idx < sorted_scores.len() {
            sorted_scores[threshold_idx]
        } else {
            *sorted_scores.last().unwrap_or(&0.0)
        };

        Ok(AngleBasedOutlierDetection {
            k: self.k,
            contamination: self.contamination,
            fast_abod: self.fast_abod,
            x_train: Some(x.clone()),
            abod_scores: Some(abod_scores),
            threshold: Some(threshold),
            _state: std::marker::PhantomData,
        })
    }
}

impl<State> AngleBasedOutlierDetection<State> {
    /// Compute ABOD scores for all points (full algorithm)
    fn compute_abod_scores(&self, x: &Array2<Float>) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = x.row(i);
            scores[i] = self.compute_abod_score(&point, x, i)?;
        }

        Ok(scores)
    }

    /// Compute Fast ABOD scores (using k-NN for efficiency)
    fn compute_fast_abod_scores(&self, x: &Array2<Float>) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = x.row(i);

            // Find k nearest neighbors
            let neighbors = self.find_k_nearest_neighbors(&point, x, i)?;

            // Compute ABOD score using only nearest neighbors
            scores[i] = self.compute_abod_score_with_neighbors(&point, x, &neighbors)?;
        }

        Ok(scores)
    }

    /// Compute ABOD score for a single point using all other points
    fn compute_abod_score(
        &self,
        point: &ArrayView1<Float>,
        x: &Array2<Float>,
        point_idx: usize,
    ) -> NeighborsResult<Float> {
        let n_samples = x.nrows();
        if n_samples < 3 {
            return Ok(0.0);
        }

        let mut angle_variances = Vec::new();

        // For each pair of other points, compute the angle variance
        for i in 0..n_samples {
            if i == point_idx {
                continue;
            }
            for j in (i + 1)..n_samples {
                if j == point_idx {
                    continue;
                }

                let p1 = x.row(i);
                let p2 = x.row(j);

                let angle = self.compute_angle(point, &p1, &p2)?;
                angle_variances.push(angle);
            }
        }

        if angle_variances.is_empty() {
            return Ok(0.0);
        }

        // Compute variance of angles
        let mean_angle = angle_variances.iter().sum::<Float>() / angle_variances.len() as Float;
        let variance = angle_variances
            .iter()
            .map(|&angle| (angle - mean_angle).powi(2))
            .sum::<Float>()
            / angle_variances.len() as Float;

        Ok(variance)
    }

    /// Compute ABOD score using only k nearest neighbors
    fn compute_abod_score_with_neighbors(
        &self,
        point: &ArrayView1<Float>,
        x: &Array2<Float>,
        neighbors: &[usize],
    ) -> NeighborsResult<Float> {
        if neighbors.len() < 2 {
            return Ok(0.0);
        }

        let mut angle_variances = Vec::new();

        // For each pair of neighbors, compute the angle
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let p1 = x.row(neighbors[i]);
                let p2 = x.row(neighbors[j]);

                let angle = self.compute_angle(point, &p1, &p2)?;
                angle_variances.push(angle);
            }
        }

        if angle_variances.is_empty() {
            return Ok(0.0);
        }

        // Compute variance of angles
        let mean_angle = angle_variances.iter().sum::<Float>() / angle_variances.len() as Float;
        let variance = angle_variances
            .iter()
            .map(|&angle| (angle - mean_angle).powi(2))
            .sum::<Float>()
            / angle_variances.len() as Float;

        Ok(variance)
    }

    /// Compute the angle between three points (angle at query point)
    fn compute_angle(
        &self,
        query: &ArrayView1<Float>,
        p1: &ArrayView1<Float>,
        p2: &ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        // Vectors from query to p1 and p2
        let v1: Array1<Float> = p1 - query;
        let v2: Array1<Float> = p2 - query;

        // Compute dot product and norms
        let dot_product = v1.dot(&v2);
        let norm_v1 = v1.mapv(|x| x * x).sum().sqrt();
        let norm_v2 = v2.mapv(|x| x * x).sum().sqrt();

        // Avoid division by zero
        if norm_v1 == 0.0 || norm_v2 == 0.0 {
            return Ok(0.0);
        }

        // Compute cosine of angle
        let cos_angle = dot_product / (norm_v1 * norm_v2);

        // Clamp to valid range for acos
        let cos_angle = cos_angle.clamp(-1.0, 1.0);

        // Return angle in radians
        Ok(cos_angle.acos())
    }

    /// Find k nearest neighbors using Euclidean distance
    fn find_k_nearest_neighbors(
        &self,
        point: &ArrayView1<Float>,
        x: &Array2<Float>,
        point_idx: usize,
    ) -> NeighborsResult<Vec<usize>> {
        let mut distances: Vec<(Float, usize)> = Vec::new();

        for (i, other) in x.axis_iter(Axis(0)).enumerate() {
            if i != point_idx {
                let dist = self.euclidean_distance(point, &other);
                distances.push((dist, i));
            }
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.k.min(distances.len());

        Ok(distances.into_iter().take(k).map(|(_, idx)| idx).collect())
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> Float {
        (p1 - p2).mapv(|x| x * x).sum().sqrt()
    }
}

impl AngleBasedOutlierDetection<sklears_core::traits::Trained> {
    /// Predict outliers on training data
    pub fn predict(&self, x: &Features) -> NeighborsResult<Array1<i32>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let x_train = self.x_train.as_ref().unwrap();
        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            });
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let score = if self.fast_abod && x_train.nrows() > 1000 {
                // Find k nearest neighbors in training data
                let neighbors = self.find_k_nearest_neighbors(&sample, x_train, usize::MAX)?;
                self.compute_abod_score_with_neighbors(&sample, x_train, &neighbors)?
            } else {
                self.compute_abod_score(&sample, x_train, usize::MAX)?
            };

            // Predict outlier if score is below threshold (lower ABOD score = more outlier-like)
            predictions[i] = if score <= self.threshold.unwrap_or(0.0) {
                1
            } else {
                -1
            };
        }

        Ok(predictions)
    }

    /// Get ABOD scores for input data
    pub fn decision_function(&self, x: &Features) -> NeighborsResult<Array1<Float>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let x_train = self.x_train.as_ref().unwrap();
        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            });
        }

        let mut scores = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            scores[i] = if self.fast_abod && x_train.nrows() > 1000 {
                // Find k nearest neighbors in training data
                let neighbors = self.find_k_nearest_neighbors(&sample, x_train, usize::MAX)?;
                self.compute_abod_score_with_neighbors(&sample, x_train, &neighbors)?
            } else {
                self.compute_abod_score(&sample, x_train, usize::MAX)?
            };
        }

        Ok(scores)
    }

    /// Get the threshold used for outlier detection
    pub fn threshold(&self) -> Option<Float> {
        self.threshold
    }

    /// Get training ABOD scores
    pub fn training_scores(&self) -> Option<&Array1<Float>> {
        self.abod_scores.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_abod_construction() {
        let abod = AngleBasedOutlierDetection::new();
        assert_eq!(abod.k, 20);
        assert_abs_diff_eq!(abod.contamination, 0.1, epsilon = 1e-10);
        assert!(abod.fast_abod);
    }

    #[test]
    fn test_abod_with_parameters() {
        let abod = AngleBasedOutlierDetection::new()
            .with_k(10)
            .with_contamination(0.05)
            .with_fast_abod(false);

        assert_eq!(abod.k, 10);
        assert_abs_diff_eq!(abod.contamination, 0.05, epsilon = 1e-10);
        assert!(!abod.fast_abod);
    }

    #[test]
    fn test_abod_fit_predict() {
        // Create data with one clear outlier
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0, // Normal
                1.5, 1.5, // Normal
                2.0, 2.0, // Normal
                2.5, 2.5, // Normal
                10.0, 10.0, // Outlier
            ],
        )
        .unwrap();

        let abod = AngleBasedOutlierDetection::new()
            .with_k(3)
            .with_contamination(0.2)
            .with_fast_abod(false);

        let fitted = abod.fit(&data, &()).unwrap();
        let predictions = fitted.predict(&data).unwrap();

        // Should detect 1 outlier
        let outlier_count = predictions.iter().filter(|&&x| x == 1).count();
        assert!(outlier_count >= 1, "Should detect at least one outlier");

        // Verify we have valid predictions
        assert_eq!(predictions.len(), 5);
        for &pred in predictions.iter() {
            assert!(pred == 1 || pred == -1, "Predictions should be 1 or -1");
        }
    }

    #[test]
    fn test_abod_decision_function() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 10.0, 10.0, // Outlier
            ],
        )
        .unwrap();

        let abod = AngleBasedOutlierDetection::new()
            .with_k(2)
            .with_fast_abod(false);

        let fitted = abod.fit(&data, &()).unwrap();
        let scores = fitted.decision_function(&data).unwrap();

        assert_eq!(scores.len(), 4);
        // All scores should be finite
        for &score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_abod_angle_computation() {
        let abod = AngleBasedOutlierDetection::new();

        // Test 90-degree angle
        let query = array![0.0, 0.0];
        let p1 = array![1.0, 0.0];
        let p2 = array![0.0, 1.0];

        let angle = abod
            .compute_angle(&query.view(), &p1.view(), &p2.view())
            .unwrap();
        assert_abs_diff_eq!(angle, std::f64::consts::PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_abod_fast_mode() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 10.0, 10.0, // Outlier
            ],
        )
        .unwrap();

        let abod = AngleBasedOutlierDetection::new()
            .with_k(3)
            .with_contamination(0.2)
            .with_fast_abod(true);

        let fitted = abod.fit(&data, &()).unwrap();
        let predictions = fitted.predict(&data).unwrap();

        // Should work correctly in fast mode
        assert_eq!(predictions.len(), 6);
        let outlier_count = predictions.iter().filter(|&&x| x == 1).count();
        assert!(outlier_count >= 1);
    }

    #[test]
    fn test_abod_empty_input() {
        let data = Array2::zeros((0, 2));
        let abod = AngleBasedOutlierDetection::new();
        let result = abod.fit(&data, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_abod_invalid_k() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let abod = AngleBasedOutlierDetection::new().with_k(5); // k > n_samples
        let result = abod.fit(&data, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_abod_threshold() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let abod = AngleBasedOutlierDetection::new()
            .with_k(2) // Use k=2 for small dataset
            .with_contamination(0.25);
        let fitted = abod.fit(&data, &()).unwrap();

        assert!(fitted.threshold().is_some());
        assert!(fitted.threshold().unwrap() >= 0.0);
    }

    #[test]
    fn test_abod_training_scores() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let abod = AngleBasedOutlierDetection::new().with_k(2); // Use k=2 for small dataset
        let fitted = abod.fit(&data, &()).unwrap();

        let scores = fitted.training_scores();
        assert!(scores.is_some());
        assert_eq!(scores.unwrap().len(), 3);
    }
}
