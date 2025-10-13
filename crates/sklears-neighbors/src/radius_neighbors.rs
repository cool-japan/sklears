//! Radius-based neighbor algorithms for classification and regression

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict};
use sklears_core::types::Features;
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Strategy for determining radius in adaptive radius algorithms
#[derive(Debug, Clone, Copy)]
pub enum RadiusStrategy {
    /// Fixed radius for all points
    Fixed(Float),
    /// Adaptive radius based on k-th nearest neighbor distance
    Adaptive {
        /// Number of neighbors to consider for radius computation
        k: usize,
        /// Multiplier for the k-th distance (radius = multiplier * k_distance)
        multiplier: Float,
    },
}

/// Radius neighbors classifier
#[derive(Debug, Clone)]
pub struct RadiusNeighborsClassifier<State = sklears_core::traits::Untrained> {
    /// Radius of the neighborhood
    pub radius: Float,
    /// Distance metric
    pub metric: Distance,
    /// Weights for neighbors ('uniform' or 'distance')
    pub weights: crate::knn::WeightStrategy,
    /// Default class for samples with no neighbors
    pub outlier_label: Option<Int>,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Training labels (only available after fitting)
    pub(crate) y_train: Option<Array1<Int>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

/// Radius neighbors regressor
#[derive(Debug, Clone)]
pub struct RadiusNeighborsRegressor<State = sklears_core::traits::Untrained> {
    /// Radius of the neighborhood
    pub radius: Float,
    /// Distance metric
    pub metric: Distance,
    /// Weights for neighbors ('uniform' or 'distance')
    pub weights: crate::knn::WeightStrategy,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Training labels (only available after fitting)
    pub(crate) y_train: Option<Array1<Float>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl RadiusNeighborsClassifier {
    /// Create a new radius neighbors classifier
    pub fn new(radius: Float) -> Self {
        Self {
            radius,
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            outlier_label: None,
            x_train: None,
            y_train: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the weight strategy
    pub fn with_weights(mut self, weights: crate::knn::WeightStrategy) -> Self {
        self.weights = weights;
        self
    }

    /// Set the outlier label for samples with no neighbors
    pub fn with_outlier_label(mut self, outlier_label: Int) -> Self {
        self.outlier_label = Some(outlier_label);
        self
    }
}

impl RadiusNeighborsRegressor {
    /// Create a new radius neighbors regressor
    pub fn new(radius: Float) -> Self {
        Self {
            radius,
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            x_train: None,
            y_train: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the weight strategy
    pub fn with_weights(mut self, weights: crate::knn::WeightStrategy) -> Self {
        self.weights = weights;
        self
    }
}

impl Estimator for RadiusNeighborsClassifier {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for RadiusNeighborsRegressor {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for RadiusNeighborsClassifier {
    type Fitted = RadiusNeighborsClassifier<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Int>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            }
            .into());
        }

        if self.radius <= 0.0 {
            return Err(NeighborsError::InvalidRadius(self.radius).into());
        }

        Ok(RadiusNeighborsClassifier {
            radius: self.radius,
            metric: self.metric,
            weights: self.weights,
            outlier_label: self.outlier_label,
            x_train: Some(x.clone()),
            y_train: Some(y.clone()),
            _state: std::marker::PhantomData,
        })
    }
}

impl Fit<Features, Array1<Float>> for RadiusNeighborsRegressor {
    type Fitted = RadiusNeighborsRegressor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            }
            .into());
        }

        if self.radius <= 0.0 {
            return Err(NeighborsError::InvalidRadius(self.radius).into());
        }

        Ok(RadiusNeighborsRegressor {
            radius: self.radius,
            metric: self.metric,
            weights: self.weights,
            x_train: Some(x.clone()),
            y_train: Some(y.clone()),
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>> for RadiusNeighborsClassifier<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let x_train = self.x_train.as_ref().unwrap();
        let y_train = self.y_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let neighbors = self.find_neighbors(&sample, x_train, y_train)?;
            predictions[i] = self.predict_sample(&neighbors)?;
        }

        Ok(predictions)
    }
}

impl Predict<Features, Array1<Float>> for RadiusNeighborsRegressor<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Float>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let x_train = self.x_train.as_ref().unwrap();
        let y_train = self.y_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let neighbors = self.find_neighbors_regression(&sample, x_train, y_train)?;
            predictions[i] = self.predict_sample_regression(&neighbors)?;
        }

        Ok(predictions)
    }
}

impl RadiusNeighborsClassifier<sklears_core::traits::Trained> {
    /// Find all neighbors within radius for a sample
    fn find_neighbors(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        x_train: &Array2<Float>,
        y_train: &Array1<Int>,
    ) -> NeighborsResult<Vec<(Float, Int)>> {
        let distances = self.metric.to_matrix(sample, &x_train.view());

        let neighbors: Vec<(Float, Int)> = distances
            .iter()
            .zip(y_train.iter())
            .filter_map(|(&dist, &label)| {
                if dist <= self.radius {
                    Some((dist, label))
                } else {
                    None
                }
            })
            .collect();

        Ok(neighbors)
    }

    /// Predict class for a single sample based on its neighbors
    fn predict_sample(&self, neighbors: &[(Float, Int)]) -> NeighborsResult<Int> {
        if neighbors.is_empty() {
            return match self.outlier_label {
                Some(label) => Ok(label),
                None => Err(NeighborsError::NoNeighbors),
            };
        }

        match self.weights {
            crate::knn::WeightStrategy::Uniform => {
                // Count votes for each class
                let mut class_counts: HashMap<Int, usize> = HashMap::new();
                for (_, label) in neighbors {
                    *class_counts.entry(*label).or_insert(0) += 1;
                }

                // Return the class with the most votes
                class_counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(class, _)| class)
                    .ok_or(NeighborsError::NoNeighbors)
            }
            crate::knn::WeightStrategy::Distance => {
                // Weight votes by inverse distance
                let mut class_weights: HashMap<Int, Float> = HashMap::new();
                for (distance, label) in neighbors {
                    let weight = if *distance == 0.0 {
                        Float::INFINITY // Exact match gets infinite weight
                    } else {
                        1.0 / distance
                    };
                    *class_weights.entry(*label).or_insert(0.0) += weight;
                }

                // Return the class with the highest weighted vote
                class_weights
                    .into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(class, _)| class)
                    .ok_or(NeighborsError::NoNeighbors)
            }
        }
    }
}

impl RadiusNeighborsRegressor<sklears_core::traits::Trained> {
    /// Find all neighbors within radius for a sample (regression version)
    fn find_neighbors_regression(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
    ) -> NeighborsResult<Vec<(Float, Float)>> {
        let distances = self.metric.to_matrix(sample, &x_train.view());

        let neighbors: Vec<(Float, Float)> = distances
            .iter()
            .zip(y_train.iter())
            .filter_map(|(&dist, &target)| {
                if dist <= self.radius {
                    Some((dist, target))
                } else {
                    None
                }
            })
            .collect();

        Ok(neighbors)
    }

    /// Predict value for a single sample based on its neighbors
    fn predict_sample_regression(&self, neighbors: &[(Float, Float)]) -> NeighborsResult<Float> {
        if neighbors.is_empty() {
            return Err(NeighborsError::NoNeighbors);
        }

        match self.weights {
            crate::knn::WeightStrategy::Uniform => {
                // Simple average
                let sum: Float = neighbors.iter().map(|(_, value)| value).sum();
                Ok(sum / neighbors.len() as Float)
            }
            crate::knn::WeightStrategy::Distance => {
                // Weighted average by inverse distance
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (distance, value) in neighbors {
                    let weight = if *distance == 0.0 {
                        return Ok(*value); // Return exact match immediately
                    } else {
                        1.0 / distance
                    };
                    weighted_sum += weight * value;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    Ok(weighted_sum / total_weight)
                } else {
                    Err(NeighborsError::NoNeighbors)
                }
            }
        }
    }
}

/// Adaptive radius neighbors classifier that adjusts radius based on local density
#[derive(Debug, Clone)]
pub struct AdaptiveRadiusNeighborsClassifier<State = sklears_core::traits::Untrained> {
    /// Strategy for radius computation
    pub radius_strategy: RadiusStrategy,
    /// Distance metric
    pub metric: Distance,
    /// Weights for neighbors ('uniform' or 'distance')
    pub weights: crate::knn::WeightStrategy,
    /// Default class for samples with no neighbors
    pub outlier_label: Option<Int>,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Training labels (only available after fitting)
    pub(crate) y_train: Option<Array1<Int>>,
    /// Pre-computed k-distances for adaptive radius (only available after fitting)
    pub(crate) k_distances: Option<Array1<Float>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

/// Adaptive radius neighbors regressor that adjusts radius based on local density
#[derive(Debug, Clone)]
pub struct AdaptiveRadiusNeighborsRegressor<State = sklears_core::traits::Untrained> {
    /// Strategy for radius computation
    pub radius_strategy: RadiusStrategy,
    /// Distance metric
    pub metric: Distance,
    /// Weights for neighbors ('uniform' or 'distance')
    pub weights: crate::knn::WeightStrategy,
    /// Training data (only available after fitting)
    pub(crate) x_train: Option<Array2<Float>>,
    /// Training labels (only available after fitting)
    pub(crate) y_train: Option<Array1<Float>>,
    /// Pre-computed k-distances for adaptive radius (only available after fitting)
    pub(crate) k_distances: Option<Array1<Float>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl AdaptiveRadiusNeighborsClassifier {
    /// Create a new adaptive radius neighbors classifier with fixed radius
    pub fn new_fixed(radius: Float) -> Self {
        Self {
            radius_strategy: RadiusStrategy::Fixed(radius),
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            outlier_label: None,
            x_train: None,
            y_train: None,
            k_distances: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Create a new adaptive radius neighbors classifier with adaptive radius
    pub fn new_adaptive(k: usize, multiplier: Float) -> Self {
        Self {
            radius_strategy: RadiusStrategy::Adaptive { k, multiplier },
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            outlier_label: None,
            x_train: None,
            y_train: None,
            k_distances: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the weight strategy
    pub fn with_weights(mut self, weights: crate::knn::WeightStrategy) -> Self {
        self.weights = weights;
        self
    }

    /// Set the outlier label for samples with no neighbors
    pub fn with_outlier_label(mut self, outlier_label: Int) -> Self {
        self.outlier_label = Some(outlier_label);
        self
    }

    /// Compute k-distances for all training points
    fn compute_k_distances(
        x: &Array2<Float>,
        k: usize,
        metric: &Distance,
    ) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut k_distances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let distances = metric.to_matrix(&sample, &x.view());

            let mut sample_distances: Vec<Float> = distances.iter().copied().collect();
            sample_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // k-distance is the distance to the k-th nearest neighbor (excluding self)
            k_distances[i] = if sample_distances.len() > k {
                sample_distances[k] // k+1-th element (0-indexed) since we include self
            } else {
                sample_distances.last().copied().unwrap_or(0.0)
            };
        }

        Ok(k_distances)
    }
}

impl AdaptiveRadiusNeighborsRegressor {
    /// Create a new adaptive radius neighbors regressor with fixed radius
    pub fn new_fixed(radius: Float) -> Self {
        Self {
            radius_strategy: RadiusStrategy::Fixed(radius),
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            x_train: None,
            y_train: None,
            k_distances: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Create a new adaptive radius neighbors regressor with adaptive radius
    pub fn new_adaptive(k: usize, multiplier: Float) -> Self {
        Self {
            radius_strategy: RadiusStrategy::Adaptive { k, multiplier },
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            x_train: None,
            y_train: None,
            k_distances: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the weight strategy
    pub fn with_weights(mut self, weights: crate::knn::WeightStrategy) -> Self {
        self.weights = weights;
        self
    }

    /// Compute k-distances for all training points
    fn compute_k_distances(
        x: &Array2<Float>,
        k: usize,
        metric: &Distance,
    ) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut k_distances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let distances = metric.to_matrix(&sample, &x.view());

            let mut sample_distances: Vec<Float> = distances.iter().copied().collect();
            sample_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // k-distance is the distance to the k-th nearest neighbor (excluding self)
            k_distances[i] = if sample_distances.len() > k {
                sample_distances[k] // k+1-th element (0-indexed) since we include self
            } else {
                sample_distances.last().copied().unwrap_or(0.0)
            };
        }

        Ok(k_distances)
    }
}

impl Estimator for AdaptiveRadiusNeighborsClassifier {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for AdaptiveRadiusNeighborsRegressor {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for AdaptiveRadiusNeighborsClassifier {
    type Fitted = AdaptiveRadiusNeighborsClassifier<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Int>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            }
            .into());
        }

        // Validate radius strategy
        match self.radius_strategy {
            RadiusStrategy::Fixed(radius) => {
                if radius <= 0.0 {
                    return Err(NeighborsError::InvalidRadius(radius).into());
                }
            }
            RadiusStrategy::Adaptive { k, multiplier } => {
                if k == 0 || k >= x.nrows() {
                    return Err(NeighborsError::InvalidNeighbors(k).into());
                }
                if multiplier <= 0.0 {
                    return Err(NeighborsError::InvalidInput(format!(
                        "Multiplier must be positive, got {}",
                        multiplier
                    ))
                    .into());
                }
            }
        }

        // Pre-compute k-distances for adaptive radius
        let k_distances = match self.radius_strategy {
            RadiusStrategy::Fixed(_) => None,
            RadiusStrategy::Adaptive { k, .. } => {
                Some(Self::compute_k_distances(x, k, &self.metric)?)
            }
        };

        Ok(AdaptiveRadiusNeighborsClassifier {
            radius_strategy: self.radius_strategy,
            metric: self.metric,
            weights: self.weights,
            outlier_label: self.outlier_label,
            x_train: Some(x.clone()),
            y_train: Some(y.clone()),
            k_distances,
            _state: std::marker::PhantomData,
        })
    }
}

impl Fit<Features, Array1<Float>> for AdaptiveRadiusNeighborsRegressor {
    type Fitted = AdaptiveRadiusNeighborsRegressor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            }
            .into());
        }

        // Validate radius strategy
        match self.radius_strategy {
            RadiusStrategy::Fixed(radius) => {
                if radius <= 0.0 {
                    return Err(NeighborsError::InvalidRadius(radius).into());
                }
            }
            RadiusStrategy::Adaptive { k, multiplier } => {
                if k == 0 || k >= x.nrows() {
                    return Err(NeighborsError::InvalidNeighbors(k).into());
                }
                if multiplier <= 0.0 {
                    return Err(NeighborsError::InvalidInput(format!(
                        "Multiplier must be positive, got {}",
                        multiplier
                    ))
                    .into());
                }
            }
        }

        // Pre-compute k-distances for adaptive radius
        let k_distances = match self.radius_strategy {
            RadiusStrategy::Fixed(_) => None,
            RadiusStrategy::Adaptive { k, .. } => {
                Some(Self::compute_k_distances(x, k, &self.metric)?)
            }
        };

        Ok(AdaptiveRadiusNeighborsRegressor {
            radius_strategy: self.radius_strategy,
            metric: self.metric,
            weights: self.weights,
            x_train: Some(x.clone()),
            y_train: Some(y.clone()),
            k_distances,
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>>
    for AdaptiveRadiusNeighborsClassifier<sklears_core::traits::Trained>
{
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let x_train = self.x_train.as_ref().unwrap();
        let y_train = self.y_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let neighbors = self.find_adaptive_neighbors(&sample, x_train, y_train)?;
            predictions[i] = self.predict_sample(&neighbors)?;
        }

        Ok(predictions)
    }
}

impl Predict<Features, Array1<Float>>
    for AdaptiveRadiusNeighborsRegressor<sklears_core::traits::Trained>
{
    fn predict(&self, x: &Features) -> Result<Array1<Float>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let x_train = self.x_train.as_ref().unwrap();
        let y_train = self.y_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_train.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let neighbors = self.find_adaptive_neighbors_regression(&sample, x_train, y_train)?;
            predictions[i] = self.predict_sample_regression(&neighbors)?;
        }

        Ok(predictions)
    }
}

impl AdaptiveRadiusNeighborsClassifier<sklears_core::traits::Trained> {
    /// Compute k-distances for all training points
    fn compute_k_distances(
        x: &Array2<Float>,
        k: usize,
        metric: &Distance,
    ) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut k_distances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let distances = metric.to_matrix(&sample, &x.view());

            let mut sample_distances: Vec<Float> = distances.iter().copied().collect();
            sample_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // k-distance is the distance to the k-th nearest neighbor (excluding self)
            k_distances[i] = if sample_distances.len() > k {
                sample_distances[k] // k-th neighbor (0-indexed, self is at index 0)
            } else {
                sample_distances.last().copied().unwrap_or(0.0)
            };
        }

        Ok(k_distances)
    }

    /// Find all adaptive neighbors for a sample (classification)
    fn find_adaptive_neighbors(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        x_train: &Array2<Float>,
        y_train: &Array1<Int>,
    ) -> NeighborsResult<Vec<(Float, Int)>> {
        let distances = self.metric.to_matrix(sample, &x_train.view());

        match self.radius_strategy {
            RadiusStrategy::Fixed(radius) => {
                let neighbors: Vec<(Float, Int)> = distances
                    .iter()
                    .zip(y_train.iter())
                    .filter_map(|(&dist, &label)| {
                        if dist <= radius {
                            Some((dist, label))
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(neighbors)
            }
            RadiusStrategy::Adaptive { k: _, multiplier } => {
                // For adaptive radius in prediction, use the closest training point's k-distance
                let k_distances = self.k_distances.as_ref().unwrap();

                // Find the closest training point to determine adaptive radius
                let min_distance_idx = distances
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let adaptive_radius = k_distances[min_distance_idx] * multiplier;

                let neighbors: Vec<(Float, Int)> = distances
                    .iter()
                    .zip(y_train.iter())
                    .filter_map(|(&dist, &label)| {
                        if dist <= adaptive_radius {
                            Some((dist, label))
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(neighbors)
            }
        }
    }

    /// Predict class for a single sample based on its neighbors
    fn predict_sample(&self, neighbors: &[(Float, Int)]) -> NeighborsResult<Int> {
        if neighbors.is_empty() {
            return match self.outlier_label {
                Some(label) => Ok(label),
                None => Err(NeighborsError::NoNeighbors),
            };
        }

        match self.weights {
            crate::knn::WeightStrategy::Uniform => {
                let mut class_counts: HashMap<Int, usize> = HashMap::new();
                for (_, label) in neighbors {
                    *class_counts.entry(*label).or_insert(0) += 1;
                }

                class_counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(class, _)| class)
                    .ok_or(NeighborsError::NoNeighbors)
            }
            crate::knn::WeightStrategy::Distance => {
                let mut class_weights: HashMap<Int, Float> = HashMap::new();
                for (distance, label) in neighbors {
                    let weight = if *distance == 0.0 {
                        Float::INFINITY
                    } else {
                        1.0 / distance
                    };
                    *class_weights.entry(*label).or_insert(0.0) += weight;
                }

                class_weights
                    .into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(class, _)| class)
                    .ok_or(NeighborsError::NoNeighbors)
            }
        }
    }
}

impl AdaptiveRadiusNeighborsRegressor<sklears_core::traits::Trained> {
    /// Compute k-distances for all training points
    fn compute_k_distances(
        x: &Array2<Float>,
        k: usize,
        metric: &Distance,
    ) -> NeighborsResult<Array1<Float>> {
        let n_samples = x.nrows();
        let mut k_distances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = x.row(i);
            let distances = metric.to_matrix(&sample, &x.view());

            let mut sample_distances: Vec<Float> = distances.iter().copied().collect();
            sample_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            k_distances[i] = if sample_distances.len() > k {
                sample_distances[k]
            } else {
                sample_distances.last().copied().unwrap_or(0.0)
            };
        }

        Ok(k_distances)
    }

    /// Find all adaptive neighbors for a sample (regression)
    fn find_adaptive_neighbors_regression(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
    ) -> NeighborsResult<Vec<(Float, Float)>> {
        let distances = self.metric.to_matrix(sample, &x_train.view());

        match self.radius_strategy {
            RadiusStrategy::Fixed(radius) => {
                let neighbors: Vec<(Float, Float)> = distances
                    .iter()
                    .zip(y_train.iter())
                    .filter_map(|(&dist, &target)| {
                        if dist <= radius {
                            Some((dist, target))
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(neighbors)
            }
            RadiusStrategy::Adaptive { k: _, multiplier } => {
                let k_distances = self.k_distances.as_ref().unwrap();

                let min_distance_idx = distances
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let adaptive_radius = k_distances[min_distance_idx] * multiplier;

                let neighbors: Vec<(Float, Float)> = distances
                    .iter()
                    .zip(y_train.iter())
                    .filter_map(|(&dist, &target)| {
                        if dist <= adaptive_radius {
                            Some((dist, target))
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(neighbors)
            }
        }
    }

    /// Predict value for a single sample based on its neighbors
    fn predict_sample_regression(&self, neighbors: &[(Float, Float)]) -> NeighborsResult<Float> {
        if neighbors.is_empty() {
            return Err(NeighborsError::NoNeighbors);
        }

        match self.weights {
            crate::knn::WeightStrategy::Uniform => {
                let sum: Float = neighbors.iter().map(|(_, value)| value).sum();
                Ok(sum / neighbors.len() as Float)
            }
            crate::knn::WeightStrategy::Distance => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (distance, value) in neighbors {
                    let weight = if *distance == 0.0 {
                        return Ok(*value);
                    } else {
                        1.0 / distance
                    };
                    weighted_sum += weight * value;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    Ok(weighted_sum / total_weight)
                } else {
                    Err(NeighborsError::NoNeighbors)
                }
            }
        }
    }
}

/// Find all neighbors within a given radius
pub fn radius_neighbors<T: Clone>(
    sample: &scirs2_core::ndarray::ArrayView1<Float>,
    x: &Array2<Float>,
    y: &Array1<T>,
    radius: Float,
    metric: Distance,
) -> Vec<(usize, Float, T)> {
    let distances = metric.to_matrix(sample, &x.view());

    distances
        .iter()
        .zip(y.iter())
        .enumerate()
        .filter_map(|(i, (&dist, label))| {
            if dist <= radius {
                Some((i, dist, label.clone()))
            } else {
                None
            }
        })
        .collect()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_radius_neighbors_classifier() {
        // Create a dataset with clear clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0 cluster
                1.1, 1.1, // Class 0 cluster
                1.2, 1.2, // Class 0 cluster
                5.0, 5.0, // Class 1 cluster
                5.1, 5.1, // Class 1 cluster
                5.2, 5.2, // Class 1 cluster
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let classifier = RadiusNeighborsClassifier::new(0.5);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Test prediction within clusters
        let x_test = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.05, 1.05, // Should be class 0
                5.05, 5.05, // Should be class 1
            ],
        )
        .unwrap();

        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_radius_neighbors_regressor() {
        // Create a simple dataset
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0]; // y = 2 * x

        let regressor = RadiusNeighborsRegressor::new(0.5);
        let fitted = regressor.fit(&x, &y).unwrap();

        // Test prediction
        let x_test = Array2::from_shape_vec((1, 1), vec![2.1]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        // Should predict close to 4.0 (only the point at x=2 is within radius)
        assert_abs_diff_eq!(predictions[0], 4.0, epsilon = 0.1);
    }

    #[test]
    fn test_radius_neighbors_no_neighbors() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 10.0]).unwrap();
        let y = array![0, 1];

        let classifier = RadiusNeighborsClassifier::new(0.5);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Test point far from any training data
        let x_test = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let result = fitted.predict(&x_test);

        // Should fail with no neighbors
        assert!(result.is_err());
    }

    #[test]
    fn test_radius_neighbors_with_outlier_label() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 10.0]).unwrap();
        let y = array![0, 1];

        let classifier = RadiusNeighborsClassifier::new(0.5).with_outlier_label(-1);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Test point far from any training data
        let x_test = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        // Should predict the outlier label
        assert_eq!(predictions[0], -1);
    }

    #[test]
    fn test_radius_neighbors_function() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 5.0]).unwrap();
        let y = array![10, 20, 50];

        let sample = array![1.5];
        let neighbors = radius_neighbors(&sample.view(), &x, &y, 1.0, Distance::Euclidean);

        // Should find neighbors at positions 0 (dist=0.5) and 1 (dist=0.5)
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].2, 10); // First neighbor
        assert_eq!(neighbors[1].2, 20); // Second neighbor
    }

    #[test]
    fn test_adaptive_radius_neighbors_classifier_fixed() {
        // Test that AdaptiveRadiusNeighborsClassifier with fixed radius works like RadiusNeighborsClassifier
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = AdaptiveRadiusNeighborsClassifier::new_fixed(0.5);
        let fitted = classifier.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 5.05, 5.05]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_adaptive_radius_neighbors_classifier_adaptive() {
        // Test adaptive radius functionality
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0 - dense cluster
                1.1, 1.1, // Class 0 - dense cluster
                1.2, 1.2, // Class 0 - dense cluster
                5.0, 5.0, // Class 1 - sparse area
                7.0, 7.0, // Class 1 - sparse area
                9.0, 9.0, // Class 1 - sparse area
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        // Use adaptive radius based on 2nd nearest neighbor with multiplier 1.5
        let classifier = AdaptiveRadiusNeighborsClassifier::new_adaptive(2, 1.5);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Test points in both dense and sparse regions
        let x_test = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.15, 1.15, // In dense cluster
                6.0, 6.0, // Between sparse points
            ],
        )
        .unwrap();

        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 2);
        // Specific predictions depend on the adaptive radius computation
        // but both should be valid predictions
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_adaptive_radius_neighbors_regressor_fixed() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0]; // y = 2 * x

        let regressor = AdaptiveRadiusNeighborsRegressor::new_fixed(0.5);
        let fitted = regressor.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((1, 1), vec![2.1]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        // Should predict close to 4.0
        assert_abs_diff_eq!(predictions[0], 4.0, epsilon = 0.1);
    }

    #[test]
    fn test_adaptive_radius_neighbors_regressor_adaptive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 1.1, 1.2, 5.0, 5.5]).unwrap();
        let y = array![1.0, 1.1, 1.2, 5.0, 5.5];

        // Use adaptive radius with k=1 and multiplier=2.0
        let regressor = AdaptiveRadiusNeighborsRegressor::new_adaptive(1, 2.0);
        let fitted = regressor.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((2, 1), vec![1.05, 5.25]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 2);
        // Predictions should be reasonable given the adaptive radius
        assert!(predictions[0] > 0.5 && predictions[0] < 2.0);
        assert!(predictions[1] > 4.0 && predictions[1] < 6.0);
    }

    #[test]
    fn test_adaptive_radius_neighbors_errors() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = array![0, 1, 2];

        // Test invalid k (too large)
        let classifier = AdaptiveRadiusNeighborsClassifier::new_adaptive(5, 1.0);
        let result = classifier.fit(&x, &y);
        assert!(result.is_err());

        // Test invalid multiplier (negative)
        let classifier = AdaptiveRadiusNeighborsClassifier::new_adaptive(1, -1.0);
        let result = classifier.fit(&x, &y);
        assert!(result.is_err());

        // Test invalid fixed radius (negative)
        let classifier = AdaptiveRadiusNeighborsClassifier::new_fixed(-1.0);
        let result = classifier.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_radius_strategy() {
        // Test RadiusStrategy enum
        let fixed_strategy = RadiusStrategy::Fixed(1.0);
        let adaptive_strategy = RadiusStrategy::Adaptive {
            k: 3,
            multiplier: 1.5,
        };

        // Test that we can pattern match on the strategies
        match fixed_strategy {
            RadiusStrategy::Fixed(radius) => assert_eq!(radius, 1.0),
            _ => panic!("Expected Fixed strategy"),
        }

        match adaptive_strategy {
            RadiusStrategy::Adaptive { k, multiplier } => {
                assert_eq!(k, 3);
                assert_eq!(multiplier, 1.5);
            }
            _ => panic!("Expected Adaptive strategy"),
        }
    }
}
