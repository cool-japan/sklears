//! Incremental and streaming nearest neighbor algorithms

use crate::{Distance, NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, ArrayView1, Axis};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict};
use sklears_core::types::{Features, Float, Int};
use std::collections::{HashMap, VecDeque};

/// Incremental KNN classifier for streaming data
#[derive(Debug, Clone)]
pub struct IncrementalKNeighborsClassifier<State = sklears_core::traits::Untrained> {
    /// Number of neighbors to consider
    pub k: usize,
    /// Distance metric to use
    pub metric: Distance,
    /// Weight strategy for neighbors
    pub weights: crate::knn::WeightStrategy,
    /// Maximum number of samples to keep in memory
    pub max_samples: usize,
    /// Strategy for handling memory overflow
    pub memory_strategy: MemoryStrategy,
    /// Training data buffer (only available after fitting)
    pub(crate) x_buffer: Option<VecDeque<Array1<Float>>>,
    /// Training labels buffer (only available after fitting)
    pub(crate) y_buffer: Option<VecDeque<Int>>,
    /// Total number of samples seen
    pub(crate) n_samples_seen: usize,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

/// Incremental KNN regressor for streaming data
#[derive(Debug, Clone)]
pub struct IncrementalKNeighborsRegressor<State = sklears_core::traits::Untrained> {
    pub k: usize,
    pub metric: Distance,
    pub weights: crate::knn::WeightStrategy,
    pub max_samples: usize,
    pub memory_strategy: MemoryStrategy,
    pub(crate) x_buffer: Option<VecDeque<Array1<Float>>>,
    pub(crate) y_buffer: Option<VecDeque<Float>>,
    pub(crate) n_samples_seen: usize,
    pub(crate) _state: std::marker::PhantomData<State>,
}

/// Strategy for handling memory overflow in incremental learning
#[derive(Debug, Clone, Copy)]
pub enum MemoryStrategy {
    /// Keep only the most recent samples (FIFO)
    KeepRecent,
    /// Keep a random subset of samples
    Random,
    /// Keep samples with highest diversity (farthest from existing samples)
    Diversity,
    /// Keep samples that are most representative (closest to cluster centers)
    Representative,
}

impl IncrementalKNeighborsClassifier {
    /// Create a new incremental KNN classifier
    pub fn new(k: usize, max_samples: usize) -> Self {
        Self {
            k,
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            max_samples,
            memory_strategy: MemoryStrategy::KeepRecent,
            x_buffer: None,
            y_buffer: None,
            n_samples_seen: 0,
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

    /// Set the memory strategy
    pub fn with_memory_strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.memory_strategy = strategy;
        self
    }
}

impl IncrementalKNeighborsRegressor {
    /// Create a new incremental KNN regressor
    pub fn new(k: usize, max_samples: usize) -> Self {
        Self {
            k,
            metric: Distance::default(),
            weights: crate::knn::WeightStrategy::Uniform,
            max_samples,
            memory_strategy: MemoryStrategy::KeepRecent,
            x_buffer: None,
            y_buffer: None,
            n_samples_seen: 0,
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

    /// Set the memory strategy
    pub fn with_memory_strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.memory_strategy = strategy;
        self
    }
}

impl Estimator for IncrementalKNeighborsClassifier {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for IncrementalKNeighborsRegressor {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for IncrementalKNeighborsClassifier {
    type Fitted = IncrementalKNeighborsClassifier<sklears_core::traits::Trained>;

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

        if self.k == 0 || self.k > self.max_samples {
            return Err(NeighborsError::InvalidNeighbors(self.k).into());
        }

        // Initialize buffers
        let mut x_buffer = VecDeque::with_capacity(self.max_samples);
        let mut y_buffer = VecDeque::with_capacity(self.max_samples);

        // Add initial samples
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            x_buffer.push_back(row.to_owned());
            y_buffer.push_back(y[i]);

            if x_buffer.len() > self.max_samples {
                // Apply memory strategy
                match self.memory_strategy {
                    MemoryStrategy::KeepRecent => {
                        x_buffer.pop_front();
                        y_buffer.pop_front();
                    }
                    _ => {
                        // For now, default to KeepRecent for other strategies
                        x_buffer.pop_front();
                        y_buffer.pop_front();
                    }
                }
            }
        }

        Ok(IncrementalKNeighborsClassifier {
            k: self.k,
            metric: self.metric,
            weights: self.weights,
            max_samples: self.max_samples,
            memory_strategy: self.memory_strategy,
            x_buffer: Some(x_buffer),
            y_buffer: Some(y_buffer),
            n_samples_seen: x.nrows(),
            _state: std::marker::PhantomData,
        })
    }
}

impl Fit<Features, Array1<Float>> for IncrementalKNeighborsRegressor {
    type Fitted = IncrementalKNeighborsRegressor<sklears_core::traits::Trained>;

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

        if self.k == 0 || self.k > self.max_samples {
            return Err(NeighborsError::InvalidNeighbors(self.k).into());
        }

        // Initialize buffers
        let mut x_buffer = VecDeque::with_capacity(self.max_samples);
        let mut y_buffer = VecDeque::with_capacity(self.max_samples);

        // Add initial samples
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            x_buffer.push_back(row.to_owned());
            y_buffer.push_back(y[i]);

            if x_buffer.len() > self.max_samples {
                match self.memory_strategy {
                    MemoryStrategy::KeepRecent => {
                        x_buffer.pop_front();
                        y_buffer.pop_front();
                    }
                    _ => {
                        x_buffer.pop_front();
                        y_buffer.pop_front();
                    }
                }
            }
        }

        Ok(IncrementalKNeighborsRegressor {
            k: self.k,
            metric: self.metric,
            weights: self.weights,
            max_samples: self.max_samples,
            memory_strategy: self.memory_strategy,
            x_buffer: Some(x_buffer),
            y_buffer: Some(y_buffer),
            n_samples_seen: x.nrows(),
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>>
    for IncrementalKNeighborsClassifier<sklears_core::traits::Trained>
{
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let x_buffer = self.x_buffer.as_ref().unwrap();
        let y_buffer = self.y_buffer.as_ref().unwrap();

        if x_buffer.is_empty() {
            return Err(
                NeighborsError::InvalidInput("No training data available".to_string()).into(),
            );
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, query_point) in x.axis_iter(Axis(0)).enumerate() {
            let neighbors = self.find_neighbors(&query_point, x_buffer, y_buffer)?;
            predictions[i] = self.predict_sample(&neighbors)?;
        }

        Ok(predictions)
    }
}

impl Predict<Features, Array1<Float>>
    for IncrementalKNeighborsRegressor<sklears_core::traits::Trained>
{
    fn predict(&self, x: &Features) -> Result<Array1<Float>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let x_buffer = self.x_buffer.as_ref().unwrap();
        let y_buffer = self.y_buffer.as_ref().unwrap();

        if x_buffer.is_empty() {
            return Err(
                NeighborsError::InvalidInput("No training data available".to_string()).into(),
            );
        }

        let mut predictions = Array1::zeros(x.nrows());

        for (i, query_point) in x.axis_iter(Axis(0)).enumerate() {
            let neighbors = self.find_neighbors_regression(&query_point, x_buffer, y_buffer)?;
            predictions[i] = self.predict_sample_regression(&neighbors)?;
        }

        Ok(predictions)
    }
}

impl IncrementalKNeighborsClassifier<sklears_core::traits::Trained> {
    /// Add new samples to the incremental classifier
    pub fn partial_fit(&mut self, x: &Features, y: &Array1<Int>) -> NeighborsResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            });
        }

        // Add new samples
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let x_buffer = self.x_buffer.as_mut().unwrap();
            let y_buffer = self.y_buffer.as_mut().unwrap();

            x_buffer.push_back(row.to_owned());
            y_buffer.push_back(y[i]);

            if x_buffer.len() > self.max_samples {
                // Apply memory strategy directly to avoid borrowing conflicts
                match self.memory_strategy {
                    MemoryStrategy::KeepRecent => {
                        self.x_buffer.as_mut().unwrap().pop_front();
                        self.y_buffer.as_mut().unwrap().pop_front();
                    }
                    MemoryStrategy::Random => {
                        // Remove a random sample
                        use scirs2_core::random::*;
                        let mut rng = thread_rng();
                        let buffer_len = self.x_buffer.as_ref().unwrap().len();
                        let remove_idx = rng.gen_range(0..buffer_len);
                        self.x_buffer.as_mut().unwrap().remove(remove_idx);
                        self.y_buffer.as_mut().unwrap().remove(remove_idx);
                    }
                    MemoryStrategy::Diversity => {
                        // Remove the sample that is most similar to others (lowest diversity)
                        let remove_idx = self.find_least_diverse_sample()?;
                        self.x_buffer.as_mut().unwrap().remove(remove_idx);
                        self.y_buffer.as_mut().unwrap().remove(remove_idx);
                    }
                    MemoryStrategy::Representative => {
                        // Remove the sample that is most representative (closest to centroid)
                        let remove_idx = self.find_most_representative_sample()?;
                        self.x_buffer.as_mut().unwrap().remove(remove_idx);
                        self.y_buffer.as_mut().unwrap().remove(remove_idx);
                    }
                }
            }
        }

        self.n_samples_seen += x.nrows();
        Ok(())
    }

    /// Find k-nearest neighbors for a query point
    fn find_neighbors(
        &self,
        query_point: &ArrayView1<Float>,
        x_buffer: &VecDeque<Array1<Float>>,
        y_buffer: &VecDeque<Int>,
    ) -> NeighborsResult<Vec<(Float, Int)>> {
        let mut distances: Vec<(Float, Int)> = x_buffer
            .iter()
            .zip(y_buffer.iter())
            .map(|(sample, &label)| {
                let distance = self.metric.calculate(query_point, &sample.view());
                (distance, label)
            })
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(self.k);

        Ok(distances)
    }

    /// Predict class for a single sample
    fn predict_sample(&self, neighbors: &[(Float, Int)]) -> NeighborsResult<Int> {
        if neighbors.is_empty() {
            return Err(NeighborsError::NoNeighbors);
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

    /// Apply the memory strategy when buffer overflows
    fn apply_memory_strategy(
        &self,
        x_buffer: &mut VecDeque<Array1<Float>>,
        y_buffer: &mut VecDeque<Int>,
    ) -> NeighborsResult<()> {
        match self.memory_strategy {
            MemoryStrategy::KeepRecent => {
                x_buffer.pop_front();
                y_buffer.pop_front();
            }
            MemoryStrategy::Random => {
                // Remove a random sample
                use scirs2_core::random::*;
                let mut rng = thread_rng();
                let remove_idx = rng.gen_range(0..x_buffer.len());
                x_buffer.remove(remove_idx);
                y_buffer.remove(remove_idx);
            }
            MemoryStrategy::Diversity | MemoryStrategy::Representative => {
                // For now, default to KeepRecent
                // These would require more sophisticated algorithms
                x_buffer.pop_front();
                y_buffer.pop_front();
            }
        }
        Ok(())
    }
}

impl IncrementalKNeighborsRegressor<sklears_core::traits::Trained> {
    /// Add new samples to the incremental regressor
    pub fn partial_fit(&mut self, x: &Features, y: &Array1<Float>) -> NeighborsResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            });
        }

        // Add new samples
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let x_buffer = self.x_buffer.as_mut().unwrap();
            let y_buffer = self.y_buffer.as_mut().unwrap();

            x_buffer.push_back(row.to_owned());
            y_buffer.push_back(y[i]);

            if x_buffer.len() > self.max_samples {
                // Apply memory strategy directly to avoid borrowing conflicts
                match self.memory_strategy {
                    MemoryStrategy::KeepRecent => {
                        self.x_buffer.as_mut().unwrap().pop_front();
                        self.y_buffer.as_mut().unwrap().pop_front();
                    }
                    MemoryStrategy::Random => {
                        // Remove a random sample
                        use scirs2_core::random::*;
                        let mut rng = thread_rng();
                        let buffer_len = self.x_buffer.as_ref().unwrap().len();
                        let remove_idx = rng.gen_range(0..buffer_len);
                        self.x_buffer.as_mut().unwrap().remove(remove_idx);
                        self.y_buffer.as_mut().unwrap().remove(remove_idx);
                    }
                    MemoryStrategy::Diversity => {
                        // Remove the sample that is most similar to others (lowest diversity)
                        let remove_idx = self.find_least_diverse_sample()?;
                        self.x_buffer.as_mut().unwrap().remove(remove_idx);
                        self.y_buffer.as_mut().unwrap().remove(remove_idx);
                    }
                    MemoryStrategy::Representative => {
                        // Remove the sample that is most representative (closest to centroid)
                        let remove_idx = self.find_most_representative_sample()?;
                        self.x_buffer.as_mut().unwrap().remove(remove_idx);
                        self.y_buffer.as_mut().unwrap().remove(remove_idx);
                    }
                }
            }
        }

        self.n_samples_seen += x.nrows();
        Ok(())
    }

    /// Find k-nearest neighbors for a query point (regression)
    fn find_neighbors_regression(
        &self,
        query_point: &ArrayView1<Float>,
        x_buffer: &VecDeque<Array1<Float>>,
        y_buffer: &VecDeque<Float>,
    ) -> NeighborsResult<Vec<(Float, Float)>> {
        let mut distances: Vec<(Float, Float)> = x_buffer
            .iter()
            .zip(y_buffer.iter())
            .map(|(sample, &target)| {
                let distance = self.metric.calculate(query_point, &sample.view());
                (distance, target)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(self.k);

        Ok(distances)
    }

    /// Predict value for a single sample
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

    /// Apply the memory strategy when buffer overflows (regression version)
    fn apply_memory_strategy(
        &self,
        x_buffer: &mut VecDeque<Array1<Float>>,
        y_buffer: &mut VecDeque<Float>,
    ) -> NeighborsResult<()> {
        match self.memory_strategy {
            MemoryStrategy::KeepRecent => {
                x_buffer.pop_front();
                y_buffer.pop_front();
            }
            MemoryStrategy::Random => {
                use scirs2_core::random::*;
                let mut rng = thread_rng();
                let remove_idx = rng.gen_range(0..x_buffer.len());
                x_buffer.remove(remove_idx);
                y_buffer.remove(remove_idx);
            }
            MemoryStrategy::Diversity | MemoryStrategy::Representative => {
                x_buffer.pop_front();
                y_buffer.pop_front();
            }
        }
        Ok(())
    }

    /// Get the current number of samples in the buffer
    pub fn n_samples_in_buffer(&self) -> usize {
        self.x_buffer.as_ref().map_or(0, |buf| buf.len())
    }

    /// Get the total number of samples seen
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Find the least diverse sample (most similar to others)
    fn find_least_diverse_sample(&self) -> NeighborsResult<usize> {
        let x_buffer = self.x_buffer.as_ref().unwrap();
        if x_buffer.len() <= 1 {
            return Ok(0);
        }

        let mut min_diversity = Float::INFINITY;
        let mut least_diverse_idx = 0;

        for (i, sample_i) in x_buffer.iter().enumerate() {
            let mut total_distance = 0.0;
            for (j, sample_j) in x_buffer.iter().enumerate() {
                if i != j {
                    total_distance += self.metric.calculate(&sample_i.view(), &sample_j.view());
                }
            }
            let diversity = total_distance / (x_buffer.len() - 1) as Float;

            if diversity < min_diversity {
                min_diversity = diversity;
                least_diverse_idx = i;
            }
        }

        Ok(least_diverse_idx)
    }

    /// Find the most representative sample (closest to centroid)
    fn find_most_representative_sample(&self) -> NeighborsResult<usize> {
        let x_buffer = self.x_buffer.as_ref().unwrap();
        if x_buffer.len() <= 1 {
            return Ok(0);
        }

        // Calculate centroid
        let n_features = x_buffer[0].len();
        let mut centroid = Array1::zeros(n_features);
        for sample in x_buffer {
            centroid += sample;
        }
        centroid /= x_buffer.len() as Float;

        // Find closest sample to centroid
        let mut min_distance = Float::INFINITY;
        let mut most_representative_idx = 0;

        for (i, sample) in x_buffer.iter().enumerate() {
            let distance = self.metric.calculate(&sample.view(), &centroid.view());
            if distance < min_distance {
                min_distance = distance;
                most_representative_idx = i;
            }
        }

        Ok(most_representative_idx)
    }
}

impl IncrementalKNeighborsClassifier<sklears_core::traits::Trained> {
    /// Get the current number of samples in the buffer
    pub fn n_samples_in_buffer(&self) -> usize {
        self.x_buffer.as_ref().map_or(0, |buf| buf.len())
    }

    /// Get the total number of samples seen
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Find the least diverse sample (most similar to others)
    fn find_least_diverse_sample(&self) -> NeighborsResult<usize> {
        let x_buffer = self.x_buffer.as_ref().unwrap();
        if x_buffer.len() <= 1 {
            return Ok(0);
        }

        let mut min_diversity = Float::INFINITY;
        let mut least_diverse_idx = 0;

        for (i, sample_i) in x_buffer.iter().enumerate() {
            let mut total_distance = 0.0;
            for (j, sample_j) in x_buffer.iter().enumerate() {
                if i != j {
                    total_distance += self.metric.calculate(&sample_i.view(), &sample_j.view());
                }
            }
            let diversity = total_distance / (x_buffer.len() - 1) as Float;

            if diversity < min_diversity {
                min_diversity = diversity;
                least_diverse_idx = i;
            }
        }

        Ok(least_diverse_idx)
    }

    /// Find the most representative sample (closest to centroid)
    fn find_most_representative_sample(&self) -> NeighborsResult<usize> {
        let x_buffer = self.x_buffer.as_ref().unwrap();
        if x_buffer.len() <= 1 {
            return Ok(0);
        }

        // Calculate centroid
        let n_features = x_buffer[0].len();
        let mut centroid = Array1::zeros(n_features);
        for sample in x_buffer {
            centroid += sample;
        }
        centroid /= x_buffer.len() as Float;

        // Find closest sample to centroid
        let mut min_distance = Float::INFINITY;
        let mut most_representative_idx = 0;

        for (i, sample) in x_buffer.iter().enumerate() {
            let distance = self.metric.calculate(&sample.view(), &centroid.view());
            if distance < min_distance {
                min_distance = distance;
                most_representative_idx = i;
            }
        }

        Ok(most_representative_idx)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};
    use sklears_core::traits::Predict;

    #[test]
    fn test_incremental_knn_classifier() {
        // Initial training data
        let x_train = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y_train = array![0, 0, 1, 1];

        let classifier = IncrementalKNeighborsClassifier::new(3, 10);
        let mut fitted = classifier.fit(&x_train, &y_train).unwrap();

        // Test initial prediction
        let x_test = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 5.05, 5.05]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);

        // Add more data incrementally
        let x_new = Array2::from_shape_vec((2, 2), vec![2.0, 2.0, 6.0, 6.0]).unwrap();
        let y_new = array![0, 1];
        fitted.partial_fit(&x_new, &y_new).unwrap();

        // Test prediction after incremental update
        let predictions_new = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions_new.len(), 2);

        // Check that buffer size is correct
        assert_eq!(fitted.n_samples_in_buffer(), 6);
        assert_eq!(fitted.n_samples_seen(), 6);
    }

    #[test]
    fn test_incremental_knn_regressor() {
        let x_train = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_train = array![2.0, 4.0, 6.0, 8.0]; // y = 2 * x

        let regressor = IncrementalKNeighborsRegressor::new(2, 10);
        let mut fitted = regressor.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((1, 1), vec![2.5]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        // Should predict around 5.0 (average of 4.0 and 6.0)
        assert!((predictions[0] - 5.0).abs() < 1.0);

        // Add more data incrementally
        let x_new = Array2::from_shape_vec((2, 1), vec![5.0, 6.0]).unwrap();
        let y_new = array![10.0, 12.0];
        fitted.partial_fit(&x_new, &y_new).unwrap();

        assert_eq!(fitted.n_samples_in_buffer(), 6);
        assert_eq!(fitted.n_samples_seen(), 6);
    }

    #[test]
    fn test_memory_overflow() {
        let x_train = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y_train = array![0, 1, 2];

        // Set max_samples to 2 to force overflow
        let classifier = IncrementalKNeighborsClassifier::new(1, 2);
        let mut fitted = classifier.fit(&x_train, &y_train).unwrap();

        // Buffer should only contain 2 samples (most recent)
        assert_eq!(fitted.n_samples_in_buffer(), 2);
        assert_eq!(fitted.n_samples_seen(), 3);

        // Add more data to test overflow handling
        let x_new = Array2::from_shape_vec((2, 2), vec![4.0, 4.0, 5.0, 5.0]).unwrap();
        let y_new = array![3, 4];
        fitted.partial_fit(&x_new, &y_new).unwrap();

        // Buffer should still be 2, but total seen should be 5
        assert_eq!(fitted.n_samples_in_buffer(), 2);
        assert_eq!(fitted.n_samples_seen(), 5);
    }

    #[test]
    fn test_memory_strategies() {
        let x_train = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let y_train = array![0, 1];

        // Test different memory strategies
        let strategies = [
            MemoryStrategy::KeepRecent,
            MemoryStrategy::Random,
            MemoryStrategy::Diversity,
            MemoryStrategy::Representative,
        ];

        for strategy in &strategies {
            let classifier =
                IncrementalKNeighborsClassifier::new(1, 3).with_memory_strategy(*strategy);
            let fitted = classifier.fit(&x_train, &y_train).unwrap();

            // Should work without error
            assert_eq!(fitted.n_samples_in_buffer(), 2);
        }
    }

    #[test]
    fn test_incremental_errors() {
        let x_train = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let y_train = array![0, 1];

        // Test invalid k
        let classifier = IncrementalKNeighborsClassifier::new(0, 10);
        let result = classifier.fit(&x_train, &y_train);
        assert!(result.is_err());

        // Test k > max_samples
        let classifier = IncrementalKNeighborsClassifier::new(15, 10);
        let result = classifier.fit(&x_train, &y_train);
        assert!(result.is_err());
    }
}
