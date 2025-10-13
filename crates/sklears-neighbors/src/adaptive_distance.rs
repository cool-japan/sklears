//! Adaptive distance metrics that adjust based on data characteristics
//!
//! This module implements distance metrics that adapt to local data properties,
//! providing better similarity measurements in heterogeneous data distributions.

use crate::distance::Distance;
use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::types::Float;

/// Adaptive distance metric that adjusts based on local density
///
/// This metric modifies distances based on the local density of the data,
/// giving more weight to differences in dense regions and less in sparse regions.
#[derive(Debug, Clone)]
pub struct AdaptiveDensityDistance {
    /// Base distance metric
    base_distance: Distance,
    /// Number of neighbors for density estimation
    k_density: usize,
    /// Adaptation strength (0.0 = no adaptation, 1.0 = full adaptation)
    adaptation_strength: Float,
    /// Cached density values for training data
    density_cache: Option<Array1<Float>>,
    /// Training data for reference
    training_data: Option<Array2<Float>>,
}

impl AdaptiveDensityDistance {
    /// Create a new adaptive density distance metric
    pub fn new(base_distance: Distance, k_density: usize) -> Self {
        Self {
            base_distance,
            k_density,
            adaptation_strength: 0.5,
            density_cache: None,
            training_data: None,
        }
    }

    /// Set adaptation strength
    pub fn with_adaptation_strength(mut self, strength: Float) -> Self {
        self.adaptation_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Fit the adaptive distance metric to training data
    pub fn fit(&mut self, data: &ArrayView2<Float>) -> NeighborsResult<()> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        // Store training data
        self.training_data = Some(data.to_owned());

        // Compute local density for each point
        let n_samples = data.nrows();
        let mut densities = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut distances = Vec::new();

            // Compute distances to all other points
            for j in 0..n_samples {
                if i != j {
                    let dist = self.base_distance.calculate(&data.row(i), &data.row(j));
                    distances.push(dist);
                }
            }

            // Sort distances and take k-th nearest neighbor distance
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k_neighbor_dist = distances
                .get(std::cmp::min(self.k_density, distances.len()) - 1)
                .copied()
                .unwrap_or(1.0);

            // Density is inverse of k-neighbor distance
            densities[i] = if k_neighbor_dist > 0.0 {
                1.0 / k_neighbor_dist
            } else {
                1e10 // Very high density for identical points
            };
        }

        self.density_cache = Some(densities);
        Ok(())
    }

    /// Calculate adaptive distance between two points
    pub fn calculate(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        let base_dist = self.base_distance.calculate(a, b);

        if let (Some(training_data), Some(density_cache)) =
            (&self.training_data, &self.density_cache)
        {
            // Find closest points in training data to estimate local density
            let density_a = self.estimate_density(a, training_data, density_cache);
            let density_b = self.estimate_density(b, training_data, density_cache);

            // Average density in the region
            let local_density = (density_a + density_b) / 2.0;

            // Adaptive scaling factor based on density
            let scale_factor = 1.0 + self.adaptation_strength * (1.0 / (1.0 + local_density));

            base_dist * scale_factor
        } else {
            base_dist
        }
    }

    /// Estimate density for a new point based on training data
    fn estimate_density(
        &self,
        point: &ArrayView1<Float>,
        training_data: &Array2<Float>,
        density_cache: &Array1<Float>,
    ) -> Float {
        let mut min_dist = Float::INFINITY;
        let mut closest_idx = 0;

        // Find closest training point
        for (i, row) in training_data.axis_iter(Axis(0)).enumerate() {
            let dist = self.base_distance.calculate(point, &row);
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        density_cache[closest_idx]
    }
}

/// Context-dependent distance that adapts based on feature relevance
///
/// This metric learns different feature weights for different regions of the feature space.
#[derive(Debug, Clone)]
pub struct ContextDependentDistance {
    /// Base distance metric
    base_distance: Distance,
    /// Number of contexts (clusters)
    n_contexts: usize,
    /// Context centroids
    context_centroids: Option<Array2<Float>>,
    /// Feature weights for each context
    context_weights: Option<Array2<Float>>,
    /// Context assignments for training data
    context_assignments: Option<Array1<usize>>,
}

impl ContextDependentDistance {
    /// Create a new context-dependent distance metric
    pub fn new(base_distance: Distance, n_contexts: usize) -> Self {
        Self {
            base_distance,
            n_contexts,
            context_centroids: None,
            context_weights: None,
            context_assignments: None,
        }
    }

    /// Fit the context-dependent distance to training data
    pub fn fit(&mut self, data: &ArrayView2<Float>) -> NeighborsResult<()> {
        if data.is_empty() {
            return Err(NeighborsError::EmptyInput);
        }

        let n_features = data.ncols();
        let n_samples = data.nrows();

        // Simple k-means clustering to find contexts
        let mut centroids = Array2::zeros((self.n_contexts, n_features));
        let mut assignments = Array1::zeros(n_samples);

        // Initialize centroids randomly
        for i in 0..self.n_contexts {
            if i < n_samples {
                centroids.row_mut(i).assign(&data.row(i));
            }
        }

        // K-means iterations
        for _iter in 0..10 {
            // Assign points to closest centroid
            for (i, point) in data.axis_iter(Axis(0)).enumerate() {
                let mut min_dist = Float::INFINITY;
                let mut best_context = 0;

                for j in 0..self.n_contexts {
                    let dist = self.base_distance.calculate(&point, &centroids.row(j));
                    if dist < min_dist {
                        min_dist = dist;
                        best_context = j;
                    }
                }
                assignments[i] = best_context;
            }

            // Update centroids
            for j in 0..self.n_contexts {
                let context_points: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &assignment)| assignment == j)
                    .map(|(idx, _)| idx)
                    .collect();

                if !context_points.is_empty() {
                    let mut centroid = Array1::zeros(n_features);
                    for &idx in &context_points {
                        centroid = centroid + data.row(idx);
                    }
                    centroid /= context_points.len() as Float;
                    centroids.row_mut(j).assign(&centroid);
                }
            }
        }

        // Learn feature weights for each context
        let mut weights = Array2::ones((self.n_contexts, n_features));

        for context in 0..self.n_contexts {
            let context_points: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == context)
                .map(|(idx, _)| idx)
                .collect();

            if context_points.len() > 1 {
                // Compute feature variances in this context
                let context_centroid = centroids.row(context);
                let mut feature_vars = Array1::<Float>::zeros(n_features);

                for &idx in &context_points {
                    let point = data.row(idx);
                    let diff = &point - &context_centroid;
                    feature_vars += &(diff.mapv(|x| x * x));
                }
                feature_vars /= context_points.len() as Float;

                // Weight features inversely to their variance (high variance = less discriminative)
                for f in 0..n_features {
                    weights[[context, f]] = if feature_vars[f] > 1e-10 {
                        1.0 / (1.0 + feature_vars[f])
                    } else {
                        1.0
                    };
                }
            }
        }

        self.context_centroids = Some(centroids);
        self.context_weights = Some(weights);
        self.context_assignments = Some(assignments);
        Ok(())
    }

    /// Calculate context-dependent distance
    pub fn calculate(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        if let (Some(centroids), Some(weights)) = (&self.context_centroids, &self.context_weights) {
            // Find most relevant context for these points
            let context_a = self.find_context(a, centroids);
            let context_b = self.find_context(b, centroids);

            // Use average weights from both contexts
            let weight_a = weights.row(context_a);
            let weight_b = weights.row(context_b);
            let avg_weights = (&weight_a + &weight_b) / 2.0;

            // Apply weighted distance
            self.weighted_distance(a, b, &avg_weights.view())
        } else {
            self.base_distance.calculate(a, b)
        }
    }

    /// Find the most relevant context for a point
    fn find_context(&self, point: &ArrayView1<Float>, centroids: &Array2<Float>) -> usize {
        let mut min_dist = Float::INFINITY;
        let mut best_context = 0;

        for (i, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
            let dist = self.base_distance.calculate(point, &centroid);
            if dist < min_dist {
                min_dist = dist;
                best_context = i;
            }
        }

        best_context
    }

    /// Calculate weighted distance
    fn weighted_distance(
        &self,
        a: &ArrayView1<Float>,
        b: &ArrayView1<Float>,
        weights: &ArrayView1<Float>,
    ) -> Float {
        match self.base_distance {
            Distance::Euclidean => {
                let diff = a - b;
                let weighted_diff = &diff * weights;
                weighted_diff.dot(&diff).sqrt()
            }
            Distance::Manhattan => {
                let diff = a - b;
                let weighted_diff = &diff.mapv(|x| x.abs()) * weights;
                weighted_diff.sum()
            }
            _ => {
                // For other distances, use simple feature selection based on weights
                let threshold = 0.5;
                let selected_features: Vec<usize> = weights
                    .iter()
                    .enumerate()
                    .filter(|(_, &w)| w > threshold)
                    .map(|(i, _)| i)
                    .collect();

                if selected_features.is_empty() {
                    self.base_distance.calculate(a, b)
                } else {
                    let a_selected = Array1::from_iter(selected_features.iter().map(|&i| a[i]));
                    let b_selected = Array1::from_iter(selected_features.iter().map(|&i| b[i]));
                    self.base_distance
                        .calculate(&a_selected.view(), &b_selected.view())
                }
            }
        }
    }
}

/// Ensemble distance metric that combines multiple distance measures
#[derive(Debug, Clone)]
pub struct EnsembleDistance {
    /// Component distance metrics
    distances: Vec<Distance>,
    /// Weights for each distance metric
    weights: Vec<Float>,
    /// Combination method
    combination_method: CombinationMethod,
}

/// Methods for combining multiple distance measures
#[derive(Debug, Clone)]
pub enum CombinationMethod {
    /// Weighted average
    WeightedAverage,
    /// Minimum distance
    Minimum,
    /// Maximum distance
    Maximum,
    /// Harmonic mean
    HarmonicMean,
    /// Geometric mean
    GeometricMean,
}

impl EnsembleDistance {
    /// Create a new ensemble distance metric
    pub fn new(distances: Vec<Distance>, weights: Vec<Float>) -> NeighborsResult<Self> {
        if distances.is_empty() {
            return Err(NeighborsError::InvalidInput(
                "No distance metrics provided".to_string(),
            ));
        }

        if distances.len() != weights.len() {
            return Err(NeighborsError::InvalidInput(
                "Distances and weights must have same length".to_string(),
            ));
        }

        let weight_sum: Float = weights.iter().sum();
        if weight_sum <= 0.0 {
            return Err(NeighborsError::InvalidInput(
                "Weights must sum to positive value".to_string(),
            ));
        }

        // Normalize weights
        let normalized_weights: Vec<Float> = weights.iter().map(|w| w / weight_sum).collect();

        Ok(Self {
            distances,
            weights: normalized_weights,
            combination_method: CombinationMethod::WeightedAverage,
        })
    }

    /// Set the combination method
    pub fn with_combination_method(mut self, method: CombinationMethod) -> Self {
        self.combination_method = method;
        self
    }

    /// Calculate ensemble distance
    pub fn calculate(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        let individual_distances: Vec<Float> = self
            .distances
            .iter()
            .map(|dist| dist.calculate(a, b))
            .collect();

        match self.combination_method {
            CombinationMethod::WeightedAverage => individual_distances
                .iter()
                .zip(self.weights.iter())
                .map(|(d, w)| d * w)
                .sum(),
            CombinationMethod::Minimum => individual_distances
                .iter()
                .fold(Float::INFINITY, |a, &b| a.min(b)),
            CombinationMethod::Maximum => individual_distances
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b)),
            CombinationMethod::HarmonicMean => {
                let n = individual_distances.len() as Float;
                let harmonic_sum: Float =
                    individual_distances.iter().map(|d| 1.0 / (d + 1e-10)).sum();
                n / harmonic_sum
            }
            CombinationMethod::GeometricMean => {
                let product: Float = individual_distances.iter().product();
                product.powf(1.0 / individual_distances.len() as Float)
            }
        }
    }
}

/// Online adaptive distance metric that updates based on streaming data
#[derive(Debug, Clone)]
pub struct OnlineAdaptiveDistance {
    /// Base distance metric
    base_distance: Distance,
    /// Feature importance weights (updated online)
    feature_weights: Array1<Float>,
    /// Number of samples seen
    n_samples: usize,
    /// Learning rate for online updates
    learning_rate: Float,
    /// Forgetting factor for old samples
    forgetting_factor: Float,
}

impl OnlineAdaptiveDistance {
    /// Create a new online adaptive distance metric
    pub fn new(base_distance: Distance, n_features: usize) -> Self {
        Self {
            base_distance,
            feature_weights: Array1::ones(n_features),
            n_samples: 0,
            learning_rate: 0.01,
            forgetting_factor: 0.99,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set forgetting factor
    pub fn with_forgetting_factor(mut self, forgetting_factor: Float) -> Self {
        self.forgetting_factor = forgetting_factor;
        self
    }

    /// Update weights based on new sample and its neighbors
    pub fn update(
        &mut self,
        sample: &ArrayView1<Float>,
        neighbors: &[ArrayView1<Float>],
        target_similarity: Float,
    ) {
        if neighbors.is_empty() {
            return;
        }

        // Apply forgetting to existing weights
        self.feature_weights *= self.forgetting_factor;

        // Compute gradient based on neighbor similarities
        for neighbor in neighbors {
            let current_distance = self.calculate(sample, neighbor);
            let error = current_distance - target_similarity;

            // Gradient of weighted distance w.r.t. weights
            let diff = sample - neighbor;
            let gradient = &diff.mapv(|x| x * x * error.signum());

            // Update weights
            self.feature_weights = &self.feature_weights - &(gradient * self.learning_rate);
        }

        // Ensure weights stay positive and normalized
        self.feature_weights = self.feature_weights.mapv(|w| w.max(0.1));
        let weight_sum = self.feature_weights.sum();
        if weight_sum > 0.0 {
            self.feature_weights =
                &self.feature_weights / weight_sum * self.feature_weights.len() as Float;
        }

        self.n_samples += 1;
    }

    /// Calculate adaptive distance
    pub fn calculate(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
        match self.base_distance {
            Distance::Euclidean => {
                let diff = a - b;
                let weighted_diff = &diff * &self.feature_weights;
                weighted_diff.dot(&diff).sqrt()
            }
            Distance::Manhattan => {
                let diff = a - b;
                let weighted_diff = &diff.mapv(|x| x.abs()) * &self.feature_weights;
                weighted_diff.sum()
            }
            _ => {
                // For other distances, apply weights as feature selection
                let avg_weight = self.feature_weights.mean().unwrap_or(1.0);
                let selected_indices: Vec<usize> = self
                    .feature_weights
                    .iter()
                    .enumerate()
                    .filter(|(_, &w)| w > avg_weight)
                    .map(|(i, _)| i)
                    .collect();

                if selected_indices.is_empty() {
                    self.base_distance.calculate(a, b)
                } else {
                    let a_selected = Array1::from_iter(selected_indices.iter().map(|&i| a[i]));
                    let b_selected = Array1::from_iter(selected_indices.iter().map(|&i| b[i]));
                    self.base_distance
                        .calculate(&a_selected.view(), &b_selected.view())
                }
            }
        }
    }

    /// Get current feature weights
    pub fn get_feature_weights(&self) -> &Array1<Float> {
        &self.feature_weights
    }

    /// Get number of samples seen
    pub fn get_n_samples(&self) -> usize {
        self.n_samples
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_density_distance() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1])
            .unwrap();

        let mut adaptive_dist = AdaptiveDensityDistance::new(Distance::Euclidean, 2);
        adaptive_dist.fit(&data.view()).unwrap();

        let a = data.row(0);
        let b = data.row(1);
        let distance = adaptive_dist.calculate(&a, &b);

        assert!(distance > 0.0);
    }

    #[test]
    fn test_context_dependent_distance() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 10.0, 10.0, 10.1, 10.1, 10.0, 10.1,
            ],
        )
        .unwrap();

        let mut context_dist = ContextDependentDistance::new(Distance::Euclidean, 2);
        context_dist.fit(&data.view()).unwrap();

        let a = data.row(0);
        let b = data.row(1);
        let distance = context_dist.calculate(&a, &b);

        assert!(distance > 0.0);
    }

    #[test]
    fn test_ensemble_distance() {
        let distances = vec![Distance::Euclidean, Distance::Manhattan];
        let weights = vec![0.6, 0.4];

        let ensemble = EnsembleDistance::new(distances, weights).unwrap();

        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0];
        let distance = ensemble.calculate(&a.view(), &b.view());

        assert!(distance > 0.0);
    }

    #[test]
    fn test_online_adaptive_distance() {
        let mut online_dist = OnlineAdaptiveDistance::new(Distance::Euclidean, 2);

        let sample = array![1.0, 2.0];
        let neighbor = array![1.1, 2.1];
        let neighbors = vec![neighbor.view()];

        online_dist.update(&sample.view(), &neighbors, 0.5);

        let distance = online_dist.calculate(&sample.view(), &neighbor.view());
        assert!(distance > 0.0);
        assert_eq!(online_dist.get_n_samples(), 1);
    }

    #[test]
    fn test_ensemble_different_combination_methods() {
        let distances = vec![Distance::Euclidean, Distance::Manhattan];
        let weights = vec![0.5, 0.5];

        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0];

        // Test different combination methods
        let methods = vec![
            CombinationMethod::WeightedAverage,
            CombinationMethod::Minimum,
            CombinationMethod::Maximum,
            CombinationMethod::HarmonicMean,
            CombinationMethod::GeometricMean,
        ];

        for method in methods {
            let ensemble = EnsembleDistance::new(distances.clone(), weights.clone())
                .unwrap()
                .with_combination_method(method);

            let distance = ensemble.calculate(&a.view(), &b.view());
            assert!(distance > 0.0);
        }
    }
}
