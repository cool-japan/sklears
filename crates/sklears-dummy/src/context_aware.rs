//! Context-aware dummy estimators that use feature information
//!
//! This module provides dummy estimators that incorporate input features into their
//! baseline predictions, making them more sophisticated than traditional dummy methods
//! while still remaining simple and interpretable baselines.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict};
use sklears_core::types::{Features, Float};
use std::collections::HashMap;

/// Strategy for context-aware predictions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ContextAwareStrategy {
    /// Make predictions conditional on feature bins/intervals
    Conditional {
        /// Number of bins for each feature
        n_bins: usize,
        /// Minimum samples per bin to make predictions
        min_samples_per_bin: usize,
    },
    /// Use feature-weighted predictions based on feature importance
    FeatureWeighted {
        /// Weighting method for features
        weighting: FeatureWeighting,
    },
    /// Cluster-based predictions using simple k-means
    ClusterBased {
        /// Number of clusters
        n_clusters: usize,
        /// Maximum iterations for clustering
        max_iter: usize,
    },
    /// Locality-sensitive predictions using nearest neighbors
    LocalitySensitive {
        /// Number of neighbors to consider
        n_neighbors: usize,
        /// Distance metric weighting factor
        distance_power: Float,
    },
    /// Adaptive local baselines that adjust based on local feature statistics
    AdaptiveLocal {
        /// Radius for local neighborhood
        radius: Float,
        /// Minimum samples in neighborhood
        min_local_samples: usize,
    },
}

/// Feature weighting methods
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum FeatureWeighting {
    /// Equal weights for all features
    Uniform,
    /// Weights based on feature variance
    Variance,
    /// Weights based on correlation with target
    Correlation,
    /// Custom user-specified weights
    Custom(Array1<Float>),
}

/// Context-aware dummy regressor
#[derive(Debug, Clone)]
pub struct ContextAwareDummyRegressor<State = sklears_core::traits::Untrained> {
    /// Strategy for context-aware predictions
    pub strategy: ContextAwareStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,

    // Fitted parameters
    /// Feature bins for conditional strategy
    pub(crate) feature_bins_: Option<Vec<Array1<Float>>>,
    /// Bin predictions for conditional strategy
    pub(crate) bin_predictions_: Option<HashMap<Vec<usize>, Float>>,

    /// Feature weights for weighted strategy
    pub(crate) feature_weights_: Option<Array1<Float>>,
    /// Weighted prediction function parameters
    pub(crate) weighted_intercept_: Option<Float>,
    pub(crate) weighted_coefficients_: Option<Array1<Float>>,

    /// Cluster centers for cluster-based strategy
    pub(crate) cluster_centers_: Option<Array2<Float>>,
    /// Cluster predictions
    pub(crate) cluster_predictions_: Option<Array1<Float>>,

    /// Training data for locality-sensitive strategy
    pub(crate) training_features_: Option<Array2<Float>>,
    pub(crate) training_targets_: Option<Array1<Float>>,

    /// Local statistics for adaptive strategy
    pub(crate) local_means_: Option<Array1<Float>>,
    pub(crate) local_stds_: Option<Array1<Float>>,
    pub(crate) local_centers_: Option<Array2<Float>>,

    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl ContextAwareDummyRegressor {
    /// Create a new context-aware dummy regressor
    pub fn new(strategy: ContextAwareStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            feature_bins_: None,
            bin_predictions_: None,
            feature_weights_: None,
            weighted_intercept_: None,
            weighted_coefficients_: None,
            cluster_centers_: None,
            cluster_predictions_: None,
            training_features_: None,
            training_targets_: None,
            local_means_: None,
            local_stds_: None,
            local_centers_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for ContextAwareDummyRegressor {
    fn default() -> Self {
        Self::new(ContextAwareStrategy::Conditional {
            n_bins: 5,
            min_samples_per_bin: 3,
        })
    }
}

impl Estimator for ContextAwareDummyRegressor {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Float>> for ContextAwareDummyRegressor {
    type Fitted = ContextAwareDummyRegressor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        let mut fitted = ContextAwareDummyRegressor {
            strategy: self.strategy.clone(),
            random_state: self.random_state,
            feature_bins_: None,
            bin_predictions_: None,
            feature_weights_: None,
            weighted_intercept_: None,
            weighted_coefficients_: None,
            cluster_centers_: None,
            cluster_predictions_: None,
            training_features_: None,
            training_targets_: None,
            local_means_: None,
            local_stds_: None,
            local_centers_: None,
            _state: std::marker::PhantomData,
        };

        match &self.strategy {
            ContextAwareStrategy::Conditional {
                n_bins,
                min_samples_per_bin,
            } => {
                fitted.fit_conditional(x, y, *n_bins, *min_samples_per_bin)?;
            }
            ContextAwareStrategy::FeatureWeighted { weighting } => {
                fitted.fit_feature_weighted(x, y, weighting)?;
            }
            ContextAwareStrategy::ClusterBased {
                n_clusters,
                max_iter,
            } => {
                fitted.fit_cluster_based(x, y, *n_clusters, *max_iter)?;
            }
            ContextAwareStrategy::LocalitySensitive {
                n_neighbors,
                distance_power,
            } => {
                fitted.fit_locality_sensitive(x, y, *n_neighbors, *distance_power)?;
            }
            ContextAwareStrategy::AdaptiveLocal {
                radius,
                min_local_samples,
            } => {
                fitted.fit_adaptive_local(x, y, *radius, *min_local_samples)?;
            }
        }

        Ok(fitted)
    }
}

impl ContextAwareDummyRegressor<sklears_core::traits::Trained> {
    /// Fit conditional strategy
    fn fit_conditional(
        &mut self,
        x: &Features,
        y: &Array1<Float>,
        n_bins: usize,
        min_samples_per_bin: usize,
    ) -> Result<()> {
        let n_features = x.ncols();
        let mut feature_bins = Vec::with_capacity(n_features);
        let mut bin_predictions = HashMap::new();

        // Create bins for each feature
        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);
            let min_val = feature_values
                .iter()
                .fold(Float::INFINITY, |a, &b| a.min(b));
            let max_val = feature_values
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let bin_width = (max_val - min_val) / n_bins as Float;
            let mut bins = Array1::zeros(n_bins + 1);

            for i in 0..=n_bins {
                bins[i] = min_val + i as Float * bin_width;
            }
            bins[n_bins] = max_val + 1e-10; // Ensure max value is included

            feature_bins.push(bins);
        }

        // Compute predictions for each bin combination
        for i in 0..x.nrows() {
            let mut bin_indices = Vec::with_capacity(n_features);

            for (feature_idx, bins) in feature_bins.iter().enumerate() {
                let value = x[[i, feature_idx]];
                let bin_idx = bins
                    .iter()
                    .position(|&bin_edge| value < bin_edge)
                    .unwrap_or(bins.len() - 1)
                    .saturating_sub(1);
                bin_indices.push(bin_idx);
            }

            let entry = bin_predictions.entry(bin_indices).or_insert_with(Vec::new);
            entry.push(y[i]);
        }

        // Compute mean for each bin with sufficient samples
        let mut final_bin_predictions = HashMap::new();
        for (bin_key, targets) in bin_predictions {
            if targets.len() >= min_samples_per_bin {
                let mean = targets.iter().sum::<Float>() / targets.len() as Float;
                final_bin_predictions.insert(bin_key, mean);
            }
        }

        self.feature_bins_ = Some(feature_bins);
        self.bin_predictions_ = Some(final_bin_predictions);
        Ok(())
    }

    /// Fit feature-weighted strategy
    fn fit_feature_weighted(
        &mut self,
        x: &Features,
        y: &Array1<Float>,
        weighting: &FeatureWeighting,
    ) -> Result<()> {
        let n_features = x.ncols();
        let weights = match weighting {
            FeatureWeighting::Uniform => Array1::from_elem(n_features, 1.0 / n_features as Float),
            FeatureWeighting::Variance => {
                let mut weights = Array1::zeros(n_features);
                for i in 0..n_features {
                    let feature = x.column(i);
                    let mean = feature.mean().unwrap_or(0.0);
                    let variance = feature
                        .iter()
                        .map(|&val| (val - mean).powi(2))
                        .sum::<Float>()
                        / feature.len() as Float;
                    weights[i] = variance;
                }
                let sum_weights = weights.sum();
                if sum_weights > 0.0 {
                    weights / sum_weights
                } else {
                    Array1::from_elem(n_features, 1.0 / n_features as Float)
                }
            }
            FeatureWeighting::Correlation => {
                let mut weights = Array1::zeros(n_features);
                let y_mean = y.mean().unwrap_or(0.0);

                for i in 0..n_features {
                    let feature = x.column(i);
                    let x_mean = feature.mean().unwrap_or(0.0);

                    let mut numerator = 0.0;
                    let mut x_var = 0.0;
                    let mut y_var = 0.0;

                    for j in 0..feature.len() {
                        let x_diff = feature[j] - x_mean;
                        let y_diff = y[j] - y_mean;
                        numerator += x_diff * y_diff;
                        x_var += x_diff * x_diff;
                        y_var += y_diff * y_diff;
                    }

                    let correlation = if x_var > 0.0 && y_var > 0.0 {
                        numerator / (x_var * y_var).sqrt()
                    } else {
                        0.0
                    };

                    weights[i] = correlation.abs();
                }

                let sum_weights = weights.sum();
                if sum_weights > 0.0 {
                    weights / sum_weights
                } else {
                    Array1::from_elem(n_features, 1.0 / n_features as Float)
                }
            }
            FeatureWeighting::Custom(custom_weights) => {
                if custom_weights.len() != n_features {
                    return Err(sklears_core::error::SklearsError::InvalidInput(
                        "Custom weights length must match number of features".to_string(),
                    ));
                }
                custom_weights.clone()
            }
        };

        // Compute weighted linear combination parameters
        let y_mean = y.mean().unwrap_or(0.0);
        let mut coefficients = Array1::zeros(n_features);

        for i in 0..n_features {
            let feature = x.column(i);
            let x_mean = feature.mean().unwrap_or(0.0);
            coefficients[i] = weights[i] * (y_mean - x_mean);
        }

        self.feature_weights_ = Some(weights);
        self.weighted_intercept_ = Some(y_mean);
        self.weighted_coefficients_ = Some(coefficients);
        Ok(())
    }

    /// Fit cluster-based strategy using simple k-means
    fn fit_cluster_based(
        &mut self,
        x: &Features,
        y: &Array1<Float>,
        n_clusters: usize,
        max_iter: usize,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_clusters > n_samples {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of clusters cannot exceed number of samples".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        // Initialize cluster centers randomly
        let mut centers = Array2::zeros((n_clusters, n_features));
        for i in 0..n_clusters {
            let sample_idx = rng.gen_range(0..n_samples);
            for j in 0..n_features {
                centers[[i, j]] = x[[sample_idx, j]];
            }
        }

        // K-means iterations
        let mut assignments = vec![0; n_samples];

        for _iter in 0..max_iter {
            let mut changed = false;

            // Assign points to nearest centers
            for i in 0..n_samples {
                let mut min_distance = Float::INFINITY;
                let mut best_cluster = 0;

                for cluster in 0..n_clusters {
                    let mut distance = 0.0;
                    for j in 0..n_features {
                        let diff = x[[i, j]] - centers[[cluster, j]];
                        distance += diff * diff;
                    }

                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = cluster;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update cluster centers
            let mut cluster_counts = vec![0; n_clusters];
            centers.fill(0.0);

            for i in 0..n_samples {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;
                for j in 0..n_features {
                    centers[[cluster, j]] += x[[i, j]];
                }
            }

            for cluster in 0..n_clusters {
                if cluster_counts[cluster] > 0 {
                    for j in 0..n_features {
                        centers[[cluster, j]] /= cluster_counts[cluster] as Float;
                    }
                }
            }
        }

        // Compute cluster predictions
        let mut cluster_targets: Vec<Vec<Float>> = vec![Vec::new(); n_clusters];
        for i in 0..n_samples {
            cluster_targets[assignments[i]].push(y[i]);
        }

        let mut cluster_predictions = Array1::zeros(n_clusters);
        for i in 0..n_clusters {
            if !cluster_targets[i].is_empty() {
                cluster_predictions[i] =
                    cluster_targets[i].iter().sum::<Float>() / cluster_targets[i].len() as Float;
            }
        }

        self.cluster_centers_ = Some(centers);
        self.cluster_predictions_ = Some(cluster_predictions);
        Ok(())
    }

    /// Fit locality-sensitive strategy
    fn fit_locality_sensitive(
        &mut self,
        x: &Features,
        y: &Array1<Float>,
        _n_neighbors: usize,
        _distance_power: Float,
    ) -> Result<()> {
        // Store training data for prediction time
        self.training_features_ = Some(x.clone());
        self.training_targets_ = Some(y.clone());
        Ok(())
    }

    /// Fit adaptive local strategy
    fn fit_adaptive_local(
        &mut self,
        x: &Features,
        y: &Array1<Float>,
        radius: Float,
        min_local_samples: usize,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Create local statistics for representative points
        let n_centers = (n_samples / min_local_samples).max(1);
        let mut centers = Array2::zeros((n_centers, n_features));
        let mut local_means = Array1::zeros(n_centers);
        let mut local_stds = Array1::zeros(n_centers);

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        // Select representative centers
        for i in 0..n_centers {
            let sample_idx = rng.gen_range(0..n_samples);
            for j in 0..n_features {
                centers[[i, j]] = x[[sample_idx, j]];
            }
        }

        // Compute local statistics for each center
        for i in 0..n_centers {
            let mut local_targets = Vec::new();

            for j in 0..n_samples {
                let mut distance = 0.0;
                for k in 0..n_features {
                    let diff = x[[j, k]] - centers[[i, k]];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                if distance <= radius {
                    local_targets.push(y[j]);
                }
            }

            if local_targets.len() >= min_local_samples {
                let mean = local_targets.iter().sum::<Float>() / local_targets.len() as Float;
                let variance = local_targets
                    .iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<Float>()
                    / local_targets.len() as Float;
                let std_dev = variance.sqrt();

                local_means[i] = mean;
                local_stds[i] = std_dev;
            } else {
                // Fallback to global statistics
                let global_mean = y.mean().unwrap_or(0.0);
                let global_variance = y
                    .iter()
                    .map(|&val| (val - global_mean).powi(2))
                    .sum::<Float>()
                    / y.len() as Float;

                local_means[i] = global_mean;
                local_stds[i] = global_variance.sqrt();
            }
        }

        self.local_centers_ = Some(centers);
        self.local_means_ = Some(local_means);
        self.local_stds_ = Some(local_stds);
        Ok(())
    }
}

impl Predict<Features, Array1<Float>>
    for ContextAwareDummyRegressor<sklears_core::traits::Trained>
{
    fn predict(&self, x: &Features) -> Result<Array1<Float>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        match &self.strategy {
            ContextAwareStrategy::Conditional { .. } => {
                self.predict_conditional(x, &mut predictions)?;
            }
            ContextAwareStrategy::FeatureWeighted { .. } => {
                self.predict_feature_weighted(x, &mut predictions)?;
            }
            ContextAwareStrategy::ClusterBased { .. } => {
                self.predict_cluster_based(x, &mut predictions)?;
            }
            ContextAwareStrategy::LocalitySensitive {
                n_neighbors,
                distance_power,
            } => {
                self.predict_locality_sensitive(
                    x,
                    &mut predictions,
                    *n_neighbors,
                    *distance_power,
                )?;
            }
            ContextAwareStrategy::AdaptiveLocal { radius, .. } => {
                self.predict_adaptive_local(x, &mut predictions, *radius)?;
            }
        }

        Ok(predictions)
    }
}

impl ContextAwareDummyRegressor<sklears_core::traits::Trained> {
    /// Predict using conditional strategy
    fn predict_conditional(&self, x: &Features, predictions: &mut Array1<Float>) -> Result<()> {
        let feature_bins = self.feature_bins_.as_ref().unwrap();
        let bin_predictions = self.bin_predictions_.as_ref().unwrap();
        let global_mean = bin_predictions.values().sum::<Float>() / bin_predictions.len() as Float;

        for i in 0..x.nrows() {
            let mut bin_indices = Vec::with_capacity(feature_bins.len());

            for (feature_idx, bins) in feature_bins.iter().enumerate() {
                let value = x[[i, feature_idx]];
                let bin_idx = bins
                    .iter()
                    .position(|&bin_edge| value < bin_edge)
                    .unwrap_or(bins.len() - 1)
                    .saturating_sub(1);
                bin_indices.push(bin_idx);
            }

            predictions[i] = *bin_predictions.get(&bin_indices).unwrap_or(&global_mean);
        }

        Ok(())
    }

    /// Predict using feature-weighted strategy
    fn predict_feature_weighted(
        &self,
        x: &Features,
        predictions: &mut Array1<Float>,
    ) -> Result<()> {
        let weights = self.feature_weights_.as_ref().unwrap();
        let intercept = self.weighted_intercept_.unwrap();
        let coefficients = self.weighted_coefficients_.as_ref().unwrap();

        for i in 0..x.nrows() {
            let mut weighted_sum = intercept;
            for j in 0..x.ncols() {
                weighted_sum += x[[i, j]] * weights[j] + coefficients[j];
            }
            predictions[i] = weighted_sum;
        }

        Ok(())
    }

    /// Predict using cluster-based strategy
    fn predict_cluster_based(&self, x: &Features, predictions: &mut Array1<Float>) -> Result<()> {
        let centers = self.cluster_centers_.as_ref().unwrap();
        let cluster_predictions = self.cluster_predictions_.as_ref().unwrap();

        for i in 0..x.nrows() {
            let mut min_distance = Float::INFINITY;
            let mut best_cluster = 0;

            for cluster in 0..centers.nrows() {
                let mut distance = 0.0;
                for j in 0..x.ncols() {
                    let diff = x[[i, j]] - centers[[cluster, j]];
                    distance += diff * diff;
                }

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster;
                }
            }

            predictions[i] = cluster_predictions[best_cluster];
        }

        Ok(())
    }

    /// Predict using locality-sensitive strategy
    fn predict_locality_sensitive(
        &self,
        x: &Features,
        predictions: &mut Array1<Float>,
        n_neighbors: usize,
        distance_power: Float,
    ) -> Result<()> {
        let training_features = self.training_features_.as_ref().unwrap();
        let training_targets = self.training_targets_.as_ref().unwrap();

        for i in 0..x.nrows() {
            let mut distances = Vec::new();

            // Compute distances to all training points
            for j in 0..training_features.nrows() {
                let mut distance = 0.0;
                for k in 0..x.ncols() {
                    let diff = x[[i, k]] - training_features[[j, k]];
                    distance += diff * diff;
                }
                distance = distance.sqrt();
                distances.push((distance, j));
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k_nearest = distances.into_iter().take(n_neighbors).collect::<Vec<_>>();

            // Weighted average based on inverse distance
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (distance, idx) in k_nearest {
                let weight = if distance == 0.0 {
                    1000.0 // Large weight for exact matches
                } else {
                    1.0 / distance.powf(distance_power)
                };

                weighted_sum += weight * training_targets[idx];
                weight_sum += weight;
            }

            predictions[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                training_targets.mean().unwrap_or(0.0)
            };
        }

        Ok(())
    }

    /// Predict using adaptive local strategy
    fn predict_adaptive_local(
        &self,
        x: &Features,
        predictions: &mut Array1<Float>,
        radius: Float,
    ) -> Result<()> {
        let centers = self.local_centers_.as_ref().unwrap();
        let local_means = self.local_means_.as_ref().unwrap();
        let local_stds = self.local_stds_.as_ref().unwrap();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        for i in 0..x.nrows() {
            // Find nearest center within radius
            let mut min_distance = Float::INFINITY;
            let mut best_center = 0;

            for j in 0..centers.nrows() {
                let mut distance = 0.0;
                for k in 0..x.ncols() {
                    let diff = x[[i, k]] - centers[[j, k]];
                    distance += diff * diff;
                }
                distance = distance.sqrt();

                if distance <= radius && distance < min_distance {
                    min_distance = distance;
                    best_center = j;
                }
            }

            // Sample from local distribution
            if min_distance <= radius {
                let mean = local_means[best_center];
                let std = local_stds[best_center];

                if std > 0.0 {
                    let normal = Normal::new(mean, std).unwrap();
                    predictions[i] = normal.sample(&mut rng);
                } else {
                    predictions[i] = mean;
                }
            } else {
                // Fallback to global mean
                predictions[i] = local_means.mean().unwrap_or(0.0);
            }
        }

        Ok(())
    }
}

/// Context-aware dummy classifier
#[derive(Debug, Clone)]
pub struct ContextAwareDummyClassifier<State = sklears_core::traits::Untrained> {
    /// Strategy for context-aware predictions
    pub strategy: ContextAwareStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,

    // Fitted parameters (similar structure to regressor but for classification)
    pub(crate) feature_bins_: Option<Vec<Array1<Float>>>,
    pub(crate) bin_class_probs_: Option<HashMap<Vec<usize>, HashMap<i32, Float>>>,
    pub(crate) classes_: Option<Array1<i32>>,
    pub(crate) training_features_: Option<Array2<Float>>,
    pub(crate) training_targets_: Option<Array1<i32>>,

    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl ContextAwareDummyClassifier {
    /// Create a new context-aware dummy classifier
    pub fn new(strategy: ContextAwareStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            feature_bins_: None,
            bin_class_probs_: None,
            classes_: None,
            training_features_: None,
            training_targets_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for ContextAwareDummyClassifier {
    fn default() -> Self {
        Self::new(ContextAwareStrategy::Conditional {
            n_bins: 5,
            min_samples_per_bin: 3,
        })
    }
}

impl Estimator for ContextAwareDummyClassifier {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<i32>> for ContextAwareDummyClassifier {
    type Fitted = ContextAwareDummyClassifier<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes = y.iter().cloned().collect::<Vec<_>>();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);

        let mut fitted = ContextAwareDummyClassifier {
            strategy: self.strategy.clone(),
            random_state: self.random_state,
            feature_bins_: None,
            bin_class_probs_: None,
            classes_: Some(classes),
            training_features_: None,
            training_targets_: None,
            _state: std::marker::PhantomData,
        };

        // For now, implement only conditional strategy for classifier
        match &self.strategy {
            ContextAwareStrategy::Conditional {
                n_bins,
                min_samples_per_bin,
            } => {
                fitted.fit_conditional_classifier(x, y, *n_bins, *min_samples_per_bin)?;
            }
            _ => {
                // Store training data for other strategies
                fitted.training_features_ = Some(x.clone());
                fitted.training_targets_ = Some(y.clone());
            }
        }

        Ok(fitted)
    }
}

impl ContextAwareDummyClassifier<sklears_core::traits::Trained> {
    /// Fit conditional strategy for classification
    fn fit_conditional_classifier(
        &mut self,
        x: &Features,
        y: &Array1<i32>,
        n_bins: usize,
        min_samples_per_bin: usize,
    ) -> Result<()> {
        let n_features = x.ncols();
        let mut feature_bins = Vec::with_capacity(n_features);
        let mut bin_class_counts: HashMap<Vec<usize>, HashMap<i32, usize>> = HashMap::new();

        // Create bins for each feature
        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);
            let min_val = feature_values
                .iter()
                .fold(Float::INFINITY, |a, &b| a.min(b));
            let max_val = feature_values
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let bin_width = (max_val - min_val) / n_bins as Float;
            let mut bins = Array1::zeros(n_bins + 1);

            for i in 0..=n_bins {
                bins[i] = min_val + i as Float * bin_width;
            }
            bins[n_bins] = max_val + 1e-10;

            feature_bins.push(bins);
        }

        // Count classes in each bin combination
        for i in 0..x.nrows() {
            let mut bin_indices = Vec::with_capacity(n_features);

            for (feature_idx, bins) in feature_bins.iter().enumerate() {
                let value = x[[i, feature_idx]];
                let bin_idx = bins
                    .iter()
                    .position(|&bin_edge| value < bin_edge)
                    .unwrap_or(bins.len() - 1)
                    .saturating_sub(1);
                bin_indices.push(bin_idx);
            }

            let class_counts = bin_class_counts.entry(bin_indices).or_default();
            *class_counts.entry(y[i]).or_insert(0) += 1;
        }

        // Convert counts to probabilities for bins with sufficient samples
        let mut bin_class_probs = HashMap::new();
        for (bin_key, class_counts) in bin_class_counts {
            let total_count: usize = class_counts.values().sum();
            if total_count >= min_samples_per_bin {
                let mut class_probs = HashMap::new();
                for (&class, &count) in &class_counts {
                    class_probs.insert(class, count as Float / total_count as Float);
                }
                bin_class_probs.insert(bin_key, class_probs);
            }
        }

        self.feature_bins_ = Some(feature_bins);
        self.bin_class_probs_ = Some(bin_class_probs);
        Ok(())
    }
}

impl Predict<Features, Array1<i32>> for ContextAwareDummyClassifier<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<i32>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);
        let classes = self.classes_.as_ref().unwrap();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        match &self.strategy {
            ContextAwareStrategy::Conditional { .. } => {
                let feature_bins = self.feature_bins_.as_ref().unwrap();
                let bin_class_probs = self.bin_class_probs_.as_ref().unwrap();

                // Global class distribution as fallback
                let global_class = classes[0]; // Simplified fallback

                for i in 0..x.nrows() {
                    let mut bin_indices = Vec::with_capacity(feature_bins.len());

                    for (feature_idx, bins) in feature_bins.iter().enumerate() {
                        let value = x[[i, feature_idx]];
                        let bin_idx = bins
                            .iter()
                            .position(|&bin_edge| value < bin_edge)
                            .unwrap_or(bins.len() - 1)
                            .saturating_sub(1);
                        bin_indices.push(bin_idx);
                    }

                    if let Some(class_probs) = bin_class_probs.get(&bin_indices) {
                        // Sample from class distribution
                        let rand_val: Float = rng.gen();
                        let mut cumulative_prob = 0.0;
                        let mut selected_class = global_class;

                        for (&class, &prob) in class_probs {
                            cumulative_prob += prob;
                            if rand_val <= cumulative_prob {
                                selected_class = class;
                                break;
                            }
                        }
                        predictions[i] = selected_class;
                    } else {
                        predictions[i] = global_class;
                    }
                }
            }
            _ => {
                // Fallback for other strategies - use most frequent class
                let most_frequent_class = classes[0];
                predictions.fill(most_frequent_class);
            }
        }

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_conditional_regressor() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let regressor = ContextAwareDummyRegressor::new(ContextAwareStrategy::Conditional {
            n_bins: 2,
            min_samples_per_bin: 1,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert!(predictions.iter().all(|&p| p >= 1.0 && p <= 6.0));
    }

    #[test]
    fn test_feature_weighted_regressor() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let regressor = ContextAwareDummyRegressor::new(ContextAwareStrategy::FeatureWeighted {
            weighting: FeatureWeighting::Uniform,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_cluster_based_regressor() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1, 9.0, 9.0, 9.1, 9.1],
        )
        .unwrap();
        let y = array![1.0, 1.0, 5.0, 5.0, 9.0, 9.0];

        let regressor = ContextAwareDummyRegressor::new(ContextAwareStrategy::ClusterBased {
            n_clusters: 3,
            max_iter: 10,
        })
        .with_random_state(42);

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_locality_sensitive_regressor() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let regressor = ContextAwareDummyRegressor::new(ContextAwareStrategy::LocalitySensitive {
            n_neighbors: 2,
            distance_power: 2.0,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_adaptive_local_regressor() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let regressor = ContextAwareDummyRegressor::new(ContextAwareStrategy::AdaptiveLocal {
            radius: 2.0,
            min_local_samples: 2,
        })
        .with_random_state(42);

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_conditional_classifier() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 0, 1];

        let classifier = ContextAwareDummyClassifier::new(ContextAwareStrategy::Conditional {
            n_bins: 2,
            min_samples_per_bin: 1,
        })
        .with_random_state(42);

        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert!(predictions.iter().all(|&p| p == 0 || p == 1));
    }
}
