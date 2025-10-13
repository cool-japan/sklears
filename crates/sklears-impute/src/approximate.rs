//! Approximate imputation algorithms for fast processing
//!
//! This module provides fast approximation methods for imputation when speed
//! is more important than perfect accuracy. These methods trade off some
//! accuracy for significant performance gains.

// ✅ SciRS2 Policy compliant imports
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{Random, Rng};
// use scirs2_core::simd::{SimdOps}; // Note: SimdArray and auto_vectorize not available
// use scirs2_core::parallel::{}; // Note: ParallelExecutor, ChunkStrategy not available

use crate::core::Imputer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Configuration for approximate imputation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximateConfig {
    /// Target accuracy vs speed trade-off (0.0 = fastest, 1.0 = most accurate)
    pub accuracy_level: f64,
    /// Maximum processing time per feature (in seconds)
    pub max_time_per_feature: Duration,
    /// Sample size for approximation algorithms
    pub sample_size: usize,
    /// Use randomized algorithms
    pub use_randomization: bool,
    /// Enable early stopping
    pub early_stopping: bool,
    /// Convergence tolerance for iterative methods
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl Default for ApproximateConfig {
    fn default() -> Self {
        Self {
            accuracy_level: 0.8,
            max_time_per_feature: Duration::from_secs(1),
            sample_size: 1000,
            use_randomization: true,
            early_stopping: true,
            tolerance: 1e-3,
            max_iterations: 10,
        }
    }
}

/// Fast approximation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApproximationStrategy {
    /// Random sampling approximation
    RandomSampling,
    /// Sketching-based approximation
    Sketching,
    /// Local approximation using nearest chunks
    LocalApproximation,
    /// Linear approximation
    LinearApproximation,
    /// Hash-based approximation
    HashBased,
}

/// Approximate KNN Imputer with fast neighbor search
#[derive(Debug)]
pub struct ApproximateKNNImputer<S = Untrained> {
    state: S,
    n_neighbors: usize,
    weights: String,
    missing_values: f64,
    config: ApproximateConfig,
    strategy: ApproximationStrategy,
}

/// Trained state for approximate KNN imputer
#[derive(Debug)]
pub struct ApproximateKNNImputerTrained {
    reference_samples: Array2<f64>,
    sample_indices: Vec<usize>,
    n_features_in_: usize,
    config: ApproximateConfig,
    strategy: ApproximationStrategy,
    locality_hash: Option<LocalityHashTable>,
}

/// Locality-sensitive hash table for fast neighbor search
#[derive(Debug)]
pub struct LocalityHashTable {
    hash_functions: Vec<RandomHashFunction>,
    buckets: HashMap<Vec<u32>, Vec<usize>>,
    num_hash_functions: usize,
    bucket_width: f64,
}

/// Random hash function for LSH
#[derive(Debug, Clone)]
pub struct RandomHashFunction {
    random_vector: Array1<f64>,
    offset: f64,
    bucket_width: f64,
}

/// Approximate Simple Imputer with sampling
#[derive(Debug)]
pub struct ApproximateSimpleImputer<S = Untrained> {
    state: S,
    strategy: String,
    missing_values: f64,
    config: ApproximateConfig,
}

/// Trained state for approximate simple imputer
#[derive(Debug)]
pub struct ApproximateSimpleImputerTrained {
    approximate_statistics_: Array1<f64>,
    confidence_intervals_: Array2<f64>, // [feature, (lower, upper)]
    n_features_in_: usize,
    config: ApproximateConfig,
}

/// Sketching-based Imputer
#[derive(Debug)]
pub struct SketchingImputer<S = Untrained> {
    state: S,
    sketch_size: usize,
    missing_values: f64,
    config: ApproximateConfig,
    hash_family: HashFamily,
}

/// Trained state for sketching imputer
#[derive(Debug)]
pub struct SketchingImputerTrained {
    sketches: Vec<CountSketch>,
    n_features_in_: usize,
    config: ApproximateConfig,
}

/// Count sketch data structure
#[derive(Debug, Clone)]
pub struct CountSketch {
    sketch: Array1<f64>,
    hash_functions: Vec<(usize, i32)>, // (hash_function_index, sign)
    size: usize,
}

/// Hash family for sketching
#[derive(Debug, Clone)]
pub enum HashFamily {
    /// Universal hash family
    Universal,
    /// Polynomial hash family
    Polynomial,
    /// MurmurHash family
    Murmur,
}

/// Randomized Iterative Imputer
#[derive(Debug)]
pub struct RandomizedIterativeImputer<S = Untrained> {
    state: S,
    max_iter: usize,
    missing_values: f64,
    config: ApproximateConfig,
    random_order: bool,
    subsample_features: f64,
}

/// Trained state for randomized iterative imputer
pub struct RandomizedIterativeImputerTrained {
    estimators_: Vec<Box<dyn Imputer>>,
    feature_order: Vec<usize>,
    n_features_in_: usize,
    config: ApproximateConfig,
}

impl ApproximateKNNImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            weights: "uniform".to_string(),
            missing_values: f64::NAN,
            config: ApproximateConfig::default(),
            strategy: ApproximationStrategy::RandomSampling,
        }
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    pub fn weights(mut self, weights: String) -> Self {
        self.weights = weights;
        self
    }

    pub fn approximate_config(mut self, config: ApproximateConfig) -> Self {
        self.config = config;
        self
    }

    pub fn strategy(mut self, strategy: ApproximationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn accuracy_level(mut self, level: f64) -> Self {
        self.config.accuracy_level = level.clamp(0.0, 1.0);
        self
    }

    pub fn sample_size(mut self, size: usize) -> Self {
        self.config.sample_size = size;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for ApproximateKNNImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ApproximateKNNImputer<Untrained> {
    type Config = ApproximateConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ApproximateKNNImputer<Untrained> {
    type Fitted = ApproximateKNNImputer<ApproximateKNNImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        // Determine sample size based on accuracy level
        let effective_sample_size = ((self.config.sample_size as f64 * self.config.accuracy_level)
            as usize)
            .min(n_samples)
            .max(self.n_neighbors * 10); // Ensure minimum samples

        // Sample training data for approximation
        let (reference_samples, sample_indices) =
            self.sample_training_data(&X, effective_sample_size)?;

        // Build locality hash table if using hash-based strategy
        let locality_hash = match self.strategy {
            ApproximationStrategy::HashBased => {
                Some(self.build_locality_hash_table(&reference_samples)?)
            }
            _ => None,
        };

        Ok(ApproximateKNNImputer {
            state: ApproximateKNNImputerTrained {
                reference_samples,
                sample_indices,
                n_features_in_: n_features,
                config: self.config,
                strategy: self.strategy,
                locality_hash,
            },
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            missing_values: self.missing_values,
            config: Default::default(),
            strategy: ApproximationStrategy::RandomSampling,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for ApproximateKNNImputer<ApproximateKNNImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        // Process samples in parallel
        X_imputed
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n_features {
                    if self.is_missing(row[j]) {
                        // Find approximate neighbors
                        if let Ok(neighbors) = self.find_approximate_neighbors(&row.to_owned(), j) {
                            if !neighbors.is_empty() {
                                if let Ok(imputed_value) = self.compute_weighted_average(&neighbors)
                                {
                                    row[j] = imputed_value;
                                }
                            }
                        }
                    }
                }
            });

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl ApproximateKNNImputer<Untrained> {
    /// Sample training data for approximation
    fn sample_training_data(
        &self,
        X: &Array2<f64>,
        sample_size: usize,
    ) -> Result<(Array2<f64>, Vec<usize>), SklearsError> {
        let n_samples = X.nrows();

        if sample_size >= n_samples {
            return Ok((X.clone(), (0..n_samples).collect()));
        }

        // Create random sample indices
        let mut rng = Random::default();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Fisher-Yates shuffle for random sampling
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        indices.truncate(sample_size);
        indices.sort(); // Keep sorted for consistent results

        // Extract sampled rows
        let mut sampled_data = Array2::<f64>::zeros((sample_size, X.ncols()));
        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            sampled_data.row_mut(new_idx).assign(&X.row(orig_idx));
        }

        Ok((sampled_data, indices))
    }

    /// Build locality-sensitive hash table
    fn build_locality_hash_table(
        &self,
        data: &Array2<f64>,
    ) -> Result<LocalityHashTable, SklearsError> {
        let n_features = data.ncols();
        let num_hash_functions = (self.config.accuracy_level * 10.0) as usize + 2;
        let bucket_width = 1.0 / (self.config.accuracy_level + 0.1);

        let mut hash_functions = Vec::new();
        let mut rng = Random::default();

        // Create random hash functions
        for _ in 0..num_hash_functions {
            let mut random_vector = Array1::<f64>::zeros(n_features);
            for i in 0..n_features {
                // Generate standard normal using Box-Muller transform
                let u1 = rng.gen::<f64>();
                let u2 = rng.gen::<f64>();
                let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                random_vector[i] = z;
            }
            let offset = rng.gen::<f64>() * bucket_width;

            hash_functions.push(RandomHashFunction {
                random_vector,
                offset,
                bucket_width,
            });
        }

        // Hash all data points
        let mut buckets = HashMap::new();
        for (row_idx, row) in data.rows().into_iter().enumerate() {
            let hash_values = self.compute_hash_values(&row.to_owned(), &hash_functions);
            buckets
                .entry(hash_values)
                .or_insert_with(Vec::new)
                .push(row_idx);
        }

        Ok(LocalityHashTable {
            hash_functions,
            buckets,
            num_hash_functions,
            bucket_width,
        })
    }

    /// Compute hash values for a data point
    fn compute_hash_values(
        &self,
        point: &Array1<f64>,
        hash_functions: &[RandomHashFunction],
    ) -> Vec<u32> {
        hash_functions
            .iter()
            .map(|hash_fn| {
                let dot_product: f64 = point
                    .iter()
                    .zip(hash_fn.random_vector.iter())
                    .filter(|(&x, _)| !self.is_missing(x))
                    .map(|(&x, &h)| x * h)
                    .sum();

                ((dot_product + hash_fn.offset) / hash_fn.bucket_width).floor() as u32
            })
            .collect()
    }
}

impl ApproximateKNNImputer<ApproximateKNNImputerTrained> {
    /// Find approximate neighbors for a query point
    fn find_approximate_neighbors(
        &self,
        query_row: &Array1<f64>,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, SklearsError> {
        match self.state.strategy {
            ApproximationStrategy::RandomSampling => {
                self.find_neighbors_random_sampling(query_row, target_feature)
            }
            ApproximationStrategy::HashBased => {
                self.find_neighbors_hash_based(query_row, target_feature)
            }
            ApproximationStrategy::LocalApproximation => {
                self.find_neighbors_local_approximation(query_row, target_feature)
            }
            _ => self.find_neighbors_random_sampling(query_row, target_feature),
        }
    }

    /// Find neighbors using random sampling
    fn find_neighbors_random_sampling(
        &self,
        query_row: &Array1<f64>,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, SklearsError> {
        let mut neighbors = Vec::new();
        let max_candidates = (self.n_neighbors * 3).min(self.state.reference_samples.nrows());

        // Randomly sample candidates for distance computation
        let mut rng = Random::default();
        let mut candidate_indices: Vec<usize> = (0..self.state.reference_samples.nrows()).collect();

        for i in (1..candidate_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            candidate_indices.swap(i, j);
        }

        candidate_indices.truncate(max_candidates);

        for &idx in &candidate_indices {
            let ref_row = self.state.reference_samples.row(idx);

            if self.is_missing(ref_row[target_feature]) {
                continue;
            }

            let distance = self.compute_approximate_distance(query_row, &ref_row.to_owned());
            if distance.is_finite() {
                neighbors.push((distance, ref_row[target_feature]));
            }
        }

        // Sort by distance and take k nearest
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.truncate(self.n_neighbors);

        Ok(neighbors)
    }

    /// Find neighbors using hash-based approach
    fn find_neighbors_hash_based(
        &self,
        query_row: &Array1<f64>,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, SklearsError> {
        if let Some(ref hash_table) = self.state.locality_hash {
            let query_hash = self.compute_query_hash_values(query_row, &hash_table.hash_functions);
            let mut candidates = HashSet::new();

            // Get candidates from the same bucket
            if let Some(bucket_candidates) = hash_table.buckets.get(&query_hash) {
                candidates.extend(bucket_candidates);
            }

            // If not enough candidates, check neighboring buckets
            if candidates.len() < self.n_neighbors * 2 {
                for (hash_key, bucket_candidates) in &hash_table.buckets {
                    let hamming_distance = self.hamming_distance(&query_hash, hash_key);
                    if hamming_distance <= 2 {
                        // Allow some hash collisions
                        candidates.extend(bucket_candidates);
                    }
                    if candidates.len() >= self.n_neighbors * 3 {
                        break;
                    }
                }
            }

            if candidates.is_empty() {
                return self.find_neighbors_random_sampling(query_row, target_feature);
            }

            // Compute distances to candidates
            let mut neighbors = Vec::new();
            for &idx in &candidates {
                let ref_row = self.state.reference_samples.row(idx);

                if self.is_missing(ref_row[target_feature]) {
                    continue;
                }

                let distance = self.compute_approximate_distance(query_row, &ref_row.to_owned());
                if distance.is_finite() {
                    neighbors.push((distance, ref_row[target_feature]));
                }
            }

            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            neighbors.truncate(self.n_neighbors);

            if neighbors.is_empty() {
                return self.find_neighbors_random_sampling(query_row, target_feature);
            }

            Ok(neighbors)
        } else {
            self.find_neighbors_random_sampling(query_row, target_feature)
        }
    }

    /// Find neighbors using local approximation
    fn find_neighbors_local_approximation(
        &self,
        query_row: &Array1<f64>,
        target_feature: usize,
    ) -> Result<Vec<(f64, f64)>, SklearsError> {
        // Use a subset of features for distance computation
        let n_features = query_row.len();
        let subset_size = ((n_features as f64 * self.state.config.accuracy_level) as usize).max(1);

        let mut rng = Random::default();
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        for i in (1..feature_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            feature_indices.swap(i, j);
        }
        feature_indices.truncate(subset_size);
        feature_indices.sort();

        // Compute distances using subset of features
        let mut neighbors = Vec::new();
        for ref_row in self.state.reference_samples.rows() {
            if self.is_missing(ref_row[target_feature]) {
                continue;
            }

            let distance =
                self.compute_subset_distance(query_row, &ref_row.to_owned(), &feature_indices);
            if distance.is_finite() {
                neighbors.push((distance, ref_row[target_feature]));
            }
        }

        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.truncate(self.n_neighbors);

        Ok(neighbors)
    }

    /// Compute hash values for query point
    fn compute_query_hash_values(
        &self,
        query_row: &Array1<f64>,
        hash_functions: &[RandomHashFunction],
    ) -> Vec<u32> {
        hash_functions
            .iter()
            .map(|hash_fn| {
                let dot_product: f64 = query_row
                    .iter()
                    .zip(hash_fn.random_vector.iter())
                    .filter(|(&x, _)| !self.is_missing(x))
                    .map(|(&x, &h)| x * h)
                    .sum();

                ((dot_product + hash_fn.offset) / hash_fn.bucket_width).floor() as u32
            })
            .collect()
    }

    /// Compute Hamming distance between hash values
    fn hamming_distance(&self, hash1: &[u32], hash2: &[u32]) -> usize {
        hash1
            .iter()
            .zip(hash2.iter())
            .map(|(a, b)| if a == b { 0 } else { 1 })
            .sum()
    }

    /// Compute approximate distance (using fewer features)
    fn compute_approximate_distance(&self, row1: &Array1<f64>, row2: &Array1<f64>) -> f64 {
        let mut sum_sq = 0.0;
        let mut valid_count = 0;

        // Use sampling to reduce computation
        let sample_rate = self.state.config.accuracy_level;
        let mut rng = Random::default();

        for (i, (&x1, &x2)) in row1.iter().zip(row2.iter()).enumerate() {
            // Skip some features based on sampling rate
            if rng.gen::<f64>() > sample_rate {
                continue;
            }

            if !self.is_missing(x1) && !self.is_missing(x2) {
                sum_sq += (x1 - x2).powi(2);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            (sum_sq / valid_count as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Compute distance using subset of features
    fn compute_subset_distance(
        &self,
        row1: &Array1<f64>,
        row2: &Array1<f64>,
        feature_indices: &[usize],
    ) -> f64 {
        let mut sum_sq = 0.0;
        let mut valid_count = 0;

        for &idx in feature_indices {
            let x1 = row1[idx];
            let x2 = row2[idx];

            if !self.is_missing(x1) && !self.is_missing(x2) {
                sum_sq += (x1 - x2).powi(2);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            (sum_sq / valid_count as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Compute weighted average of neighbor values
    fn compute_weighted_average(&self, neighbors: &[(f64, f64)]) -> Result<f64, SklearsError> {
        if neighbors.is_empty() {
            return Ok(0.0);
        }

        match self.weights.as_str() {
            "uniform" => {
                let sum: f64 = neighbors.iter().map(|(_, value)| value).sum();
                Ok(sum / neighbors.len() as f64)
            }
            "distance" => {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for &(distance, value) in neighbors {
                    let weight = if distance > 0.0 { 1.0 / distance } else { 1e6 };
                    weighted_sum += weight * value;
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    Ok(weighted_sum / weight_sum)
                } else {
                    Ok(neighbors[0].1)
                }
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown weights: {}",
                self.weights
            ))),
        }
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

// Implement Approximate Simple Imputer
impl ApproximateSimpleImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            strategy: "mean".to_string(),
            missing_values: f64::NAN,
            config: ApproximateConfig::default(),
        }
    }

    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn approximate_config(mut self, config: ApproximateConfig) -> Self {
        self.config = config;
        self
    }

    pub fn sample_size(mut self, size: usize) -> Self {
        self.config.sample_size = size;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for ApproximateSimpleImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ApproximateSimpleImputer<Untrained> {
    type Config = ApproximateConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ApproximateSimpleImputer<Untrained> {
    type Fitted = ApproximateSimpleImputer<ApproximateSimpleImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        // Determine sample size for approximation
        let sample_size = (self.config.sample_size as f64 * self.config.accuracy_level) as usize;
        let effective_sample_size = sample_size.min(n_samples);

        // Compute approximate statistics using sampling
        let (approximate_statistics, confidence_intervals) =
            self.compute_approximate_statistics(&X, effective_sample_size)?;

        Ok(ApproximateSimpleImputer {
            state: ApproximateSimpleImputerTrained {
                approximate_statistics_: approximate_statistics,
                confidence_intervals_: confidence_intervals,
                n_features_in_: n_features,
                config: self.config,
            },
            strategy: self.strategy,
            missing_values: self.missing_values,
            config: Default::default(),
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for ApproximateSimpleImputer<ApproximateSimpleImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        // Apply imputation in parallel
        X_imputed
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                for (j, value) in row.iter_mut().enumerate() {
                    if self.is_missing(*value) {
                        *value = self.state.approximate_statistics_[j];
                    }
                }
            });

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl ApproximateSimpleImputer<Untrained> {
    /// Compute approximate statistics using sampling
    fn compute_approximate_statistics(
        &self,
        X: &Array2<f64>,
        sample_size: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), SklearsError> {
        let (n_samples, n_features) = X.dim();
        let mut approximate_statistics = Array1::<f64>::zeros(n_features);
        let mut confidence_intervals = Array2::<f64>::zeros((n_features, 2)); // [lower, upper]

        // Use bootstrap sampling for confidence intervals
        let num_bootstrap_samples = 100;

        for j in 0..n_features {
            let mut bootstrap_estimates = Vec::new();

            for _ in 0..num_bootstrap_samples {
                // Sample with replacement
                let mut rng = Random::default();
                let mut sample_values = Vec::new();

                for _ in 0..sample_size {
                    let sample_idx = rng.gen_range(0..n_samples);
                    let value = X[[sample_idx, j]];
                    if !self.is_missing(value) {
                        sample_values.push(value);
                    }
                }

                if sample_values.is_empty() {
                    continue;
                }

                let estimate = match self.strategy.as_str() {
                    "mean" => sample_values.iter().sum::<f64>() / sample_values.len() as f64,
                    "median" => {
                        let mut sorted = sample_values.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mid = sorted.len() / 2;
                        if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        }
                    }
                    _ => sample_values.iter().sum::<f64>() / sample_values.len() as f64,
                };

                bootstrap_estimates.push(estimate);
            }

            if !bootstrap_estimates.is_empty() {
                // Main estimate
                approximate_statistics[j] =
                    bootstrap_estimates.iter().sum::<f64>() / bootstrap_estimates.len() as f64;

                // Confidence interval (5th and 95th percentiles)
                bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let lower_idx = (bootstrap_estimates.len() as f64 * 0.05) as usize;
                let upper_idx = (bootstrap_estimates.len() as f64 * 0.95) as usize;

                confidence_intervals[[j, 0]] =
                    bootstrap_estimates[lower_idx.min(bootstrap_estimates.len() - 1)];
                confidence_intervals[[j, 1]] =
                    bootstrap_estimates[upper_idx.min(bootstrap_estimates.len() - 1)];
            }
        }

        Ok((approximate_statistics, confidence_intervals))
    }
}

impl ApproximateSimpleImputer<ApproximateSimpleImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    /// Get confidence intervals for imputed values
    pub fn confidence_intervals(&self) -> &Array2<f64> {
        &self.state.confidence_intervals_
    }

    /// Get approximate statistics
    pub fn statistics(&self) -> &Array1<f64> {
        &self.state.approximate_statistics_
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_approximate_simple_imputer() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let imputer = ApproximateSimpleImputer::new()
            .strategy("mean".to_string())
            .sample_size(100);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Check that NaN was replaced (value should be reasonable)
        assert!(!X_imputed[[1, 1]].is_nan());
        assert!(X_imputed[[1, 1]] > 0.0);
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_approximate_knn_imputer() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ];

        let imputer = ApproximateKNNImputer::new()
            .n_neighbors(2)
            .weights("uniform".to_string())
            .accuracy_level(0.8)
            .sample_size(3);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Verify that missing value was imputed
        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_approximate_config() {
        let config = ApproximateConfig {
            accuracy_level: 0.5,
            sample_size: 500,
            use_randomization: false,
            ..Default::default()
        };

        let imputer = ApproximateSimpleImputer::new().approximate_config(config.clone());

        assert_eq!(imputer.config.accuracy_level, 0.5);
        assert_eq!(imputer.config.sample_size, 500);
        assert_eq!(imputer.config.use_randomization, false);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_hash_based_strategy() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        let imputer = ApproximateKNNImputer::new()
            .n_neighbors(2)
            .strategy(ApproximationStrategy::HashBased)
            .accuracy_level(0.9);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Verify that missing value was imputed
        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_confidence_intervals() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let imputer = ApproximateSimpleImputer::new().strategy("mean".to_string());

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let confidence_intervals = fitted.confidence_intervals();

        // Check that confidence intervals exist and make sense
        assert_eq!(confidence_intervals.shape(), &[3, 2]);

        for j in 0..3 {
            let lower = confidence_intervals[[j, 0]];
            let upper = confidence_intervals[[j, 1]];
            assert!(
                lower <= upper,
                "Lower bound should be <= upper bound for feature {}",
                j
            );
        }
    }
}
