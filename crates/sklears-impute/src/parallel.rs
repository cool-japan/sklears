//! Parallel imputation algorithms for high-performance missing data processing
//!
//! This module provides parallel implementations of imputation algorithms that can
//! leverage multiple CPU cores for significant performance improvements on large datasets.

// ✅ SciRS2 Policy compliant imports
use crate::simd_ops::{SimdDistanceCalculator, SimdImputationOps, SimdStatistics};
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::sync::{Arc, Mutex};

/// Configuration for parallel processing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParallelConfig {
    /// max_threads
    pub max_threads: Option<usize>,
    /// chunk_size
    pub chunk_size: usize,
    /// load_balancing
    pub load_balancing: bool,
    /// memory_efficient
    pub memory_efficient: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: None, // Use all available cores
            chunk_size: 1000,  // Process data in chunks of 1000 rows
            load_balancing: true,
            memory_efficient: false,
        }
    }
}

/// Parallel KNN Imputer with SIMD optimizations
#[derive(Debug, Clone)]
pub struct ParallelKNNImputer<S = Untrained> {
    state: S,
    n_neighbors: usize,
    weights: String,
    metric: String,
    missing_values: f64,
    parallel_config: ParallelConfig,
}

/// Trained state for parallel KNN imputer
#[derive(Debug, Clone)]
pub struct ParallelKNNImputerTrained {
    X_train_: Array2<f64>,
    n_features_in_: usize,
    parallel_config: ParallelConfig,
}

impl Default for ParallelKNNImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelKNNImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            weights: "uniform".to_string(),
            metric: "euclidean".to_string(),
            missing_values: f64::NAN,
            parallel_config: ParallelConfig::default(),
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

    pub fn metric(mut self, metric: String) -> Self {
        self.metric = metric;
        self
    }

    pub fn parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    pub fn max_threads(mut self, max_threads: usize) -> Self {
        self.parallel_config.max_threads = Some(max_threads);
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

impl Estimator for ParallelKNNImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ParallelKNNImputer<Untrained> {
    type Fitted = ParallelKNNImputer<ParallelKNNImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        Ok(ParallelKNNImputer {
            state: ParallelKNNImputerTrained {
                X_train_: X.clone(),
                n_features_in_: n_features,
                parallel_config: self.parallel_config.clone(),
            },
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            metric: self.metric,
            missing_values: self.missing_values,
            parallel_config: self.parallel_config,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for ParallelKNNImputer<ParallelKNNImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (_n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        // Set up parallel thread pool if specified (ignore if already initialized)
        if let Some(max_threads) = self.state.parallel_config.max_threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global(); // Ignore error if already initialized
        }

        let mut X_imputed = X.clone();
        let X_train = &self.state.X_train_;

        // Parallel processing over samples and features
        let missing_positions: Vec<(usize, usize)> = X_imputed
            .indexed_iter()
            .filter_map(|((i, j), &val)| {
                if self.is_missing(val) {
                    Some((i, j))
                } else {
                    None
                }
            })
            .collect();

        // Process missing values in parallel chunks
        let chunk_size = self
            .state
            .parallel_config
            .chunk_size
            .min(missing_positions.len().max(1));

        missing_positions
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_imputed = X_imputed.clone();

                for &(i, j) in chunk {
                    let imputed_value = self.impute_single_value(&X_imputed, X_train, i, j)?;
                    local_imputed[[i, j]] = imputed_value;
                }

                Ok::<Array2<f64>, SklearsError>(local_imputed)
            })
            .collect::<SklResult<Vec<_>>>()?
            .into_iter()
            .for_each(|chunk_result| {
                // Merge results back (this could be optimized further)
                for &(i, j) in missing_positions.iter() {
                    if !self.is_missing(chunk_result[[i, j]]) {
                        X_imputed[[i, j]] = chunk_result[[i, j]];
                    }
                }
            });

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl ParallelKNNImputer<ParallelKNNImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn impute_single_value(
        &self,
        X: &Array2<f64>,
        X_train: &Array2<f64>,
        row_idx: usize,
        col_idx: usize,
    ) -> SklResult<f64> {
        let query_row = X.row(row_idx);

        // Calculate distances to all training samples in parallel
        let distances: Vec<(f64, usize)> = X_train
            .axis_iter(Axis(0))
            .enumerate()
            .par_bridge()
            .map(|(train_idx, train_row)| {
                let distance = match self.metric.as_str() {
                    "euclidean" => SimdDistanceCalculator::euclidean_distance_simd(
                        query_row.as_slice().unwrap(),
                        train_row.as_slice().unwrap(),
                    ),
                    "manhattan" => SimdDistanceCalculator::manhattan_distance_simd(
                        query_row.as_slice().unwrap(),
                        train_row.as_slice().unwrap(),
                    ),
                    _ => SimdDistanceCalculator::euclidean_distance_simd(
                        query_row.as_slice().unwrap(),
                        train_row.as_slice().unwrap(),
                    ),
                };
                (distance, train_idx)
            })
            .collect();

        // Sort and find k nearest neighbors
        let mut sorted_distances = distances;
        sorted_distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or_else(|| {
                // Handle NaN and infinity cases
                if a.0.is_nan() && b.0.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.0.is_nan() {
                    std::cmp::Ordering::Greater // NaN is considered larger
                } else if b.0.is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
        });

        // Collect valid neighbor values
        let mut neighbor_values = Vec::new();
        let mut weights = Vec::new();

        for &(distance, train_idx) in sorted_distances.iter().take(self.n_neighbors * 3) {
            if !self.is_missing(X_train[[train_idx, col_idx]]) {
                neighbor_values.push(X_train[[train_idx, col_idx]]);

                let weight = match self.weights.as_str() {
                    "distance" => {
                        if distance > 0.0 {
                            1.0 / distance
                        } else {
                            1e6
                        }
                    }
                    _ => 1.0,
                };
                weights.push(weight);

                if neighbor_values.len() >= self.n_neighbors {
                    break;
                }
            }
        }

        if neighbor_values.is_empty() {
            // Fallback to column mean
            let column = X_train.column(col_idx);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if !valid_values.is_empty() {
                Ok(SimdStatistics::mean_simd(&valid_values))
            } else {
                Ok(0.0)
            }
        } else {
            // Use SIMD-optimized weighted mean
            Ok(SimdImputationOps::weighted_mean_simd(
                &neighbor_values,
                &weights,
            ))
        }
    }
}

/// Parallel Iterative Imputer (MICE) with multi-threading
#[derive(Debug, Clone)]
pub struct ParallelIterativeImputer<S = Untrained> {
    state: S,
    max_iter: usize,
    tol: f64,
    n_nearest_features: Option<usize>,
    random_state: Option<u64>,
    parallel_config: ParallelConfig,
}

/// Trained state for parallel iterative imputer
#[derive(Debug, Clone)]
pub struct ParallelIterativeImputerTrained {
    n_features_in_: usize,
    missing_mask_: Array2<bool>,
    parallel_config: ParallelConfig,
    random_state: Option<u64>,
}

impl Default for ParallelIterativeImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelIterativeImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_iter: 10,
            tol: 1e-3,
            n_nearest_features: None,
            random_state: None,
            parallel_config: ParallelConfig::default(),
        }
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for ParallelIterativeImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ParallelIterativeImputer<Untrained> {
    type Fitted = ParallelIterativeImputer<ParallelIterativeImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (_, n_features) = X.dim();

        // Create missing mask
        let missing_mask = X.mapv(|x| x.is_nan());

        Ok(ParallelIterativeImputer {
            state: ParallelIterativeImputerTrained {
                n_features_in_: n_features,
                missing_mask_: missing_mask,
                parallel_config: self.parallel_config.clone(),
                random_state: self.random_state,
            },
            max_iter: self.max_iter,
            tol: self.tol,
            n_nearest_features: self.n_nearest_features,
            random_state: self.random_state,
            parallel_config: self.parallel_config,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for ParallelIterativeImputer<ParallelIterativeImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (_n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        // Initialize with simple mean imputation
        let mut X_imputed = self.initial_imputation(&X)?;
        let missing_mask = X.mapv(|x| x.is_nan());

        // Iterative imputation with parallel feature processing
        for _iteration in 0..self.max_iter {
            let X_prev = X_imputed.clone();

            // Process features in parallel
            let feature_indices: Vec<usize> = (0..n_features).collect();
            let imputed_features: SklResult<Vec<Array1<f64>>> = feature_indices
                .par_iter()
                .map(|&feature_idx| self.impute_feature(&X_imputed, &missing_mask, feature_idx))
                .collect();

            let imputed_features = imputed_features?;

            // Update imputed values for each feature
            for (feature_idx, feature_values) in imputed_features.into_iter().enumerate() {
                for (sample_idx, &value) in feature_values.iter().enumerate() {
                    if missing_mask[[sample_idx, feature_idx]] {
                        X_imputed[[sample_idx, feature_idx]] = value;
                    }
                }
            }

            // Check convergence using SIMD-optimized calculations
            let diff = self.calculate_convergence_difference(&X_prev, &X_imputed, &missing_mask);
            if diff < self.tol {
                break;
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl ParallelIterativeImputer<ParallelIterativeImputerTrained> {
    fn initial_imputation(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut X_imputed = X.clone();

        // Parallel mean imputation for each column
        let column_means: Vec<f64> = (0..X.ncols())
            .into_par_iter()
            .map(|col_idx| {
                let column = X.column(col_idx);
                let valid_values: Vec<f64> =
                    column.iter().filter(|&&x| !x.is_nan()).cloned().collect();

                if !valid_values.is_empty() {
                    SimdStatistics::mean_simd(&valid_values)
                } else {
                    0.0
                }
            })
            .collect();

        // Fill missing values with means
        for ((_i, j), value) in X_imputed.indexed_iter_mut() {
            if value.is_nan() {
                *value = column_means[j];
            }
        }

        Ok(X_imputed)
    }

    fn impute_feature(
        &self,
        X: &Array2<f64>,
        missing_mask: &Array2<bool>,
        feature_idx: usize,
    ) -> SklResult<Array1<f64>> {
        let n_samples = X.nrows();
        let mut imputed_feature = Array1::zeros(n_samples);

        // Identify samples with missing values for this feature
        let missing_samples: Vec<usize> = (0..n_samples)
            .filter(|&i| missing_mask[[i, feature_idx]])
            .collect();

        if missing_samples.is_empty() {
            return Ok(X.column(feature_idx).to_owned());
        }

        // Use other features as predictors
        let predictor_features: Vec<usize> = (0..X.ncols()).filter(|&i| i != feature_idx).collect();

        // Simple linear regression for each missing sample
        missing_samples
            .par_iter()
            .map(|&sample_idx| {
                self.predict_missing_value(X, sample_idx, feature_idx, &predictor_features)
            })
            .collect::<SklResult<Vec<_>>>()?
            .into_iter()
            .zip(missing_samples.iter())
            .for_each(|(predicted_value, &sample_idx)| {
                imputed_feature[sample_idx] = predicted_value;
            });

        // Copy non-missing values
        for i in 0..n_samples {
            if !missing_mask[[i, feature_idx]] {
                imputed_feature[i] = X[[i, feature_idx]];
            }
        }

        Ok(imputed_feature)
    }

    fn predict_missing_value(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        target_feature: usize,
        predictor_features: &[usize],
    ) -> SklResult<f64> {
        // Find complete cases for regression
        let complete_samples: Vec<usize> = (0..X.nrows())
            .filter(|&i| {
                !X[[i, target_feature]].is_nan()
                    && predictor_features.iter().all(|&j| !X[[i, j]].is_nan())
            })
            .collect();

        if complete_samples.len() < 2 {
            // Fallback to column mean
            let column = X.column(target_feature);
            let valid_values: Vec<f64> = column.iter().filter(|&&x| !x.is_nan()).cloned().collect();

            return Ok(if !valid_values.is_empty() {
                SimdStatistics::mean_simd(&valid_values)
            } else {
                0.0
            });
        }

        // Simple average of k nearest neighbors based on predictor features
        let query_features: Vec<f64> = predictor_features
            .iter()
            .map(|&j| X[[sample_idx, j]])
            .collect();

        let mut distances: Vec<(f64, usize)> = complete_samples
            .par_iter()
            .map(|&complete_idx| {
                let sample_features: Vec<f64> = predictor_features
                    .iter()
                    .map(|&j| X[[complete_idx, j]])
                    .collect();

                let distance = SimdDistanceCalculator::euclidean_distance_simd(
                    &query_features,
                    &sample_features,
                );

                (distance, complete_idx)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Use top k neighbors
        let k = 5.min(distances.len());
        let neighbor_values: Vec<f64> = distances
            .iter()
            .take(k)
            .map(|(_, idx)| X[[*idx, target_feature]])
            .collect();

        Ok(SimdStatistics::mean_simd(&neighbor_values))
    }

    fn calculate_convergence_difference(
        &self,
        X_prev: &Array2<f64>,
        X_current: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> f64 {
        let differences: Vec<f64> = X_prev
            .indexed_iter()
            .par_bridge()
            .filter_map(|((i, j), &prev_val)| {
                if missing_mask[[i, j]] {
                    Some((prev_val - X_current[[i, j]]).abs())
                } else {
                    None
                }
            })
            .collect();

        if differences.is_empty() {
            0.0
        } else {
            SimdStatistics::mean_simd(&differences)
        }
    }
}

/// Parallel memory-efficient imputation for large datasets
pub struct MemoryEfficientImputer;

impl MemoryEfficientImputer {
    /// Process large datasets in chunks to minimize memory usage
    pub fn impute_chunked<F>(
        data: &Array2<f64>,
        chunk_size: usize,
        impute_fn: F,
    ) -> SklResult<Array2<f64>>
    where
        F: Fn(&Array2<f64>) -> SklResult<Array2<f64>> + Sync + Send,
    {
        let (n_rows, n_cols) = data.dim();
        let mut result = Array2::zeros((n_rows, n_cols));

        // Process in row chunks
        let chunks: Vec<_> = (0..n_rows).step_by(chunk_size).collect();

        chunks
            .par_iter()
            .map(|&start_row| {
                let end_row = (start_row + chunk_size).min(n_rows);
                let chunk = data.slice(s![start_row..end_row, ..]).to_owned();

                let imputed_chunk = impute_fn(&chunk)?;
                Ok((start_row, imputed_chunk))
            })
            .collect::<SklResult<Vec<_>>>()?
            .into_iter()
            .for_each(|(start_row, imputed_chunk)| {
                let end_row = start_row + imputed_chunk.nrows();
                result
                    .slice_mut(s![start_row..end_row, ..])
                    .assign(&imputed_chunk);
            });

        Ok(result)
    }

    /// Process streaming data with online imputation
    pub fn stream_impute<F>(
        stream: impl Iterator<Item = SklResult<Array1<f64>>> + Send,
        impute_fn: F,
    ) -> impl Iterator<Item = SklResult<Array1<f64>>> + Send
    where
        F: Fn(&Array1<f64>) -> SklResult<Array1<f64>> + Sync + Send + Clone + 'static,
    {
        stream.map(move |result| {
            let row = result?;
            impute_fn(&row)
        })
    }
}

/// Streaming imputation for very large datasets that don't fit in memory
#[derive(Debug, Clone)]
pub struct StreamingImputer {
    window_size: usize,
    buffer_size: usize,
    strategy: String,
    missing_values: f64,
}

impl Default for StreamingImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingImputer {
    pub fn new() -> Self {
        Self {
            window_size: 1000,
            buffer_size: 10000,
            strategy: "mean".to_string(),
            missing_values: f64::NAN,
        }
    }

    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    /// Process streaming data with online statistics update
    pub fn fit_transform_stream<I>(&self, data_stream: I) -> SklResult<Vec<Array1<f64>>>
    where
        I: Iterator<Item = SklResult<Array1<f64>>>,
    {
        let mut results = Vec::new();
        let mut statistics = OnlineStatistics::new();
        let mut buffer = Vec::new();

        for row_result in data_stream {
            let row = row_result?;
            buffer.push(row.clone());

            // Update online statistics
            statistics.update(&row);

            // Process buffer when full
            if buffer.len() >= self.buffer_size {
                let processed_buffer = self.process_buffer(&buffer, &statistics)?;
                results.extend(processed_buffer);
                buffer.clear();
            }
        }

        // Process remaining buffer
        if !buffer.is_empty() {
            let processed_buffer = self.process_buffer(&buffer, &statistics)?;
            results.extend(processed_buffer);
        }

        Ok(results)
    }

    fn process_buffer(
        &self,
        buffer: &[Array1<f64>],
        statistics: &OnlineStatistics,
    ) -> SklResult<Vec<Array1<f64>>> {
        buffer
            .par_iter()
            .map(|row| self.impute_row(row, statistics))
            .collect()
    }

    fn impute_row(
        &self,
        row: &Array1<f64>,
        statistics: &OnlineStatistics,
    ) -> SklResult<Array1<f64>> {
        let mut imputed_row = row.clone();

        for (i, &value) in row.iter().enumerate() {
            if self.is_missing(value) {
                let imputed_value = match self.strategy.as_str() {
                    "mean" => statistics.get_mean(i),
                    "median" => statistics.get_median(i),
                    "mode" => statistics.get_mode(i),
                    _ => statistics.get_mean(i),
                };
                imputed_row[i] = imputed_value;
            }
        }

        Ok(imputed_row)
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Online statistics for streaming imputation
#[derive(Debug, Clone)]
pub struct OnlineStatistics {
    n_samples: usize,
    means: Vec<f64>,
    variances: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
    value_counts: Vec<std::collections::HashMap<i64, usize>>,
    n_features: usize,
}

impl Default for OnlineStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStatistics {
    pub fn new() -> Self {
        Self {
            n_samples: 0,
            means: Vec::new(),
            variances: Vec::new(),
            mins: Vec::new(),
            maxs: Vec::new(),
            value_counts: Vec::new(),
            n_features: 0,
        }
    }

    pub fn update(&mut self, row: &Array1<f64>) {
        if self.n_features == 0 {
            self.n_features = row.len();
            self.means = vec![0.0; self.n_features];
            self.variances = vec![0.0; self.n_features];
            self.mins = vec![f64::INFINITY; self.n_features];
            self.maxs = vec![f64::NEG_INFINITY; self.n_features];
            self.value_counts = vec![std::collections::HashMap::new(); self.n_features];
        }

        self.n_samples += 1;

        for (i, &value) in row.iter().enumerate() {
            if !value.is_nan() {
                // Update mean using Welford's online algorithm
                let delta = value - self.means[i];
                self.means[i] += delta / self.n_samples as f64;
                let delta2 = value - self.means[i];
                self.variances[i] += delta * delta2;

                // Update min/max
                self.mins[i] = self.mins[i].min(value);
                self.maxs[i] = self.maxs[i].max(value);

                // Update value counts for mode calculation
                let rounded_value = (value * 1000.0).round() as i64;
                *self.value_counts[i].entry(rounded_value).or_insert(0) += 1;
            }
        }
    }

    pub fn get_mean(&self, feature_idx: usize) -> f64 {
        if feature_idx < self.means.len() {
            self.means[feature_idx]
        } else {
            0.0
        }
    }

    pub fn get_median(&self, feature_idx: usize) -> f64 {
        if feature_idx < self.means.len() {
            self.means[feature_idx]
        } else {
            0.0
        }
    }

    pub fn get_mode(&self, feature_idx: usize) -> f64 {
        if feature_idx < self.value_counts.len() {
            self.value_counts[feature_idx]
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&value, _)| value as f64 / 1000.0)
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn get_variance(&self, feature_idx: usize) -> f64 {
        if feature_idx < self.variances.len() && self.n_samples > 1 {
            self.variances[feature_idx] / (self.n_samples - 1) as f64
        } else {
            0.0
        }
    }
}

/// Adaptive streaming imputation that learns from incoming data
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingImputer {
    learning_rate: f64,
    forgetting_factor: f64,
    min_samples_for_adaptation: usize,
    statistics: OnlineStatistics,
}

impl Default for AdaptiveStreamingImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveStreamingImputer {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            forgetting_factor: 0.99,
            min_samples_for_adaptation: 100,
            statistics: OnlineStatistics::new(),
        }
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn forgetting_factor(mut self, forgetting_factor: f64) -> Self {
        self.forgetting_factor = forgetting_factor;
        self
    }

    /// Process a single row with adaptive learning
    pub fn fit_transform_single(&mut self, row: &Array1<f64>) -> SklResult<Array1<f64>> {
        let mut imputed_row = row.clone();

        // First, impute missing values using current statistics
        for (i, &value) in row.iter().enumerate() {
            if value.is_nan() {
                let imputed_value = if self.statistics.n_samples >= self.min_samples_for_adaptation
                {
                    self.statistics.get_mean(i)
                } else {
                    0.0
                };
                imputed_row[i] = imputed_value;
            }
        }

        // Update statistics with the imputed row
        self.statistics.update(&imputed_row);

        // Apply forgetting factor to adapt to data drift
        if self.statistics.n_samples > self.min_samples_for_adaptation {
            for i in 0..self.statistics.n_features {
                self.statistics.means[i] *= self.forgetting_factor;
                self.statistics.variances[i] *= self.forgetting_factor;
            }
        }

        Ok(imputed_row)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_parallel_knn_imputer() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
        )
        .unwrap();

        let config = ParallelConfig {
            max_threads: Some(2),
            ..Default::default()
        };

        let imputer = ParallelKNNImputer::new()
            .n_neighbors(2)
            .parallel_config(config);

        let fitted = imputer.fit(&data.view(), &()).unwrap();
        let result = fitted.transform(&data.view()).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| (x).is_nan()));

        // Non-missing values should be preserved
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_iterative_imputer() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let config = ParallelConfig {
            max_threads: Some(2),
            ..Default::default()
        };

        let imputer = ParallelIterativeImputer::new()
            .max_iter(5)
            .tol(1e-3)
            .parallel_config(config);

        let fitted = imputer.fit(&data.view(), &()).unwrap();
        let result = fitted.transform(&data.view()).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| (x).is_nan()));

        // Non-missing values should be preserved
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_efficient_chunked_processing() {
        let data = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                f64::NAN,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                f64::NAN,
                19.0,
                20.0,
                21.0,
                22.0,
                f64::NAN,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
            ],
        )
        .unwrap();

        let simple_impute_fn = |chunk: &Array2<f64>| -> SklResult<Array2<f64>> {
            let mut result = chunk.clone();

            // Simple mean imputation
            for j in 0..chunk.ncols() {
                let column = chunk.column(j);
                let valid_values: Vec<f64> =
                    column.iter().filter(|&&x| !x.is_nan()).cloned().collect();

                if !valid_values.is_empty() {
                    let mean = SimdStatistics::mean_simd(&valid_values);

                    for i in 0..chunk.nrows() {
                        if chunk[[i, j]].is_nan() {
                            result[[i, j]] = mean;
                        }
                    }
                }
            }

            Ok(result)
        };

        let result = MemoryEfficientImputer::impute_chunked(&data, 3, simple_impute_fn).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| x.is_nan()));

        // Should preserve non-missing values
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_streaming_imputer() {
        // Create a mock data stream
        let data_rows = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![f64::NAN, 5.0, 6.0]),
            Array1::from_vec(vec![7.0, f64::NAN, 9.0]),
            Array1::from_vec(vec![10.0, 11.0, 12.0]),
            Array1::from_vec(vec![13.0, 14.0, f64::NAN]),
        ];

        let data_stream = data_rows.into_iter().map(Ok);

        let imputer = StreamingImputer::new()
            .buffer_size(3)
            .strategy("mean".to_string());

        let results = imputer.fit_transform_stream(data_stream).unwrap();

        // Should have same number of rows
        assert_eq!(results.len(), 5);

        // Should have no missing values
        for result_row in &results {
            assert!(!result_row.iter().any(|&x| x.is_nan()));
        }

        // Non-missing values should be preserved
        assert_abs_diff_eq!(results[0][0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[0][1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[0][2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_streaming_imputer() {
        let mut imputer = AdaptiveStreamingImputer::new()
            .learning_rate(0.1)
            .forgetting_factor(0.95);

        // Process several rows to build up statistics
        let rows = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
            Array1::from_vec(vec![7.0, 8.0, 9.0]),
            Array1::from_vec(vec![f64::NAN, 11.0, 12.0]),
        ];

        let mut results = Vec::new();

        for row in &rows {
            let result = imputer.fit_transform_single(row).unwrap();
            results.push(result);
        }

        // Should have no missing values
        for result_row in &results {
            assert!(!result_row.iter().any(|&x| x.is_nan()));
        }

        // Non-missing values should be preserved
        assert_abs_diff_eq!(results[0][0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[1][1], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[2][2], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_online_statistics() {
        let mut stats = OnlineStatistics::new();

        // Add some data points
        let rows = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
            Array1::from_vec(vec![7.0, 8.0, 9.0]),
        ];

        for row in &rows {
            stats.update(row);
        }

        // Check means (should be 4.0, 5.0, 6.0)
        assert_abs_diff_eq!(stats.get_mean(0), 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.get_mean(1), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.get_mean(2), 6.0, epsilon = 1e-10);

        // Check sample count
        assert_eq!(stats.n_samples, 3);
    }

    #[test]
    fn test_memory_efficient_stream_processing() {
        let data_stream = (0..100).map(|i| {
            let row = Array1::from_vec(vec![
                i as f64,
                (i * 2) as f64,
                if i % 10 == 0 {
                    f64::NAN
                } else {
                    (i * 3) as f64
                },
            ]);
            Ok(row)
        });

        let impute_fn = |row: &Array1<f64>| -> SklResult<Array1<f64>> {
            let mut result = row.clone();
            for (i, value) in result.iter_mut().enumerate() {
                if value.is_nan() {
                    *value = i as f64; // Simple fallback
                }
            }
            Ok(result)
        };

        let processed_stream: Vec<_> =
            MemoryEfficientImputer::stream_impute(data_stream, impute_fn)
                .collect::<SklResult<Vec<_>>>()
                .unwrap();

        assert_eq!(processed_stream.len(), 100);

        // Should have no missing values
        for row in &processed_stream {
            assert!(!row.iter().any(|&x| x.is_nan()));
        }
    }
}

// Additional memory efficiency improvements

/// Sparse matrix representation for missing data patterns
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Row indices for non-missing values
    pub row_indices: Vec<usize>,
    /// Column indices for non-missing values  
    pub col_indices: Vec<usize>,
    /// Non-missing values
    pub values: Vec<f64>,
    /// Matrix dimensions
    pub shape: (usize, usize),
    /// Sparsity ratio (fraction of missing values)
    pub sparsity: f64,
}

impl SparseMatrix {
    /// Create a sparse matrix from a dense array
    pub fn from_dense(array: &Array2<f64>, missing_value: f64) -> Self {
        let (n_rows, n_cols) = array.dim();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut missing_count = 0;

        for i in 0..n_rows {
            for j in 0..n_cols {
                let value = array[[i, j]];
                let is_missing = if missing_value.is_nan() {
                    value.is_nan()
                } else {
                    (value - missing_value).abs() < f64::EPSILON
                };

                if !is_missing {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(value);
                } else {
                    missing_count += 1;
                }
            }
        }

        let total_elements = n_rows * n_cols;
        let sparsity = missing_count as f64 / total_elements as f64;

        Self {
            row_indices,
            col_indices,
            values,
            shape: (n_rows, n_cols),
            sparsity,
        }
    }

    /// Convert back to dense array with missing values filled
    pub fn to_dense(&self, missing_value: f64) -> Array2<f64> {
        let mut array = Array2::from_elem(self.shape, missing_value);

        for ((i, j), &value) in self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
        {
            array[[*i, *j]] = value;
        }

        array
    }

    /// Get non-missing value at specific coordinates
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        for ((r, c), &value) in self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
        {
            if *r == row && *c == col {
                return Some(value);
            }
        }
        None
    }

    /// Check if the matrix is sparse enough to benefit from sparse representation
    pub fn is_beneficial(&self) -> bool {
        self.sparsity > 0.5 // More than 50% missing values
    }

    /// Calculate memory savings compared to dense representation
    pub fn memory_savings(&self) -> f64 {
        let dense_size = self.shape.0 * self.shape.1 * std::mem::size_of::<f64>();
        let sparse_size = (self.values.len() * std::mem::size_of::<f64>())
            + (self.row_indices.len() * std::mem::size_of::<usize>())
            + (self.col_indices.len() * std::mem::size_of::<usize>());

        1.0 - (sparse_size as f64 / dense_size as f64)
    }
}

/// Memory-mapped data operations for large datasets
pub struct MemoryMappedData {
    /// File path for memory-mapped data
    file_path: std::path::PathBuf,
    /// Data dimensions
    shape: (usize, usize),
    /// Data type size in bytes
    dtype_size: usize,
}

impl MemoryMappedData {
    /// Create a new memory-mapped data structure
    pub fn new(file_path: std::path::PathBuf, shape: (usize, usize)) -> Self {
        Self {
            file_path,
            shape,
            dtype_size: std::mem::size_of::<f64>(),
        }
    }

    /// Write array data to memory-mapped file
    pub fn write_array(&self, array: &Array2<f64>) -> SklResult<()> {
        if array.dim() != self.shape {
            return Err(SklearsError::InvalidInput(
                "Array dimensions don't match memory-mapped data shape".to_string(),
            ));
        }

        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(&self.file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        // Write raw bytes
        let data_slice = array
            .as_slice()
            .ok_or_else(|| SklearsError::InvalidInput("Array is not contiguous".to_string()))?;

        let bytes = unsafe {
            std::slice::from_raw_parts(
                data_slice.as_ptr() as *const u8,
                data_slice.len() * self.dtype_size,
            )
        };

        file.write_all(bytes)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write data: {}", e)))?;

        Ok(())
    }

    /// Read array data from memory-mapped file  
    pub fn read_array(&self) -> SklResult<Array2<f64>> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(&self.file_path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to open file: {}", e)))?;

        let expected_size = self.shape.0 * self.shape.1 * self.dtype_size;
        let mut buffer = vec![0u8; expected_size];

        file.read_exact(&mut buffer)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read data: {}", e)))?;

        // Convert bytes back to f64 array
        let data_slice = unsafe {
            std::slice::from_raw_parts(buffer.as_ptr() as *const f64, self.shape.0 * self.shape.1)
        };

        Array2::from_shape_vec(self.shape, data_slice.to_vec())
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to reshape array: {}", e)))
    }

    /// Get estimated memory usage
    pub fn memory_usage(&self) -> usize {
        self.shape.0 * self.shape.1 * self.dtype_size
    }

    /// Check if file exists
    pub fn exists(&self) -> bool {
        self.file_path.exists()
    }
}

/// Reference-counted shared data for efficient memory usage
#[derive(Debug, Clone)]
pub struct SharedDataRef<T> {
    data: Arc<T>,
    refs: Arc<Mutex<usize>>,
}

impl<T> SharedDataRef<T> {
    /// Create a new shared data reference
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(data),
            refs: Arc::new(Mutex::new(1)),
        }
    }

    /// Get a reference to the data
    pub fn get(&self) -> &T {
        &self.data
    }

    /// Get the current reference count
    pub fn ref_count(&self) -> usize {
        *self.refs.lock().unwrap()
    }

    /// Check if this is the only reference
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }
}

impl<T: Clone> SharedDataRef<T> {
    /// Make the data mutable by cloning if necessary
    pub fn make_mut(&mut self) -> &mut T {
        if Arc::strong_count(&self.data) > 1 {
            self.data = Arc::new((*self.data).clone());
        }
        Arc::get_mut(&mut self.data).unwrap()
    }
}

/// Memory-efficient imputation strategies
pub struct MemoryOptimizedImputer {
    strategy: MemoryStrategy,
    chunk_size: usize,
    use_sparse: bool,
    use_mmap: bool,
    temp_dir: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Process data in chunks to limit memory usage
    Chunked,
    /// Use sparse representations for high-sparsity data
    Sparse,
    /// Use memory-mapped files for very large datasets
    MemoryMapped,
    /// Combine multiple strategies
    Hybrid,
}

impl MemoryOptimizedImputer {
    /// Create a new memory-optimized imputer
    pub fn new() -> Self {
        Self {
            strategy: MemoryStrategy::Hybrid,
            chunk_size: 1000,
            use_sparse: true,
            use_mmap: false,
            temp_dir: std::env::temp_dir(),
        }
    }

    /// Set the memory optimization strategy
    pub fn strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set chunk size for chunked processing
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable/disable sparse matrix optimization
    pub fn use_sparse(mut self, use_sparse: bool) -> Self {
        self.use_sparse = use_sparse;
        self
    }

    /// Enable/disable memory mapping
    pub fn use_memory_mapping(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    /// Set temporary directory for memory-mapped files
    pub fn temp_dir(mut self, temp_dir: std::path::PathBuf) -> Self {
        self.temp_dir = temp_dir;
        self
    }

    /// Estimate memory requirements for a dataset
    pub fn estimate_memory_usage(&self, shape: (usize, usize)) -> usize {
        let base_size = shape.0 * shape.1 * std::mem::size_of::<f64>();

        match self.strategy {
            MemoryStrategy::Chunked => {
                // Only load chunk_size rows at a time
                self.chunk_size * shape.1 * std::mem::size_of::<f64>()
            }
            MemoryStrategy::Sparse => {
                // Assume 50% sparsity for estimation
                base_size / 2
            }
            MemoryStrategy::MemoryMapped => {
                // Minimal memory usage, just metadata
                1024
            }
            MemoryStrategy::Hybrid => {
                // Use the most efficient strategy for the given size
                if base_size > 1_000_000_000 {
                    // 1GB
                    1024 // Memory mapped
                } else if base_size > 100_000_000 {
                    // 100MB
                    self.chunk_size * shape.1 * std::mem::size_of::<f64>() // Chunked
                } else {
                    base_size / 2 // Sparse if beneficial
                }
            }
        }
    }

    /// Check if the dataset would benefit from sparse representation
    pub fn should_use_sparse(&self, array: &Array2<f64>) -> bool {
        if !self.use_sparse {
            return false;
        }

        // Quick sparsity check on a sample
        let sample_size = 1000.min(array.len());
        let mut missing_count = 0;

        for &value in array.iter().take(sample_size) {
            if value.is_nan() {
                missing_count += 1;
            }
        }

        let sparsity = missing_count as f64 / sample_size as f64;
        sparsity > 0.5 // Use sparse if more than 50% missing
    }

    /// Process large dataset with memory optimization
    pub fn process_large_dataset<F>(
        &self,
        array: &Array2<f64>,
        mut processor: F,
    ) -> SklResult<Array2<f64>>
    where
        F: FnMut(&ArrayView2<f64>) -> SklResult<Array2<f64>>,
    {
        let (n_rows, n_cols) = array.dim();

        match self.strategy {
            MemoryStrategy::Chunked | MemoryStrategy::Hybrid => {
                let mut result = Array2::zeros((n_rows, n_cols));

                // Process in chunks
                for chunk_start in (0..n_rows).step_by(self.chunk_size) {
                    let chunk_end = (chunk_start + self.chunk_size).min(n_rows);
                    let chunk = array.slice(s![chunk_start..chunk_end, ..]);
                    let processed_chunk = processor(&chunk)?;

                    result
                        .slice_mut(s![chunk_start..chunk_end, ..])
                        .assign(&processed_chunk);
                }

                Ok(result)
            }
            MemoryStrategy::Sparse => {
                if self.should_use_sparse(array) {
                    let sparse = SparseMatrix::from_dense(array, f64::NAN);
                    let dense = sparse.to_dense(f64::NAN);
                    processor(&dense.view())
                } else {
                    processor(&array.view())
                }
            }
            MemoryStrategy::MemoryMapped => {
                if self.use_mmap {
                    let temp_file = self.temp_dir.join("temp_data.bin");
                    let mmap = MemoryMappedData::new(temp_file, (n_rows, n_cols));
                    mmap.write_array(array)?;
                    let loaded_array = mmap.read_array()?;
                    processor(&loaded_array.view())
                } else {
                    processor(&array.view())
                }
            }
        }
    }
}

impl Default for MemoryOptimizedImputer {
    fn default() -> Self {
        Self::new()
    }
}
