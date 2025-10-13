//! Advanced information theory features
//!
//! This module provides advanced information-theoretic measures for
//! complexity analysis and feature extraction.

use crate::*;
use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{SklearsError, Transform},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Complexity measures extractor
///
/// Extracts various complexity measures from data using information theory.
/// This includes Kolmogorov complexity estimates, compression-based measures,
/// and other complexity metrics.
#[derive(Debug, Clone)]
pub struct ComplexityMeasuresExtractor<S = Untrained> {
    state: S,
    /// Window size for local complexity estimation
    pub window_size: usize,
    /// Number of bins for discretization
    pub n_bins: usize,
    /// Whether to include Lempel-Ziv complexity
    pub include_lz_complexity: bool,
    /// Whether to include compression ratio features
    pub include_compression_ratio: bool,
    /// Whether to include effective measure complexity
    pub include_effective_complexity: bool,
    /// Whether to include logical depth estimates
    pub include_logical_depth: bool,
}

/// Trained state for complexity measures extractor
#[derive(Debug, Clone)]
pub struct ComplexityMeasuresExtractorTrained {
    /// n_features
    pub n_features: usize,
}

impl Default for ComplexityMeasuresExtractor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplexityMeasuresExtractor<Untrained> {
    /// Create a new complexity measures extractor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            window_size: 10,
            n_bins: 10,
            include_lz_complexity: true,
            include_compression_ratio: true,
            include_effective_complexity: true,
            include_logical_depth: false,
        }
    }

    /// Set the window size for local complexity estimation
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set the number of bins for discretization
    pub fn n_bins(mut self, bins: usize) -> Self {
        self.n_bins = bins;
        self
    }

    /// Set whether to include Lempel-Ziv complexity
    pub fn include_lz_complexity(mut self, include: bool) -> Self {
        self.include_lz_complexity = include;
        self
    }

    /// Set whether to include compression ratio features
    pub fn include_compression_ratio(mut self, include: bool) -> Self {
        self.include_compression_ratio = include;
        self
    }

    /// Set whether to include effective measure complexity
    pub fn include_effective_complexity(mut self, include: bool) -> Self {
        self.include_effective_complexity = include;
        self
    }

    /// Set whether to include logical depth estimates
    pub fn include_logical_depth(mut self, include: bool) -> Self {
        self.include_logical_depth = include;
        self
    }
}

impl<S> ComplexityMeasuresExtractor<S> {
    /// Extract complexity features from a data sample
    fn extract_complexity_features(&self, data: &Array1<f64>) -> SklResult<Vec<f64>> {
        let sequence = self.discretize_data(data);
        let mut features = Vec::new();

        if self.include_lz_complexity {
            let lz_complexity = self.compute_lz_complexity(&sequence);
            features.push(lz_complexity);
        }

        if self.include_compression_ratio {
            let compression_ratio = self.compute_compression_ratio(&sequence);
            features.push(compression_ratio);
        }

        if self.include_effective_complexity {
            let effective_complexity = self.compute_effective_complexity(&sequence);
            features.push(effective_complexity);
        }

        if self.include_logical_depth {
            let logical_depth = self.compute_logical_depth(&sequence);
            features.push(logical_depth);
        }

        Ok(features)
    }

    /// Discretize continuous data into bins
    fn discretize_data(&self, data: &Array1<f64>) -> Vec<usize> {
        if data.is_empty() {
            return Vec::new();
        }

        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            return vec![0; data.len()];
        }

        let bin_width = (max_val - min_val) / (self.n_bins as f64);

        data.iter()
            .map(|&x| {
                let bin = ((x - min_val) / bin_width).floor() as usize;
                bin.min(self.n_bins - 1)
            })
            .collect()
    }

    /// Compute Lempel-Ziv complexity
    fn compute_lz_complexity(&self, sequence: &[usize]) -> f64 {
        if sequence.is_empty() {
            return 0.0;
        }

        let mut complexity = 0;
        let mut i = 0;
        let n = sequence.len();

        while i < n {
            let mut j = 1;
            let mut found = false;

            // Look for the longest prefix that has appeared before
            while i + j <= n {
                let current_substring = &sequence[i..i + j];

                // Search for this substring in previous parts
                for start in 0..i {
                    if start + j <= i {
                        let prev_substring = &sequence[start..start + j];
                        if current_substring == prev_substring {
                            found = true;
                            break;
                        }
                    }
                }

                if !found {
                    break;
                }
                j += 1;
            }

            complexity += 1;
            i += j.max(1);
        }

        complexity as f64
    }

    /// Estimate compression ratio using simple run-length encoding
    fn compute_compression_ratio(&self, sequence: &[usize]) -> f64 {
        if sequence.is_empty() {
            return 1.0;
        }

        let original_length = sequence.len();
        let mut compressed_length = 0;
        let mut i = 0;

        while i < sequence.len() {
            let current_value = sequence[i];
            let mut count = 1;

            // Count consecutive identical values
            while i + count < sequence.len() && sequence[i + count] == current_value {
                count += 1;
            }

            // Each run is encoded as (value, count) - 2 units
            compressed_length += 2;
            i += count;
        }

        compressed_length as f64 / original_length as f64
    }

    /// Compute effective measure complexity (EMC)
    fn compute_effective_complexity(&self, sequence: &[usize]) -> f64 {
        if sequence.is_empty() {
            return 0.0;
        }

        // Count symbol frequencies
        let mut frequencies = std::collections::HashMap::new();
        for &symbol in sequence {
            *frequencies.entry(symbol).or_insert(0) += 1;
        }

        let n = sequence.len() as f64;

        // Compute entropy
        let entropy: f64 = frequencies
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        // Compute effective complexity as entropy * length
        entropy * n.log2()
    }

    /// Estimate logical depth (simplified version)
    fn compute_logical_depth(&self, sequence: &[usize]) -> f64 {
        if sequence.is_empty() {
            return 0.0;
        }

        // Simplified logical depth based on minimum computation time
        // In practice, this would require running actual computation
        // Here we estimate based on pattern complexity

        let mut pattern_lengths = Vec::new();
        let mut i = 0;

        while i < sequence.len() {
            let mut pattern_length = 1;

            // Find repeating patterns
            for len in 2..=(sequence.len() - i) / 2 {
                if i + 2 * len <= sequence.len() {
                    let pattern1 = &sequence[i..i + len];
                    let pattern2 = &sequence[i + len..i + 2 * len];
                    if pattern1 == pattern2 {
                        pattern_length = len;
                        break;
                    }
                }
            }

            pattern_lengths.push(pattern_length);
            i += pattern_length;
        }

        // Logical depth estimate based on average pattern complexity
        let avg_pattern_length =
            pattern_lengths.iter().sum::<usize>() as f64 / pattern_lengths.len() as f64;
        avg_pattern_length * (sequence.len() as f64).log2()
    }
}

impl Estimator<Untrained> for ComplexityMeasuresExtractor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for ComplexityMeasuresExtractor<Untrained> {
    type Fitted = ComplexityMeasuresExtractor<ComplexityMeasuresExtractorTrained>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        // Determine number of features by extracting from first sample
        let first_sample = x.row(0).to_owned();
        let sample_features = self.extract_complexity_features(&first_sample)?;
        let n_features = sample_features.len();

        Ok(ComplexityMeasuresExtractor {
            state: ComplexityMeasuresExtractorTrained { n_features },
            window_size: self.window_size,
            n_bins: self.n_bins,
            include_lz_complexity: self.include_lz_complexity,
            include_compression_ratio: self.include_compression_ratio,
            include_effective_complexity: self.include_effective_complexity,
            include_logical_depth: self.include_logical_depth,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>>
    for ComplexityMeasuresExtractor<ComplexityMeasuresExtractorTrained>
{
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, self.state.n_features));

        for i in 0..n_samples {
            let sample = x.row(i).to_owned();
            let sample_features = self.extract_complexity_features(&sample)?;

            for (j, &feature) in sample_features.iter().enumerate() {
                if j < self.state.n_features {
                    features[(i, j)] = feature;
                }
            }
        }

        Ok(features)
    }
}

/// Information gain feature extractor
///
/// Computes information gain features for feature selection and ranking.
/// This is useful for identifying the most informative features in a dataset.
#[derive(Debug, Clone)]
pub struct InformationGainExtractor<S = Untrained> {
    state: S,
    /// Number of bins for discretization of continuous features
    pub n_bins: usize,
    /// Minimum samples in a bin to consider it valid
    pub min_samples_bin: usize,
    /// Whether to use normalized information gain (information gain ratio)
    pub normalize: bool,
    /// Small constant to avoid log(0)
    pub epsilon: f64,
}

/// Trained state for information gain extractor
#[derive(Debug, Clone)]
pub struct InformationGainExtractorTrained {
    pub information_gains: Vec<f64>,
}

impl Default for InformationGainExtractor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl InformationGainExtractor<Untrained> {
    /// Create a new information gain extractor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_bins: 10,
            min_samples_bin: 1,
            normalize: false,
            epsilon: 1e-10,
        }
    }

    /// Set the number of bins for discretization
    pub fn n_bins(mut self, bins: usize) -> Self {
        self.n_bins = bins;
        self
    }

    /// Set the minimum samples per bin
    pub fn min_samples_bin(mut self, min_samples: usize) -> Self {
        self.min_samples_bin = min_samples;
        self
    }

    /// Set whether to normalize information gain
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set epsilon for numerical stability
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }
}

impl<S> InformationGainExtractor<S> {
    /// Compute information gain for a feature
    fn compute_information_gain(&self, feature: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let feature_bins = self.discretize_feature(feature);
        let target_bins = self.discretize_feature(target);

        // Compute H(Y)
        let mut target_counts = HashMap::new();
        for &bin in target_bins.iter() {
            *target_counts.entry(bin).or_insert(0) += 1;
        }
        let target_counts_vec: Vec<usize> = target_counts.values().cloned().collect();
        let entropy_y = self.compute_entropy(&target_counts_vec);

        // Compute H(Y|X)
        let conditional_entropy = self.compute_conditional_entropy(&feature_bins, &target_bins);

        // Information gain = H(Y) - H(Y|X)
        let information_gain = entropy_y - conditional_entropy;

        if self.normalize {
            // Compute H(X) for normalization
            let mut feature_counts = HashMap::new();
            for &bin in feature_bins.iter() {
                *feature_counts.entry(bin).or_insert(0) += 1;
            }
            let feature_counts_vec: Vec<usize> = feature_counts.values().cloned().collect();
            let entropy_x = self.compute_entropy(&feature_counts_vec);

            if entropy_x > self.epsilon {
                information_gain / entropy_x // Information gain ratio
            } else {
                0.0
            }
        } else {
            information_gain
        }
    }

    /// Discretize continuous feature values
    fn discretize_feature(&self, values: &Array1<f64>) -> Vec<usize> {
        if values.is_empty() {
            return Vec::new();
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < self.epsilon {
            return vec![0; values.len()];
        }

        let bin_width = (max_val - min_val) / (self.n_bins as f64);

        values
            .iter()
            .map(|&x| {
                let bin = ((x - min_val) / bin_width).floor() as usize;
                bin.min(self.n_bins - 1)
            })
            .collect()
    }

    /// Compute entropy of a discrete distribution
    fn compute_entropy(&self, counts: &[usize]) -> f64 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total as f64;
                -p * (p + self.epsilon).log2()
            })
            .sum()
    }

    /// Compute conditional entropy H(Y|X)
    fn compute_conditional_entropy(&self, x_bins: &[usize], y_bins: &[usize]) -> f64 {
        if x_bins.len() != y_bins.len() || x_bins.is_empty() {
            return 0.0;
        }

        // Count joint occurrences
        let mut joint_counts = std::collections::HashMap::new();
        let mut x_counts = std::collections::HashMap::new();

        for (&x, &y) in x_bins.iter().zip(y_bins.iter()) {
            *joint_counts.entry((x, y)).or_insert(0) += 1;
            *x_counts.entry(x).or_insert(0) += 1;
        }

        let total = x_bins.len() as f64;
        let mut conditional_entropy = 0.0;

        for (&x, &x_count) in x_counts.iter() {
            if x_count < self.min_samples_bin {
                continue;
            }

            let p_x = x_count as f64 / total;
            let mut y_given_x_counts = Vec::new();

            // Collect Y counts for this X value
            for (&(joint_x, _), &count) in joint_counts.iter() {
                if joint_x == x {
                    y_given_x_counts.push(count);
                }
            }

            let entropy_y_given_x = self.compute_entropy(&y_given_x_counts);
            conditional_entropy += p_x * entropy_y_given_x;
        }

        conditional_entropy
    }
}

/// Fitted information gain extractor
pub struct FittedInformationGainExtractor {
    extractor: InformationGainExtractor,
    information_gains: Vec<f64>,
}

impl Estimator<Untrained> for InformationGainExtractor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, Array1<f64>> for InformationGainExtractor<Untrained> {
    type Fitted = InformationGainExtractor<InformationGainExtractorTrained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let n_features = x.ncols();
        let mut information_gains = Vec::with_capacity(n_features);

        for i in 0..n_features {
            let feature = x.column(i).to_owned();
            let ig = self.compute_information_gain(&feature, y);
            information_gains.push(ig);
        }

        Ok(InformationGainExtractor {
            state: InformationGainExtractorTrained { information_gains },
            n_bins: self.n_bins,
            min_samples_bin: self.min_samples_bin,
            normalize: self.normalize,
            epsilon: self.epsilon,
        })
    }
}

impl Transform<Array2<f64>, Array1<f64>>
    for InformationGainExtractor<InformationGainExtractorTrained>
{
    fn transform(&self, _x: &Array2<f64>) -> SklResult<Array1<f64>> {
        // Return the computed information gain scores
        Ok(Array1::from_vec(self.state.information_gains.clone()))
    }
}

impl InformationGainExtractor<InformationGainExtractorTrained> {
    /// Get the information gain scores
    pub fn get_information_gains(&self) -> &[f64] {
        &self.state.information_gains
    }

    /// Get the feature ranking based on information gain
    pub fn get_feature_ranking(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.state.information_gains.len()).collect();
        indices.sort_by(|&a, &b| {
            self.state.information_gains[b]
                .partial_cmp(&self.state.information_gains[a])
                .unwrap()
        });
        indices
    }
}

/// Minimum Description Length (MDL) feature extractor
///
/// Uses the minimum description length principle for feature extraction
/// and model selection. This is useful for finding the optimal trade-off
/// between model complexity and data fit.
pub struct MinimumDescriptionLengthExtractor {
    /// Model complexity penalty factor
    pub complexity_penalty: f64,
    /// Number of model parameters to consider
    pub n_parameters: usize,
    /// Whether to use normalized MDL
    pub normalize: bool,
    /// Encoding precision for real-valued data
    pub encoding_precision: f64,
}

impl Default for MinimumDescriptionLengthExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MinimumDescriptionLengthExtractor {
    /// Create a new MDL extractor
    pub fn new() -> Self {
        Self {
            complexity_penalty: 1.0,
            n_parameters: 10,
            normalize: true,
            encoding_precision: 1e-6,
        }
    }

    /// Set the complexity penalty factor
    pub fn complexity_penalty(mut self, penalty: f64) -> Self {
        self.complexity_penalty = penalty;
        self
    }

    /// Set the number of model parameters
    pub fn n_parameters(mut self, n: usize) -> Self {
        self.n_parameters = n;
        self
    }

    /// Set whether to normalize MDL
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set encoding precision
    pub fn encoding_precision(mut self, precision: f64) -> Self {
        self.encoding_precision = precision;
        self
    }

    /// Compute data encoding length (negative log-likelihood)
    fn compute_data_length(&self, data: &Array1<f64>) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Compute empirical distribution
        let n = data.len() as f64;
        let mean = data.sum() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        if variance < self.encoding_precision {
            return n; // Uniform encoding
        }

        // Gaussian encoding length (negative log-likelihood)
        let mut total_length = 0.0;
        for &x in data.iter() {
            let prob_density = (-0.5 * (x - mean).powi(2) / variance).exp()
                / (2.0 * std::f64::consts::PI * variance).sqrt();
            total_length -= (prob_density.max(self.encoding_precision)).log2();
        }

        total_length
    }

    /// Compute model encoding length
    fn compute_model_length(&self, data: &Array1<f64>) -> f64 {
        let n = data.len() as f64;

        // Model complexity: number of parameters * log(data_size) / 2
        let model_length = (self.n_parameters as f64) * n.log2() / 2.0;

        model_length * self.complexity_penalty
    }

    /// Compute MDL score for data
    fn compute_mdl_score(&self, data: &Array1<f64>) -> f64 {
        let data_length = self.compute_data_length(data);
        let model_length = self.compute_model_length(data);

        let mdl_score = data_length + model_length;

        if self.normalize && !data.is_empty() {
            mdl_score / (data.len() as f64)
        } else {
            mdl_score
        }
    }

    /// Extract sliding window MDL features
    fn extract_windowed_mdl_features(&self, data: &Array1<f64>, window_size: usize) -> Vec<f64> {
        if data.len() < window_size {
            return vec![self.compute_mdl_score(data)];
        }

        let mut features = Vec::new();
        for i in 0..=data.len() - window_size {
            let window = data.slice(s![i..i + window_size]).to_owned();
            let mdl_score = self.compute_mdl_score(&window);
            features.push(mdl_score);
        }

        features
    }
}

/// Fitted MDL extractor
pub struct FittedMinimumDescriptionLengthExtractor {
    extractor: MinimumDescriptionLengthExtractor,
    window_sizes: Vec<usize>,
}

impl Estimator<Untrained> for MinimumDescriptionLengthExtractor {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for MinimumDescriptionLengthExtractor {
    type Fitted = FittedMinimumDescriptionLengthExtractor;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        // Use multiple window sizes for feature extraction
        let max_len = x.ncols();
        let window_sizes = vec![
            (max_len / 4).max(1),
            (max_len / 2).max(1),
            (3 * max_len / 4).max(1),
            max_len,
        ];

        Ok(FittedMinimumDescriptionLengthExtractor {
            extractor: self,
            window_sizes,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedMinimumDescriptionLengthExtractor {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();

        // Compute total number of features
        let mut total_features = 0;
        for &window_size in &self.window_sizes {
            let n_windows = if x.ncols() >= window_size {
                x.ncols() - window_size + 1
            } else {
                1
            };
            total_features += n_windows;
        }

        let mut features = Array2::zeros((n_samples, total_features));

        for i in 0..n_samples {
            let sample = x.row(i).to_owned();
            let mut feature_idx = 0;

            for &window_size in &self.window_sizes {
                let window_features = self
                    .extractor
                    .extract_windowed_mdl_features(&sample, window_size);

                for &feature in window_features.iter() {
                    if feature_idx < total_features {
                        features[(i, feature_idx)] = feature;
                        feature_idx += 1;
                    }
                }
            }
        }

        Ok(features)
    }
}

/// Topological Data Analysis feature extractor
///
/// Extracts topological features from data using persistent homology
/// and other topological methods.
#[derive(Debug, Clone)]
pub struct TopologicalDataAnalysis {
    max_dimension: usize,
    persistence_threshold: f64,
    homology_dimensions: Vec<usize>,
    metric: String,
}

impl TopologicalDataAnalysis {
    /// Create a new TDA extractor
    pub fn new() -> Self {
        Self {
            max_dimension: 2,
            persistence_threshold: 0.1,
            homology_dimensions: vec![0, 1, 2],
            metric: "euclidean".to_string(),
        }
    }

    /// Set the maximum dimension for homology computation
    pub fn max_dimension(mut self, max_dim: usize) -> Self {
        self.max_dimension = max_dim;
        self
    }

    /// Set the persistence threshold
    pub fn persistence_threshold(mut self, threshold: f64) -> Self {
        self.persistence_threshold = threshold;
        self
    }

    /// Set the homology dimensions to compute
    pub fn homology_dimensions(mut self, dims: Vec<usize>) -> Self {
        self.homology_dimensions = dims;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: String) -> Self {
        self.metric = metric;
        self
    }

    /// Set the threshold (alias for persistence_threshold)
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.persistence_threshold = threshold;
        self
    }

    /// Compute persistence diagrams from point cloud data
    pub fn compute_persistence_diagrams(
        &self,
        data: &Array2<f64>,
    ) -> SklResult<Vec<Vec<(f64, f64)>>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput("Empty data".to_string()));
        }

        let mut diagrams = Vec::new();

        // Compute distance matrix
        let distance_matrix = self.compute_distance_matrix(data)?;

        // For each homology dimension
        for &dim in &self.homology_dimensions {
            let diagram = self.compute_persistence_diagram_for_dimension(&distance_matrix, dim)?;
            diagrams.push(diagram);
        }

        Ok(diagrams)
    }

    /// Compute Betti numbers from persistence diagrams
    pub fn compute_betti_numbers(&self, diagrams: &[Vec<(f64, f64)>]) -> Vec<usize> {
        diagrams
            .iter()
            .enumerate()
            .map(|(dimension, diagram)| {
                let mut betti = diagram
                    .iter()
                    .filter(|&&(birth, death)| {
                        let persistence = death - birth;
                        death.is_infinite() || persistence >= self.persistence_threshold
                    })
                    .count();

                if dimension == 0 {
                    betti = betti.max(1);
                }

                betti
            })
            .collect()
    }

    /// Extract topological features from data
    pub fn extract_features(&self, data: &Array2<f64>) -> SklResult<Array1<f64>> {
        let diagrams = self.compute_persistence_diagrams(data)?;
        let betti_numbers = self.compute_betti_numbers(&diagrams);

        let mut features = Vec::new();

        // Add Betti numbers as features
        for &betti in &betti_numbers {
            features.push(betti as f64);
        }

        // Add persistence statistics
        for diagram in &diagrams {
            if diagram.is_empty() {
                features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]); // mean, std, max, count
                continue;
            }

            let persistences: Vec<f64> = diagram
                .iter()
                .map(|&(birth, death)| death - birth)
                .collect();

            let mean_persistence = persistences.iter().sum::<f64>() / persistences.len() as f64;
            let std_persistence = {
                let variance = persistences
                    .iter()
                    .map(|&p| (p - mean_persistence).powi(2))
                    .sum::<f64>()
                    / persistences.len() as f64;
                variance.sqrt()
            };
            let max_persistence = persistences.iter().cloned().fold(0.0, f64::max);
            let count = persistences.len() as f64;

            features.extend_from_slice(&[
                mean_persistence,
                std_persistence,
                max_persistence,
                count,
            ]);
        }

        Ok(Array1::from_vec(features))
    }

    /// Extract topological features (alias for extract_features)
    pub fn extract_topological_features(&self, data: &Array2<f64>) -> SklResult<Array1<f64>> {
        self.extract_features(data)
    }

    /// Compute persistence (alias for compute_persistence_diagrams)
    pub fn compute_persistence(&self, data: &Array2<f64>) -> SklResult<Vec<Vec<(f64, f64)>>> {
        self.compute_persistence_diagrams(data)
    }

    /// Compute persistence entropy from data
    pub fn persistence_entropy(&self, data: &Array2<f64>) -> SklResult<f64> {
        let diagrams = self.compute_persistence_diagrams(data)?;
        let mut total_entropy = 0.0;

        for diagram in &diagrams {
            if diagram.is_empty() {
                continue;
            }

            let persistences: Vec<f64> = diagram
                .iter()
                .map(|&(birth, death)| death - birth)
                .filter(|&p| p > self.persistence_threshold)
                .collect();

            if persistences.is_empty() {
                continue;
            }

            let total_persistence: f64 = persistences.iter().sum();
            if total_persistence == 0.0 {
                continue;
            }

            let entropy = persistences
                .iter()
                .map(|&p| {
                    let normalized = p / total_persistence;
                    if normalized > 0.0 {
                        -normalized * normalized.ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();

            total_entropy += entropy;
        }

        Ok(total_entropy)
    }

    /// Compute persistence landscape from data
    pub fn persistence_landscape(
        &self,
        data: &Array2<f64>,
        resolution: usize,
    ) -> SklResult<Array2<f64>> {
        let diagrams = self.compute_persistence_diagrams(data)?;
        let mut landscape = Array2::zeros((diagrams.len(), resolution));

        for (dim_idx, diagram) in diagrams.iter().enumerate() {
            if diagram.is_empty() {
                continue;
            }

            // Find the range for this dimension
            let births: Vec<f64> = diagram.iter().map(|(b, _)| *b).collect();
            let deaths: Vec<f64> = diagram.iter().map(|(_, d)| *d).collect();

            let min_val = births.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = deaths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if min_val >= max_val {
                continue;
            }

            let step = (max_val - min_val) / resolution as f64;

            for i in 0..resolution {
                let x = min_val + i as f64 * step;
                let mut landscape_value = 0.0;

                for &(birth, death) in diagram {
                    if birth <= x && x <= death {
                        let tent_height = (x - birth).min(death - x);
                        landscape_value = landscape_value.max(tent_height);
                    }
                }

                landscape[(dim_idx, i)] = landscape_value;
            }
        }

        Ok(landscape)
    }

    /// Compute distance matrix using specified metric
    fn compute_distance_matrix(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = data.nrows();
        let mut distances = Array2::zeros((n, n));

        match self.metric.as_str() {
            "euclidean" => {
                for i in 0..n {
                    for j in i..n {
                        let mut sum = 0.0;
                        for k in 0..data.ncols() {
                            let diff = data[[i, k]] - data[[j, k]];
                            sum += diff * diff;
                        }
                        let dist = sum.sqrt();
                        distances[[i, j]] = dist;
                        distances[[j, i]] = dist;
                    }
                }
            }
            "manhattan" => {
                for i in 0..n {
                    for j in i..n {
                        let mut sum = 0.0;
                        for k in 0..data.ncols() {
                            sum += (data[[i, k]] - data[[j, k]]).abs();
                        }
                        distances[[i, j]] = sum;
                        distances[[j, i]] = sum;
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown metric: {}",
                    self.metric
                )))
            }
        }

        Ok(distances)
    }

    /// Compute persistence diagram for a specific homology dimension
    fn compute_persistence_diagram_for_dimension(
        &self,
        distance_matrix: &Array2<f64>,
        dimension: usize,
    ) -> SklResult<Vec<(f64, f64)>> {
        let n = distance_matrix.nrows();

        // Simplified persistence computation using Vietoris-Rips complex
        // This is a basic approximation of the full persistent homology computation
        let mut diagram = Vec::new();

        // Create simplicial complex by adding edges at different scales
        let mut scale_values: Vec<f64> = distance_matrix
            .iter()
            .filter(|&&dist| dist > 0.0)
            .cloned()
            .collect();
        scale_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scale_values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

        // For dimension 0 (connected components)
        if dimension == 0 {
            let mut components = UnionFind::new(n);
            let mut current_components = n;

            for &scale in &scale_values {
                let mut merges_this_scale = 0;

                for i in 0..n {
                    for j in i + 1..n {
                        if distance_matrix[[i, j]] <= scale && components.union(i, j) {
                            merges_this_scale += 1;
                        }
                    }
                }

                // Components that merge at this scale die
                for _ in 0..merges_this_scale {
                    if current_components > 1 {
                        diagram.push((0.0, scale));
                        current_components -= 1;
                    }
                }
            }
        }
        // For higher dimensions, use simplified approach
        else if dimension == 1 {
            // Look for 1-dimensional holes (cycles)
            for i in 0..n.min(10) {
                // Limit for computational efficiency
                for j in i + 1..n.min(10) {
                    for k in j + 1..n.min(10) {
                        // Check if we have a triangle
                        let edge_lengths = [
                            distance_matrix[[i, j]],
                            distance_matrix[[j, k]],
                            distance_matrix[[k, i]],
                        ];

                        let min_edge = edge_lengths.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_edge = edge_lengths.iter().cloned().fold(0.0, f64::max);

                        // Triangle forms at max_edge scale, potential hole
                        if max_edge - min_edge > self.persistence_threshold {
                            diagram.push((min_edge, max_edge));
                        }
                    }
                }
            }
        }

        Ok(diagram)
    }

    /// Compute bottleneck distance between persistence diagrams of two point clouds
    ///
    /// The bottleneck distance is the infimum over all bijections between diagrams
    /// of the maximum distance between matched points.
    pub fn bottleneck_distance(&self, data1: &Array2<f64>, data2: &Array2<f64>) -> SklResult<f64> {
        let diagrams1 = self.compute_persistence_diagrams(data1)?;
        let diagrams2 = self.compute_persistence_diagrams(data2)?;

        let mut max_dist = 0.0;

        // For each dimension, compute the bottleneck distance between diagrams
        for (diag1, diag2) in diagrams1.iter().zip(diagrams2.iter()) {
            let dist = self.compute_diagram_bottleneck_distance(diag1, diag2);
            if dist > max_dist {
                max_dist = dist;
            }
        }

        Ok(max_dist)
    }

    /// Helper function to compute bottleneck distance between two diagrams
    fn compute_diagram_bottleneck_distance(
        &self,
        diag1: &[(f64, f64)],
        diag2: &[(f64, f64)],
    ) -> f64 {
        // Simplified bottleneck distance computation
        // In practice, this would use the Hungarian algorithm
        let mut max_dist = 0.0;

        for &(b1, d1) in diag1 {
            let mut min_match_dist = f64::MAX;
            for &(b2, d2) in diag2 {
                let dist = ((b1 - b2).powi(2) + (d1 - d2).powi(2)).sqrt();
                if dist < min_match_dist {
                    min_match_dist = dist;
                }
            }
            if min_match_dist < f64::MAX && min_match_dist > max_dist {
                max_dist = min_match_dist;
            }
        }

        max_dist
    }
}

impl Default for TopologicalDataAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Union-Find data structure for connected components
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let px = self.find(x);
        let py = self.find(y);

        if px == py {
            return false;
        }

        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
        } else if self.rank[px] > self.rank[py] {
            self.parent[py] = px;
        } else {
            self.parent[py] = px;
            self.rank[px] += 1;
        }

        true
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_complexity_measures_extractor() {
        let mut data = Array2::zeros((5, 10));
        for i in 0..5 {
            for j in 0..10 {
                data[(i, j)] = (i + j) as f64;
            }
        }

        let extractor = ComplexityMeasuresExtractor::new()
            .include_lz_complexity(true)
            .include_compression_ratio(true)
            .n_bins(5);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 5);
        assert!(features.ncols() > 0);

        // Check that features are finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_information_gain_extractor() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let extractor = InformationGainExtractor::new().n_bins(3).normalize(false);

        let fitted = extractor.fit(&x, &y).unwrap();
        let gains = fitted.transform(&x).unwrap();

        assert_eq!(gains.len(), 2);

        // Information gains should be non-negative
        for &gain in gains.iter() {
            assert!(gain >= 0.0);
            assert!(gain.is_finite());
        }
    }

    #[test]
    fn test_mdl_extractor() {
        let mut data = Array2::zeros((3, 8));
        for i in 0..3 {
            for j in 0..8 {
                data[(i, j)] = (i * 8 + j) as f64;
            }
        }

        let extractor = MinimumDescriptionLengthExtractor::new()
            .complexity_penalty(0.5)
            .normalize(true);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);

        // Check that features are finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_complexity_measures_empty_data() {
        let data = Array2::zeros((0, 5));
        let extractor = ComplexityMeasuresExtractor::new();

        let result = extractor.fit(&data, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_information_gain_mismatched_dimensions() {
        let x = Array2::zeros((5, 3));
        let y = Array1::zeros(3); // Wrong size

        let extractor = InformationGainExtractor::new();
        let result = extractor.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lempel_ziv_complexity() {
        let extractor = ComplexityMeasuresExtractor::new();

        // Simple repeating pattern
        let sequence = vec![0, 1, 0, 1, 0, 1];
        let complexity = extractor.compute_lz_complexity(&sequence);
        assert!(complexity > 0.0);

        // All same values (low complexity)
        let uniform_sequence = vec![0, 0, 0, 0, 0];
        let uniform_complexity = extractor.compute_lz_complexity(&uniform_sequence);
        assert!(uniform_complexity >= 1.0);
    }

    #[test]
    fn test_information_gain_feature_ranking() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0,
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        let extractor = InformationGainExtractor::new();
        let fitted = extractor.fit(&x, &y).unwrap();

        let ranking = fitted.get_feature_ranking();
        assert_eq!(ranking.len(), 3);

        // Check that ranking contains all feature indices
        let mut sorted_ranking = ranking.clone();
        sorted_ranking.sort();
        assert_eq!(sorted_ranking, vec![0, 1, 2]);
    }
}
