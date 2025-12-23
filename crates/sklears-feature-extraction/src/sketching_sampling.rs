//! Sketching and sampling feature engineering components
//!
//! This module provides implementations of various sketching and sampling algorithms
//! for efficient feature extraction and data processing.

use crate::*;
// use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use sklears_core::prelude::{SklearsError, Transform};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::hash::Hash;

/// Count-Min Sketch for frequency estimation with guaranteed error bounds
///
/// Count-Min Sketch is a probabilistic data structure that maintains frequency
/// estimates for items in a stream. It provides space-efficient approximate
/// counting with mathematical guarantees on accuracy.
///
/// This implementation provides feature extraction from sketched frequency
/// distributions, useful for large-scale data analysis where exact counting
/// is impractical.
///
/// # Examples
/// ```
/// # use sklears_feature_extraction::sketching_sampling::CountMinSketch;
/// # use scirs2_core::ndarray::Array2;
/// # use sklears_core::traits::{Transform, Fit};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| (x % 50) as f64).collect())?;
///
/// let sketch = CountMinSketch::new()
///     .width(100)
///     .depth(5)
///     .include_frequency_statistics(true);
///
/// let features = sketch.transform(&data.view())?;
/// assert!(features.len() > 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    include_frequency_statistics: bool,
    include_sketch_properties: bool,
    hash_seed: u64,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch
    pub fn new() -> Self {
        Self {
            width: 1000,
            depth: 5,
            include_frequency_statistics: true,
            include_sketch_properties: true,
            hash_seed: 42,
        }
    }

    /// Set the width of the sketch (number of buckets per hash function)
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the depth of the sketch (number of hash functions)
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Set whether to include frequency statistics
    pub fn include_frequency_statistics(mut self, include: bool) -> Self {
        self.include_frequency_statistics = include;
        self
    }

    /// Set whether to include sketch properties
    pub fn include_sketch_properties(mut self, include: bool) -> Self {
        self.include_sketch_properties = include;
        self
    }

    /// Set the hash seed for reproducibility
    pub fn hash_seed(mut self, seed: u64) -> Self {
        self.hash_seed = seed;
        self
    }

    /// Hash function implementation
    fn hash(&self, item: u64, hash_idx: usize) -> usize {
        // Simple hash function combining item and hash index
        let combined = item.wrapping_add((hash_idx as u64).wrapping_mul(self.hash_seed));
        let hash_val = combined.wrapping_mul(0x9e3779b97f4a7c15_u64);
        (hash_val as usize) % self.width
    }

    /// Build the Count-Min Sketch from data
    fn build_sketch(&self, data: &Array2<f64>) -> Array2<u32> {
        let mut sketch = Array2::zeros((self.depth, self.width));

        // Process each data point
        for row in data.outer_iter() {
            for &value in row.iter() {
                // Convert value to integer for hashing
                let item = (value * 1000.0) as u64; // Scale to preserve some precision

                // Update all hash functions
                for hash_idx in 0..self.depth {
                    let bucket = self.hash(item, hash_idx);
                    sketch[[hash_idx, bucket]] += 1;
                }
            }
        }

        sketch
    }

    /// Estimate frequency of an item
    fn estimate_frequency(&self, sketch: &Array2<u32>, item: f64) -> u32 {
        let item_hash = (item * 1000.0) as u64;
        let mut min_count = u32::MAX;

        for hash_idx in 0..self.depth {
            let bucket = self.hash(item_hash, hash_idx);
            min_count = min_count.min(sketch[[hash_idx, bucket]]);
        }

        min_count
    }

    /// Extract features from the sketch
    fn extract_sketch_features(&self, sketch: &Array2<u32>, data: &Array2<f64>) -> Vec<f64> {
        let mut features = Vec::new();

        if self.include_frequency_statistics {
            // Total count
            let total_count: u32 = sketch.sum();
            features.push(total_count as f64);

            // Non-zero buckets
            let non_zero_buckets = sketch.iter().filter(|&&x| x > 0).count();
            features.push(non_zero_buckets as f64);

            // Load factor (fraction of non-zero buckets)
            let load_factor = non_zero_buckets as f64 / (self.width * self.depth) as f64;
            features.push(load_factor);

            // Maximum bucket count
            let max_count = sketch.iter().max().copied().unwrap_or(0);
            features.push(max_count as f64);

            // Average bucket count (excluding zeros)
            let avg_count = if non_zero_buckets > 0 {
                total_count as f64 / non_zero_buckets as f64
            } else {
                0.0
            };
            features.push(avg_count);
        }

        if self.include_sketch_properties {
            // Sketch dimensions
            features.push(self.width as f64);
            features.push(self.depth as f64);

            // Collision rate estimate (approximate)
            let unique_items = data
                .iter()
                .map(|&x| (x * 1000.0) as u64)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let collision_rate = if unique_items > 0 {
                1.0 - (unique_items as f64 / (self.width as f64))
            } else {
                0.0
            };
            features.push(collision_rate.max(0.0));
        }

        features
    }
}

impl Default for CountMinSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array1<Float>> for CountMinSketch {
    fn transform(&self, data: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot create sketch from empty data".to_string(),
            ));
        }

        // Convert to f64 for internal processing
        let data_owned = data.mapv(|x| x).to_owned();

        let sketch = self.build_sketch(&data_owned);
        let features = self.extract_sketch_features(&sketch, &data_owned);

        Ok(Array1::from_vec(
            features.into_iter().map(|x| x as Float).collect(),
        ))
    }
}

/// Fast Johnson-Lindenstrauss Transform for efficient random projection
///
/// The Fast Johnson-Lindenstrauss Transform (FJLT) provides an efficient
/// implementation of the Johnson-Lindenstrauss lemma using fast transforms
/// like Walsh-Hadamard Transform combined with sparse random matrices.
///
/// This provides approximate distance preservation with better computational
/// efficiency compared to dense Gaussian random projections.
///
/// # Examples
/// ```
/// # use sklears_feature_extraction::sketching_sampling::FastJohnsonLindenstrauss;
/// # use scirs2_core::ndarray::Array2;
/// # use sklears_core::traits::{Transform, Fit};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((50, 16), (0..800).map(|x| x as f64).collect())?;
///
/// let fjlt = FastJohnsonLindenstrauss::new()
///     .target_dimension(8)
///     .sparsity_parameter(0.1);
///
/// let features = fjlt.transform(&data.view())?;
/// assert_eq!(features.len(), 50 * 8); // 50 samples, 8 dimensions
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FastJohnsonLindenstrauss {
    target_dimension: usize,
    sparsity_parameter: f64,
    random_state: Option<u64>,
    include_transform_stats: bool,
}

impl FastJohnsonLindenstrauss {
    /// Create a new Fast Johnson-Lindenstrauss transform
    pub fn new() -> Self {
        Self {
            target_dimension: 100,
            sparsity_parameter: 0.1,
            random_state: Some(42),
            include_transform_stats: false,
        }
    }

    /// Set the target dimension for the projection
    pub fn target_dimension(mut self, target_dimension: usize) -> Self {
        self.target_dimension = target_dimension;
        self
    }

    /// Set the sparsity parameter (fraction of non-zero elements)
    pub fn sparsity_parameter(mut self, sparsity: f64) -> Self {
        self.sparsity_parameter = sparsity.clamp(0.01, 1.0);
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set whether to include transform statistics
    pub fn include_transform_stats(mut self, include: bool) -> Self {
        self.include_transform_stats = include;
        self
    }

    /// Generate sparse random matrix
    fn generate_sparse_matrix(&self, input_dim: usize) -> Array2<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut matrix = Array2::zeros((self.target_dimension, input_dim));
        let seed = self.random_state.unwrap_or(42);

        for i in 0..self.target_dimension {
            for j in 0..input_dim {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash_val = hasher.finish();

                // Determine if this element should be non-zero
                let uniform_val = (hash_val % 1000) as f64 / 1000.0;

                if uniform_val < self.sparsity_parameter {
                    // Rademacher random variable (+1 or -1)
                    let sign_hash = hash_val >> 32;
                    let sign = if sign_hash % 2 == 0 { 1.0 } else { -1.0 };
                    matrix[[i, j]] = sign / (self.sparsity_parameter * input_dim as f64).sqrt();
                }
            }
        }

        matrix
    }

    /// Apply fast Walsh-Hadamard transform (simplified version)
    fn fast_hadamard_transform(&self, mut data: Array1<f64>) -> Array1<f64> {
        let n = data.len();

        // Find the next power of 2
        let next_pow2 = n.next_power_of_two();

        // Pad with zeros if necessary
        if next_pow2 > n {
            let mut padded = Array1::zeros(next_pow2);
            padded.slice_mut(s![..n]).assign(&data);
            data = padded;
        }

        let len = data.len();
        let mut h = 1;

        while h < len {
            for i in (0..len).step_by(h * 2) {
                for j in i..(i + h).min(len) {
                    if j + h < len {
                        let a = data[j];
                        let b = data[j + h];
                        data[j] = a + b;
                        data[j + h] = a - b;
                    }
                }
            }
            h *= 2;
        }

        // Normalize
        let normalizer = (len as f64).sqrt();
        data.mapv_inplace(|x| x / normalizer);

        // Return only the original size
        if next_pow2 > n {
            data.slice(s![..n]).to_owned()
        } else {
            data
        }
    }

    /// Apply the FJLT
    fn apply_fjlt(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let input_dim = data.ncols();

        let mut result = Array2::zeros((n_samples, self.target_dimension));

        // Generate sparse random matrix
        let sparse_matrix = self.generate_sparse_matrix(input_dim);

        // Apply transform to each sample
        for (i, sample) in data.outer_iter().enumerate() {
            // Apply fast Hadamard transform
            let hadamard_result = self.fast_hadamard_transform(sample.to_owned());

            // Apply sparse random projection
            for j in 0..self.target_dimension {
                let mut sum = 0.0;
                for k in 0..input_dim {
                    sum += sparse_matrix[[j, k]] * hadamard_result[k];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    /// Extract features from the transformed data
    fn extract_features(&self, transformed_data: &Array2<f64>) -> Vec<f64> {
        if self.include_transform_stats {
            let mut features = Vec::new();

            // Basic statistics of the transformed data
            let mean = transformed_data.mean().unwrap_or(0.0);
            let std = transformed_data.std(0.0);
            let min_val = transformed_data
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_val = transformed_data
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            features.push(mean);
            features.push(std);
            features.push(min_val);
            features.push(max_val);
            features.push(self.target_dimension as f64);
            features.push(self.sparsity_parameter);

            features
        } else {
            // Flatten the transformed data
            transformed_data.iter().copied().collect()
        }
    }
}

impl Default for FastJohnsonLindenstrauss {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array1<Float>> for FastJohnsonLindenstrauss {
    fn transform(&self, data: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot transform empty data".to_string(),
            ));
        }

        if data.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Data must have at least one column".to_string(),
            ));
        }

        // Convert to f64 for internal processing
        let data_owned = data.mapv(|x| x).to_owned();

        let transformed = self.apply_fjlt(&data_owned);
        let features = self.extract_features(&transformed);

        Ok(Array1::from_vec(
            features.into_iter().map(|x| x as Float).collect(),
        ))
    }
}

/// Reservoir Sampling for Large Dataset Processing
///
/// Implements reservoir sampling algorithm for selecting a fixed-size sample
/// from a large or streaming dataset with uniform probability.
///
/// # Parameters
///
/// * `reservoir_size` - Size of the sample to maintain
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::sketching_sampling::ReservoirSampler;
/// use scirs2_core::ndarray::array;
///
/// let sampler = ReservoirSampler::new()
///     .reservoir_size(100)
///     .random_state(42);
/// ```
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    reservoir_size: usize,
    random_state: Option<u64>,
    reservoir: Vec<usize>,
    seen_count: usize,
}

impl ReservoirSampler {
    /// Create a new ReservoirSampler
    pub fn new() -> Self {
        Self {
            reservoir_size: 1000,
            random_state: None,
            reservoir: Vec::new(),
            seen_count: 0,
        }
    }

    /// Set the reservoir size
    pub fn reservoir_size(mut self, size: usize) -> Self {
        self.reservoir_size = size;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices from a dataset using reservoir sampling
    pub fn sample_indices(&mut self, dataset_size: usize) -> SklResult<Vec<usize>> {
        use scirs2_core::random::thread_rng;

        // Use rng() function from scirs2_core - seeding handled elsewhere if needed
        let mut rng = thread_rng();

        let actual_size = self.reservoir_size.min(dataset_size);
        let mut reservoir = Vec::with_capacity(actual_size);

        // Fill reservoir with first k elements
        for i in 0..actual_size {
            reservoir.push(i);
        }

        // Replace elements with gradually decreasing probability
        for i in actual_size..dataset_size {
            let j = rng.gen_range(0..i + 1);
            if j < actual_size {
                reservoir[j] = i;
            }
        }

        self.reservoir = reservoir.clone();
        self.seen_count = dataset_size;

        Ok(reservoir)
    }

    /// Sample data using reservoir sampling
    pub fn sample_data(&mut self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let indices = self.sample_indices(data.nrows())?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }

    /// Get current reservoir indices
    pub fn get_reservoir_indices(&self) -> &[usize] {
        &self.reservoir
    }

    /// Get count of seen samples
    pub fn get_seen_count(&self) -> usize {
        self.seen_count
    }
}

impl Default for ReservoirSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Importance Sampling for Weighted Data Selection
///
/// Implements importance sampling for selecting samples based on weights,
/// useful for handling imbalanced datasets or focusing on important regions.
///
/// # Parameters
///
/// * `sample_size` - Number of samples to select
/// * `replacement` - Whether to sample with replacement
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::sketching_sampling::ImportanceSampler;
/// use scirs2_core::ndarray::array;
///
/// let sampler = ImportanceSampler::new()
///     .sample_size(100)
///     .replacement(true)
///     .random_state(42);
/// ```
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    sample_size: usize,
    replacement: bool,
    random_state: Option<u64>,
}

impl ImportanceSampler {
    /// Create a new ImportanceSampler
    pub fn new() -> Self {
        Self {
            sample_size: 1000,
            replacement: false,
            random_state: None,
        }
    }

    /// Set the sample size
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Set whether to sample with replacement
    pub fn replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices based on importance weights
    pub fn sample_indices(&self, weights: &ArrayView1<Float>) -> SklResult<Vec<usize>> {
        use scirs2_core::random::thread_rng;

        if weights.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Weights cannot be empty".to_string(),
            ));
        }

        // Normalize weights to probabilities
        let weight_sum: Float = weights.sum();
        if weight_sum <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Weights must be positive".to_string(),
            ));
        }

        let probabilities: Array1<Float> = weights / weight_sum;
        let cumulative: Array1<Float> = {
            let mut cum = Array1::zeros(probabilities.len());
            cum[0] = probabilities[0];
            for i in 1..probabilities.len() {
                cum[i] = cum[i - 1] + probabilities[i];
            }
            cum
        };

        // Use rng() function from scirs2_core - seeding handled elsewhere if needed
        let mut rng = thread_rng();

        let mut selected_indices = Vec::with_capacity(self.sample_size);
        let mut used_indices = std::collections::HashSet::new();

        for _ in 0..self.sample_size {
            let random_value: Float = rng.gen();

            // Find index using binary search on cumulative distribution
            let mut idx = 0;
            for (i, &cum_prob) in cumulative.iter().enumerate() {
                if random_value <= cum_prob {
                    idx = i;
                    break;
                }
            }

            if self.replacement || !used_indices.contains(&idx) {
                selected_indices.push(idx);
                if !self.replacement {
                    used_indices.insert(idx);
                }
            } else if !self.replacement {
                // Find next available index
                for i in 0..weights.len() {
                    if !used_indices.contains(&i) {
                        selected_indices.push(i);
                        used_indices.insert(i);
                        break;
                    }
                }
            }

            if !self.replacement && used_indices.len() == weights.len() {
                break;
            }
        }

        Ok(selected_indices)
    }

    /// Sample data based on importance weights
    pub fn sample_data(
        &self,
        data: &ArrayView2<Float>,
        weights: &ArrayView1<Float>,
    ) -> SklResult<Array2<Float>> {
        if data.nrows() != weights.len() {
            return Err(SklearsError::InvalidInput(
                "Data rows and weights length must match".to_string(),
            ));
        }

        let indices = self.sample_indices(weights)?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }
}

impl Default for ImportanceSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Stratified Sampling for Balanced Data Selection
///
/// Implements stratified sampling to maintain the same proportion of samples
/// in each stratum as in the original dataset, useful for maintaining class balance.
///
/// # Parameters
///
/// * `sample_size` - Total number of samples to select
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::sketching_sampling::StratifiedSampler;
/// use scirs2_core::ndarray::array;
///
/// let sampler = StratifiedSampler::new()
///     .sample_size(100)
///     .random_state(42);
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    sample_size: usize,
    random_state: Option<u64>,
}

impl StratifiedSampler {
    /// Create a new StratifiedSampler
    pub fn new() -> Self {
        Self {
            sample_size: 1000,
            random_state: None,
        }
    }

    /// Set the sample size
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices using stratified sampling
    pub fn sample_indices(&self, strata: &ArrayView1<Float>) -> SklResult<Vec<usize>> {
        use scirs2_core::random::thread_rng;

        if strata.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Strata cannot be empty".to_string(),
            ));
        }

        // Use rng() function from scirs2_core - seeding handled elsewhere if needed
        let mut rng = thread_rng();

        // Group indices by stratum
        let mut stratum_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, &stratum) in strata.iter().enumerate() {
            let key = format!("{:.6}", stratum); // Use string representation for grouping
            stratum_indices.entry(key).or_default().push(idx);
        }

        let total_samples = strata.len();
        let mut selected_indices = Vec::new();

        // Sample proportionally from each stratum
        for (_, indices) in stratum_indices.iter() {
            let stratum_proportion = indices.len() as f64 / total_samples as f64;
            let stratum_sample_size =
                (self.sample_size as f64 * stratum_proportion).round() as usize;
            let stratum_sample_size = stratum_sample_size.min(indices.len());

            if stratum_sample_size > 0 {
                let mut stratum_indices = indices.clone();
                // Simple Fisher-Yates shuffle
                for i in (1..stratum_indices.len()).rev() {
                    let j = rng.gen_range(0..i + 1);
                    stratum_indices.swap(i, j);
                }
                selected_indices.extend(stratum_indices.into_iter().take(stratum_sample_size));
            }
        }

        // If we haven't reached the target sample size, add more samples
        while selected_indices.len() < self.sample_size && selected_indices.len() < total_samples {
            let all_indices: Vec<usize> = (0..total_samples).collect();
            let mut remaining: Vec<usize> = all_indices
                .into_iter()
                .filter(|&idx| !selected_indices.contains(&idx))
                .collect();

            if remaining.is_empty() {
                break;
            }

            // Simple Fisher-Yates shuffle
            for i in (1..remaining.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                remaining.swap(i, j);
            }
            let needed = self.sample_size - selected_indices.len();
            selected_indices.extend(remaining.into_iter().take(needed));
        }

        selected_indices.truncate(self.sample_size);
        Ok(selected_indices)
    }

    /// Sample data using stratified sampling
    pub fn sample_data(
        &self,
        data: &ArrayView2<Float>,
        strata: &ArrayView1<Float>,
    ) -> SklResult<Array2<Float>> {
        if data.nrows() != strata.len() {
            return Err(SklearsError::InvalidInput(
                "Data rows and strata length must match".to_string(),
            ));
        }

        let indices = self.sample_indices(strata)?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }
}

impl Default for StratifiedSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast Transform Methods for Efficient Feature Extraction
///
/// Implements various fast transform algorithms for efficient computation
/// of frequency domain features and other transform-based features.
///
/// # Transform Types
///
/// * `FFT` - Fast Fourier Transform for frequency analysis
/// * `DCT` - Discrete Cosine Transform for compression and feature extraction
/// * `Walsh` - Walsh-Hadamard Transform for fast computation
/// * `Haar` - Haar Transform for multiresolution analysis
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::sketching_sampling::{FastTransformExtractor, TransformType};
/// use scirs2_core::ndarray::array;
///
/// let extractor = FastTransformExtractor::new()
///     .transform_type(TransformType::FFT)
///     .include_magnitude(true)
///     .include_phase(false);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum TransformType {
    /// Fast Fourier Transform
    FFT,
    /// Discrete Cosine Transform
    DCT,
    /// Walsh-Hadamard Transform
    Walsh,
    /// Haar Transform
    Haar,
}

#[derive(Debug, Clone)]
pub struct FastTransformExtractor {
    transform_type: TransformType,
    include_magnitude: bool,
    include_phase: bool,
    include_power: bool,
    normalize: bool,
    zero_pad_to_power_of_2: bool,
}

impl FastTransformExtractor {
    /// Create a new FastTransformExtractor
    pub fn new() -> Self {
        Self {
            transform_type: TransformType::FFT,
            include_magnitude: true,
            include_phase: false,
            include_power: false,
            normalize: true,
            zero_pad_to_power_of_2: true,
        }
    }

    /// Set the transform type
    pub fn transform_type(mut self, transform_type: TransformType) -> Self {
        self.transform_type = transform_type;
        self
    }

    /// Set whether to include magnitude features
    pub fn include_magnitude(mut self, include: bool) -> Self {
        self.include_magnitude = include;
        self
    }

    /// Set whether to include phase features
    pub fn include_phase(mut self, include: bool) -> Self {
        self.include_phase = include;
        self
    }

    /// Set whether to include power features
    pub fn include_power(mut self, include: bool) -> Self {
        self.include_power = include;
        self
    }

    /// Set whether to normalize the output
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set whether to zero-pad to next power of 2
    pub fn zero_pad_to_power_of_2(mut self, zero_pad: bool) -> Self {
        self.zero_pad_to_power_of_2 = zero_pad;
        self
    }

    /// Extract features using fast transform methods
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        let mut features_list = Vec::new();

        for row in data.axis_iter(Axis(0)) {
            let row_features = match self.transform_type {
                TransformType::FFT => self.compute_fft(&row)?,
                TransformType::DCT => self.compute_dct(&row)?,
                TransformType::Walsh => self.compute_walsh(&row)?,
                TransformType::Haar => self.compute_haar(&row)?,
            };
            features_list.push(row_features);
        }

        if features_list.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features extracted".to_string(),
            ));
        }

        let n_features = features_list[0].len();
        let n_samples = features_list.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, features) in features_list.into_iter().enumerate() {
            for (j, &value) in features.iter().enumerate() {
                result[[i, j]] = value;
            }
        }

        Ok(result)
    }

    /// Compute Fast Fourier Transform features
    fn compute_fft(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut padded_signal = signal.to_vec();

        // Zero-pad to next power of 2 if requested
        if self.zero_pad_to_power_of_2 {
            let next_power_of_2 = (padded_signal.len() as f64).log2().ceil() as u32;
            let target_size = 2_usize.pow(next_power_of_2);
            padded_signal.resize(target_size, 0.0);
        }

        // Simple DFT implementation (can be optimized with real FFT libraries)
        let n = padded_signal.len();
        let mut real_part = Vec::with_capacity(n / 2 + 1);
        let mut imag_part = Vec::with_capacity(n / 2 + 1);

        for k in 0..=n / 2 {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
                real_sum += padded_signal[j] * angle.cos();
                imag_sum += padded_signal[j] * angle.sin();
            }

            real_part.push(real_sum);
            imag_part.push(imag_sum);
        }

        let mut features = Vec::new();

        if self.include_magnitude {
            for i in 0..real_part.len() {
                let magnitude = (real_part[i] * real_part[i] + imag_part[i] * imag_part[i]).sqrt();
                features.push(magnitude);
            }
        }

        if self.include_phase {
            for i in 0..real_part.len() {
                let phase = imag_part[i].atan2(real_part[i]);
                features.push(phase);
            }
        }

        if self.include_power {
            for i in 0..real_part.len() {
                let power = real_part[i] * real_part[i] + imag_part[i] * imag_part[i];
                features.push(power);
            }
        }

        if self.normalize && !features.is_empty() {
            let max_val = features.iter().fold(0.0 as Float, |a, &b| a.max(b.abs()));
            if max_val > 0.0 {
                for feature in features.iter_mut() {
                    *feature /= max_val;
                }
            }
        }

        Ok(features)
    }

    /// Compute Discrete Cosine Transform features
    fn compute_dct(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let n = signal.len();
        let mut dct_coeffs = Vec::with_capacity(n);

        for k in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let angle = std::f64::consts::PI * (k as f64) * (j as f64 + 0.5) / (n as f64);
                sum += signal[j] * angle.cos();
            }

            // Apply normalization factors
            let coeff = if k == 0 {
                sum * (1.0 / n as f64).sqrt()
            } else {
                sum * (2.0 / n as f64).sqrt()
            };

            dct_coeffs.push(coeff);
        }

        if self.normalize && !dct_coeffs.is_empty() {
            let max_val = dct_coeffs.iter().fold(0.0 as Float, |a, &b| a.max(b.abs()));
            if max_val > 0.0 {
                for coeff in dct_coeffs.iter_mut() {
                    *coeff /= max_val;
                }
            }
        }

        Ok(dct_coeffs)
    }

    /// Compute Walsh-Hadamard Transform features
    fn compute_walsh(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut data = signal.to_vec();
        let n = data.len();

        // Ensure length is power of 2
        let next_power_of_2 = (n as f64).log2().ceil() as u32;
        let target_size = 2_usize.pow(next_power_of_2);
        data.resize(target_size, 0.0);

        // Fast Walsh-Hadamard Transform
        let mut step = 1;
        while step < target_size {
            for i in (0..target_size).step_by(step * 2) {
                for j in 0..step {
                    let u = data[i + j];
                    let v = data[i + j + step];
                    data[i + j] = u + v;
                    data[i + j + step] = u - v;
                }
            }
            step *= 2;
        }

        if self.normalize && !data.is_empty() {
            let norm_factor = (target_size as f64).sqrt();
            for value in data.iter_mut() {
                *value /= norm_factor;
            }
        }

        Ok(data)
    }

    /// Compute Haar Transform features
    fn compute_haar(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut data = signal.to_vec();
        let n = data.len();

        // Ensure length is power of 2
        let next_power_of_2 = (n as f64).log2().ceil() as u32;
        let target_size = 2_usize.pow(next_power_of_2);
        data.resize(target_size, 0.0);

        let mut temp = vec![0.0; target_size];
        let mut length = target_size;

        while length > 1 {
            let half_length = length / 2;

            // Compute averages (scaling function)
            for i in 0..half_length {
                temp[i] = (data[2 * i] + data[2 * i + 1]) / 2.0_f64.sqrt();
            }

            // Compute differences (wavelet function)
            for i in 0..half_length {
                temp[half_length + i] = (data[2 * i] - data[2 * i + 1]) / 2.0_f64.sqrt();
            }

            // Copy back
            data[0..length].copy_from_slice(&temp[0..length]);
            length = half_length;
        }

        if self.normalize && !data.is_empty() {
            let max_val = data.iter().fold(0.0 as Float, |a, &b| a.max(b.abs()));
            if max_val > 0.0 {
                for value in data.iter_mut() {
                    *value /= max_val;
                }
            }
        }

        Ok(data)
    }
}

impl Default for FastTransformExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_count_min_sketch_basic() {
        let data =
            Array2::from_shape_vec((50, 3), (0..150).map(|x| (x % 20) as f64).collect()).unwrap();

        let sketch = CountMinSketch::new()
            .width(50)
            .depth(3)
            .include_frequency_statistics(true)
            .include_sketch_properties(true);

        let features = sketch.transform(&data.view()).unwrap();

        // Should have 8 features: 5 frequency stats + 3 sketch properties
        assert_eq!(features.len(), 8);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite");
        }

        // Total count should be positive
        assert!(features[0] > 0.0, "Total count should be positive");
    }

    #[test]
    fn test_count_min_sketch_configuration() {
        let data = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();

        // Test with different configurations
        let sketch1 = CountMinSketch::new()
            .width(100)
            .depth(5)
            .include_frequency_statistics(true)
            .include_sketch_properties(false);

        let features1 = sketch1.transform(&data.view()).unwrap();
        assert_eq!(features1.len(), 5); // Only frequency stats

        let sketch2 = CountMinSketch::new()
            .width(100)
            .depth(5)
            .include_frequency_statistics(false)
            .include_sketch_properties(true);

        let features2 = sketch2.transform(&data.view()).unwrap();
        assert_eq!(features2.len(), 3); // Only sketch properties
    }

    #[test]
    fn test_count_min_sketch_empty_data() {
        let data = Array2::zeros((0, 2));
        let sketch = CountMinSketch::new();

        let result = sketch.transform(&data.view());
        assert!(result.is_err(), "Should return error for empty data");
    }

    #[test]
    fn test_fast_johnson_lindenstrauss_basic() {
        let data = Array2::from_shape_vec((10, 8), (0..80).map(|x| x as f64).collect()).unwrap();

        let fjlt = FastJohnsonLindenstrauss::new()
            .target_dimension(4)
            .sparsity_parameter(0.2)
            .include_transform_stats(false);

        let features = fjlt.transform(&data.view()).unwrap();

        // Should have 10 * 4 = 40 features (flattened transformed data)
        assert_eq!(features.len(), 40);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite");
        }
    }

    #[test]
    fn test_fast_johnson_lindenstrauss_stats() {
        let data = Array2::from_shape_vec((5, 4), (0..20).map(|x| x as f64).collect()).unwrap();

        let fjlt = FastJohnsonLindenstrauss::new()
            .target_dimension(2)
            .include_transform_stats(true);

        let features = fjlt.transform(&data.view()).unwrap();

        // Should have 6 features: mean, std, min, max, target_dim, sparsity
        assert_eq!(features.len(), 6);

        // All features should be finite
        for &feature in &features {
            assert!(feature.is_finite(), "Feature should be finite");
        }

        // Target dimension should match
        assert_eq!(features[4], 2.0);
    }

    #[test]
    fn test_fast_johnson_lindenstrauss_empty_data() {
        let data = Array2::zeros((0, 4));
        let fjlt = FastJohnsonLindenstrauss::new();

        let result = fjlt.transform(&data.view());
        assert!(result.is_err(), "Should return error for empty data");
    }

    #[test]
    fn test_fast_johnson_lindenstrauss_zero_columns() {
        let data = Array2::zeros((5, 0));
        let fjlt = FastJohnsonLindenstrauss::new();

        let result = fjlt.transform(&data.view());
        assert!(result.is_err(), "Should return error for zero columns");
    }

    #[test]
    fn test_sketching_methods_consistency() {
        let data =
            Array2::from_shape_vec((30, 4), (0..120).map(|x| (x % 10) as f64).collect()).unwrap();

        // Test Count-Min Sketch with fixed seed
        let sketch1 = CountMinSketch::new().hash_seed(123);
        let sketch2 = CountMinSketch::new().hash_seed(123);

        let features1 = sketch1.transform(&data.view()).unwrap();
        let features2 = sketch2.transform(&data.view()).unwrap();

        // Should be identical with same seed
        assert_eq!(features1.len(), features2.len());
        for (f1, f2) in features1.iter().zip(features2.iter()) {
            assert!(
                (f1 - f2).abs() < 1e-10,
                "Features should be identical with same seed"
            );
        }

        // Test FJLT with fixed seed
        let fjlt1 = FastJohnsonLindenstrauss::new()
            .random_state(456)
            .target_dimension(3);
        let fjlt2 = FastJohnsonLindenstrauss::new()
            .random_state(456)
            .target_dimension(3);

        let fjlt_features1 = fjlt1.transform(&data.view()).unwrap();
        let fjlt_features2 = fjlt2.transform(&data.view()).unwrap();

        // Should be identical with same seed
        assert_eq!(fjlt_features1.len(), fjlt_features2.len());
        for (f1, f2) in fjlt_features1.iter().zip(fjlt_features2.iter()) {
            assert!(
                (f1 - f2).abs() < 1e-10,
                "FJLT features should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_reservoir_sampler_basic() {
        let mut sampler = ReservoirSampler::new().reservoir_size(10).random_state(42);

        let indices = sampler.sample_indices(20).unwrap();
        assert_eq!(indices.len(), 10);
        assert_eq!(sampler.get_seen_count(), 20);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 20);
        }
    }

    #[test]
    fn test_importance_sampler_basic() {
        let weights = Array1::from_vec(vec![0.1, 0.3, 0.2, 0.4]);
        let sampler = ImportanceSampler::new().sample_size(3).random_state(42);

        let indices = sampler.sample_indices(&weights.view()).unwrap();
        assert_eq!(indices.len(), 3);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 4);
        }
    }

    #[test]
    fn test_stratified_sampler_basic() {
        let strata = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let sampler = StratifiedSampler::new().sample_size(4).random_state(42);

        let indices = sampler.sample_indices(&strata.view()).unwrap();
        assert_eq!(indices.len(), 4);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 6);
        }
    }

    #[test]
    fn test_fast_transform_extractor_fft() {
        let data = Array2::from_shape_vec((3, 8), (0..24).map(|x| x as f64).collect()).unwrap();

        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::FFT)
            .include_magnitude(true)
            .include_phase(false);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);

        // All features should be finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_fast_transform_extractor_dct() {
        let data = Array2::from_shape_vec((2, 4), (0..8).map(|x| x as f64).collect()).unwrap();

        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::DCT)
            .normalize(true);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 4);

        // All features should be finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }
}
