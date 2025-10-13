//! Probabilistic Methods Module
//!
//! This module provides probabilistic data structures and algorithms for efficient
//! feature extraction and data processing with mathematical guarantees.

use crate::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{FromPrimitive, One, Zero};
use rayon::prelude::*;
use sklears_core::prelude::{Fit, Predict, SklearsError, Transform};
use sklears_core::traits::{Estimator, Untrained};
use sklears_core::types::Float;
use std::collections::{HashMap, VecDeque};
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
/// # use sklears_feature_extraction::engineering::CountMinSketch;
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
        let data_owned = data.mapv(|x| x as f64).to_owned();

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
/// # use sklears_feature_extraction::engineering::FastJohnsonLindenstrauss;
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
        let data_owned = data.mapv(|x| x as f64).to_owned();

        let transformed = self.apply_fjlt(&data_owned);
        let features = self.extract_features(&transformed);

        Ok(Array1::from_vec(
            features.into_iter().map(|x| x as Float).collect(),
        ))
    }
}