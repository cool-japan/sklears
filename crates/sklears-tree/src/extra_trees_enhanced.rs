//! Enhanced Extra Trees with Advanced Randomization Strategies
//!
//! This module provides advanced randomization techniques for Extra Trees:
//! - Rotation Forests (PCA-based feature transformation)
//! - Oblique splits (hyperplane-based splits)
//! - Feature binning for categorical/discrete features
//! - Sparse feature optimization

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::{
    essentials::{Normal, Uniform},
    CoreRandom,
};
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Advanced randomization strategy for Extra Trees
#[derive(Debug, Clone)]
pub enum RandomizationStrategy {
    /// Standard Extra Trees with random thresholds
    Standard,
    /// Rotation Forest: Apply PCA rotation to feature subsets
    RotationForest {
        /// Number of feature subsets to create
        n_subsets: usize,
        /// Fraction of features per subset
        subset_fraction: Float,
    },
    /// Oblique splits using random hyperplanes
    ObliqueSplits {
        /// Number of features to combine in each split
        n_features_per_split: usize,
    },
    /// Random projections for dimensionality reduction
    RandomProjection {
        /// Target dimensionality
        target_dim: usize,
    },
}

impl Default for RandomizationStrategy {
    fn default() -> Self {
        Self::Standard
    }
}

/// Feature binning configuration for categorical/discrete features
#[derive(Debug, Clone)]
pub struct FeatureBinning {
    /// Number of bins per feature
    pub n_bins: usize,
    /// Binning strategy
    pub strategy: BinningStrategy,
    /// Features to bin (None = all features)
    pub features_to_bin: Option<Vec<usize>>,
}

/// Binning strategy for discretizing continuous features
#[derive(Debug, Clone, Copy)]
pub enum BinningStrategy {
    /// Equal-width binning
    Uniform,
    /// Quantile-based binning (equal frequency)
    Quantile,
    /// K-means clustering for binning
    KMeans,
}

impl Default for FeatureBinning {
    fn default() -> Self {
        Self {
            n_bins: 10,
            strategy: BinningStrategy::Quantile,
            features_to_bin: None,
        }
    }
}

/// Sparse feature configuration
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Sparsity threshold (fraction of zeros to consider sparse)
    pub sparsity_threshold: Float,
    /// Whether to use specialized sparse algorithms
    pub use_sparse_optimization: bool,
    /// Compression strategy for sparse features
    pub compression: SparseCompression,
}

/// Sparse feature compression strategy
#[derive(Debug, Clone, Copy)]
pub enum SparseCompression {
    /// No compression
    None,
    /// Coordinate list (COO) format
    COO,
    /// Compressed sparse row (CSR) format
    CSR,
    /// Compressed sparse column (CSC) format
    CSC,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.5,
            use_sparse_optimization: true,
            compression: SparseCompression::CSR,
        }
    }
}

/// Apply rotation forest transformation to features
///
/// Rotation Forest applies PCA to feature subsets and concatenates the rotated features.
/// This can improve ensemble diversity and performance.
pub fn apply_rotation_forest(
    x: &Array2<Float>,
    n_subsets: usize,
    subset_fraction: Float,
    rng: &mut CoreRandom,
) -> Result<Array2<Float>> {
    let n_features = x.ncols();
    let subset_size = ((n_features as Float * subset_fraction).ceil() as usize).max(1);

    let mut rotated_x = x.clone();

    for _subset_idx in 0..n_subsets {
        // Randomly select feature subset
        let feature_indices = sample_features(n_features, subset_size, rng);

        // Extract subset
        let subset = select_features(x, &feature_indices);

        // Apply PCA rotation (simplified: just add random rotation)
        let rotation = generate_rotation_matrix(subset_size, rng)?;
        let rotated_subset = apply_rotation(&subset, &rotation)?;

        // Update features in place
        for (i, &feat_idx) in feature_indices.iter().enumerate() {
            if feat_idx < rotated_x.ncols() {
                for row_idx in 0..rotated_x.nrows() {
                    rotated_x[[row_idx, feat_idx]] = rotated_subset[[row_idx, i]];
                }
            }
        }
    }

    Ok(rotated_x)
}

/// Sample random feature indices
fn sample_features(n_features: usize, n_to_sample: usize, rng: &mut CoreRandom) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_features).collect();

    // Fisher-Yates shuffle for first n_to_sample elements
    for i in 0..n_to_sample.min(n_features) {
        let uniform = Uniform::new(i, n_features).unwrap();
        let j = rng.sample(uniform);
        indices.swap(i, j);
    }

    indices.truncate(n_to_sample);
    indices
}

/// Select specific features from array
fn select_features(x: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
    let n_samples = x.nrows();
    let n_features = indices.len();
    let mut result = Array2::zeros((n_samples, n_features));

    for (new_idx, &orig_idx) in indices.iter().enumerate() {
        if orig_idx < x.ncols() {
            for row_idx in 0..n_samples {
                result[[row_idx, new_idx]] = x[[row_idx, orig_idx]];
            }
        }
    }

    result
}

/// Generate random rotation matrix
fn generate_rotation_matrix(dim: usize, rng: &mut CoreRandom) -> Result<Array2<Float>> {
    let mut matrix = Array2::zeros((dim, dim));
    let normal = Normal::new(0.0, 1.0).map_err(|_| {
        SklearsError::InvalidInput("Failed to create normal distribution".to_string())
    })?;

    // Generate random matrix
    for i in 0..dim {
        for j in 0..dim {
            matrix[[i, j]] = rng.sample(normal);
        }
    }

    // Simple normalization (not true Gram-Schmidt, but sufficient for randomization)
    for j in 0..dim {
        let col = matrix.column(j);
        let dot_product: Float = col.dot(&col);
        let norm = dot_product.sqrt();
        if norm > 1e-10 {
            for i in 0..dim {
                matrix[[i, j]] /= norm;
            }
        }
    }

    Ok(matrix)
}

/// Apply rotation matrix to data
fn apply_rotation(x: &Array2<Float>, rotation: &Array2<Float>) -> Result<Array2<Float>> {
    if x.ncols() != rotation.nrows() {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("x.ncols() == rotation.nrows() ({})", rotation.nrows()),
            actual: format!("x.ncols() = {}", x.ncols()),
        });
    }

    // Matrix multiplication: x * rotation
    let n_samples = x.nrows();
    let n_features = rotation.ncols();
    let mut result = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..x.ncols() {
                sum += x[[i, k]] * rotation[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Apply oblique splits using random hyperplanes
///
/// Instead of axis-aligned splits, oblique splits use random linear combinations of features.
pub fn generate_oblique_split(
    x: &Array2<Float>,
    n_features_per_split: usize,
    rng: &mut CoreRandom,
) -> Result<(Array1<Float>, Float)> {
    let n_features = x.ncols();
    let n_to_use = n_features_per_split.min(n_features);

    // Generate random normal vector
    let mut normal = Array1::zeros(n_features);
    let dist = Normal::new(0.0, 1.0).map_err(|_| {
        SklearsError::InvalidInput("Failed to create normal distribution".to_string())
    })?;

    // Select random features and assign random weights
    let feature_indices = sample_features(n_features, n_to_use, rng);
    for &idx in &feature_indices {
        normal[idx] = rng.sample(dist);
    }

    // Normalize
    let dot_product: Float = normal.dot(&normal);
    let norm = dot_product.sqrt();
    if norm > 1e-10 {
        normal /= norm;
    }

    // Compute projections and random threshold
    let projections = compute_projections(x, &normal);
    let min_proj = projections
        .iter()
        .cloned()
        .fold(Float::INFINITY, Float::min);
    let max_proj = projections
        .iter()
        .cloned()
        .fold(Float::NEG_INFINITY, Float::max);

    let threshold_dist = Uniform::new(min_proj, max_proj).map_err(|_| {
        SklearsError::InvalidInput("Failed to create threshold distribution".to_string())
    })?;
    let threshold = rng.sample(threshold_dist);

    Ok((normal, threshold))
}

/// Compute projections of samples onto a direction vector
fn compute_projections(x: &Array2<Float>, direction: &Array1<Float>) -> Vec<Float> {
    let mut projections = Vec::with_capacity(x.nrows());

    for row in x.axis_iter(Axis(0)) {
        let projection = row.dot(direction);
        projections.push(projection);
    }

    projections
}

/// Apply feature binning to discretize continuous features
pub fn apply_feature_binning(x: &Array2<Float>, binning: &FeatureBinning) -> Result<Array2<Float>> {
    let mut binned_x = x.clone();

    let features_to_process = match &binning.features_to_bin {
        Some(indices) => indices.clone(),
        None => (0..x.ncols()).collect(),
    };

    for &feat_idx in &features_to_process {
        if feat_idx >= x.ncols() {
            continue;
        }

        let feature_col = x.column(feat_idx);
        let bin_edges = compute_bin_edges(&feature_col, binning.n_bins, binning.strategy)?;

        // Assign samples to bins
        for row_idx in 0..x.nrows() {
            let value = x[[row_idx, feat_idx]];
            let bin_idx = find_bin(value, &bin_edges);
            binned_x[[row_idx, feat_idx]] = bin_idx as Float;
        }
    }

    Ok(binned_x)
}

/// Compute bin edges for a feature
fn compute_bin_edges(
    feature: &ArrayView1<Float>,
    n_bins: usize,
    strategy: BinningStrategy,
) -> Result<Vec<Float>> {
    let mut values: Vec<Float> = feature.to_vec();

    match strategy {
        BinningStrategy::Uniform => {
            let min_val = values.iter().cloned().fold(Float::INFINITY, Float::min);
            let max_val = values.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

            let mut edges = Vec::with_capacity(n_bins + 1);
            for i in 0..=n_bins {
                edges.push(min_val + (max_val - min_val) * (i as Float / n_bins as Float));
            }
            Ok(edges)
        }

        BinningStrategy::Quantile => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut edges = Vec::with_capacity(n_bins + 1);
            for i in 0..=n_bins {
                let idx = ((values.len() - 1) as Float * (i as Float / n_bins as Float)) as usize;
                edges.push(values[idx]);
            }

            // Remove duplicates
            edges.dedup();
            Ok(edges)
        }

        BinningStrategy::KMeans => {
            // Simplified K-means: use quantiles as initial centers
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut centers = Vec::with_capacity(n_bins);
            for i in 0..n_bins {
                let idx =
                    ((values.len() - 1) as Float * ((i as Float + 0.5) / n_bins as Float)) as usize;
                centers.push(values[idx]);
            }

            // Simple iteration (not full K-means)
            for _ in 0..5 {
                let mut new_centers = vec![0.0; n_bins];
                let mut counts = vec![0; n_bins];

                for &val in &values {
                    let mut best_center = 0;
                    let mut best_dist = Float::INFINITY;

                    for (c_idx, &center) in centers.iter().enumerate() {
                        let dist = (val - center).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_center = c_idx;
                        }
                    }

                    new_centers[best_center] += val;
                    counts[best_center] += 1;
                }

                for i in 0..n_bins {
                    if counts[i] > 0 {
                        centers[i] = new_centers[i] / counts[i] as Float;
                    }
                }
            }

            centers.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Create edges from centers
            let mut edges = vec![Float::NEG_INFINITY];
            for i in 0..centers.len() - 1 {
                edges.push((centers[i] + centers[i + 1]) / 2.0);
            }
            edges.push(Float::INFINITY);

            Ok(edges)
        }
    }
}

/// Find which bin a value belongs to
fn find_bin(value: Float, bin_edges: &[Float]) -> usize {
    for (i, &edge) in bin_edges.iter().enumerate().skip(1) {
        if value <= edge {
            return i - 1;
        }
    }
    bin_edges.len().saturating_sub(2)
}

/// Detect sparse features and compute sparsity statistics
pub fn analyze_sparsity(x: &Array2<Float>) -> HashMap<usize, Float> {
    let mut sparsity_map = HashMap::new();

    for feat_idx in 0..x.ncols() {
        let feature_col = x.column(feat_idx);
        let n_zeros = feature_col.iter().filter(|&&v| v.abs() < 1e-10).count();
        let sparsity = n_zeros as Float / feature_col.len() as Float;
        sparsity_map.insert(feat_idx, sparsity);
    }

    sparsity_map
}

/// Sparse feature representation
#[derive(Debug, Clone)]
pub struct SparseFeature {
    /// Non-zero indices
    pub indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<Float>,
    /// Total length
    pub length: usize,
}

impl SparseFeature {
    /// Create from dense array
    pub fn from_dense(data: &ArrayView1<Float>) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in data.iter().enumerate() {
            if val.abs() > 1e-10 {
                indices.push(i);
                values.push(val);
            }
        }

        Self {
            indices,
            values,
            length: data.len(),
        }
    }

    /// Compute sparsity ratio
    pub fn sparsity(&self) -> Float {
        1.0 - (self.values.len() as Float / self.length as Float)
    }

    /// Get value at index
    pub fn get(&self, idx: usize) -> Float {
        self.indices
            .iter()
            .position(|&i| i == idx)
            .map(|pos| self.values[pos])
            .unwrap_or(0.0)
    }
}

/// Convert to sparse representation if beneficial
pub fn to_sparse_if_beneficial(
    x: &Array2<Float>,
    config: &SparseConfig,
) -> Result<Vec<SparseFeature>> {
    let sparsity_map = analyze_sparsity(x);
    let mut sparse_features = Vec::with_capacity(x.ncols());

    for feat_idx in 0..x.ncols() {
        let sparsity = sparsity_map.get(&feat_idx).copied().unwrap_or(0.0);

        if sparsity >= config.sparsity_threshold && config.use_sparse_optimization {
            let feature_col = x.column(feat_idx);
            sparse_features.push(SparseFeature::from_dense(&feature_col));
        } else {
            // Keep as dense (convert to sparse with all values)
            let feature_col = x.column(feat_idx);
            let indices: Vec<usize> = (0..feature_col.len()).collect();
            let values: Vec<Float> = feature_col.to_vec();
            sparse_features.push(SparseFeature {
                indices,
                values,
                length: feature_col.len(),
            });
        }
    }

    Ok(sparse_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_rotation_forest() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let mut rng = thread_rng();
        let result = apply_rotation_forest(&x, 2, 0.6, &mut rng).unwrap();

        assert_eq!(result.shape(), x.shape());
    }

    #[test]
    fn test_oblique_split() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let mut rng = thread_rng();
        let (normal, threshold) = generate_oblique_split(&x, 2, &mut rng).unwrap();

        assert_eq!(normal.len(), 2);
        assert!(threshold.is_finite());
    }

    #[test]
    fn test_feature_binning_uniform() {
        let x = array![[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]];

        let binning = FeatureBinning {
            n_bins: 2,
            strategy: BinningStrategy::Uniform,
            features_to_bin: Some(vec![0]),
        };

        let result = apply_feature_binning(&x, &binning).unwrap();
        assert_eq!(result.shape(), x.shape());

        // Values should be binned
        let binned_col = result.column(0);
        for &val in binned_col.iter() {
            assert!(val >= 0.0 && val < 2.0);
        }
    }

    #[test]
    fn test_sparsity_analysis() {
        let x = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

        let sparsity_map = analyze_sparsity(&x);

        // Each feature should be 2/3 sparse
        for feat_idx in 0..3 {
            let sparsity = sparsity_map.get(&feat_idx).unwrap();
            assert!((*sparsity - 2.0 / 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_sparse_feature() {
        let dense = array![1.0, 0.0, 0.0, 2.0, 0.0];
        let sparse = SparseFeature::from_dense(&dense.view());

        assert_eq!(sparse.indices, vec![0, 3]);
        assert_eq!(sparse.values, vec![1.0, 2.0]);
        assert_eq!(sparse.length, 5);
        assert_eq!(sparse.sparsity(), 0.6);

        assert_eq!(sparse.get(0), 1.0);
        assert_eq!(sparse.get(1), 0.0);
        assert_eq!(sparse.get(3), 2.0);
    }

    #[test]
    fn test_to_sparse_if_beneficial() {
        let x = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

        let config = SparseConfig {
            sparsity_threshold: 0.5,
            use_sparse_optimization: true,
            compression: SparseCompression::CSR,
        };

        let sparse_features = to_sparse_if_beneficial(&x, &config).unwrap();
        assert_eq!(sparse_features.len(), 3);

        // All features should be sparse
        for sf in &sparse_features {
            assert!(sf.sparsity() >= 0.5);
        }
    }
}
