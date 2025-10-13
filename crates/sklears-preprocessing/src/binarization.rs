//! Binarization transformers
//!
//! This module provides transformers for binarizing data:
//! - Binarizer: Binarize data according to a threshold
//! - KBinsDiscretizer: Discretize continuous features into bins

use scirs2_core::ndarray::{Array1, Array2};
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};

/// Configuration for Binarizer
#[derive(Debug, Clone)]
pub struct BinarizerConfig {
    /// Feature values below or equal to this are replaced by 0, above it by 1
    pub threshold: Float,
    /// Whether to copy the input array
    pub copy: bool,
}

impl Default for BinarizerConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            copy: true,
        }
    }
}

/// Binarizer transforms data to binary values based on a threshold
pub struct Binarizer<State = Untrained> {
    config: BinarizerConfig,
    state: PhantomData<State>,
}

impl Binarizer<Untrained> {
    /// Create a new Binarizer with default configuration
    pub fn new() -> Self {
        Self {
            config: BinarizerConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a new Binarizer with specified threshold
    pub fn with_threshold(threshold: Float) -> Self {
        Self {
            config: BinarizerConfig {
                threshold,
                copy: true,
            },
            state: PhantomData,
        }
    }

    /// Set the threshold
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set whether to copy the input array
    pub fn copy(mut self, copy: bool) -> Self {
        self.config.copy = copy;
        self
    }
}

impl Default for Binarizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Binarizer<Untrained> {
    type Config = BinarizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for Binarizer<Trained> {
    type Config = BinarizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for Binarizer<Untrained> {
    type Fitted = Binarizer<Trained>;

    fn fit(self, _x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        // Binarizer doesn't need to learn anything from the data
        Ok(Binarizer {
            config: self.config,
            state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for Binarizer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let result = if self.config.copy {
            x.clone()
        } else {
            x.to_owned()
        };

        Ok(result.mapv(|v| if v > self.config.threshold { 1.0 } else { 0.0 }))
    }
}

impl Transform<Array1<Float>, Array1<Float>> for Binarizer<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let result = if self.config.copy {
            x.clone()
        } else {
            x.to_owned()
        };

        Ok(result.mapv(|v| if v > self.config.threshold { 1.0 } else { 0.0 }))
    }
}

/// Discretization strategy for KBinsDiscretizer
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiscretizationStrategy {
    /// All bins have identical widths
    Uniform,
    /// All bins have the same number of points
    Quantile,
    /// Bins are clustered using k-means
    KMeans,
}

/// Encoding method for discretized values
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiscretizerEncoding {
    /// Encode as one-hot vectors
    OneHot,
    /// Encode with the bin identifier as an integer
    Ordinal,
}

/// Configuration for KBinsDiscretizer
#[derive(Debug, Clone)]
pub struct KBinsDiscretizerConfig {
    /// Number of bins to produce
    pub n_bins: usize,
    /// Encoding method
    pub encode: DiscretizerEncoding,
    /// Strategy used to define the widths of the bins
    pub strategy: DiscretizationStrategy,
    /// Subsample size for KMeans strategy
    pub subsample: Option<usize>,
    /// Random state for KMeans
    pub random_state: Option<u64>,
}

impl Default for KBinsDiscretizerConfig {
    fn default() -> Self {
        Self {
            n_bins: 5,
            encode: DiscretizerEncoding::OneHot,
            strategy: DiscretizationStrategy::Quantile,
            subsample: Some(200_000),
            random_state: None,
        }
    }
}

/// KBinsDiscretizer bins continuous data into intervals
pub struct KBinsDiscretizer<State = Untrained> {
    config: KBinsDiscretizerConfig,
    state: PhantomData<State>,
    /// The edges of each bin for each feature
    bin_edges_: Option<Vec<Array1<Float>>>,
    /// Number of bins for each feature
    n_bins_: Option<Vec<usize>>,
}

impl KBinsDiscretizer<Untrained> {
    /// Create a new KBinsDiscretizer with default configuration
    pub fn new() -> Self {
        Self {
            config: KBinsDiscretizerConfig::default(),
            state: PhantomData,
            bin_edges_: None,
            n_bins_: None,
        }
    }

    /// Set the number of bins
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        if n_bins < 2 {
            panic!("n_bins must be at least 2");
        }
        self.config.n_bins = n_bins;
        self
    }

    /// Set the encoding method
    pub fn encode(mut self, encode: DiscretizerEncoding) -> Self {
        self.config.encode = encode;
        self
    }

    /// Set the discretization strategy
    pub fn strategy(mut self, strategy: DiscretizationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }
}

impl Default for KBinsDiscretizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for KBinsDiscretizer<Untrained> {
    type Config = KBinsDiscretizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for KBinsDiscretizer<Trained> {
    type Config = KBinsDiscretizerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute uniform bin edges
fn compute_uniform_bins(data: &Array1<Float>, n_bins: usize) -> Array1<Float> {
    let min_val = data.iter().cloned().fold(Float::INFINITY, Float::min);
    let max_val = data.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

    if (max_val - min_val).abs() < Float::EPSILON {
        // All values are the same
        return Array1::from_vec(vec![min_val - 0.5, max_val + 0.5]);
    }

    let width = (max_val - min_val) / n_bins as Float;
    let mut edges = Vec::with_capacity(n_bins + 1);

    for i in 0..=n_bins {
        edges.push(min_val + i as Float * width);
    }

    // Extend the last edge slightly to include the maximum value
    edges[n_bins] = max_val + Float::EPSILON;

    Array1::from_vec(edges)
}

/// Compute quantile bin edges
fn compute_quantile_bins(data: &Array1<Float>, n_bins: usize) -> Array1<Float> {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_samples = sorted_data.len();
    let mut edges = Vec::with_capacity(n_bins + 1);

    // Add minimum
    edges.push(sorted_data[0]);

    // Add quantile edges
    for i in 1..n_bins {
        let idx = (i * n_samples) / n_bins;
        let value = sorted_data[idx.min(n_samples - 1)];

        // Avoid duplicate edges
        if value > edges.last().unwrap() + Float::EPSILON {
            edges.push(value);
        }
    }

    // Add maximum
    edges.push(sorted_data[n_samples - 1] + Float::EPSILON);

    // Ensure we have at least 2 bins
    if edges.len() < 3 {
        edges.clear();
        edges.push(sorted_data[0]);
        edges.push(sorted_data[n_samples - 1] + Float::EPSILON);
    }

    Array1::from_vec(edges)
}

impl Fit<Array2<Float>, ()> for KBinsDiscretizer<Untrained> {
    type Fitted = KBinsDiscretizer<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_features = x.ncols();
        let mut bin_edges = Vec::with_capacity(n_features);
        let mut n_bins = Vec::with_capacity(n_features);

        // Compute bin edges for each feature
        for j in 0..n_features {
            let feature_data = x.column(j).to_owned();

            let edges = match self.config.strategy {
                DiscretizationStrategy::Uniform => {
                    compute_uniform_bins(&feature_data, self.config.n_bins)
                }
                DiscretizationStrategy::Quantile => {
                    compute_quantile_bins(&feature_data, self.config.n_bins)
                }
                DiscretizationStrategy::KMeans => {
                    // For now, fall back to quantile
                    compute_quantile_bins(&feature_data, self.config.n_bins)
                }
            };

            n_bins.push(edges.len() - 1);
            bin_edges.push(edges);
        }

        Ok(KBinsDiscretizer {
            config: self.config,
            state: PhantomData,
            bin_edges_: Some(bin_edges),
            n_bins_: Some(n_bins),
        })
    }
}

/// Find the bin index for a value given bin edges
fn find_bin(value: Float, edges: &Array1<Float>) -> usize {
    // Binary search for the bin
    let n_edges = edges.len();

    if value <= edges[0] {
        return 0;
    }
    if value >= edges[n_edges - 1] {
        return n_edges - 2;
    }

    let mut left = 0;
    let mut right = n_edges - 1;

    while left < right - 1 {
        let mid = (left + right) / 2;
        if value < edges[mid] {
            right = mid;
        } else {
            left = mid;
        }
    }

    left
}

impl Transform<Array2<Float>, Array2<Float>> for KBinsDiscretizer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let bin_edges = self.bin_edges_.as_ref().unwrap();
        let n_bins = self.n_bins_.as_ref().unwrap();

        match self.config.encode {
            DiscretizerEncoding::Ordinal => {
                let mut result = Array2::zeros((n_samples, n_features));

                for i in 0..n_samples {
                    for j in 0..n_features {
                        let bin_idx = find_bin(x[[i, j]], &bin_edges[j]);
                        result[[i, j]] = bin_idx as Float;
                    }
                }

                Ok(result)
            }
            DiscretizerEncoding::OneHot => {
                // Calculate total number of columns for one-hot encoding
                let total_bins: usize = n_bins.iter().sum();
                let mut result = Array2::zeros((n_samples, total_bins));

                for i in 0..n_samples {
                    let mut col_offset = 0;
                    for j in 0..n_features {
                        let bin_idx = find_bin(x[[i, j]], &bin_edges[j]);
                        result[[i, col_offset + bin_idx]] = 1.0;
                        col_offset += n_bins[j];
                    }
                }

                Ok(result)
            }
        }
    }
}

impl KBinsDiscretizer<Trained> {
    /// Get the bin edges for each feature
    pub fn bin_edges(&self) -> &Vec<Array1<Float>> {
        self.bin_edges_.as_ref().unwrap()
    }

    /// Get the number of bins for each feature
    pub fn n_bins(&self) -> &Vec<usize> {
        self.n_bins_.as_ref().unwrap()
    }

    /// Transform back from bin indices to representative values (inverse transform)
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let bin_edges = self.bin_edges_.as_ref().unwrap();
        let n_features = bin_edges.len();

        match self.config.encode {
            DiscretizerEncoding::Ordinal => {
                if x.ncols() != n_features {
                    return Err(SklearsError::InvalidInput(
                        "Input must have the same number of features as during fit".to_string(),
                    ));
                }

                let mut result = Array2::zeros(x.dim());

                for i in 0..x.nrows() {
                    for j in 0..n_features {
                        let bin_idx = x[[i, j]] as usize;
                        let edges = &bin_edges[j];

                        if bin_idx >= edges.len() - 1 {
                            return Err(SklearsError::InvalidInput(format!(
                                "Invalid bin index {bin_idx} for feature {j}"
                            )));
                        }

                        // Use bin center as representative value
                        result[[i, j]] = (edges[bin_idx] + edges[bin_idx + 1]) / 2.0;
                    }
                }

                Ok(result)
            }
            DiscretizerEncoding::OneHot => Err(SklearsError::InvalidInput(
                "Inverse transform not supported for one-hot encoding".to_string(),
            )),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_binarizer() {
        let x = array![[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0],];

        let binarizer = Binarizer::with_threshold(0.0).fit(&x, &()).unwrap();

        let x_bin = binarizer.transform(&x).unwrap();

        let expected = array![[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],];

        assert_eq!(x_bin, expected);
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let x = array![[1.0, 2.0, 3.0, 4.0]];

        let binarizer = Binarizer::new().threshold(2.5).fit(&x, &()).unwrap();

        let x_bin = binarizer.transform(&x).unwrap();
        let expected = array![[0.0, 0.0, 1.0, 1.0]];

        assert_eq!(x_bin, expected);
    }

    #[test]
    fn test_binarizer_1d() {
        let x = array![1.0, -1.0, 2.0, 0.0];

        let binarizer = Binarizer::new().fit(&array![[0.0]], &()).unwrap();

        let x_bin = binarizer.transform(&x).unwrap();
        let expected = array![1.0, 0.0, 1.0, 0.0];

        assert_eq!(x_bin, expected);
    }

    #[test]
    fn test_kbins_discretizer_uniform() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0],];

        let discretizer = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(DiscretizationStrategy::Uniform)
            .encode(DiscretizerEncoding::Ordinal)
            .fit(&x, &())
            .unwrap();

        let x_disc = discretizer.transform(&x).unwrap();

        // With 3 bins and uniform strategy: [0, 2), [2, 4), [4, 5+ε]
        // Values should be binned as: 0, 0, 1, 1, 2, 2
        assert_eq!(
            x_disc.column(0).to_vec(),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );
    }

    #[test]
    fn test_kbins_discretizer_quantile() {
        let x = array![[0.0], [1.0], [1.0], [2.0], [3.0], [10.0],];

        let discretizer = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(DiscretizationStrategy::Quantile)
            .encode(DiscretizerEncoding::Ordinal)
            .fit(&x, &())
            .unwrap();

        let x_disc = discretizer.transform(&x).unwrap();

        // Check that each bin has approximately the same number of samples
        let bin_counts = vec![
            x_disc.iter().filter(|&&v| v == 0.0).count(),
            x_disc.iter().filter(|&&v| v == 1.0).count(),
            x_disc.iter().filter(|&&v| v == 2.0).count(),
        ];

        // Each bin should have about 2 samples
        for count in bin_counts {
            assert!(count >= 1 && count <= 3);
        }
    }

    #[test]
    fn test_kbins_discretizer_onehot() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0],];

        let discretizer = KBinsDiscretizer::new()
            .n_bins(2)
            .encode(DiscretizerEncoding::OneHot)
            .fit(&x, &())
            .unwrap();

        let x_disc = discretizer.transform(&x).unwrap();

        // With 2 bins per feature and 2 features, we should get 4 columns
        assert_eq!(x_disc.ncols(), 4);

        // Each row should have exactly 2 ones (one per feature)
        for i in 0..x_disc.nrows() {
            let row_sum: Float = x_disc.row(i).sum();
            assert_eq!(row_sum, 2.0);
        }
    }

    #[test]
    fn test_kbins_discretizer_inverse_transform() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0],];

        let discretizer = KBinsDiscretizer::new()
            .n_bins(3)
            .strategy(DiscretizationStrategy::Uniform)
            .encode(DiscretizerEncoding::Ordinal)
            .fit(&x, &())
            .unwrap();

        let x_disc = discretizer.transform(&x).unwrap();
        let x_inv = discretizer.inverse_transform(&x_disc).unwrap();

        // The inverse transform should produce values close to the bin centers
        // Bins: [0, 2), [2, 4), [4, 5+ε] with centers at 1, 3, ~4.5
        assert!(x_inv[[0, 0]] < 2.0); // First bin center
        assert!(x_inv[[2, 0]] > 2.0 && x_inv[[2, 0]] < 4.0); // Second bin center
        assert!(x_inv[[4, 0]] > 4.0); // Third bin center
    }

    #[test]
    fn test_find_bin() {
        let edges = array![0.0, 2.0, 4.0, 6.0];

        assert_eq!(find_bin(-1.0, &edges), 0);
        assert_eq!(find_bin(0.0, &edges), 0);
        assert_eq!(find_bin(1.0, &edges), 0);
        assert_eq!(find_bin(2.0, &edges), 1);
        assert_eq!(find_bin(3.0, &edges), 1);
        assert_eq!(find_bin(4.0, &edges), 2);
        assert_eq!(find_bin(5.0, &edges), 2);
        assert_eq!(find_bin(6.0, &edges), 2);
        assert_eq!(find_bin(7.0, &edges), 2);
    }
}
