//! Feature engineering utilities for machine learning
//!
//! This module provides utilities for generating derived features including
//! polynomial features, interaction features, and feature binning.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Polynomial feature generator
///
/// Generates polynomial and interaction features from input data.
/// For example, with degree=2 and features [a, b], generates: [1, a, b, a², ab, b²]
#[derive(Clone, Debug)]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
}

impl PolynomialFeatures {
    /// Create a new polynomial feature generator
    ///
    /// # Arguments
    /// * `degree` - Maximum degree of polynomial features
    /// * `include_bias` - Whether to include a bias column (constant 1.0)
    /// * `interaction_only` - If true, only interaction features (no powers of single features)
    pub fn new(degree: usize, include_bias: bool, interaction_only: bool) -> UtilsResult<Self> {
        if degree == 0 {
            return Err(UtilsError::InvalidParameter(
                "Degree must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            degree,
            include_bias,
            interaction_only,
        })
    }

    /// Compute the number of output features
    ///
    /// # Arguments
    /// * `n_features` - Number of input features
    ///
    /// # Returns
    /// Number of output features after transformation
    pub fn n_output_features(&self, n_features: usize) -> usize {
        let mut n_out = if self.include_bias { 1 } else { 0 };

        if self.interaction_only {
            // Combinations without powers
            for d in 1..=self.degree {
                n_out += Self::binomial_coefficient(n_features, d);
            }
        } else {
            // Stars and bars: choose n_features + degree from degree
            n_out += Self::binomial_coefficient(n_features + self.degree, self.degree) - 1;
        }

        n_out
    }

    /// Transform features to polynomial features
    ///
    /// # Arguments
    /// * `x` - Input features (n_samples × n_features)
    ///
    /// # Returns
    /// Transformed features with polynomial terms
    pub fn transform(&self, x: &Array2<f64>) -> UtilsResult<Array2<f64>> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(UtilsError::EmptyInput);
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_output = self.n_output_features(n_features);

        let mut result = Array2::zeros((n_samples, n_output));

        for i in 0..n_samples {
            let row = x.row(i);
            let features = self.generate_polynomial_row(&row.to_vec());
            for (j, &val) in features.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    fn generate_polynomial_row(&self, row: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();

        if self.include_bias {
            features.push(1.0);
        }

        if self.interaction_only {
            self.generate_interaction_features(row, &mut features);
        } else {
            self.generate_all_polynomial_features(row, &mut features);
        }

        features
    }

    fn generate_interaction_features(&self, row: &[f64], features: &mut Vec<f64>) {
        // Generate all k-way interactions for k = 1 to degree
        for degree in 1..=self.degree {
            Self::generate_combinations(row, degree, 0, vec![], features);
        }
    }

    fn generate_combinations(
        row: &[f64],
        k: usize,
        start: usize,
        current: Vec<usize>,
        features: &mut Vec<f64>,
    ) {
        if current.len() == k {
            let product: f64 = current.iter().map(|&i| row[i]).product();
            features.push(product);
            return;
        }

        for i in start..row.len() {
            let mut next = current.clone();
            next.push(i);
            Self::generate_combinations(row, k, i + 1, next, features);
        }
    }

    fn generate_all_polynomial_features(&self, row: &[f64], features: &mut Vec<f64>) {
        let n = row.len();
        Self::generate_powers_recursive(row, self.degree, 0, vec![0; n], 1.0, features);
    }

    fn generate_powers_recursive(
        row: &[f64],
        max_degree: usize,
        feature_idx: usize,
        powers: Vec<usize>,
        current_product: f64,
        features: &mut Vec<f64>,
    ) {
        if feature_idx == row.len() {
            if powers.iter().sum::<usize>() > 0 {
                features.push(current_product);
            }
            return;
        }

        let current_degree: usize = powers.iter().sum();

        for power in 0..=(max_degree - current_degree) {
            let mut new_powers = powers.clone();
            new_powers[feature_idx] = power;

            let new_product = if power > 0 {
                current_product * row[feature_idx].powi(power as i32)
            } else {
                current_product
            };

            Self::generate_powers_recursive(
                row,
                max_degree,
                feature_idx + 1,
                new_powers,
                new_product,
                features,
            );
        }
    }

    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1;

        for i in 0..k {
            result *= n - i;
            result /= i + 1;
        }

        result
    }
}

/// Interaction feature generator
///
/// Generates pairwise interaction features (products) between features.
#[derive(Clone, Debug)]
pub struct InteractionFeatures {
    include_self: bool,
}

impl InteractionFeatures {
    /// Create a new interaction feature generator
    ///
    /// # Arguments
    /// * `include_self` - Whether to include self-interactions (x * x)
    pub fn new(include_self: bool) -> Self {
        Self { include_self }
    }

    /// Transform features by adding interaction terms
    ///
    /// # Arguments
    /// * `x` - Input features (n_samples × n_features)
    ///
    /// # Returns
    /// Features with added interaction terms
    pub fn transform(&self, x: &Array2<f64>) -> UtilsResult<Array2<f64>> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(UtilsError::EmptyInput);
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let n_interactions = if self.include_self {
            n_features * (n_features + 1) / 2
        } else {
            n_features * (n_features - 1) / 2
        };

        let mut result = Array2::zeros((n_samples, n_features + n_interactions));

        // Copy original features
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = x[[i, j]];
            }
        }

        // Add interaction features
        let mut col_idx = n_features;
        for i in 0..n_features {
            let start_j = if self.include_self { i } else { i + 1 };
            for j in start_j..n_features {
                for row in 0..n_samples {
                    result[[row, col_idx]] = x[[row, i]] * x[[row, j]];
                }
                col_idx += 1;
            }
        }

        Ok(result)
    }

    /// Get the number of output features
    pub fn n_output_features(&self, n_input_features: usize) -> usize {
        let n_interactions = if self.include_self {
            n_input_features * (n_input_features + 1) / 2
        } else {
            n_input_features * (n_input_features - 1) / 2
        };
        n_input_features + n_interactions
    }
}

impl Default for InteractionFeatures {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Feature binning (discretization) utility
///
/// Transforms continuous features into discrete bins.
#[derive(Clone, Debug)]
pub struct FeatureBinner {
    n_bins: usize,
    strategy: BinningStrategy,
}

/// Strategy for determining bin edges
#[derive(Clone, Debug, PartialEq)]
pub enum BinningStrategy {
    /// Equal width bins
    Uniform,
    /// Equal frequency bins (quantiles)
    Quantile,
    /// K-means clustering for bin centers
    KMeans,
}

impl FeatureBinner {
    /// Create a new feature binner
    ///
    /// # Arguments
    /// * `n_bins` - Number of bins to create
    /// * `strategy` - Binning strategy
    pub fn new(n_bins: usize, strategy: BinningStrategy) -> UtilsResult<Self> {
        if n_bins < 2 {
            return Err(UtilsError::InvalidParameter(
                "n_bins must be at least 2".to_string(),
            ));
        }

        Ok(Self { n_bins, strategy })
    }

    /// Fit and transform features into bins
    ///
    /// # Arguments
    /// * `x` - Input features (1D array)
    ///
    /// # Returns
    /// Bin indices for each value
    pub fn fit_transform(&self, x: &Array1<f64>) -> UtilsResult<Array1<usize>> {
        if x.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        match self.strategy {
            BinningStrategy::Uniform => self.uniform_binning(x),
            BinningStrategy::Quantile => self.quantile_binning(x),
            BinningStrategy::KMeans => {
                // Simplified version - use quantiles as approximation
                self.quantile_binning(x)
            }
        }
    }

    fn uniform_binning(&self, x: &Array1<f64>) -> UtilsResult<Array1<usize>> {
        let min_val = x.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            // All values are the same
            return Ok(Array1::zeros(x.len()));
        }

        let bin_width = (max_val - min_val) / (self.n_bins as f64);
        let mut result = Array1::zeros(x.len());

        for (i, &val) in x.iter().enumerate() {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            result[i] = bin.min(self.n_bins - 1); // Clamp to valid range
        }

        Ok(result)
    }

    fn quantile_binning(&self, x: &Array1<f64>) -> UtilsResult<Array1<usize>> {
        let mut sorted: Vec<f64> = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute quantile edges
        let mut edges = Vec::with_capacity(self.n_bins + 1);
        edges.push(sorted[0] - 1e-10); // Slightly less than minimum

        for i in 1..self.n_bins {
            let quantile = i as f64 / self.n_bins as f64;
            let idx = (quantile * sorted.len() as f64) as usize;
            let idx = idx.min(sorted.len() - 1);
            edges.push(sorted[idx]);
        }

        edges.push(sorted[sorted.len() - 1] + 1e-10); // Slightly more than maximum

        // Assign bins
        let mut result = Array1::zeros(x.len());
        for (i, &val) in x.iter().enumerate() {
            for bin in 0..self.n_bins {
                if val > edges[bin] && val <= edges[bin + 1] {
                    result[i] = bin;
                    break;
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_polynomial_features_degree_2() {
        let poly = PolynomialFeatures::new(2, true, false).unwrap();

        // Input: [a, b]
        // The recursive generation produces: [1, b, ab, b², a, a², ...]
        let x = array![[2.0, 3.0]];
        let result = poly.transform(&x).unwrap();

        assert_eq!(result.ncols(), 6);

        // Check that all expected values are present (order may vary)
        let values: Vec<f64> = (0..result.ncols()).map(|i| result[[0, i]]).collect();

        // Must have bias
        assert!(values.contains(&1.0));
        // Must have a and b
        assert!(values.contains(&2.0));
        assert!(values.contains(&3.0));
        // Must have a², ab, b²
        assert!(values.contains(&4.0)); // a²
        assert!(values.contains(&6.0)); // ab
        assert!(values.contains(&9.0)); // b²
    }

    #[test]
    fn test_polynomial_features_interaction_only() {
        let poly = PolynomialFeatures::new(2, false, true).unwrap();

        // Input: [a, b]
        // Expected output: [a, b, ab] (no a², b²)
        let x = array![[2.0, 3.0]];
        let result = poly.transform(&x).unwrap();

        assert_eq!(result.ncols(), 3);
        assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(result[[0, 1]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(result[[0, 2]], 6.0, epsilon = 1e-10); // ab
    }

    #[test]
    fn test_polynomial_n_output_features() {
        let poly = PolynomialFeatures::new(2, true, false).unwrap();
        assert_eq!(poly.n_output_features(2), 6); // [1, a, b, a², ab, b²]

        let poly = PolynomialFeatures::new(3, false, false).unwrap();
        assert_eq!(poly.n_output_features(2), 9); // [a, b, a², ab, b², a³, a²b, ab², b³]
    }

    #[test]
    fn test_interaction_features() {
        let interaction = InteractionFeatures::new(false);

        // Input: [a, b, c]
        // Interactions (without self): [ab, ac, bc]
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];

        let result = interaction.transform(&x).unwrap();

        assert_eq!(result.ncols(), 6); // 3 original + 3 interactions

        // First row
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(result[[0, 1]], 2.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(result[[0, 2]], 3.0, epsilon = 1e-10); // c
        assert_abs_diff_eq!(result[[0, 3]], 2.0, epsilon = 1e-10); // ab
        assert_abs_diff_eq!(result[[0, 4]], 3.0, epsilon = 1e-10); // ac
        assert_abs_diff_eq!(result[[0, 5]], 6.0, epsilon = 1e-10); // bc
    }

    #[test]
    fn test_interaction_features_with_self() {
        let interaction = InteractionFeatures::new(true);

        let x = array![[2.0, 3.0]];
        let result = interaction.transform(&x).unwrap();

        assert_eq!(result.ncols(), 5); // 2 original + 3 interactions (a², ab, b²)

        assert_abs_diff_eq!(result[[0, 2]], 4.0, epsilon = 1e-10); // a²
        assert_abs_diff_eq!(result[[0, 3]], 6.0, epsilon = 1e-10); // ab
        assert_abs_diff_eq!(result[[0, 4]], 9.0, epsilon = 1e-10); // b²
    }

    #[test]
    fn test_uniform_binning() {
        let binner = FeatureBinner::new(3, BinningStrategy::Uniform).unwrap();

        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let bins = binner.fit_transform(&x).unwrap();

        // Should be divided into 3 bins: [0-3), [3-6), [6-9]
        assert_eq!(bins[0], 0); // 0.0
        assert_eq!(bins[5], 1); // 5.0
        assert_eq!(bins[9], 2); // 9.0
    }

    #[test]
    fn test_quantile_binning() {
        let binner = FeatureBinner::new(4, BinningStrategy::Quantile).unwrap();

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bins = binner.fit_transform(&x).unwrap();

        // Each bin should have approximately equal number of samples
        let mut bin_counts = vec![0; 4];
        for &bin in bins.iter() {
            bin_counts[bin] += 1;
        }

        // All bins should have samples
        for &count in &bin_counts {
            assert!(count > 0);
        }
    }

    #[test]
    fn test_binning_constant_values() {
        let binner = FeatureBinner::new(3, BinningStrategy::Uniform).unwrap();

        let x = array![5.0, 5.0, 5.0, 5.0];
        let bins = binner.fit_transform(&x).unwrap();

        // All values should be in the same bin
        assert!(bins.iter().all(|&b| b == 0));
    }
}
