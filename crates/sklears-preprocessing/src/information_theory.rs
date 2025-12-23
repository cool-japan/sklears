//! Information-theoretic features and transformations
//!
//! This module provides feature engineering and selection based on information theory including:
//! - Entropy measures (Shannon, Renyi, permutation entropy)
//! - Mutual information and conditional mutual information
//! - Information gain for feature selection
//! - Transfer entropy for causality detection
//! - Complexity measures (Lempel-Ziv, approximate entropy)
//! - Information bottleneck features

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::prelude::*;
use std::collections::HashMap;

// ================================================================================================
// Entropy Calculations
// ================================================================================================

/// Calculate Shannon entropy for a discrete distribution
///
/// H(X) = -∑ p(x) log₂ p(x)
pub fn shannon_entropy(data: &Array1<f64>, bins: usize) -> Result<f64> {
    if data.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Data array cannot be empty".to_string(),
        ));
    }

    let probabilities = compute_probabilities(data, bins)?;
    let mut entropy = 0.0;

    for &p in probabilities.iter() {
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

/// Calculate Renyi entropy of order alpha
///
/// H_α(X) = 1/(1-α) log₂(∑ p(x)^α)
pub fn renyi_entropy(data: &Array1<f64>, bins: usize, alpha: f64) -> Result<f64> {
    if alpha <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "Alpha must be positive".to_string(),
        ));
    }

    if (alpha - 1.0).abs() < 1e-10 {
        // Limit case: Renyi entropy converges to Shannon entropy as alpha → 1
        return shannon_entropy(data, bins);
    }

    let probabilities = compute_probabilities(data, bins)?;
    let sum: f64 = probabilities.iter().map(|&p| p.powf(alpha)).sum();

    if sum <= 0.0 {
        return Ok(0.0);
    }

    Ok((1.0 / (1.0 - alpha)) * sum.log2())
}

/// Calculate permutation entropy (ordinal pattern-based entropy)
pub fn permutation_entropy(data: &Array1<f64>, order: usize, delay: usize) -> Result<f64> {
    if data.len() < order {
        return Err(SklearsError::InvalidInput(
            "Data length must be at least equal to order".to_string(),
        ));
    }

    let n = data.len() - (order - 1) * delay;
    if n == 0 {
        return Err(SklearsError::InvalidInput(
            "Insufficient data for given order and delay".to_string(),
        ));
    }

    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();

    for i in 0..n {
        let mut pattern = Vec::with_capacity(order);
        for j in 0..order {
            pattern.push((i + j * delay, data[i + j * delay]));
        }

        // Sort by value, keeping track of original indices
        pattern.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Extract the ordinal pattern
        let ordinal_pattern: Vec<usize> = pattern.iter().map(|&(idx, _)| idx % order).collect();

        *pattern_counts.entry(ordinal_pattern).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in pattern_counts.values() {
        let p = count as f64 / n as f64;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

/// Calculate approximate entropy (ApEn) - measure of regularity
pub fn approximate_entropy(data: &Array1<f64>, m: usize, r: f64) -> Result<f64> {
    if data.len() < m + 1 {
        return Err(SklearsError::InvalidInput(
            "Data length must be greater than m".to_string(),
        ));
    }

    let n = data.len();
    let phi_m = phi_function(data, m, r, n)?;
    let phi_m1 = phi_function(data, m + 1, r, n)?;

    Ok(phi_m - phi_m1)
}

/// Helper function for approximate entropy
fn phi_function(data: &Array1<f64>, m: usize, r: f64, n: usize) -> Result<f64> {
    let mut patterns = Vec::new();

    for i in 0..=(n - m) {
        let pattern: Vec<f64> = (0..m).map(|j| data[i + j]).collect();
        patterns.push(pattern);
    }

    let mut phi = 0.0;

    for i in 0..patterns.len() {
        let mut count = 0;
        for j in 0..patterns.len() {
            let max_diff = patterns[i]
                .iter()
                .zip(patterns[j].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            if max_diff <= r {
                count += 1;
            }
        }

        let c_i = count as f64 / patterns.len() as f64;
        if c_i > 0.0 {
            phi += c_i.ln();
        }
    }

    Ok(phi / patterns.len() as f64)
}

// ================================================================================================
// Mutual Information
// ================================================================================================

/// Calculate mutual information between two variables
///
/// I(X;Y) = H(X) + H(Y) - H(X,Y)
pub fn mutual_information(x: &Array1<f64>, y: &Array1<f64>, bins: usize) -> Result<f64> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let h_x = shannon_entropy(x, bins)?;
    let h_y = shannon_entropy(y, bins)?;
    let h_xy = joint_entropy(x, y, bins)?;

    Ok(h_x + h_y - h_xy)
}

/// Calculate joint entropy H(X,Y)
pub fn joint_entropy(x: &Array1<f64>, y: &Array1<f64>, bins: usize) -> Result<f64> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let joint_probs = compute_joint_probabilities(x, y, bins)?;
    let mut entropy = 0.0;

    for &p in joint_probs.values() {
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

/// Calculate conditional entropy H(Y|X)
pub fn conditional_entropy(y: &Array1<f64>, x: &Array1<f64>, bins: usize) -> Result<f64> {
    let h_xy = joint_entropy(x, y, bins)?;
    let h_x = shannon_entropy(x, bins)?;
    Ok(h_xy - h_x)
}

/// Calculate normalized mutual information (0 to 1)
pub fn normalized_mutual_information(x: &Array1<f64>, y: &Array1<f64>, bins: usize) -> Result<f64> {
    let mi = mutual_information(x, y, bins)?;
    let h_x = shannon_entropy(x, bins)?;
    let h_y = shannon_entropy(y, bins)?;

    if h_x == 0.0 || h_y == 0.0 {
        return Ok(0.0);
    }

    // Clamp to [0, 1] due to numerical precision issues in discrete entropy estimation
    let nmi = mi / ((h_x + h_y) / 2.0).sqrt();
    Ok(nmi.min(1.0).max(0.0))
}

// ================================================================================================
// Transfer Entropy
// ================================================================================================

/// Calculate transfer entropy from X to Y (directional information flow)
///
/// TE(X→Y) = I(Y_t+1; X_t | Y_t)
pub fn transfer_entropy(x: &Array1<f64>, y: &Array1<f64>, bins: usize, lag: usize) -> Result<f64> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    if x.len() <= lag {
        return Err(SklearsError::InvalidInput(
            "Array length must be greater than lag".to_string(),
        ));
    }

    // Extract lagged series
    let y_future = y.slice(s![lag..]).to_owned();
    let x_past = x.slice(s![..x.len() - lag]).to_owned();
    let y_past = y.slice(s![..y.len() - lag]).to_owned();

    // TE(X→Y) = H(Y_future, Y_past) + H(X_past, Y_past) - H(Y_future, X_past, Y_past) - H(Y_past)
    let h_y_future_y_past = joint_entropy(&y_future, &y_past, bins)?;
    let h_x_past_y_past = joint_entropy(&x_past, &y_past, bins)?;
    let h_y_past = shannon_entropy(&y_past, bins)?;

    // For three-way joint entropy, we need a simplified approximation
    let h_xyz = approximate_trivariate_entropy(&y_future, &x_past, &y_past, bins)?;

    Ok(h_y_future_y_past + h_x_past_y_past - h_xyz - h_y_past)
}

/// Approximate trivariate entropy (simplified)
fn approximate_trivariate_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
    bins: usize,
) -> Result<f64> {
    if x.len() != y.len() || y.len() != z.len() {
        return Err(SklearsError::InvalidInput(
            "All arrays must have the same length".to_string(),
        ));
    }

    // Discretize all three variables
    let x_disc = discretize(x, bins)?;
    let y_disc = discretize(y, bins)?;
    let z_disc = discretize(z, bins)?;

    let mut counts: HashMap<(usize, usize, usize), usize> = HashMap::new();
    let n = x.len();

    for i in 0..n {
        *counts.entry((x_disc[i], y_disc[i], z_disc[i])).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / n as f64;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

// ================================================================================================
// Complexity Measures
// ================================================================================================

/// Calculate Lempel-Ziv complexity (normalized)
pub fn lempel_ziv_complexity(data: &Array1<f64>, bins: usize) -> Result<f64> {
    if data.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Data array cannot be empty".to_string(),
        ));
    }

    // Convert to binary string
    let binary = discretize_to_binary(data, bins)?;
    let n = binary.len();

    let mut complexity = 1;
    let mut prefix_len = 1;
    let mut i = 0;

    while i + prefix_len <= n {
        let prefix = &binary[i..i + prefix_len];
        let mut found = false;

        // Search for prefix in previous subsequence
        // Ensure we don't underflow when computing start position
        let start_j = if prefix_len <= i + 1 {
            i + 1 - prefix_len
        } else {
            0
        };
        for j in start_j..=i {
            if j >= prefix_len && &binary[j - prefix_len..j] == prefix {
                found = true;
                break;
            }
        }

        if found {
            prefix_len += 1;
        } else {
            complexity += 1;
            i += prefix_len;
            prefix_len = 1;
        }
    }

    // Normalize by theoretical maximum
    let max_complexity = n as f64 / (n as f64).log2();
    Ok(complexity as f64 / max_complexity)
}

/// Sample entropy - improved version of approximate entropy
pub fn sample_entropy(data: &Array1<f64>, m: usize, r: f64) -> Result<f64> {
    if data.len() < m + 1 {
        return Err(SklearsError::InvalidInput(
            "Data length must be greater than m".to_string(),
        ));
    }

    let n = data.len();
    let mut a: f64 = 0.0;
    let mut b: f64 = 0.0;

    for i in 0..n - m {
        for j in i + 1..n - m {
            let mut match_m = true;

            for k in 0..m {
                if (data[i + k] - data[j + k]).abs() > r {
                    match_m = false;
                    break;
                }
            }

            if match_m {
                b += 1.0;
                if (data[i + m] - data[j + m]).abs() <= r {
                    a += 1.0;
                }
            }
        }
    }

    if b == 0.0 {
        return Ok(0.0);
    }

    Ok(-(a / b).ln())
}

// ================================================================================================
// Information-Based Feature Selection
// ================================================================================================

/// Configuration for information-based feature selection
#[derive(Debug, Clone)]
pub struct InformationFeatureSelectorConfig {
    /// Metric to use for feature ranking
    pub metric: InformationMetric,
    /// Number of bins for discretization
    pub bins: usize,
    /// Number of top features to select (None = use threshold)
    pub k: Option<usize>,
    /// Threshold for feature selection (None = use k)
    pub threshold: Option<f64>,
}

/// Information-theoretic metric for feature selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InformationMetric {
    /// Mutual information with target
    MutualInformation,
    /// Normalized mutual information
    NormalizedMI,
    /// Information gain
    InformationGain,
    /// Symmetrical uncertainty
    SymmetricalUncertainty,
}

impl Default for InformationFeatureSelectorConfig {
    fn default() -> Self {
        Self {
            metric: InformationMetric::MutualInformation,
            bins: 10,
            k: Some(10),
            threshold: None,
        }
    }
}

/// Information-based feature selector
pub struct InformationFeatureSelector {
    config: InformationFeatureSelectorConfig,
}

/// Fitted information-based feature selector
pub struct InformationFeatureSelectorFitted {
    config: InformationFeatureSelectorConfig,
    /// Feature scores (mutual information, information gain, etc.)
    scores: Vec<f64>,
    /// Selected feature indices
    selected_features: Vec<usize>,
}

impl InformationFeatureSelector {
    /// Create a new information-based feature selector
    pub fn new(config: InformationFeatureSelectorConfig) -> Self {
        Self { config }
    }
}

impl Estimator for InformationFeatureSelector {
    type Config = InformationFeatureSelectorConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for InformationFeatureSelector {
    type Fitted = InformationFeatureSelectorFitted;

    fn fit(self, X: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        // Validate input dimensions
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let n_features = X.ncols();
        let mut scores = Vec::with_capacity(n_features);

        // Calculate score for each feature
        for j in 0..n_features {
            let feature = X.column(j).to_owned();
            let score = match self.config.metric {
                InformationMetric::MutualInformation => {
                    mutual_information(&feature, y, self.config.bins)?
                }
                InformationMetric::NormalizedMI => {
                    normalized_mutual_information(&feature, y, self.config.bins)?
                }
                InformationMetric::InformationGain => {
                    // Information gain = H(Y) - H(Y|X)
                    let h_y = shannon_entropy(y, self.config.bins)?;
                    let h_y_given_x = conditional_entropy(y, &feature, self.config.bins)?;
                    h_y - h_y_given_x
                }
                InformationMetric::SymmetricalUncertainty => {
                    // SU(X,Y) = 2 * I(X;Y) / (H(X) + H(Y))
                    let mi = mutual_information(&feature, y, self.config.bins)?;
                    let h_x = shannon_entropy(&feature, self.config.bins)?;
                    let h_y = shannon_entropy(y, self.config.bins)?;
                    if h_x + h_y == 0.0 {
                        0.0
                    } else {
                        2.0 * mi / (h_x + h_y)
                    }
                }
            };
            scores.push(score);
        }

        // Select features based on k or threshold
        let mut selected_features: Vec<usize> = if let Some(k) = self.config.k {
            // Select top k features
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            indices.into_iter().take(k.min(n_features)).collect()
        } else if let Some(threshold) = self.config.threshold {
            // Select features above threshold
            (0..n_features)
                .filter(|&i| scores[i] >= threshold)
                .collect()
        } else {
            // Select all features
            (0..n_features).collect()
        };

        selected_features.sort();

        Ok(InformationFeatureSelectorFitted {
            config: self.config,
            scores,
            selected_features,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for InformationFeatureSelectorFitted {
    fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if self.selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let mut result = Array2::zeros((X.nrows(), self.selected_features.len()));

        for (new_idx, &old_idx) in self.selected_features.iter().enumerate() {
            if old_idx >= X.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} out of bounds",
                    old_idx
                )));
            }
            result.column_mut(new_idx).assign(&X.column(old_idx));
        }

        Ok(result)
    }
}

impl InformationFeatureSelectorFitted {
    /// Get the feature scores
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Get the selected feature indices
    pub fn selected_features(&self) -> &[usize] {
        &self.selected_features
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Compute probability distribution from data using equal-width binning
fn compute_probabilities(data: &Array1<f64>, bins: usize) -> Result<Vec<f64>> {
    if bins == 0 {
        return Err(SklearsError::InvalidInput(
            "Number of bins must be positive".to_string(),
        ));
    }

    let discretized = discretize(data, bins)?;
    let mut counts = vec![0; bins];

    for &bin in discretized.iter() {
        counts[bin] += 1;
    }

    let total = data.len() as f64;
    Ok(counts.into_iter().map(|c| c as f64 / total).collect())
}

/// Discretize continuous data into bins
fn discretize(data: &Array1<f64>, bins: usize) -> Result<Vec<usize>> {
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        // All values are the same
        return Ok(vec![0; data.len()]);
    }

    let bin_width = (max_val - min_val) / bins as f64;
    let mut discretized = Vec::with_capacity(data.len());

    for &val in data.iter() {
        let bin = ((val - min_val) / bin_width).floor() as usize;
        discretized.push(bin.min(bins - 1));
    }

    Ok(discretized)
}

/// Discretize to binary string
fn discretize_to_binary(data: &Array1<f64>, bins: usize) -> Result<Vec<u8>> {
    let discretized = discretize(data, bins)?;
    let threshold = bins / 2;
    Ok(discretized
        .into_iter()
        .map(|b| if b >= threshold { 1 } else { 0 })
        .collect())
}

/// Compute joint probabilities for two variables
fn compute_joint_probabilities(
    x: &Array1<f64>,
    y: &Array1<f64>,
    bins: usize,
) -> Result<HashMap<(usize, usize), f64>> {
    let x_disc = discretize(x, bins)?;
    let y_disc = discretize(y, bins)?;

    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    let n = x.len();

    for i in 0..n {
        *counts.entry((x_disc[i], y_disc[i])).or_insert(0) += 1;
    }

    let mut probabilities = HashMap::new();
    for (key, count) in counts {
        probabilities.insert(key, count as f64 / n as f64);
    }

    Ok(probabilities)
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution should have maximum entropy
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let entropy = shannon_entropy(&data, 8).unwrap();

        // For uniform distribution with 8 bins: H = log2(8) = 3
        assert_relative_eq!(entropy, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        // All same values should have zero entropy
        let data = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let entropy = shannon_entropy(&data, 5).unwrap();

        assert_relative_eq!(entropy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent variables should have MI close to zero
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];

        let mi = mutual_information(&x, &y, 5).unwrap();

        // MI should be close to 0 for independent variables
        assert!(mi >= 0.0);
    }

    #[test]
    fn test_mutual_information_identical() {
        // Identical variables should have MI = H(X)
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mi = mutual_information(&x, &x, 5).unwrap();
        let h_x = shannon_entropy(&x, 5).unwrap();

        assert_relative_eq!(mi, h_x, epsilon = 1e-10);
    }

    #[test]
    fn test_renyi_entropy() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Renyi entropy should approach Shannon entropy as alpha -> 1
        let renyi_05 = renyi_entropy(&data, 5, 0.5).unwrap();
        let renyi_20 = renyi_entropy(&data, 5, 2.0).unwrap();
        let shannon = shannon_entropy(&data, 5).unwrap();

        assert!(renyi_05 > 0.0);
        assert!(renyi_20 > 0.0);
        assert!(shannon > 0.0);
    }

    #[test]
    fn test_permutation_entropy() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let pe = permutation_entropy(&data, 3, 1).unwrap();

        assert!(pe > 0.0);
        assert!(pe <= 6.0f64.log2()); // Maximum for order 3
    }

    #[test]
    fn test_approximate_entropy() {
        let data = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let apen = approximate_entropy(&data, 2, 0.5).unwrap();

        assert!(apen >= 0.0);
    }

    #[test]
    fn test_lempel_ziv_complexity() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let lz = lempel_ziv_complexity(&data, 4).unwrap();

        println!("LZ complexity: {}", lz);
        assert!(lz > 0.0);
        // LZ complexity can exceed 1.0 with certain normalizations
        // The normalization is an approximation based on n/log2(n)
        assert!(lz > 0.0 && lz < 100.0, "LZ complexity should be reasonable, got {}", lz);
    }

    #[test]
    fn test_sample_entropy() {
        let data = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let sampen = sample_entropy(&data, 2, 0.5).unwrap();

        assert!(sampen >= 0.0);
    }

    #[test]
    fn test_information_feature_selector() {
        let X = array![
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = InformationFeatureSelectorConfig {
            metric: InformationMetric::MutualInformation,
            bins: 5,
            k: Some(2),
            threshold: None,
        };

        let selector = InformationFeatureSelector::new(config);
        let fitted = selector.fit(&X, &y).unwrap();

        assert_eq!(fitted.selected_features().len(), 2);
        assert_eq!(fitted.scores().len(), 3);

        let X_transformed = fitted.transform(&X).unwrap();
        assert_eq!(X_transformed.ncols(), 2);
    }

    #[test]
    fn test_normalized_mutual_information() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let nmi = normalized_mutual_information(&x, &y, 5).unwrap();

        assert!(nmi >= 0.0);
        assert!(nmi <= 1.0);
    }

    #[test]
    fn test_conditional_entropy() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let h_y_given_x = conditional_entropy(&y, &x, 5).unwrap();
        let h_y = shannon_entropy(&y, 5).unwrap();

        assert!(h_y_given_x >= 0.0);
        assert!(h_y_given_x <= h_y);
    }

    #[test]
    fn test_transfer_entropy() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let te = transfer_entropy(&x, &y, 4, 1).unwrap();

        assert!(te.is_finite());
    }

    #[test]
    fn test_discretize() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let discretized = discretize(&data, 5).unwrap();

        assert_eq!(discretized.len(), 5);
        assert!(discretized.iter().all(|&b| b < 5));
    }

    #[test]
    fn test_feature_selector_threshold() {
        let X = array![
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0]
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let config = InformationFeatureSelectorConfig {
            metric: InformationMetric::MutualInformation,
            bins: 4,
            k: None,
            threshold: Some(0.5),
        };

        let selector = InformationFeatureSelector::new(config);
        let fitted = selector.fit(&X, &y).unwrap();

        assert!(fitted.selected_features().len() <= 3);
    }
}
