//! Statistical feature extraction
//!
//! This module provides transformers for extracting statistical features from data
//! including moments, distribution properties, and entropy-based measures.

use crate::*;
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, ArrayView1};
use sklears_core::prelude::SklearsError;

/// Statistical Moments Extractor
///
/// Extracts statistical moments and moment-based features from data including
/// raw moments, central moments, cumulants, and higher-order statistics.
///
/// # Parameters
///
/// * `max_order` - Maximum order of moments to compute
/// * `normalize` - Whether to normalize moments
/// * `include_cumulants` - Whether to include cumulant features
/// * `robust` - Whether to use robust estimators
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::statistical_features::StatisticalMomentsExtractor;
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let extractor = StatisticalMomentsExtractor::new()
///     .max_order(4)
///     .normalize(true)
///     .include_cumulants(true);
///
/// let features = extractor.extract_features(&data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StatisticalMomentsExtractor {
    max_order: usize,
    normalize: bool,
    include_cumulants: bool,
    robust: bool,
}

impl StatisticalMomentsExtractor {
    /// Create a new StatisticalMomentsExtractor
    pub fn new() -> Self {
        Self {
            max_order: 4,
            normalize: true,
            include_cumulants: true,
            robust: false,
        }
    }

    /// Set the maximum order of moments
    pub fn max_order(mut self, max_order: usize) -> Self {
        self.max_order = max_order;
        self
    }

    /// Set whether to normalize moments
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set whether to include cumulant features
    pub fn include_cumulants(mut self, include_cumulants: bool) -> Self {
        self.include_cumulants = include_cumulants;
        self
    }

    /// Set whether to use robust estimators
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Extract statistical moments from data
    pub fn extract_features(&self, data: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n = data.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }
        if n < 2 {
            return Err(SklearsError::InvalidInput(
                "Data must have at least 2 points".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Basic statistics
        let mean = if self.robust {
            self.robust_mean(data)
        } else {
            data.sum() / n as Float
        };
        features.push(mean);

        let variance = if self.robust {
            self.robust_variance(data, mean)
        } else {
            data.mapv(|x| (x - mean).powi(2)).sum() / (n - 1) as Float
        };
        features.push(variance);

        let std_dev = variance.sqrt();
        features.push(std_dev);

        // Raw moments
        for order in 1..=self.max_order {
            let raw_moment = data.mapv(|x| x.powi(order as i32)).sum() / n as Float;
            features.push(raw_moment);
        }

        // Central moments
        for order in 2..=self.max_order {
            let central_moment = data.mapv(|x| (x - mean).powi(order as i32)).sum() / n as Float;
            features.push(central_moment);
        }

        // Standardized moments (normalized by standard deviation)
        if self.normalize && std_dev > 1e-10 {
            for order in 3..=self.max_order {
                let standardized_moment = data
                    .mapv(|x| ((x - mean) / std_dev).powi(order as i32))
                    .sum()
                    / n as Float;
                features.push(standardized_moment);
            }
        }

        // Cumulants (if requested)
        if self.include_cumulants {
            let cumulants = self.compute_cumulants(data, mean, variance)?;
            features.extend_from_slice(&cumulants);
        }

        // Shape statistics
        let skewness = if std_dev > 1e-10 {
            data.mapv(|x| ((x - mean) / std_dev).powi(3)).sum() / n as Float
        } else {
            0.0
        };
        features.push(skewness);

        let kurtosis = if std_dev > 1e-10 {
            data.mapv(|x| ((x - mean) / std_dev).powi(4)).sum() / n as Float - 3.0
        // Excess kurtosis
        } else {
            0.0
        };
        features.push(kurtosis);

        // Additional moment-based features
        let coefficient_of_variation = if mean.abs() > 1e-10 {
            std_dev / mean.abs()
        } else {
            0.0
        };
        features.push(coefficient_of_variation);

        Ok(Array1::from_vec(features))
    }

    fn robust_mean(&self, data: &ArrayView1<Float>) -> Float {
        // Use median as robust mean estimator
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_data.len();

        if n % 2 == 1 {
            sorted_data[n / 2]
        } else {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
        }
    }

    fn robust_variance(&self, data: &ArrayView1<Float>, center: Float) -> Float {
        // Use median absolute deviation (MAD) scaled to match variance
        let mad_values: Vec<Float> = data.iter().map(|&x| (x - center).abs()).collect();
        let mut sorted_mad = mad_values;
        sorted_mad.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_mad.len();

        let mad = if n % 2 == 1 {
            sorted_mad[n / 2]
        } else {
            (sorted_mad[n / 2 - 1] + sorted_mad[n / 2]) / 2.0
        };

        // Scale MAD to match standard deviation for normal distribution
        (1.4826 * mad).powi(2)
    }

    fn compute_cumulants(
        &self,
        data: &ArrayView1<Float>,
        mean: Float,
        variance: Float,
    ) -> SklResult<Vec<Float>> {
        let n = data.len() as Float;
        let mut cumulants = Vec::new();

        // First cumulant is the mean
        cumulants.push(mean);

        // Second cumulant is the variance
        cumulants.push(variance);

        // Third cumulant (related to skewness)
        let third_moment = data.mapv(|x| (x - mean).powi(3)).sum() / n;
        cumulants.push(third_moment);

        // Fourth cumulant (related to kurtosis)
        let fourth_moment = data.mapv(|x| (x - mean).powi(4)).sum() / n;
        let fourth_cumulant = fourth_moment - 3.0 * variance.powi(2);
        cumulants.push(fourth_cumulant);

        Ok(cumulants)
    }
}

impl Default for StatisticalMomentsExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Distribution-Based Feature Extractor
///
/// Extracts features based on the empirical distribution of data including
/// quantiles, percentiles, histogram features, and distribution shape measures.
///
/// # Parameters
///
/// * `n_quantiles` - Number of quantiles to compute
/// * `n_bins` - Number of histogram bins
/// * `include_shape_features` - Whether to include distribution shape features
/// * `density_estimation` - Method for density estimation ("histogram", "kde")
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::statistical_features::DistributionFeaturesExtractor;
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let extractor = DistributionFeaturesExtractor::new()
///     .n_quantiles(5)
///     .n_bins(10)
///     .include_shape_features(true);
///
/// let features = extractor.extract_features(&data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DistributionFeaturesExtractor {
    n_quantiles: usize,
    n_bins: usize,
    include_shape_features: bool,
    density_estimation: String,
}

impl DistributionFeaturesExtractor {
    /// Create a new DistributionFeaturesExtractor
    pub fn new() -> Self {
        Self {
            n_quantiles: 5,
            n_bins: 10,
            include_shape_features: true,
            density_estimation: "histogram".to_string(),
        }
    }

    /// Set the number of quantiles
    pub fn n_quantiles(mut self, n_quantiles: usize) -> Self {
        self.n_quantiles = n_quantiles;
        self
    }

    /// Set the number of histogram bins
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set whether to include shape features
    pub fn include_shape_features(mut self, include_shape_features: bool) -> Self {
        self.include_shape_features = include_shape_features;
        self
    }

    /// Set the density estimation method
    pub fn density_estimation(mut self, method: String) -> Self {
        self.density_estimation = method;
        self
    }

    /// Extract distribution-based features from data
    pub fn extract_features(&self, data: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n = data.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Sort data for quantile computation
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Quantile features
        for i in 1..=self.n_quantiles {
            let quantile = i as Float / (self.n_quantiles + 1) as Float;
            let quantile_value = self.compute_quantile(&sorted_data, quantile);
            features.push(quantile_value);
        }

        // Additional percentiles (5th, 25th, 50th, 75th, 95th)
        let percentiles = [0.05, 0.25, 0.5, 0.75, 0.95];
        for &percentile in &percentiles {
            let percentile_value = self.compute_quantile(&sorted_data, percentile);
            features.push(percentile_value);
        }

        // Interquartile range
        let q25 = self.compute_quantile(&sorted_data, 0.25);
        let q75 = self.compute_quantile(&sorted_data, 0.75);
        let iqr = q75 - q25;
        features.push(iqr);

        // Range
        let range = sorted_data[n - 1] - sorted_data[0];
        features.push(range);

        // Histogram features
        let histogram = self.compute_histogram(&sorted_data)?;
        features.extend_from_slice(&histogram);

        // Shape features
        if self.include_shape_features {
            let shape_features = self.compute_shape_features(&sorted_data)?;
            features.extend_from_slice(&shape_features);
        }

        Ok(Array1::from_vec(features))
    }

    fn compute_quantile(&self, sorted_data: &[Float], quantile: Float) -> Float {
        let n = sorted_data.len();
        let index = quantile * (n - 1) as Float;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            sorted_data[lower_index]
        } else {
            let weight = index - lower_index as Float;
            sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
        }
    }

    fn compute_histogram(&self, sorted_data: &[Float]) -> SklResult<Vec<Float>> {
        if sorted_data.is_empty() {
            return Ok(vec![0.0; self.n_bins]);
        }

        let min_val = sorted_data[0];
        let max_val = sorted_data[sorted_data.len() - 1];

        if (max_val - min_val).abs() < 1e-10 {
            // All values are the same
            let mut hist = vec![0.0; self.n_bins];
            hist[0] = sorted_data.len() as Float;
            return Ok(hist);
        }

        let bin_width = (max_val - min_val) / self.n_bins as Float;
        let mut histogram = vec![0.0; self.n_bins];

        for &value in sorted_data {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(self.n_bins - 1); // Handle edge case
            histogram[bin_index] += 1.0;
        }

        // Normalize to get density
        let total_count = sorted_data.len() as Float;
        for count in &mut histogram {
            *count /= total_count;
        }

        Ok(histogram)
    }

    fn compute_shape_features(&self, sorted_data: &[Float]) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();

        // Trimmed mean (10% trim)
        let trim_amount = (sorted_data.len() as Float * 0.1).floor() as usize;
        if trim_amount > 0 && trim_amount * 2 < sorted_data.len() {
            let trimmed_slice = &sorted_data[trim_amount..sorted_data.len() - trim_amount];
            let trimmed_mean = trimmed_slice.iter().sum::<Float>() / trimmed_slice.len() as Float;
            features.push(trimmed_mean);
        } else {
            features.push(sorted_data.iter().sum::<Float>() / sorted_data.len() as Float);
        }

        // Mode approximation (most frequent bin in histogram)
        let histogram = self.compute_histogram(sorted_data)?;
        let mode_bin = histogram
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let min_val = sorted_data[0];
        let max_val = sorted_data[sorted_data.len() - 1];
        let bin_width = (max_val - min_val) / self.n_bins as Float;
        let mode_estimate = min_val + (mode_bin as Float + 0.5) * bin_width;
        features.push(mode_estimate);

        // Peak count (local maxima in histogram)
        let mut peak_count = 0;
        for i in 1..histogram.len() - 1 {
            if histogram[i] > histogram[i - 1] && histogram[i] > histogram[i + 1] {
                peak_count += 1;
            }
        }
        features.push(peak_count as Float);

        // Tail weight (sum of probabilities in tails)
        let tail_size = (self.n_bins as Float * 0.1).max(1.0) as usize;
        let left_tail: Float = histogram[0..tail_size].iter().sum();
        let right_tail: Float = histogram[histogram.len() - tail_size..].iter().sum();
        features.push(left_tail + right_tail);

        Ok(features)
    }
}

impl Default for DistributionFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy-Based Feature Extractor
///
/// Extracts entropy-based features from data including Shannon entropy,
/// Rényi entropy, sample entropy, and approximate entropy measures.
///
/// # Parameters
///
/// * `n_bins` - Number of bins for histogram-based entropy
/// * `include_conditional_entropy` - Whether to include conditional entropy features
/// * `window_size` - Window size for sample entropy computation
/// * `tolerance` - Tolerance for approximate entropy
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::statistical_features::EntropyFeaturesExtractor;
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let extractor = EntropyFeaturesExtractor::new()
///     .n_bins(8)
///     .window_size(2)
///     .tolerance(0.1);
///
/// let features = extractor.extract_features(&data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct EntropyFeaturesExtractor {
    n_bins: usize,
    include_conditional_entropy: bool,
    window_size: usize,
    tolerance: Float,
}

impl EntropyFeaturesExtractor {
    /// Create a new EntropyFeaturesExtractor
    pub fn new() -> Self {
        Self {
            n_bins: 10,
            include_conditional_entropy: true,
            window_size: 2,
            tolerance: 0.2,
        }
    }

    /// Set the number of bins for histogram-based entropy
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set whether to include conditional entropy features
    pub fn include_conditional_entropy(mut self, include_conditional_entropy: bool) -> Self {
        self.include_conditional_entropy = include_conditional_entropy;
        self
    }

    /// Set the window size for sample entropy
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the tolerance for approximate entropy
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Extract entropy-based features from data
    pub fn extract_features(&self, data: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n = data.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Shannon entropy from histogram
        let shannon_entropy = self.compute_shannon_entropy(data)?;
        features.push(shannon_entropy);

        // Rényi entropy (order 2)
        let renyi_entropy = self.compute_renyi_entropy(data, 2.0)?;
        features.push(renyi_entropy);

        // Sample entropy
        if n >= 2 * self.window_size {
            let sample_entropy = self.compute_sample_entropy(data)?;
            features.push(sample_entropy);
        } else {
            features.push(0.0); // Not enough data for sample entropy
        }

        // Approximate entropy
        if n > self.window_size {
            let approximate_entropy = self.compute_approximate_entropy(data)?;
            features.push(approximate_entropy);
        } else {
            features.push(0.0); // Not enough data for approximate entropy
        }

        // Differential entropy estimate
        let differential_entropy = self.compute_differential_entropy(data)?;
        features.push(differential_entropy);

        // Conditional entropy (if requested)
        if self.include_conditional_entropy && n >= 2 {
            let conditional_entropy = self.compute_conditional_entropy(data)?;
            features.push(conditional_entropy);
        }

        Ok(Array1::from_vec(features))
    }

    fn compute_shannon_entropy(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        let histogram = self.compute_normalized_histogram(data)?;

        let mut entropy = 0.0;
        for &prob in &histogram {
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }

        Ok(entropy)
    }

    fn compute_renyi_entropy(&self, data: &ArrayView1<Float>, alpha: Float) -> SklResult<Float> {
        if (alpha - 1.0).abs() < 1e-10 {
            return self.compute_shannon_entropy(data); // Rényi entropy converges to Shannon entropy as α → 1
        }

        let histogram = self.compute_normalized_histogram(data)?;

        let mut sum = 0.0;
        for &prob in &histogram {
            if prob > 1e-10 {
                sum += prob.powf(alpha);
            }
        }

        let entropy = if sum > 1e-10 {
            sum.ln() / (1.0 - alpha)
        } else {
            0.0
        };

        Ok(entropy)
    }

    fn compute_sample_entropy(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        let n = data.len();
        let m = self.window_size;
        let r = self.tolerance;

        if n < m + 1 {
            return Ok(0.0);
        }

        let mut a = 0;
        let mut b = 0;

        for i in 0..(n - m) {
            for j in 0..(n - m) {
                if i != j {
                    let mut match_m = true;
                    let mut match_m1 = true;

                    // Check for matches of length m
                    for k in 0..m {
                        if (data[i + k] - data[j + k]).abs() > r {
                            match_m = false;
                            match_m1 = false;
                            break;
                        }
                    }

                    if match_m {
                        b += 1;

                        // Check for matches of length m+1
                        if i < n - m && j < n - m {
                            if (data[i + m] - data[j + m]).abs() <= r {
                                a += 1;
                            } else {
                                match_m1 = false;
                            }
                        } else {
                            match_m1 = false;
                        }
                    }
                }
            }
        }

        let sample_entropy = if a > 0 && b > 0 {
            -(a as Float / b as Float).ln()
        } else {
            std::f64::consts::E // Natural logarithm base for undefined case
        };

        Ok(sample_entropy)
    }

    fn compute_approximate_entropy(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        let n = data.len();
        let m = self.window_size;
        let r = self.tolerance;

        if n < m + 1 {
            return Ok(0.0);
        }

        let phi_m = self.compute_phi(data, m, r)?;
        let phi_m1 = self.compute_phi(data, m + 1, r)?;

        Ok(phi_m - phi_m1)
    }

    fn compute_phi(&self, data: &ArrayView1<Float>, m: usize, r: Float) -> SklResult<Float> {
        let n = data.len();
        let mut phi = 0.0;

        for i in 0..(n - m + 1) {
            let mut pattern_count = 0;

            for j in 0..(n - m + 1) {
                let mut matches = true;
                for k in 0..m {
                    if (data[i + k] - data[j + k]).abs() > r {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    pattern_count += 1;
                }
            }

            if pattern_count > 0 {
                phi += (pattern_count as Float / (n - m + 1) as Float).ln();
            }
        }

        phi /= (n - m + 1) as Float;
        Ok(phi)
    }

    fn compute_differential_entropy(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        // Estimate using Gaussian assumption: h(X) = 0.5 * log(2πeσ²)
        let mean = data.sum() / data.len() as Float;
        let variance = data.mapv(|x| (x - mean).powi(2)).sum() / (data.len() - 1) as Float;

        let differential_entropy =
            0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln();
        Ok(differential_entropy)
    }

    fn compute_conditional_entropy(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        // Approximate conditional entropy H(X_{t+1} | X_t)
        let n = data.len();
        if n < 2 {
            return Ok(0.0);
        }

        // Create pairs (X_t, X_{t+1})
        let pairs: Vec<(Float, Float)> = (0..n - 1).map(|i| (data[i], data[i + 1])).collect();

        // Use histogram approach for simplicity
        let joint_entropy = self.compute_joint_entropy(&pairs)?;
        let marginal_entropy = self.compute_shannon_entropy(&data.slice(s![0..n - 1]))?;

        Ok(joint_entropy - marginal_entropy)
    }

    fn compute_joint_entropy(&self, pairs: &[(Float, Float)]) -> SklResult<Float> {
        // Simple binning approach for joint entropy
        let n_bins = self.n_bins;

        if pairs.is_empty() {
            return Ok(0.0);
        }

        // Find ranges for both dimensions
        let x_min = pairs
            .iter()
            .map(|(x, _)| *x)
            .fold(std::f64::INFINITY, |a, b| a.min(b));
        let x_max = pairs
            .iter()
            .map(|(x, _)| *x)
            .fold(std::f64::NEG_INFINITY, |a, b| a.max(b));
        let y_min = pairs
            .iter()
            .map(|(_, y)| *y)
            .fold(std::f64::INFINITY, |a, b| a.min(b));
        let y_max = pairs
            .iter()
            .map(|(_, y)| *y)
            .fold(std::f64::NEG_INFINITY, |a, b| a.max(b));

        let x_range = if (x_max - x_min).abs() < 1e-10 {
            1.0
        } else {
            x_max - x_min
        };
        let y_range = if (y_max - y_min).abs() < 1e-10 {
            1.0
        } else {
            y_max - y_min
        };

        let mut joint_hist = vec![vec![0.0; n_bins]; n_bins];

        for &(x, y) in pairs {
            let x_bin = ((x - x_min) / x_range * n_bins as Float).floor() as usize;
            let y_bin = ((y - y_min) / y_range * n_bins as Float).floor() as usize;
            let x_bin = x_bin.min(n_bins - 1);
            let y_bin = y_bin.min(n_bins - 1);
            joint_hist[x_bin][y_bin] += 1.0;
        }

        // Normalize and compute entropy
        let total = pairs.len() as Float;
        let mut entropy = 0.0;

        for i in 0..n_bins {
            for j in 0..n_bins {
                let prob = joint_hist[i][j] / total;
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
        }

        Ok(entropy)
    }

    fn compute_normalized_histogram(&self, data: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let n = data.len();
        if n == 0 {
            return Ok(vec![0.0; self.n_bins]);
        }

        let min_val = data
            .iter()
            .cloned()
            .fold(std::f64::INFINITY, |a, b| a.min(b));
        let max_val = data
            .iter()
            .cloned()
            .fold(std::f64::NEG_INFINITY, |a, b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            // All values are the same
            let mut hist = vec![0.0; self.n_bins];
            hist[0] = 1.0;
            return Ok(hist);
        }

        let bin_width = (max_val - min_val) / self.n_bins as Float;
        let mut histogram = vec![0.0; self.n_bins];

        for &value in data.iter() {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(self.n_bins - 1);
            histogram[bin_index] += 1.0;
        }

        // Normalize to get probabilities
        let total_count = n as Float;
        for count in &mut histogram {
            *count /= total_count;
        }

        Ok(histogram)
    }
}

impl Default for EntropyFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}
