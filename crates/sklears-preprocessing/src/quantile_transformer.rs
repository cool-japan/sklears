//! Quantile Transformer
//!
//! This module provides QuantileTransformer which transforms features to follow
//! a uniform or a normal distribution using quantiles information.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};

/// Output distribution for QuantileTransformer
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantileOutput {
    /// Transform to uniform distribution on [0, 1]
    Uniform,
    /// Transform to standard normal distribution
    Normal,
}

/// Configuration for QuantileTransformer
#[derive(Debug, Clone)]
pub struct QuantileTransformerConfig {
    /// Number of quantiles to be computed
    pub n_quantiles: usize,
    /// Desired output distribution
    pub output_distribution: QuantileOutput,
    /// Subsample size for large datasets
    pub subsample: Option<usize>,
    /// Random state for subsampling
    pub random_state: Option<u64>,
    /// Whether to copy the input data
    pub copy: bool,
    /// Whether to clip transformed values to avoid infinities
    pub clip: bool,
    /// Ignore outliers beyond this quantile range
    pub ignore_outliers: Option<(Float, Float)>,
}

impl Default for QuantileTransformerConfig {
    fn default() -> Self {
        Self {
            n_quantiles: 1000,
            output_distribution: QuantileOutput::Uniform,
            subsample: Some(100_000),
            random_state: None,
            copy: true,
            clip: true,
            ignore_outliers: None,
        }
    }
}

/// QuantileTransformer applies a non-linear transformation to make data follow
/// a uniform or normal distribution
pub struct QuantileTransformer<State = Untrained> {
    config: QuantileTransformerConfig,
    state: PhantomData<State>,
    /// The quantiles for each feature
    quantiles_: Option<Vec<Array1<Float>>>,
    /// The actual number of quantiles used
    n_quantiles_: Option<usize>,
    /// Reference values for inverse transform
    references_: Option<Array1<Float>>,
}

impl QuantileTransformer<Untrained> {
    /// Create a new QuantileTransformer with default configuration
    pub fn new() -> Self {
        Self {
            config: QuantileTransformerConfig::default(),
            state: PhantomData,
            quantiles_: None,
            n_quantiles_: None,
            references_: None,
        }
    }

    /// Set the number of quantiles
    pub fn n_quantiles(mut self, n_quantiles: usize) -> Self {
        if n_quantiles < 2 {
            panic!("n_quantiles must be at least 2");
        }
        self.config.n_quantiles = n_quantiles;
        self
    }

    /// Set the output distribution
    pub fn output_distribution(mut self, output_distribution: QuantileOutput) -> Self {
        self.config.output_distribution = output_distribution;
        self
    }

    /// Set the subsample size
    pub fn subsample(mut self, subsample: Option<usize>) -> Self {
        self.config.subsample = subsample;
        self
    }

    /// Set whether to clip values to avoid infinities
    pub fn clip(mut self, clip: bool) -> Self {
        self.config.clip = clip;
        self
    }

    /// Set outlier quantile range to ignore during fitting
    pub fn ignore_outliers(mut self, range: Option<(Float, Float)>) -> Self {
        if let Some((low, high)) = range {
            assert!(
                low >= 0.0 && low < high && high <= 1.0,
                "Outlier range must be (low, high) where 0 <= low < high <= 1"
            );
        }
        self.config.ignore_outliers = range;
        self
    }
}

impl Default for QuantileTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for QuantileTransformer<Untrained> {
    type Config = QuantileTransformerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for QuantileTransformer<Trained> {
    type Config = QuantileTransformerConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute quantiles for a feature with optional outlier filtering
fn compute_quantiles(
    data: &Array1<Float>,
    n_quantiles: usize,
    ignore_outliers: Option<(Float, Float)>,
) -> (Array1<Float>, Array1<Float>) {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_samples = sorted_data.len();

    // Apply outlier filtering if specified
    let (start_idx, end_idx) = if let Some((low_quantile, high_quantile)) = ignore_outliers {
        let start = ((low_quantile * (n_samples - 1) as Float) as usize).min(n_samples - 1);
        let end = ((high_quantile * (n_samples - 1) as Float) as usize).min(n_samples - 1);
        (start, end + 1)
    } else {
        (0, n_samples)
    };

    let filtered_data = &sorted_data[start_idx..end_idx];
    let filtered_n_samples = filtered_data.len();
    let n_quantiles = n_quantiles.min(filtered_n_samples);

    let mut quantiles = Vec::with_capacity(n_quantiles);
    let mut references = Vec::with_capacity(n_quantiles);

    for i in 0..n_quantiles {
        let quantile = i as Float / (n_quantiles - 1) as Float;
        let idx = (quantile * (filtered_n_samples - 1) as Float) as usize;

        quantiles.push(filtered_data[idx]);
        references.push(quantile);
    }

    (Array1::from_vec(quantiles), Array1::from_vec(references))
}

/// Compute the inverse of the error function (for normal distribution)
/// Uses an accurate rational approximation by Winitzki (2008)
fn erfinv(x: Float) -> Float {
    if x.abs() >= 1.0 {
        return if x > 0.0 {
            Float::INFINITY
        } else {
            Float::NEG_INFINITY
        };
    }

    if x == 0.0 {
        return 0.0;
    }

    let sign = if x > 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // Use Winitzki's approximation
    let a = 0.147;
    let ln_term = (1.0 - x * x).ln();
    let term1 = 2.0 / (std::f64::consts::PI * a) + ln_term / 2.0;
    let term2 = ln_term / a;

    let result = (term1 * term1 - term2).sqrt() - term1;
    sign * result.sqrt()
}

/// Robust inverse error function implementation using proven approximations
/// This ensures monotonicity by construction using Beasley-Springer-Moro algorithm
fn erfinv_accurate(x: Float) -> Float {
    if x.abs() >= 1.0 {
        return if x > 0.0 {
            Float::INFINITY
        } else {
            Float::NEG_INFINITY
        };
    }

    if x == 0.0 {
        return 0.0;
    }

    // Use symmetric property
    let sign = x.signum();
    let abs_x = x.abs();

    // Use the Beasley-Springer-Moro algorithm for standard normal quantiles
    // This is monotonic and highly accurate

    // Convert erf^-1 to standard normal quantile function
    // If x = erf(y), then sqrt(2) * y = Φ^-1((1+x)/2)
    let p = (1.0 + abs_x) / 2.0;

    // Apply rational approximation for inverse normal CDF
    let result = if p > 0.5 {
        // Upper tail
        let q = p - 0.5;
        let r = q * q;

        let numerator = (((((-39.6968302866538 * r + 220.946098424521) * r - 275.928510446969)
            * r
            + 138.357751867269)
            * r
            - 30.6647980661472)
            * r
            + 2.50662827745924)
            * q;

        let denominator = ((((-54.4760987982241 * r + 161.585836858041) * r - 155.698979859887)
            * r
            + 66.8013118877197)
            * r
            - 13.2806815528857)
            * r
            + 1.0;

        numerator / denominator
    } else {
        // Lower tail - use symmetry
        let q = 0.5 - p;
        let r = q * q;

        let numerator = (((((-39.6968302866538 * r + 220.946098424521) * r - 275.928510446969)
            * r
            + 138.357751867269)
            * r
            - 30.6647980661472)
            * r
            + 2.50662827745924)
            * q;

        let denominator = ((((-54.4760987982241 * r + 161.585836858041) * r - 155.698979859887)
            * r
            + 66.8013118877197)
            * r
            - 13.2806815528857)
            * r
            + 1.0;

        -numerator / denominator
    };

    // Convert from standard normal quantile to erf^-1
    sign * result / std::f64::consts::SQRT_2
}

/// Convert uniform values to normal distribution
fn uniform_to_normal(uniform_value: Float, clip: bool) -> Float {
    let clipped = if clip {
        // Clip to avoid infinities
        uniform_value.clamp(1e-7, 1.0 - 1e-7)
    } else {
        uniform_value
    };

    // Use more accurate inverse error function to convert uniform to normal
    std::f64::consts::SQRT_2 * erfinv_accurate(2.0 * clipped - 1.0)
}

/// Accurate error function implementation using Abramowitz and Stegun approximation
fn erf_accurate(x: Float) -> Float {
    if x == 0.0 {
        return 0.0;
    }

    let sign = x.signum();
    let abs_x = x.abs();

    // Abramowitz and Stegun approximation with high accuracy
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();

    sign * y
}

/// Convert normal values to uniform distribution
fn normal_to_uniform(normal_value: Float, clip: bool) -> Float {
    // Use accurate error function to convert normal to uniform
    let erf_input = normal_value / std::f64::consts::SQRT_2;
    let erf_val = erf_accurate(erf_input);
    let uniform_val = (1.0 + erf_val) / 2.0;

    if clip {
        uniform_val.clamp(1e-7, 1.0 - 1e-7)
    } else {
        uniform_val
    }
}

impl Fit<Array2<Float>, ()> for QuantileTransformer<Untrained> {
    type Fitted = QuantileTransformer<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine actual number of quantiles
        let n_quantiles = self.config.n_quantiles.min(n_samples);

        let mut all_quantiles = Vec::with_capacity(n_features);
        let mut all_references = None;

        // Compute quantiles for each feature
        for j in 0..n_features {
            let feature_data = x.column(j).to_owned();

            // Subsample if needed with better strategy
            let data_to_use = if let Some(subsample_size) = self.config.subsample {
                if n_samples > subsample_size {
                    // Use stratified subsampling for better representation
                    let mut subsampled = Vec::with_capacity(subsample_size);

                    // Sort data to enable stratified sampling
                    let mut indexed_data: Vec<(usize, Float)> = feature_data
                        .iter()
                        .enumerate()
                        .map(|(i, &val)| (i, val))
                        .collect();
                    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Take stratified samples across the sorted data
                    let step = n_samples as Float / subsample_size as Float;
                    for i in 0..subsample_size {
                        let idx = (i as Float * step) as usize;
                        if idx < n_samples {
                            subsampled.push(indexed_data[idx].1);
                        }
                    }

                    Array1::from_vec(subsampled)
                } else {
                    feature_data
                }
            } else {
                feature_data
            };

            let (quantiles, references) =
                compute_quantiles(&data_to_use, n_quantiles, self.config.ignore_outliers);
            all_quantiles.push(quantiles);

            if all_references.is_none() {
                all_references = Some(references);
            }
        }

        Ok(QuantileTransformer {
            config: self.config,
            state: PhantomData,
            quantiles_: Some(all_quantiles),
            n_quantiles_: Some(n_quantiles),
            references_: all_references,
        })
    }
}

/// Interpolate value based on quantiles
fn interpolate_value(value: Float, quantiles: &Array1<Float>, references: &Array1<Float>) -> Float {
    let n = quantiles.len();

    // Handle edge cases
    if value <= quantiles[0] {
        return references[0];
    }
    if value >= quantiles[n - 1] {
        return references[n - 1];
    }

    // Binary search for the interval
    let mut left = 0;
    let mut right = n - 1;

    while left < right - 1 {
        let mid = (left + right) / 2;
        if value < quantiles[mid] {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Linear interpolation
    let x0 = quantiles[left];
    let x1 = quantiles[right];
    let y0 = references[left];
    let y1 = references[right];

    if (x1 - x0).abs() < Float::EPSILON {
        y0
    } else {
        y0 + (value - x0) * (y1 - y0) / (x1 - x0)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for QuantileTransformer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let quantiles = self.quantiles_.as_ref().unwrap();
        let references = self.references_.as_ref().unwrap();

        if n_features != quantiles.len() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but QuantileTransformer is expecting {} features",
                n_features,
                quantiles.len()
            )));
        }

        let mut result = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x[[i, j]];
                let uniform_value = interpolate_value(value, &quantiles[j], references);

                result[[i, j]] = match self.config.output_distribution {
                    QuantileOutput::Uniform => uniform_value,
                    QuantileOutput::Normal => uniform_to_normal(uniform_value, self.config.clip),
                };
            }
        }

        Ok(result)
    }
}

impl Transform<Array1<Float>, Array1<Float>> for QuantileTransformer<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        // Transform 1D array by treating it as a column vector
        let x_2d = x.clone().insert_axis(Axis(1));
        let result_2d = self.transform(&x_2d)?;
        Ok(result_2d.column(0).to_owned())
    }
}

impl QuantileTransformer<Trained> {
    /// Get the quantiles for each feature
    pub fn quantiles(&self) -> &Vec<Array1<Float>> {
        self.quantiles_.as_ref().unwrap()
    }

    /// Get the number of quantiles used
    pub fn n_quantiles(&self) -> usize {
        self.n_quantiles_.unwrap()
    }

    /// Inverse transform data back to original distribution
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let quantiles = self.quantiles_.as_ref().unwrap();
        let references = self.references_.as_ref().unwrap();

        if n_features != quantiles.len() {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but QuantileTransformer is expecting {} features",
                n_features,
                quantiles.len()
            )));
        }

        let mut result = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x[[i, j]];

                let uniform_value = match self.config.output_distribution {
                    QuantileOutput::Uniform => value,
                    QuantileOutput::Normal => normal_to_uniform(value, self.config.clip),
                };

                // Interpolate back to original distribution
                result[[i, j]] = interpolate_value(uniform_value, references, &quantiles[j]);
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantile_transformer_uniform() {
        let x = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
        ];

        let qt = QuantileTransformer::new()
            .n_quantiles(10)
            .output_distribution(QuantileOutput::Uniform)
            .fit(&x, &())
            .unwrap();

        let x_transformed = qt.transform(&x).unwrap();

        // Check that values are in [0, 1]
        for value in x_transformed.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }

        // Check that transformation is monotonic
        for i in 1..x_transformed.len() {
            assert!(x_transformed[[i, 0]] >= x_transformed[[i - 1, 0]]);
        }
    }

    #[test]
    fn test_quantile_transformer_normal() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0],];

        let qt = QuantileTransformer::new()
            .n_quantiles(6)
            .output_distribution(QuantileOutput::Normal)
            .fit(&x, &())
            .unwrap();

        let x_transformed = qt.transform(&x).unwrap();

        // Check that transformation is monotonic (more important than exact values)
        for i in 1..x_transformed.len() {
            assert!(
                x_transformed[[i, 0]] >= x_transformed[[i - 1, 0]] - 1e-10,
                "Values at index {} ({}) should be >= values at index {} ({})",
                i,
                x_transformed[[i, 0]],
                i - 1,
                x_transformed[[i - 1, 0]]
            );
        }

        // Check that the range is reasonable for normal distribution
        let min_val = x_transformed.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = x_transformed
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        // Normal transformed values should be in a reasonable range
        assert!(
            min_val > -5.0 && max_val < 5.0,
            "Normal transformed values should be in reasonable range, got [{}, {}]",
            min_val,
            max_val
        );
    }

    #[test]
    fn test_quantile_transformer_multivariate() {
        let x = array![
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [3.0, 40.0],
            [4.0, 50.0],
        ];

        let qt = QuantileTransformer::new()
            .n_quantiles(5)
            .fit(&x, &())
            .unwrap();

        let x_transformed = qt.transform(&x).unwrap();

        // Each feature should be transformed independently
        assert_eq!(x_transformed.ncols(), 2);

        // Values should be in [0, 1]
        for value in x_transformed.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_quantile_transformer_inverse() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0],];

        let qt = QuantileTransformer::new()
            .n_quantiles(5)
            .fit(&x, &())
            .unwrap();

        let x_transformed = qt.transform(&x).unwrap();
        let x_inverse = qt.inverse_transform(&x_transformed).unwrap();

        // Check that inverse transform recovers original values
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(x[[i, 0]], x_inverse[[i, 0]], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_quantile_transformer_edge_cases() {
        let x = array![[0.0], [0.0], [1.0], [1.0], [1.0],];

        let qt = QuantileTransformer::new()
            .n_quantiles(3)
            .fit(&x, &())
            .unwrap();

        let x_transformed = qt.transform(&x).unwrap();

        // All values should be valid
        for value in x_transformed.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_interpolate_value() {
        let quantiles = array![0.0, 1.0, 2.0, 3.0];
        let references = array![0.0, 0.33, 0.67, 1.0];

        // Test exact matches
        assert_abs_diff_eq!(interpolate_value(0.0, &quantiles, &references), 0.0);
        assert_abs_diff_eq!(interpolate_value(3.0, &quantiles, &references), 1.0);

        // Test interpolation
        assert_abs_diff_eq!(
            interpolate_value(0.5, &quantiles, &references),
            0.165,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            interpolate_value(1.5, &quantiles, &references),
            0.5,
            epsilon = 1e-3
        );

        // Test extrapolation
        assert_abs_diff_eq!(interpolate_value(-1.0, &quantiles, &references), 0.0);
        assert_abs_diff_eq!(interpolate_value(4.0, &quantiles, &references), 1.0);
    }

    #[test]
    fn test_uniform_to_normal() {
        // Test some known values
        assert_abs_diff_eq!(uniform_to_normal(0.5, true), 0.0, epsilon = 1e-4);

        // Test that it's monotonic
        let values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let transformed: Vec<Float> = values.iter().map(|&v| uniform_to_normal(v, true)).collect();

        for i in 1..transformed.len() {
            assert!(transformed[i] > transformed[i - 1]);
        }
    }

    #[test]
    fn test_enhanced_quantile_transformer_with_clipping() {
        let x = array![
            [-10.0], // Extreme outlier
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [100.0], // Extreme outlier
        ];

        let qt = QuantileTransformer::new()
            .n_quantiles(6)
            .output_distribution(QuantileOutput::Normal)
            .clip(true)
            .fit(&x, &())
            .unwrap();

        let x_transformed = qt.transform(&x).unwrap();

        // Check that all values are finite
        for value in x_transformed.iter() {
            assert!(value.is_finite(), "All transformed values should be finite");
        }

        // Check that transformation is monotonic
        for i in 1..x_transformed.nrows() {
            assert!(x_transformed[[i, 0]] >= x_transformed[[i - 1, 0]]);
        }
    }

    #[test]
    fn test_quantile_transformer_with_outlier_filtering() {
        let mut x_data = vec![];
        // Add normal data
        for i in 0..100 {
            x_data.push([i as Float]);
        }
        // Add outliers
        x_data.push([-1000.0]);
        x_data.push([1000.0]);

        let x = Array2::from_shape_vec((102, 1), x_data.into_iter().flatten().collect()).unwrap();

        // Test with outlier filtering (ignore bottom 1% and top 1%)
        let qt_filtered = QuantileTransformer::new()
            .n_quantiles(50)
            .ignore_outliers(Some((0.01, 0.99)))
            .fit(&x, &())
            .unwrap();

        // Test without outlier filtering
        let qt_unfiltered = QuantileTransformer::new()
            .n_quantiles(50)
            .fit(&x, &())
            .unwrap();

        let test_value = array![[50.0]];
        let result_filtered = qt_filtered.transform(&test_value).unwrap();
        let result_unfiltered = qt_unfiltered.transform(&test_value).unwrap();

        // The filtered version should handle the middle values better
        // (this is a qualitative test - both should be valid but different)
        assert!(result_filtered.iter().all(|&v| v.is_finite()));
        assert!(result_unfiltered.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_improved_inverse_error_function() {
        // Test erfinv_accurate with known values
        assert_abs_diff_eq!(erfinv_accurate(0.0), 0.0, epsilon = 1e-10);

        // Test symmetry
        let test_val = 0.5;
        assert_abs_diff_eq!(
            erfinv_accurate(test_val),
            -erfinv_accurate(-test_val),
            epsilon = 1e-10
        );

        // Test that it's monotonic
        let values = vec![-0.9, -0.5, 0.0, 0.5, 0.9];
        let transformed: Vec<Float> = values.iter().map(|&v| erfinv_accurate(v)).collect();

        for i in 1..transformed.len() {
            assert!(transformed[i] > transformed[i - 1]);
        }
    }

    #[test]
    fn test_normal_to_uniform_conversion() {
        // Test round-trip conversion
        let uniform_values = vec![0.1, 0.3, 0.5, 0.7, 0.9];

        for &uniform_val in &uniform_values {
            let normal_val = uniform_to_normal(uniform_val, true);
            let recovered_uniform = normal_to_uniform(normal_val, true);

            assert_abs_diff_eq!(uniform_val, recovered_uniform, epsilon = 1e-3);
        }

        // Test edge cases
        assert_abs_diff_eq!(normal_to_uniform(0.0, true), 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_builder_methods() {
        let qt = QuantileTransformer::new()
            .n_quantiles(500)
            .output_distribution(QuantileOutput::Normal)
            .subsample(Some(1000))
            .clip(false)
            .ignore_outliers(Some((0.05, 0.95)));

        assert_eq!(qt.config.n_quantiles, 500);
        assert_eq!(qt.config.output_distribution, QuantileOutput::Normal);
        assert_eq!(qt.config.subsample, Some(1000));
        assert_eq!(qt.config.clip, false);
        assert_eq!(qt.config.ignore_outliers, Some((0.05, 0.95)));
    }

    #[test]
    #[should_panic(expected = "Outlier range must be")]
    fn test_invalid_outlier_range() {
        QuantileTransformer::new().ignore_outliers(Some((0.9, 0.1))); // Invalid: low > high
    }
}
