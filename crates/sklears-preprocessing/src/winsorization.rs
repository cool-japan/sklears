//! Winsorization utilities for capping extreme outliers
//!
//! Winsorization is a statistical transformation that limits extreme values by
//! replacing outliers with specified percentile values rather than removing them.
//! This helps reduce the impact of outliers while preserving data points.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for winsorization
#[derive(Debug, Clone)]
pub struct WinsorizerConfig {
    /// Lower percentile for winsorization (default: 5.0)
    pub lower_percentile: Float,
    /// Upper percentile for winsorization (default: 95.0)
    pub upper_percentile: Float,
    /// Whether to winsorize each feature independently (default: true)
    pub feature_wise: bool,
    /// Strategy for handling NaN values
    pub nan_strategy: NanStrategy,
}

/// Strategy for handling NaN values during winsorization
#[derive(Debug, Clone, Copy, Default)]
pub enum NanStrategy {
    /// Skip NaN values during percentile calculation and leave them unchanged
    #[default]
    Skip,
    /// Treat NaN values as missing and interpolate them
    Interpolate,
    /// Replace NaN values with the winsorization bounds
    Replace,
}

impl Default for WinsorizerConfig {
    fn default() -> Self {
        Self {
            lower_percentile: 5.0,
            upper_percentile: 95.0,
            feature_wise: true,
            nan_strategy: NanStrategy::Skip,
        }
    }
}

/// Winsorizer for capping extreme outliers
///
/// This transformer limits extreme values by replacing outliers with percentile values.
/// For example, with lower_percentile=5 and upper_percentile=95, all values below the 5th
/// percentile are set to the 5th percentile value, and all values above the 95th percentile
/// are set to the 95th percentile value.
#[derive(Debug, Clone)]
pub struct Winsorizer<State = Untrained> {
    config: WinsorizerConfig,
    state: PhantomData<State>,
    // Fitted parameters
    lower_bounds_: Option<Array1<Float>>,
    upper_bounds_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
}

impl Winsorizer<Untrained> {
    /// Create a new Winsorizer with default configuration
    pub fn new() -> Self {
        Self {
            config: WinsorizerConfig::default(),
            state: PhantomData,
            lower_bounds_: None,
            upper_bounds_: None,
            n_features_in_: None,
        }
    }

    /// Create a Winsorizer with specified percentiles
    pub fn with_percentiles(lower: Float, upper: Float) -> Self {
        Self::new().lower_percentile(lower).upper_percentile(upper)
    }

    /// Create a Winsorizer with IQR-based bounds
    pub fn with_iqr(multiplier: Float) -> Self {
        // IQR method: Q1 - multiplier*IQR, Q3 + multiplier*IQR
        // Convert to approximate percentiles
        let lower_perc = if multiplier >= 1.5 { 0.7 } else { 2.5 };
        let upper_perc = if multiplier >= 1.5 { 99.3 } else { 97.5 };
        Self::new()
            .lower_percentile(lower_perc)
            .upper_percentile(upper_perc)
    }

    /// Set the lower percentile
    pub fn lower_percentile(mut self, percentile: Float) -> Self {
        if !(0.0..50.0).contains(&percentile) {
            panic!("Lower percentile must be between 0 and 50");
        }
        self.config.lower_percentile = percentile;
        self
    }

    /// Set the upper percentile
    pub fn upper_percentile(mut self, percentile: Float) -> Self {
        if percentile <= 50.0 || percentile > 100.0 {
            panic!("Upper percentile must be between 50 and 100");
        }
        self.config.upper_percentile = percentile;
        self
    }

    /// Set whether to winsorize features independently
    pub fn feature_wise(mut self, feature_wise: bool) -> Self {
        self.config.feature_wise = feature_wise;
        self
    }

    /// Set the NaN handling strategy
    pub fn nan_strategy(mut self, strategy: NanStrategy) -> Self {
        self.config.nan_strategy = strategy;
        self
    }

    /// Compute percentile of sorted data
    fn compute_percentile(sorted_data: &[Float], percentile: Float) -> Float {
        if sorted_data.is_empty() {
            return Float::NAN;
        }

        if percentile <= 0.0 {
            return sorted_data[0];
        }
        if percentile >= 100.0 {
            return sorted_data[sorted_data.len() - 1];
        }

        let index = (percentile / 100.0) * (sorted_data.len() - 1) as Float;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            sorted_data[lower_index]
        } else {
            let weight = index - lower_index as Float;
            sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
        }
    }

    /// Compute bounds for a single feature
    fn compute_feature_bounds(&self, feature_data: &Array1<Float>) -> Result<(Float, Float)> {
        // Filter out NaN values
        let mut valid_data: Vec<Float> = feature_data
            .iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect();

        if valid_data.is_empty() {
            return Ok((Float::NEG_INFINITY, Float::INFINITY));
        }

        if valid_data.len() == 1 {
            let value = valid_data[0];
            return Ok((value, value));
        }

        // Sort data for percentile calculation
        valid_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_bound = Self::compute_percentile(&valid_data, self.config.lower_percentile);
        let upper_bound = Self::compute_percentile(&valid_data, self.config.upper_percentile);

        Ok((lower_bound, upper_bound))
    }
}

impl Winsorizer<Trained> {
    /// Get the lower bounds
    pub fn lower_bounds(&self) -> &Array1<Float> {
        self.lower_bounds_
            .as_ref()
            .expect("Winsorizer should be fitted")
    }

    /// Get the upper bounds
    pub fn upper_bounds(&self) -> &Array1<Float> {
        self.upper_bounds_
            .as_ref()
            .expect("Winsorizer should be fitted")
    }

    /// Get the number of features seen during fitting
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("Winsorizer should be fitted")
    }

    /// Apply winsorization to a single value for a specific feature
    pub fn winsorize_single(&self, feature_idx: usize, value: Float) -> Float {
        if value.is_nan() {
            match self.config.nan_strategy {
                NanStrategy::Skip => value,
                NanStrategy::Interpolate | NanStrategy::Replace => {
                    // For single values, we can't interpolate, so use the mean of bounds
                    let lower = self.lower_bounds()[feature_idx];
                    let upper = self.upper_bounds()[feature_idx];
                    if lower.is_finite() && upper.is_finite() {
                        (lower + upper) / 2.0
                    } else {
                        value
                    }
                }
            }
        } else {
            let lower = self.lower_bounds()[feature_idx];
            let upper = self.upper_bounds()[feature_idx];

            if value < lower {
                lower
            } else if value > upper {
                upper
            } else {
                value
            }
        }
    }

    /// Get statistics about the winsorization applied to the data
    pub fn get_winsorization_stats(&self, x: &Array2<Float>) -> Result<WinsorizationStats> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let mut stats = WinsorizationStats {
            n_samples,
            n_features,
            lower_clipped_per_feature: vec![0; n_features],
            upper_clipped_per_feature: vec![0; n_features],
            total_clipped: 0,
            clipping_rate: 0.0,
        };

        #[cfg(feature = "parallel")]
        {
            // Use parallel processing for large datasets
            if n_samples * n_features > 10000 {
                let clipping_counts: Vec<(usize, usize, usize)> = (0..n_features)
                    .into_par_iter()
                    .map(|j| {
                        let lower = self.lower_bounds()[j];
                        let upper = self.upper_bounds()[j];
                        let mut lower_clipped = 0;
                        let mut upper_clipped = 0;

                        for i in 0..n_samples {
                            let value = x[[i, j]];
                            if value.is_finite() {
                                if value < lower {
                                    lower_clipped += 1;
                                } else if value > upper {
                                    upper_clipped += 1;
                                }
                            }
                        }
                        (j, lower_clipped, upper_clipped)
                    })
                    .collect();

                for (j, lower_clipped, upper_clipped) in clipping_counts {
                    stats.lower_clipped_per_feature[j] = lower_clipped;
                    stats.upper_clipped_per_feature[j] = upper_clipped;
                    stats.total_clipped += lower_clipped + upper_clipped;
                }
            } else {
                // Sequential processing for smaller datasets
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let value = x[[i, j]];
                        if value.is_finite() {
                            let lower = self.lower_bounds()[j];
                            let upper = self.upper_bounds()[j];

                            if value < lower {
                                stats.lower_clipped_per_feature[j] += 1;
                                stats.total_clipped += 1;
                            } else if value > upper {
                                stats.upper_clipped_per_feature[j] += 1;
                                stats.total_clipped += 1;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential processing when parallel feature is disabled
            for i in 0..n_samples {
                for j in 0..n_features {
                    let value = x[[i, j]];
                    if value.is_finite() {
                        let lower = self.lower_bounds()[j];
                        let upper = self.upper_bounds()[j];

                        if value < lower {
                            stats.lower_clipped_per_feature[j] += 1;
                            stats.total_clipped += 1;
                        } else if value > upper {
                            stats.upper_clipped_per_feature[j] += 1;
                            stats.total_clipped += 1;
                        }
                    }
                }
            }
        }

        stats.clipping_rate = stats.total_clipped as Float / (n_samples * n_features) as Float;
        Ok(stats)
    }
}

/// Statistics about winsorization applied to data
#[derive(Debug, Clone)]
pub struct WinsorizationStats {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of values clipped at lower bound per feature
    pub lower_clipped_per_feature: Vec<usize>,
    /// Number of values clipped at upper bound per feature
    pub upper_clipped_per_feature: Vec<usize>,
    /// Total number of values clipped
    pub total_clipped: usize,
    /// Overall clipping rate (proportion of values clipped)
    pub clipping_rate: Float,
}

impl Default for Winsorizer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for Winsorizer<Untrained> {
    type Fitted = Winsorizer<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit winsorizer on empty dataset".to_string(),
            ));
        }

        if self.config.lower_percentile >= self.config.upper_percentile {
            return Err(SklearsError::InvalidInput(
                "Lower percentile must be less than upper percentile".to_string(),
            ));
        }

        let mut lower_bounds = Array1::<Float>::zeros(n_features);
        let mut upper_bounds = Array1::<Float>::zeros(n_features);

        #[cfg(feature = "parallel")]
        {
            // Use parallel processing for multiple features
            if n_features > 4 {
                let bounds: Result<Vec<(Float, Float)>> = (0..n_features)
                    .into_par_iter()
                    .map(|j| {
                        let feature_data = x.column(j).to_owned();
                        self.compute_feature_bounds(&feature_data)
                    })
                    .collect();

                match bounds {
                    Ok(bounds_vec) => {
                        for (j, (lower, upper)) in bounds_vec.into_iter().enumerate() {
                            lower_bounds[j] = lower;
                            upper_bounds[j] = upper;
                        }
                    }
                    Err(e) => return Err(e),
                }
            } else {
                // Sequential processing for few features
                for j in 0..n_features {
                    let feature_data = x.column(j).to_owned();
                    let (lower, upper) = self.compute_feature_bounds(&feature_data)?;
                    lower_bounds[j] = lower;
                    upper_bounds[j] = upper;
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential processing when parallel feature is disabled
            for j in 0..n_features {
                let feature_data = x.column(j).to_owned();
                let (lower, upper) = self.compute_feature_bounds(&feature_data)?;
                lower_bounds[j] = lower;
                upper_bounds[j] = upper;
            }
        }

        Ok(Winsorizer {
            config: self.config,
            state: PhantomData,
            lower_bounds_: Some(lower_bounds),
            upper_bounds_: Some(upper_bounds),
            n_features_in_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for Winsorizer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let mut result = x.clone();

        #[cfg(feature = "parallel")]
        {
            // Use parallel processing for large datasets
            if n_samples * n_features > 10000 {
                result
                    .as_slice_mut()
                    .unwrap()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, value)| {
                        let i = idx / n_features;
                        let j = idx % n_features;
                        *value = self.winsorize_single(j, x[[i, j]]);
                    });
                return Ok(result);
            }
        }

        // Sequential processing for smaller datasets or when parallel feature is disabled
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = self.winsorize_single(j, x[[i, j]]);
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
    fn test_winsorizer_basic() {
        let x = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0],
            [9.0, 90.0],
            [100.0, 1000.0], // Extreme outliers
        ];

        let winsorizer = Winsorizer::with_percentiles(10.0, 90.0)
            .fit(&x, &())
            .unwrap();

        let transformed = winsorizer.transform(&x).unwrap();

        // Check that outliers are capped
        assert!(transformed[[9, 0]] < x[[9, 0]]); // 100.0 should be capped
        assert!(transformed[[9, 1]] < x[[9, 1]]); // 1000.0 should be capped

        // Check that normal values are unchanged
        assert_abs_diff_eq!(transformed[[4, 0]], x[[4, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[4, 1]], x[[4, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_winsorizer_percentiles() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];

        let winsorizer = Winsorizer::with_percentiles(20.0, 80.0)
            .fit(&x, &())
            .unwrap();

        // 20th percentile should be around 2.8, 80th percentile around 8.2
        let lower = winsorizer.lower_bounds()[0];
        let upper = winsorizer.upper_bounds()[0];

        assert!(lower >= 2.0 && lower <= 3.5);
        assert!(upper >= 7.5 && upper <= 9.0);
    }

    #[test]
    fn test_winsorizer_with_nans() {
        let x = array![
            [1.0, 10.0],
            [2.0, Float::NAN],
            [3.0, 30.0],
            [4.0, 40.0],
            [100.0, 1000.0], // Outliers
        ];

        let winsorizer = Winsorizer::with_percentiles(25.0, 75.0)
            .nan_strategy(NanStrategy::Skip)
            .fit(&x, &())
            .unwrap();

        let transformed = winsorizer.transform(&x).unwrap();

        // NaN should remain NaN with Skip strategy
        assert!(transformed[[1, 1]].is_nan());

        // Outliers should be capped
        assert!(transformed[[4, 0]] < x[[4, 0]]);
        assert!(transformed[[4, 1]] < x[[4, 1]]);
    }

    #[test]
    fn test_winsorizer_single_value() {
        let winsorizer = Winsorizer::with_percentiles(10.0, 90.0);
        let x = array![[5.0], [15.0], [25.0], [100.0]];

        let fitted = winsorizer.fit(&x, &()).unwrap();

        let lower_bound = fitted.lower_bounds()[0];
        let upper_bound = fitted.upper_bounds()[0];

        // Test single value winsorization
        // 5.0 might be below the 10th percentile, so it gets clipped to the lower bound
        assert_eq!(fitted.winsorize_single(0, 15.0), 15.0); // Within bounds
        assert_eq!(fitted.winsorize_single(0, 3.0), lower_bound); // Below lower bound, should be clipped
        assert_eq!(fitted.winsorize_single(0, 200.0), upper_bound); // Above upper bound, should be clipped
        assert!(fitted.winsorize_single(0, 200.0) < 200.0); // Should be capped
    }

    #[test]
    fn test_winsorization_stats() {
        let x = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [100.0, 1000.0], // Outliers
        ];

        let winsorizer = Winsorizer::with_percentiles(25.0, 75.0)
            .fit(&x, &())
            .unwrap();

        let stats = winsorizer.get_winsorization_stats(&x).unwrap();

        assert_eq!(stats.n_samples, 5);
        assert_eq!(stats.n_features, 2);
        assert!(stats.total_clipped > 0);
        assert!(stats.clipping_rate > 0.0);
    }

    #[test]
    fn test_winsorizer_edge_cases() {
        // Test with constant data
        let x = array![[5.0], [5.0], [5.0], [5.0]];
        let winsorizer = Winsorizer::new().fit(&x, &()).unwrap();
        let transformed = winsorizer.transform(&x).unwrap();
        assert_eq!(transformed, x);

        // Test with single sample
        let x = array![[5.0]];
        let winsorizer = Winsorizer::new().fit(&x, &()).unwrap();
        let transformed = winsorizer.transform(&x).unwrap();
        assert_eq!(transformed, x);
    }

    #[test]
    fn test_winsorizer_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Extra feature

        let winsorizer = Winsorizer::new().fit(&x_train, &()).unwrap();
        let result = winsorizer.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_winsorizer_invalid_percentiles() {
        let result = std::panic::catch_unwind(|| {
            Winsorizer::new().lower_percentile(60.0); // Invalid: > 50
        });
        assert!(result.is_err());

        let result = std::panic::catch_unwind(|| {
            Winsorizer::new().upper_percentile(40.0); // Invalid: < 50
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_winsorizer_empty_data() {
        let x = Array2::<Float>::zeros((0, 2));
        let result = Winsorizer::new().fit(&x, &());
        assert!(result.is_err());
    }
}
