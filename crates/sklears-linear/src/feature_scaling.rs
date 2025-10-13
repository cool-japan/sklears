//! Automatic feature scaling for linear models
//!
//! This module provides automatic feature scaling capabilities that can be
//! integrated directly into linear models. It supports various scaling methods
//! including standardization, normalization, robust scaling, and quantile scaling.

use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Feature scaling method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingMethod {
    /// No scaling applied
    None,
    /// Standard scaling (zero mean, unit variance): (x - mean) / std
    StandardScaling,
    /// Min-max scaling to [0, 1]: (x - min) / (max - min)
    MinMaxScaling,
    /// Min-max scaling to custom range: (x - min) / (max - min) * (max_range - min_range) + min_range
    MinMaxCustom { min_range: f64, max_range: f64 },
    /// Robust scaling using median and IQR: (x - median) / IQR
    RobustScaling,
    /// Unit vector scaling: x / ||x||_2
    UnitVectorScaling,
    /// Max scaling: x / max(|x|)
    MaxAbsScaling,
    /// Quantile uniform scaling to [0, 1]
    QuantileUniform,
    /// Power transformer (Box-Cox or Yeo-Johnson)
    PowerTransformer { method: PowerTransformMethod },
}

/// Power transformation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PowerTransformMethod {
    /// Box-Cox transformation (requires positive data)
    BoxCox,
    /// Yeo-Johnson transformation (works with any data)
    YeoJohnson,
}

/// Feature scaling configuration
#[derive(Debug, Clone)]
pub struct FeatureScalingConfig {
    /// Scaling method to use
    pub method: ScalingMethod,
    /// Whether to scale features independently
    pub per_feature: bool,
    /// Whether to copy data before scaling
    pub copy: bool,
    /// Clip outliers to this many standard deviations (for robust methods)
    pub clip_outliers: Option<f64>,
    /// Minimum variance threshold for features
    pub variance_threshold: f64,
    /// Whether to handle missing values
    pub handle_missing: bool,
}

impl Default for FeatureScalingConfig {
    fn default() -> Self {
        Self {
            method: ScalingMethod::StandardScaling,
            per_feature: true,
            copy: true,
            clip_outliers: None,
            variance_threshold: 1e-8,
            handle_missing: false,
        }
    }
}

/// Statistics for feature scaling
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Feature means
    pub means: Vec<f64>,
    /// Feature standard deviations
    pub stds: Vec<f64>,
    /// Feature minimums
    pub mins: Vec<f64>,
    /// Feature maximums
    pub maxs: Vec<f64>,
    /// Feature medians
    pub medians: Vec<f64>,
    /// Feature quantiles (for quantile scaling)
    pub quantiles: Vec<Vec<f64>>,
    /// Feature scales (computed scaling factors)
    pub scales: Vec<f64>,
    /// Feature centers (computed centering values)
    pub centers: Vec<f64>,
    /// Number of features
    pub n_features: usize,
    /// Number of samples used for statistics
    pub n_samples: usize,
}

impl FeatureStats {
    fn new(n_features: usize) -> Self {
        Self {
            means: vec![0.0; n_features],
            stds: vec![1.0; n_features],
            mins: vec![f64::INFINITY; n_features],
            maxs: vec![f64::NEG_INFINITY; n_features],
            medians: vec![0.0; n_features],
            quantiles: vec![vec![]; n_features],
            scales: vec![1.0; n_features],
            centers: vec![0.0; n_features],
            n_features,
            n_samples: 0,
        }
    }
}

/// Feature scaler that can be integrated into linear models
#[derive(Debug, Clone)]
pub struct FeatureScaler {
    /// Scaling configuration
    pub config: FeatureScalingConfig,
    /// Computed feature statistics
    pub stats: Option<FeatureStats>,
    /// Whether the scaler has been fitted
    pub fitted: bool,
    /// Mapping of feature indices to exclude from scaling
    pub exclude_features: HashMap<usize, bool>,
}

impl FeatureScaler {
    /// Create a new feature scaler
    pub fn new(config: FeatureScalingConfig) -> Self {
        Self {
            config,
            stats: None,
            fitted: false,
            exclude_features: HashMap::new(),
        }
    }

    /// Create a standard scaler (most common)
    pub fn standard() -> Self {
        Self::new(FeatureScalingConfig {
            method: ScalingMethod::StandardScaling,
            ..Default::default()
        })
    }

    /// Create a min-max scaler
    pub fn min_max() -> Self {
        Self::new(FeatureScalingConfig {
            method: ScalingMethod::MinMaxScaling,
            ..Default::default()
        })
    }

    /// Create a robust scaler
    pub fn robust() -> Self {
        Self::new(FeatureScalingConfig {
            method: ScalingMethod::RobustScaling,
            ..Default::default()
        })
    }

    /// Exclude specific features from scaling
    pub fn exclude_feature(&mut self, feature_index: usize) -> &mut Self {
        self.exclude_features.insert(feature_index, true);
        self
    }

    /// Fit the scaler to training data
    pub fn fit(&mut self, x: &[Vec<f64>]) -> Result<(), SklearsError> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let n_samples = x.len();
        let n_features = x[0].len();

        // Validate input dimensions
        for (i, row) in x.iter().enumerate() {
            if row.len() != n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Inconsistent number of features at row {}: expected {}, got {}",
                    i,
                    n_features,
                    row.len()
                )));
            }
        }

        let mut stats = FeatureStats::new(n_features);
        stats.n_samples = n_samples;

        // Compute basic statistics
        self.compute_basic_stats(&mut stats, x)?;

        // Compute method-specific statistics and scaling parameters
        match self.config.method {
            ScalingMethod::None => {
                // No scaling needed
            }
            ScalingMethod::StandardScaling => {
                self.compute_standard_scaling(&mut stats)?;
            }
            ScalingMethod::MinMaxScaling | ScalingMethod::MinMaxCustom { .. } => {
                self.compute_minmax_scaling(&mut stats)?;
            }
            ScalingMethod::RobustScaling => {
                self.compute_robust_scaling(&mut stats, x)?;
            }
            ScalingMethod::UnitVectorScaling => {
                self.compute_unit_vector_scaling(&mut stats, x)?;
            }
            ScalingMethod::MaxAbsScaling => {
                self.compute_maxabs_scaling(&mut stats)?;
            }
            ScalingMethod::QuantileUniform => {
                self.compute_quantile_scaling(&mut stats, x)?;
            }
            ScalingMethod::PowerTransformer { .. } => {
                self.compute_power_transform_params(&mut stats, x)?;
            }
        }

        self.stats = Some(stats);
        self.fitted = true;
        Ok(())
    }

    /// Transform data using fitted scaler
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let stats = self.stats.as_ref().unwrap();

        if x.is_empty() {
            return Ok(vec![]);
        }

        let n_features = x[0].len();
        if n_features != stats.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features mismatch: expected {}, got {}",
                stats.n_features, n_features
            )));
        }

        let mut result = if self.config.copy {
            x.to_vec()
        } else {
            return Err(SklearsError::InvalidInput(
                "In-place transformation not supported for this method".to_string(),
            ));
        };

        self.apply_transformation(&mut result, stats)?;
        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Inverse transform (undo scaling)
    pub fn inverse_transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let stats = self.stats.as_ref().unwrap();
        let mut result = x.to_vec();

        self.apply_inverse_transformation(&mut result, stats)?;
        Ok(result)
    }

    /// Get feature statistics
    pub fn get_stats(&self) -> Option<&FeatureStats> {
        self.stats.as_ref()
    }

    fn compute_basic_stats(
        &self,
        stats: &mut FeatureStats,
        x: &[Vec<f64>],
    ) -> Result<(), SklearsError> {
        let n_samples = x.len();
        let n_features = stats.n_features;

        // Initialize accumulators
        let mut sums = vec![0.0; n_features];
        let mut sum_squares = vec![0.0; n_features];

        // Compute means and track min/max
        for row in x {
            for (j, &value) in row.iter().enumerate() {
                if self.config.handle_missing && value.is_nan() {
                    continue;
                }

                sums[j] += value;
                sum_squares[j] += value * value;
                stats.mins[j] = stats.mins[j].min(value);
                stats.maxs[j] = stats.maxs[j].max(value);
            }
        }

        // Compute means and standard deviations
        for j in 0..n_features {
            stats.means[j] = sums[j] / n_samples as f64;

            let variance = (sum_squares[j] / n_samples as f64) - (stats.means[j] * stats.means[j]);
            stats.stds[j] = variance.max(self.config.variance_threshold).sqrt();
        }

        Ok(())
    }

    fn compute_standard_scaling(&self, stats: &mut FeatureStats) -> Result<(), SklearsError> {
        for j in 0..stats.n_features {
            if self.exclude_features.contains_key(&j) {
                stats.centers[j] = 0.0;
                stats.scales[j] = 1.0;
            } else {
                stats.centers[j] = stats.means[j];
                stats.scales[j] = stats.stds[j];
            }
        }
        Ok(())
    }

    fn compute_minmax_scaling(&self, stats: &mut FeatureStats) -> Result<(), SklearsError> {
        let (min_range, max_range) = match self.config.method {
            ScalingMethod::MinMaxCustom {
                min_range,
                max_range,
            } => (min_range, max_range),
            _ => (0.0, 1.0),
        };

        for j in 0..stats.n_features {
            if self.exclude_features.contains_key(&j) {
                stats.centers[j] = 0.0;
                stats.scales[j] = 1.0;
            } else {
                let range = stats.maxs[j] - stats.mins[j];
                if range.abs() < self.config.variance_threshold {
                    stats.centers[j] = 0.0;
                    stats.scales[j] = 1.0;
                } else {
                    stats.centers[j] = stats.mins[j];
                    stats.scales[j] = range / (max_range - min_range);
                }
            }
        }
        Ok(())
    }

    fn compute_robust_scaling(
        &self,
        stats: &mut FeatureStats,
        x: &[Vec<f64>],
    ) -> Result<(), SklearsError> {
        for j in 0..stats.n_features {
            if self.exclude_features.contains_key(&j) {
                stats.centers[j] = 0.0;
                stats.scales[j] = 1.0;
                continue;
            }

            // Collect feature values for quantile computation
            let mut feature_values: Vec<f64> = x.iter().map(|row| row[j]).collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Compute median (50th percentile)
            let median_idx = feature_values.len() / 2;
            stats.medians[j] = if feature_values.len() % 2 == 0 {
                (feature_values[median_idx - 1] + feature_values[median_idx]) / 2.0
            } else {
                feature_values[median_idx]
            };

            // Compute IQR (75th - 25th percentile)
            let q1_idx = feature_values.len() / 4;
            let q3_idx = 3 * feature_values.len() / 4;
            let q1 = feature_values[q1_idx];
            let q3 = feature_values[q3_idx];
            let iqr = q3 - q1;

            stats.centers[j] = stats.medians[j];
            stats.scales[j] = if iqr.abs() < self.config.variance_threshold {
                1.0
            } else {
                iqr
            };
        }
        Ok(())
    }

    fn compute_unit_vector_scaling(
        &self,
        stats: &mut FeatureStats,
        x: &[Vec<f64>],
    ) -> Result<(), SklearsError> {
        // For unit vector scaling, we compute the L2 norm of each sample
        for j in 0..stats.n_features {
            stats.centers[j] = 0.0;
            stats.scales[j] = 1.0; // Will be computed per-sample during transformation
        }
        Ok(())
    }

    fn compute_maxabs_scaling(&self, stats: &mut FeatureStats) -> Result<(), SklearsError> {
        for j in 0..stats.n_features {
            if self.exclude_features.contains_key(&j) {
                stats.centers[j] = 0.0;
                stats.scales[j] = 1.0;
            } else {
                let max_abs = stats.maxs[j].abs().max(stats.mins[j].abs());
                stats.centers[j] = 0.0;
                stats.scales[j] = if max_abs < self.config.variance_threshold {
                    1.0
                } else {
                    max_abs
                };
            }
        }
        Ok(())
    }

    fn compute_quantile_scaling(
        &self,
        stats: &mut FeatureStats,
        x: &[Vec<f64>],
    ) -> Result<(), SklearsError> {
        let num_quantiles = 1000; // Use 1000 quantiles for smooth transformation

        for j in 0..stats.n_features {
            if self.exclude_features.contains_key(&j) {
                stats.quantiles[j] = vec![];
                continue;
            }

            let mut feature_values: Vec<f64> = x.iter().map(|row| row[j]).collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut quantiles = Vec::with_capacity(num_quantiles + 1);
            for i in 0..=num_quantiles {
                let percentile = i as f64 / num_quantiles as f64;
                let idx = (percentile * (feature_values.len() - 1) as f64) as usize;
                quantiles.push(feature_values[idx]);
            }

            stats.quantiles[j] = quantiles;
        }
        Ok(())
    }

    fn compute_power_transform_params(
        &self,
        stats: &mut FeatureStats,
        x: &[Vec<f64>],
    ) -> Result<(), SklearsError> {
        // For power transformations, we need to find optimal lambda parameters
        // This is a simplified implementation - in practice, maximum likelihood estimation would be used
        for j in 0..stats.n_features {
            if self.exclude_features.contains_key(&j) {
                continue;
            }

            let feature_values: Vec<f64> = x.iter().map(|row| row[j]).collect();

            match self.config.method {
                ScalingMethod::PowerTransformer {
                    method: PowerTransformMethod::BoxCox,
                } => {
                    // Check if all values are positive (required for Box-Cox)
                    if feature_values.iter().any(|&v| v <= 0.0) {
                        return Err(SklearsError::InvalidInput(format!(
                            "Feature {} contains non-positive values, cannot apply Box-Cox transformation",
                            j
                        )));
                    }
                    // Simplified: use lambda = 0 (log transform) as default
                    stats.scales[j] = 0.0; // lambda parameter
                }
                ScalingMethod::PowerTransformer {
                    method: PowerTransformMethod::YeoJohnson,
                } => {
                    // Simplified: use lambda = 1 (identity) as default
                    stats.scales[j] = 1.0; // lambda parameter
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    fn apply_transformation(
        &self,
        data: &mut [Vec<f64>],
        stats: &FeatureStats,
    ) -> Result<(), SklearsError> {
        match self.config.method {
            ScalingMethod::None => {
                // No transformation needed
            }
            ScalingMethod::StandardScaling
            | ScalingMethod::MinMaxScaling
            | ScalingMethod::MinMaxCustom { .. }
            | ScalingMethod::RobustScaling
            | ScalingMethod::MaxAbsScaling => {
                self.apply_center_scale_transform(data, stats)?;
            }
            ScalingMethod::UnitVectorScaling => {
                self.apply_unit_vector_transform(data)?;
            }
            ScalingMethod::QuantileUniform => {
                self.apply_quantile_transform(data, stats)?;
            }
            ScalingMethod::PowerTransformer { method } => {
                self.apply_power_transform(data, stats, method)?;
            }
        }
        Ok(())
    }

    fn apply_center_scale_transform(
        &self,
        data: &mut [Vec<f64>],
        stats: &FeatureStats,
    ) -> Result<(), SklearsError> {
        for row in data.iter_mut() {
            for (j, value) in row.iter_mut().enumerate() {
                if !self.exclude_features.contains_key(&j) {
                    *value = (*value - stats.centers[j]) / stats.scales[j];

                    // Apply outlier clipping if specified
                    if let Some(clip_threshold) = self.config.clip_outliers {
                        *value = value.max(-clip_threshold).min(clip_threshold);
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_unit_vector_transform(&self, data: &mut [Vec<f64>]) -> Result<(), SklearsError> {
        for row in data.iter_mut() {
            // Compute L2 norm of the row
            let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if norm > self.config.variance_threshold {
                for value in row.iter_mut() {
                    *value /= norm;
                }
            }
        }
        Ok(())
    }

    fn apply_quantile_transform(
        &self,
        data: &mut [Vec<f64>],
        stats: &FeatureStats,
    ) -> Result<(), SklearsError> {
        for row in data.iter_mut() {
            for (j, value) in row.iter_mut().enumerate() {
                if self.exclude_features.contains_key(&j) || stats.quantiles[j].is_empty() {
                    continue;
                }

                let quantiles = &stats.quantiles[j];

                // Find position in quantile distribution
                let pos = quantiles
                    .binary_search_by(|&q| q.partial_cmp(value).unwrap())
                    .unwrap_or_else(|i| i);

                *value = pos as f64 / (quantiles.len() - 1) as f64;
            }
        }
        Ok(())
    }

    fn apply_power_transform(
        &self,
        data: &mut [Vec<f64>],
        stats: &FeatureStats,
        method: PowerTransformMethod,
    ) -> Result<(), SklearsError> {
        for row in data.iter_mut() {
            for (j, value) in row.iter_mut().enumerate() {
                if self.exclude_features.contains_key(&j) {
                    continue;
                }

                let lambda = stats.scales[j];

                match method {
                    PowerTransformMethod::BoxCox => {
                        if *value <= 0.0 {
                            return Err(SklearsError::NumericalError(
                                "Box-Cox transformation requires positive values".to_string(),
                            ));
                        }
                        *value = if lambda.abs() < 1e-8 {
                            value.ln()
                        } else {
                            (value.powf(lambda) - 1.0) / lambda
                        };
                    }
                    PowerTransformMethod::YeoJohnson => {
                        *value = if *value >= 0.0 {
                            if lambda.abs() < 1e-8 {
                                value.ln()
                            } else {
                                ((*value + 1.0).powf(lambda) - 1.0) / lambda
                            }
                        } else {
                            if (lambda - 2.0).abs() < 1e-8 {
                                (-*value).ln()
                            } else {
                                -((-*value + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
                            }
                        };
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_inverse_transformation(
        &self,
        data: &mut [Vec<f64>],
        stats: &FeatureStats,
    ) -> Result<(), SklearsError> {
        match self.config.method {
            ScalingMethod::None => {
                // No transformation to reverse
            }
            ScalingMethod::StandardScaling
            | ScalingMethod::MinMaxScaling
            | ScalingMethod::MinMaxCustom { .. }
            | ScalingMethod::RobustScaling
            | ScalingMethod::MaxAbsScaling => {
                self.apply_inverse_center_scale_transform(data, stats)?;
            }
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Inverse transformation not implemented for this scaling method".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn apply_inverse_center_scale_transform(
        &self,
        data: &mut [Vec<f64>],
        stats: &FeatureStats,
    ) -> Result<(), SklearsError> {
        for row in data.iter_mut() {
            for (j, value) in row.iter_mut().enumerate() {
                if !self.exclude_features.contains_key(&j) {
                    *value = *value * stats.scales[j] + stats.centers[j];
                }
            }
        }
        Ok(())
    }
}

/// Builder for feature scaler configuration
pub struct FeatureScalerBuilder {
    config: FeatureScalingConfig,
}

impl FeatureScalerBuilder {
    pub fn new() -> Self {
        Self {
            config: FeatureScalingConfig::default(),
        }
    }

    pub fn method(mut self, method: ScalingMethod) -> Self {
        self.config.method = method;
        self
    }

    pub fn per_feature(mut self, per_feature: bool) -> Self {
        self.config.per_feature = per_feature;
        self
    }

    pub fn copy(mut self, copy: bool) -> Self {
        self.config.copy = copy;
        self
    }

    pub fn clip_outliers(mut self, clip_outliers: Option<f64>) -> Self {
        self.config.clip_outliers = clip_outliers;
        self
    }

    pub fn variance_threshold(mut self, threshold: f64) -> Self {
        self.config.variance_threshold = threshold;
        self
    }

    pub fn handle_missing(mut self, handle_missing: bool) -> Self {
        self.config.handle_missing = handle_missing;
        self
    }

    pub fn build(self) -> FeatureScaler {
        FeatureScaler::new(self.config)
    }
}

impl Default for FeatureScalerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_standard_scaling() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];

        let mut scaler = FeatureScaler::standard();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Check that means are approximately 0 and stds are approximately 1
        let means: Vec<f64> = (0..2)
            .map(|j| scaled.iter().map(|row| row[j]).sum::<f64>() / scaled.len() as f64)
            .collect();

        for mean in means {
            assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        }

        // Test inverse transformation
        let inverse = scaler.inverse_transform(&scaled).unwrap();
        for (i, row) in inverse.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                assert_relative_eq!(value, data[i][j], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_minmax_scaling() {
        let data = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ];

        let mut scaler = FeatureScaler::min_max();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Check that values are in [0, 1] range
        for row in &scaled {
            for &value in row {
                assert!(value >= 0.0 && value <= 1.0);
            }
        }

        // Check that min values become 0 and max values become 1
        assert_relative_eq!(scaled[0][0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[3][0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[0][1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(scaled[3][1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robust_scaling() {
        let data = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![100.0, 1000.0], // Outliers
        ];

        let mut scaler = FeatureScaler::robust();
        let scaled = scaler.fit_transform(&data).unwrap();

        // Robust scaling should be less affected by outliers
        assert!(scaled.len() == data.len());

        let stats = scaler.get_stats().unwrap();
        // Median should be 3.0 for first feature and 30.0 for second
        assert_relative_eq!(stats.medians[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.medians[1], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unit_vector_scaling() {
        let data = vec![
            vec![3.0, 4.0], // norm = 5
            vec![6.0, 8.0], // norm = 10
        ];

        let config = FeatureScalingConfig {
            method: ScalingMethod::UnitVectorScaling,
            ..Default::default()
        };
        let mut scaler = FeatureScaler::new(config);
        let scaled = scaler.fit_transform(&data).unwrap();

        // Check that each row has unit norm
        for row in &scaled {
            let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_feature_exclusion() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];

        let mut scaler = FeatureScaler::standard();
        scaler.exclude_feature(1); // Exclude second feature from scaling

        let scaled = scaler.fit_transform(&data).unwrap();

        // Second feature should remain unchanged
        for (i, row) in scaled.iter().enumerate() {
            assert_relative_eq!(row[1], data[i][1], epsilon = 1e-10);
        }

        // First and third features should be scaled
        let first_feature_mean: f64 =
            scaled.iter().map(|row| row[0]).sum::<f64>() / scaled.len() as f64;
        assert_relative_eq!(first_feature_mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scaler_builder() {
        let scaler = FeatureScalerBuilder::new()
            .method(ScalingMethod::MinMaxCustom {
                min_range: -1.0,
                max_range: 1.0,
            })
            .clip_outliers(Some(3.0))
            .variance_threshold(1e-6)
            .build();

        assert!(matches!(
            scaler.config.method,
            ScalingMethod::MinMaxCustom { .. }
        ));
        assert_eq!(scaler.config.clip_outliers, Some(3.0));
        assert_eq!(scaler.config.variance_threshold, 1e-6);
    }

    #[test]
    fn test_empty_data_handling() {
        let data: Vec<Vec<f64>> = vec![];
        let mut scaler = FeatureScaler::standard();

        let result = scaler.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_inconsistent_dimensions() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0, 6.0], // Different number of features
        ];

        let mut scaler = FeatureScaler::standard();
        let result = scaler.fit(&data);
        assert!(result.is_err());
    }
}
