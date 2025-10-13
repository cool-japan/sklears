//! Main outlier detection implementation
//!
//! This module provides the core OutlierDetector implementation with support
//! for multiple detection methods.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::core::*;
use super::simd_operations::*;

/// Univariate outlier detector for identifying anomalous data points
///
/// This detector provides several methods for identifying outliers in univariate data:
/// - Z-score: Points with |z-score| > threshold are considered outliers
/// - Modified Z-score: Uses median and MAD instead of mean and std for robustness
/// - IQR: Points outside Q1 - k*IQR or Q3 + k*IQR are considered outliers
/// - Percentile: Points outside specified percentile bounds are considered outliers
#[derive(Debug, Clone)]
pub struct OutlierDetector<State = Untrained> {
    config: OutlierDetectorConfig,
    state: PhantomData<State>,
    // Fitted parameters
    n_features_in_: Option<usize>,
    statistics_: Option<OutlierStatistics>,
    feature_params_: Option<Vec<FeatureOutlierParams>>,
    multivariate_params_: Option<MultivariateOutlierParams>,
}

impl OutlierDetector<Untrained> {
    /// Create a new OutlierDetector
    pub fn new() -> Self {
        Self {
            config: OutlierDetectorConfig::default(),
            state: PhantomData,
            n_features_in_: None,
            statistics_: None,
            feature_params_: None,
            multivariate_params_: None,
        }
    }

    /// Create Z-score based outlier detector
    pub fn z_score(threshold: Float) -> Self {
        Self {
            config: OutlierDetectorConfig {
                method: OutlierDetectionMethod::ZScore,
                threshold,
                ..Default::default()
            },
            state: PhantomData,
            n_features_in_: None,
            statistics_: None,
            feature_params_: None,
            multivariate_params_: None,
        }
    }

    /// Create IQR-based outlier detector
    pub fn iqr(multiplier: Float) -> Self {
        Self {
            config: OutlierDetectorConfig {
                method: OutlierDetectionMethod::IQR,
                iqr_multiplier: multiplier,
                ..Default::default()
            },
            state: PhantomData,
            n_features_in_: None,
            statistics_: None,
            feature_params_: None,
            multivariate_params_: None,
        }
    }

    /// Create Mahalanobis distance based outlier detector
    pub fn mahalanobis_distance(confidence_level: Float) -> Self {
        Self {
            config: OutlierDetectorConfig {
                method: OutlierDetectionMethod::MahalanobisDistance,
                confidence_level,
                ..Default::default()
            },
            state: PhantomData,
            n_features_in_: None,
            statistics_: None,
            feature_params_: None,
            multivariate_params_: None,
        }
    }

    /// Set the detection method
    pub fn method(mut self, method: OutlierDetectionMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the threshold
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Compute statistics for a single feature
    fn compute_feature_statistics(
        &self,
        feature_data: &Array1<Float>,
    ) -> Result<FeatureOutlierParams> {
        let mut params = FeatureOutlierParams::default();

        match self.config.method {
            OutlierDetectionMethod::ZScore => {
                let mean = feature_data.mean().unwrap_or(0.0);
                let std = feature_data.std(0.0);
                params.mean = Some(mean);
                params.std = Some(std);
            }
            OutlierDetectionMethod::ModifiedZScore => {
                let median = self.compute_median(feature_data);
                let mad = self.compute_mad(feature_data, median);
                params.median = Some(median);
                params.mad = Some(mad);
            }
            OutlierDetectionMethod::IQR => {
                let (q1, q3, iqr) = self.compute_quartiles(feature_data);
                params.q1 = Some(q1);
                params.q3 = Some(q3);
                params.iqr = Some(iqr);
                params.lower_bound = Some(q1 - self.config.iqr_multiplier * iqr);
                params.upper_bound = Some(q3 + self.config.iqr_multiplier * iqr);
            }
            OutlierDetectionMethod::Percentile => {
                let (lower, upper) = self.compute_percentile_bounds(feature_data);
                params.lower_percentile_value = Some(lower);
                params.upper_percentile_value = Some(upper);
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "method".to_string(),
                    reason: "Unsupported method for univariate detection".to_string(),
                });
            }
        }

        Ok(params)
    }

    /// Compute median of data
    fn compute_median(&self, data: &Array1<Float>) -> Float {
        let mut sorted: Vec<Float> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Compute Median Absolute Deviation
    fn compute_mad(&self, data: &Array1<Float>, median: Float) -> Float {
        let deviations: Vec<Float> = data.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_deviations.len();
        if n % 2 == 0 {
            (sorted_deviations[n / 2 - 1] + sorted_deviations[n / 2]) / 2.0
        } else {
            sorted_deviations[n / 2]
        }
    }

    /// Compute quartiles and IQR
    fn compute_quartiles(&self, data: &Array1<Float>) -> (Float, Float, Float) {
        let mut sorted: Vec<Float> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();

        let q1_idx = (0.25 * (n - 1) as Float) as usize;
        let q3_idx = (0.75 * (n - 1) as Float) as usize;

        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        (q1, q3, iqr)
    }

    /// Compute percentile bounds
    fn compute_percentile_bounds(&self, data: &Array1<Float>) -> (Float, Float) {
        let mut sorted: Vec<Float> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();

        let lower_idx = ((self.config.lower_percentile / 100.0) * (n - 1) as Float) as usize;
        let upper_idx = ((self.config.upper_percentile / 100.0) * (n - 1) as Float) as usize;

        (sorted[lower_idx], sorted[upper_idx])
    }

    /// Compute multivariate statistics for Mahalanobis distance
    fn compute_multivariate_statistics(
        &self,
        x: &Array2<Float>,
    ) -> Result<MultivariateOutlierParams> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Compute mean vector
        let mean = x.mean_axis(Axis(0)).unwrap().to_vec();

        // Compute covariance matrix
        let mut covariance = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in i..n_features {
                let mut cov_sum = 0.0;
                for k in 0..n_samples {
                    cov_sum += (x[[k, i]] - mean[i]) * (x[[k, j]] - mean[j]);
                }
                let cov_val = cov_sum / (n_samples - 1) as Float;
                covariance[i][j] = cov_val;
                covariance[j][i] = cov_val; // Symmetric
            }
        }

        // Compute inverse covariance matrix (simplified)
        let inv_covariance = self.matrix_inverse(&covariance)?;

        // Compute threshold using chi-squared distribution
        let threshold = self.chi_squared_threshold(n_features, self.config.confidence_level);

        Ok(MultivariateOutlierParams {
            mean,
            covariance,
            inv_covariance,
            threshold,
        })
    }

    /// Simple matrix inversion (for demo - would use proper LAPACK in practice)
    fn matrix_inverse(&self, matrix: &[Vec<Float>]) -> Result<Vec<Vec<Float>>> {
        let n = matrix.len();
        if n == 1 {
            if matrix[0][0].abs() < 1e-12 {
                return Err(SklearsError::NumericalError("Singular matrix".to_string()));
            }
            return Ok(vec![vec![1.0 / matrix[0][0]]]);
        }

        // For simplicity, return identity matrix for now
        // In practice, would use proper matrix inversion
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            inv[i][i] = 1.0;
        }
        Ok(inv)
    }

    /// Chi-squared threshold approximation
    fn chi_squared_threshold(&self, degrees_of_freedom: usize, confidence_level: Float) -> Float {
        // Simple approximation - in practice would use proper chi-squared quantile
        let df = degrees_of_freedom as Float;
        match confidence_level {
            x if x >= 0.99 => df + 3.0 * df.sqrt(),
            x if x >= 0.95 => df + 2.0 * df.sqrt(),
            x if x >= 0.90 => df + 1.5 * df.sqrt(),
            _ => df + df.sqrt(),
        }
    }
}

impl OutlierDetector<Trained> {
    /// Get fitted statistics
    pub fn statistics(&self) -> &OutlierStatistics {
        self.statistics_
            .as_ref()
            .expect("OutlierDetector should be fitted")
    }

    /// Get number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap_or(0)
    }

    /// Check if a single feature value is an outlier
    pub fn is_outlier_single(&self, feature_idx: usize, value: Float) -> bool {
        if let Some(ref params) = self.feature_params_ {
            if feature_idx < params.len() {
                return self.is_outlier_for_params(&params[feature_idx], value);
            }
        }
        false
    }

    /// Check outlier based on parameters
    fn is_outlier_for_params(&self, params: &FeatureOutlierParams, value: Float) -> bool {
        match self.config.method {
            OutlierDetectionMethod::ZScore => {
                if let (Some(mean), Some(std)) = (params.mean, params.std) {
                    let z_score = (value - mean) / std;
                    z_score.abs() > self.config.threshold
                } else {
                    false
                }
            }
            OutlierDetectionMethod::ModifiedZScore => {
                if let (Some(median), Some(mad)) = (params.median, params.mad) {
                    let modified_z = 0.6745 * (value - median) / mad;
                    modified_z.abs() > self.config.threshold
                } else {
                    false
                }
            }
            OutlierDetectionMethod::IQR => {
                if let (Some(lower), Some(upper)) = (params.lower_bound, params.upper_bound) {
                    value < lower || value > upper
                } else {
                    false
                }
            }
            OutlierDetectionMethod::Percentile => {
                if let (Some(lower), Some(upper)) =
                    (params.lower_percentile_value, params.upper_percentile_value)
                {
                    value < lower || value > upper
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Generate detailed outlier detection results
    pub fn detect_outliers(&self, x: &Array2<Float>) -> Result<OutlierDetectionResult> {
        let outlier_mask = self.transform(x)?;
        let scores = self.outlier_scores(x)?;

        let outliers: Vec<bool> = outlier_mask.iter().map(|&x| x).collect();
        let n_outliers = outliers.iter().filter(|&&x| x).count();
        let n_samples = outliers.len();

        Ok(OutlierDetectionResult {
            outliers: outliers.clone(),
            scores,
            summary: OutlierSummary {
                n_samples,
                n_outliers,
                outlier_fraction: n_outliers as Float / n_samples as Float,
                method: self.config.method,
            },
        })
    }

    /// Compute outlier scores
    pub fn outlier_scores(&self, x: &Array2<Float>) -> Result<Vec<Float>> {
        let n_samples = x.nrows();
        let mut scores = vec![0.0; n_samples];

        match self.config.method {
            OutlierDetectionMethod::ZScore => {
                if let Some(ref params) = self.feature_params_ {
                    for i in 0..n_samples {
                        let mut max_score: Float = 0.0;
                        for (j, param) in params.iter().enumerate() {
                            if let (Some(mean), Some(std)) = (param.mean, param.std) {
                                let z_score = ((x[[i, j]] - mean) / std).abs();
                                max_score = max_score.max(z_score);
                            }
                        }
                        scores[i] = max_score;
                    }
                }
            }
            OutlierDetectionMethod::MahalanobisDistance => {
                if let Some(ref params) = self.multivariate_params_ {
                    for i in 0..n_samples {
                        let sample: Vec<Float> = (0..x.ncols()).map(|j| x[[i, j]]).collect();
                        let mut centered = vec![0.0; sample.len()];

                        // Center the data
                        for (j, &val) in sample.iter().enumerate() {
                            centered[j] = val - params.mean[j];
                        }

                        // Compute Mahalanobis distance
                        let temp = simd_matvec_multiply(&params.inv_covariance, &centered);
                        scores[i] = simd_dot_product(&centered, &temp).sqrt();
                    }
                }
            }
            _ => {
                // Default scoring
                for i in 0..n_samples {
                    scores[i] = 0.0;
                }
            }
        }

        Ok(scores)
    }
}

impl Default for OutlierDetector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for OutlierDetector<Untrained> {
    type Fitted = OutlierDetector<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        let mut feature_params = None;
        let mut multivariate_params = None;

        match self.config.method {
            OutlierDetectionMethod::MahalanobisDistance => {
                multivariate_params = Some(self.compute_multivariate_statistics(x)?);
            }
            _ => {
                // Compute per-feature parameters
                let mut params = Vec::new();
                for j in 0..n_features {
                    let feature_column = x.column(j).to_owned();
                    params.push(self.compute_feature_statistics(&feature_column)?);
                }
                feature_params = Some(params);
            }
        }

        let statistics = OutlierStatistics {
            n_outliers: 0, // Will be computed during transform
            outlier_fraction: 0.0,
            feature_outlier_counts: vec![0; n_features],
        };

        Ok(OutlierDetector {
            config: self.config,
            state: PhantomData,
            n_features_in_: Some(n_features),
            statistics_: Some(statistics),
            feature_params_: feature_params,
            multivariate_params_: multivariate_params,
        })
    }
}

impl Transform<Array2<Float>, Array2<bool>> for OutlierDetector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<bool>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let mut outlier_mask = Array2::<bool>::default((n_samples, 1));

        match self.config.method {
            OutlierDetectionMethod::MahalanobisDistance => {
                if let Some(ref params) = self.multivariate_params_ {
                    for i in 0..n_samples {
                        let sample: Vec<Float> = (0..n_features).map(|j| x[[i, j]]).collect();
                        let mut centered = vec![0.0; sample.len()];

                        // Center the data
                        for (j, &val) in sample.iter().enumerate() {
                            centered[j] = val - params.mean[j];
                        }

                        // Compute Mahalanobis distance
                        let temp = simd_matvec_multiply(&params.inv_covariance, &centered);
                        let distance_squared = simd_dot_product(&centered, &temp);

                        outlier_mask[[i, 0]] = distance_squared > params.threshold;
                    }
                }
            }
            _ => {
                // Feature-wise outlier detection
                if let Some(ref params) = self.feature_params_ {
                    for i in 0..n_samples {
                        let mut is_outlier = false;
                        for (j, param) in params.iter().enumerate() {
                            if self.is_outlier_for_params(param, x[[i, j]]) {
                                is_outlier = true;
                                break; // Any feature being an outlier makes the sample an outlier
                            }
                        }
                        outlier_mask[[i, 0]] = is_outlier;
                    }
                }
            }
        }

        Ok(outlier_mask)
    }
}
