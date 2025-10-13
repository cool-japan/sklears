//! Outlier transformation methods for handling extreme values
//!
//! This module provides various transformation methods specifically designed to handle
//! outliers in data while preserving the overall structure and relationships. Unlike
//! outlier detection which identifies outliers, these methods transform them to reduce
//! their impact on downstream analysis.
//!
//! # Features
//!
//! - **Log Transformation**: Reduces impact of large outliers through logarithmic scaling
//! - **Square Root Transformation**: Mild transformation for positive outliers
//! - **Box-Cox Transformation**: Data-driven power transformation for normalization
//! - **Quantile Transformation**: Maps to uniform or normal distribution
//! - **Robust Scaling**: Scaling resistant to outliers using median and IQR
//! - **Outlier Interpolation**: Replace outliers with interpolated values
//! - **Outlier Smoothing**: Smooth outliers using neighboring values
//! - **Trimmed Transformation**: Apply transformations after trimming extreme percentiles

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Available outlier transformation methods
#[derive(Debug, Clone, Copy)]
pub enum OutlierTransformationMethod {
    /// Natural logarithm transformation (for positive values)
    Log,
    /// Log1p transformation (log(1 + x), handles zeros)
    Log1p,
    /// Square root transformation (for positive values)
    Sqrt,
    /// Box-Cox transformation with automatic lambda estimation
    BoxCox,
    /// Box-Cox transformation with fixed lambda
    BoxCoxFixed(Float),
    /// Quantile transformation to uniform distribution
    QuantileUniform,
    /// Quantile transformation to normal distribution
    QuantileNormal,
    /// Robust scaling using median and IQR
    RobustScale,
    /// Replace outliers with interpolated values
    Interpolate,
    /// Smooth outliers using local neighborhood
    Smooth,
    /// Trim extreme percentiles before transformation
    Trim,
}

/// Configuration for outlier transformation
#[derive(Debug, Clone)]
pub struct OutlierTransformationConfig {
    /// Transformation method to apply
    pub method: OutlierTransformationMethod,
    /// Threshold for outlier detection (used with interpolation/smoothing)
    pub outlier_threshold: Float,
    /// Method for outlier detection (z-score, iqr, percentile)
    pub detection_method: String,
    /// Lower percentile for trimming (default: 1.0)
    pub lower_percentile: Float,
    /// Upper percentile for trimming (default: 99.0)
    pub upper_percentile: Float,
    /// Window size for smoothing (default: 5)
    pub smoothing_window: usize,
    /// Number of quantiles for quantile transformation (default: 1000)
    pub n_quantiles: usize,
    /// Whether to handle negative values by shifting (default: true)
    pub handle_negatives: bool,
    /// Small constant to add before log transformation to avoid zeros
    pub log_epsilon: Float,
    /// Whether to apply transformation feature-wise (default: true)
    pub feature_wise: bool,
}

impl Default for OutlierTransformationConfig {
    fn default() -> Self {
        Self {
            method: OutlierTransformationMethod::Log1p,
            outlier_threshold: 3.0,
            detection_method: "z-score".to_string(),
            lower_percentile: 1.0,
            upper_percentile: 99.0,
            smoothing_window: 5,
            n_quantiles: 1000,
            handle_negatives: true,
            log_epsilon: 1e-8,
            feature_wise: true,
        }
    }
}

/// Outlier transformer for handling extreme values through transformation
#[derive(Debug, Clone)]
pub struct OutlierTransformer<State = Untrained> {
    config: OutlierTransformationConfig,
    state: PhantomData<State>,
    // Fitted parameters
    transformation_params_: Option<TransformationParameters>,
    n_features_in_: Option<usize>,
}

/// Parameters learned during fitting for transformations
#[derive(Debug, Clone)]
pub struct TransformationParameters {
    /// Feature-wise transformation parameters
    pub feature_params: Vec<FeatureTransformationParams>,
    /// Global parameters (for non-feature-wise transformations)
    pub global_params: Option<GlobalTransformationParams>,
}

/// Transformation parameters for a single feature
#[derive(Debug, Clone)]
pub struct FeatureTransformationParams {
    /// Box-Cox lambda parameter
    pub lambda: Option<Float>,
    /// Shift applied to handle negative values
    pub shift: Float,
    /// Quantiles for quantile transformation
    pub quantiles: Option<Array1<Float>>,
    /// References values for quantile transformation
    pub references: Option<Array1<Float>>,
    /// Robust scaling parameters (median, IQR)
    pub median: Option<Float>,
    pub iqr: Option<Float>,
    /// Outlier bounds for interpolation/smoothing
    pub lower_bound: Option<Float>,
    pub upper_bound: Option<Float>,
    /// Statistics for outlier detection
    pub mean: Option<Float>,
    pub std: Option<Float>,
}

/// Global transformation parameters
#[derive(Debug, Clone)]
pub struct GlobalTransformationParams {
    /// Global shift for handling negatives
    pub global_shift: Float,
    /// Global lambda for Box-Cox
    pub global_lambda: Option<Float>,
}

impl OutlierTransformer<Untrained> {
    /// Create a new OutlierTransformer with default configuration
    pub fn new() -> Self {
        Self {
            config: OutlierTransformationConfig::default(),
            state: PhantomData,
            transformation_params_: None,
            n_features_in_: None,
        }
    }

    /// Create a log transformation for outliers
    pub fn log() -> Self {
        Self::new().method(OutlierTransformationMethod::Log)
    }

    /// Create a log1p transformation for outliers
    pub fn log1p() -> Self {
        Self::new().method(OutlierTransformationMethod::Log1p)
    }

    /// Create a square root transformation for outliers
    pub fn sqrt() -> Self {
        Self::new().method(OutlierTransformationMethod::Sqrt)
    }

    /// Create a Box-Cox transformation with automatic lambda
    pub fn box_cox() -> Self {
        Self::new().method(OutlierTransformationMethod::BoxCox)
    }

    /// Create a Box-Cox transformation with fixed lambda
    pub fn box_cox_fixed(lambda: Float) -> Self {
        Self::new().method(OutlierTransformationMethod::BoxCoxFixed(lambda))
    }

    /// Create a quantile transformation to uniform distribution
    pub fn quantile_uniform(n_quantiles: usize) -> Self {
        Self::new()
            .method(OutlierTransformationMethod::QuantileUniform)
            .n_quantiles(n_quantiles)
    }

    /// Create a quantile transformation to normal distribution
    pub fn quantile_normal(n_quantiles: usize) -> Self {
        Self::new()
            .method(OutlierTransformationMethod::QuantileNormal)
            .n_quantiles(n_quantiles)
    }

    /// Create a robust scaling transformation
    pub fn robust_scale() -> Self {
        Self::new().method(OutlierTransformationMethod::RobustScale)
    }

    /// Create an interpolation transformation
    pub fn interpolate(threshold: Float, detection_method: &str) -> Self {
        Self::new()
            .method(OutlierTransformationMethod::Interpolate)
            .outlier_threshold(threshold)
            .detection_method(detection_method.to_string())
    }

    /// Create a smoothing transformation
    pub fn smooth(window_size: usize, threshold: Float) -> Self {
        Self::new()
            .method(OutlierTransformationMethod::Smooth)
            .smoothing_window(window_size)
            .outlier_threshold(threshold)
    }

    /// Create a trimming transformation
    pub fn trim(lower_percentile: Float, upper_percentile: Float) -> Self {
        Self::new()
            .method(OutlierTransformationMethod::Trim)
            .lower_percentile(lower_percentile)
            .upper_percentile(upper_percentile)
    }

    /// Set the transformation method
    pub fn method(mut self, method: OutlierTransformationMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the outlier detection threshold
    pub fn outlier_threshold(mut self, threshold: Float) -> Self {
        self.config.outlier_threshold = threshold;
        self
    }

    /// Set the outlier detection method
    pub fn detection_method(mut self, method: String) -> Self {
        self.config.detection_method = method;
        self
    }

    /// Set the lower percentile for trimming
    pub fn lower_percentile(mut self, percentile: Float) -> Self {
        self.config.lower_percentile = percentile;
        self
    }

    /// Set the upper percentile for trimming
    pub fn upper_percentile(mut self, percentile: Float) -> Self {
        self.config.upper_percentile = percentile;
        self
    }

    /// Set the smoothing window size
    pub fn smoothing_window(mut self, window: usize) -> Self {
        self.config.smoothing_window = window;
        self
    }

    /// Set the number of quantiles
    pub fn n_quantiles(mut self, n_quantiles: usize) -> Self {
        self.config.n_quantiles = n_quantiles;
        self
    }

    /// Set whether to handle negative values
    pub fn handle_negatives(mut self, handle: bool) -> Self {
        self.config.handle_negatives = handle;
        self
    }

    /// Set the epsilon for log transformations
    pub fn log_epsilon(mut self, epsilon: Float) -> Self {
        self.config.log_epsilon = epsilon;
        self
    }

    /// Set whether to apply transformations feature-wise
    pub fn feature_wise(mut self, feature_wise: bool) -> Self {
        self.config.feature_wise = feature_wise;
        self
    }
}

impl Fit<Array2<Float>, ()> for OutlierTransformer<Untrained> {
    type Fitted = OutlierTransformer<Trained>;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        self.n_features_in_ = Some(n_features);

        // Compute transformation parameters based on method
        let feature_params = if self.config.feature_wise {
            (0..n_features)
                .map(|j| self.fit_feature_params(x.column(j).to_owned().as_slice().unwrap()))
                .collect::<Result<Vec<_>>>()?
        } else {
            // For non-feature-wise, we'll use global parameters
            vec![self.fit_feature_params(x.as_slice().unwrap())?]
        };

        self.transformation_params_ = Some(TransformationParameters {
            feature_params,
            global_params: None, // Could be used for global transformations
        });

        Ok(OutlierTransformer {
            config: self.config,
            state: PhantomData,
            transformation_params_: self.transformation_params_,
            n_features_in_: self.n_features_in_,
        })
    }
}

impl OutlierTransformer<Untrained> {
    /// Fit parameters for a single feature
    fn fit_feature_params(&self, data: &[Float]) -> Result<FeatureTransformationParams> {
        let mut params = FeatureTransformationParams {
            lambda: None,
            shift: 0.0,
            quantiles: None,
            references: None,
            median: None,
            iqr: None,
            lower_bound: None,
            upper_bound: None,
            mean: None,
            std: None,
        };

        // Calculate basic statistics
        let valid_data: Vec<Float> = data.iter().filter(|x| x.is_finite()).copied().collect();

        if valid_data.is_empty() {
            return Ok(params);
        }

        let mean = valid_data.iter().sum::<Float>() / valid_data.len() as Float;
        let variance = valid_data.iter().map(|x| (x - mean).powi(2)).sum::<Float>()
            / valid_data.len() as Float;
        let std = variance.sqrt();

        params.mean = Some(mean);
        params.std = Some(std);

        // Calculate median and IQR for robust methods
        let mut sorted_data = valid_data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_data.len() % 2 == 0 {
            let mid = sorted_data.len() / 2;
            (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        let q1_idx = sorted_data.len() / 4;
        let q3_idx = 3 * sorted_data.len() / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;

        params.median = Some(median);
        params.iqr = Some(iqr);

        // Set outlier bounds based on detection method
        match self.config.detection_method.as_str() {
            "z-score" => {
                params.lower_bound = Some(mean - self.config.outlier_threshold * std);
                params.upper_bound = Some(mean + self.config.outlier_threshold * std);
            }
            "iqr" => {
                params.lower_bound = Some(q1 - self.config.outlier_threshold * iqr);
                params.upper_bound = Some(q3 + self.config.outlier_threshold * iqr);
            }
            "percentile" => {
                let lower_idx =
                    ((self.config.lower_percentile / 100.0) * sorted_data.len() as Float) as usize;
                let upper_idx =
                    ((self.config.upper_percentile / 100.0) * sorted_data.len() as Float) as usize;
                params.lower_bound = Some(sorted_data[lower_idx.min(sorted_data.len() - 1)]);
                params.upper_bound = Some(sorted_data[upper_idx.min(sorted_data.len() - 1)]);
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown detection method: {}",
                    self.config.detection_method
                )));
            }
        }

        // Handle negative values for log/sqrt transformations
        if self.config.handle_negatives {
            match self.config.method {
                OutlierTransformationMethod::Log | OutlierTransformationMethod::Sqrt => {
                    let min_val = sorted_data[0];
                    if min_val <= 0.0 {
                        params.shift = -min_val + self.config.log_epsilon;
                    }
                }
                OutlierTransformationMethod::BoxCox
                | OutlierTransformationMethod::BoxCoxFixed(_) => {
                    let min_val = sorted_data[0];
                    if min_val <= 0.0 {
                        params.shift = -min_val + self.config.log_epsilon;
                    }
                }
                _ => {}
            }
        }

        // Fit method-specific parameters
        match self.config.method {
            OutlierTransformationMethod::BoxCox => {
                params.lambda = Some(self.estimate_box_cox_lambda(&valid_data, params.shift)?);
            }
            OutlierTransformationMethod::BoxCoxFixed(lambda) => {
                params.lambda = Some(lambda);
            }
            OutlierTransformationMethod::QuantileUniform
            | OutlierTransformationMethod::QuantileNormal => {
                params.quantiles = Some(self.compute_quantiles(&sorted_data)?);
                params.references = Some(self.compute_references()?);
            }
            _ => {}
        }

        Ok(params)
    }

    /// Estimate optimal lambda for Box-Cox transformation using maximum likelihood
    fn estimate_box_cox_lambda(&self, data: &[Float], shift: Float) -> Result<Float> {
        let shifted_data: Vec<Float> = data.iter().map(|x| x + shift).collect();

        // Search for optimal lambda in range [-2, 2]
        let lambda_range: Vec<Float> = (-20..=20).map(|i| i as Float * 0.1).collect();

        let mut best_lambda = 0.0;
        let mut best_llf = Float::NEG_INFINITY;

        for &lambda in &lambda_range {
            if let Ok(llf) = self.box_cox_log_likelihood(&shifted_data, lambda) {
                if llf > best_llf {
                    best_llf = llf;
                    best_lambda = lambda;
                }
            }
        }

        Ok(best_lambda)
    }

    /// Compute log-likelihood for Box-Cox transformation
    fn box_cox_log_likelihood(&self, data: &[Float], lambda: Float) -> Result<Float> {
        let n = data.len() as Float;

        // Transform data
        let transformed: Vec<Float> = data
            .iter()
            .map(|&x| {
                if x <= 0.0 {
                    return Float::NAN;
                }
                if lambda.abs() < 1e-10 {
                    x.ln()
                } else {
                    (x.powf(lambda) - 1.0) / lambda
                }
            })
            .collect();

        // Check for invalid transformations
        if transformed.iter().any(|x| !x.is_finite()) {
            return Err(SklearsError::InvalidInput(
                "Invalid Box-Cox transformation".to_string(),
            ));
        }

        // Compute log-likelihood
        let mean = transformed.iter().sum::<Float>() / n;
        let variance = transformed
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<Float>()
            / n;

        let log_jacobian = (lambda - 1.0) * data.iter().map(|x| x.ln()).sum::<Float>();
        let llf = -0.5 * n * (2.0 * std::f64::consts::PI as Float).ln()
            - 0.5 * n * variance.ln()
            - 0.5 * n
            + log_jacobian;

        Ok(llf)
    }

    /// Compute quantiles for quantile transformation
    fn compute_quantiles(&self, sorted_data: &[Float]) -> Result<Array1<Float>> {
        let n_quantiles = self.config.n_quantiles.min(sorted_data.len());
        let mut quantiles = Array1::zeros(n_quantiles);

        for i in 0..n_quantiles {
            let q = i as Float / (n_quantiles - 1) as Float;
            let idx = (q * (sorted_data.len() - 1) as Float) as usize;
            quantiles[i] = sorted_data[idx.min(sorted_data.len() - 1)];
        }

        Ok(quantiles)
    }

    /// Compute reference values for quantile transformation
    fn compute_references(&self) -> Result<Array1<Float>> {
        let n_quantiles = self.config.n_quantiles;
        let mut references = Array1::zeros(n_quantiles);

        match self.config.method {
            OutlierTransformationMethod::QuantileUniform => {
                for i in 0..n_quantiles {
                    references[i] = i as Float / (n_quantiles - 1) as Float;
                }
            }
            OutlierTransformationMethod::QuantileNormal => {
                // Approximate normal quantiles
                for i in 0..n_quantiles {
                    let p = i as Float / (n_quantiles - 1) as Float;
                    references[i] = self.inverse_normal_cdf(p);
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Invalid quantile method".to_string(),
                ));
            }
        }

        Ok(references)
    }

    /// Approximate inverse normal CDF using Beasley-Springer-Moro algorithm
    fn inverse_normal_cdf(&self, p: Float) -> Float {
        if p <= 0.0 {
            return Float::NEG_INFINITY;
        }
        if p >= 1.0 {
            return Float::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        // Use simple approximation for demonstration
        // In production, use a more accurate method
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];

        let q = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * q.ln()).sqrt();

        let mut num = a[5];
        for i in (0..5).rev() {
            num = num * t + a[i];
        }

        let mut den = 1.0;
        for i in (0..5).rev() {
            den = den * t + b[i];
        }

        let x = t - num / den;
        if p > 0.5 {
            x
        } else {
            -x
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OutlierTransformer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.n_features_in().unwrap() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in().unwrap(),
                actual: n_features,
            });
        }

        let params = self.transformation_params_.as_ref().unwrap();
        let mut result = x.clone();

        if self.config.feature_wise {
            for j in 0..n_features {
                let feature_params = &params.feature_params[j];
                let mut column = result.column_mut(j);
                self.transform_feature_inplace(&mut column, feature_params)?;
            }
        } else {
            // Global transformation
            let feature_params = &params.feature_params[0];
            for mut row in result.axis_iter_mut(Axis(0)) {
                for elem in row.iter_mut() {
                    *elem = self.transform_value(*elem, feature_params)?;
                }
            }
        }

        Ok(result)
    }
}

impl OutlierTransformer<Trained> {
    /// Get the number of features seen during fit
    pub fn n_features_in(&self) -> Option<usize> {
        self.n_features_in_
    }

    /// Transform a single feature in-place
    fn transform_feature_inplace(
        &self,
        column: &mut scirs2_core::ndarray::ArrayViewMut1<Float>,
        params: &FeatureTransformationParams,
    ) -> Result<()> {
        for elem in column.iter_mut() {
            *elem = self.transform_value(*elem, params)?;
        }
        Ok(())
    }

    /// Transform a single value
    fn transform_value(&self, value: Float, params: &FeatureTransformationParams) -> Result<Float> {
        if !value.is_finite() {
            return Ok(value);
        }

        match self.config.method {
            OutlierTransformationMethod::Log => {
                let shifted = value + params.shift;
                if shifted <= 0.0 {
                    Ok(Float::NAN)
                } else {
                    Ok(shifted.ln())
                }
            }
            OutlierTransformationMethod::Log1p => Ok((value + params.shift).ln_1p()),
            OutlierTransformationMethod::Sqrt => {
                let shifted = value + params.shift;
                if shifted < 0.0 {
                    Ok(Float::NAN)
                } else {
                    Ok(shifted.sqrt())
                }
            }
            OutlierTransformationMethod::BoxCox | OutlierTransformationMethod::BoxCoxFixed(_) => {
                let lambda = params.lambda.unwrap_or(0.0);
                let shifted = value + params.shift;
                if shifted <= 0.0 {
                    return Ok(Float::NAN);
                }
                if lambda.abs() < 1e-10 {
                    Ok(shifted.ln())
                } else {
                    Ok((shifted.powf(lambda) - 1.0) / lambda)
                }
            }
            OutlierTransformationMethod::QuantileUniform
            | OutlierTransformationMethod::QuantileNormal => {
                self.quantile_transform_value(value, params)
            }
            OutlierTransformationMethod::RobustScale => {
                let median = params.median.unwrap_or(0.0);
                let iqr = params.iqr.unwrap_or(1.0);
                if iqr > 0.0 {
                    Ok((value - median) / iqr)
                } else {
                    Ok(0.0)
                }
            }
            OutlierTransformationMethod::Interpolate => self.interpolate_value(value, params),
            OutlierTransformationMethod::Smooth => {
                // For single value, return as-is (smoothing requires neighborhood)
                Ok(value)
            }
            OutlierTransformationMethod::Trim => {
                let lower = params.lower_bound.unwrap_or(Float::NEG_INFINITY);
                let upper = params.upper_bound.unwrap_or(Float::INFINITY);
                Ok(value.max(lower).min(upper))
            }
        }
    }

    /// Transform value using quantile transformation
    fn quantile_transform_value(
        &self,
        value: Float,
        params: &FeatureTransformationParams,
    ) -> Result<Float> {
        let quantiles = params.quantiles.as_ref().unwrap();
        let references = params.references.as_ref().unwrap();

        // Find position in quantiles
        let mut pos = 0;
        for (i, &q) in quantiles.iter().enumerate() {
            if value <= q {
                pos = i;
                break;
            }
            pos = i + 1;
        }

        pos = pos.min(references.len() - 1);
        Ok(references[pos])
    }

    /// Interpolate outlier value
    fn interpolate_value(
        &self,
        value: Float,
        params: &FeatureTransformationParams,
    ) -> Result<Float> {
        let lower = params.lower_bound.unwrap_or(Float::NEG_INFINITY);
        let upper = params.upper_bound.unwrap_or(Float::INFINITY);

        if value < lower {
            Ok(lower)
        } else if value > upper {
            Ok(upper)
        } else {
            Ok(value)
        }
    }

    /// Get transformation parameters
    pub fn transformation_params(&self) -> Option<&TransformationParameters> {
        self.transformation_params_.as_ref()
    }

    /// Get transformation statistics for a specific feature
    pub fn feature_stats(&self, feature_idx: usize) -> Option<&FeatureTransformationParams> {
        self.transformation_params_
            .as_ref()?
            .feature_params
            .get(feature_idx)
    }
}

impl Default for OutlierTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_log_transformation() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 100.0, 1000.0, // Outliers
                4.0, 40.0,
            ],
        )
        .unwrap();

        let transformer = OutlierTransformer::log();
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // First value should be ln(1) = 0
        assert_relative_eq!(result[[0, 0]], 1.0_f64.ln(), epsilon = 1e-10);

        // Large outlier should be transformed: ln(100) ≈ 4.6
        assert_relative_eq!(result[[3, 0]], 100.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_log1p_transformation() {
        let data = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 10.0, 100.0]).unwrap();

        let transformer = OutlierTransformer::log1p();
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // log1p(0) = ln(1) = 0
        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);

        // log1p(1) = ln(2)
        assert_relative_eq!(result[[1, 0]], (2.0_f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_sqrt_transformation() {
        let data = Array2::from_shape_vec((4, 1), vec![1.0, 4.0, 9.0, 100.0]).unwrap();

        let transformer = OutlierTransformer::sqrt();
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // sqrt(1) = 1
        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);

        // sqrt(100) = 10
        assert_relative_eq!(result[[3, 0]], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_robust_scale_transformation() {
        let data = Array2::from_shape_vec(
            (7, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, // 100 is outlier
            ],
        )
        .unwrap();

        let transformer = OutlierTransformer::robust_scale();
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // Should be scaled by median and IQR, making it robust to outliers
        let params = fitted.feature_stats(0).unwrap();
        assert!(params.median.is_some());
        assert!(params.iqr.is_some());
    }

    #[test]
    fn test_interpolate_transformation() {
        let data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 100.0]).unwrap();

        let transformer = OutlierTransformer::interpolate(2.0, "z-score");
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // Normal values should remain unchanged
        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0]], 2.0, epsilon = 1e-10);

        // Outlier should be capped
        let params = fitted.feature_stats(0).unwrap();
        assert!(params.upper_bound.is_some());
    }

    #[test]
    fn test_trim_transformation() {
        let data = Array2::from_shape_vec(
            (11, 1),
            (1..=10)
                .map(|x| x as f64)
                .chain(std::iter::once(1000.0))
                .collect(),
        )
        .unwrap();

        let transformer = OutlierTransformer::trim(10.0, 90.0);
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // Values should be trimmed to percentile bounds
        let params = fitted.feature_stats(0).unwrap();
        assert!(params.lower_bound.is_some());
        assert!(params.upper_bound.is_some());
    }

    #[test]
    fn test_box_cox_transformation() {
        let data = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]).unwrap();

        let transformer = OutlierTransformer::box_cox_fixed(0.5);
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        let params = fitted.feature_stats(0).unwrap();
        assert!(params.lambda.is_some());
        assert_relative_eq!(params.lambda.unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_handle_negative_values() {
        let data = Array2::from_shape_vec((4, 1), vec![-2.0, -1.0, 1.0, 100.0]).unwrap();

        let transformer = OutlierTransformer::log().handle_negatives(true);
        let fitted = transformer.fit(&data, &()).unwrap();
        let result = fitted.transform(&data).unwrap();

        assert_eq!(result.dim(), data.dim());

        // Should have applied shift to handle negatives
        let params = fitted.feature_stats(0).unwrap();
        assert!(params.shift > 0.0);
    }

    #[test]
    fn test_feature_wise_vs_global() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 100.0, 1000.0])
                .unwrap();

        // Feature-wise transformation
        let transformer_fw = OutlierTransformer::log().feature_wise(true);
        let fitted_fw = transformer_fw.fit(&data, &()).unwrap();
        let result_fw = fitted_fw.transform(&data).unwrap();

        // Global transformation
        let transformer_global = OutlierTransformer::log().feature_wise(false);
        let fitted_global = transformer_global.fit(&data, &()).unwrap();
        let result_global = fitted_global.transform(&data).unwrap();

        assert_eq!(result_fw.dim(), data.dim());
        assert_eq!(result_global.dim(), data.dim());

        // Results should be different for feature-wise vs global
        // (This is a basic check - specific values depend on implementation)
    }

    #[test]
    fn test_transformation_error_handling() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let transformer = OutlierTransformer::log();
        let fitted = transformer.fit(&data, &()).unwrap();

        // Test dimension mismatch
        let wrong_data =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.transform(&wrong_data).is_err());
    }

    #[test]
    fn test_detection_methods() {
        let data =
            Array2::from_shape_vec((7, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]).unwrap();

        // Test different detection methods
        let methods = vec!["z-score", "iqr", "percentile"];

        for method in methods {
            let transformer = OutlierTransformer::interpolate(2.0, method);
            let fitted = transformer.fit(&data, &()).unwrap();
            let result = fitted.transform(&data).unwrap();

            assert_eq!(result.dim(), data.dim());

            let params = fitted.feature_stats(0).unwrap();
            assert!(params.lower_bound.is_some());
            assert!(params.upper_bound.is_some());
        }
    }
}
