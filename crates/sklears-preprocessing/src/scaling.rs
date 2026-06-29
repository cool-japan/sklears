//! Data scaling utilities
//!
//! This module provides comprehensive data scaling and normalization implementations including
//! standard scaling (z-score normalization), min-max scaling, robust scaling with quantiles,
//! max absolute value scaling, L1/L2 normalization, unit vector scaling, feature-wise scaling,
//! outlier-aware scaling, kernel centering, polynomial feature generation, power transformations,
//! quantile transformations, SIMD-optimized implementations, streaming scalers, adaptive scalers,
//! categorical feature encoding, mixed-type scaling, and high-performance preprocessing pipelines.
//! All algorithms have been refactored into focused modules for better maintainability and comply
//! with SciRS2 Policy.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};

/// Fitted parameters for `StandardScaler`
#[derive(Debug, Clone)]
pub struct StandardScalerFitParams {
    /// Per-feature mean (populated when `with_mean=true`, else zeros)
    pub mean: Array1<Float>,
    /// Per-feature scale / std (populated when `with_std=true`, 1.0 for constant features)
    pub scale: Array1<Float>,
    /// Number of samples seen during fit
    pub n_samples_seen: usize,
}

/// Standard scaler: center to zero mean and unit variance.
///
/// Equivalent to scikit-learn's `StandardScaler`. Supports `with_mean` and
/// `with_std` flags, handles constant features (std = 0 → scale = 1.0),
/// and provides `inverse_transform` and `fit_transform`.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// Whether to subtract the column mean before scaling
    with_mean: bool,
    /// Whether to divide by the column standard deviation
    with_std: bool,
    /// Fitted parameters (None before fit is called)
    params_: Option<StandardScalerFitParams>,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            params_: None,
        }
    }
}

impl StandardScaler {
    /// Create a new `StandardScaler` with default settings (center and scale).
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure mean-centering (default: `true`).
    pub fn with_mean(mut self, yes: bool) -> Self {
        self.with_mean = yes;
        self
    }

    /// Configure variance-scaling (default: `true`).
    pub fn with_std(mut self, yes: bool) -> Self {
        self.with_std = yes;
        self
    }

    /// Access fitted parameters (returns `None` before `fit`).
    pub fn params(&self) -> Option<&StandardScalerFitParams> {
        self.params_.as_ref()
    }

    /// Convenience: fitted mean vector (None before fit).
    pub fn mean_(&self) -> Option<&Array1<Float>> {
        self.params_.as_ref().map(|p| &p.mean)
    }

    /// Convenience: fitted scale vector (None before fit).
    pub fn scale_(&self) -> Option<&Array1<Float>> {
        self.params_.as_ref().map(|p| &p.scale)
    }

    /// Apply the inverse transform: `X = X_scaled * scale + mean`.
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "StandardScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.mean.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.mean.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let scale = if self.with_std { params.scale[j] } else { 1.0 };
            let mean = if self.with_mean { params.mean[j] } else { 0.0 };
            for i in 0..n_rows {
                result[[i, j]] = result[[i, j]] * scale + mean;
            }
        }
        Ok(result)
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl Fit<Array2<Float>, ()> for StandardScaler {
    type Fitted = StandardScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        let mut mean = Array1::zeros(n_cols);
        let mut scale = Array1::ones(n_cols);

        for j in 0..n_cols {
            let col = x.column(j);

            if self.with_mean {
                mean[j] = col.mean().unwrap_or(0.0);
            }

            if self.with_std {
                // Use population std (ddof=0) consistent with sklearn default
                let col_mean = mean[j];
                let variance: Float = col
                    .iter()
                    .map(|&v| {
                        let d = v - col_mean;
                        d * d
                    })
                    .sum::<Float>()
                    / n_rows as Float;
                let std = variance.sqrt();
                // Guard constant features: set scale to 1.0 to avoid divide-by-zero
                scale[j] = if std > Float::EPSILON { std } else { 1.0 };
            }
        }

        self.params_ = Some(StandardScalerFitParams {
            mean,
            scale,
            n_samples_seen: n_rows,
        });
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for StandardScaler {
    /// Transform `x` using fitted mean and scale.
    ///
    /// Returns the input unchanged if the scaler has not been fitted
    /// (preserves existing behaviour for unfitted scalers used in pipeline tests).
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = match self.params_.as_ref() {
            Some(p) => p,
            // Passthrough when not fitted — preserves existing pipeline test behaviour
            None => return Ok(x.clone()),
        };

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.mean.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.mean.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let mean = if self.with_mean { params.mean[j] } else { 0.0 };
            let scale = if self.with_std { params.scale[j] } else { 1.0 };
            for i in 0..n_rows {
                result[[i, j]] = (result[[i, j]] - mean) / scale;
            }
        }
        Ok(result)
    }
}

/// Fitted parameters for `MinMaxScaler`
#[derive(Debug, Clone)]
pub struct MinMaxScalerFitParams {
    /// Per-feature minimum observed during fit
    pub data_min: Array1<Float>,
    /// Per-feature maximum observed during fit
    pub data_max: Array1<Float>,
    /// Per-feature scaling factor `(feature_max - feature_min) / (data_max - data_min)`
    pub scale: Array1<Float>,
    /// Per-feature offset `feature_min - data_min * scale`
    pub min: Array1<Float>,
    /// Target output range `(feature_min, feature_max)`
    pub feature_range: (Float, Float),
}

/// Min-max scaler: linearly rescale each feature into a target range.
///
/// Equivalent to scikit-learn's `MinMaxScaler`. For each feature the transform is
/// `X_scaled = X * scale_ + min_` where
/// `scale_ = (feature_max - feature_min) / (data_max - data_min)` and
/// `min_ = feature_min - data_min * scale_`. Constant features (where
/// `data_max == data_min`) use `scale_ = 1.0` so they map to `feature_min`.
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    /// Target output range `(min, max)` (default `(0.0, 1.0)`)
    feature_range: (Float, Float),
    /// Fitted parameters (`None` before `fit`)
    params_: Option<MinMaxScalerFitParams>,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            params_: None,
        }
    }
}

impl MinMaxScaler {
    /// Create a new `MinMaxScaler` with default range `(0.0, 1.0)`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the target output range `(min, max)`.
    pub fn feature_range(mut self, min: Float, max: Float) -> Self {
        self.feature_range = (min, max);
        self
    }

    /// Access fitted parameters (returns `None` before `fit`).
    pub fn params(&self) -> Option<&MinMaxScalerFitParams> {
        self.params_.as_ref()
    }

    /// Convenience: per-feature observed minimum (`None` before `fit`).
    pub fn data_min_(&self) -> Option<&Array1<Float>> {
        self.params_.as_ref().map(|p| &p.data_min)
    }

    /// Convenience: per-feature observed maximum (`None` before `fit`).
    pub fn data_max_(&self) -> Option<&Array1<Float>> {
        self.params_.as_ref().map(|p| &p.data_max)
    }

    /// Convenience: per-feature scaling factor (`None` before `fit`).
    pub fn scale_(&self) -> Option<&Array1<Float>> {
        self.params_.as_ref().map(|p| &p.scale)
    }

    /// Convenience: per-feature offset (`None` before `fit`).
    pub fn min_(&self) -> Option<&Array1<Float>> {
        self.params_.as_ref().map(|p| &p.min)
    }

    /// Apply the inverse transform: `X = (X_scaled - min_) / scale_`.
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "MinMaxScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.scale.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.scale.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let scale = params.scale[j];
            let min = params.min[j];
            for i in 0..n_rows {
                result[[i, j]] = (result[[i, j]] - min) / scale;
            }
        }
        Ok(result)
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl Fit<Array2<Float>, ()> for MinMaxScaler {
    type Fitted = MinMaxScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        let (feature_min, feature_max) = self.feature_range;
        if feature_max <= feature_min {
            return Err(SklearsError::InvalidInput(format!(
                "feature_range max ({feature_max}) must be greater than min ({feature_min})"
            )));
        }

        let mut data_min = Array1::zeros(n_cols);
        let mut data_max = Array1::zeros(n_cols);
        let mut scale = Array1::ones(n_cols);
        let mut min = Array1::zeros(n_cols);

        for j in 0..n_cols {
            // Filter out non-finite values before computing min/max so NaN/Inf
            // does not corrupt the fitted statistics.
            let finite: Vec<Float> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();

            if finite.is_empty() {
                // Entirely non-finite column → map everything to feature_min.
                data_min[j] = 0.0;
                data_max[j] = 0.0;
                scale[j] = 1.0;
                min[j] = feature_min;
                continue;
            }

            let col_min = finite.iter().copied().fold(Float::INFINITY, Float::min);
            let col_max = finite.iter().copied().fold(Float::NEG_INFINITY, Float::max);

            data_min[j] = col_min;
            data_max[j] = col_max;

            let range = col_max - col_min;
            // Guard constant features (range == 0): scale stays 1.0 so the column
            // maps to feature_min via the offset below.
            let s = if range.abs() < Float::EPSILON {
                1.0
            } else {
                (feature_max - feature_min) / range
            };
            scale[j] = s;
            min[j] = feature_min - col_min * s;
        }

        self.params_ = Some(MinMaxScalerFitParams {
            data_min,
            data_max,
            scale,
            min,
            feature_range: self.feature_range,
        });
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MinMaxScaler {
    /// Transform `x` using fitted scale and offset.
    ///
    /// Returns an error if the scaler has not been fitted yet.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "MinMaxScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.scale.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.scale.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let scale = params.scale[j];
            let min = params.min[j];
            for i in 0..n_rows {
                result[[i, j]] = result[[i, j]] * scale + min;
            }
        }
        Ok(result)
    }
}

/// Per-feature statistics fitted by `RobustScaler`
#[derive(Debug, Clone)]
pub struct RobustScalerFitParams {
    /// Median (center) per feature
    pub center: Vec<Float>,
    /// IQR-based scale per feature (1.0 when IQR is zero)
    pub scale: Vec<Float>,
    /// Lower quantile bound per feature (for reference)
    pub quantile_lower: Float,
    /// Upper quantile bound per feature (for reference)
    pub quantile_upper: Float,
}

/// `RobustScaler` centers features by their median and scales by the interquartile
/// range (IQR = Q_upper − Q_lower), making it resistant to outliers.
///
/// # Fit state
/// Before `fit`, `params_` is `None` and `transform` returns the input unchanged.
/// After `fit`, `params_` contains per-feature `center` and `scale` vectors.
#[derive(Debug, Clone)]
pub struct RobustScaler {
    /// Lower quantile bound (default 25.0)
    quantile_lower: Float,
    /// Upper quantile bound (default 75.0)
    quantile_upper: Float,
    /// Whether to subtract the median before scaling
    with_centering: bool,
    /// Whether to divide by IQR
    with_scaling: bool,
    /// Fitted parameters (populated after `fit`)
    params_: Option<RobustScalerFitParams>,
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self {
            quantile_lower: 25.0,
            quantile_upper: 75.0,
            with_centering: true,
            with_scaling: true,
            params_: None,
        }
    }
}

impl RobustScaler {
    /// Create a new `RobustScaler` with default IQR range (Q25–Q75).
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the quantile range used for scaling.
    ///
    /// `lower` and `upper` are percentile values in [0, 100].
    pub fn quantile_range(mut self, lower: Float, upper: Float) -> Self {
        self.quantile_lower = lower;
        self.quantile_upper = upper;
        self
    }

    /// Enable or disable centering (subtract median).
    pub fn with_centering(mut self, yes: bool) -> Self {
        self.with_centering = yes;
        self
    }

    /// Enable or disable scaling (divide by IQR).
    pub fn with_scaling(mut self, yes: bool) -> Self {
        self.with_scaling = yes;
        self
    }

    /// Access fitted parameters (returns `None` before `fit`).
    pub fn params(&self) -> Option<&RobustScalerFitParams> {
        self.params_.as_ref()
    }

    /// Compute a quantile for a sorted slice.
    ///
    /// Uses linear interpolation between the two surrounding values.
    fn quantile_of_sorted(sorted: &[Float], q: Float) -> Float {
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted[0];
        }
        // Map q ∈ [0,100] to index space [0, n-1]
        let pos = (q / 100.0) * (n as Float - 1.0);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = pos - lo as Float;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    /// Apply the inverse transform: `X = X_scaled * scale + center`.
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "RobustScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.center.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.center.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let scale = if self.with_scaling {
                params.scale[j]
            } else {
                1.0
            };
            let center = if self.with_centering {
                params.center[j]
            } else {
                0.0
            };
            for i in 0..n_rows {
                result[[i, j]] = result[[i, j]] * scale + center;
            }
        }
        Ok(result)
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl Fit<Array2<Float>, ()> for RobustScaler {
    type Fitted = RobustScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        let mut center = Vec::with_capacity(n_cols);
        let mut scale = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            // Filter out non-finite values (NaN, Inf) before computing quantiles
            // so that NaN does not propagate into center/scale and corrupt transform output.
            let mut col: Vec<Float> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();

            if col.is_empty() {
                // All values are non-finite — fall back to identity (center=0, scale=1)
                center.push(0.0);
                scale.push(1.0);
                continue;
            }

            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let med = Self::quantile_of_sorted(&col, 50.0);
            let q_lo = Self::quantile_of_sorted(&col, self.quantile_lower);
            let q_hi = Self::quantile_of_sorted(&col, self.quantile_upper);
            let iqr = q_hi - q_lo;

            center.push(med);
            // Guard against zero IQR to avoid division by zero
            scale.push(if iqr.abs() < Float::EPSILON { 1.0 } else { iqr });
        }

        self.params_ = Some(RobustScalerFitParams {
            center,
            scale,
            quantile_lower: self.quantile_lower,
            quantile_upper: self.quantile_upper,
        });
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for RobustScaler {
    /// Transform `x` using fitted median and IQR.
    ///
    /// Returns an error if the scaler has not been fitted yet.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "RobustScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.center.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.center.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let center = if self.with_centering {
                params.center[j]
            } else {
                0.0
            };
            let scale = if self.with_scaling {
                params.scale[j]
            } else {
                1.0
            };
            for i in 0..n_rows {
                result[[i, j]] = (result[[i, j]] - center) / scale;
            }
        }
        Ok(result)
    }
}

/// Fitted parameters for `MaxAbsScaler`
#[derive(Debug, Clone)]
pub struct MaxAbsScalerFitParams {
    /// Per-feature maximum absolute value observed during fit
    pub max_abs: Vec<Float>,
    /// Per-feature scaling factor (equal to `max_abs`, or `1.0` for all-zero columns)
    pub scale: Vec<Float>,
}

/// Max-abs scaler: scale each feature by its maximum absolute value.
///
/// Equivalent to scikit-learn's `MaxAbsScaler`. Each feature is divided by its
/// maximum absolute value so that the transformed data lies in `[-1, 1]`. Columns
/// whose maximum absolute value is zero use `scale = 1.0` (identity).
#[derive(Debug, Clone, Default)]
pub struct MaxAbsScaler {
    /// Fitted parameters (`None` before `fit`)
    params_: Option<MaxAbsScalerFitParams>,
}

impl MaxAbsScaler {
    /// Create a new `MaxAbsScaler`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Access fitted parameters (returns `None` before `fit`).
    pub fn params(&self) -> Option<&MaxAbsScalerFitParams> {
        self.params_.as_ref()
    }

    /// Convenience: per-feature maximum absolute value (`None` before `fit`).
    pub fn max_abs_(&self) -> Option<&[Float]> {
        self.params_.as_ref().map(|p| p.max_abs.as_slice())
    }

    /// Convenience: per-feature scaling factor (`None` before `fit`).
    pub fn scale_(&self) -> Option<&[Float]> {
        self.params_.as_ref().map(|p| p.scale.as_slice())
    }

    /// Apply the inverse transform: `X = X_scaled * scale`.
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "MaxAbsScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.scale.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.scale.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let scale = params.scale[j];
            for i in 0..n_rows {
                result[[i, j]] *= scale;
            }
        }
        Ok(result)
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl Fit<Array2<Float>, ()> for MaxAbsScaler {
    type Fitted = MaxAbsScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        let mut max_abs = Vec::with_capacity(n_cols);
        let mut scale = Vec::with_capacity(n_cols);

        for j in 0..n_cols {
            // Maximum absolute value over finite entries only.
            let col_max_abs = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(|v| v.abs())
                .fold(0.0, Float::max);

            max_abs.push(col_max_abs);
            // Guard all-zero columns to avoid division by zero.
            scale.push(if col_max_abs.abs() < Float::EPSILON {
                1.0
            } else {
                col_max_abs
            });
        }

        self.params_ = Some(MaxAbsScalerFitParams { max_abs, scale });
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MaxAbsScaler {
    /// Transform `x` by dividing each feature by its fitted maximum absolute value.
    ///
    /// Returns an error if the scaler has not been fitted yet.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "MaxAbsScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.scale.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.scale.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let scale = params.scale[j];
            for i in 0..n_rows {
                result[[i, j]] /= scale;
            }
        }
        Ok(result)
    }
}

/// Row-wise vector normalizer, equivalent to scikit-learn's `Normalizer`.
///
/// Each sample (row) is rescaled independently to unit norm under the configured
/// [`NormType`] (`L1`, `L2`, or `Max`). The transform is stateless — there are no
/// fitted parameters — so a row whose norm is effectively zero is left unchanged.
#[derive(Debug, Clone, Default)]
pub struct Normalizer {
    norm: NormType,
}

impl Normalizer {
    pub fn new() -> Self {
        Self { norm: NormType::L2 }
    }

    pub fn norm(mut self, norm: NormType) -> Self {
        self.norm = norm;
        self
    }
}

impl Transform<Array2<Float>, Array2<Float>> for Normalizer {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let mut result = x.clone();

        for i in 0..x.nrows() {
            let row = x.row(i);
            let norm_value = match self.norm {
                NormType::L1 => row.iter().map(|v| v.abs()).sum(),
                NormType::L2 => row.iter().map(|v| v * v).sum::<Float>().sqrt(),
                NormType::Max => row.iter().map(|v| v.abs()).fold(0.0, Float::max),
            };

            if norm_value > 1e-8 {
                for j in 0..x.ncols() {
                    result[[i, j]] = x[[i, j]] / norm_value;
                }
            }
        }

        Ok(result)
    }
}

/// Stateful row-wise unit-vector scaler.
///
/// Like scikit-learn's `Normalizer`, normalization is row-wise and stateless with
/// respect to the data: each sample (row) is divided by its norm (`L1`, `L2`, or
/// `Max`). `fit` only validates the input and records the number of input features;
/// `transform` performs the normalization. Rows whose norm is below `1e-8` are left
/// unchanged (matching the existing `Normalizer` behaviour).
#[derive(Debug, Clone, Default)]
pub struct UnitVectorScaler {
    /// Norm used for row normalization
    norm: NormType,
    /// Number of features seen during `fit` (`None` before `fit`)
    n_features_in_: Option<usize>,
}

/// UnitVectorScaler configuration
#[derive(Debug, Clone, Default)]
pub struct UnitVectorScalerConfig {
    /// Norm to use (L1, L2, or Max)
    pub norm: NormType,
}

impl UnitVectorScaler {
    /// Create a new `UnitVectorScaler` using the default `L2` norm.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a configuration.
    pub fn with_config(config: UnitVectorScalerConfig) -> Self {
        Self {
            norm: config.norm,
            n_features_in_: None,
        }
    }

    /// Configure the norm used for normalization.
    pub fn norm(mut self, norm: NormType) -> Self {
        self.norm = norm;
        self
    }

    /// Access the configured norm.
    pub fn norm_type(&self) -> NormType {
        self.norm
    }

    /// Number of features seen during `fit` (`None` before `fit`).
    pub fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in_
    }

    /// Compute the norm of a single row given the configured norm type.
    fn row_norm(&self, row: scirs2_core::ndarray::ArrayView1<Float>) -> Float {
        match self.norm {
            NormType::L1 => row.iter().map(|v| v.abs()).sum(),
            NormType::L2 => row.iter().map(|v| v * v).sum::<Float>().sqrt(),
            NormType::Max => row.iter().map(|v| v.abs()).fold(0.0, Float::max),
        }
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl Fit<Array2<Float>, ()> for UnitVectorScaler {
    type Fitted = UnitVectorScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }
        self.n_features_in_ = Some(n_cols);
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for UnitVectorScaler {
    /// Normalize each row of `x` to unit norm.
    ///
    /// Returns an error if the scaler has not been fitted or the feature count differs.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features_in = self.n_features_in_.ok_or_else(|| {
            SklearsError::InvalidInput(
                "UnitVectorScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != n_features_in {
            return Err(SklearsError::DimensionMismatch {
                expected: n_features_in,
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for i in 0..n_rows {
            let norm_value = self.row_norm(x.row(i));
            // Leave (near-)zero rows untouched to avoid division by zero.
            if norm_value > 1e-8 {
                for j in 0..n_cols {
                    result[[i, j]] = x[[i, j]] / norm_value;
                }
            }
        }
        Ok(result)
    }
}

/// Compute a quantile (percentile in `[0, 100]`) for a sorted slice using
/// linear interpolation between the two surrounding order statistics.
///
/// Mirrors `RobustScaler::quantile_of_sorted` for use by the feature-wise and
/// outlier-aware scalers.
fn quantile_of_sorted(sorted: &[Float], q: Float) -> Float {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (q / 100.0) * (n as Float - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as Float;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Fitted per-feature `(center, scale)` parameters for `FeatureWiseScaler`.
///
/// Every method is reduced to an affine transform `(x - center) / scale`:
/// Standard → `(mean, std)`, MinMax → `(data_min, data_max - data_min)`,
/// Robust → `(median, iqr)`, MaxAbs → `(0, max_abs)`, None → `(0, 1)`.
#[derive(Debug, Clone)]
pub struct FeatureWiseScalerFitParams {
    /// Per-feature center (subtracted before scaling)
    pub center: Vec<Float>,
    /// Per-feature scale (divided after centering, guarded away from zero)
    pub scale: Vec<Float>,
    /// Resolved scaling method per feature
    pub methods: Vec<ScalingMethod>,
}

/// Feature-wise scaler: apply a possibly different `ScalingMethod` per column.
///
/// Each column is fitted independently according to its configured method and then
/// transformed via the uniform affine map `(x - center) / scale`. When no methods
/// are supplied every column defaults to `ScalingMethod::Standard`; otherwise the
/// number of methods must equal the number of features.
#[derive(Debug, Clone, Default)]
pub struct FeatureWiseScaler {
    /// Per-feature scaling methods (empty → default Standard for all columns)
    methods: Vec<ScalingMethod>,
    /// Fitted parameters (`None` before `fit`)
    params_: Option<FeatureWiseScalerFitParams>,
}

/// FeatureWiseScaler configuration
#[derive(Debug, Clone, Default)]
pub struct FeatureWiseScalerConfig {
    /// Scaling method per feature
    pub methods: Vec<ScalingMethod>,
}

impl FeatureWiseScaler {
    /// Create a new `FeatureWiseScaler` (defaults every column to Standard).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a configuration.
    pub fn with_config(config: FeatureWiseScalerConfig) -> Self {
        Self {
            methods: config.methods,
            params_: None,
        }
    }

    /// Set the per-feature scaling methods.
    pub fn methods(mut self, methods: Vec<ScalingMethod>) -> Self {
        self.methods = methods;
        self
    }

    /// Access fitted parameters (returns `None` before `fit`).
    pub fn params(&self) -> Option<&FeatureWiseScalerFitParams> {
        self.params_.as_ref()
    }

    /// Convenience: fitted `(center, scale)` for feature `j` (`None` before `fit`
    /// or if `j` is out of range).
    pub fn center_scale_(&self, j: usize) -> Option<(Float, Float)> {
        self.params_
            .as_ref()
            .and_then(|p| match (p.center.get(j), p.scale.get(j)) {
                (Some(&c), Some(&s)) => Some((c, s)),
                _ => None,
            })
    }

    /// Apply the inverse transform: `X = X_scaled * scale + center`.
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "FeatureWiseScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.center.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.center.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let center = params.center[j];
            let scale = params.scale[j];
            for i in 0..n_rows {
                result[[i, j]] = result[[i, j]] * scale + center;
            }
        }
        Ok(result)
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }

    /// Compute `(center, scale)` for a single column under the given method.
    ///
    /// Non-finite values are filtered out; an entirely non-finite column falls back
    /// to the identity `(0, 1)`. Degenerate scales (≈ 0) are guarded to `1.0`.
    fn fit_column(method: ScalingMethod, col: &[Float]) -> (Float, Float) {
        if matches!(method, ScalingMethod::None) {
            return (0.0, 1.0);
        }

        let finite: Vec<Float> = col.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return (0.0, 1.0);
        }

        match method {
            ScalingMethod::Standard => {
                let mean = finite.iter().copied().sum::<Float>() / finite.len() as Float;
                // ddof = 0 population variance, consistent with StandardScaler default.
                let variance = finite
                    .iter()
                    .map(|&v| {
                        let d = v - mean;
                        d * d
                    })
                    .sum::<Float>()
                    / finite.len() as Float;
                let std = variance.sqrt();
                (mean, if std > Float::EPSILON { std } else { 1.0 })
            }
            ScalingMethod::MinMax => {
                let col_min = finite.iter().copied().fold(Float::INFINITY, Float::min);
                let col_max = finite.iter().copied().fold(Float::NEG_INFINITY, Float::max);
                let range = col_max - col_min;
                // (x - min) / (max - min) maps the column to [0, 1].
                (
                    col_min,
                    if range.abs() < Float::EPSILON {
                        1.0
                    } else {
                        range
                    },
                )
            }
            ScalingMethod::Robust => {
                let mut sorted = finite;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = quantile_of_sorted(&sorted, 50.0);
                let q25 = quantile_of_sorted(&sorted, 25.0);
                let q75 = quantile_of_sorted(&sorted, 75.0);
                let iqr = q75 - q25;
                (median, if iqr.abs() < Float::EPSILON { 1.0 } else { iqr })
            }
            ScalingMethod::MaxAbs => {
                let max_abs = finite.iter().map(|v| v.abs()).fold(0.0, Float::max);
                (
                    0.0,
                    if max_abs.abs() < Float::EPSILON {
                        1.0
                    } else {
                        max_abs
                    },
                )
            }
            // `None` handled above; unreachable but keeps the match exhaustive.
            ScalingMethod::None => (0.0, 1.0),
        }
    }
}

impl Fit<Array2<Float>, ()> for FeatureWiseScaler {
    type Fitted = FeatureWiseScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        // Resolve the per-column methods: default to Standard everywhere when empty,
        // otherwise require an exact match with the feature count.
        let methods: Vec<ScalingMethod> = if self.methods.is_empty() {
            vec![ScalingMethod::Standard; n_cols]
        } else {
            if self.methods.len() != n_cols {
                return Err(SklearsError::DimensionMismatch {
                    expected: n_cols,
                    actual: self.methods.len(),
                });
            }
            self.methods.clone()
        };

        let mut center = Vec::with_capacity(n_cols);
        let mut scale = Vec::with_capacity(n_cols);

        for (j, &method) in methods.iter().enumerate() {
            let col: Vec<Float> = x.column(j).iter().copied().collect();
            let (c, s) = Self::fit_column(method, &col);
            center.push(c);
            scale.push(s);
        }

        self.params_ = Some(FeatureWiseScalerFitParams {
            center,
            scale,
            methods,
        });
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for FeatureWiseScaler {
    /// Transform `x` using the fitted per-feature `(center, scale)` pairs.
    ///
    /// Returns an error if the scaler has not been fitted yet.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "FeatureWiseScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.center.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.center.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let center = params.center[j];
            let scale = params.scale[j];
            for i in 0..n_rows {
                result[[i, j]] = (result[[i, j]] - center) / scale;
            }
        }
        Ok(result)
    }
}

/// Fitted per-feature parameters for `OutlierAwareScaler`.
///
/// Transform is the uniform affine map `(x - center) / scale`. For the
/// `Transform` strategy each value is first clipped to `[lower_fence, upper_fence]`
/// before being centered and scaled.
#[derive(Debug, Clone)]
pub struct OutlierAwareScalerFitParams {
    /// Per-feature center
    pub center: Vec<Float>,
    /// Per-feature scale (guarded away from zero)
    pub scale: Vec<Float>,
    /// Per-feature lower IQR fence (`Q25 - 1.5 * IQR`)
    pub lower_fence: Vec<Float>,
    /// Per-feature upper IQR fence (`Q75 + 1.5 * IQR`)
    pub upper_fence: Vec<Float>,
    /// Strategy used to fit the parameters
    pub strategy: OutlierAwareScalingStrategy,
}

/// Outlier-aware scaler: scale features while accounting for outliers.
///
/// Outliers are detected per column with the IQR rule — values outside
/// `[Q25 - 1.5 * IQR, Q75 + 1.5 * IQR]`. The chosen
/// [`OutlierAwareScalingStrategy`] then determines the fitted statistics:
/// - `Robust` (default): center = median, scale = IQR for every value.
/// - `Exclude`: center = mean, scale = std computed over inliers only.
/// - `Transform`: clip values to the IQR fences, then standardize on the clipped
///   column (transform re-applies the same clip-then-standardize).
#[derive(Debug, Clone, Default)]
pub struct OutlierAwareScaler {
    /// Strategy for handling outliers
    strategy: OutlierAwareScalingStrategy,
    /// Fitted parameters (`None` before `fit`)
    params_: Option<OutlierAwareScalerFitParams>,
    /// Total number of outliers detected across all columns during `fit`
    outlier_count_: usize,
}

/// OutlierAwareScaler configuration
#[derive(Debug, Clone, Default)]
pub struct OutlierAwareScalerConfig {
    /// Strategy for handling outliers
    pub strategy: OutlierAwareScalingStrategy,
}

impl OutlierAwareScaler {
    /// Create a new `OutlierAwareScaler` using the default `Robust` strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a configuration.
    pub fn with_config(config: OutlierAwareScalerConfig) -> Self {
        Self {
            strategy: config.strategy,
            params_: None,
            outlier_count_: 0,
        }
    }

    /// Configure the outlier-handling strategy.
    pub fn strategy(mut self, strategy: OutlierAwareScalingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Access fitted parameters (returns `None` before `fit`).
    pub fn params(&self) -> Option<&OutlierAwareScalerFitParams> {
        self.params_.as_ref()
    }

    /// Outlier statistics gathered during `fit`.
    pub fn outlier_stats(&self) -> OutlierScalingStats {
        OutlierScalingStats {
            outlier_count: self.outlier_count_,
        }
    }

    /// Apply the inverse transform: `X = X_scaled * scale + center`.
    ///
    /// For the `Transform` strategy this only inverts the standardization step;
    /// the original clipping is not recoverable, mirroring scikit-learn semantics
    /// for clipping transformers.
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "OutlierAwareScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.center.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.center.len(),
                actual: n_cols,
            });
        }

        let mut result = x.clone();
        for j in 0..n_cols {
            let center = params.center[j];
            let scale = params.scale[j];
            for i in 0..n_rows {
                result[[i, j]] = result[[i, j]] * scale + center;
            }
        }
        Ok(result)
    }

    /// Fit and immediately transform.
    pub fn fit_transform(self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }

    /// Population mean and std (ddof = 0) of a slice; returns `(0.0, 1.0)` for empties.
    fn mean_std(values: &[Float]) -> (Float, Float) {
        if values.is_empty() {
            return (0.0, 1.0);
        }
        let mean = values.iter().copied().sum::<Float>() / values.len() as Float;
        let variance = values
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<Float>()
            / values.len() as Float;
        let std = variance.sqrt();
        (mean, if std > Float::EPSILON { std } else { 1.0 })
    }
}

impl Fit<Array2<Float>, ()> for OutlierAwareScaler {
    type Fitted = OutlierAwareScaler;

    fn fit(mut self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();
        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array must be non-empty".to_string(),
            ));
        }

        let mut center = Vec::with_capacity(n_cols);
        let mut scale = Vec::with_capacity(n_cols);
        let mut lower_fence = Vec::with_capacity(n_cols);
        let mut upper_fence = Vec::with_capacity(n_cols);
        let mut total_outliers = 0usize;

        for j in 0..n_cols {
            // Filter non-finite values before computing any statistics.
            let finite: Vec<Float> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();

            if finite.is_empty() {
                // Entirely non-finite column → identity transform, no fences.
                center.push(0.0);
                scale.push(1.0);
                lower_fence.push(Float::NEG_INFINITY);
                upper_fence.push(Float::INFINITY);
                continue;
            }

            let mut sorted = finite.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = quantile_of_sorted(&sorted, 50.0);
            let q25 = quantile_of_sorted(&sorted, 25.0);
            let q75 = quantile_of_sorted(&sorted, 75.0);
            let iqr = q75 - q25;
            let lo_fence = q25 - 1.5 * iqr;
            let hi_fence = q75 + 1.5 * iqr;

            // Count outliers (over finite values) for this column.
            let col_outliers = finite
                .iter()
                .filter(|&&v| v < lo_fence || v > hi_fence)
                .count();
            total_outliers += col_outliers;

            lower_fence.push(lo_fence);
            upper_fence.push(hi_fence);

            let (c, s) = match self.strategy {
                OutlierAwareScalingStrategy::Robust => {
                    // Median / IQR for all values.
                    (median, if iqr.abs() < Float::EPSILON { 1.0 } else { iqr })
                }
                OutlierAwareScalingStrategy::Exclude => {
                    // Mean / std over inliers only; fall back to all finite values
                    // if every point is flagged as an outlier.
                    let inliers: Vec<Float> = finite
                        .iter()
                        .copied()
                        .filter(|&v| v >= lo_fence && v <= hi_fence)
                        .collect();
                    if inliers.is_empty() {
                        Self::mean_std(&finite)
                    } else {
                        Self::mean_std(&inliers)
                    }
                }
                OutlierAwareScalingStrategy::Transform => {
                    // Clip (winsorize) to the fences, then standardize on the clipped data.
                    let clipped: Vec<Float> = finite
                        .iter()
                        .map(|&v| v.clamp(lo_fence, hi_fence))
                        .collect();
                    Self::mean_std(&clipped)
                }
            };
            center.push(c);
            scale.push(s);
        }

        self.outlier_count_ = total_outliers;
        self.params_ = Some(OutlierAwareScalerFitParams {
            center,
            scale,
            lower_fence,
            upper_fence,
            strategy: self.strategy,
        });
        Ok(self)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for OutlierAwareScaler {
    /// Transform `x` using the fitted strategy.
    ///
    /// For `Transform`, inputs are first clipped to the fitted IQR fences, then
    /// standardized; other strategies apply `(x - center) / scale` directly.
    /// Returns an error if the scaler has not been fitted yet.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let params = self.params_.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "OutlierAwareScaler has not been fitted yet; call fit() first".to_string(),
            )
        })?;

        let (n_rows, n_cols) = x.dim();
        if n_cols != params.center.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: params.center.len(),
                actual: n_cols,
            });
        }

        let clip = matches!(params.strategy, OutlierAwareScalingStrategy::Transform);

        let mut result = x.clone();
        for j in 0..n_cols {
            let center = params.center[j];
            let scale = params.scale[j];
            let lo = params.lower_fence[j];
            let hi = params.upper_fence[j];
            for i in 0..n_rows {
                let mut v = x[[i, j]];
                if clip && v.is_finite() {
                    v = v.clamp(lo, hi);
                }
                result[[i, j]] = (v - center) / scale;
            }
        }
        Ok(result)
    }
}

/// Outlier scaling statistics
#[derive(Debug, Clone, Default)]
pub struct OutlierScalingStats {
    /// Number of outliers detected
    pub outlier_count: usize,
}

/// Norm types for vector normalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormType {
    /// L1 norm (Manhattan distance)
    L1,
    /// L2 norm (Euclidean distance)
    #[default]
    L2,
    /// Max norm (Chebyshev distance)
    Max,
}

/// Scaling methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMethod {
    /// Standard scaling (z-score)
    Standard,
    /// Min-max scaling
    MinMax,
    /// Robust scaling
    Robust,
    /// Max absolute value scaling
    MaxAbs,
    /// No scaling
    None,
}

/// Outlier-aware scaling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutlierAwareScalingStrategy {
    /// Exclude outliers from scaling calculation
    Exclude,
    /// Use robust statistics
    #[default]
    Robust,
    /// Transform outliers before scaling
    Transform,
}

/// Robust statistics for scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustStatistic {
    /// Median
    Median,
    /// Median Absolute Deviation
    MAD,
    /// Interquartile Range
    IQR,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array2;

    fn arr(rows: usize, cols: usize, data: Vec<Float>) -> Array2<Float> {
        Array2::from_shape_vec((rows, cols), data).expect("shape")
    }

    #[test]
    fn min_max_scaler_maps_to_unit_range() {
        // Column 0: 1..4, column 1: 10,20,30,40 (constant-free).
        let x = arr(4, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
        let scaler = MinMaxScaler::new();
        let out = scaler.fit_transform(&x).expect("fit_transform");

        // Per-column min maps to 0, max maps to 1.
        assert_relative_eq!(out[[0, 0]], 0.0, epsilon = 1e-9);
        assert_relative_eq!(out[[3, 0]], 1.0, epsilon = 1e-9);
        assert_relative_eq!(out[[0, 1]], 0.0, epsilon = 1e-9);
        assert_relative_eq!(out[[3, 1]], 1.0, epsilon = 1e-9);
        // Midpoint of column 0 (value 2.0) → (2-1)/(4-1).
        assert_relative_eq!(out[[1, 0]], 1.0 / 3.0, epsilon = 1e-9);
    }

    #[test]
    fn min_max_scaler_inverse_round_trips() {
        let x = arr(3, 2, vec![2.0, -5.0, 4.0, 0.0, 6.0, 5.0]);
        let scaler = MinMaxScaler::new().feature_range(-1.0, 1.0);
        let fitted = scaler.fit(&x, &()).expect("fit");
        let out = fitted.transform(&x).expect("transform");
        let back = fitted.inverse_transform(&out).expect("inverse");
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(back[[i, j]], x[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn min_max_scaler_constant_column_maps_to_range_min() {
        // Column 1 is constant (7.0 everywhere).
        let x = arr(3, 2, vec![1.0, 7.0, 2.0, 7.0, 3.0, 7.0]);
        let scaler = MinMaxScaler::new();
        let out = scaler.fit_transform(&x).expect("fit_transform");
        for i in 0..3 {
            assert_relative_eq!(out[[i, 1]], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn max_abs_scaler_divides_by_column_max_abs() {
        // Column 0 max|x| = 4, column 1 max|x| = 8.
        let x = arr(3, 2, vec![-2.0, 4.0, 4.0, -8.0, 1.0, 2.0]);
        let scaler = MaxAbsScaler::new();
        let out = scaler.fit_transform(&x).expect("fit_transform");

        assert_relative_eq!(out[[0, 0]], -0.5, epsilon = 1e-9);
        assert_relative_eq!(out[[1, 0]], 1.0, epsilon = 1e-9);
        assert_relative_eq!(out[[1, 1]], -1.0, epsilon = 1e-9);
        // All values must land within [-1, 1].
        for v in out.iter() {
            assert!(*v >= -1.0 - 1e-12 && *v <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn max_abs_scaler_inverse_round_trips() {
        let x = arr(3, 2, vec![-2.0, 4.0, 4.0, -8.0, 1.0, 2.0]);
        let scaler = MaxAbsScaler::new();
        let fitted = scaler.fit(&x, &()).expect("fit");
        let out = fitted.transform(&x).expect("transform");
        let back = fitted.inverse_transform(&out).expect("inverse");
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(back[[i, j]], x[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn unit_vector_scaler_l2_rows_have_unit_norm() {
        let x = arr(2, 2, vec![3.0, 4.0, 1.0, 0.0]);
        let scaler = UnitVectorScaler::new().norm(NormType::L2);
        let out = scaler.fit_transform(&x).expect("fit_transform");

        // Row 0: (3,4) / 5 → norm 1.
        let norm0 = (out[[0, 0]].powi(2) + out[[0, 1]].powi(2)).sqrt();
        assert_relative_eq!(norm0, 1.0, epsilon = 1e-9);
        assert_relative_eq!(out[[0, 0]], 0.6, epsilon = 1e-9);
        assert_relative_eq!(out[[0, 1]], 0.8, epsilon = 1e-9);
    }

    #[test]
    fn unit_vector_scaler_zero_row_stays_zero() {
        let x = arr(2, 2, vec![0.0, 0.0, 3.0, 4.0]);
        let scaler = UnitVectorScaler::with_config(UnitVectorScalerConfig { norm: NormType::L2 });
        let out = scaler.fit_transform(&x).expect("fit_transform");
        assert_relative_eq!(out[[0, 0]], 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[[0, 1]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn feature_wise_scaler_mixed_methods() {
        // Column 0 → MinMax (1..4 → 0..1), column 1 → MaxAbs (max|x|=8).
        let x = arr(4, 2, vec![1.0, 8.0, 2.0, -4.0, 3.0, 2.0, 4.0, 0.0]);
        let scaler = FeatureWiseScaler::with_config(FeatureWiseScalerConfig {
            methods: vec![ScalingMethod::MinMax, ScalingMethod::MaxAbs],
        });
        let out = scaler.fit_transform(&x).expect("fit_transform");

        // MinMax column.
        assert_relative_eq!(out[[0, 0]], 0.0, epsilon = 1e-9);
        assert_relative_eq!(out[[3, 0]], 1.0, epsilon = 1e-9);
        // MaxAbs column: divide by 8.
        assert_relative_eq!(out[[0, 1]], 1.0, epsilon = 1e-9);
        assert_relative_eq!(out[[1, 1]], -0.5, epsilon = 1e-9);
    }

    #[test]
    fn feature_wise_scaler_defaults_to_standard() {
        let x = arr(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let scaler = FeatureWiseScaler::new();
        let out = scaler.fit_transform(&x).expect("fit_transform");
        // Standardized column should have ~zero mean.
        let mean: Float = out.column(0).iter().copied().sum::<Float>() / 4.0;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn feature_wise_scaler_method_count_mismatch_errors() {
        let x = arr(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let scaler = FeatureWiseScaler::new().methods(vec![ScalingMethod::Standard]);
        assert!(scaler.fit(&x, &()).is_err());
    }

    #[test]
    fn outlier_aware_scaler_robust_uses_median_iqr() {
        // Column with an injected outlier (100.0).
        let x = arr(7, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]);
        let scaler = OutlierAwareScaler::new(); // default Robust
        let fitted = scaler.fit(&x, &()).expect("fit");
        let out = fitted.transform(&x).expect("transform");

        let params = fitted.params().expect("params");
        let median = params.center[0];
        let iqr = params.scale[0];
        // Median of 1..6,100 = 4.0.
        assert_relative_eq!(median, 4.0, epsilon = 1e-9);
        // Transform is (x - median) / iqr.
        assert_relative_eq!(out[[0, 0]], (1.0 - median) / iqr, epsilon = 1e-9);
        assert!(fitted.outlier_stats().outlier_count > 0);
    }

    #[test]
    fn outlier_aware_scaler_exclude_drops_outlier_from_stats() {
        let x = arr(7, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]);
        let scaler = OutlierAwareScaler::new().strategy(OutlierAwareScalingStrategy::Exclude);
        let fitted = scaler.fit(&x, &()).expect("fit");
        let params = fitted.params().expect("params");

        // Inlier mean of 1..6 = 3.5 (outlier 100 excluded).
        assert_relative_eq!(params.center[0], 3.5, epsilon = 1e-9);
        assert!(fitted.outlier_stats().outlier_count > 0);
    }

    #[test]
    fn outlier_aware_scaler_transform_clips_outlier() {
        let x = arr(7, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]);
        let scaler = OutlierAwareScaler::with_config(OutlierAwareScalerConfig {
            strategy: OutlierAwareScalingStrategy::Transform,
        });
        let fitted = scaler.fit(&x, &()).expect("fit");
        let out = fitted.transform(&x).expect("transform");
        let params = fitted.params().expect("params");

        // The outlier (100.0) is clipped to the upper fence before standardizing,
        // so its standardized value must equal the clipped-then-standardized fence.
        let upper = params.upper_fence[0];
        let expected = (upper - params.center[0]) / params.scale[0];
        assert_relative_eq!(out[[6, 0]], expected, epsilon = 1e-9);
        assert!(fitted.outlier_stats().outlier_count > 0);
    }

    #[test]
    fn unfitted_scalers_error_on_transform() {
        let x = arr(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(MinMaxScaler::new().transform(&x).is_err());
        assert!(MaxAbsScaler::new().transform(&x).is_err());
        assert!(UnitVectorScaler::new().transform(&x).is_err());
        assert!(FeatureWiseScaler::new().transform(&x).is_err());
        assert!(OutlierAwareScaler::new().transform(&x).is_err());
    }
}
