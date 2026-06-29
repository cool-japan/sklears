//! Time series processing pipeline transformers.
//!
//! This module provides real, composable feature-engineering transformers for
//! temporally ordered data. Each input is treated as a feature matrix of shape
//! `(n_timesteps, n_series)` where every column is an independent time series
//! sampled at the same regular cadence and ordered from oldest (row `0`) to
//! newest (last row).
//!
//! Provided transformers:
//!
//! - [`LagFeatures`] — shifts each series by one or more lags to expose past
//!   values as predictive features.
//! - [`RollingWindow`] — trailing rolling-window aggregations (mean, sum, min,
//!   max, standard deviation).
//! - [`Differencing`] — length-preserving regular and/or seasonal differencing
//!   to remove trend and seasonality and induce stationarity.
//! - [`TemporalTrainTestSplit`] — chronological (non-shuffled) train/test split
//!   that never leaks future information into the training window.
//!
//! Initial rows for which a lag/window/difference is undefined are filled with a
//! configurable sentinel (NaN by default), matching the established feature-store
//! convention so downstream steps can drop or impute them explicitly.
//!
//! The canonical 1-D differencing path delegates to
//! [`scirs2_series::transformations::difference_transform`] so behaviour stays
//! consistent with the wider `SciRS2` time-series stack; the matrix transformers
//! are implemented directly to preserve row alignment required for pipeline
//! concatenation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};

use crate::PipelineStep;

/// Aggregation applied over a trailing rolling window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingAggregation {
    /// Arithmetic mean of the window.
    Mean,
    /// Sum of the window.
    Sum,
    /// Minimum value in the window.
    Min,
    /// Maximum value in the window.
    Max,
    /// Sample standard deviation of the window (divisor `n - 1`).
    Std,
}

impl RollingAggregation {
    /// Apply the aggregation to a fully populated window slice.
    fn apply(self, window: &[Float]) -> Float {
        let n = window.len() as Float;
        match self {
            RollingAggregation::Sum => window.iter().copied().sum(),
            RollingAggregation::Mean => window.iter().copied().sum::<Float>() / n,
            RollingAggregation::Min => window.iter().copied().fold(Float::INFINITY, Float::min),
            RollingAggregation::Max => window.iter().copied().fold(Float::NEG_INFINITY, Float::max),
            RollingAggregation::Std => {
                if window.len() < 2 {
                    return Float::NAN;
                }
                let mean = window.iter().copied().sum::<Float>() / n;
                let variance = window
                    .iter()
                    .map(|&v| {
                        let d = v - mean;
                        d * d
                    })
                    .sum::<Float>()
                    / (n - 1.0);
                variance.sqrt()
            }
        }
    }
}

/// Lag-feature transformer.
///
/// For each requested lag `k` and each input column, emits a column whose value
/// at row `t` is the input value at row `t - k`. The first `k` rows of every
/// generated lag column are filled with `fill_value`.
///
/// The output column order is *(lag-major, then original column-major)*: for
/// lags `[1, 2]` and input columns `[c0, c1]` the output is
/// `[c0_lag1, c1_lag1, c0_lag2, c1_lag2]`.
#[derive(Debug, Clone)]
pub struct LagFeatures {
    lags: Vec<usize>,
    fill_value: Float,
}

impl LagFeatures {
    /// Create a lag-feature transformer for the given strictly-positive lags.
    ///
    /// Returns an error if `lags` is empty or contains a zero lag (a zero lag is
    /// just the identity column and would be ambiguous as a "lag" feature).
    pub fn new(lags: Vec<usize>) -> SklResult<Self> {
        if lags.is_empty() {
            return Err(SklearsError::InvalidInput(
                "LagFeatures requires at least one lag".to_string(),
            ));
        }
        if lags.contains(&0) {
            return Err(SklearsError::InvalidInput(
                "LagFeatures lags must be strictly positive".to_string(),
            ));
        }
        Ok(Self {
            lags,
            fill_value: Float::NAN,
        })
    }

    /// Set the value used to fill rows where the lag is undefined.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: Float) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// Compute lag features for a 2-D input without requiring a fit step.
    pub fn lag(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        if n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "LagFeatures requires at least one input column".to_string(),
            ));
        }

        let out_cols = self.lags.len() * n_cols;
        let mut out = Array2::from_elem((n_rows, out_cols), self.fill_value);

        for (lag_idx, &lag) in self.lags.iter().enumerate() {
            let col_base = lag_idx * n_cols;
            for row in lag..n_rows {
                for col in 0..n_cols {
                    out[[row, col_base + col]] = x[[row - lag, col]];
                }
            }
        }
        Ok(out)
    }

    /// Number of output features produced for `n_input_cols` input columns.
    #[must_use]
    pub fn n_output_features(&self, n_input_cols: usize) -> usize {
        self.lags.len() * n_input_cols
    }
}

/// Rolling-window aggregation transformer.
///
/// Emits, for each input column, a column holding the trailing rolling
/// aggregation over a window of `window` consecutive rows ending at the current
/// row (inclusive). Rows with fewer than `min_periods` observations available
/// are filled with `fill_value`.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    window: usize,
    aggregation: RollingAggregation,
    min_periods: usize,
    fill_value: Float,
}

impl RollingWindow {
    /// Create a rolling-window transformer.
    ///
    /// `window` must be at least 1. `min_periods` defaults to `window` (only
    /// fully populated windows produce a value); use [`RollingWindow::with_min_periods`]
    /// to allow partial leading windows.
    pub fn new(window: usize, aggregation: RollingAggregation) -> SklResult<Self> {
        if window == 0 {
            return Err(SklearsError::InvalidInput(
                "RollingWindow window must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            window,
            aggregation,
            min_periods: window,
            fill_value: Float::NAN,
        })
    }

    /// Set the minimum number of observations in a window required to emit a
    /// value. Clamped to `1..=window`.
    pub fn with_min_periods(mut self, min_periods: usize) -> SklResult<Self> {
        if min_periods == 0 || min_periods > self.window {
            return Err(SklearsError::InvalidInput(format!(
                "min_periods must be in 1..={}",
                self.window
            )));
        }
        self.min_periods = min_periods;
        Ok(self)
    }

    /// Set the fill value used before `min_periods` observations are available.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: Float) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// Compute rolling aggregations for a 2-D input.
    pub fn roll(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        if n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "RollingWindow requires at least one input column".to_string(),
            ));
        }

        let mut out = Array2::from_elem((n_rows, n_cols), self.fill_value);
        let mut buffer: Vec<Float> = Vec::with_capacity(self.window);

        for col in 0..n_cols {
            for row in 0..n_rows {
                let start = row + 1 - (row + 1).min(self.window);
                let available = row - start + 1;
                if available < self.min_periods {
                    continue;
                }
                buffer.clear();
                for r in start..=row {
                    buffer.push(x[[r, col]]);
                }
                out[[row, col]] = self.aggregation.apply(&buffer);
            }
        }
        Ok(out)
    }
}

/// Length-preserving differencing transformer.
///
/// Applies optional seasonal differencing (`x[t] - x[t - seasonal_lag]`) of the
/// requested seasonal order, followed by regular differencing (`x[t] - x[t-1]`)
/// of the requested order. Unlike the shrinking 1-D primitive in `scirs2-series`,
/// this keeps the original row count by filling the leading undefined rows with
/// `fill_value`, so the result can be concatenated alongside other features.
#[derive(Debug, Clone)]
pub struct Differencing {
    order: usize,
    seasonal_lag: Option<usize>,
    seasonal_order: usize,
    fill_value: Float,
}

impl Differencing {
    /// Create a regular differencing transformer of the given order.
    ///
    /// `order` must be at least 1.
    pub fn new(order: usize) -> SklResult<Self> {
        if order == 0 {
            return Err(SklearsError::InvalidInput(
                "Differencing order must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            order,
            seasonal_lag: None,
            seasonal_order: 0,
            fill_value: Float::NAN,
        })
    }

    /// Add seasonal differencing with the given lag (e.g. 12 for monthly data
    /// with yearly seasonality) and seasonal order (number of times applied).
    pub fn with_seasonal(mut self, seasonal_lag: usize, seasonal_order: usize) -> SklResult<Self> {
        if seasonal_lag == 0 {
            return Err(SklearsError::InvalidInput(
                "seasonal_lag must be strictly positive".to_string(),
            ));
        }
        self.seasonal_lag = Some(seasonal_lag);
        self.seasonal_order = seasonal_order;
        Ok(self)
    }

    /// Set the fill value for the leading undefined rows.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: Float) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// Number of leading rows that are undefined (and hence filled) after the
    /// configured differencing.
    #[must_use]
    pub fn warmup_rows(&self) -> usize {
        self.order + self.seasonal_lag.map_or(0, |lag| lag * self.seasonal_order)
    }

    /// Difference a single column in place over `[0, valid_len)`, returning the
    /// number of leading rows that became undefined.
    fn difference_column(&self, column: &mut [Float]) -> usize {
        let mut undefined = 0usize;

        // Seasonal differencing first.
        if let (Some(lag), order) = (self.seasonal_lag, self.seasonal_order) {
            for _ in 0..order {
                for t in (lag..column.len()).rev() {
                    column[t] -= column[t - lag];
                }
                undefined = (undefined + lag).min(column.len());
            }
        }

        // Regular differencing.
        for _ in 0..self.order {
            for t in (1..column.len()).rev() {
                column[t] -= column[t - 1];
            }
            undefined = (undefined + 1).min(column.len());
        }

        undefined
    }

    /// Apply length-preserving differencing to a 2-D input.
    pub fn difference(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        if n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Differencing requires at least one input column".to_string(),
            ));
        }
        if n_rows <= self.warmup_rows() {
            return Err(SklearsError::InvalidInput(format!(
                "Series length {n_rows} is too short for differencing warmup {}",
                self.warmup_rows()
            )));
        }

        let mut out = x.to_owned();
        for col in 0..n_cols {
            let mut column: Vec<Float> = out.column(col).to_vec();
            let undefined = self.difference_column(&mut column);
            for (row, value) in column.iter().enumerate() {
                out[[row, col]] = if row < undefined {
                    self.fill_value
                } else {
                    *value
                };
            }
        }
        Ok(out)
    }

    /// Canonical 1-D differencing that delegates to `scirs2-series`.
    ///
    /// Returns the (shrunk) differenced series exactly as the wider `SciRS2`
    /// time-series stack computes it (length reduced by the warmup). Use this
    /// when you need stack-consistent output rather than the length-preserving
    /// matrix variant.
    pub fn difference_series(&self, series: &ArrayView1<'_, Float>) -> SklResult<Array1<Float>> {
        let owned = series.to_owned();
        // Apply seasonal differencing of the requested seasonal order, then the
        // regular order, mirroring `difference_column` ordering.
        let mut current = owned;
        if let (Some(lag), order) = (self.seasonal_lag, self.seasonal_order) {
            for _ in 0..order {
                let (diffed, _params) = scirs2_series::transformations::difference_transform::<
                    Float,
                    _,
                >(&current, 0, Some(lag))
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?;
                current = diffed;
            }
        }
        let (diffed, _params) = scirs2_series::transformations::difference_transform::<Float, _>(
            &current, self.order, None,
        )
        .map_err(|e| SklearsError::InvalidInput(e.to_string()))?;
        Ok(diffed)
    }
}

/// Result of a chronological train/test split.
#[derive(Debug, Clone)]
pub struct TemporalSplit {
    /// Inclusive-exclusive row range `[0, train_end)` used for training.
    pub train_range: (usize, usize),
    /// Inclusive-exclusive row range `[train_end + gap, n)` used for testing.
    pub test_range: (usize, usize),
}

impl TemporalSplit {
    /// Number of training rows.
    #[must_use]
    pub fn train_len(&self) -> usize {
        self.train_range.1 - self.train_range.0
    }

    /// Number of test rows.
    #[must_use]
    pub fn test_len(&self) -> usize {
        self.test_range.1 - self.test_range.0
    }

    /// Slice the training rows out of a feature matrix.
    #[must_use]
    pub fn train_rows(&self, x: &ArrayView2<'_, Float>) -> Array2<Float> {
        x.slice(scirs2_core::ndarray::s![
            self.train_range.0..self.train_range.1,
            ..
        ])
        .to_owned()
    }

    /// Slice the test rows out of a feature matrix.
    #[must_use]
    pub fn test_rows(&self, x: &ArrayView2<'_, Float>) -> Array2<Float> {
        x.slice(scirs2_core::ndarray::s![
            self.test_range.0..self.test_range.1,
            ..
        ])
        .to_owned()
    }
}

/// Chronological train/test splitter for temporally ordered data.
///
/// Splits rows in time order so that the test window always lies strictly after
/// the training window. An optional `gap` of rows between the two windows can be
/// used to mimic a forecasting horizon and avoid leakage from overlapping
/// lag/rolling features straddling the boundary.
#[derive(Debug, Clone)]
pub struct TemporalTrainTestSplit {
    test_fraction: f64,
    gap: usize,
}

impl TemporalTrainTestSplit {
    /// Create a splitter reserving `test_fraction` of the most recent rows for
    /// testing. `test_fraction` must lie in the open interval `(0, 1)`.
    pub fn new(test_fraction: f64) -> SklResult<Self> {
        if !(test_fraction > 0.0 && test_fraction < 1.0) {
            return Err(SklearsError::InvalidInput(
                "test_fraction must be in the open interval (0, 1)".to_string(),
            ));
        }
        Ok(Self {
            test_fraction,
            gap: 0,
        })
    }

    /// Insert a gap of `gap` rows between the train and test windows.
    #[must_use]
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Compute the split for a series of `n_rows` observations.
    pub fn split(&self, n_rows: usize) -> SklResult<TemporalSplit> {
        if n_rows == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot split an empty series".to_string(),
            ));
        }

        let test_len = ((n_rows as f64) * self.test_fraction).round() as usize;
        let test_len = test_len.clamp(1, n_rows.saturating_sub(1));
        let test_start = n_rows - test_len;
        // The training window ends `gap` rows before the test window begins.
        let train_end = test_start.saturating_sub(self.gap);

        if train_end == 0 {
            return Err(SklearsError::InvalidInput(
                "Configured test_fraction and gap leave no training rows".to_string(),
            ));
        }

        Ok(TemporalSplit {
            train_range: (0, train_end),
            test_range: (test_start, n_rows),
        })
    }
}

// --- sklears Estimator / Fit / Transform / PipelineStep integration ---------

/// Configuration shared by the stateless time-series transformers.
///
/// These transformers do not learn parameters from data; the configuration is
/// fixed at construction time. The struct exists to satisfy the
/// [`Estimator`] contract used throughout the pipeline machinery.
#[derive(Debug, Clone, Default)]
pub struct TimeSeriesTransformConfig;

macro_rules! impl_ts_estimator {
    ($ty:ty) => {
        impl Estimator for $ty {
            type Config = TimeSeriesTransformConfig;
            type Error = SklearsError;
            type Float = Float;

            fn config(&self) -> &Self::Config {
                static CONFIG: TimeSeriesTransformConfig = TimeSeriesTransformConfig;
                &CONFIG
            }
        }
    };
}

impl_ts_estimator!(LagFeatures);
impl_ts_estimator!(RollingWindow);
impl_ts_estimator!(Differencing);

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for LagFeatures {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.lag(x)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for RollingWindow {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.roll(x)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for Differencing {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.difference(x)
    }
}

/// The stateless transformers are trivially fittable: fitting is a no-op that
/// simply returns `self`, since all behaviour is fixed at construction.
macro_rules! impl_ts_fit {
    ($ty:ty) => {
        impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for $ty {
            type Fitted = $ty;

            fn fit(
                self,
                _x: &ArrayView2<'_, Float>,
                _y: &Option<&ArrayView1<'_, Float>>,
            ) -> SklResult<Self::Fitted> {
                Ok(self)
            }
        }
    };
}

impl_ts_fit!(LagFeatures);
impl_ts_fit!(RollingWindow);
impl_ts_fit!(Differencing);

macro_rules! impl_ts_pipeline_step {
    ($ty:ty) => {
        impl PipelineStep for $ty {
            fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
                Transform::transform(self, x)
            }

            fn fit(
                &mut self,
                _x: &ArrayView2<'_, Float>,
                _y: Option<&ArrayView1<'_, Float>>,
            ) -> SklResult<()> {
                Ok(())
            }

            fn clone_step(&self) -> Box<dyn PipelineStep> {
                Box::new(self.clone())
            }
        }
    };
}

impl_ts_pipeline_step!(LagFeatures);
impl_ts_pipeline_step!(RollingWindow);
impl_ts_pipeline_step!(Differencing);

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn lag_features_shift_columns() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let lagger = LagFeatures::new(vec![1])
            .expect("valid lag")
            .with_fill_value(0.0);
        let out = lagger.lag(&x.view()).expect("lag ok");
        assert_eq!(out.shape(), &[4, 1]);
        assert_eq!(out[[0, 0]], 0.0); // undefined -> fill
        assert_eq!(out[[1, 0]], 1.0);
        assert_eq!(out[[2, 0]], 2.0);
        assert_eq!(out[[3, 0]], 3.0);
    }

    #[test]
    fn lag_features_multiple_lags_column_layout() {
        let x = array![[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]];
        let lagger = LagFeatures::new(vec![1, 2])
            .expect("valid")
            .with_fill_value(0.0);
        let out = lagger.lag(&x.view()).expect("lag ok");
        // Columns: [c0_lag1, c1_lag1, c0_lag2, c1_lag2]
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(out[[2, 0]], 20.0); // c0 lag1 at t=2 -> x[1,0]
        assert_eq!(out[[2, 1]], 200.0); // c1 lag1 at t=2 -> x[1,1]
        assert_eq!(out[[2, 2]], 10.0); // c0 lag2 at t=2 -> x[0,0]
        assert_eq!(out[[2, 3]], 100.0); // c1 lag2 at t=2 -> x[0,1]
    }

    #[test]
    fn lag_features_reject_zero_lag() {
        assert!(LagFeatures::new(vec![0]).is_err());
        assert!(LagFeatures::new(vec![]).is_err());
    }

    #[test]
    fn rolling_mean_trailing_window() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let roller = RollingWindow::new(3, RollingAggregation::Mean)
            .expect("valid")
            .with_fill_value(0.0);
        let out = roller.roll(&x.view()).expect("roll ok");
        assert_eq!(out[[0, 0]], 0.0); // <3 obs
        assert_eq!(out[[1, 0]], 0.0); // <3 obs
        assert!((out[[2, 0]] - 2.0).abs() < 1e-12); // mean(1,2,3)
        assert!((out[[3, 0]] - 3.0).abs() < 1e-12); // mean(2,3,4)
        assert!((out[[4, 0]] - 4.0).abs() < 1e-12); // mean(3,4,5)
    }

    #[test]
    fn rolling_min_max_sum() {
        let x = array![[3.0], [1.0], [2.0], [5.0]];
        let min = RollingWindow::new(2, RollingAggregation::Min)
            .expect("v")
            .with_fill_value(0.0)
            .roll(&x.view())
            .expect("ok");
        assert_eq!(min[[1, 0]], 1.0); // min(3,1)
        assert_eq!(min[[3, 0]], 2.0); // min(2,5)

        let max = RollingWindow::new(2, RollingAggregation::Max)
            .expect("v")
            .with_fill_value(0.0)
            .roll(&x.view())
            .expect("ok");
        assert_eq!(max[[3, 0]], 5.0);

        let sum = RollingWindow::new(2, RollingAggregation::Sum)
            .expect("v")
            .with_fill_value(0.0)
            .roll(&x.view())
            .expect("ok");
        assert_eq!(sum[[2, 0]], 3.0); // 1+2
    }

    #[test]
    fn rolling_std_matches_sample_formula() {
        let x = array![[2.0], [4.0], [6.0]];
        let std = RollingWindow::new(3, RollingAggregation::Std)
            .expect("v")
            .roll(&x.view())
            .expect("ok");
        // sample std of [2,4,6] = sqrt(((2-4)^2+(4-4)^2+(6-4)^2)/2) = sqrt(4) = 2
        assert!((std[[2, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn differencing_first_order_preserves_length() {
        let x = array![[1.0], [3.0], [6.0], [10.0]];
        let diff = Differencing::new(1).expect("v").with_fill_value(0.0);
        let out = diff.difference(&x.view()).expect("ok");
        assert_eq!(out.shape(), &[4, 1]);
        assert_eq!(out[[0, 0]], 0.0); // warmup fill
        assert_eq!(out[[1, 0]], 2.0); // 3-1
        assert_eq!(out[[2, 0]], 3.0); // 6-3
        assert_eq!(out[[3, 0]], 4.0); // 10-6
    }

    #[test]
    fn differencing_seasonal_then_regular() {
        // 8 points, seasonal lag 4, then first difference.
        let x = array![[1.0], [2.0], [3.0], [4.0], [10.0], [20.0], [30.0], [40.0]];
        let diff = Differencing::new(1)
            .expect("v")
            .with_seasonal(4, 1)
            .expect("v")
            .with_fill_value(0.0);
        let out = diff.difference(&x.view()).expect("ok");
        assert_eq!(out.shape(), &[8, 1]);
        // First 5 rows are warmup (seasonal lag 4 + order 1).
        for r in 0..5 {
            assert_eq!(out[[r, 0]], 0.0);
        }
        // Seasonal diff: x[t]-x[t-4] => [.,.,.,., 9,18,27,36]; first diff of that
        // at t=5 -> 18-9=9, t=6 -> 27-18=9, t=7 -> 36-27=9.
        assert!((out[[5, 0]] - 9.0).abs() < 1e-12);
        assert!((out[[6, 0]] - 9.0).abs() < 1e-12);
        assert!((out[[7, 0]] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn differencing_series_matches_scirs2_series() {
        let series = array![1.0, 3.0, 6.0, 10.0, 15.0];
        let diff = Differencing::new(1).expect("v");
        let out = diff.difference_series(&series.view()).expect("ok");
        // scirs2-series shrinks: [2,3,4,5]
        assert_eq!(out.len(), 4);
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[3] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn differencing_rejects_too_short_series() {
        let x = array![[1.0], [2.0]];
        let diff = Differencing::new(1)
            .expect("v")
            .with_seasonal(4, 1)
            .expect("v");
        assert!(diff.difference(&x.view()).is_err());
    }

    #[test]
    fn temporal_split_is_chronological() {
        let split = TemporalTrainTestSplit::new(0.25).expect("v");
        let s = split.split(8).expect("ok");
        // 25% of 8 = 2 test rows.
        assert_eq!(s.train_range, (0, 6));
        assert_eq!(s.test_range, (6, 8));
        assert_eq!(s.train_len(), 6);
        assert_eq!(s.test_len(), 2);
        // Train strictly precedes test.
        assert!(s.train_range.1 <= s.test_range.0);
    }

    #[test]
    fn temporal_split_with_gap() {
        let split = TemporalTrainTestSplit::new(0.2).expect("v").with_gap(2);
        let s = split.split(10).expect("ok");
        // 20% of 10 = 2 test rows -> test starts at 8; gap 2 -> train ends at 6.
        assert_eq!(s.test_range, (8, 10));
        assert_eq!(s.train_range, (0, 6));
    }

    #[test]
    fn temporal_split_rows_slicing() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let split = TemporalTrainTestSplit::new(0.25).expect("v");
        let s = split.split(4).expect("ok");
        let train = s.train_rows(&x.view());
        let test = s.test_rows(&x.view());
        assert_eq!(train.nrows(), 3);
        assert_eq!(test.nrows(), 1);
        assert_eq!(test[[0, 0]], 3.0);
    }

    #[test]
    fn temporal_split_rejects_invalid_fraction() {
        assert!(TemporalTrainTestSplit::new(0.0).is_err());
        assert!(TemporalTrainTestSplit::new(1.0).is_err());
        assert!(TemporalTrainTestSplit::new(-0.1).is_err());
    }

    #[test]
    fn lag_features_as_pipeline_step() {
        let x = array![[1.0], [2.0], [3.0]];
        let step: Box<dyn PipelineStep> =
            Box::new(LagFeatures::new(vec![1]).expect("v").with_fill_value(-1.0));
        let out = step.transform(&x.view()).expect("ok");
        assert_eq!(out[[0, 0]], -1.0);
        assert_eq!(out[[1, 0]], 1.0);
    }
}
