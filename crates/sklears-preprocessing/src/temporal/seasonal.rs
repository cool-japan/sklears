//! Seasonal decomposition for time series analysis
//!
//! This module provides seasonal decomposition capabilities to extract
//! trend, seasonal, and residual components from time series data.

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for SeasonalDecomposer
#[derive(Debug, Clone)]
pub struct SeasonalDecomposerConfig {
    /// Season length (e.g., 12 for monthly data, 7 for daily data)
    pub season_length: usize,
    /// Decomposition method (additive or multiplicative)
    pub method: DecompositionMethod,
    /// Whether to include trend component as feature
    pub include_trend: bool,
    /// Whether to include seasonal component as feature
    pub include_seasonal: bool,
    /// Whether to include residual component as feature
    pub include_residual: bool,
    /// Whether to include seasonal strength (seasonality measure)
    pub include_seasonal_strength: bool,
    /// Whether to include trend strength (trend measure)
    pub include_trend_strength: bool,
}

/// Decomposition method for seasonal decomposition
#[derive(Debug, Clone, Copy)]
pub enum DecompositionMethod {
    /// Additive decomposition: y = trend + seasonal + residual
    Additive,
    /// Multiplicative decomposition: y = trend * seasonal * residual
    Multiplicative,
}

impl Default for SeasonalDecomposerConfig {
    fn default() -> Self {
        Self {
            season_length: 12, // Default to monthly seasonality
            method: DecompositionMethod::Additive,
            include_trend: true,
            include_seasonal: true,
            include_residual: false,
            include_seasonal_strength: true,
            include_trend_strength: true,
        }
    }
}

/// SeasonalDecomposer for extracting seasonal patterns and trend from time series
#[derive(Debug, Clone)]
pub struct SeasonalDecomposer<S> {
    config: SeasonalDecomposerConfig,
    n_features_out_: Option<usize>,
    _phantom: PhantomData<S>,
}

impl SeasonalDecomposer<Untrained> {
    /// Create a new SeasonalDecomposer
    pub fn new() -> Self {
        Self {
            config: SeasonalDecomposerConfig::default(),
            n_features_out_: None,
            _phantom: PhantomData,
        }
    }

    /// Set the season length
    pub fn season_length(mut self, season_length: usize) -> Self {
        self.config.season_length = season_length;
        self
    }

    /// Set the decomposition method
    pub fn method(mut self, method: DecompositionMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set whether to include trend component
    pub fn include_trend(mut self, include_trend: bool) -> Self {
        self.config.include_trend = include_trend;
        self
    }

    /// Set whether to include seasonal component
    pub fn include_seasonal(mut self, include_seasonal: bool) -> Self {
        self.config.include_seasonal = include_seasonal;
        self
    }

    /// Set whether to include residual component
    pub fn include_residual(mut self, include_residual: bool) -> Self {
        self.config.include_residual = include_residual;
        self
    }

    /// Set whether to include seasonal strength
    pub fn include_seasonal_strength(mut self, include_seasonal_strength: bool) -> Self {
        self.config.include_seasonal_strength = include_seasonal_strength;
        self
    }

    /// Set whether to include trend strength
    pub fn include_trend_strength(mut self, include_trend_strength: bool) -> Self {
        self.config.include_trend_strength = include_trend_strength;
        self
    }

    /// Calculate number of output features
    fn calculate_n_features_out(&self) -> usize {
        let mut count = 0;
        if self.config.include_trend {
            count += 1;
        }
        if self.config.include_seasonal {
            count += 1;
        }
        if self.config.include_residual {
            count += 1;
        }
        if self.config.include_seasonal_strength {
            count += 1;
        }
        if self.config.include_trend_strength {
            count += 1;
        }
        count
    }
}

impl SeasonalDecomposer<Trained> {
    /// Get the number of output features
    pub fn n_features_out(&self) -> usize {
        self.n_features_out_.expect("Decomposer should be fitted")
    }

    /// Perform classical seasonal decomposition
    fn decompose(
        &self,
        data: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        let n = data.len();
        let season_length = self.config.season_length;

        if n < 2 * season_length {
            return Err(SklearsError::InvalidInput(format!(
                "Data length {} is too short for season length {}",
                n, season_length
            )));
        }

        // Calculate seasonal component using centered moving averages
        let mut trend = Array1::<Float>::zeros(n);
        let mut seasonal = Array1::<Float>::zeros(n);

        // Calculate trend using centered moving average
        let half_season = season_length / 2;
        for i in half_season..(n - half_season) {
            let start = i - half_season;
            let end = i + half_season + 1;
            let sum: Float = data.slice(s![start..end]).sum();
            trend[i] = sum / season_length as Float;
        }

        // Fill trend endpoints using linear extrapolation
        if half_season > 0 {
            let start_slope = (trend[half_season + 1] - trend[half_season]) / 1.0;
            let end_slope = (trend[n - half_season - 1] - trend[n - half_season - 2]) / 1.0;

            for i in 0..half_season {
                trend[i] = trend[half_season] - start_slope * (half_season - i) as Float;
            }
            for i in (n - half_season)..n {
                trend[i] =
                    trend[n - half_season - 1] + end_slope * (i - (n - half_season - 1)) as Float;
            }
        }

        // Calculate seasonal component
        let mut seasonal_averages = vec![0.0; season_length];
        let mut seasonal_counts = vec![0; season_length];

        match self.config.method {
            DecompositionMethod::Additive => {
                // Calculate detrended series
                let detrended: Array1<Float> = data - &trend;

                // Average seasonal effects
                for (i, &value) in detrended.iter().enumerate() {
                    let season_idx = i % season_length;
                    seasonal_averages[season_idx] += value;
                    seasonal_counts[season_idx] += 1;
                }

                for i in 0..season_length {
                    if seasonal_counts[i] > 0 {
                        seasonal_averages[i] /= seasonal_counts[i] as Float;
                    }
                }

                // Center seasonal component (sum should be 0)
                let seasonal_mean =
                    seasonal_averages.iter().sum::<Float>() / season_length as Float;
                for avg in &mut seasonal_averages {
                    *avg -= seasonal_mean;
                }

                // Fill seasonal array
                for i in 0..n {
                    seasonal[i] = seasonal_averages[i % season_length];
                }
            }
            DecompositionMethod::Multiplicative => {
                // Calculate detrended series
                let detrended: Array1<Float> =
                    data.mapv(|x| x.max(1e-10)) / trend.mapv(|x| x.max(1e-10));

                // Average seasonal effects
                for (i, &value) in detrended.iter().enumerate() {
                    let season_idx = i % season_length;
                    seasonal_averages[season_idx] += value;
                    seasonal_counts[season_idx] += 1;
                }

                for i in 0..season_length {
                    if seasonal_counts[i] > 0 {
                        seasonal_averages[i] /= seasonal_counts[i] as Float;
                    }
                }

                // Center seasonal component (geometric mean should be 1)
                let geometric_mean = seasonal_averages.iter().map(|&x| x.ln()).sum::<Float>()
                    / season_length as Float;
                let geometric_mean = geometric_mean.exp();

                for avg in &mut seasonal_averages {
                    *avg /= geometric_mean;
                }

                // Fill seasonal array
                for i in 0..n {
                    seasonal[i] = seasonal_averages[i % season_length];
                }
            }
        }

        // Calculate residual component
        let residual = match self.config.method {
            DecompositionMethod::Additive => data - &trend - &seasonal,
            DecompositionMethod::Multiplicative => {
                let trend_seasonal = &trend * &seasonal;
                data.mapv(|x| x.max(1e-10)) / trend_seasonal.mapv(|x| x.max(1e-10))
            }
        };

        Ok((trend, seasonal, residual))
    }

    /// Calculate seasonal strength (F-statistic based measure)
    fn calculate_seasonal_strength(&self, data: &Array1<Float>, seasonal: &Array1<Float>) -> Float {
        let seasonal_var = seasonal.var(0.0);
        let residual = match self.config.method {
            DecompositionMethod::Additive => data - seasonal,
            DecompositionMethod::Multiplicative => {
                data.mapv(|x| x.max(1e-10)) / seasonal.mapv(|x| x.max(1e-10))
            }
        };
        let residual_var = residual.var(0.0);

        if residual_var > 1e-10 {
            (seasonal_var / residual_var).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Calculate trend strength (coefficient of variation of trend)
    fn calculate_trend_strength(&self, trend: &Array1<Float>) -> Float {
        let trend_mean = trend.mean().unwrap_or(0.0);
        let trend_std = trend.std(0.0);

        if trend_mean.abs() > 1e-10 {
            (trend_std / trend_mean.abs()).min(1.0)
        } else {
            0.0
        }
    }
}

impl Default for SeasonalDecomposer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array1<Float>, ()> for SeasonalDecomposer<Untrained> {
    type Fitted = SeasonalDecomposer<Trained>;

    fn fit(self, x: &Array1<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_features_out = self.calculate_n_features_out();

        if n_features_out == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "feature_selection".to_string(),
                reason: "No features selected for extraction".to_string(),
            });
        }

        if x.len() < 2 * self.config.season_length {
            return Err(SklearsError::InvalidInput(format!(
                "Data length {} is too short for season length {}",
                x.len(),
                self.config.season_length
            )));
        }

        Ok(SeasonalDecomposer {
            config: self.config,
            n_features_out_: Some(n_features_out),
            _phantom: PhantomData,
        })
    }
}

impl Transform<Array1<Float>, Array2<Float>> for SeasonalDecomposer<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array2<Float>> {
        let n_samples = x.len();
        let n_features_out = self.n_features_out();

        let (trend, seasonal, residual) = self.decompose(x)?;

        let mut result = Array2::<Float>::zeros((n_samples, n_features_out));
        let mut feature_idx = 0;

        if self.config.include_trend {
            for (i, &value) in trend.iter().enumerate() {
                result[[i, feature_idx]] = value;
            }
            feature_idx += 1;
        }

        if self.config.include_seasonal {
            for (i, &value) in seasonal.iter().enumerate() {
                result[[i, feature_idx]] = value;
            }
            feature_idx += 1;
        }

        if self.config.include_residual {
            for (i, &value) in residual.iter().enumerate() {
                result[[i, feature_idx]] = value;
            }
            feature_idx += 1;
        }

        if self.config.include_seasonal_strength {
            let seasonal_strength = self.calculate_seasonal_strength(x, &seasonal);
            for i in 0..n_samples {
                result[[i, feature_idx]] = seasonal_strength;
            }
            feature_idx += 1;
        }

        if self.config.include_trend_strength {
            let trend_strength = self.calculate_trend_strength(&trend);
            for i in 0..n_samples {
                result[[i, feature_idx]] = trend_strength;
            }
        }

        Ok(result)
    }
}
