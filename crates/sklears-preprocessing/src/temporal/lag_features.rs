//! Lag feature generation for time series data
//!
//! This module provides utilities for creating lag features from time series data,
//! which are essential for many temporal modeling tasks.

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for LagFeatureGenerator
#[derive(Debug, Clone)]
pub struct LagFeatureGeneratorConfig {
    /// Number of lag periods to generate
    pub lags: Vec<usize>,
    /// Whether to drop missing values (first n rows where lags are not available)
    pub drop_na: bool,
    /// Fill value for missing lag values
    pub fill_value: Option<Float>,
}

impl Default for LagFeatureGeneratorConfig {
    fn default() -> Self {
        Self {
            lags: vec![1, 2, 3], // Default to 3 lags
            drop_na: false,
            fill_value: Some(0.0),
        }
    }
}

/// LagFeatureGenerator for creating lag features from time series data
#[derive(Debug, Clone)]
pub struct LagFeatureGenerator<S> {
    config: LagFeatureGeneratorConfig,
    n_features_out_: Option<usize>,
    _phantom: PhantomData<S>,
}

impl LagFeatureGenerator<Untrained> {
    /// Create a new LagFeatureGenerator
    pub fn new() -> Self {
        Self {
            config: LagFeatureGeneratorConfig::default(),
            n_features_out_: None,
            _phantom: PhantomData,
        }
    }

    /// Set the lag periods
    pub fn lags(mut self, lags: Vec<usize>) -> Self {
        self.config.lags = lags;
        self
    }

    /// Set whether to drop missing values
    pub fn drop_na(mut self, drop_na: bool) -> Self {
        self.config.drop_na = drop_na;
        self
    }

    /// Set the fill value for missing lag values
    pub fn fill_value(mut self, fill_value: Float) -> Self {
        self.config.fill_value = Some(fill_value);
        self
    }
}

impl LagFeatureGenerator<Trained> {
    /// Get the number of output features
    pub fn n_features_out(&self) -> usize {
        self.n_features_out_.expect("Generator should be fitted")
    }
}

impl Default for LagFeatureGenerator<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for LagFeatureGenerator<Untrained> {
    type Fitted = LagFeatureGenerator<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_input_features) = x.dim();
        let n_features_out = n_input_features * (self.config.lags.len() + 1); // Original + lags

        if self.config.lags.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "lags".to_string(),
                reason: "At least one lag must be specified".to_string(),
            });
        }

        Ok(LagFeatureGenerator {
            config: self.config,
            n_features_out_: Some(n_features_out),
            _phantom: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for LagFeatureGenerator<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let max_lag = *self.config.lags.iter().max().unwrap_or(&0);

        if max_lag >= n_samples {
            return Err(SklearsError::InvalidInput(
                "Maximum lag cannot be greater than or equal to number of samples".to_string(),
            ));
        }

        let n_features_out = self.n_features_out();
        let mut result = Array2::<Float>::zeros((n_samples, n_features_out));

        // Copy original features
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = x[[i, j]];
            }
        }

        // Generate lag features
        let mut feature_idx = n_features;
        for &lag in &self.config.lags {
            for j in 0..n_features {
                for i in 0..n_samples {
                    if i >= lag {
                        result[[i, feature_idx]] = x[[i - lag, j]];
                    } else {
                        // Fill missing values
                        result[[i, feature_idx]] = self.config.fill_value.unwrap_or(0.0);
                    }
                }
                feature_idx += 1;
            }
        }

        // Drop rows with missing values if requested
        if self.config.drop_na && max_lag > 0 {
            let valid_rows = n_samples - max_lag;
            let mut trimmed_result = Array2::<Float>::zeros((valid_rows, n_features_out));
            for i in 0..valid_rows {
                for j in 0..n_features_out {
                    trimmed_result[[i, j]] = result[[i + max_lag, j]];
                }
            }
            Ok(trimmed_result)
        } else {
            Ok(result)
        }
    }
}
