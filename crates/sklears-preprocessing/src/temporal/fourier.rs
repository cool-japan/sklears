//! Fourier-based feature generation for time series
//!
//! This module provides frequency domain feature extraction using
//! Discrete Fourier Transform (DFT) for time series analysis.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for FourierFeatureGenerator
#[derive(Debug, Clone)]
pub struct FourierFeatureGeneratorConfig {
    /// Number of Fourier components to extract
    pub n_components: usize,
    /// Whether to include DC component (frequency 0)
    pub include_dc: bool,
    /// Whether to include phase information
    pub include_phase: bool,
    /// Whether to normalize features
    pub normalize: bool,
}

impl Default for FourierFeatureGeneratorConfig {
    fn default() -> Self {
        Self {
            n_components: 10,
            include_dc: true,
            include_phase: false,
            normalize: true,
        }
    }
}

/// FourierFeatureGenerator for extracting frequency domain features
#[derive(Debug, Clone)]
pub struct FourierFeatureGenerator<S> {
    config: FourierFeatureGeneratorConfig,
    n_features_out_: Option<usize>,
    _phantom: PhantomData<S>,
}

impl FourierFeatureGenerator<Untrained> {
    /// Create a new FourierFeatureGenerator
    pub fn new() -> Self {
        Self {
            config: FourierFeatureGeneratorConfig::default(),
            n_features_out_: None,
            _phantom: PhantomData,
        }
    }

    /// Set the number of Fourier components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set whether to include DC component
    pub fn include_dc(mut self, include_dc: bool) -> Self {
        self.config.include_dc = include_dc;
        self
    }

    /// Set whether to include phase information
    pub fn include_phase(mut self, include_phase: bool) -> Self {
        self.config.include_phase = include_phase;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Calculate number of output features
    fn calculate_n_features_out(&self) -> usize {
        let mut count = self.config.n_components;
        if !self.config.include_dc {
            count = count.saturating_sub(1);
        }
        if self.config.include_phase {
            count *= 2; // magnitude and phase for each component
        }
        count
    }
}

impl FourierFeatureGenerator<Trained> {
    /// Get the number of output features
    pub fn n_features_out(&self) -> usize {
        self.n_features_out_.expect("Generator should be fitted")
    }

    /// Compute Discrete Fourier Transform (simplified implementation)
    fn compute_dft(&self, data: &Array1<Float>) -> Vec<(Float, Float)> {
        let n = data.len();
        let mut result = Vec::new();

        let max_k = self.config.n_components.min(n / 2 + 1);
        let start_k = if self.config.include_dc { 0 } else { 1 };

        for k in start_k..max_k {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (i, &x) in data.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as Float * i as Float / n as Float;
                real_sum += x * angle.cos();
                imag_sum += x * angle.sin();
            }

            let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            let phase = imag_sum.atan2(real_sum);

            result.push((magnitude, phase));
        }

        result
    }
}

impl Default for FourierFeatureGenerator<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array1<Float>, ()> for FourierFeatureGenerator<Untrained> {
    type Fitted = FourierFeatureGenerator<Trained>;

    fn fit(self, x: &Array1<Float>, _y: &()) -> Result<Self::Fitted> {
        let n_features_out = self.calculate_n_features_out();

        if n_features_out == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_components".to_string(),
                reason: "Number of output features must be greater than 0".to_string(),
            });
        }

        if x.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Data must have at least 2 points for Fourier analysis".to_string(),
            ));
        }

        Ok(FourierFeatureGenerator {
            config: self.config,
            n_features_out_: Some(n_features_out),
            _phantom: PhantomData,
        })
    }
}

impl Transform<Array1<Float>, Array1<Float>> for FourierFeatureGenerator<Trained> {
    fn transform(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fourier_components = self.compute_dft(x);
        let mut features = Vec::new();

        for (magnitude, phase) in fourier_components {
            features.push(magnitude);
            if self.config.include_phase {
                features.push(phase);
            }
        }

        let mut result = Array1::from_vec(features);

        // Normalize if requested
        if self.config.normalize {
            let max_val = result.iter().cloned().fold(0.0, Float::max);
            if max_val > 1e-10 {
                result.mapv_inplace(|x| x / max_val);
            }
        }

        Ok(result)
    }
}
