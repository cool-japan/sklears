//! Composite signal feature extractors and utility functions
//!
//! This module provides high-level signal processing tools including:
//! - Comprehensive signal feature extraction combining multiple methods
//! - Statistical, spectral, and temporal feature extraction
//! - Utility functions for windowing and correlation analysis
//! - Configurable feature selection and extraction pipelines

use scirs2_core::ndarray::{s, Array1, ArrayView1};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use std::f64::consts::PI;

use super::frequency_analysis::FrequencyDomainExtractor;

/// Comprehensive signal feature extractor
///
/// Combines multiple feature extraction methods to provide a comprehensive
/// characterization of signal properties across different domains.
#[derive(Debug, Clone)]
pub struct SignalFeatureExtractor {
    include_statistical: bool,
    include_spectral: bool,
    include_temporal: bool,
    n_fft: usize,
    sample_rate: f64,
}

impl SignalFeatureExtractor {
    /// Create a new comprehensive signal feature extractor
    ///
    /// Default configuration:
    /// - Include statistical features: true
    /// - Include spectral features: true
    /// - Include temporal features: true
    /// - FFT size: 1024
    /// - Sample rate: 250.0 Hz
    pub fn new() -> Self {
        Self {
            include_statistical: true,
            include_spectral: true,
            include_temporal: true,
            n_fft: 1024,
            sample_rate: 250.0,
        }
    }

    /// Set whether to include statistical features
    ///
    /// Statistical features include: mean, std, min, max, range, skewness, kurtosis
    pub fn include_statistical(mut self, include: bool) -> Self {
        self.include_statistical = include;
        self
    }

    /// Set whether to include spectral features
    ///
    /// Spectral features include: frequency domain analysis, band powers,
    /// spectral statistics, peak frequency, spectral edge
    pub fn include_spectral(mut self, include: bool) -> Self {
        self.include_spectral = include;
        self
    }

    /// Set whether to include temporal features
    ///
    /// Temporal features include: zero crossing rate, RMS energy, peak-to-peak amplitude
    pub fn include_temporal(mut self, include: bool) -> Self {
        self.include_temporal = include;
        self
    }

    /// Alias for include_temporal
    pub fn include_time_domain(mut self, include: bool) -> Self {
        self.include_temporal = include;
        self
    }

    /// Alias for include_spectral
    pub fn include_frequency_domain(mut self, include: bool) -> Self {
        self.include_spectral = include;
        self
    }

    /// Set the FFT size for spectral analysis
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the sample rate for frequency analysis
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Extract comprehensive features from signal
    ///
    /// Returns a feature vector combining statistical, spectral, and temporal features
    /// based on the configuration. The exact number and order of features depends on
    /// which feature types are enabled.
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let mut features = Vec::new();

        if self.include_statistical {
            let stats = self.extract_statistical_features(signal);
            features.extend_from_slice(&stats);
        }

        if self.include_spectral {
            let spectral = self.extract_spectral_features(signal)?;
            features.extend_from_slice(&spectral);
        }

        if self.include_temporal {
            let temporal = self.extract_temporal_features(signal);
            features.extend_from_slice(&temporal);
        }

        Ok(Array1::from_vec(features))
    }

    /// Extract statistical features from signal
    ///
    /// Computes basic statistical measures that characterize the distribution
    /// and shape of the signal values.
    fn extract_statistical_features(&self, signal: &ArrayView1<f64>) -> Vec<f64> {
        let mean = signal.mean().unwrap_or(0.0);
        let std = signal.std(0.0);
        let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        // Skewness and kurtosis approximations
        let variance = std * std;
        let mut skewness = 0.0;
        let mut kurtosis = 0.0;

        if variance > 0.0 {
            for &val in signal.iter() {
                let normalized = (val - mean) / std;
                skewness += normalized.powi(3);
                kurtosis += normalized.powi(4);
            }
            skewness /= signal.len() as f64;
            kurtosis = kurtosis / signal.len() as f64 - 3.0; // Excess kurtosis
        }

        vec![mean, std, min, max, range, skewness, kurtosis]
    }

    /// Extract spectral features from signal
    ///
    /// Uses frequency domain analysis to extract features related to the
    /// signal's frequency content and spectral characteristics.
    fn extract_spectral_features(&self, signal: &ArrayView1<f64>) -> SklResult<Vec<f64>> {
        let freq_extractor = FrequencyDomainExtractor::new()
            .n_fft(self.n_fft)
            .sample_rate(self.sample_rate);

        let freq_features = freq_extractor.extract_features(signal)?;
        Ok(freq_features.to_vec())
    }

    /// Extract temporal features from signal
    ///
    /// Computes features that characterize the signal's temporal dynamics
    /// and time-domain properties.
    fn extract_temporal_features(&self, signal: &ArrayView1<f64>) -> Vec<f64> {
        let mut features = Vec::new();

        // Zero crossing rate
        let mut zero_crossings = 0;
        for i in 1..signal.len() {
            if (signal[i - 1] >= 0.0) != (signal[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zcr = zero_crossings as f64 / (signal.len() - 1) as f64;
        features.push(zcr);

        // RMS energy
        let rms = (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt();
        features.push(rms);

        // Peak-to-peak amplitude
        let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        features.push(max - min);

        features
    }

    /// Get the expected number of features based on current configuration
    pub fn feature_count(&self) -> usize {
        let mut count = 0;

        if self.include_statistical {
            count += 7; // mean, std, min, max, range, skewness, kurtosis
        }

        if self.include_spectral {
            count += 11; // Based on FrequencyDomainExtractor default output
        }

        if self.include_temporal {
            count += 3; // zero crossing rate, RMS, peak-to-peak
        }

        count
    }

    /// Get feature names for the current configuration
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        if self.include_statistical {
            names.extend_from_slice(&[
                "mean".to_string(),
                "std".to_string(),
                "min".to_string(),
                "max".to_string(),
                "range".to_string(),
                "skewness".to_string(),
                "kurtosis".to_string(),
            ]);
        }

        if self.include_spectral {
            // Names based on FrequencyDomainExtractor default bands
            names.extend_from_slice(&[
                "delta_power".to_string(),
                "theta_power".to_string(),
                "alpha_power".to_string(),
                "beta_power".to_string(),
                "gamma_power".to_string(),
                "spectral_mean".to_string(),
                "spectral_std".to_string(),
                "spectral_skewness".to_string(),
                "spectral_kurtosis".to_string(),
                "peak_frequency".to_string(),
                "spectral_edge".to_string(),
            ]);
        }

        if self.include_temporal {
            names.extend_from_slice(&[
                "zero_crossing_rate".to_string(),
                "rms_energy".to_string(),
                "peak_to_peak".to_string(),
            ]);
        }

        names
    }
}

impl Default for SignalFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply window function to signal
///
/// Applies various window functions to reduce spectral leakage in frequency analysis.
/// Commonly used before FFT operations to improve spectral estimation quality.
pub fn apply_window(signal: &ArrayView1<f64>, window_type: &str) -> SklResult<Array1<f64>> {
    if signal.is_empty() {
        return Err(SklearsError::InvalidInput("Empty signal".to_string()));
    }

    let n = signal.len();
    let mut windowed = Array1::zeros(n);

    match window_type.to_lowercase().as_str() {
        "hanning" | "hann" => {
            for i in 0..n {
                let window_val = if i == 0 || i == n - 1 {
                    0.0
                } else {
                    0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                };
                windowed[i] = signal[i] * window_val;
            }
        }
        "hamming" => {
            for i in 0..n {
                let base = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
                let window_val = if i == 0 || i == n - 1 { 0.0 } else { base };
                windowed[i] = signal[i] * window_val;
            }
        }
        "blackman" => {
            for i in 0..n {
                let arg = 2.0 * PI * i as f64 / (n - 1) as f64;
                let base = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
                let window_val = if i == 0 || i == n - 1 { 0.0 } else { base };
                windowed[i] = signal[i] * window_val;
            }
        }
        "rectangular" | "rect" => {
            windowed.assign(signal);
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown window type: {}",
                window_type
            )));
        }
    }

    Ok(windowed)
}

/// Cross-correlation between two signals
///
/// Computes the cross-correlation function between two signals, which measures
/// the similarity between the signals as a function of time lag.
pub fn cross_correlate(
    signal1: &ArrayView1<f64>,
    signal2: &ArrayView1<f64>,
) -> SklResult<Array1<f64>> {
    if signal1.is_empty() || signal2.is_empty() {
        return Err(SklearsError::InvalidInput("Empty signal".to_string()));
    }

    let n1 = signal1.len();
    let n2 = signal2.len();
    let result_len = n1 + n2 - 1;
    let mut result = Array1::zeros(result_len);

    // Simple cross-correlation implementation
    for i in 0..result_len {
        let mut sum = 0.0;
        for j in 0..n1 {
            let k = i as i32 - j as i32;
            if k >= 0 && k < n2 as i32 {
                sum += signal1[j] * signal2[k as usize];
            }
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Normalized cross-correlation between two signals
///
/// Computes the normalized cross-correlation, which ranges from -1 to 1
/// and provides a measure of similarity independent of signal amplitude.
pub fn normalized_cross_correlate(
    signal1: &ArrayView1<f64>,
    signal2: &ArrayView1<f64>,
) -> SklResult<Array1<f64>> {
    let correlation = cross_correlate(signal1, signal2)?;

    // Compute normalization factors
    let norm1 = signal1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2 = signal2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm1 > 0.0 && norm2 > 0.0 {
        Ok(correlation / (norm1 * norm2))
    } else {
        Ok(correlation)
    }
}

/// Auto-correlation of a signal
///
/// Computes the auto-correlation function, which is the cross-correlation
/// of a signal with itself. Useful for detecting periodic patterns.
pub fn auto_correlate(signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
    cross_correlate(signal, signal)
}

/// Generate a window function of specified type and length
///
/// Utility function to create window functions that can be applied to signals
/// for various signal processing operations.
pub fn generate_window(window_type: &str, length: usize) -> SklResult<Array1<f64>> {
    if length == 0 {
        return Err(SklearsError::InvalidInput(
            "Window length must be positive".to_string(),
        ));
    }

    let mut window = Array1::zeros(length);

    match window_type.to_lowercase().as_str() {
        "hanning" | "hann" => {
            for i in 0..length {
                window[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / (length - 1) as f64).cos());
            }
        }
        "hamming" => {
            for i in 0..length {
                window[i] = 0.54 - 0.46 * (2.0 * PI * i as f64 / (length - 1) as f64).cos();
            }
        }
        "blackman" => {
            for i in 0..length {
                let arg = 2.0 * PI * i as f64 / (length - 1) as f64;
                window[i] = 0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos();
            }
        }
        "rectangular" | "rect" => {
            window.fill(1.0);
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown window type: {}",
                window_type
            )));
        }
    }

    Ok(window)
}

/// Convolve two signals
///
/// Computes the convolution of two 1D signals, which represents the amount of overlap
/// of one signal as it is shifted over another.
pub fn convolve(signal: &ArrayView1<f64>, kernel: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
    if signal.is_empty() || kernel.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Empty signal or kernel".to_string(),
        ));
    }

    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let result_len = signal_len + kernel_len - 1;
    let mut result = Array1::zeros(result_len);

    // Simple convolution implementation
    for i in 0..result_len {
        let mut sum = 0.0;
        for j in 0..kernel_len {
            let signal_idx = i as i32 - j as i32;
            if signal_idx >= 0 && signal_idx < signal_len as i32 {
                sum += signal[signal_idx as usize] * kernel[j];
            }
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Convolve with mode (same, valid, full)
///
/// Computes convolution with different output modes:
/// - "full": Full convolution (default behavior)
/// - "same": Output same size as first input
/// - "valid": Output only where they completely overlap
pub fn convolve_mode(
    signal: &ArrayView1<f64>,
    kernel: &ArrayView1<f64>,
    mode: &str,
) -> SklResult<Array1<f64>> {
    let full_conv = convolve(signal, kernel)?;
    let signal_len = signal.len();
    let kernel_len = kernel.len();

    match mode.to_lowercase().as_str() {
        "full" => Ok(full_conv),
        "same" => {
            let start = kernel_len / 2;
            let end = start + signal_len;
            if end <= full_conv.len() {
                Ok(full_conv.slice(s![start..end]).to_owned())
            } else {
                Ok(full_conv)
            }
        }
        "valid" => {
            if signal_len >= kernel_len {
                let start = kernel_len - 1;
                let end = signal_len;
                if end <= full_conv.len() {
                    Ok(full_conv.slice(s![start..end]).to_owned())
                } else {
                    Ok(Array1::zeros(0))
                }
            } else {
                Ok(Array1::zeros(0))
            }
        }
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown convolution mode: {}",
            mode
        ))),
    }
}
