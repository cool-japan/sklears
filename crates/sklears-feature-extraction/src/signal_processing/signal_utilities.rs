//! Signal processing utilities and preprocessing tools
//!
//! This module provides essential signal processing utilities including:
//! - Signal resampling with multiple interpolation methods
//! - Envelope detection using Hilbert transform and rectification
//! - Signal conditioning and preprocessing operations
//! - Quality metrics for processed signals

use scirs2_core::ndarray::{Array1, ArrayView1};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use std::f64::consts::PI;

/// Resampler for changing sample rate
///
/// Provides signal resampling capabilities with multiple interpolation methods.
/// Useful for standardizing sample rates across different data sources.
#[derive(Debug, Clone)]
pub struct Resampler {
    target_rate: f64,
    original_rate: f64,
    method: String,
}

impl Resampler {
    /// Create a new resampler
    ///
    /// Default configuration:
    /// - Target rate: 22050.0 Hz
    /// - Original rate: 44100.0 Hz
    /// - Method: "linear" interpolation
    pub fn new() -> Self {
        Self {
            target_rate: 22050.0,
            original_rate: 44100.0,
            method: "linear".to_string(),
        }
    }

    /// Set the target sample rate
    pub fn target_rate(mut self, target_rate: f64) -> Self {
        self.target_rate = target_rate;
        self
    }

    /// Set the original sample rate
    pub fn original_rate(mut self, original_rate: f64) -> Self {
        self.original_rate = original_rate;
        self
    }

    /// Set the resampling method
    ///
    /// Supported methods:
    /// - "linear": Linear interpolation (default, good quality)
    /// - "nearest": Nearest neighbor (fastest, lower quality)
    /// - "cubic": Catmull-Rom cubic interpolation (smooth upsampling)
    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    /// Resample the input signal to the target rate
    ///
    /// Performs sample rate conversion using the configured interpolation method.
    /// Returns the resampled signal with the new length based on the rate ratio.
    pub fn resample(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let ratio = self.target_rate / self.original_rate;
        let new_length = (signal.len() as f64 * ratio) as usize;

        if new_length == 0 {
            return Err(SklearsError::InvalidInput(
                "Invalid resampling ratio".to_string(),
            ));
        }

        let mut resampled = Array1::zeros(new_length);

        match self.method.as_str() {
            "linear" => {
                for i in 0..new_length {
                    let orig_index = i as f64 / ratio;
                    let left_idx = orig_index.floor() as usize;
                    let right_idx = (left_idx + 1).min(signal.len() - 1);
                    let frac = orig_index - orig_index.floor();

                    if left_idx < signal.len() {
                        resampled[i] = signal[left_idx] * (1.0 - frac) + signal[right_idx] * frac;
                    }
                }
            }
            "nearest" => {
                for i in 0..new_length {
                    let orig_index = ((i as f64 / ratio).round() as usize).min(signal.len() - 1);
                    resampled[i] = signal[orig_index];
                }
            }
            "cubic" => {
                for i in 0..new_length {
                    let orig_pos = i as f64 / ratio;
                    let idx = orig_pos.floor() as isize;
                    let t = orig_pos - idx as f64;

                    let p0 = Self::sample_with_clamp(signal, idx - 1);
                    let p1 = Self::sample_with_clamp(signal, idx);
                    let p2 = Self::sample_with_clamp(signal, idx + 1);
                    let p3 = Self::sample_with_clamp(signal, idx + 2);

                    resampled[i] = Self::catmull_rom(p0, p1, p2, p3, t);
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown resampling method: {}",
                    self.method
                )));
            }
        }

        Ok(resampled)
    }

    fn sample_with_clamp(signal: &ArrayView1<f64>, index: isize) -> f64 {
        let clamped = index.clamp(0, signal.len() as isize - 1) as usize;
        signal[clamped]
    }

    fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
        let t2 = t * t;
        let t3 = t2 * t;

        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }

    /// Extract resampling quality features
    ///
    /// Returns metrics that characterize the resampling operation:
    /// - Length ratio (new_length / original_length)
    /// - Energy preservation ratio (maintains signal power)
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let resampled = self.resample(signal)?;

        // Extract features from resampled signal
        let mut features = Vec::new();

        // Length ratio
        features.push(resampled.len() as f64 / signal.len() as f64);

        // Energy preservation ratio
        let orig_energy = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        let resamp_energy = resampled.iter().map(|x| x * x).sum::<f64>() / resampled.len() as f64;
        features.push(resamp_energy / orig_energy.max(1e-10));

        Ok(Array1::from_vec(features))
    }

    /// Get the resampling ratio
    pub fn ratio(&self) -> f64 {
        self.target_rate / self.original_rate
    }

    /// Check if resampling is upsampling (increasing sample rate)
    pub fn is_upsampling(&self) -> bool {
        self.target_rate > self.original_rate
    }

    /// Check if resampling is downsampling (decreasing sample rate)
    pub fn is_downsampling(&self) -> bool {
        self.target_rate < self.original_rate
    }

    /// Estimate the new signal length after resampling
    pub fn estimate_new_length(&self, original_length: usize) -> usize {
        (original_length as f64 * self.ratio()) as usize
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Envelope Detector extractor
///
/// Extracts the amplitude envelope of signals using various detection methods.
/// Useful for analyzing signal dynamics and amplitude modulation.
#[derive(Debug, Clone)]
pub struct EnvelopeDetector {
    method: String,
    n_fft: usize,
    window: String,
}

impl EnvelopeDetector {
    /// Create a new envelope detector
    ///
    /// Default configuration:
    /// - Method: "hilbert" (Hilbert transform)
    /// - FFT size: 1024 (for frequency domain methods)
    /// - Window: "hanning"
    pub fn new() -> Self {
        Self {
            method: "hilbert".to_string(),
            n_fft: 1024,
            window: "hanning".to_string(),
        }
    }

    /// Set the envelope detection method
    ///
    /// Supported methods:
    /// - "hilbert": Uses Hilbert transform for analytic signal envelope
    /// - "rectify": Full-wave rectification followed by low-pass filtering
    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    /// Set the FFT size for frequency domain methods
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the window function for frequency domain processing
    pub fn window(mut self, window: String) -> Self {
        self.window = window;
        self
    }

    /// Extract envelope features from signal
    ///
    /// Returns the amplitude envelope of the input signal using the configured method.
    /// The envelope represents the slowly-varying amplitude of the signal.
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let envelope = match self.method.as_str() {
            "hilbert" => self.hilbert_envelope(signal)?,
            "rectify" => self.rectified_envelope(signal)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown envelope method: {}",
                    self.method
                )))
            }
        };

        Ok(envelope)
    }

    /// Compute envelope using Hilbert transform
    ///
    /// Uses the Hilbert transform to create the analytic signal, then computes
    /// the magnitude to get the instantaneous amplitude envelope.
    fn hilbert_envelope(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let n = signal.len();
        let mut envelope = Array1::zeros(n);

        for i in 0..n {
            let real_part = signal[i];

            // Simple Hilbert transform approximation
            let mut imag_part = 0.0;
            for k in 0..n {
                if k != i {
                    let kernel = 1.0 / (PI * (i as f64 - k as f64));
                    imag_part += signal[k] * kernel;
                }
            }

            envelope[i] = (real_part * real_part + imag_part * imag_part).sqrt();
        }

        Ok(envelope)
    }

    /// Compute envelope using rectification and low-pass filtering
    ///
    /// Full-wave rectifies the signal (takes absolute value) then applies
    /// low-pass filtering to smooth the result and extract the envelope.
    fn rectified_envelope(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Full-wave rectification followed by low-pass filtering
        let rectified: Array1<f64> = signal.mapv(|x| x.abs());

        // Simple moving average as low-pass filter
        let window_size = 20;
        let mut smoothed = Array1::zeros(rectified.len());

        for i in window_size..rectified.len() - window_size {
            let mut sum = 0.0;
            for j in i - window_size..i + window_size {
                sum += rectified[j];
            }
            smoothed[i] = sum / (2 * window_size) as f64;
        }

        Ok(smoothed)
    }

    /// Extract envelope statistics
    ///
    /// Computes statistical features from the envelope signal:
    /// - Mean envelope value
    /// - Standard deviation of envelope
    /// - Maximum envelope value
    /// - Envelope dynamic range
    pub fn envelope_statistics(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let envelope = self.extract_features(signal)?;

        let mut stats = Vec::new();

        // Mean envelope
        let mean = envelope.mean().unwrap_or(0.0);
        stats.push(mean);

        // Standard deviation
        let std = envelope.std(0.0);
        stats.push(std);

        // Maximum envelope
        let max_val = envelope.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        stats.push(max_val);

        // Dynamic range (max - min)
        let min_val = envelope.iter().cloned().fold(f64::INFINITY, f64::min);
        let dynamic_range = max_val - min_val;
        stats.push(dynamic_range);

        Ok(Array1::from_vec(stats))
    }

    /// Detect envelope peaks
    ///
    /// Finds local maxima in the envelope signal that could represent
    /// important amplitude events or modulation patterns.
    pub fn detect_peaks(&self, signal: &ArrayView1<f64>, min_height: f64) -> SklResult<Vec<usize>> {
        let envelope = self.extract_features(signal)?;
        let mut peaks = Vec::new();

        // Simple peak detection: local maxima above threshold
        for i in 1..envelope.len() - 1 {
            if envelope[i] > envelope[i - 1]
                && envelope[i] > envelope[i + 1]
                && envelope[i] > min_height
            {
                peaks.push(i);
            }
        }

        Ok(peaks)
    }

    /// Alias for extract_features
    pub fn extract(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.extract_features(signal)
    }
}

impl Default for EnvelopeDetector {
    fn default() -> Self {
        Self::new()
    }
}
