//! Phase-based signal analysis and feature extraction
//!
//! This module provides comprehensive phase analysis tools including:
//! - Phase-based feature extraction from STFT analysis
//! - Instantaneous frequency and phase derivative calculations
//! - Phase coherence and entropy measurements
//! - Hilbert transform for analytic signal computation
//! - Advanced phase domain characteristics

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use std::f64::consts::PI;

/// Phase-based features extractor
///
/// Extracts features based on the phase characteristics of signals in the frequency domain.
#[derive(Debug, Clone)]
pub struct PhaseBasedExtractor {
    n_fft: usize,
    hop_length: usize,
    include_instantaneous_frequency: bool,
    include_phase_derivative: bool,
    include_phase_variance: bool,
}

impl PhaseBasedExtractor {
    /// Create a new phase-based extractor
    pub fn new() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 512,
            include_instantaneous_frequency: true,
            include_phase_derivative: true,
            include_phase_variance: true,
        }
    }

    /// Set the FFT size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set whether to include instantaneous frequency
    pub fn include_instantaneous_frequency(mut self, include: bool) -> Self {
        self.include_instantaneous_frequency = include;
        self
    }

    /// Set whether to include phase derivative
    pub fn include_phase_derivative(mut self, include: bool) -> Self {
        self.include_phase_derivative = include;
        self
    }

    /// Set whether to include phase variance
    pub fn include_phase_variance(mut self, include: bool) -> Self {
        self.include_phase_variance = include;
        self
    }

    /// Extract phase-based features from signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        // Compute STFT to get phase information
        let (magnitude, phase) = self.stft(signal)?;

        let mut features = Vec::new();

        if self.include_instantaneous_frequency {
            let inst_freq = self.compute_instantaneous_frequency(&phase);
            features.extend_from_slice(&self.summarize_time_series(&inst_freq));
        }

        if self.include_phase_derivative {
            let phase_deriv = self.compute_phase_derivative(&phase);
            features.extend_from_slice(&self.summarize_matrix(&phase_deriv));
        }

        if self.include_phase_variance {
            let phase_var = self.compute_phase_variance(&phase);
            features.extend_from_slice(&phase_var.to_vec());
        }

        // Phase coherence
        let coherence = self.compute_phase_coherence(&phase);
        features.push(coherence);

        // Phase entropy
        let entropy = self.compute_phase_entropy(&phase);
        features.push(entropy);

        Ok(Array1::from_vec(features))
    }

    /// Compute STFT to get magnitude and phase
    fn stft(&self, signal: &ArrayView1<f64>) -> SklResult<(Array2<f64>, Array2<f64>)> {
        let n_frames = (signal.len() - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;

        let mut magnitude = Array2::zeros((n_freqs, n_frames));
        let mut phase = Array2::zeros((n_freqs, n_frames));

        let window = self.hanning_window(self.n_fft);

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = start + self.n_fft;

            if end > signal.len() {
                break;
            }

            let windowed: Vec<f64> = signal
                .slice(s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            let (mag, ph) = self.fft_complex(&windowed);

            for (i, (&m, &p)) in mag.iter().zip(ph.iter()).enumerate() {
                magnitude[[i, frame]] = m;
                phase[[i, frame]] = p;
            }
        }

        Ok((magnitude, phase))
    }

    /// Generate Hanning window
    fn hanning_window(&self, n: usize) -> Array1<f64> {
        Array1::from_iter(
            (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())),
        )
    }

    /// Compute complex FFT
    fn fft_complex(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let n_freqs = n / 2 + 1;
        let mut magnitudes = vec![0.0; n_freqs];
        let mut phases = vec![0.0; n_freqs];

        for k in 0..n_freqs {
            let mut real = 0.0;
            let mut imag = 0.0;

            for n_idx in 0..n {
                let angle = -2.0 * PI * k as f64 * n_idx as f64 / n as f64;
                real += signal[n_idx] * angle.cos();
                imag += signal[n_idx] * angle.sin();
            }

            magnitudes[k] = (real * real + imag * imag).sqrt();
            phases[k] = imag.atan2(real);
        }

        (magnitudes, phases)
    }

    /// Compute instantaneous frequency
    fn compute_instantaneous_frequency(&self, phase: &Array2<f64>) -> Array1<f64> {
        let n_frames = phase.ncols();
        let mut inst_freq = Array1::zeros(n_frames);

        if n_frames < 2 {
            return inst_freq;
        }

        for frame in 1..n_frames {
            let mut weighted_freq = 0.0;
            let mut total_weight = 0.0;

            for freq_bin in 0..phase.nrows() {
                let phase_diff = phase[[freq_bin, frame]] - phase[[freq_bin, frame - 1]];
                let unwrapped_diff = self.unwrap_phase(phase_diff);

                // Weight by frequency bin (higher frequencies contribute more)
                let weight = freq_bin as f64 + 1.0;
                weighted_freq += unwrapped_diff * weight;
                total_weight += weight;
            }

            if total_weight > 0.0 {
                inst_freq[frame] = weighted_freq / total_weight;
            }
        }

        inst_freq
    }

    /// Unwrap phase difference
    fn unwrap_phase(&self, phase_diff: f64) -> f64 {
        let mut unwrapped = phase_diff;
        while unwrapped > PI {
            unwrapped -= 2.0 * PI;
        }
        while unwrapped < -PI {
            unwrapped += 2.0 * PI;
        }
        unwrapped
    }

    /// Compute phase derivative across frequency
    fn compute_phase_derivative(&self, phase: &Array2<f64>) -> Array2<f64> {
        let (n_freqs, n_frames) = phase.dim();
        let mut phase_deriv = Array2::zeros((n_freqs - 1, n_frames));

        for frame in 0..n_frames {
            for freq in 1..n_freqs {
                let phase_diff = phase[[freq, frame]] - phase[[freq - 1, frame]];
                phase_deriv[[freq - 1, frame]] = self.unwrap_phase(phase_diff);
            }
        }

        phase_deriv
    }

    /// Compute phase variance across time
    fn compute_phase_variance(&self, phase: &Array2<f64>) -> Array1<f64> {
        let n_freqs = phase.nrows();
        let mut phase_var = Array1::zeros(n_freqs);

        for freq in 0..n_freqs {
            let phase_row = phase.row(freq);
            let mean_phase = phase_row.mean().unwrap_or(0.0);
            let variance = phase_row
                .iter()
                .map(|&p| (p - mean_phase).powi(2))
                .sum::<f64>()
                / phase_row.len() as f64;
            phase_var[freq] = variance;
        }

        phase_var
    }

    /// Compute phase coherence
    fn compute_phase_coherence(&self, phase: &Array2<f64>) -> f64 {
        let (n_freqs, n_frames) = phase.dim();
        if n_frames < 2 {
            return 0.0;
        }

        let mut coherence_sum = 0.0;
        let mut count = 0;

        for freq in 0..n_freqs {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for frame in 0..n_frames {
                let p = phase[[freq, frame]];
                real_sum += p.cos();
                imag_sum += p.sin();
            }

            let coherence = (real_sum * real_sum + imag_sum * imag_sum).sqrt() / n_frames as f64;
            coherence_sum += coherence;
            count += 1;
        }

        if count > 0 {
            coherence_sum / count as f64
        } else {
            0.0
        }
    }

    /// Compute phase entropy
    fn compute_phase_entropy(&self, phase: &Array2<f64>) -> f64 {
        // Discretize phase values into bins
        let n_bins = 16;
        let mut histogram = vec![0; n_bins];
        let total_samples = phase.len();

        for &p in phase.iter() {
            // Map phase from [-π, π] to [0, n_bins-1]
            let normalized = (p + PI) / (2.0 * PI);
            let bin = ((normalized * n_bins as f64).floor() as usize).min(n_bins - 1);
            histogram[bin] += 1;
        }

        // Compute entropy
        let mut entropy = 0.0;
        for &count in &histogram {
            if count > 0 {
                let p = count as f64 / total_samples as f64;
                entropy -= p * p.ln();
            }
        }

        entropy / (n_bins as f64).ln() // Normalize by max entropy
    }

    /// Summarize time series with statistics
    fn summarize_time_series(&self, data: &Array1<f64>) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0; 5];
        }

        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        vec![mean, std, min, max, range]
    }

    /// Summarize matrix with statistics
    fn summarize_matrix(&self, data: &Array2<f64>) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0; 5];
        }

        let flattened: Vec<f64> = data.iter().cloned().collect();
        let arr = Array1::from_vec(flattened);
        self.summarize_time_series(&arr)
    }
}

impl Default for PhaseBasedExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Hilbert Transform extractor
///
/// Computes the analytic signal using Hilbert transform for phase and amplitude analysis.
#[derive(Debug, Clone)]
pub struct HilbertTransform {
    n_fft: usize,
}

impl HilbertTransform {
    pub fn new() -> Self {
        Self { n_fft: 1024 }
    }

    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let analytic = self.analytic_signal(signal)?;
        let mut features = Vec::new();

        // Instantaneous amplitude
        let inst_amplitude: Vec<f64> = analytic
            .iter()
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect();
        features.push(inst_amplitude.iter().sum::<f64>() / inst_amplitude.len() as f64); // Mean
        features.push(
            inst_amplitude.iter().map(|x| x * x).sum::<f64>().sqrt() / inst_amplitude.len() as f64,
        ); // RMS

        // Instantaneous phase
        let inst_phase: Vec<f64> = analytic.iter().map(|(r, i)| i.atan2(*r)).collect();
        let mut phase_diffs = Vec::new();
        for i in 1..inst_phase.len() {
            let mut diff = inst_phase[i] - inst_phase[i - 1];
            while diff > PI {
                diff -= 2.0 * PI;
            }
            while diff < -PI {
                diff += 2.0 * PI;
            }
            phase_diffs.push(diff);
        }
        if !phase_diffs.is_empty() {
            features.push(phase_diffs.iter().sum::<f64>() / phase_diffs.len() as f64);
            // Mean freq
        }

        Ok(Array1::from_vec(features))
    }

    pub fn transform(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let analytic = self.analytic_signal(signal)?;
        let complex_signal: Vec<f64> = analytic.iter().map(|(r, i)| r * r + i * i).collect();
        Ok(Array1::from_vec(complex_signal))
    }

    pub fn instantaneous_amplitude(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let analytic = self.analytic_signal(signal)?;
        let amplitudes: Vec<f64> = analytic
            .iter()
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect();
        Ok(Array1::from_vec(amplitudes))
    }

    pub fn instantaneous_phase(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let analytic = self.analytic_signal(signal)?;
        let phases: Vec<f64> = analytic.iter().map(|(r, i)| i.atan2(*r)).collect();
        Ok(Array1::from_vec(phases))
    }

    fn analytic_signal(&self, signal: &ArrayView1<f64>) -> SklResult<Vec<(f64, f64)>> {
        let n = signal.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let real_part = signal[i];

            // Simple approximation of Hilbert transform using 90-degree phase shift
            let mut imag_part = 0.0;
            for k in 0..n {
                if k != i {
                    let kernel = 1.0 / (PI * (i as f64 - k as f64));
                    imag_part += signal[k] * kernel;
                }
            }

            result.push((real_part, imag_part));
        }

        Ok(result)
    }
}

impl Default for HilbertTransform {
    fn default() -> Self {
        Self::new()
    }
}
