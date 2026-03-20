//! Frequency domain analysis and feature extraction
//!
//! This module provides comprehensive frequency domain analysis tools including:
//! - Power spectral density analysis with Welch's method
//! - Filter bank feature extraction with configurable bands
//! - Short-Time Fourier Transform (STFT) analysis
//! - Spectral statistics and characteristics

use scirs2_core::ndarray::{concatenate, s, Array1, Array2, ArrayView1, Axis};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use std::f64::consts::PI;

/// Frequency domain features extractor
///
/// Extracts features from the frequency domain representation of signals
/// including power spectral density, frequency band energies, and spectral statistics.
#[derive(Debug, Clone)]
pub struct FrequencyDomainExtractor {
    n_fft: usize,
    window: String,
    normalize: bool,
    frequency_bands: Vec<(f64, f64)>,
    sample_rate: f64,
}

impl FrequencyDomainExtractor {
    /// Create a new frequency domain extractor
    pub fn new() -> Self {
        Self {
            n_fft: 1024,
            window: "hanning".to_string(),
            normalize: true,
            frequency_bands: vec![
                (0.0, 4.0),    // Delta
                (4.0, 8.0),    // Theta
                (8.0, 13.0),   // Alpha
                (13.0, 30.0),  // Beta
                (30.0, 100.0), // Gamma
            ],
            sample_rate: 250.0,
        }
    }

    /// Set the FFT size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the window function
    pub fn window(mut self, window: String) -> Self {
        self.window = window;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set frequency bands for feature extraction
    pub fn frequency_bands(mut self, bands: Vec<(f64, f64)>) -> Self {
        self.frequency_bands = bands;
        self
    }

    /// Set sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Extract frequency domain features from signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        // Zero-pad short signals up to the FFT size so that we can still
        // provide spectral estimates instead of failing for short recordings.
        let psd = if signal.len() < self.n_fft {
            let mut padded = Array1::zeros(self.n_fft);
            padded.slice_mut(s![..signal.len()]).assign(signal);
            self.compute_psd(&padded.view())?
        } else {
            self.compute_psd(signal)?
        };
        let frequencies = self.get_frequencies();

        let mut features = Vec::new();

        // Extract band powers
        for &(fmin, fmax) in &self.frequency_bands {
            let band_power = self.compute_band_power(&psd, &frequencies, fmin, fmax);
            features.push(band_power);
        }

        // Extract spectral statistics
        let spectral_mean = self.compute_spectral_mean(&psd, &frequencies);
        let spectral_std = self.compute_spectral_std(&psd, &frequencies, spectral_mean);
        let spectral_skewness =
            self.compute_spectral_skewness(&psd, &frequencies, spectral_mean, spectral_std);
        let spectral_kurtosis =
            self.compute_spectral_kurtosis(&psd, &frequencies, spectral_mean, spectral_std);

        features.extend_from_slice(&[
            spectral_mean,
            spectral_std,
            spectral_skewness,
            spectral_kurtosis,
        ]);

        // Extract peak frequency
        let peak_freq = self.compute_peak_frequency(&psd, &frequencies);
        features.push(peak_freq);

        // Extract spectral edge frequency (95% of power)
        let edge_freq = self.compute_spectral_edge(&psd, &frequencies, 0.95);
        features.push(edge_freq);

        let mut result = Array1::from_vec(features);

        if self.normalize {
            let sum: f64 = result.iter().map(|x| x.abs()).sum();
            if sum > 0.0 {
                result.mapv_inplace(|x| x / sum);
            }
        }

        Ok(result)
    }

    /// Compute power spectral density using Welch's method
    fn compute_psd(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let window = self.generate_window(self.n_fft)?;
        let n_freqs = self.n_fft / 2 + 1;
        let mut psd = Array1::zeros(n_freqs);

        let overlap = self.n_fft / 2;
        let step = self.n_fft - overlap;
        let n_segments = (signal.len() - overlap) / step;

        if n_segments == 0 {
            return Err(SklearsError::InvalidInput(
                "Signal too short for analysis".to_string(),
            ));
        }

        for segment in 0..n_segments {
            let start = segment * step;
            let end = start + self.n_fft;

            if end > signal.len() {
                break;
            }

            // Apply window
            let windowed: Vec<f64> = signal
                .slice(s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute FFT magnitude squared
            let spectrum = self.fft_magnitude_squared(&windowed);

            for (i, &power) in spectrum.iter().enumerate() {
                psd[i] += power;
            }
        }

        // Average across segments
        psd.mapv_inplace(|x| x / n_segments as f64);

        Ok(psd)
    }

    /// Generate window function
    fn generate_window(&self, n: usize) -> SklResult<Array1<f64>> {
        match self.window.as_str() {
            "hanning" => {
                Ok(Array1::from_iter((0..n).map(|i| {
                    0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                })))
            }
            "hamming" => {
                Ok(Array1::from_iter((0..n).map(|i| {
                    0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
                })))
            }
            "blackman" => Ok(Array1::from_iter((0..n).map(|i| {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
                    + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            }))),
            "rectangular" | "none" => Ok(Array1::ones(n)),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown window type: {}",
                self.window
            ))),
        }
    }

    /// Compute FFT magnitude squared
    fn fft_magnitude_squared(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let n_freqs = n / 2 + 1;
        let mut magnitudes = vec![0.0; n_freqs];

        for k in 0..n_freqs {
            let mut real = 0.0;
            let mut imag = 0.0;

            for n_idx in 0..n {
                let angle = -2.0 * PI * k as f64 * n_idx as f64 / n as f64;
                real += signal[n_idx] * angle.cos();
                imag += signal[n_idx] * angle.sin();
            }

            magnitudes[k] = real * real + imag * imag;
        }

        magnitudes
    }

    /// Get frequency bins
    fn get_frequencies(&self) -> Array1<f64> {
        let n_freqs = self.n_fft / 2 + 1;
        Array1::from_iter((0..n_freqs).map(|k| k as f64 * self.sample_rate / self.n_fft as f64))
    }

    /// Compute power in a frequency band
    fn compute_band_power(
        &self,
        psd: &Array1<f64>,
        frequencies: &Array1<f64>,
        fmin: f64,
        fmax: f64,
    ) -> f64 {
        let mut power = 0.0;
        for (i, &freq) in frequencies.iter().enumerate() {
            if freq >= fmin && freq <= fmax {
                power += psd[i];
            }
        }
        power
    }

    /// Compute spectral mean
    fn compute_spectral_mean(&self, psd: &Array1<f64>, frequencies: &Array1<f64>) -> f64 {
        let total_power: f64 = psd.sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = psd
            .iter()
            .zip(frequencies.iter())
            .map(|(&power, &freq)| power * freq)
            .sum();

        weighted_sum / total_power
    }

    /// Compute spectral standard deviation
    fn compute_spectral_std(&self, psd: &Array1<f64>, frequencies: &Array1<f64>, mean: f64) -> f64 {
        let total_power: f64 = psd.sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let variance: f64 = psd
            .iter()
            .zip(frequencies.iter())
            .map(|(&power, &freq)| power * (freq - mean).powi(2))
            .sum::<f64>()
            / total_power;

        variance.sqrt()
    }

    /// Compute spectral skewness
    fn compute_spectral_skewness(
        &self,
        psd: &Array1<f64>,
        frequencies: &Array1<f64>,
        mean: f64,
        std: f64,
    ) -> f64 {
        if std == 0.0 {
            return 0.0;
        }

        let total_power: f64 = psd.sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let skewness: f64 = psd
            .iter()
            .zip(frequencies.iter())
            .map(|(&power, &freq)| power * ((freq - mean) / std).powi(3))
            .sum::<f64>()
            / total_power;

        skewness
    }

    /// Compute spectral kurtosis
    fn compute_spectral_kurtosis(
        &self,
        psd: &Array1<f64>,
        frequencies: &Array1<f64>,
        mean: f64,
        std: f64,
    ) -> f64 {
        if std == 0.0 {
            return 0.0;
        }

        let total_power: f64 = psd.sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let kurtosis: f64 = psd
            .iter()
            .zip(frequencies.iter())
            .map(|(&power, &freq)| power * ((freq - mean) / std).powi(4))
            .sum::<f64>()
            / total_power;

        kurtosis - 3.0 // Excess kurtosis
    }

    /// Compute peak frequency
    fn compute_peak_frequency(&self, psd: &Array1<f64>, frequencies: &Array1<f64>) -> f64 {
        let max_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        frequencies[max_idx]
    }

    /// Compute spectral edge frequency
    fn compute_spectral_edge(
        &self,
        psd: &Array1<f64>,
        frequencies: &Array1<f64>,
        threshold: f64,
    ) -> f64 {
        let total_power: f64 = psd.sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let target_power = threshold * total_power;
        let mut cumulative_power = 0.0;

        for (i, &power) in psd.iter().enumerate() {
            cumulative_power += power;
            if cumulative_power >= target_power {
                return frequencies[i];
            }
        }

        frequencies[frequencies.len() - 1]
    }
}

impl Default for FrequencyDomainExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter bank features extractor
///
/// Applies a bank of bandpass filters to extract features from different frequency bands.
#[derive(Debug, Clone)]
pub struct FilterBankExtractor {
    filter_bank: Vec<(f64, f64)>, // (center_freq, bandwidth) pairs
    sample_rate: f64,
    n_fft: usize,
    feature_type: String, // "energy", "power", "magnitude"
}

impl FilterBankExtractor {
    /// Create a new filter bank extractor
    pub fn new() -> Self {
        Self {
            filter_bank: vec![
                (10.0, 5.0),  // 7.5-12.5 Hz
                (20.0, 10.0), // 15-25 Hz
                (40.0, 20.0), // 30-50 Hz
                (80.0, 40.0), // 60-100 Hz
            ],
            sample_rate: 250.0,
            n_fft: 1024,
            feature_type: "energy".to_string(),
        }
    }

    /// Set the filter bank (center frequency, bandwidth pairs)
    pub fn filter_bank(mut self, filters: Vec<(f64, f64)>) -> Self {
        self.filter_bank = filters;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the FFT size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the feature type
    pub fn feature_type(mut self, feature_type: String) -> Self {
        self.feature_type = feature_type;
        self
    }

    /// Extract filter bank features from signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let spectrum = self.compute_spectrum(signal)?;
        let frequencies = self.get_frequencies();

        let mut features = Vec::with_capacity(self.filter_bank.len());

        for &(center_freq, bandwidth) in &self.filter_bank {
            let fmin = center_freq - bandwidth / 2.0;
            let fmax = center_freq + bandwidth / 2.0;

            let feature_value = self.extract_band_feature(&spectrum, &frequencies, fmin, fmax);
            features.push(feature_value);
        }

        Ok(Array1::from_vec(features))
    }

    /// Compute spectrum
    fn compute_spectrum(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.len() < self.n_fft {
            return Err(SklearsError::InvalidInput(
                "Signal too short for FFT".to_string(),
            ));
        }

        // Use zero-padding if necessary
        let mut padded_signal = signal.to_owned();
        if signal.len() % self.n_fft != 0 {
            let padding_size = self.n_fft - signal.len() % self.n_fft;
            let padding = Array1::zeros(padding_size);
            padded_signal = concatenate![Axis(0), padded_signal, padding];
        }

        // Apply Hanning window
        let window = self.hanning_window(self.n_fft);
        let n_segments = padded_signal.len() / self.n_fft;
        let n_freqs = self.n_fft / 2 + 1;
        let mut spectrum = Array1::zeros(n_freqs);

        for segment in 0..n_segments {
            let start = segment * self.n_fft;
            let end = start + self.n_fft;

            let windowed: Vec<f64> = padded_signal
                .slice(s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            let segment_spectrum = self.fft_magnitude(&windowed);
            for (i, &mag) in segment_spectrum.iter().enumerate() {
                spectrum[i] += mag;
            }
        }

        // Average across segments
        spectrum.mapv_inplace(|x| x / n_segments as f64);

        Ok(spectrum)
    }

    /// Generate Hanning window
    fn hanning_window(&self, n: usize) -> Array1<f64> {
        Array1::from_iter(
            (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())),
        )
    }

    /// Compute FFT magnitude
    fn fft_magnitude(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let n_freqs = n / 2 + 1;
        let mut magnitudes = vec![0.0; n_freqs];

        for k in 0..n_freqs {
            let mut real = 0.0;
            let mut imag = 0.0;

            for n_idx in 0..n {
                let angle = -2.0 * PI * k as f64 * n_idx as f64 / n as f64;
                real += signal[n_idx] * angle.cos();
                imag += signal[n_idx] * angle.sin();
            }

            magnitudes[k] = (real * real + imag * imag).sqrt();
        }

        magnitudes
    }

    /// Get frequency bins
    fn get_frequencies(&self) -> Array1<f64> {
        let n_freqs = self.n_fft / 2 + 1;
        Array1::from_iter((0..n_freqs).map(|k| k as f64 * self.sample_rate / self.n_fft as f64))
    }

    /// Extract feature from frequency band
    fn extract_band_feature(
        &self,
        spectrum: &Array1<f64>,
        frequencies: &Array1<f64>,
        fmin: f64,
        fmax: f64,
    ) -> f64 {
        let mut values = Vec::new();

        for (i, &freq) in frequencies.iter().enumerate() {
            if freq >= fmin && freq <= fmax {
                values.push(spectrum[i]);
            }
        }

        if values.is_empty() {
            return 0.0;
        }

        match self.feature_type.as_str() {
            "energy" => values.iter().map(|x| x * x).sum(),
            "power" => values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64,
            "magnitude" => values.iter().sum::<f64>() / values.len() as f64,
            "max" => values.iter().cloned().fold(0.0, f64::max),
            _ => values.iter().sum::<f64>() / values.len() as f64,
        }
    }
}

impl Default for FilterBankExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Short-Time Fourier Transform (STFT) analyzer
///
/// Computes time-frequency representation of signals using overlapping windowed FFTs.
#[derive(Debug, Clone)]
pub struct STFT {
    n_fft: usize,
    hop_length: usize,
    window: String,
    sample_rate: f64,
}

impl STFT {
    pub fn new() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 512,
            window: "hanning".to_string(),
            sample_rate: 22050.0,
        }
    }

    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn window(mut self, window: String) -> Self {
        self.window = window;
        self
    }

    pub fn window_size(mut self, window_size: usize) -> Self {
        self.n_fft = window_size;
        self
    }

    pub fn window_type(mut self, window_type: String) -> Self {
        self.window = window_type;
        self
    }

    pub fn transform(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        // Transform returns (n_time_frames, n_freq_bins) - transposed from extract_features
        let features = self.extract_features(signal)?;
        // Transpose so rows are time frames and columns are frequency bins
        Ok(features.t().to_owned())
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        if signal.len() < self.n_fft {
            return Err(SklearsError::InvalidInput(
                "Signal too short for FFT size".to_string(),
            ));
        }

        let n_frames = (signal.len() - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;
        let mut result = Array2::zeros((n_freqs, n_frames));

        let window = self.generate_window()?;

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

            let magnitudes = self.fft_magnitude(&windowed);
            for (i, &mag) in magnitudes.iter().enumerate() {
                result[[i, frame]] = mag;
            }
        }

        Ok(result)
    }

    fn generate_window(&self) -> SklResult<Array1<f64>> {
        match self.window.as_str() {
            "hanning" => Ok(Array1::from_iter((0..self.n_fft).map(|i| {
                0.5 * (1.0 - (2.0 * PI * i as f64 / (self.n_fft - 1) as f64).cos())
            }))),
            _ => Ok(Array1::ones(self.n_fft)),
        }
    }

    fn fft_magnitude(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let n_freqs = n / 2 + 1;
        let mut magnitudes = vec![0.0; n_freqs];

        for k in 0..n_freqs {
            let mut real = 0.0;
            let mut imag = 0.0;

            for n_idx in 0..n {
                let angle = -2.0 * PI * k as f64 * n_idx as f64 / n as f64;
                real += signal[n_idx] * angle.cos();
                imag += signal[n_idx] * angle.sin();
            }

            magnitudes[k] = (real * real + imag * imag).sqrt();
        }

        magnitudes
    }
}

impl Default for STFT {
    fn default() -> Self {
        Self::new()
    }
}
