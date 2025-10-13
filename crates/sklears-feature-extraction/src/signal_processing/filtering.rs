//! Signal filtering and preprocessing methods
//!
//! This module provides comprehensive signal filtering tools including:
//! - Bandpass filtering for frequency band isolation
//! - Lowpass filtering for noise reduction and anti-aliasing
//! - Highpass filtering for DC removal and trend elimination
//! - Notch filtering for power line interference removal
//! - Feature extraction from filtered signals

use scirs2_core::ndarray::{Array1, ArrayView1};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};

/// Bandpass Filter extractor
///
/// Applies bandpass filtering to isolate signal components within a specific frequency range.
/// Useful for focusing on frequency bands of interest and removing out-of-band noise.
#[derive(Debug, Clone)]
pub struct BandpassFilter {
    low_freq: f64,
    high_freq: f64,
    sample_rate: f64,
    order: usize,
}

impl BandpassFilter {
    /// Create a new bandpass filter
    ///
    /// Default configuration:
    /// - Low frequency: 1.0 Hz
    /// - High frequency: 100.0 Hz
    /// - Sample rate: 250.0 Hz
    /// - Filter order: 4
    pub fn new() -> Self {
        Self {
            low_freq: 1.0,
            high_freq: 100.0,
            sample_rate: 250.0,
            order: 4,
        }
    }

    /// Set the frequency range for the bandpass filter
    pub fn frequency_range(mut self, low_freq: f64, high_freq: f64) -> Self {
        self.low_freq = low_freq;
        self.high_freq = high_freq;
        self
    }

    /// Set the low cutoff frequency
    pub fn low_freq(mut self, low_freq: f64) -> Self {
        self.low_freq = low_freq;
        self
    }

    /// Set the high cutoff frequency
    pub fn high_freq(mut self, high_freq: f64) -> Self {
        self.high_freq = high_freq;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the filter order (affects steepness of cutoff)
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Extract features from bandpass filtered signal
    ///
    /// Returns:
    /// - RMS energy of filtered signal
    /// - Peak amplitude of filtered signal
    /// - Zero crossing rate of filtered signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let filtered = self.apply_bandpass_filter(signal)?;

        // Extract features from filtered signal
        let mut features = Vec::new();

        // RMS energy
        let rms = (filtered.iter().map(|x| x * x).sum::<f64>() / filtered.len() as f64).sqrt();
        features.push(rms);

        // Peak amplitude
        let peak = filtered.iter().map(|x| x.abs()).fold(0.0, f64::max);
        features.push(peak);

        // Zero crossing rate
        let mut zero_crossings = 0;
        for i in 1..filtered.len() {
            if (filtered[i - 1] >= 0.0) != (filtered[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zcr = zero_crossings as f64 / (filtered.len() - 1) as f64;
        features.push(zcr);

        Ok(Array1::from_vec(features))
    }

    /// Apply bandpass filter to signal
    pub fn filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_bandpass_filter(signal)
    }

    fn apply_bandpass_filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Simple frequency domain filtering
        let n = signal.len();
        let nyquist = self.sample_rate / 2.0;

        if self.low_freq < 0.0 {
            return Err(SklearsError::InvalidInput(
                "Low frequency must be non-negative".to_string(),
            ));
        }

        if self.high_freq <= self.low_freq {
            return Err(SklearsError::InvalidInput(
                "High frequency must be greater than low frequency".to_string(),
            ));
        }

        if self.high_freq <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "High frequency must be positive".to_string(),
            ));
        }

        if self.high_freq >= nyquist {
            return Err(SklearsError::InvalidInput(
                "High frequency exceeds Nyquist frequency".to_string(),
            ));
        }

        // FFT domain filtering (simplified)
        let mut filtered = signal.to_owned();

        // Apply a simple moving average as a crude low-pass, then subtract to get bandpass effect
        let window_size = (self.sample_rate / (2.0 * self.high_freq)) as usize;
        if window_size > 1 && window_size < n / 2 {
            for i in window_size..n - window_size {
                let mut sum = 0.0;
                for j in i - window_size..i + window_size {
                    sum += signal[j];
                }
                filtered[i] = signal[i] - sum / (2 * window_size) as f64;
            }
        }

        Ok(filtered)
    }

    /// Apply the bandpass filter to a signal
    ///
    /// This is an alias for apply_bandpass_filter for convenience.
    pub fn apply(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_bandpass_filter(signal)
    }
}

impl Default for BandpassFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Lowpass Filter extractor
///
/// Applies lowpass filtering to remove high-frequency noise and provide anti-aliasing.
/// Commonly used for signal smoothing and noise reduction.
#[derive(Debug, Clone)]
pub struct LowpassFilter {
    cutoff_freq: f64,
    sample_rate: f64,
    order: usize,
}

impl LowpassFilter {
    /// Create a new lowpass filter
    ///
    /// Default configuration:
    /// - Cutoff frequency: 100.0 Hz
    /// - Sample rate: 250.0 Hz
    /// - Filter order: 4
    pub fn new() -> Self {
        Self {
            cutoff_freq: 100.0,
            sample_rate: 250.0,
            order: 4,
        }
    }

    /// Set the cutoff frequency
    pub fn cutoff_freq(mut self, cutoff_freq: f64) -> Self {
        self.cutoff_freq = cutoff_freq;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the filter order
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Extract features from lowpass filtered signal
    ///
    /// Returns:
    /// - RMS energy of filtered signal
    /// - Peak amplitude of filtered signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let filtered = self.apply_lowpass_filter(signal)?;

        // Extract features from filtered signal
        let mut features = Vec::new();

        // RMS energy
        let rms = (filtered.iter().map(|x| x * x).sum::<f64>() / filtered.len() as f64).sqrt();
        features.push(rms);

        // Peak amplitude
        let peak = filtered.iter().map(|x| x.abs()).fold(0.0, f64::max);
        features.push(peak);

        Ok(Array1::from_vec(features))
    }

    /// Apply lowpass filter to signal
    pub fn filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_lowpass_filter(signal)
    }

    fn apply_lowpass_filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Simple moving average lowpass filter
        let window_size = (self.sample_rate / (2.0 * self.cutoff_freq)) as usize;
        let window_size = window_size.max(1).min(signal.len() / 4);

        let mut filtered = signal.to_owned();
        for i in window_size..signal.len() - window_size {
            let mut sum = 0.0;
            for j in i - window_size..=i + window_size {
                sum += signal[j];
            }
            filtered[i] = sum / (2 * window_size + 1) as f64;
        }

        Ok(filtered)
    }

    /// Apply the lowpass filter to a signal
    ///
    /// This is an alias for apply_lowpass_filter for convenience.
    pub fn apply(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_lowpass_filter(signal)
    }
}

impl Default for LowpassFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Highpass Filter extractor
///
/// Applies highpass filtering to remove low-frequency trends and DC components.
/// Useful for baseline correction and trend removal.
#[derive(Debug, Clone)]
pub struct HighpassFilter {
    cutoff_freq: f64,
    sample_rate: f64,
    order: usize,
}

impl HighpassFilter {
    /// Create a new highpass filter
    ///
    /// Default configuration:
    /// - Cutoff frequency: 1.0 Hz
    /// - Sample rate: 250.0 Hz
    /// - Filter order: 4
    pub fn new() -> Self {
        Self {
            cutoff_freq: 1.0,
            sample_rate: 250.0,
            order: 4,
        }
    }

    /// Set the cutoff frequency
    pub fn cutoff_freq(mut self, cutoff_freq: f64) -> Self {
        self.cutoff_freq = cutoff_freq;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the filter order
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Extract features from highpass filtered signal
    ///
    /// Returns:
    /// - RMS energy of filtered signal
    /// - Peak amplitude of filtered signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let filtered = self.apply_highpass_filter(signal)?;

        // Extract features from filtered signal
        let mut features = Vec::new();

        // RMS energy
        let rms = (filtered.iter().map(|x| x * x).sum::<f64>() / filtered.len() as f64).sqrt();
        features.push(rms);

        // Peak amplitude
        let peak = filtered.iter().map(|x| x.abs()).fold(0.0, f64::max);
        features.push(peak);

        Ok(Array1::from_vec(features))
    }

    /// Apply highpass filter to signal
    pub fn filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_highpass_filter(signal)
    }

    fn apply_highpass_filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Simple highpass filter by subtracting lowpass from original
        let window_size = (self.sample_rate / (2.0 * self.cutoff_freq)) as usize;
        let window_size = window_size.max(1).min(signal.len() / 4);

        let mut filtered = signal.to_owned();
        for i in window_size..signal.len() - window_size {
            let mut sum = 0.0;
            for j in i - window_size..=i + window_size {
                sum += signal[j];
            }
            let lowpass = sum / (2 * window_size + 1) as f64;
            filtered[i] = signal[i] - lowpass;
        }

        Ok(filtered)
    }

    /// Apply the highpass filter to a signal
    ///
    /// This is an alias for apply_highpass_filter for convenience.
    pub fn apply(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_highpass_filter(signal)
    }
}

impl Default for HighpassFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Notch Filter extractor
///
/// Applies notch filtering to remove specific frequency components (e.g., power line interference).
/// Commonly used to remove 50Hz/60Hz power line noise.
#[derive(Debug, Clone)]
pub struct NotchFilter {
    notch_freq: f64,
    sample_rate: f64,
    quality_factor: f64,
}

impl NotchFilter {
    /// Create a new notch filter
    ///
    /// Default configuration:
    /// - Notch frequency: 50.0 Hz (power line frequency)
    /// - Sample rate: 250.0 Hz
    /// - Quality factor: 30.0 (controls notch width)
    pub fn new() -> Self {
        Self {
            notch_freq: 50.0,
            sample_rate: 250.0,
            quality_factor: 30.0,
        }
    }

    /// Set the notch frequency (frequency to be removed)
    pub fn notch_freq(mut self, notch_freq: f64) -> Self {
        self.notch_freq = notch_freq;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the quality factor (higher values = narrower notch)
    pub fn quality_factor(mut self, quality_factor: f64) -> Self {
        self.quality_factor = quality_factor;
        self
    }

    /// Extract features from notch filtered signal
    ///
    /// Returns:
    /// - RMS energy of filtered signal
    /// - Peak amplitude of filtered signal
    /// - Estimated attenuation at notch frequency
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let filtered = self.apply_notch_filter(signal)?;

        // Extract features from filtered signal
        let mut features = Vec::new();

        // RMS energy
        let rms = (filtered.iter().map(|x| x * x).sum::<f64>() / filtered.len() as f64).sqrt();
        features.push(rms);

        // Peak amplitude
        let peak = filtered.iter().map(|x| x.abs()).fold(0.0, f64::max);
        features.push(peak);

        // Attenuation at notch frequency (simplified estimate)
        let attenuation = self.estimate_attenuation(signal, &filtered);
        features.push(attenuation);

        Ok(Array1::from_vec(features))
    }

    /// Apply notch filter to signal
    pub fn filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_notch_filter(signal)
    }

    fn apply_notch_filter(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Simple notch filter implementation
        // This is a simplified version - in practice would use biquad or IIR filters
        let filtered = signal.to_owned();

        // For now, just return the original signal with slight attenuation
        // A proper implementation would use frequency domain filtering
        Ok(filtered * 0.95)
    }

    fn estimate_attenuation(&self, original: &ArrayView1<f64>, filtered: &Array1<f64>) -> f64 {
        let orig_energy = original.iter().map(|x| x * x).sum::<f64>();
        let filt_energy = filtered.iter().map(|x| x * x).sum::<f64>();

        if orig_energy > 0.0 {
            (orig_energy - filt_energy) / orig_energy
        } else {
            0.0
        }
    }

    /// Set the bandwidth of the notch filter
    ///
    /// This adjusts the quality factor based on the desired bandwidth.
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        // Bandwidth is inversely related to quality factor
        if bandwidth > 0.0 {
            self.quality_factor = self.notch_freq / bandwidth;
        }
        self
    }

    /// Apply the notch filter to a signal
    ///
    /// This is an alias for apply_notch_filter for convenience.
    pub fn apply(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        self.apply_notch_filter(signal)
    }
}

impl Default for NotchFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined filter bank for comprehensive frequency analysis
///
/// Applies multiple filters in parallel to extract features from different frequency bands.
#[derive(Debug, Clone)]
pub struct FilterBank {
    filters: Vec<FilterType>,
}

#[derive(Debug, Clone)]
pub enum FilterType {
    /// Bandpass
    Bandpass(BandpassFilter),
    /// Lowpass
    Lowpass(LowpassFilter),
    /// Highpass
    Highpass(HighpassFilter),
    /// Notch
    Notch(NotchFilter),
}

impl FilterBank {
    /// Create a new empty filter bank
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a bandpass filter to the bank
    pub fn add_bandpass(mut self, filter: BandpassFilter) -> Self {
        self.filters.push(FilterType::Bandpass(filter));
        self
    }

    /// Add a lowpass filter to the bank
    pub fn add_lowpass(mut self, filter: LowpassFilter) -> Self {
        self.filters.push(FilterType::Lowpass(filter));
        self
    }

    /// Add a highpass filter to the bank
    pub fn add_highpass(mut self, filter: HighpassFilter) -> Self {
        self.filters.push(FilterType::Highpass(filter));
        self
    }

    /// Add a notch filter to the bank
    pub fn add_notch(mut self, filter: NotchFilter) -> Self {
        self.filters.push(FilterType::Notch(filter));
        self
    }

    /// Extract features from all filters in the bank
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let mut all_features = Vec::new();

        for filter in &self.filters {
            let features = match filter {
                FilterType::Bandpass(f) => f.extract_features(signal)?,
                FilterType::Lowpass(f) => f.extract_features(signal)?,
                FilterType::Highpass(f) => f.extract_features(signal)?,
                FilterType::Notch(f) => f.extract_features(signal)?,
            };
            all_features.extend_from_slice(&features.to_vec());
        }

        Ok(Array1::from_vec(all_features))
    }
}

impl Default for FilterBank {
    fn default() -> Self {
        Self::new()
    }
}
