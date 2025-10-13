//! Spectral Decomposition Module
//!
//! This module provides comprehensive spectral analysis functionality including:
//! - Short-Time Fourier Transform (STFT) with multiple window functions
//! - Power Spectral Density (PSD) computation
//! - Spectral peak detection and analysis
//! - Cross-spectral analysis for multi-channel signals
//! - Advanced window functions (Kaiser, Tukey, Gaussian, etc.)
//! - Frequency domain analysis with comprehensive error handling
//!
//! # Examples
//!
//! ## Basic STFT Analysis
//! ```rust,ignore
//! use sklears_decomposition::signal_processing::spectral_decomposition::{
//!     SpectralDecomposition, WindowFunction
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a test signal
//! let signal = Array1::linspace(0.0, 1.0, 1024);
//!
//! // Set up spectral decomposition with Hanning window
//! let spectral = SpectralDecomposition::new(128)
//!     .window_function(WindowFunction::Hanning)
//!     .overlap(64);
//!
//! // Compute STFT
//! let result = spectral.stft(&signal)?;
//!
//! // Extract power spectral density
//! let psd = result.power_spectral_density();
//! # Ok(())
//! # }
//! ```

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::f64::consts::PI;

/// Spectral decomposition configuration for frequency domain analysis
///
/// Provides comprehensive spectral analysis capabilities including STFT computation,
/// power spectral density analysis, and multi-channel cross-spectral analysis.
#[derive(Debug, Clone)]
pub struct SpectralDecomposition {
    /// Window size for spectral analysis
    pub window_size: usize,
    /// Overlap between windows (in samples)
    pub overlap: usize,
    /// Window function type
    pub window_function: WindowFunction,
    /// Zero-padding factor for FFT
    pub zero_padding_factor: usize,
    /// Sampling rate (Hz)
    pub sampling_rate: Float,
}

/// Window function types for spectral analysis
///
/// Each window function provides different trade-offs between frequency resolution
/// and spectral leakage characteristics.
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    /// Rectangular window (no tapering)
    Rectangular,
    /// Hanning window (good general-purpose window)
    Hanning,
    /// Hamming window (slightly better sidelobe suppression than Hanning)
    Hamming,
    /// Blackman window (excellent sidelobe suppression)
    Blackman,
    /// Kaiser window with beta parameter for adjustable characteristics
    Kaiser(Float),
    /// Tukey window (tapered cosine window) with alpha parameter
    Tukey(Float),
    /// Gaussian window with sigma parameter
    Gaussian(Float),
    /// Bartlett window (triangular window)
    Bartlett,
    /// Blackman-Harris window (superior sidelobe suppression)
    BlackmanHarris,
}

impl SpectralDecomposition {
    /// Create a new spectral decomposition instance
    ///
    /// # Arguments
    /// * `window_size` - Size of the analysis window in samples
    ///
    /// # Example
    /// ```rust,ignore
    /// use sklears_decomposition::signal_processing::spectral_decomposition::SpectralDecomposition;
    ///
    /// let spectral = SpectralDecomposition::new(512);
    /// ```
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            overlap: window_size / 2,
            window_function: WindowFunction::Hanning,
            zero_padding_factor: 1,
            sampling_rate: 1.0,
        }
    }

    /// Set overlap between windows
    ///
    /// # Arguments
    /// * `overlap` - Number of samples to overlap between consecutive windows
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set window function
    ///
    /// # Arguments
    /// * `window_function` - Type of window function to apply
    pub fn window_function(mut self, window_function: WindowFunction) -> Self {
        self.window_function = window_function;
        self
    }

    /// Set zero-padding factor for improved frequency resolution
    ///
    /// # Arguments
    /// * `factor` - Zero-padding factor (1 = no padding, 2 = double length, etc.)
    pub fn zero_padding_factor(mut self, factor: usize) -> Self {
        self.zero_padding_factor = factor;
        self
    }

    /// Set sampling rate
    ///
    /// # Arguments
    /// * `sampling_rate` - Sampling rate in Hz
    pub fn sampling_rate(mut self, sampling_rate: Float) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }

    /// Compute Short-Time Fourier Transform (STFT)
    ///
    /// Performs windowed Fourier transform analysis to provide time-frequency representation.
    ///
    /// # Arguments
    /// * `signal` - Input signal for analysis
    ///
    /// # Returns
    /// * `SpectralResult` containing magnitude, phase, frequency and time information
    ///
    /// # Errors
    /// * Returns `SklearsError::InvalidInput` if signal is too short or overlap is invalid
    pub fn stft(&self, signal: &Array1<Float>) -> Result<SpectralResult> {
        self.validate_parameters(signal)?;

        let n = signal.len();
        let hop_size = self.window_size - self.overlap;
        let n_frames = (n - self.overlap) / hop_size;
        let fft_size = self.window_size * self.zero_padding_factor;
        let n_freq_bins = fft_size / 2 + 1;

        let mut magnitude = Array2::zeros((n_freq_bins, n_frames));
        let mut phase = Array2::zeros((n_freq_bins, n_frames));

        // Generate window function
        let window = self.generate_window()?;

        for frame in 0..n_frames {
            let start = frame * hop_size;
            let end = (start + self.window_size).min(n);

            if end - start < self.window_size {
                break;
            }

            // Apply window to signal segment
            let mut windowed_signal = Array1::zeros(fft_size);
            for i in 0..self.window_size {
                if start + i < n {
                    windowed_signal[i] = signal[start + i] * window[i];
                }
            }

            // Compute FFT
            let (mag, ph) = self.compute_dft(&windowed_signal)?;

            for freq in 0..n_freq_bins {
                magnitude[[freq, frame]] = mag[freq];
                phase[[freq, frame]] = ph[freq];
            }
        }

        Ok(SpectralResult {
            magnitude,
            phase,
            frequencies: self.generate_frequency_bins(),
            times: self.generate_time_bins(hop_size as Float, n_frames),
            sampling_rate: self.sampling_rate,
            window_function: self.window_function,
        })
    }

    /// Compute cross-spectral analysis for multi-channel signals
    ///
    /// Analyzes phase and magnitude relationships between multiple signal channels.
    ///
    /// # Arguments
    /// * `signals` - Matrix where each row is a signal channel
    ///
    /// # Returns
    /// * `CrossSpectralResult` containing cross-spectral information
    pub fn cross_spectral_analysis(&self, signals: &Array2<Float>) -> Result<CrossSpectralResult> {
        let (n_channels, signal_length) = signals.dim();

        if n_channels < 2 {
            return Err(SklearsError::InvalidInput(
                "Cross-spectral analysis requires at least 2 channels".to_string(),
            ));
        }

        // Compute STFT for each channel
        let mut channel_stfts = Vec::new();
        for ch in 0..n_channels {
            let signal = signals.slice(s![ch, ..]).to_owned();
            let stft_result = self.stft(&signal)?;
            channel_stfts.push(stft_result);
        }

        // Compute cross-spectral matrices
        let (n_freq, n_time) = channel_stfts[0].magnitude.dim();
        let mut cross_spectrum = Array2::zeros((n_freq, n_time));
        let mut coherence = Array2::zeros((n_freq, n_time));
        let mut phase_difference = Array2::zeros((n_freq, n_time));

        for freq in 0..n_freq {
            for time in 0..n_time {
                // Compute cross-spectrum between first two channels as example
                let mag1 = channel_stfts[0].magnitude[[freq, time]];
                let mag2 = channel_stfts[1].magnitude[[freq, time]];
                let phase1 = channel_stfts[0].phase[[freq, time]];
                let phase2 = channel_stfts[1].phase[[freq, time]];

                cross_spectrum[[freq, time]] = mag1 * mag2;
                coherence[[freq, time]] = if mag1 > 1e-12 && mag2 > 1e-12 {
                    (cross_spectrum[[freq, time]] / (mag1 * mag1 + mag2 * mag2)).min(1.0)
                } else {
                    0.0
                };
                phase_difference[[freq, time]] = phase1 - phase2;
            }
        }

        Ok(CrossSpectralResult {
            cross_spectrum,
            coherence,
            phase_difference,
            frequencies: channel_stfts[0].frequencies.clone(),
            times: channel_stfts[0].times.clone(),
            n_channels,
        })
    }

    /// Validate input parameters
    fn validate_parameters(&self, signal: &Array1<Float>) -> Result<()> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        if self.window_size == 0 {
            return Err(SklearsError::InvalidInput(
                "Window size must be positive".to_string(),
            ));
        }

        if self.overlap >= self.window_size {
            return Err(SklearsError::InvalidInput(
                "Overlap must be less than window size".to_string(),
            ));
        }

        if signal.len() < self.window_size {
            return Err(SklearsError::InvalidInput(
                "Signal must be at least as long as window size".to_string(),
            ));
        }

        if self.zero_padding_factor == 0 {
            return Err(SklearsError::InvalidInput(
                "Zero padding factor must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate window function based on specified type
    fn generate_window(&self) -> Result<Array1<Float>> {
        let mut window = Array1::zeros(self.window_size);

        match self.window_function {
            WindowFunction::Rectangular => {
                window.fill(1.0);
            }
            WindowFunction::Hanning => {
                for i in 0..self.window_size {
                    let phase = 2.0 * PI * i as Float / (self.window_size - 1) as Float;
                    window[i] = 0.5 * (1.0 - phase.cos());
                }
            }
            WindowFunction::Hamming => {
                for i in 0..self.window_size {
                    let phase = 2.0 * PI * i as Float / (self.window_size - 1) as Float;
                    window[i] = 0.54 - 0.46 * phase.cos();
                }
            }
            WindowFunction::Blackman => {
                for i in 0..self.window_size {
                    let phase = 2.0 * PI * i as Float / (self.window_size - 1) as Float;
                    window[i] = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                }
            }
            WindowFunction::Kaiser(beta) => {
                let beta = beta.max(0.0);
                let norm = self.modified_bessel_i0(beta);
                for i in 0..self.window_size {
                    let x = 2.0 * i as Float / (self.window_size - 1) as Float - 1.0;
                    let arg = beta * (1.0 - x * x).max(0.0).sqrt();
                    window[i] = self.modified_bessel_i0(arg) / norm;
                }
            }
            WindowFunction::Tukey(alpha) => {
                let alpha = alpha.clamp(0.0, 1.0);
                let n_taper = (alpha * self.window_size as Float / 2.0) as usize;

                for i in 0..self.window_size {
                    if i < n_taper {
                        let phase = PI * i as Float / n_taper as Float;
                        window[i] = 0.5 * (1.0 - phase.cos());
                    } else if i >= self.window_size - n_taper {
                        let phase = PI * (self.window_size - 1 - i) as Float / n_taper as Float;
                        window[i] = 0.5 * (1.0 - phase.cos());
                    } else {
                        window[i] = 1.0;
                    }
                }
            }
            WindowFunction::Gaussian(sigma) => {
                let sigma = sigma.max(0.1);
                let center = (self.window_size - 1) as Float / 2.0;
                for i in 0..self.window_size {
                    let x = (i as Float - center) / (sigma * center);
                    window[i] = (-0.5 * x * x).exp();
                }
            }
            WindowFunction::Bartlett => {
                let n_half = (self.window_size - 1) as Float / 2.0;
                for i in 0..self.window_size {
                    window[i] = 1.0 - (i as Float - n_half).abs() / n_half;
                }
            }
            WindowFunction::BlackmanHarris => {
                for i in 0..self.window_size {
                    let phase = 2.0 * PI * i as Float / (self.window_size - 1) as Float;
                    window[i] = 0.35875 - 0.48829 * phase.cos() + 0.14128 * (2.0 * phase).cos()
                        - 0.01168 * (3.0 * phase).cos();
                }
            }
        }

        Ok(window)
    }

    /// Modified Bessel function I0 for Kaiser window
    fn modified_bessel_i0(&self, x: Float) -> Float {
        let mut result = 1.0;
        let mut term = 1.0;
        let x_half = x / 2.0;

        for n in 1..50 {
            term *= x_half * x_half / (n * n) as Float;
            result += term;
            if term < 1e-12 {
                break;
            }
        }

        result
    }

    /// Compute DFT using simple algorithm (in practice, would use FFT library)
    fn compute_dft(&self, signal: &Array1<Float>) -> Result<(Array1<Float>, Array1<Float>)> {
        let n = signal.len();
        let n_freq = n / 2 + 1;
        let mut magnitude = Array1::zeros(n_freq);
        let mut phase = Array1::zeros(n_freq);

        for k in 0..n_freq {
            let mut real = 0.0;
            let mut imag = 0.0;

            for n_idx in 0..n {
                let angle = -2.0 * PI * k as Float * n_idx as Float / n as Float;
                real += signal[n_idx] * angle.cos();
                imag += signal[n_idx] * angle.sin();
            }

            magnitude[k] = (real * real + imag * imag).sqrt();
            phase[k] = imag.atan2(real);
        }

        Ok((magnitude, phase))
    }

    /// Generate frequency bins for the spectrum
    fn generate_frequency_bins(&self) -> Array1<Float> {
        let fft_size = self.window_size * self.zero_padding_factor;
        let n_freq = fft_size / 2 + 1;
        let mut frequencies = Array1::zeros(n_freq);

        for i in 0..n_freq {
            frequencies[i] = i as Float * self.sampling_rate / fft_size as Float;
        }

        frequencies
    }

    /// Generate time bins for the spectrogram
    fn generate_time_bins(&self, hop_size: Float, n_frames: usize) -> Array1<Float> {
        let mut times = Array1::zeros(n_frames);

        for i in 0..n_frames {
            times[i] = i as Float * hop_size / self.sampling_rate;
        }

        times
    }
}

/// Result of spectral decomposition analysis
///
/// Contains the complete time-frequency representation of the analyzed signal.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Magnitude spectrogram (frequency × time)
    pub magnitude: Array2<Float>,
    /// Phase spectrogram (frequency × time)
    pub phase: Array2<Float>,
    /// Frequency bins (Hz)
    pub frequencies: Array1<Float>,
    /// Time bins (seconds)
    pub times: Array1<Float>,
    /// Sampling rate used for analysis
    pub sampling_rate: Float,
    /// Window function used for analysis
    pub window_function: WindowFunction,
}

impl SpectralResult {
    /// Compute power spectral density
    ///
    /// Returns the squared magnitude of the STFT, representing the power distribution
    /// across frequency and time.
    pub fn power_spectral_density(&self) -> Array2<Float> {
        self.magnitude.mapv(|x| x * x)
    }

    /// Extract spectral peaks for each time frame
    ///
    /// Identifies local maxima in the magnitude spectrum that exceed a threshold.
    ///
    /// # Arguments
    /// * `threshold` - Minimum magnitude threshold for peak detection
    ///
    /// # Returns
    /// * Vector of peaks for each time frame as (frequency_bin, magnitude) tuples
    pub fn spectral_peaks(&self, threshold: Float) -> Vec<Vec<(usize, Float)>> {
        let (n_freq, n_frames) = self.magnitude.dim();
        let mut peaks = Vec::new();

        for frame in 0..n_frames {
            let mut frame_peaks = Vec::new();

            for freq in 1..n_freq - 1 {
                let current = self.magnitude[[freq, frame]];
                let prev = self.magnitude[[freq - 1, frame]];
                let next = self.magnitude[[freq + 1, frame]];

                if current > prev && current > next && current > threshold {
                    frame_peaks.push((freq, current));
                }
            }

            peaks.push(frame_peaks);
        }

        peaks
    }

    /// Compute spectral centroid for each time frame
    ///
    /// The spectral centroid indicates the "brightness" of the spectrum.
    pub fn spectral_centroid(&self) -> Array1<Float> {
        let (n_freq, n_frames) = self.magnitude.dim();
        let mut centroids = Array1::zeros(n_frames);

        for frame in 0..n_frames {
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;

            for freq in 0..n_freq {
                let mag = self.magnitude[[freq, frame]];
                weighted_sum += self.frequencies[freq] * mag;
                magnitude_sum += mag;
            }

            centroids[frame] = if magnitude_sum > 1e-12 {
                weighted_sum / magnitude_sum
            } else {
                0.0
            };
        }

        centroids
    }

    /// Compute spectral rolloff for each time frame
    ///
    /// Frequency below which a specified percentage of total spectral energy lies.
    ///
    /// # Arguments
    /// * `rolloff_percentage` - Percentage of total energy (e.g., 0.85 for 85%)
    pub fn spectral_rolloff(&self, rolloff_percentage: Float) -> Array1<Float> {
        let (n_freq, n_frames) = self.magnitude.dim();
        let mut rolloffs = Array1::zeros(n_frames);
        let threshold = rolloff_percentage.clamp(0.0, 1.0);

        for frame in 0..n_frames {
            let psd = self.magnitude.slice(s![.., frame]).mapv(|x| x * x);
            let total_energy: Float = psd.sum();
            let target_energy = total_energy * threshold;

            let mut cumulative_energy = 0.0;
            let mut rolloff_freq = self.frequencies[n_freq - 1];

            for freq in 0..n_freq {
                cumulative_energy += psd[freq];
                if cumulative_energy >= target_energy {
                    rolloff_freq = self.frequencies[freq];
                    break;
                }
            }

            rolloffs[frame] = rolloff_freq;
        }

        rolloffs
    }
}

/// Result of cross-spectral analysis for multi-channel signals
#[derive(Debug, Clone)]
pub struct CrossSpectralResult {
    /// Cross-spectral magnitude between channels
    pub cross_spectrum: Array2<Float>,
    /// Coherence between channels (0-1)
    pub coherence: Array2<Float>,
    /// Phase difference between channels
    pub phase_difference: Array2<Float>,
    /// Frequency bins
    pub frequencies: Array1<Float>,
    /// Time bins
    pub times: Array1<Float>,
    /// Number of signal channels analyzed
    pub n_channels: usize,
}

impl CrossSpectralResult {
    /// Compute average coherence across time for each frequency
    pub fn average_coherence(&self) -> Array1<Float> {
        self.coherence.mean_axis(Axis(1)).unwrap()
    }

    /// Find frequency bands with high coherence
    ///
    /// # Arguments
    /// * `coherence_threshold` - Minimum coherence value (0-1)
    ///
    /// # Returns
    /// * Vector of frequency indices with high coherence
    pub fn high_coherence_frequencies(&self, coherence_threshold: Float) -> Vec<usize> {
        let avg_coherence = self.average_coherence();
        avg_coherence
            .iter()
            .enumerate()
            .filter_map(|(idx, &coh)| {
                if coh > coherence_threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_decomposition_creation() {
        let spectral = SpectralDecomposition::new(128)
            .overlap(64)
            .window_function(WindowFunction::Hanning)
            .sampling_rate(44100.0);

        assert_eq!(spectral.window_size, 128);
        assert_eq!(spectral.overlap, 64);
        assert_eq!(spectral.sampling_rate, 44100.0);
    }

    #[test]
    fn test_window_functions() {
        let spectral = SpectralDecomposition::new(32);

        let window_functions = vec![
            WindowFunction::Rectangular,
            WindowFunction::Hanning,
            WindowFunction::Hamming,
            WindowFunction::Blackman,
            WindowFunction::Kaiser(2.0),
            WindowFunction::Tukey(0.5),
            WindowFunction::Gaussian(0.4),
            WindowFunction::Bartlett,
            WindowFunction::BlackmanHarris,
        ];

        for window_func in window_functions {
            let spectral_with_window = spectral.clone().window_function(window_func);
            let window = spectral_with_window.generate_window().unwrap();

            assert_eq!(window.len(), 32);
            assert!(!window.iter().any(|&x| x.is_nan() || x.is_infinite()));
        }
    }

    #[test]
    fn test_stft_basic_functionality() {
        let spectral = SpectralDecomposition::new(32)
            .overlap(16)
            .sampling_rate(1000.0);

        // Create a simple test signal (sine wave)
        let mut signal = Array1::zeros(128);
        for i in 0..128 {
            signal[i] = (2.0 * PI * 10.0 * i as Float / 1000.0).sin();
        }

        let result = spectral.stft(&signal).unwrap();

        assert_eq!(result.magnitude.nrows(), 17); // (32/2 + 1) frequency bins
        assert!(result.magnitude.ncols() > 0);
        assert_eq!(result.phase.dim(), result.magnitude.dim());
        assert_eq!(result.frequencies.len(), result.magnitude.nrows());
        assert_eq!(result.times.len(), result.magnitude.ncols());
    }

    #[test]
    fn test_power_spectral_density() {
        let spectral = SpectralDecomposition::new(16);
        let signal = Array1::ones(64);

        let result = spectral.stft(&signal).unwrap();
        let psd = result.power_spectral_density();

        assert_eq!(psd.dim(), result.magnitude.dim());
        assert!(psd.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_spectral_peaks() {
        let spectral = SpectralDecomposition::new(16);
        let mut signal = Array1::zeros(64);

        // Add some peaks
        for i in 0..64 {
            signal[i] = (2.0 * PI * 2.0 * i as Float / 16.0).sin() + 0.1;
        }

        let result = spectral.stft(&signal).unwrap();
        let peaks = result.spectral_peaks(0.5);

        assert_eq!(peaks.len(), result.magnitude.ncols());
    }

    #[test]
    fn test_spectral_centroid() {
        let spectral = SpectralDecomposition::new(16).sampling_rate(1000.0);
        let signal = Array1::ones(64);

        let result = spectral.stft(&signal).unwrap();
        let centroids = result.spectral_centroid();

        assert_eq!(centroids.len(), result.magnitude.ncols());
        assert!(centroids.iter().all(|&x| x >= 0.0 && x <= 500.0)); // Within Nyquist
    }

    #[test]
    fn test_cross_spectral_analysis() {
        let spectral = SpectralDecomposition::new(16);

        // Create two-channel signal
        let mut signals = Array2::zeros((2, 64));
        for i in 0..64 {
            signals[[0, i]] = (2.0 * PI * i as Float / 16.0).sin();
            signals[[1, i]] = (2.0 * PI * i as Float / 16.0).cos(); // 90° phase shift
        }

        let result = spectral.cross_spectral_analysis(&signals).unwrap();

        assert_eq!(result.n_channels, 2);
        assert_eq!(result.cross_spectrum.nrows(), 9); // (16/2 + 1) frequency bins
        assert_eq!(result.coherence.dim(), result.cross_spectrum.dim());
        assert_eq!(result.phase_difference.dim(), result.cross_spectrum.dim());
    }

    #[test]
    fn test_error_handling() {
        let spectral = SpectralDecomposition::new(32);

        // Empty signal
        let empty_signal = Array1::zeros(0);
        assert!(spectral.stft(&empty_signal).is_err());

        // Signal too short
        let short_signal = Array1::zeros(16); // Less than window size
        assert!(spectral.stft(&short_signal).is_err());

        // Invalid overlap
        let invalid_spectral = SpectralDecomposition::new(32).overlap(32);
        let signal = Array1::zeros(64);
        assert!(invalid_spectral.stft(&signal).is_err());
    }

    #[test]
    fn test_kaiser_window_properties() {
        let spectral = SpectralDecomposition::new(32).window_function(WindowFunction::Kaiser(2.0));

        let window = spectral.generate_window().unwrap();

        // Kaiser window should be symmetric and have maximum at center
        let center = window.len() / 2;
        assert!(window[center] >= window[0]);
        assert!(window[center] >= window[window.len() - 1]);
    }

    #[test]
    fn test_spectral_rolloff() {
        let spectral = SpectralDecomposition::new(16).sampling_rate(1000.0);
        let signal = Array1::ones(64);

        let result = spectral.stft(&signal).unwrap();
        let rolloffs = result.spectral_rolloff(0.85);

        assert_eq!(rolloffs.len(), result.magnitude.ncols());
        assert!(rolloffs.iter().all(|&x| x >= 0.0 && x <= 500.0));
    }
}
