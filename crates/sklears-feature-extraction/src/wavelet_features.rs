//! Wavelet and Time-Frequency Feature Extraction
//!
//! This module provides comprehensive wavelet-based and time-frequency feature extraction:
//!
//! - `TimeSeriesWaveletExtractor`: Discrete wavelet transform features for 1D time series
//! - `TimeFrequencyExtractor`: Combined time-frequency analysis with STFT and wavelets
//!
//! These extractors enable time-frequency analysis and multi-resolution signal decomposition.

use crate::{Float, SklResult, SklearsError};
use scirs2_core::ndarray::{s, Array1, ArrayView1};

/// Wavelet types for time series analysis
#[derive(Debug, Clone, Copy)]
pub enum TimeSeriesWaveletType {
    /// Haar
    Haar,
    /// Daubechies4
    Daubechies4,
    /// Daubechies8
    Daubechies8,
    /// Biorthogonal
    Biorthogonal,
    /// Coiflets
    Coiflets,
}

/// Feature types to extract from wavelet coefficients
#[derive(Debug, Clone, Copy)]
pub enum WaveletFeatureType {
    /// Mean
    Mean,
    /// Variance
    Variance,
    /// Energy
    Energy,
    /// Entropy
    Entropy,
    /// Skewness
    Skewness,
    /// Kurtosis
    Kurtosis,
    /// MaxCoeff
    MaxCoeff,
    /// MinCoeff
    MinCoeff,
}

/// Wavelet decomposition result
struct WaveletDecomposition {
    approximation: Array1<Float>,
    details: Vec<Array1<Float>>,
}

/// Time series wavelet transform feature extractor
///
/// Extracts features from time series using discrete wavelet transform (DWT).
/// This is different from the image wavelet features - it's specifically designed
/// for 1D time series analysis and includes time-frequency decomposition.
pub struct TimeSeriesWaveletExtractor {
    wavelet_type: TimeSeriesWaveletType,
    num_levels: usize,
    feature_types: Vec<WaveletFeatureType>,
    include_scaleogram: bool,
    include_energy_ratios: bool,
}

impl TimeSeriesWaveletExtractor {
    /// Create a new time series wavelet extractor
    pub fn new() -> Self {
        Self {
            wavelet_type: TimeSeriesWaveletType::Haar,
            num_levels: 4,
            feature_types: vec![
                WaveletFeatureType::Mean,
                WaveletFeatureType::Variance,
                WaveletFeatureType::Energy,
                WaveletFeatureType::Entropy,
            ],
            include_scaleogram: false,
            include_energy_ratios: true,
        }
    }

    /// Set wavelet type
    pub fn wavelet_type(mut self, wavelet_type: TimeSeriesWaveletType) -> Self {
        self.wavelet_type = wavelet_type;
        self
    }

    /// Set number of decomposition levels
    pub fn num_levels(mut self, levels: usize) -> Self {
        self.num_levels = levels;
        self
    }

    /// Set feature types to extract
    pub fn feature_types(mut self, feature_types: Vec<WaveletFeatureType>) -> Self {
        self.feature_types = feature_types;
        self
    }

    /// Include scaleogram features (time-frequency representation)
    pub fn include_scaleogram(mut self, include: bool) -> Self {
        self.include_scaleogram = include;
        self
    }

    /// Include energy ratios between levels
    pub fn include_energy_ratios(mut self, include: bool) -> Self {
        self.include_energy_ratios = include;
        self
    }

    /// Extract wavelet features from time series
    pub fn extract_features(&self, ts: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        if ts.len() < 4 {
            return Err(SklearsError::InvalidInput(
                "Time series too short for wavelet analysis".to_string(),
            ));
        }

        // Pad to power of 2 for efficient computation
        let padded_ts = self.pad_to_power_of_2(ts)?;

        // Perform wavelet decomposition
        let decomposition = self.wavelet_decompose(&padded_ts)?;

        let mut features = Vec::new();

        // Extract features from each level
        for level in 0..self.num_levels {
            if level < decomposition.details.len() {
                let detail_features = self.extract_level_features(&decomposition.details[level])?;
                features.extend(detail_features);
            }
        }

        // Extract features from approximation coefficients
        let approx_features = self.extract_level_features(&decomposition.approximation)?;
        features.extend(approx_features);

        // Energy ratios between levels
        if self.include_energy_ratios {
            let energy_ratios = self.calculate_energy_ratios(&decomposition)?;
            features.extend(energy_ratios);
        }

        // Scaleogram features (time-frequency analysis)
        if self.include_scaleogram {
            let scaleogram_features = self.extract_scaleogram_features(&padded_ts)?;
            features.extend(scaleogram_features);
        }

        Ok(Array1::from_vec(features))
    }

    /// Pad time series to next power of 2
    fn pad_to_power_of_2(&self, ts: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let len = ts.len();
        let next_power = (len as Float).log2().ceil() as usize;
        let padded_len = 2_usize.pow(next_power as u32);

        let mut padded = Array1::zeros(padded_len);
        padded.slice_mut(s![..len]).assign(ts);

        // Pad with zeros or replicate the boundary values
        for i in len..padded_len {
            padded[i] = ts[len - 1]; // Replicate last value
        }

        Ok(padded)
    }

    /// Perform discrete wavelet transform decomposition
    fn wavelet_decompose(&self, signal: &Array1<Float>) -> SklResult<WaveletDecomposition> {
        let (low_pass, high_pass) = self.get_wavelet_filters()?;

        let mut approximation = signal.clone();
        let mut details = Vec::new();

        for _ in 0..self.num_levels {
            if approximation.len() < 2 {
                break;
            }

            let (approx, detail) = self.single_level_dwt(&approximation, &low_pass, &high_pass)?;
            details.push(detail);
            approximation = approx;
        }

        Ok(WaveletDecomposition {
            approximation,
            details,
        })
    }

    /// Get wavelet filter coefficients
    fn get_wavelet_filters(&self) -> SklResult<(Vec<Float>, Vec<Float>)> {
        match self.wavelet_type {
            TimeSeriesWaveletType::Haar => {
                let low_pass = vec![
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ];
                let high_pass = vec![
                    -std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ];
                Ok((low_pass, high_pass))
            }
            TimeSeriesWaveletType::Daubechies4 => {
                let low_pass = vec![
                    0.48296291314469025,
                    0.8365163037378079,
                    0.2241438680420134,
                    -0.12940952255092145,
                ];
                let high_pass = vec![
                    -0.12940952255092145,
                    -0.2241438680420134,
                    0.8365163037378079,
                    -0.48296291314469025,
                ];
                Ok((low_pass, high_pass))
            }
            TimeSeriesWaveletType::Daubechies8 => {
                let low_pass = vec![
                    0.23037781330885523,
                    0.7148465705525415,
                    0.6308807679295904,
                    -0.02798376941698385,
                    -0.18703481171888114,
                    0.030841381835986965,
                    0.032883011666982945,
                    -0.010597401784997278,
                ];
                let high_pass = vec![
                    -0.010597401784997278,
                    -0.032883011666982945,
                    0.030841381835986965,
                    0.18703481171888114,
                    -0.02798376941698385,
                    -0.6308807679295904,
                    0.7148465705525415,
                    -0.23037781330885523,
                ];
                Ok((low_pass, high_pass))
            }
            TimeSeriesWaveletType::Biorthogonal => {
                // Biorthogonal 2.2 wavelet filters
                let low_pass = vec![
                    -0.1767766952966369,
                    0.3535533905932738,
                    1.0606601717798214,
                    0.3535533905932738,
                    -0.1767766952966369,
                ];
                let high_pass = vec![0.0, -0.5, 1.0, -0.5, 0.0];
                Ok((low_pass, high_pass))
            }
            TimeSeriesWaveletType::Coiflets => {
                // Coiflets 2 (6-tap) wavelet filters
                let low_pass = vec![
                    -0.051429864679368207,
                    0.23389032745436126,
                    0.6054328720162073,
                    0.6054328720162073,
                    0.23389032745436126,
                    -0.051429864679368207,
                ];
                let high_pass = vec![
                    -0.051429864679368207,
                    -0.23389032745436126,
                    0.6054328720162073,
                    -0.6054328720162073,
                    0.23389032745436126,
                    0.051429864679368207,
                ];
                Ok((low_pass, high_pass))
            }
        }
    }

    /// Single level discrete wavelet transform
    fn single_level_dwt(
        &self,
        signal: &Array1<Float>,
        low_pass: &[Float],
        high_pass: &[Float],
    ) -> SklResult<(Array1<Float>, Array1<Float>)> {
        let n = signal.len();
        let filter_len = low_pass.len();

        // Convolution and downsampling
        let mut approx = Vec::new();
        let mut detail = Vec::new();

        for i in (0..n).step_by(2) {
            let mut low_sum = 0.0;
            let mut high_sum = 0.0;

            for j in 0..filter_len {
                let idx = (i + j) % n; // Circular boundary conditions
                low_sum += signal[idx] * low_pass[j];
                high_sum += signal[idx] * high_pass[j];
            }

            approx.push(low_sum);
            detail.push(high_sum);
        }

        Ok((Array1::from_vec(approx), Array1::from_vec(detail)))
    }

    /// Extract features from wavelet coefficients
    fn extract_level_features(&self, coeffs: &Array1<Float>) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();

        for &feature_type in &self.feature_types {
            let feature_value = match feature_type {
                WaveletFeatureType::Mean => coeffs.mean().unwrap_or(0.0),
                WaveletFeatureType::Variance => coeffs.var(0.0),
                WaveletFeatureType::Energy => coeffs.iter().map(|x| x * x).sum::<Float>(),
                WaveletFeatureType::Entropy => self.calculate_wavelet_entropy(coeffs)?,
                WaveletFeatureType::Skewness => self.calculate_skewness(coeffs),
                WaveletFeatureType::Kurtosis => self.calculate_kurtosis(coeffs),
                WaveletFeatureType::MaxCoeff => {
                    coeffs.iter().cloned().fold(Float::NEG_INFINITY, Float::max)
                }
                WaveletFeatureType::MinCoeff => {
                    coeffs.iter().cloned().fold(Float::INFINITY, Float::min)
                }
            };

            features.push(feature_value);
        }

        Ok(features)
    }

    /// Calculate wavelet entropy
    fn calculate_wavelet_entropy(&self, coeffs: &Array1<Float>) -> SklResult<Float> {
        let energy: Float = coeffs.iter().map(|x| x * x).sum();

        if energy < 1e-10 {
            return Ok(0.0);
        }

        let mut entropy = 0.0;
        for &coeff in coeffs.iter() {
            let prob = (coeff * coeff) / energy;
            if prob > 1e-10 {
                entropy -= prob * prob.log2();
            }
        }

        Ok(entropy)
    }

    /// Calculate skewness of coefficients
    fn calculate_skewness(&self, coeffs: &Array1<Float>) -> Float {
        let n = coeffs.len() as Float;
        if n < 3.0 {
            return 0.0;
        }

        let mean = coeffs.mean().unwrap_or(0.0);
        let variance = coeffs.var(0.0);

        if variance < 1e-10 {
            return 0.0;
        }

        let std_dev = variance.sqrt();
        let third_moment: Float = coeffs
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<Float>()
            / n;

        third_moment
    }

    /// Calculate kurtosis of coefficients
    fn calculate_kurtosis(&self, coeffs: &Array1<Float>) -> Float {
        let n = coeffs.len() as Float;
        if n < 4.0 {
            return 0.0;
        }

        let mean = coeffs.mean().unwrap_or(0.0);
        let variance = coeffs.var(0.0);

        if variance < 1e-10 {
            return 0.0;
        }

        let std_dev = variance.sqrt();
        let fourth_moment: Float = coeffs
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<Float>()
            / n;

        fourth_moment - 3.0 // Excess kurtosis
    }

    /// Calculate energy ratios between decomposition levels
    fn calculate_energy_ratios(
        &self,
        decomposition: &WaveletDecomposition,
    ) -> SklResult<Vec<Float>> {
        let mut energies = Vec::new();

        // Energy of approximation
        let approx_energy: Float = decomposition.approximation.iter().map(|x| x * x).sum();
        energies.push(approx_energy);

        // Energy of details
        for detail in &decomposition.details {
            let detail_energy: Float = detail.iter().map(|x| x * x).sum();
            energies.push(detail_energy);
        }

        let total_energy: Float = energies.iter().sum();

        let mut ratios = Vec::new();
        for energy in energies {
            let ratio = if total_energy > 1e-10 {
                energy / total_energy
            } else {
                0.0
            };
            ratios.push(ratio);
        }

        Ok(ratios)
    }

    /// Extract scaleogram features (time-frequency representation)
    fn extract_scaleogram_features(&self, signal: &Array1<Float>) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();

        // Continuous wavelet transform approximation using scales
        let scales = vec![1.0, 2.0, 4.0, 8.0, 16.0];

        for scale in scales {
            let scaled_coeffs = self.continuous_wavelet_transform(signal, scale)?;

            // Extract statistical features from each scale
            let mean = scaled_coeffs.mean().unwrap_or(0.0);
            let variance = scaled_coeffs.var(0.0);
            let max_val = scaled_coeffs
                .iter()
                .cloned()
                .fold(Float::NEG_INFINITY, Float::max);

            features.extend_from_slice(&[mean, variance, max_val]);
        }

        Ok(features)
    }

    /// Simplified continuous wavelet transform for a single scale
    fn continuous_wavelet_transform(
        &self,
        signal: &Array1<Float>,
        scale: Float,
    ) -> SklResult<Array1<Float>> {
        let n = signal.len();
        let mut coeffs = Array1::zeros(n);

        // Morlet wavelet approximation
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let t = (j as Float - i as Float) / scale;
                let wavelet_val = (t * t / 2.0).exp() * (5.0 * t).cos(); // Simplified Morlet
                sum += signal[j] * wavelet_val;
            }
            coeffs[i] = sum / scale.sqrt();
        }

        Ok(coeffs)
    }
}

impl Default for TimeSeriesWaveletExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Time-frequency feature extractor
///
/// Combines multiple time-frequency analysis methods including
/// wavelets, spectrograms, and other time-frequency representations.
pub struct TimeFrequencyExtractor {
    window_size: usize,
    hop_size: usize,
    num_frequency_bins: usize,
    include_spectral_features: bool,
    include_wavelet_features: bool,
    include_instantaneous_features: bool,
}

impl TimeFrequencyExtractor {
    /// Create a new time-frequency extractor
    pub fn new() -> Self {
        Self {
            window_size: 256,
            hop_size: 128,
            num_frequency_bins: 64,
            include_spectral_features: true,
            include_wavelet_features: true,
            include_instantaneous_features: false,
        }
    }

    /// Set window size for STFT
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set hop size for STFT
    pub fn hop_size(mut self, size: usize) -> Self {
        self.hop_size = size;
        self
    }

    /// Set number of frequency bins
    pub fn num_frequency_bins(mut self, bins: usize) -> Self {
        self.num_frequency_bins = bins;
        self
    }

    /// Include spectral features from STFT
    pub fn include_spectral_features(mut self, include: bool) -> Self {
        self.include_spectral_features = include;
        self
    }

    /// Include wavelet-based features
    pub fn include_wavelet_features(mut self, include: bool) -> Self {
        self.include_wavelet_features = include;
        self
    }

    /// Include instantaneous frequency and phase features
    pub fn include_instantaneous_features(mut self, include: bool) -> Self {
        self.include_instantaneous_features = include;
        self
    }

    /// Extract time-frequency features
    pub fn extract_features(&self, ts: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        if ts.len() < self.window_size {
            return Err(SklearsError::InvalidInput(
                "Time series too short for time-frequency analysis".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Spectral features from STFT
        if self.include_spectral_features {
            let spectral_features = self.extract_spectral_features(ts)?;
            features.extend(spectral_features);
        }

        // Wavelet-based time-frequency features
        if self.include_wavelet_features {
            let wavelet_extractor = TimeSeriesWaveletExtractor::new()
                .wavelet_type(TimeSeriesWaveletType::Daubechies4)
                .num_levels(3)
                .include_scaleogram(true);

            let wavelet_features = wavelet_extractor.extract_features(ts)?;
            features.extend(wavelet_features.to_vec());
        }

        // Instantaneous features
        if self.include_instantaneous_features {
            let instantaneous_features = self.extract_instantaneous_features(ts)?;
            features.extend(instantaneous_features);
        }

        Ok(Array1::from_vec(features))
    }

    /// Extract spectral features from short-time Fourier transform
    fn extract_spectral_features(&self, ts: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let num_windows = (ts.len() - self.window_size) / self.hop_size + 1;
        let mut spectral_centroids = Vec::new();
        let mut spectral_bandwidths = Vec::new();
        let mut spectral_rolloffs = Vec::new();

        for i in 0..num_windows {
            let start = i * self.hop_size;
            let end = start + self.window_size;

            if end <= ts.len() {
                let window = ts.slice(s![start..end]);
                let windowed = self.apply_hanning_window(&window.to_owned())?;

                // Compute FFT for this window
                let fft_result = self.compute_fft(&windowed)?;
                let max_freq = fft_result.len() / 2; // Nyquist frequency

                // Spectral centroid
                let mut weighted_sum = 0.0;
                let mut total_power = 0.0;

                for (i, &(real, imag)) in fft_result.iter().take(max_freq).enumerate() {
                    let power = real * real + imag * imag;
                    weighted_sum += i as Float * power;
                    total_power += power;
                }

                let centroid = if total_power > 1e-10 {
                    weighted_sum / total_power
                } else {
                    0.0
                };
                spectral_centroids.push(centroid);

                // Spectral bandwidth
                let mut bandwidth_sum = 0.0;
                for (i, &(real, imag)) in fft_result.iter().take(max_freq).enumerate() {
                    let power = real * real + imag * imag;
                    bandwidth_sum += (i as Float - centroid).powi(2) * power;
                }

                let bandwidth = if total_power > 1e-10 {
                    (bandwidth_sum / total_power).sqrt()
                } else {
                    0.0
                };
                spectral_bandwidths.push(bandwidth);

                // Spectral rolloff
                let target_energy = 0.85 * total_power;
                let mut cumulative_energy = 0.0;
                let mut rolloff = 0.0;

                for (i, &(real, imag)) in fft_result.iter().take(max_freq).enumerate() {
                    let power = real * real + imag * imag;
                    cumulative_energy += power;
                    if cumulative_energy >= target_energy {
                        rolloff = i as Float;
                        break;
                    }
                }
                spectral_rolloffs.push(rolloff);
            }
        }

        let mut features = Vec::new();

        // Statistical features of spectral measures across time
        features.push(spectral_centroids.iter().sum::<Float>() / spectral_centroids.len() as Float);
        features
            .push(spectral_bandwidths.iter().sum::<Float>() / spectral_bandwidths.len() as Float);
        features.push(spectral_rolloffs.iter().sum::<Float>() / spectral_rolloffs.len() as Float);

        // Variance of spectral measures
        let centroid_mean = features[features.len() - 3];
        let bandwidth_mean = features[features.len() - 2];
        let rolloff_mean = features[features.len() - 1];

        let centroid_var = spectral_centroids
            .iter()
            .map(|x| (x - centroid_mean).powi(2))
            .sum::<Float>()
            / spectral_centroids.len() as Float;
        let bandwidth_var = spectral_bandwidths
            .iter()
            .map(|x| (x - bandwidth_mean).powi(2))
            .sum::<Float>()
            / spectral_bandwidths.len() as Float;
        let rolloff_var = spectral_rolloffs
            .iter()
            .map(|x| (x - rolloff_mean).powi(2))
            .sum::<Float>()
            / spectral_rolloffs.len() as Float;

        features.extend_from_slice(&[centroid_var, bandwidth_var, rolloff_var]);

        Ok(features)
    }

    /// Extract instantaneous frequency and phase features
    fn extract_instantaneous_features(&self, ts: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        // Hilbert transform approximation for analytic signal
        let analytic_signal = self.hilbert_transform(ts)?;

        let mut instantaneous_freqs = Vec::new();
        let mut instantaneous_phases = Vec::new();

        for i in 1..analytic_signal.len() {
            let (real_prev, imag_prev) = analytic_signal[i - 1];
            let (real_curr, imag_curr) = analytic_signal[i];

            // Instantaneous phase
            let phase_prev = imag_prev.atan2(real_prev);
            let phase_curr = imag_curr.atan2(real_curr);

            // Instantaneous frequency (derivative of phase)
            let mut freq = phase_curr - phase_prev;

            // Unwrap phase
            if freq > std::f64::consts::PI {
                freq -= 2.0 * std::f64::consts::PI;
            } else if freq < -std::f64::consts::PI {
                freq += 2.0 * std::f64::consts::PI;
            }

            instantaneous_freqs.push(freq);
            instantaneous_phases.push(phase_curr);
        }

        let mut features = Vec::new();

        // Statistical features of instantaneous frequency
        if !instantaneous_freqs.is_empty() {
            let mean_freq =
                instantaneous_freqs.iter().sum::<Float>() / instantaneous_freqs.len() as Float;
            let var_freq = instantaneous_freqs
                .iter()
                .map(|x| (x - mean_freq).powi(2))
                .sum::<Float>()
                / instantaneous_freqs.len() as Float;

            features.extend_from_slice(&[mean_freq, var_freq]);
        }

        // Phase-based features
        if !instantaneous_phases.is_empty() {
            let phase_variance = instantaneous_phases
                .iter()
                .map(|x| x.powi(2))
                .sum::<Float>()
                / instantaneous_phases.len() as Float;

            features.push(phase_variance);
        }

        Ok(features)
    }

    /// Apply Hanning window
    fn apply_hanning_window(&self, signal: &Array1<Float>) -> SklResult<Array1<Float>> {
        let n = signal.len();
        let mut windowed = Array1::zeros(n);

        for i in 0..n {
            let window_val =
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as Float / (n - 1) as Float).cos());
            windowed[i] = signal[i] * window_val;
        }

        Ok(windowed)
    }

    /// Simplified FFT computation
    fn compute_fft(&self, signal: &Array1<Float>) -> SklResult<Vec<(Float, Float)>> {
        let n = signal.len();
        let mut result = vec![(0.0, 0.0); n];

        // Simple DFT (for production, use a proper FFT implementation)
        for k in 0..n {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * j) as Float / n as Float;
                real_sum += signal[j] * angle.cos();
                imag_sum += signal[j] * angle.sin();
            }

            result[k] = (real_sum, imag_sum);
        }

        Ok(result)
    }

    /// Simplified Hilbert transform
    fn hilbert_transform(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<(Float, Float)>> {
        let n = signal.len();
        let mut result = Vec::with_capacity(n);

        // Simple Hilbert transform approximation
        for i in 0..n {
            let mut hilbert_val = 0.0;

            for j in 0..n {
                if i != j {
                    hilbert_val += signal[j] / (std::f64::consts::PI * (i as Float - j as Float));
                }
            }

            result.push((signal[i], hilbert_val));
        }

        Ok(result)
    }
}

impl Default for TimeFrequencyExtractor {
    fn default() -> Self {
        Self::new()
    }
}
