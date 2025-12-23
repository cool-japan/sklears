//! Time series feature extraction
//!
//! This module provides transformers for extracting features from time series data
//! including temporal patterns, frequency domain analysis, and sliding window statistics.

use crate::*;
// use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::prelude::SklearsError;

/// Temporal Feature Extractor
///
/// Extracts temporal features from time series data including lag features,
/// rolling statistics, trend, and seasonality components.
///
/// # Parameters
///
/// * `n_lags` - Number of lag features to generate
/// * `window_sizes` - Window sizes for rolling statistics
/// * `extract_trend` - Whether to extract trend features
/// * `extract_seasonality` - Whether to extract seasonality features
/// * `seasonal_periods` - Periods for seasonality extraction
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::time_series_features::TemporalFeatureExtractor;
/// use scirs2_core::ndarray::{array, Array1, ArrayView1, ArrayView2};
///
/// let ts_data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let extractor = TemporalFeatureExtractor::new()
///     .n_lags(3)
///     .window_sizes(vec![3, 5])
///     .extract_trend(true);
///
/// let features = extractor.extract_features(&ts_data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TemporalFeatureExtractor {
    n_lags: usize,
    window_sizes: Vec<usize>,
    extract_trend: bool,
    extract_seasonality: bool,
    seasonal_periods: Vec<usize>,
    fill_method: String, // "forward", "backward", "mean", "zero"
}

impl TemporalFeatureExtractor {
    /// Create a new TemporalFeatureExtractor
    pub fn new() -> Self {
        Self {
            n_lags: 5,
            window_sizes: vec![3, 7, 14],
            extract_trend: true,
            extract_seasonality: true,
            seasonal_periods: vec![7, 12, 24],
            fill_method: "forward".to_string(),
        }
    }

    /// Set the number of lag features
    pub fn n_lags(mut self, n_lags: usize) -> Self {
        self.n_lags = n_lags;
        self
    }

    /// Set the window sizes for rolling statistics
    pub fn window_sizes(mut self, window_sizes: Vec<usize>) -> Self {
        self.window_sizes = window_sizes;
        self
    }

    /// Set whether to extract trend features
    pub fn extract_trend(mut self, extract_trend: bool) -> Self {
        self.extract_trend = extract_trend;
        self
    }

    /// Set whether to extract seasonality features
    pub fn extract_seasonality(mut self, extract_seasonality: bool) -> Self {
        self.extract_seasonality = extract_seasonality;
        self
    }

    /// Set the seasonal periods
    pub fn seasonal_periods(mut self, seasonal_periods: Vec<usize>) -> Self {
        self.seasonal_periods = seasonal_periods;
        self
    }

    /// Set the fill method for missing values
    pub fn fill_method(mut self, fill_method: String) -> Self {
        self.fill_method = fill_method;
        self
    }

    /// Extract temporal features from a time series
    pub fn extract_features(&self, ts: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n = ts.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Time series cannot be empty".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Lag features
        for lag in 1..=self.n_lags {
            if lag >= n {
                features.push(0.0); // Use zero for unavailable lags
            } else {
                features.push(ts[n - 1 - lag]);
            }
        }

        // Rolling statistics
        for &window_size in &self.window_sizes {
            if window_size <= n {
                let start_idx = n.saturating_sub(window_size);
                let window = ts.slice(s![start_idx..]);

                // Rolling mean
                let mean = window.sum() / window_size as Float;
                features.push(mean);

                // Rolling std
                let std = (window.mapv(|x| (x - mean).powi(2)).sum() / window_size as Float).sqrt();
                features.push(std);

                // Rolling min/max
                features.push(
                    *window
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap(),
                );
                features.push(
                    *window
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap(),
                );
            } else {
                // Use entire series if window is larger than series
                let mean = ts.sum() / n as Float;
                let std = (ts.mapv(|x| (x - mean).powi(2)).sum() / n as Float).sqrt();
                let min_val = *ts.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let max_val = *ts.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

                features.extend_from_slice(&[mean, std, min_val, max_val]);
            }
        }

        // Trend features
        if self.extract_trend {
            let trend_features = self.extract_trend_features(ts)?;
            features.extend_from_slice(&trend_features);
        }

        // Seasonality features
        if self.extract_seasonality {
            for &period in &self.seasonal_periods {
                let seasonality_features = self.extract_seasonality_features(ts, period)?;
                features.extend_from_slice(&seasonality_features);
            }
        }

        Ok(Array1::from_vec(features))
    }

    fn extract_trend_features(&self, ts: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let n = ts.len();
        if n < 2 {
            return Ok(vec![0.0, 0.0]); // slope, intercept
        }

        // Simple linear trend estimation using least squares
        let x: Vec<Float> = (0..n).map(|i| i as Float).collect();
        let y: Vec<Float> = ts.to_vec();

        let x_mean = x.iter().sum::<Float>() / n as Float;
        let y_mean = y.iter().sum::<Float>() / n as Float;

        let numerator: Float = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();
        let denominator: Float = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();

        let slope = if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        };
        let intercept = y_mean - slope * x_mean;

        Ok(vec![slope, intercept])
    }

    fn extract_seasonality_features(
        &self,
        ts: &ArrayView1<Float>,
        period: usize,
    ) -> SklResult<Vec<Float>> {
        let n = ts.len();
        if n < period {
            return Ok(vec![0.0, 0.0]); // mean, amplitude
        }

        // Simple seasonality extraction: compute mean and amplitude within each period
        let mut seasonal_means = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];

        for (i, &val) in ts.iter().enumerate() {
            let season_idx = i % period;
            seasonal_means[season_idx] += val;
            seasonal_counts[season_idx] += 1;
        }

        // Normalize means
        for i in 0..period {
            if seasonal_counts[i] > 0 {
                seasonal_means[i] /= seasonal_counts[i] as Float;
            }
        }

        let overall_mean = seasonal_means.iter().sum::<Float>() / period as Float;
        let amplitude = seasonal_means
            .iter()
            .map(|&x| (x - overall_mean).abs())
            .fold(0.0_f64, |acc, x| acc.max(x));

        Ok(vec![overall_mean, amplitude])
    }
}

impl Default for TemporalFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Sliding Window Feature Extractor
///
/// Extracts features from sliding windows of time series data including
/// various statistical measures, moving averages, and volatility metrics.
///
/// # Parameters
///
/// * `window_size` - Size of the sliding window
/// * `step_size` - Step size for window sliding
/// * `features` - List of features to extract ("mean", "std", "min", "max", "median", "volatility", "skewness", "kurtosis")
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::time_series_features::SlidingWindowFeatures;
/// use scirs2_core::ndarray::{array, Array1, ArrayView1, ArrayView2};
///
/// let ts_data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let extractor = SlidingWindowFeatures::new()
///     .window_size(5)
///     .step_size(1)
///     .features(vec!["mean".to_string(), "std".to_string(), "volatility".to_string()]);
///
/// let features = extractor.extract_features(&ts_data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SlidingWindowFeatures {
    window_size: usize,
    step_size: usize,
    features: Vec<String>,
}

impl SlidingWindowFeatures {
    /// Create a new SlidingWindowFeatures extractor
    pub fn new() -> Self {
        Self {
            window_size: 10,
            step_size: 1,
            features: vec![
                "mean".to_string(),
                "std".to_string(),
                "min".to_string(),
                "max".to_string(),
                "volatility".to_string(),
            ],
        }
    }

    /// Set the window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the step size
    pub fn step_size(mut self, step_size: usize) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set the features to extract
    pub fn features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Extract sliding window features from a time series
    pub fn extract_features(&self, ts: &ArrayView1<Float>) -> SklResult<Array2<Float>> {
        let n = ts.len();
        if n < self.window_size {
            return Err(SklearsError::InvalidInput(format!(
                "Time series length ({}) must be >= window_size ({})",
                n, self.window_size
            )));
        }

        let n_windows = (n - self.window_size) / self.step_size + 1;
        let n_features_per_window = self.features.len();
        let mut result = Array2::zeros((n_windows, n_features_per_window));

        for (window_idx, start_idx) in (0..=n - self.window_size)
            .step_by(self.step_size)
            .enumerate()
        {
            let window = ts.slice(s![start_idx..start_idx + self.window_size]);

            for (feat_idx, feature_name) in self.features.iter().enumerate() {
                let feature_value = match feature_name.as_str() {
                    "mean" => window.sum() / self.window_size as Float,
                    "std" => {
                        let mean = window.sum() / self.window_size as Float;
                        (window.mapv(|x| (x - mean).powi(2)).sum() / self.window_size as Float)
                            .sqrt()
                    }
                    "min" => *window
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap(),
                    "max" => *window
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap(),
                    "median" => {
                        let mut sorted_window = window.to_vec();
                        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        if self.window_size % 2 == 1 {
                            sorted_window[self.window_size / 2]
                        } else {
                            (sorted_window[self.window_size / 2 - 1]
                                + sorted_window[self.window_size / 2])
                                / 2.0
                        }
                    }
                    "volatility" => {
                        if self.window_size > 1 {
                            let returns: Vec<Float> = window
                                .windows(2)
                                .into_iter()
                                .map(|pair| (pair[1] - pair[0]) / pair[0].max(1e-10))
                                .collect();
                            let mean_return =
                                returns.iter().sum::<Float>() / returns.len() as Float;
                            (returns
                                .iter()
                                .map(|&r| (r - mean_return).powi(2))
                                .sum::<Float>()
                                / returns.len() as Float)
                                .sqrt()
                        } else {
                            0.0
                        }
                    }
                    "skewness" => self.compute_skewness(&window),
                    "kurtosis" => self.compute_kurtosis(&window),
                    _ => 0.0, // Unknown feature
                };

                result[[window_idx, feat_idx]] = feature_value;
            }
        }

        Ok(result)
    }

    fn compute_skewness(&self, data: &ArrayView1<Float>) -> Float {
        let n = data.len() as Float;
        if n < 3.0 {
            return 0.0;
        }

        let mean = data.sum() / n;
        let variance = data.mapv(|x| (x - mean).powi(2)).sum() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        data.mapv(|x| ((x - mean) / std_dev).powi(3)).sum() / n
    }

    fn compute_kurtosis(&self, data: &ArrayView1<Float>) -> Float {
        let n = data.len() as Float;
        if n < 4.0 {
            return 0.0;
        }

        let mean = data.sum() / n;
        let variance = data.mapv(|x| (x - mean).powi(2)).sum() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        let kurtosis = data.mapv(|x| ((x - mean) / std_dev).powi(4)).sum() / n;
        kurtosis - 3.0 // Excess kurtosis (subtract 3 for normal distribution)
    }
}

impl Default for SlidingWindowFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Fourier Transform Feature Extractor
///
/// Extracts frequency domain features from time series data using Fast Fourier Transform.
/// Provides power spectrum analysis, dominant frequencies, and spectral features.
///
/// # Parameters
///
/// * `n_frequencies` - Number of frequency components to extract
/// * `include_power_spectrum` - Whether to include power spectrum features
/// * `include_phase` - Whether to include phase information
/// * `normalize` - Whether to normalize the features
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::time_series_features::FourierTransformFeatures;
/// use scirs2_core::ndarray::array;
///
/// let ts_data = array![1.0, 2.0, 1.0, -1.0, -2.0, -1.0, 1.0, 2.0];
/// let extractor = FourierTransformFeatures::new()
///     .n_frequencies(4)
///     .include_power_spectrum(true)
///     .normalize(true);
///
/// let features = extractor.extract_features(&ts_data.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FourierTransformFeatures {
    n_frequencies: usize,
    include_power_spectrum: bool,
    include_phase: bool,
    normalize: bool,
    window_function: String, // "hanning", "hamming", "blackman", "none"
}

impl FourierTransformFeatures {
    /// Create a new FourierTransformFeatures extractor
    pub fn new() -> Self {
        Self {
            n_frequencies: 10,
            include_power_spectrum: true,
            include_phase: false,
            normalize: true,
            window_function: "hanning".to_string(),
        }
    }

    /// Set the number of frequency components to extract
    pub fn n_frequencies(mut self, n_frequencies: usize) -> Self {
        self.n_frequencies = n_frequencies;
        self
    }

    /// Set whether to include power spectrum features
    pub fn include_power_spectrum(mut self, include_power_spectrum: bool) -> Self {
        self.include_power_spectrum = include_power_spectrum;
        self
    }

    /// Set whether to include phase information
    pub fn include_phase(mut self, include_phase: bool) -> Self {
        self.include_phase = include_phase;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the window function
    pub fn window_function(mut self, window_function: String) -> Self {
        self.window_function = window_function;
        self
    }

    /// Extract Fourier transform features from a time series
    pub fn extract_features(&self, ts: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n = ts.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Time series cannot be empty".to_string(),
            ));
        }

        // Apply window function
        let windowed_ts = self.apply_window(ts)?;

        // Compute DFT (simplified implementation - not optimized FFT)
        let fft_result = self.compute_dft(&windowed_ts)?;

        let mut features = Vec::new();

        // Extract frequency components up to n_frequencies
        let max_freq = (self.n_frequencies.min(n / 2)).max(1);

        if self.include_power_spectrum {
            for i in 0..max_freq {
                let power = fft_result[i].0.powi(2) + fft_result[i].1.powi(2); // |X[k]|^2
                features.push(power);
            }
        }

        if self.include_phase {
            for i in 0..max_freq {
                let phase = fft_result[i].1.atan2(fft_result[i].0); // angle(X[k])
                features.push(phase);
            }
        }

        // Add spectral features
        let spectral_features = self.compute_spectral_features(&fft_result, max_freq)?;
        features.extend_from_slice(&spectral_features);

        // Normalize if requested
        if self.normalize && !features.is_empty() {
            let max_val = features.iter().cloned().fold(0.0, Float::max);
            if max_val > 1e-10 {
                for feature in &mut features {
                    *feature /= max_val;
                }
            }
        }

        Ok(Array1::from_vec(features))
    }

    fn apply_window(&self, ts: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n = ts.len();
        let mut windowed = Array1::zeros(n);

        match self.window_function.as_str() {
            "hanning" => {
                for i in 0..n {
                    let w = 0.5
                        * (1.0
                            - (2.0 * std::f64::consts::PI * i as Float / (n - 1) as Float).cos());
                    windowed[i] = ts[i] * w;
                }
            }
            "hamming" => {
                for i in 0..n {
                    let w = 0.54
                        - 0.46 * (2.0 * std::f64::consts::PI * i as Float / (n - 1) as Float).cos();
                    windowed[i] = ts[i] * w;
                }
            }
            "blackman" => {
                for i in 0..n {
                    let w = 0.42
                        - 0.5 * (2.0 * std::f64::consts::PI * i as Float / (n - 1) as Float).cos()
                        + 0.08 * (4.0 * std::f64::consts::PI * i as Float / (n - 1) as Float).cos();
                    windowed[i] = ts[i] * w;
                }
            }
            _ => {
                // No window applied
                windowed.assign(ts);
            }
        }

        Ok(windowed)
    }

    fn compute_dft(&self, ts: &Array1<Float>) -> SklResult<Vec<(Float, Float)>> {
        let n = ts.len();
        let mut result = vec![(0.0, 0.0); n];

        // Simple DFT implementation (O(n^2) - for production use, implement FFT)
        for k in 0..n {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * j) as Float / n as Float;
                real_sum += ts[j] * angle.cos();
                imag_sum += ts[j] * angle.sin();
            }

            result[k] = (real_sum, imag_sum);
        }

        Ok(result)
    }

    fn compute_spectral_features(
        &self,
        fft_result: &[(Float, Float)],
        max_freq: usize,
    ) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();

        // Spectral centroid (center of mass of spectrum)
        let mut weighted_sum = 0.0;
        let mut total_power = 0.0;

        for i in 0..max_freq {
            let power = fft_result[i].0.powi(2) + fft_result[i].1.powi(2);
            weighted_sum += i as Float * power;
            total_power += power;
        }

        let spectral_centroid = if total_power > 1e-10 {
            weighted_sum / total_power
        } else {
            0.0
        };
        features.push(spectral_centroid);

        // Spectral bandwidth
        let mut bandwidth_sum = 0.0;
        for i in 0..max_freq {
            let power = fft_result[i].0.powi(2) + fft_result[i].1.powi(2);
            bandwidth_sum += (i as Float - spectral_centroid).powi(2) * power;
        }

        let spectral_bandwidth = if total_power > 1e-10 {
            (bandwidth_sum / total_power).sqrt()
        } else {
            0.0
        };
        features.push(spectral_bandwidth);

        // Spectral rolloff (frequency below which 85% of energy is contained)
        let target_energy = 0.85 * total_power;
        let mut cumulative_energy = 0.0;
        let mut rolloff_freq = 0.0;

        for i in 0..max_freq {
            let power = fft_result[i].0.powi(2) + fft_result[i].1.powi(2);
            cumulative_energy += power;
            if cumulative_energy >= target_energy {
                rolloff_freq = i as Float;
                break;
            }
        }
        features.push(rolloff_freq);

        Ok(features)
    }
}

impl Default for FourierTransformFeatures {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Wavelet-based Time Series Feature Extraction
// =============================================================================

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
    /// Basic
    Basic,
    /// Extended
    Extended,
}

/// Wavelet decomposition result
struct WaveletDecomposition {
    approximation: Array1<Float>,
    details: Vec<Array1<Float>>,
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
                WaveletFeatureType::Basic => {
                    // Basic: compute mean as a simple feature
                    coeffs.mean().unwrap_or(0.0)
                }
                WaveletFeatureType::Extended => {
                    // Extended: compute variance as an extended feature
                    coeffs.var(0.0)
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

// =============================================================================
// Time-Frequency Feature Extraction
// =============================================================================

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

// =============================================================================
// Temporal-Spatial Feature Extraction
// =============================================================================

/// Temporal-Spatial Feature Extractor
///
/// Extracts features that capture both temporal and spatial relationships
/// in multi-dimensional time series data. This includes autocorrelation,
/// temporal lag correlations, gradients, coherence, and interaction features.
///
/// # Parameters
///
/// * `max_temporal_lag` - Maximum temporal lag for correlation analysis
/// * `spatial_window_size` - Spatial window size for spatial feature computation
/// * `include_autocorrelation` - Whether to include spatial autocorrelation features
/// * `include_temporal_correlations` - Whether to include temporal correlation features
/// * `include_gradients` - Whether to include gradient features
/// * `include_coherence` - Whether to include coherence features
/// * `include_interactions` - Whether to include interaction features
/// * `normalize_features` - Whether to normalize features
///
/// # Examples
///
/// ```
/// use sklears_feature_extraction::time_series_features::TemporalSpatialExtractor;
/// use scirs2_core::ndarray::Array2;
///
/// let extractor = TemporalSpatialExtractor::new()
///     .max_temporal_lag(3)
///     .include_gradients(true);
///
/// let data = Array2::from_elem((10, 3), 1.0);
/// let features = extractor.extract_features(&data.view()).unwrap();
/// ```
pub struct TemporalSpatialExtractor {
    max_temporal_lag: usize,
    spatial_window_size: usize,
    include_autocorrelation: bool,
    include_temporal_correlations: bool,
    include_gradients: bool,
    include_coherence: bool,
    include_interactions: bool,
    normalize_features: bool,
}

impl TemporalSpatialExtractor {
    /// Create a new TemporalSpatialExtractor
    pub fn new() -> Self {
        Self {
            max_temporal_lag: 5,
            spatial_window_size: 3,
            include_autocorrelation: true,
            include_temporal_correlations: true,
            include_gradients: true,
            include_coherence: true,
            include_interactions: true,
            normalize_features: true,
        }
    }

    /// Set the maximum temporal lag for correlation analysis
    pub fn max_temporal_lag(mut self, lag: usize) -> Self {
        self.max_temporal_lag = lag;
        self
    }

    /// Set the spatial window size for spatial feature computation
    pub fn spatial_window_size(mut self, size: usize) -> Self {
        self.spatial_window_size = size;
        self
    }

    /// Set whether to include spatial autocorrelation features
    pub fn include_autocorrelation(mut self, include: bool) -> Self {
        self.include_autocorrelation = include;
        self
    }

    /// Set whether to include temporal correlation features
    pub fn include_temporal_correlations(mut self, include: bool) -> Self {
        self.include_temporal_correlations = include;
        self
    }

    /// Set whether to include gradient features
    pub fn include_gradients(mut self, include: bool) -> Self {
        self.include_gradients = include;
        self
    }

    /// Set whether to include coherence features
    pub fn include_coherence(mut self, include: bool) -> Self {
        self.include_coherence = include;
        self
    }

    /// Set whether to include interaction features
    pub fn include_interactions(mut self, include: bool) -> Self {
        self.include_interactions = include;
        self
    }

    /// Set whether to normalize features
    pub fn normalize_features(mut self, normalize: bool) -> Self {
        self.normalize_features = normalize;
        self
    }

    /// Extract temporal-spatial features from multi-dimensional time series
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        let (n_timesteps, _n_dimensions) = data.dim();

        if n_timesteps < self.max_temporal_lag + 1 {
            return Err(SklearsError::InvalidInput(
                "Not enough timesteps for specified temporal lag".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Spatial autocorrelation features
        if self.include_autocorrelation {
            let autocorr_features = self.compute_spatial_autocorrelation(data)?;
            features.extend(autocorr_features);
        }

        // Temporal lag correlation features
        if self.include_temporal_correlations {
            let temporal_features = self.compute_temporal_correlations(data)?;
            features.extend(temporal_features);
        }

        // Gradient features
        if self.include_gradients {
            let gradient_features = self.compute_gradients(data)?;
            features.extend(gradient_features);
        }

        // Coherence features
        if self.include_coherence {
            let coherence_features = self.compute_coherence(data)?;
            features.extend(coherence_features);
        }

        // Interaction features
        if self.include_interactions {
            let interaction_features = self.compute_interactions(data)?;
            features.extend(interaction_features);
        }

        if features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected for extraction".to_string(),
            ));
        }

        let mut feature_array = Array1::from_vec(features);

        // Normalize features if requested
        if self.normalize_features {
            let mean = feature_array.mean().unwrap_or(0.0);
            let std = feature_array.std(0.0);

            if std > 1e-10 {
                feature_array = (feature_array - mean) / std;
            }
        }

        Ok(feature_array)
    }

    /// Compute spatial autocorrelation features
    fn compute_spatial_autocorrelation(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let (_, n_dimensions) = data.dim();
        let mut features = Vec::new();

        // Moran's I statistic for spatial autocorrelation
        for t in 0..data.nrows() {
            let time_slice = data.row(t);

            // Compute spatial weights (inverse distance based)
            let mut weights = Array2::zeros((n_dimensions, n_dimensions));
            for i in 0..n_dimensions {
                for j in 0..n_dimensions {
                    if i != j {
                        let distance = ((i as f64 - j as f64).abs() + 1.0).recip();
                        weights[[i, j]] = distance;
                    }
                }
            }

            // Normalize weights
            let weight_sum: Float = weights.sum();
            if weight_sum > 0.0 {
                weights /= weight_sum;
            }

            // Compute Moran's I
            let mean_val = time_slice.mean().unwrap_or(0.0);
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 0..n_dimensions {
                for j in 0..n_dimensions {
                    numerator +=
                        weights[[i, j]] * (time_slice[i] - mean_val) * (time_slice[j] - mean_val);
                }
                denominator += (time_slice[i] - mean_val).powi(2);
            }

            let morans_i = if denominator > 1e-10 {
                numerator / denominator
            } else {
                0.0
            };

            features.push(morans_i);
        }

        // Global spatial autocorrelation across all timesteps
        let global_autocorr = features.iter().sum::<Float>() / features.len() as Float;
        features.push(global_autocorr);

        Ok(features)
    }

    /// Compute temporal lag correlation features
    fn compute_temporal_correlations(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let (n_timesteps, n_dimensions) = data.dim();
        let mut features = Vec::new();

        for dim in 0..n_dimensions {
            let dim_series = data.column(dim);

            for lag in 1..=self.max_temporal_lag {
                if lag < n_timesteps {
                    let series1 = dim_series.slice(s![..n_timesteps - lag]);
                    let series2 = dim_series.slice(s![lag..]);

                    // Compute Pearson correlation
                    let mean1 = series1.mean().unwrap_or(0.0);
                    let mean2 = series2.mean().unwrap_or(0.0);

                    let mut numerator = 0.0;
                    let mut denom1 = 0.0;
                    let mut denom2 = 0.0;

                    for i in 0..series1.len() {
                        let diff1 = series1[i] - mean1;
                        let diff2 = series2[i] - mean2;
                        numerator += diff1 * diff2;
                        denom1 += diff1 * diff1;
                        denom2 += diff2 * diff2;
                    }

                    let correlation = if denom1 > 1e-10 && denom2 > 1e-10 {
                        numerator / (denom1.sqrt() * denom2.sqrt())
                    } else {
                        0.0
                    };

                    features.push(correlation);
                }
            }
        }

        Ok(features)
    }

    /// Compute gradient features (temporal and spatial gradients)
    fn compute_gradients(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let (n_timesteps, n_dimensions) = data.dim();
        let mut features = Vec::new();

        // Temporal gradients
        for dim in 0..n_dimensions {
            let mut temporal_gradients = Vec::new();
            for t in 1..n_timesteps {
                let gradient = data[[t, dim]] - data[[t - 1, dim]];
                temporal_gradients.push(gradient);
            }

            if !temporal_gradients.is_empty() {
                let mean_grad =
                    temporal_gradients.iter().sum::<Float>() / temporal_gradients.len() as Float;
                let var_grad = temporal_gradients
                    .iter()
                    .map(|&g| (g - mean_grad).powi(2))
                    .sum::<Float>()
                    / temporal_gradients.len() as Float;

                features.push(mean_grad);
                features.push(var_grad.sqrt());
            }
        }

        // Spatial gradients
        for t in 0..n_timesteps {
            let mut spatial_gradients = Vec::new();
            for dim in 1..n_dimensions {
                let gradient = data[[t, dim]] - data[[t, dim - 1]];
                spatial_gradients.push(gradient);
            }

            if !spatial_gradients.is_empty() {
                let mean_grad =
                    spatial_gradients.iter().sum::<Float>() / spatial_gradients.len() as Float;
                features.push(mean_grad);
            }
        }

        Ok(features)
    }

    /// Compute coherence features (measure of synchronization)
    fn compute_coherence(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let (_n_timesteps, n_dimensions) = data.dim();
        let mut features = Vec::new();

        // Phase coherence between dimensions
        for i in 0..n_dimensions {
            for j in (i + 1)..n_dimensions {
                let series1 = data.column(i);
                let series2 = data.column(j);

                // Compute instantaneous phases using Hilbert transform approximation
                let phase_diff = self.compute_phase_difference(&series1, &series2)?;

                // Compute phase locking value (PLV)
                let mut complex_sum_real = 0.0;
                let mut complex_sum_imag = 0.0;

                for &diff in &phase_diff {
                    complex_sum_real += diff.cos();
                    complex_sum_imag += diff.sin();
                }

                let plv = ((complex_sum_real.powi(2) + complex_sum_imag.powi(2)).sqrt()
                    / phase_diff.len() as Float)
                    .abs();
                features.push(plv);
            }
        }

        // Global coherence
        if !features.is_empty() {
            let global_coherence = (features.iter().sum::<Float>() / features.len() as Float).abs();
            features.push(global_coherence);
        }

        Ok(features)
    }

    /// Compute interaction features between dimensions
    fn compute_interactions(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let (_, n_dimensions) = data.dim();
        let mut features = Vec::new();

        // Cross-dimensional variance ratios
        for i in 0..n_dimensions {
            for j in (i + 1)..n_dimensions {
                let series1 = data.column(i);
                let series2 = data.column(j);

                let var1 = series1.var(0.0);
                let var2 = series2.var(0.0);

                let var_ratio = if var2 > 1e-10 {
                    var1 / var2
                } else if var1 > 1e-10 {
                    var1 / 1e-10
                } else {
                    1.0
                };

                features.push(var_ratio);
            }
        }

        // Cross-dimensional energy ratios
        for i in 0..n_dimensions {
            for j in (i + 1)..n_dimensions {
                let series1 = data.column(i);
                let series2 = data.column(j);

                let energy1: Float = series1.iter().map(|&x| x.powi(2)).sum();
                let energy2: Float = series2.iter().map(|&x| x.powi(2)).sum();

                let energy_ratio = if energy2 > 1e-10 {
                    energy1 / energy2
                } else if energy1 > 1e-10 {
                    energy1 / 1e-10
                } else {
                    1.0
                };

                features.push(energy_ratio);
            }
        }

        Ok(features)
    }

    /// Compute phase difference between two time series
    fn compute_phase_difference(
        &self,
        series1: &ArrayView1<Float>,
        series2: &ArrayView1<Float>,
    ) -> SklResult<Vec<Float>> {
        if series1.len() != series2.len() {
            return Err(SklearsError::InvalidInput(
                "Time series must have same length".to_string(),
            ));
        }

        let mut phase_diff = Vec::new();

        // Simple phase estimation using derivative approximation
        for i in 1..series1.len() {
            let phase1 = (series1[i] - series1[i - 1]).atan2(series1[i]);
            let phase2 = (series2[i] - series2[i - 1]).atan2(series2[i]);

            let mut diff = phase1 - phase2;

            // Wrap phase difference to [-π, π]
            while diff > std::f64::consts::PI {
                diff -= 2.0 * std::f64::consts::PI;
            }
            while diff < -std::f64::consts::PI {
                diff += 2.0 * std::f64::consts::PI;
            }

            phase_diff.push(diff);
        }

        Ok(phase_diff)
    }
}

impl Default for TemporalSpatialExtractor {
    fn default() -> Self {
        Self::new()
    }
}
