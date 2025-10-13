//! Temporal Feature Extraction
//!
//! This module provides comprehensive time series feature extraction capabilities:
//!
//! - `TemporalFeatureExtractor`: Lag features, rolling statistics, trend, and seasonality
//! - `SlidingWindowFeatures`: Statistical measures from sliding windows
//! - `FourierTransformFeatures`: Frequency domain features using DFT
//!
//! These extractors enable rich temporal pattern recognition for time series analysis.

use crate::{Float, SklResult, SklearsError};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};

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
/// use sklears_feature_extraction::temporal_features::TemporalFeatureExtractor;
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
/// use sklears_feature_extraction::temporal_features::SlidingWindowFeatures;
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
/// use sklears_feature_extraction::temporal_features::FourierTransformFeatures;
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
