//! Fast Transform Methods Module
//!
//! This module provides efficient transform algorithms for feature extraction
//! including FFT, DCT, Walsh-Hadamard, and Haar transforms.

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::prelude::SklearsError;
use sklears_core::types::Float;

/// Types of fast transforms available
#[derive(Debug, Clone, PartialEq)]
pub enum TransformType {
    /// Fast Fourier Transform
    FFT,
    /// Discrete Cosine Transform
    DCT,
    /// Walsh-Hadamard Transform
    Walsh,
    /// Haar Transform
    Haar,
}

/// Fast Transform Methods for Efficient Feature Extraction
///
/// Implements various fast transform algorithms for efficient computation
/// of frequency domain features and other transform-based features.
///
/// This extractor provides multiple transform types optimized for different
/// use cases, from frequency analysis to compression and multiresolution analysis.
///
/// # Transform Types
///
/// * `FFT` - Fast Fourier Transform for frequency analysis
/// * `DCT` - Discrete Cosine Transform for compression and feature extraction
/// * `Walsh` - Walsh-Hadamard Transform for fast computation
/// * `Haar` - Haar Transform for multiresolution analysis
///
/// # Parameters
///
/// * `transform_type` - Type of transform to apply
/// * `include_magnitude` - Whether to include magnitude features
/// * `include_phase` - Whether to include phase features
/// * `include_power` - Whether to include power spectrum features
/// * `normalize` - Whether to normalize the output
/// * `zero_pad_to_power_of_2` - Whether to zero-pad to next power of 2
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::{FastTransformExtractor, TransformType};
/// # use scirs2_core::ndarray::Array2;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((3, 8), (0..24).map(|x| x as f64).collect())?;
///
/// let extractor = FastTransformExtractor::new()
///     .transform_type(TransformType::FFT)
///     .include_magnitude(true)
///     .include_phase(false);
///
/// let features = extractor.extract_features(&data.view())?;
/// assert_eq!(features.nrows(), 3);
/// assert!(features.ncols() > 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FastTransformExtractor {
    transform_type: TransformType,
    include_magnitude: bool,
    include_phase: bool,
    include_power: bool,
    normalize: bool,
    zero_pad_to_power_of_2: bool,
}

impl FastTransformExtractor {
    /// Create a new FastTransformExtractor
    pub fn new() -> Self {
        Self {
            transform_type: TransformType::FFT,
            include_magnitude: true,
            include_phase: false,
            include_power: false,
            normalize: true,
            zero_pad_to_power_of_2: true,
        }
    }

    /// Set the transform type
    pub fn transform_type(mut self, transform_type: TransformType) -> Self {
        self.transform_type = transform_type;
        self
    }

    /// Set whether to include magnitude features
    pub fn include_magnitude(mut self, include: bool) -> Self {
        self.include_magnitude = include;
        self
    }

    /// Set whether to include phase features
    pub fn include_phase(mut self, include: bool) -> Self {
        self.include_phase = include;
        self
    }

    /// Set whether to include power spectrum features
    pub fn include_power(mut self, include: bool) -> Self {
        self.include_power = include;
        self
    }

    /// Set whether to normalize the output
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set whether to zero-pad to the next power of 2
    pub fn zero_pad_to_power_of_2(mut self, zero_pad: bool) -> Self {
        self.zero_pad_to_power_of_2 = zero_pad;
        self
    }

    /// Extract features using the specified transform
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        let mut features_list = Vec::new();

        for row in data.axis_iter(Axis(0)) {
            let row_features = match self.transform_type {
                TransformType::FFT => self.compute_fft(&row)?,
                TransformType::DCT => self.compute_dct(&row)?,
                TransformType::Walsh => self.compute_walsh(&row)?,
                TransformType::Haar => self.compute_haar(&row)?,
            };
            features_list.push(row_features);
        }

        if features_list.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features extracted".to_string(),
            ));
        }

        let n_features = features_list[0].len();
        let n_samples = features_list.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, features) in features_list.into_iter().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                result[[i, j]] = feature;
            }
        }

        Ok(result)
    }

    /// Compute Fast Fourier Transform features
    fn compute_fft(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut padded_signal = signal.to_vec();

        // Zero-pad to power of 2 if requested
        if self.zero_pad_to_power_of_2 && !padded_signal.is_empty() {
            let next_power_of_2 = (padded_signal.len() as f64).log2().ceil() as u32;
            let target_size = 2_usize.pow(next_power_of_2);
            padded_signal.resize(target_size, 0.0);
        }

        // Simple DFT implementation (can be optimized with real FFT libraries)
        let n = padded_signal.len();
        let mut real_part = Vec::with_capacity(n / 2 + 1);
        let mut imag_part = Vec::with_capacity(n / 2 + 1);

        for k in 0..=n / 2 {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
                real_sum += padded_signal[j] * angle.cos();
                imag_sum += padded_signal[j] * angle.sin();
            }

            real_part.push(real_sum);
            imag_part.push(imag_sum);
        }

        let mut features = Vec::new();

        if self.include_magnitude {
            for (real, imag) in real_part.iter().zip(imag_part.iter()) {
                let magnitude = (real * real + imag * imag).sqrt();
                features.push(magnitude);
            }
        }

        if self.include_phase {
            for (real, imag) in real_part.iter().zip(imag_part.iter()) {
                let phase = imag.atan2(*real);
                features.push(phase);
            }
        }

        if self.include_power {
            for (real, imag) in real_part.iter().zip(imag_part.iter()) {
                let power = real * real + imag * imag;
                features.push(power);
            }
        }

        if self.normalize && !features.is_empty() {
            let max_val = features.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_val > 0.0 {
                for feature in &mut features {
                    *feature /= max_val;
                }
            }
        }

        Ok(features)
    }

    /// Compute Discrete Cosine Transform features
    fn compute_dct(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let data = signal.to_vec();
        let n = data.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut dct_coeffs = Vec::with_capacity(n);

        for k in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                let angle = std::f64::consts::PI * (k as f64) * (2.0 * i as f64 + 1.0) / (2.0 * n as f64);
                sum += data[i] * angle.cos();
            }

            // Apply DCT normalization
            let coeff = if k == 0 {
                sum * (1.0 / n as f64).sqrt()
            } else {
                sum * (2.0 / n as f64).sqrt()
            };

            dct_coeffs.push(coeff);
        }

        if self.normalize && !dct_coeffs.is_empty() {
            let max_val = dct_coeffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_val > 0.0 {
                for coeff in &mut dct_coeffs {
                    *coeff /= max_val;
                }
            }
        }

        Ok(dct_coeffs)
    }

    /// Compute Walsh-Hadamard Transform features
    fn compute_walsh(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut data = signal.to_vec();
        let n = data.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // Ensure length is power of 2
        let next_power_of_2 = (n as f64).log2().ceil() as u32;
        let target_size = 2_usize.pow(next_power_of_2);
        data.resize(target_size, 0.0);

        // Fast Walsh-Hadamard Transform
        let mut step = 1;
        while step < target_size {
            for i in (0..target_size).step_by(step * 2) {
                for j in 0..step {
                    let u = data[i + j];
                    let v = data[i + j + step];
                    data[i + j] = u + v;
                    data[i + j + step] = u - v;
                }
            }
            step *= 2;
        }

        if self.normalize && !data.is_empty() {
            let norm_factor = (target_size as f64).sqrt();
            for value in &mut data {
                *value /= norm_factor;
            }
        }

        // Return only original size if we padded
        if target_size > n {
            data.truncate(n);
        }

        Ok(data)
    }

    /// Compute Haar Transform features
    fn compute_haar(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        let mut data = signal.to_vec();
        let n = data.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // Ensure length is power of 2
        let next_power_of_2 = (n as f64).log2().ceil() as u32;
        let target_size = 2_usize.pow(next_power_of_2);
        data.resize(target_size, 0.0);

        let mut temp = vec![0.0; target_size];
        let mut h = target_size;

        while h > 1 {
            h /= 2;
            for i in 0..h {
                let k = 2 * i;
                temp[i] = (data[k] + data[k + 1]) / 2.0_f64.sqrt();
                temp[i + h] = (data[k] - data[k + 1]) / 2.0_f64.sqrt();
            }
            data[..2 * h].copy_from_slice(&temp[..2 * h]);
        }

        if self.normalize && !data.is_empty() {
            let norm_factor = (target_size as f64).sqrt();
            for value in &mut data {
                *value /= norm_factor;
            }
        }

        // Return only original size if we padded
        if target_size > n {
            data.truncate(n);
        }

        Ok(data)
    }
}

impl Default for FastTransformExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast Walsh-Hadamard Transform utility
///
/// Standalone implementation of the Walsh-Hadamard transform for use
/// in other algorithms like Fast Johnson-Lindenstrauss Transform.
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::fast_hadamard_transform;
/// # use scirs2_core::ndarray::Array1;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// let transformed = fast_hadamard_transform(data)?;
/// assert_eq!(transformed.len(), 8);
/// # Ok(())
/// # }
/// ```
pub fn fast_hadamard_transform(mut data: Array1<f64>) -> SklResult<Array1<f64>> {
    let n = data.len();

    if n == 0 {
        return Ok(data);
    }

    // Find the next power of 2
    let next_pow2 = n.next_power_of_two();

    // Pad with zeros if necessary
    if next_pow2 > n {
        let mut padded = Array1::zeros(next_pow2);
        padded.slice_mut(scirs2_core::ndarray::s![..n]).assign(&data);
        data = padded;
    }

    let len = data.len();
    let mut h = 1;

    while h < len {
        for i in (0..len).step_by(h * 2) {
            for j in i..(i + h).min(len) {
                if j + h < len {
                    let a = data[j];
                    let b = data[j + h];
                    data[j] = a + b;
                    data[j + h] = a - b;
                }
            }
        }
        h *= 2;
    }

    // Normalize
    let normalizer = (len as f64).sqrt();
    data.mapv_inplace(|x| x / normalizer);

    // Return only the original size
    if next_pow2 > n {
        Ok(data.slice(scirs2_core::ndarray::s![..n]).to_owned())
    } else {
        Ok(data)
    }
}

/// Spectral Features Extractor
///
/// Extracts various spectral features from signals including energy,
/// spectral centroid, rolloff, and other frequency-domain characteristics.
///
/// # Parameters
///
/// * `sample_rate` - Sample rate of the signal (for frequency calculations)
/// * `include_energy` - Whether to include spectral energy
/// * `include_centroid` - Whether to include spectral centroid
/// * `include_rolloff` - Whether to include spectral rolloff
/// * `include_flux` - Whether to include spectral flux
/// * `rolloff_threshold` - Threshold for spectral rolloff calculation
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::SpectralFeaturesExtractor;
/// # use scirs2_core::ndarray::Array2;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((3, 16), (0..48).map(|x| x as f64).collect())?;
///
/// let extractor = SpectralFeaturesExtractor::new()
///     .sample_rate(44100.0)
///     .include_energy(true)
///     .include_centroid(true)
///     .include_rolloff(true);
///
/// let features = extractor.extract_features(&data.view())?;
/// assert_eq!(features.nrows(), 3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SpectralFeaturesExtractor {
    sample_rate: f64,
    include_energy: bool,
    include_centroid: bool,
    include_rolloff: bool,
    include_flux: bool,
    rolloff_threshold: f64,
}

impl SpectralFeaturesExtractor {
    /// Create a new SpectralFeaturesExtractor
    pub fn new() -> Self {
        Self {
            sample_rate: 44100.0,
            include_energy: true,
            include_centroid: true,
            include_rolloff: true,
            include_flux: false,
            rolloff_threshold: 0.85,
        }
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set whether to include spectral energy
    pub fn include_energy(mut self, include: bool) -> Self {
        self.include_energy = include;
        self
    }

    /// Set whether to include spectral centroid
    pub fn include_centroid(mut self, include: bool) -> Self {
        self.include_centroid = include;
        self
    }

    /// Set whether to include spectral rolloff
    pub fn include_rolloff(mut self, include: bool) -> Self {
        self.include_rolloff = include;
        self
    }

    /// Set whether to include spectral flux
    pub fn include_flux(mut self, include: bool) -> Self {
        self.include_flux = include;
        self
    }

    /// Set the rolloff threshold
    pub fn rolloff_threshold(mut self, threshold: f64) -> Self {
        self.rolloff_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Extract spectral features
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        let mut features_list = Vec::new();

        for row in data.axis_iter(Axis(0)) {
            let row_features = self.compute_spectral_features(&row)?;
            features_list.push(row_features);
        }

        if features_list.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let n_features = features_list[0].len();
        let n_samples = features_list.len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, features) in features_list.into_iter().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                result[[i, j]] = feature;
            }
        }

        Ok(result)
    }

    /// Compute spectral features for a single signal
    fn compute_spectral_features(&self, signal: &ArrayView1<Float>) -> SklResult<Vec<Float>> {
        // First compute the magnitude spectrum
        let fft_extractor = FastTransformExtractor::new()
            .transform_type(TransformType::FFT)
            .include_magnitude(true)
            .include_phase(false)
            .normalize(false);

        let spectrum_data = Array2::from_shape_vec((1, signal.len()), signal.to_vec())
            .map_err(|_| SklearsError::InvalidInput("Failed to reshape signal".to_string()))?;

        let spectrum_result = fft_extractor.extract_features(&spectrum_data.view())?;
        let spectrum = spectrum_result.row(0);

        let mut features = Vec::new();

        if self.include_energy {
            let energy = spectrum.iter().map(|&x| x * x).sum::<Float>();
            features.push(energy);
        }

        if self.include_centroid {
            let centroid = self.compute_spectral_centroid(&spectrum);
            features.push(centroid);
        }

        if self.include_rolloff {
            let rolloff = self.compute_spectral_rolloff(&spectrum);
            features.push(rolloff);
        }

        if self.include_flux {
            // For flux, we would need the previous frame, so we'll use a simple approximation
            let flux = spectrum.iter().map(|&x| x.abs()).sum::<Float>() / spectrum.len() as Float;
            features.push(flux);
        }

        Ok(features)
    }

    /// Compute spectral centroid
    fn compute_spectral_centroid(&self, spectrum: &ArrayView1<Float>) -> Float {
        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let frequency = i as f64 * self.sample_rate / (2.0 * spectrum.len() as f64);
            weighted_sum += frequency * magnitude;
            total_magnitude += magnitude;
        }

        if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            0.0
        }
    }

    /// Compute spectral rolloff
    fn compute_spectral_rolloff(&self, spectrum: &ArrayView1<Float>) -> Float {
        let total_energy: Float = spectrum.iter().map(|&x| x * x).sum();
        let threshold = total_energy * self.rolloff_threshold;

        let mut cumulative_energy = 0.0;
        for (i, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude * magnitude;
            if cumulative_energy >= threshold {
                let frequency = i as f64 * self.sample_rate / (2.0 * spectrum.len() as f64);
                return frequency;
            }
        }

        // If we reach here, return the Nyquist frequency
        self.sample_rate / 2.0
    }
}

impl Default for SpectralFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_fast_transform_extractor_fft() {
        let data = Array2::from_shape_vec((3, 8), (0..24).map(|x| x as f64).collect()).unwrap();
        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::FFT)
            .include_magnitude(true)
            .include_phase(false);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);

        // All features should be finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_fast_transform_extractor_dct() {
        let data = Array2::from_shape_vec((2, 4), (0..8).map(|x| x as f64).collect()).unwrap();
        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::DCT)
            .normalize(true);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 4);

        // All features should be finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_fast_transform_extractor_walsh() {
        let data = Array2::from_shape_vec((2, 8), (0..16).map(|x| x as f64).collect()).unwrap();
        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::Walsh)
            .normalize(true);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 2);
        assert!(features.ncols() > 0);

        // All features should be finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_fast_transform_extractor_haar() {
        let data = Array2::from_shape_vec((2, 4), (0..8).map(|x| x as f64).collect()).unwrap();
        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::Haar)
            .normalize(true);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 2);
        assert_eq!(features.ncols(), 4);

        // All features should be finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_fast_hadamard_transform() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let transformed = fast_hadamard_transform(data).unwrap();
        assert_eq!(transformed.len(), 8);

        // All values should be finite
        for &value in transformed.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_spectral_features_extractor() {
        let data = Array2::from_shape_vec((2, 16), (0..32).map(|x| x as f64).collect()).unwrap();
        let extractor = SpectralFeaturesExtractor::new()
            .sample_rate(44100.0)
            .include_energy(true)
            .include_centroid(true)
            .include_rolloff(true);

        let features = extractor.extract_features(&data.view()).unwrap();
        assert_eq!(features.nrows(), 2);
        assert!(features.ncols() > 0);

        // All features should be finite and non-negative for these metrics
        for &value in features.iter() {
            assert!(value.is_finite());
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_transform_empty_data() {
        let empty_data = Array2::zeros((0, 0));
        let extractor = FastTransformExtractor::new();

        let result = extractor.extract_features(&empty_data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_consistency() {
        let data = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f64).collect()).unwrap();
        let extractor = FastTransformExtractor::new()
            .transform_type(TransformType::DCT)
            .normalize(true);

        let features1 = extractor.extract_features(&data.view()).unwrap();
        let features2 = extractor.extract_features(&data.view()).unwrap();

        // Results should be identical
        assert_eq!(features1.shape(), features2.shape());
        for (a, b) in features1.iter().zip(features2.iter()) {
            assert_eq!(a, b);
        }
    }
}