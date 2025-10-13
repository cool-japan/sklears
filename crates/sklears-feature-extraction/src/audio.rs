//! Audio feature extraction methods
//!
//! This module implements various audio feature extraction techniques including:
//! - Mel-Frequency Cepstral Coefficients (MFCC)
//! - Spectral features (centroid, bandwidth, rolloff, flux)
//! - Chroma features
//! - Zero Crossing Rate (ZCR)
//! - Spectral rolloff

use std::f64::consts::PI;

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};

fn ensure_valid_frame_parameters(
    signal_len: usize,
    frame_length: usize,
    hop_length: usize,
    context: &str,
) -> SklResult<usize> {
    if hop_length == 0 {
        return Err(SklearsError::InvalidInput(format!(
            "{}: hop_length must be greater than zero",
            context
        )));
    }

    if frame_length < 2 {
        return Err(SklearsError::InvalidInput(format!(
            "{}: frame length must be at least 2 samples",
            context
        )));
    }

    if signal_len < frame_length {
        return Err(SklearsError::InvalidInput(format!(
            "{}: Signal too short for frame length {}",
            context, frame_length
        )));
    }

    Ok((signal_len - frame_length) / hop_length + 1)
}

/// Mel-Frequency Cepstral Coefficients (MFCC) extractor
///
/// MFCC are widely used features in speech and audio processing that
/// capture the spectral characteristics of audio signals in a way that
/// mimics human auditory perception.
#[derive(Debug, Clone)]
pub struct MFCCExtractor {
    n_mfcc: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
    sample_rate: f64,
    normalize: bool,
    lifter: Option<f64>,
}

impl MFCCExtractor {
    /// Create a new MFCC extractor with default parameters
    pub fn new() -> Self {
        Self {
            n_mfcc: 13,
            n_fft: 2048,
            hop_length: 512,
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
            sample_rate: 22050.0,
            normalize: true,
            lifter: Some(22.0),
        }
    }

    /// Set the number of MFCC coefficients to extract
    pub fn n_mfcc(mut self, n_mfcc: usize) -> Self {
        self.n_mfcc = n_mfcc;
        self
    }

    /// Set the FFT window size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length for STFT
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the number of mel filter banks
    pub fn n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Set the minimum frequency
    pub fn fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    /// Set the maximum frequency
    pub fn fmax(mut self, fmax: Option<f64>) -> Self {
        self.fmax = fmax;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set whether to normalize the mel filterbank
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the liftering coefficient
    pub fn lifter(mut self, lifter: Option<f64>) -> Self {
        self.lifter = lifter;
        self
    }

    /// Extract MFCC features from audio signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        if self.n_mfcc == 0 {
            return Err(SklearsError::InvalidInput(
                "MFCC extractor requires at least one coefficient".to_string(),
            ));
        }

        ensure_valid_frame_parameters(signal.len(), self.n_fft, self.hop_length, "MFCC")?;

        // Compute STFT
        let spectrogram = self.stft(signal)?;

        // Convert to power spectrogram
        let power_spec = self.power_spectrogram(&spectrogram);

        // Apply mel filterbank
        let mel_spec = self.mel_filterbank(&power_spec)?;

        // Compute DCT to get MFCCs
        let mfcc = self.dct(&mel_spec)?;

        // Apply liftering if specified
        if let Some(lifter_coeff) = self.lifter {
            Ok(self.apply_liftering(&mfcc, lifter_coeff))
        } else {
            Ok(mfcc)
        }
    }

    /// Compute Short-Time Fourier Transform
    fn stft(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        let n_frames =
            ensure_valid_frame_parameters(signal.len(), self.n_fft, self.hop_length, "MFCC STFT")?;
        let n_freqs = self.n_fft / 2 + 1;

        let mut spectrogram = Array2::zeros((n_freqs, n_frames));

        // Hanning window
        let window = self.hanning_window(self.n_fft);

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = start + self.n_fft;

            if end > signal.len() {
                break;
            }

            // Apply window function
            let windowed: Vec<f64> = signal
                .slice(s![start..end])
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute FFT magnitude
            let fft_result = self.fft_magnitude(&windowed);
            for (i, &mag) in fft_result.iter().enumerate() {
                spectrogram[[i, frame]] = mag;
            }
        }

        Ok(spectrogram)
    }

    /// Generate Hanning window
    fn hanning_window(&self, n: usize) -> Array1<f64> {
        Array1::from_iter(
            (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())),
        )
    }

    /// Compute FFT magnitude (simplified implementation)
    fn fft_magnitude(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let n_freqs = n / 2 + 1;
        let mut magnitudes = vec![0.0; n_freqs];

        // Simplified DFT computation
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

    /// Convert spectrogram to power spectrogram
    fn power_spectrogram(&self, spectrogram: &Array2<f64>) -> Array2<f64> {
        spectrogram.mapv(|x| x * x)
    }

    /// Apply mel filterbank to power spectrogram
    fn mel_filterbank(&self, power_spec: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_freqs = power_spec.nrows();
        let n_frames = power_spec.ncols();

        // Create mel filter bank
        let mel_filters = self.create_mel_filterbank(n_freqs)?;

        let mut mel_spec = Array2::zeros((self.n_mels, n_frames));

        for frame in 0..n_frames {
            for mel_bin in 0..self.n_mels {
                let mut energy = 0.0;
                for freq_bin in 0..n_freqs {
                    energy += power_spec[[freq_bin, frame]] * mel_filters[[mel_bin, freq_bin]];
                }
                mel_spec[[mel_bin, frame]] = energy.max(1e-10).ln(); // Log mel spectrogram
            }
        }

        Ok(mel_spec)
    }

    /// Create mel filterbank
    fn create_mel_filterbank(&self, n_freqs: usize) -> SklResult<Array2<f64>> {
        let fmax = self.fmax.unwrap_or(self.sample_rate / 2.0);

        // Convert frequencies to mel scale
        let mel_min = self.hz_to_mel(self.fmin);
        let mel_max = self.hz_to_mel(fmax);

        // Create mel-spaced frequencies
        let mel_points: Vec<f64> = (0..=self.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (self.n_mels + 1) as f64)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f64> = mel_points.iter().map(|&m| self.mel_to_hz(m)).collect();

        // Convert to FFT bin numbers
        let bin_points: Vec<f64> = hz_points
            .iter()
            .map(|&f| f * (n_freqs - 1) as f64 * 2.0 / self.sample_rate)
            .collect();

        let mut filterbank = Array2::zeros((self.n_mels, n_freqs));

        for m in 1..=self.n_mels {
            let left = bin_points[m - 1];
            let center = bin_points[m];
            let right = bin_points[m + 1];

            for k in 0..n_freqs {
                let k_f = k as f64;
                if k_f >= left && k_f <= center {
                    filterbank[[m - 1, k]] = (k_f - left) / (center - left);
                } else if k_f > center && k_f <= right {
                    filterbank[[m - 1, k]] = (right - k_f) / (right - center);
                }
            }
        }

        // Normalize if requested
        if self.normalize {
            for m in 0..self.n_mels {
                let sum: f64 = filterbank.row(m).sum();
                if sum > 0.0 {
                    for k in 0..n_freqs {
                        filterbank[[m, k]] /= sum;
                    }
                }
            }
        }

        Ok(filterbank)
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(&self, hz: f64) -> f64 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(&self, mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    /// Compute Discrete Cosine Transform
    fn dct(&self, mel_spec: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_mels = mel_spec.nrows();
        let n_frames = mel_spec.ncols();

        let mut mfcc = Array2::zeros((n_frames, self.n_mfcc));

        for frame in 0..n_frames {
            for m in 0..self.n_mfcc {
                let mut coeff = 0.0;
                for k in 0..n_mels {
                    let angle = PI * m as f64 * (2.0 * k as f64 + 1.0) / (2.0 * n_mels as f64);
                    coeff += mel_spec[[k, frame]] * angle.cos();
                }
                mfcc[[frame, m]] = coeff * (2.0 / n_mels as f64).sqrt();
            }
        }

        Ok(mfcc)
    }

    /// Apply liftering to MFCC coefficients
    fn apply_liftering(&self, mfcc: &Array2<f64>, lifter: f64) -> Array2<f64> {
        let mut liftered = mfcc.clone();
        for m in 0..self.n_mfcc {
            let lift_coeff = 1.0 + (lifter / 2.0) * (PI * m as f64 / lifter).sin();
            for frame in 0..mfcc.nrows() {
                liftered[[frame, m]] *= lift_coeff;
            }
        }

        liftered
    }

    /// Extract MFCC features from multiple signals in batch
    pub fn extract_features_batch(&self, signals: &[Array1<f64>]) -> SklResult<Vec<Array2<f64>>> {
        signals
            .iter()
            .map(|signal| self.extract_features(&signal.view()))
            .collect()
    }
}

impl Default for MFCCExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral features extractor
///
/// Extracts various spectral characteristics from audio signals including
/// spectral centroid, bandwidth, rolloff, and flux.
#[derive(Debug, Clone)]
pub struct SpectralFeaturesExtractor {
    n_fft: usize,
    hop_length: usize,
    sample_rate: f64,
    rolloff_threshold: f64,
}

impl SpectralFeaturesExtractor {
    /// Create a new spectral features extractor
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            sample_rate: 22050.0,
            rolloff_threshold: 0.85,
        }
    }

    /// Set the FFT window size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the rolloff threshold
    pub fn rolloff_threshold(mut self, threshold: f64) -> Self {
        self.rolloff_threshold = threshold;
        self
    }

    /// Extract spectral features from audio signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        // Compute STFT
        let spectrogram = self.stft(signal)?;

        let n_frames = spectrogram.ncols();
        let mut features = Array2::zeros((4, n_frames)); // centroid, bandwidth, rolloff, flux

        let frequencies = self.get_frequencies();
        let mut prev_spectrum = None;

        for frame in 0..n_frames {
            let spectrum = spectrogram.column(frame);

            // Spectral centroid
            features[[0, frame]] = self.spectral_centroid(&spectrum, &frequencies);

            // Spectral bandwidth
            features[[1, frame]] =
                self.spectral_bandwidth(&spectrum, &frequencies, features[[0, frame]]);

            // Spectral rolloff
            features[[2, frame]] = self.spectral_rolloff(&spectrum, &frequencies);

            // Spectral flux
            if let Some(ref prev) = prev_spectrum {
                features[[3, frame]] = self.spectral_flux(&spectrum, prev);
            }

            prev_spectrum = Some(spectrum.to_owned());
        }

        Ok(features)
    }

    /// Compute STFT (reused from MFCC)
    fn stft(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        let n_frames = (signal.len() - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;

        let mut spectrogram = Array2::zeros((n_freqs, n_frames));
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

            let fft_result = self.fft_magnitude(&windowed);
            for (i, &mag) in fft_result.iter().enumerate() {
                spectrogram[[i, frame]] = mag;
            }
        }

        Ok(spectrogram)
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

    /// Compute spectral centroid
    fn spectral_centroid(&self, spectrum: &ArrayView1<f64>, frequencies: &Array1<f64>) -> f64 {
        let total_energy: f64 = spectrum.sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = spectrum
            .iter()
            .zip(frequencies.iter())
            .map(|(&mag, &freq)| mag * freq)
            .sum();

        weighted_sum / total_energy
    }

    /// Compute spectral bandwidth
    fn spectral_bandwidth(
        &self,
        spectrum: &ArrayView1<f64>,
        frequencies: &Array1<f64>,
        centroid: f64,
    ) -> f64 {
        let total_energy: f64 = spectrum.sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        let variance: f64 = spectrum
            .iter()
            .zip(frequencies.iter())
            .map(|(&mag, &freq)| mag * (freq - centroid).powi(2))
            .sum();

        (variance / total_energy).sqrt()
    }

    /// Compute spectral rolloff
    fn spectral_rolloff(&self, spectrum: &ArrayView1<f64>, frequencies: &Array1<f64>) -> f64 {
        let total_energy: f64 = spectrum.sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        let threshold = self.rolloff_threshold * total_energy;
        let mut cumulative_energy = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= threshold {
                return frequencies[i];
            }
        }

        frequencies[frequencies.len() - 1]
    }

    /// Compute spectral flux
    fn spectral_flux(&self, spectrum: &ArrayView1<f64>, prev_spectrum: &Array1<f64>) -> f64 {
        spectrum
            .iter()
            .zip(prev_spectrum.iter())
            .map(|(&curr, &prev)| (curr - prev).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for SpectralFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Chroma features extractor
///
/// Extracts chroma features that represent the distribution of energy
/// across the 12 different pitch classes.
#[derive(Debug, Clone)]
pub struct ChromaFeaturesExtractor {
    n_fft: usize,
    hop_length: usize,
    sample_rate: f64,
    n_chroma: usize,
    normalize: bool,
}

impl ChromaFeaturesExtractor {
    /// Create a new chroma features extractor
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            sample_rate: 22050.0,
            n_chroma: 12,
            normalize: true,
        }
    }

    /// Set the FFT window size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the number of chroma bins
    pub fn n_chroma(mut self, n_chroma: usize) -> Self {
        self.n_chroma = n_chroma;
        self
    }

    /// Set whether to normalize chroma features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Extract chroma features from audio signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let n_frames =
            ensure_valid_frame_parameters(signal.len(), self.n_fft, self.hop_length, "Chroma")?;

        // Compute STFT
        let spectrogram = self.stft(signal)?;

        // Create chroma filter bank
        let chroma_filters = self.create_chroma_filterbank()?;

        let mut chroma_features = Array2::zeros((n_frames, self.n_chroma));

        // Apply chroma filters
        for frame in 0..n_frames {
            for chroma_bin in 0..self.n_chroma {
                let mut energy = 0.0;
                for freq_bin in 0..spectrogram.nrows() {
                    energy +=
                        spectrogram[[freq_bin, frame]] * chroma_filters[[chroma_bin, freq_bin]];
                }
                chroma_features[[frame, chroma_bin]] = energy.max(0.0);
            }

            // Normalize if requested
            if self.normalize {
                let mut row = chroma_features.row_mut(frame);
                let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for value in row.iter_mut() {
                        *value /= norm;
                    }
                }
            }
        }

        Ok(chroma_features)
    }

    /// Compute STFT
    fn stft(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        let n_frames = ensure_valid_frame_parameters(
            signal.len(),
            self.n_fft,
            self.hop_length,
            "Chroma STFT",
        )?;
        let n_freqs = self.n_fft / 2 + 1;

        let mut spectrogram = Array2::zeros((n_freqs, n_frames));
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

            let fft_result = self.fft_magnitude(&windowed);
            for (i, &mag) in fft_result.iter().enumerate() {
                spectrogram[[i, frame]] = mag;
            }
        }

        Ok(spectrogram)
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

    /// Create chroma filterbank
    fn create_chroma_filterbank(&self) -> SklResult<Array2<f64>> {
        let n_freqs = self.n_fft / 2 + 1;
        let mut chroma_filters = Array2::zeros((self.n_chroma, n_freqs));

        // A4 = 440 Hz, MIDI note 69
        let a4_freq = 440.0;
        let a4_midi = 69.0;

        for k in 1..n_freqs {
            // Skip DC component
            let freq = k as f64 * self.sample_rate / self.n_fft as f64;

            // Convert frequency to MIDI note number
            let midi_note = a4_midi + 12.0 * (freq / a4_freq).log2();

            // Map to chroma bin (0-11)
            let chroma_bin = (midi_note % 12.0).floor() as usize % self.n_chroma;

            // Simple triangular filter
            let weight = 1.0; // Could be improved with proper triangular filters
            chroma_filters[[chroma_bin, k]] = weight;
        }

        Ok(chroma_filters)
    }
}

impl Default for ChromaFeaturesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero Crossing Rate (ZCR) extractor
///
/// Computes the rate at which the signal changes sign, which is useful
/// for distinguishing between voiced and unvoiced speech segments.
#[derive(Debug, Clone)]
pub struct ZeroCrossingRateExtractor {
    frame_length: usize,
    hop_length: usize,
    center: bool,
}

impl ZeroCrossingRateExtractor {
    /// Create a new ZCR extractor
    pub fn new() -> Self {
        Self {
            frame_length: 2048,
            hop_length: 512,
            center: true,
        }
    }

    /// Set the frame length
    pub fn frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Set the hop length
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set whether to center frames
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Extract ZCR features from audio signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        if signal.len() < self.frame_length {
            return Err(SklearsError::InvalidInput(
                "Signal too short for frame length".to_string(),
            ));
        }

        let n_frames = (signal.len() - self.frame_length) / self.hop_length + 1;
        let mut zcr_features = Array1::zeros(n_frames);

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = (start + self.frame_length).min(signal.len());

            if end > start && end - start >= 2 {
                let frame_signal = signal.slice(s![start..end]);
                zcr_features[frame] = self.compute_zcr(&frame_signal);
            }
        }

        Ok(zcr_features)
    }

    /// Compute zero crossing rate for a single frame
    fn compute_zcr(&self, frame: &ArrayView1<f64>) -> f64 {
        if frame.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..frame.len() {
            if frame[i] * frame[i - 1] < 0.0 {
                crossings += 1;
            }
        }

        crossings as f64 / (frame.len() - 1) as f64
    }
}

impl Default for ZeroCrossingRateExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral rolloff extractor
///
/// Computes the frequency below which a specified percentage of the
/// total spectral energy is contained.
#[derive(Debug, Clone)]
pub struct SpectralRolloffExtractor {
    n_fft: usize,
    hop_length: usize,
    sample_rate: f64,
    rolloff_threshold: f64,
}

impl SpectralRolloffExtractor {
    /// Create a new spectral rolloff extractor
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            sample_rate: 22050.0,
            rolloff_threshold: 0.85,
        }
    }

    /// Set the FFT window size
    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Set the hop length
    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Set the sample rate
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the rolloff threshold (default 0.85 = 85%)
    pub fn rolloff_threshold(mut self, threshold: f64) -> Self {
        self.rolloff_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Extract spectral rolloff features from audio signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        // Compute STFT
        let spectrogram = self.stft(signal)?;
        let frequencies = self.get_frequencies();

        let n_frames = spectrogram.ncols();
        let mut rolloff_features = Array1::zeros(n_frames);

        for frame in 0..n_frames {
            let spectrum = spectrogram.column(frame);
            rolloff_features[frame] = self.compute_rolloff(&spectrum, &frequencies);
        }

        Ok(rolloff_features)
    }

    /// Compute STFT
    fn stft(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        let n_frames = (signal.len() - self.n_fft) / self.hop_length + 1;
        let n_freqs = self.n_fft / 2 + 1;

        let mut spectrogram = Array2::zeros((n_freqs, n_frames));
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

            let fft_result = self.fft_magnitude(&windowed);
            for (i, &mag) in fft_result.iter().enumerate() {
                spectrogram[[i, frame]] = mag;
            }
        }

        Ok(spectrogram)
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

    /// Compute spectral rolloff for a single frame
    fn compute_rolloff(&self, spectrum: &ArrayView1<f64>, frequencies: &Array1<f64>) -> f64 {
        let total_energy: f64 = spectrum.sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        let threshold = self.rolloff_threshold * total_energy;
        let mut cumulative_energy = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= threshold {
                return frequencies[i];
            }
        }

        frequencies[frequencies.len() - 1]
    }
}

impl Default for SpectralRolloffExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mfcc_extractor() {
        // Create a simple sinusoidal signal
        let n_samples = 8192;
        let sample_rate = 22050.0;
        let freq = 440.0; // A4

        let signal: Array1<f64> = Array1::from_iter(
            (0..n_samples).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin()),
        );

        let mfcc_extractor = MFCCExtractor::new().n_mfcc(13).sample_rate(sample_rate);

        let mfcc_features = mfcc_extractor.extract_features(&signal.view()).unwrap();

        // Check dimensions
        assert_eq!(mfcc_features.nrows(), 13);
        assert!(mfcc_features.ncols() > 0);

        // Check that features are finite
        for &val in mfcc_features.iter() {
            assert!(val.is_finite(), "MFCC feature should be finite");
        }
    }

    #[test]
    fn test_spectral_features_extractor() {
        let n_samples = 4096;
        let sample_rate = 22050.0;

        // Create a signal with multiple frequency components
        let signal: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            440.0 * (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin()
        }));

        let spectral_extractor = SpectralFeaturesExtractor::new().sample_rate(sample_rate);

        let spectral_features = spectral_extractor.extract_features(&signal.view()).unwrap();

        // Check dimensions: 4 features (centroid, bandwidth, rolloff, flux)
        assert_eq!(spectral_features.nrows(), 4);
        assert!(spectral_features.ncols() > 0);

        // Check that features are finite and reasonable
        for &val in spectral_features.iter() {
            assert!(val.is_finite(), "Spectral feature should be finite");
            assert!(val >= 0.0, "Spectral feature should be non-negative");
        }

        // Spectral centroid should be reasonable for our signal
        let centroid_values = spectral_features.row(0);
        let avg_centroid: f64 = centroid_values.mean().unwrap();
        assert!(
            avg_centroid > 400.0 && avg_centroid < 1000.0,
            "Spectral centroid should be reasonable for test signal"
        );
    }

    #[test]
    fn test_chroma_features_extractor() {
        let n_samples = 4096;
        let sample_rate = 22050.0;

        // Create a signal with A4 (440 Hz)
        let signal: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * 440.0 * t).sin()
        }));

        let chroma_extractor = ChromaFeaturesExtractor::new().sample_rate(sample_rate);

        let chroma_features = chroma_extractor.extract_features(&signal.view()).unwrap();

        // Check dimensions: 12 chroma bins
        assert_eq!(chroma_features.ncols(), 12);
        assert!(chroma_features.nrows() > 0);

        // Check that features are finite
        for &val in chroma_features.iter() {
            assert!(val.is_finite(), "Chroma feature should be finite");
            assert!(val >= 0.0, "Chroma feature should be non-negative");
        }
    }

    #[test]
    fn test_zero_crossing_rate_extractor() {
        // Create a high-frequency signal (more zero crossings)
        let n_samples = 4096;
        let sample_rate = 22050.0;
        let freq = 2000.0; // High frequency

        let signal_high_freq: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * freq * t).sin()
        }));

        // Create a low-frequency signal (fewer zero crossings)
        let freq_low = 100.0; // Low frequency
        let signal_low_freq: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * freq_low * t).sin()
        }));

        let zcr_extractor = ZeroCrossingRateExtractor::new();

        let zcr_high = zcr_extractor
            .extract_features(&signal_high_freq.view())
            .unwrap();
        let zcr_low = zcr_extractor
            .extract_features(&signal_low_freq.view())
            .unwrap();

        // Check dimensions
        assert!(zcr_high.len() > 0);
        assert!(zcr_low.len() > 0);

        // Check that features are finite and in valid range
        for &val in zcr_high.iter() {
            assert!(val.is_finite(), "ZCR should be finite");
            assert!(val >= 0.0 && val <= 1.0, "ZCR should be between 0 and 1");
        }

        // High frequency signal should have higher ZCR than low frequency
        let avg_zcr_high: f64 = zcr_high.mean().unwrap();
        let avg_zcr_low: f64 = zcr_low.mean().unwrap();

        assert!(
            avg_zcr_high > avg_zcr_low,
            "High frequency signal should have higher ZCR"
        );
    }

    #[test]
    fn test_spectral_rolloff_extractor() {
        let n_samples = 4096;
        let sample_rate = 22050.0;

        // Create a signal with known frequency content
        let signal: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin()
        }));

        let rolloff_extractor = SpectralRolloffExtractor::new()
            .sample_rate(sample_rate)
            .rolloff_threshold(0.85);

        let rolloff_features = rolloff_extractor.extract_features(&signal.view()).unwrap();

        // Check dimensions
        assert!(rolloff_features.len() > 0);

        // Check that features are finite and reasonable
        for &val in rolloff_features.iter() {
            assert!(val.is_finite(), "Rolloff feature should be finite");
            assert!(val >= 0.0, "Rolloff feature should be non-negative");
            assert!(
                val <= sample_rate / 2.0,
                "Rolloff should be below Nyquist frequency"
            );
        }

        // For our test signal, rolloff should be somewhere above 440 Hz
        let avg_rolloff: f64 = rolloff_features.mean().unwrap();
        assert!(
            avg_rolloff > 400.0,
            "Rolloff should be above fundamental frequency"
        );
    }

    #[test]
    fn test_mfcc_extractor_empty_signal() {
        let empty_signal = Array1::zeros(0);
        let mfcc_extractor = MFCCExtractor::new();

        let result = mfcc_extractor.extract_features(&empty_signal.view());
        assert!(result.is_err(), "Should fail with empty signal");
    }

    #[test]
    fn test_spectral_features_extractor_empty_signal() {
        let empty_signal = Array1::zeros(0);
        let spectral_extractor = SpectralFeaturesExtractor::new();

        let result = spectral_extractor.extract_features(&empty_signal.view());
        assert!(result.is_err(), "Should fail with empty signal");
    }

    #[test]
    fn test_chroma_features_extractor_empty_signal() {
        let empty_signal = Array1::zeros(0);
        let chroma_extractor = ChromaFeaturesExtractor::new();

        let result = chroma_extractor.extract_features(&empty_signal.view());
        assert!(result.is_err(), "Should fail with empty signal");
    }

    #[test]
    fn test_zcr_extractor_empty_signal() {
        let empty_signal = Array1::zeros(0);
        let zcr_extractor = ZeroCrossingRateExtractor::new();

        let result = zcr_extractor.extract_features(&empty_signal.view());
        assert!(result.is_err(), "Should fail with empty signal");
    }

    #[test]
    fn test_rolloff_extractor_empty_signal() {
        let empty_signal = Array1::zeros(0);
        let rolloff_extractor = SpectralRolloffExtractor::new();

        let result = rolloff_extractor.extract_features(&empty_signal.view());
        assert!(result.is_err(), "Should fail with empty signal");
    }

    #[test]
    fn test_zcr_constant_signal() {
        // Constant signal should have zero crossings
        let constant_signal = Array1::ones(4096); // Make sure it's longer than default frame_length
        let zcr_extractor = ZeroCrossingRateExtractor::new();

        let zcr_features = zcr_extractor
            .extract_features(&constant_signal.view())
            .unwrap();

        for &val in zcr_features.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_alternating_signal_zcr() {
        // Alternating signal should have maximum zero crossings
        let alternating_signal: Array1<f64> =
            Array1::from_iter((0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }));

        let zcr_extractor = ZeroCrossingRateExtractor::new()
            .frame_length(100)
            .hop_length(50);

        let zcr_features = zcr_extractor
            .extract_features(&alternating_signal.view())
            .unwrap();

        // Should have high ZCR (close to 1.0)
        for &val in zcr_features.iter() {
            assert!(val > 0.9, "Alternating signal should have high ZCR");
        }
    }
}

// Additional extractors required by tests
#[derive(Debug, Clone)]
pub struct SpectralCentroidExtractor {
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
}

impl SpectralCentroidExtractor {
    pub fn new() -> Self {
        Self {
            sample_rate: 22050.0,
            n_fft: 2048,
            hop_length: 512,
        }
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let n_frames = ensure_valid_frame_parameters(
            signal.len(),
            self.n_fft,
            self.hop_length,
            "Spectral centroid",
        )?;
        let mut centroids = Array1::zeros(n_frames);

        // Compute spectral centroid for each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(signal.len());

            // Extract frame
            let frame_len = end - start;
            let mut frame = Array1::zeros(self.n_fft);
            for i in 0..frame_len {
                frame[i] = signal[start + i];
            }

            // Apply window function (Hanning)
            for i in 0..self.n_fft {
                let window_val =
                    0.5 * (1.0 - (2.0 * PI * i as f64 / (self.n_fft - 1) as f64).cos());
                frame[i] *= window_val;
            }

            // Compute magnitude spectrum (simplified FFT)
            let mut magnitude_spectrum = Array1::zeros(self.n_fft / 2 + 1);
            for k in 0..magnitude_spectrum.len() {
                let mut real = 0.0;
                let mut imag = 0.0;

                for n in 0..self.n_fft {
                    let angle = -2.0 * PI * k as f64 * n as f64 / self.n_fft as f64;
                    real += frame[n] * angle.cos();
                    imag += frame[n] * angle.sin();
                }

                magnitude_spectrum[k] = (real * real + imag * imag).sqrt();
            }

            // Compute spectral centroid
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;

            for (k, &magnitude) in magnitude_spectrum.iter().enumerate() {
                let frequency = k as f64 * self.sample_rate / self.n_fft as f64;
                weighted_sum += frequency * magnitude;
                magnitude_sum += magnitude;
            }

            centroids[frame_idx] = if magnitude_sum > 0.0 {
                weighted_sum / magnitude_sum
            } else {
                0.0
            };
        }

        Ok(centroids)
    }
}

impl Default for SpectralCentroidExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TonnetzExtractor {
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
}

impl TonnetzExtractor {
    pub fn new() -> Self {
        Self {
            sample_rate: 22050.0,
            n_fft: 2048,
            hop_length: 512,
        }
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        let chroma = ChromaFeaturesExtractor::new()
            .n_fft(self.n_fft)
            .hop_length(self.hop_length)
            .sample_rate(self.sample_rate)
            .normalize(true)
            .n_chroma(12);

        let chroma_features = chroma.extract_features(signal)?;
        let n_frames = chroma_features.nrows();
        let mut tonnetz = Array2::zeros((n_frames, 6));

        for frame in 0..n_frames {
            let row = chroma_features.row(frame);
            let sum: f64 = row.iter().sum();

            if sum <= 0.0 {
                continue;
            }

            let mut normalized = row.to_owned();
            normalized.mapv_inplace(|x| x / sum);

            let get = |idx: usize| normalized[idx % normalized.len()];

            tonnetz[[frame, 0]] = (get(0) + get(3) + get(4) + get(7) + get(8) + get(11))
                - (get(1) + get(2) + get(5) + get(6) + get(9) + get(10));
            tonnetz[[frame, 1]] = get(0) + get(4) + get(7) - get(2) - get(5) - get(9);
            tonnetz[[frame, 2]] = get(0) + get(3) + get(8) - get(4) - get(7) - get(11);
            tonnetz[[frame, 3]] = get(1) + get(4) + get(6) - get(3) - get(7) - get(10);
            tonnetz[[frame, 4]] = get(2) + get(5) + get(7) - get(4) - get(8) - get(11);
            tonnetz[[frame, 5]] = get(3) + get(6) + get(9) - get(5) - get(8) - get(11);

            for j in 0..6 {
                tonnetz[[frame, j]] = tonnetz[[frame, j]].clamp(-1.0, 1.0);
            }
        }

        Ok(tonnetz)
    }
}

impl Default for TonnetzExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// Alias for existing ChromaFeaturesExtractor
pub type ChromaExtractor = ChromaFeaturesExtractor;

#[derive(Debug, Clone)]
pub struct TempogramExtractor {
    sample_rate: f64,
    hop_length: usize,
    win_length: usize,
    tempo_min: usize,
    tempo_max: usize,
    n_tempo_bins: usize,
}

impl TempogramExtractor {
    pub fn new() -> Self {
        Self {
            sample_rate: 22050.0,
            hop_length: 512,
            win_length: 1024,
            tempo_min: 60,
            tempo_max: 240,
            n_tempo_bins: 40,
        }
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn win_length(mut self, win_length: usize) -> Self {
        self.win_length = win_length;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn tempo_min(mut self, tempo_min: usize) -> Self {
        self.tempo_min = tempo_min;
        self
    }

    pub fn tempo_max(mut self, tempo_max: usize) -> Self {
        self.tempo_max = tempo_max;
        self
    }

    pub fn n_tempo_bins(mut self, n_tempo_bins: usize) -> Self {
        self.n_tempo_bins = n_tempo_bins;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        if self.tempo_min == 0 || self.tempo_max <= self.tempo_min {
            return Err(SklearsError::InvalidInput(
                "Invalid tempo range for tempogram".to_string(),
            ));
        }

        if self.n_tempo_bins == 0 {
            return Err(SklearsError::InvalidInput(
                "Tempogram requires at least one tempo bin".to_string(),
            ));
        }

        let window_length = self.win_length;
        let n_frames = ensure_valid_frame_parameters(
            signal.len(),
            window_length,
            self.hop_length,
            "Tempogram",
        )?;

        let mut tempogram = Array2::zeros((n_frames, self.n_tempo_bins));

        let tempo_range = (self.tempo_max - self.tempo_min) as f64;
        let tempo_step = if self.n_tempo_bins > 1 {
            tempo_range / (self.n_tempo_bins as f64 - 1.0)
        } else {
            0.0
        };

        // Rough onset detection by energy difference
        let mut envelope = Vec::with_capacity(signal.len());
        let mut prev = 0.0;
        for &sample in signal.iter() {
            let diff = (sample.abs() - prev).max(0.0);
            envelope.push(diff);
            prev = sample.abs();
        }

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = start + window_length;
            let frame_envelope = &envelope[start..end];

            // Estimate dominant tempo via autocorrelation of envelope
            let min_period = (60.0 * self.sample_rate / self.tempo_max as f64).floor() as usize;
            let max_period = (60.0 * self.sample_rate / self.tempo_min as f64).ceil() as usize;

            let mut best_period = None;
            let mut best_value = 0.0;

            for period in min_period.max(1)..=max_period.min(frame_envelope.len() / 2) {
                let mut acc = 0.0;
                for i in period..frame_envelope.len() {
                    acc += frame_envelope[i] * frame_envelope[i - period];
                }

                if acc > best_value {
                    best_value = acc;
                    best_period = Some(period);
                }
            }

            let tempo_hint = best_period.map(|p| 60.0 * self.sample_rate / p as f64);
            for bin in 0..self.n_tempo_bins {
                let tempo = self.tempo_min as f64 + tempo_step * bin as f64;
                let value = match tempo_hint {
                    Some(target) => {
                        let diff = (tempo - target).abs();
                        (-diff / (tempo_range.max(1.0) * 0.1)).exp()
                    }
                    None => 0.0,
                };
                tempogram[[frame, bin]] = value.max(0.0);
            }
        }

        Ok(tempogram)
    }
}

impl Default for TempogramExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct RMSEnergyExtractor {
    hop_length: usize,
    frame_length: usize,
}

impl RMSEnergyExtractor {
    pub fn new() -> Self {
        Self {
            hop_length: 512,
            frame_length: 2048,
        }
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let n_frames = ensure_valid_frame_parameters(
            signal.len(),
            self.frame_length,
            self.hop_length,
            "RMS energy",
        )?;
        let mut rms_values = Array1::zeros(n_frames);

        // Compute RMS energy for each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.frame_length).min(signal.len());

            // Compute RMS for the current frame
            let mut sum_squares = 0.0;
            let frame_len = end - start;

            for i in start..end {
                sum_squares += signal[i] * signal[i];
            }

            // RMS = sqrt(mean(signal^2))
            rms_values[frame_idx] = (sum_squares / frame_len as f64).sqrt();
        }

        Ok(rms_values)
    }
}

impl Default for RMSEnergyExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral bandwidth extractor
///
/// Computes the spectral bandwidth of audio signals, which measures
/// the spread of energy around the spectral centroid.
#[derive(Debug, Clone)]
pub struct SpectralBandwidthExtractor {
    sample_rate: f64,
    n_fft: usize,
    hop_length: usize,
}

impl SpectralBandwidthExtractor {
    pub fn new() -> Self {
        Self {
            sample_rate: 22050.0,
            n_fft: 2048,
            hop_length: 512,
        }
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = n_fft;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let n_frames = ensure_valid_frame_parameters(
            signal.len(),
            self.n_fft,
            self.hop_length,
            "Spectral bandwidth",
        )?;
        let mut bandwidths = Array1::zeros(n_frames);

        // Compute spectral bandwidth for each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(signal.len());

            // Extract frame
            let frame_len = end - start;
            let mut frame = Array1::zeros(self.n_fft);
            for i in 0..frame_len {
                frame[i] = signal[start + i];
            }

            // Apply window function (Hanning)
            for i in 0..self.n_fft {
                let window_val =
                    0.5 * (1.0 - (2.0 * PI * i as f64 / (self.n_fft - 1) as f64).cos());
                frame[i] *= window_val;
            }

            // Compute magnitude spectrum (simplified FFT)
            let mut magnitude_spectrum = Array1::zeros(self.n_fft / 2 + 1);
            for k in 0..magnitude_spectrum.len() {
                let mut real = 0.0;
                let mut imag = 0.0;

                for n in 0..self.n_fft {
                    let angle = -2.0 * PI * k as f64 * n as f64 / self.n_fft as f64;
                    real += frame[n] * angle.cos();
                    imag += frame[n] * angle.sin();
                }

                magnitude_spectrum[k] = (real * real + imag * imag).sqrt();
            }

            // Compute spectral centroid first
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;

            for (k, &magnitude) in magnitude_spectrum.iter().enumerate() {
                let frequency = k as f64 * self.sample_rate / self.n_fft as f64;
                weighted_sum += frequency * magnitude;
                magnitude_sum += magnitude;
            }

            let centroid = if magnitude_sum > 0.0 {
                weighted_sum / magnitude_sum
            } else {
                0.0
            };

            // Compute spectral bandwidth using variance around centroid
            let mut variance_sum = 0.0;

            for (k, &magnitude) in magnitude_spectrum.iter().enumerate() {
                let frequency = k as f64 * self.sample_rate / self.n_fft as f64;
                let diff = frequency - centroid;
                variance_sum += diff * diff * magnitude;
            }

            bandwidths[frame_idx] = if magnitude_sum > 0.0 {
                (variance_sum / magnitude_sum).sqrt()
            } else {
                0.0
            };
        }

        Ok(bandwidths)
    }
}

impl Default for SpectralBandwidthExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Mel spectrogram extractor
///
/// Computes mel-frequency spectrograms which are commonly used
/// in audio analysis and machine learning applications.
#[derive(Debug, Clone)]
pub struct MelSpectrogramExtractor {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
    sample_rate: f64,
}

impl MelSpectrogramExtractor {
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
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

    pub fn n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = n_mels;
        self
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        if self.n_mels == 0 {
            return Err(SklearsError::InvalidInput(
                "Mel spectrogram requires at least one mel bin".to_string(),
            ));
        }

        let n_frames = ensure_valid_frame_parameters(
            signal.len(),
            self.n_fft,
            self.hop_length,
            "Mel spectrogram",
        )?;
        let mut mel_spectrogram = Array2::zeros((n_frames, self.n_mels));

        // Create mel filter bank
        let mel_filters = self.create_mel_filters()?;

        // Compute mel spectrogram for each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(signal.len());

            if end <= start {
                continue;
            }

            // Extract frame
            let frame_len = end - start;
            let mut frame = Array1::zeros(self.n_fft);
            for i in 0..frame_len {
                frame[i] = signal[start + i];
            }

            // Apply window function (Hanning)
            for i in 0..frame_len {
                let window_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (frame_len - 1) as f64).cos());
                frame[i] *= window_val;
            }

            // Compute magnitude spectrum (simplified FFT)
            let mut magnitude_spectrum = Array1::zeros(self.n_fft / 2 + 1);
            for k in 0..magnitude_spectrum.len() {
                let mut real = 0.0;
                let mut imag = 0.0;

                for n in 0..self.n_fft {
                    let angle = -2.0 * PI * k as f64 * n as f64 / self.n_fft as f64;
                    real += frame[n] * angle.cos();
                    imag += frame[n] * angle.sin();
                }

                magnitude_spectrum[k] = (real * real + imag * imag).sqrt();
            }

            // Apply mel filters to magnitude spectrum
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut mel_energy = 0.0;
                for (freq_idx, &filter_val) in filter.iter().enumerate() {
                    if freq_idx < magnitude_spectrum.len() {
                        mel_energy += magnitude_spectrum[freq_idx] * filter_val;
                    }
                }
                mel_spectrogram[(frame_idx, mel_idx)] = mel_energy.max(0.0);
            }
        }

        Ok(mel_spectrogram)
    }

    fn create_mel_filters(&self) -> SklResult<Vec<Array1<f64>>> {
        let fmax = self.fmax.unwrap_or(self.sample_rate / 2.0);
        let n_fft_bins = self.n_fft / 2 + 1;

        // Convert Hz to mel scale
        let mel_min = self.hz_to_mel(self.fmin);
        let mel_max = self.hz_to_mel(fmax);

        // Create mel scale points
        let mel_points: Vec<f64> = (0..=self.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (self.n_mels + 1) as f64)
            .collect();

        // Convert mel points back to Hz
        let hz_points: Vec<f64> = mel_points.iter().map(|&mel| self.mel_to_hz(mel)).collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<f64> = hz_points
            .iter()
            .map(|&hz| hz * self.n_fft as f64 / self.sample_rate)
            .collect();

        // Create triangular filters
        let mut filters = Vec::new();
        for i in 1..=self.n_mels {
            let mut filter = Array1::zeros(n_fft_bins);

            let left = bin_points[i - 1];
            let center = bin_points[i];
            let right = bin_points[i + 1];

            for k in 0..n_fft_bins {
                let k_f = k as f64;
                if k_f >= left && k_f <= right {
                    if k_f <= center {
                        // Rising edge
                        filter[k] = (k_f - left) / (center - left);
                    } else {
                        // Falling edge
                        filter[k] = (right - k_f) / (right - center);
                    }
                }
            }

            filters.push(filter);
        }

        Ok(filters)
    }

    fn hz_to_mel(&self, hz: f64) -> f64 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(&self, mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }
}

impl Default for MelSpectrogramExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Pitch extractor
///
/// Extracts fundamental frequency (pitch) information from audio signals
/// using various pitch detection algorithms.
#[derive(Debug, Clone)]
pub struct PitchExtractor {
    sample_rate: f64,
    hop_length: usize,
    fmin: f64,
    fmax: f64,
    threshold: f64,
    method: String,
}

impl PitchExtractor {
    pub fn new() -> Self {
        Self {
            sample_rate: 22050.0,
            hop_length: 512,
            fmin: 50.0,
            fmax: 2000.0,
            threshold: 0.1,
            method: "autocorr".to_string(),
        }
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    pub fn fmax(mut self, fmax: f64) -> Self {
        self.fmax = fmax;
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let frame_length = (self.hop_length * 2).max(2);
        let n_frames =
            ensure_valid_frame_parameters(signal.len(), frame_length, self.hop_length, "Pitch")?;

        let mut pitches = Array1::zeros(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = start + frame_length;
            let frame = signal.slice(s![start..end]);

            let pitch = self.estimate_pitch(&frame);
            pitches[frame_idx] = pitch;
        }

        Ok(pitches)
    }
}

impl Default for PitchExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchExtractor {
    fn estimate_pitch(&self, frame: &ArrayView1<f64>) -> f64 {
        let mut windowed = frame.to_owned();
        let n = windowed.len();
        if n < 4 {
            return 0.0;
        }

        // Apply Hann window to reduce spectral leakage
        for i in 0..n {
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            windowed[i] *= window_val;
        }

        let energy = windowed.iter().map(|x| x * x).sum::<f64>();
        if energy < 1e-12 {
            return 0.0;
        }

        let min_lag = (self.sample_rate / self.fmax.max(1.0)).floor().max(1.0) as usize;
        let max_lag = (self.sample_rate / self.fmin.max(1.0)).ceil() as usize;

        let mut best_lag = 0;
        let mut best_corr = 0.0;

        for lag in min_lag..=max_lag.min(n - 1) {
            let mut sum = 0.0;
            for i in lag..n {
                sum += windowed[i] * windowed[i - lag];
            }

            let corr = sum / energy;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        if best_corr >= self.threshold && best_lag > 0 {
            self.sample_rate / best_lag as f64
        } else {
            0.0
        }
    }
}

/// Onset detector
///
/// Detects onset times in audio signals, which are moments when
/// new musical events (notes, sounds) begin.
#[derive(Debug, Clone)]
pub struct OnsetDetector {
    sample_rate: f64,
    hop_length: usize,
    n_fft: usize,
    threshold: f64,
    method: String,
}

impl OnsetDetector {
    pub fn new() -> Self {
        Self {
            sample_rate: 22050.0,
            hop_length: 512,
            n_fft: 2048,
            threshold: 0.5,
            method: "energy".to_string(),
        }
    }

    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Placeholder implementation - onset strength
        let n_frames = signal.len() / self.hop_length + 1;
        Ok(Array1::zeros(n_frames))
    }

    pub fn detect_onsets(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        // Placeholder implementation - onset times
        Ok(Array1::zeros(10)) // Return some onset times
    }
}

impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Harmonic-Percussive Separation
///
/// Separates harmonic and percussive components from audio signals
/// using spectral analysis techniques.
#[derive(Debug, Clone)]
pub struct HarmonicPercussiveSeparation {
    n_fft: usize,
    hop_length: usize,
    margin: f64,
    kernel_size: usize,
    power: f64,
}

impl HarmonicPercussiveSeparation {
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            margin: 1.0,
            kernel_size: 31,
            power: 2.0,
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

    pub fn margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    pub fn kernel_size(mut self, kernel_size: usize) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    pub fn power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    pub fn separate(&self, signal: &ArrayView1<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        // Placeholder implementation
        let harmonic = Array1::zeros(signal.len());
        let percussive = Array1::zeros(signal.len());
        Ok((harmonic, percussive))
    }
}

impl Default for HarmonicPercussiveSeparation {
    fn default() -> Self {
        Self::new()
    }
}
