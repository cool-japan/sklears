//! Comprehensive signal processing and feature extraction module
//!
//! This module provides a complete suite of signal processing tools for feature extraction,
//! systematically organized into focused, maintainable modules following the 2000-line policy.
//!
//! # Architecture
//!
//! The signal processing functionality is organized into specialized modules:
//!
//! ## Core Analysis Modules
//!
//! ### Frequency Domain Analysis
//! - **frequency_analysis**: Power spectral density, filter banks, STFT
//! - **phase_analysis**: Phase-based features, Hilbert transform, instantaneous frequency
//! - **transform_methods**: Wavelet transforms with multiple wavelet types
//!
//! ### Time Domain Analysis
//! - **time_series_analysis**: Autoregressive modeling, cross-correlation analysis
//! - **signal_utilities**: Resampling, envelope detection, signal conditioning
//!
//! ### Signal Conditioning
//! - **filtering**: Bandpass, lowpass, highpass, and notch filtering
//!
//! ### Comprehensive Extraction
//! - **composite_extractors**: Multi-domain feature extraction and utility functions
//!
//! # Usage Examples
//!
//! ## Basic Frequency Analysis
//! ```rust
//! use sklears_feature_extraction::signal_processing::FrequencyDomainExtractor;
//! use scirs2_core::ndarray::Array1;
//!
//! let signal = Array1::from_vec(vec![1.0, 2.0, 1.0, -1.0, -2.0, -1.0]);
//! let extractor = FrequencyDomainExtractor::new()
//!     .n_fft(512)
//!     .sample_rate(250.0);
//!
//! let features = extractor.extract_features(&signal.view()).unwrap();
//! ```
//!
//! ## Time Series Analysis
//! ```rust
//! use sklears_feature_extraction::signal_processing::AutoregressiveExtractor;
//! use scirs2_core::ndarray::Array1;
//!
//! let signal = Array1::from_vec(vec![1.0, 1.5, 2.0, 1.8, 1.2, 0.8, 1.0, 1.3]);
//! let extractor = AutoregressiveExtractor::new()
//!     .order(3)
//!     .method("yule_walker".to_string());
//!
//! let ar_coeffs = extractor.extract_features(&signal.view()).unwrap();
//! ```
//!
//! ## Comprehensive Feature Extraction
//! ```rust
//! use sklears_feature_extraction::signal_processing::SignalFeatureExtractor;
//! use scirs2_core::ndarray::Array1;
//!
//! let signal = Array1::from_vec((0..1000).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>());
//! let extractor = SignalFeatureExtractor::new()
//!     .include_statistical(true)
//!     .include_spectral(true)
//!     .include_temporal(true);
//!
//! let features = extractor.extract_features(&signal.view()).unwrap();
//! ```
//!
//! # Feature Categories
//!
//! ## Statistical Features
//! - Mean, standard deviation, min/max values
//! - Range, skewness, kurtosis
//! - Distribution characteristics
//!
//! ## Spectral Features
//! - Power spectral density analysis
//! - Frequency band powers (delta, theta, alpha, beta, gamma)
//! - Spectral statistics (mean, std, skewness, kurtosis)
//! - Peak frequency, spectral edge frequency
//!
//! ## Temporal Features
//! - Zero crossing rate
//! - RMS energy
//! - Peak-to-peak amplitude
//! - Envelope characteristics
//!
//! ## Phase Features
//! - Instantaneous frequency
//! - Phase derivatives and variance
//! - Phase coherence and entropy
//! - Hilbert transform analysis
//!
//! ## Transform Features
//! - Wavelet decomposition energies
//! - Multi-resolution analysis
//! - Time-frequency representations
//!
//! # Performance Considerations
//!
//! - All modules use efficient SciRS2 array operations
//! - FFT operations are optimized for common sizes (powers of 2)
//! - Memory usage is optimized through careful array management
//! - Parallel processing capabilities where applicable

/// Frequency domain analysis and feature extraction
///
/// Provides tools for spectral analysis including power spectral density,
/// filter bank analysis, and short-time Fourier transform (STFT).
pub mod frequency_analysis;

/// Time series analysis and autoregressive modeling
///
/// Implements autoregressive model fitting, cross-correlation analysis,
/// and temporal pattern recognition methods.
pub mod time_series_analysis;

/// Phase-based signal analysis and feature extraction
///
/// Advanced phase analysis tools including instantaneous frequency calculation,
/// phase derivatives, coherence measures, and Hilbert transform methods.
pub mod phase_analysis;

/// Signal transform methods and wavelet analysis
///
/// Comprehensive wavelet transform implementations with multiple wavelet types,
/// multi-level decomposition, and energy-based feature extraction.
pub mod transform_methods;

/// Signal filtering and preprocessing methods
///
/// Complete filtering toolkit including bandpass, lowpass, highpass, and notch filters
/// with configurable parameters and feature extraction from filtered signals.
pub mod filtering;

/// Signal processing utilities and preprocessing tools
///
/// Essential utilities including signal resampling, envelope detection,
/// and signal conditioning operations.
pub mod signal_utilities;

/// Composite extractors and utility functions
///
/// High-level feature extraction combining multiple methods, utility functions
/// for windowing and correlation analysis, and comprehensive feature pipelines.
pub mod composite_extractors;

// Re-export core structures for convenient access

/// Frequency domain analysis exports
pub use frequency_analysis::{FilterBankExtractor, FrequencyDomainExtractor, STFT};

/// Time series analysis exports
pub use time_series_analysis::{AutoregressiveExtractor, CrossCorrelationExtractor};

/// Phase analysis exports
pub use phase_analysis::{HilbertTransform, PhaseBasedExtractor};

/// Transform methods exports
pub use transform_methods::WaveletTransform;

/// Filtering exports
pub use filtering::{
    BandpassFilter, FilterBank, FilterType, HighpassFilter, LowpassFilter, NotchFilter,
};

/// Signal utilities exports
pub use signal_utilities::{EnvelopeDetector, Resampler};

/// Composite extractors exports
pub use composite_extractors::{
    apply_window, auto_correlate, convolve, convolve_mode, cross_correlate, generate_window,
    normalized_cross_correlate, SignalFeatureExtractor,
};

/// Comprehensive signal feature extractor (primary interface)
///
/// This is the main high-level interface for signal feature extraction,
/// combining statistical, spectral, and temporal analysis methods.
pub type SignalProcessor = SignalFeatureExtractor;

/// Frequency domain processor (alias for convenience)
pub type FrequencyProcessor = FrequencyDomainExtractor;

/// Time series processor (alias for convenience)
pub type TimeSeriesProcessor = AutoregressiveExtractor;

/// Phase processor (alias for convenience)
pub type PhaseProcessor = PhaseBasedExtractor;

/// Wavelet processor (alias for convenience)
pub type WaveletProcessor = WaveletTransform;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_frequency_domain_extractor() {
        let signal = Array1::from_iter(
            (0..512).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 250.0).sin()),
        );

        let extractor = FrequencyDomainExtractor::new()
            .n_fft(256)
            .sample_rate(250.0);

        let features = extractor.extract_features(&signal.view()).unwrap();
        assert!(features.len() > 0);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_autoregressive_extractor() {
        let signal = Array1::from_iter((0..100).map(|i| i as f64 + 0.1 * (i as f64).sin()));

        let extractor = AutoregressiveExtractor::new()
            .order(5)
            .method("yule_walker".to_string());

        let features = extractor.extract_features(&signal.view()).unwrap();
        assert_eq!(features.len(), 5);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_phase_based_extractor() {
        let signal = Array1::from_iter(
            (0..512).map(|i| (2.0 * std::f64::consts::PI * 15.0 * i as f64 / 250.0).sin()),
        );

        let extractor = PhaseBasedExtractor::new().n_fft(256).hop_length(128);

        let features = extractor.extract_features(&signal.view()).unwrap();
        assert!(features.len() > 0);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_wavelet_transform() {
        let signal = Array1::from_iter(
            (0..256).map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 100.0).sin()),
        );

        let extractor = WaveletTransform::new()
            .wavelet("haar".to_string())
            .levels(4);

        let features = extractor.extract_features(&signal.view()).unwrap();
        assert!(features.len() <= 5); // 4 detail + 1 approximation
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_bandpass_filter() {
        let signal = Array1::from_iter(
            (0..500).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 250.0).sin()),
        );

        let filter = BandpassFilter::new()
            .frequency_range(8.0, 12.0)
            .sample_rate(250.0);

        let features = filter.extract_features(&signal.view()).unwrap();
        assert_eq!(features.len(), 3); // RMS, peak, ZCR
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_signal_feature_extractor() {
        let signal = Array1::from_iter(
            (0..512).map(|i| (2.0 * std::f64::consts::PI * 20.0 * i as f64 / 250.0).sin()),
        );

        let extractor = SignalFeatureExtractor::new()
            .include_statistical(true)
            .include_spectral(true)
            .include_temporal(true)
            .sample_rate(250.0);

        let features = extractor.extract_features(&signal.view()).unwrap();
        let expected_count = extractor.feature_count();
        assert_eq!(features.len(), expected_count);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_cross_correlation() {
        let signal1 = Array1::from_iter((0..100).map(|i| (i as f64 * 0.1).sin()));
        let signal2 = Array1::from_iter((0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()));

        let correlation = cross_correlate(&signal1.view(), &signal2.view()).unwrap();
        assert_eq!(correlation.len(), signal1.len() + signal2.len() - 1);
        assert!(correlation.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_apply_window() {
        let signal = Array1::ones(128);

        let windowed = apply_window(&signal.view(), "hanning").unwrap();
        assert_eq!(windowed.len(), signal.len());
        assert!(windowed[0] < windowed[64]); // Window should taper at edges
        assert!(windowed.iter().all(|&x| x.is_finite()));
    }
}
