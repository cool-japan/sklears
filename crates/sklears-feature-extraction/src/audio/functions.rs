//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
pub(crate) fn ensure_valid_frame_parameters(
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
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;
    use std::f64::consts::PI;
    #[test]
    fn test_mfcc_extractor() {
        let n_samples = 8192;
        let sample_rate = 22050.0;
        let freq = 440.0;
        let signal: Array1<f64> = Array1::from_iter(
            (0..n_samples).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin()),
        );
        let mfcc_extractor = MFCCExtractor::new().n_mfcc(13).sample_rate(sample_rate);
        let mfcc_features = mfcc_extractor.extract_features(&signal.view()).unwrap();
        assert_eq!(mfcc_features.nrows(), 13);
        assert!(mfcc_features.ncols() > 0);
        for &val in mfcc_features.iter() {
            assert!(val.is_finite(), "MFCC feature should be finite");
        }
    }
    #[test]
    fn test_spectral_features_extractor() {
        let n_samples = 4096;
        let sample_rate = 22050.0;
        let signal: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            440.0 * (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin()
        }));
        let spectral_extractor = SpectralFeaturesExtractor::new().sample_rate(sample_rate);
        let spectral_features = spectral_extractor.extract_features(&signal.view()).unwrap();
        assert_eq!(spectral_features.nrows(), 4);
        assert!(spectral_features.ncols() > 0);
        for &val in spectral_features.iter() {
            assert!(val.is_finite(), "Spectral feature should be finite");
            assert!(val >= 0.0, "Spectral feature should be non-negative");
        }
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
        let signal: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * 440.0 * t).sin()
        }));
        let chroma_extractor = ChromaFeaturesExtractor::new().sample_rate(sample_rate);
        let chroma_features = chroma_extractor.extract_features(&signal.view()).unwrap();
        assert_eq!(chroma_features.ncols(), 12);
        assert!(chroma_features.nrows() > 0);
        for &val in chroma_features.iter() {
            assert!(val.is_finite(), "Chroma feature should be finite");
            assert!(val >= 0.0, "Chroma feature should be non-negative");
        }
    }
    #[test]
    fn test_zero_crossing_rate_extractor() {
        let n_samples = 4096;
        let sample_rate = 22050.0;
        let freq = 2000.0;
        let signal_high_freq: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * freq * t).sin()
        }));
        let freq_low = 100.0;
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
        assert!(zcr_high.len() > 0);
        assert!(zcr_low.len() > 0);
        for &val in zcr_high.iter() {
            assert!(val.is_finite(), "ZCR should be finite");
            assert!(val >= 0.0 && val <= 1.0, "ZCR should be between 0 and 1");
        }
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
        let signal: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin()
        }));
        let rolloff_extractor = SpectralRolloffExtractor::new()
            .sample_rate(sample_rate)
            .rolloff_threshold(0.85);
        let rolloff_features = rolloff_extractor.extract_features(&signal.view()).unwrap();
        assert!(rolloff_features.len() > 0);
        for &val in rolloff_features.iter() {
            assert!(val.is_finite(), "Rolloff feature should be finite");
            assert!(val >= 0.0, "Rolloff feature should be non-negative");
            assert!(
                val <= sample_rate / 2.0,
                "Rolloff should be below Nyquist frequency"
            );
        }
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
        let constant_signal = Array1::ones(4096);
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
        let alternating_signal: Array1<f64> =
            Array1::from_iter((0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }));
        let zcr_extractor = ZeroCrossingRateExtractor::new()
            .frame_length(100)
            .hop_length(50);
        let zcr_features = zcr_extractor
            .extract_features(&alternating_signal.view())
            .unwrap();
        for &val in zcr_features.iter() {
            assert!(val > 0.9, "Alternating signal should have high ZCR");
        }
    }
}
pub type ChromaExtractor = ChromaFeaturesExtractor;
