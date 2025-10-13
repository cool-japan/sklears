//! Signal processing tests
//!
//! This module contains tests for signal processing operations including
//! STFT, wavelet transforms, Hilbert transforms, bandpass filters,
//! envelope detection, and general signal feature extraction.

use crate::signal_processing;
use scirs2_core::ndarray::Array1;

#[test]
fn test_signal_stft() {
    // Create a signal with two frequency components
    let sample_rate = 1000;
    let duration = 1.0;
    let n_samples = (sample_rate as f64 * duration) as usize;

    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 150.0 * t).sin()
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let stft = signal_processing::STFT::new()
        .window_size(256)
        .hop_length(128)
        .window_type("hann".to_string());

    let spectrogram = stft.transform(&signal.view()).unwrap();

    // Should have frequency bins and time frames
    let n_freq_bins = 256 / 2 + 1; // For real signals
    let n_time_frames = (signal.len() - 256) / 128 + 1;

    assert_eq!(spectrogram.nrows(), n_time_frames);
    assert_eq!(spectrogram.ncols(), n_freq_bins);

    // All spectrogram values should be non-negative (magnitude)
    for &val in spectrogram.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}

#[test]
fn test_signal_wavelet_transform() {
    let signal: Vec<f64> = (0..64)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin())
        .collect();

    let wavelet = signal_processing::WaveletTransform::new()
        .wavelet_type("db4".to_string())
        .levels(3);

    let signal_array = Array1::from_vec(signal.clone());
    let coefficients = wavelet.transform(&signal_array.view()).unwrap();

    // Should have coefficients for each decomposition level
    assert!(coefficients.len() > 0);

    // All coefficients should be finite
    for &coeff in coefficients.iter() {
        assert!(coeff.is_finite());
    }

    // Test inverse transform
    let reconstructed = wavelet.inverse_transform(&coefficients).unwrap();
    assert_eq!(reconstructed.len(), signal.len());

    for &val in reconstructed.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_signal_hilbert_transform() {
    let signal: Vec<f64> = (0..100)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin())
        .collect();

    let hilbert = signal_processing::HilbertTransform::new();
    let signal_arr = Array1::from_vec(signal.clone());

    let analytic_signal = hilbert.transform(&signal_arr.view()).unwrap();

    // Analytic signal should have same length as input
    assert_eq!(analytic_signal.len(), signal.len());

    // All values should be finite
    for &val in analytic_signal.iter() {
        assert!(val.is_finite());
    }

    // Test instantaneous amplitude extraction
    let amplitude = hilbert.instantaneous_amplitude(&signal_arr.view()).unwrap();
    assert_eq!(amplitude.len(), signal.len());

    for &amp in amplitude.iter() {
        assert!(amp.is_finite());
        assert!(amp >= 0.0); // Amplitude should be non-negative
    }

    // Test instantaneous phase extraction
    let phase = hilbert.instantaneous_phase(&signal_arr.view()).unwrap();
    assert_eq!(phase.len(), signal.len());

    for &ph in phase.iter() {
        assert!(ph.is_finite());
    }
}

#[test]
fn test_signal_bandpass_filter() {
    let sample_rate = 1000;
    let signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Signal with components at 50Hz, 150Hz, and 300Hz
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + (2.0 * std::f64::consts::PI * 150.0 * t).sin()
                + (2.0 * std::f64::consts::PI * 300.0 * t).sin()
        })
        .collect();

    let filter = signal_processing::BandpassFilter::new()
        .low_freq(100.0)
        .high_freq(200.0)
        .sample_rate(sample_rate as f64)
        .order(4);

    let signal_array = Array1::from_vec(signal.clone());
    let filtered = filter.apply(&signal_array.view()).unwrap();

    assert_eq!(filtered.len(), signal.len());

    // All filtered values should be finite
    for &val in filtered.iter() {
        assert!(val.is_finite());
    }

    // The 150Hz component should be preserved, while 50Hz and 300Hz should be attenuated
    // This is hard to test precisely without spectral analysis, so we just check basic properties
    let filtered_energy: f64 = filtered.iter().map(|x| x * x).sum();
    assert!(filtered_energy > 0.0); // Some energy should remain
}

#[test]
fn test_signal_envelope_detector() {
    // Create an amplitude-modulated signal
    let signal: Vec<f64> = (0..200)
        .map(|i| {
            let t = i as f64 / 100.0;
            let carrier = (2.0 * std::f64::consts::PI * 10.0 * t).sin();
            let envelope = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * 1.0 * t).sin();
            envelope * carrier
        })
        .collect();

    let detector = signal_processing::EnvelopeDetector::new().method("hilbert".to_string());

    let signal_array = Array1::from_vec(signal.clone());
    let envelope = detector.extract(&signal_array.view()).unwrap();

    assert_eq!(envelope.len(), signal.len());

    // All envelope values should be finite and non-negative
    for &val in envelope.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }

    // The envelope should follow the modulation pattern
    // (specific testing would require more detailed analysis)
}

#[test]
fn test_signal_feature_extractor() {
    let signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / 100.0;
            (2.0 * std::f64::consts::PI * t).sin() + 0.1 * (t * t).sin()
        })
        .collect();

    let extractor = signal_processing::SignalFeatureExtractor::new()
        .include_time_domain(true)
        .include_frequency_domain(true)
        .include_statistical(true);

    let signal_array = Array1::from_vec(signal.clone());
    let features = extractor.extract_features(&signal_array.view()).unwrap();

    // Should have multiple feature types
    assert!(features.len() > 10); // Expect various time/freq/statistical features

    // All features should be finite
    for &feat in features.iter() {
        assert!(feat.is_finite());
    }
}

#[test]
fn test_lowpass_filter() {
    let sample_rate = 1000;
    let signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Mix of low and high frequency components
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + (2.0 * std::f64::consts::PI * 300.0 * t).sin()
        })
        .collect();

    let filter = signal_processing::LowpassFilter::new()
        .cutoff_freq(150.0)
        .sample_rate(sample_rate as f64)
        .order(6);

    let signal_array = Array1::from_vec(signal.clone());
    let filtered = filter.apply(&signal_array.view()).unwrap();

    assert_eq!(filtered.len(), signal.len());

    for &val in filtered.iter() {
        assert!(val.is_finite());
    }

    // Low frequency component should be preserved, high frequency attenuated
    let filtered_energy: f64 = filtered.iter().map(|x| x * x).sum();
    let original_energy: f64 = signal.iter().map(|x| x * x).sum();
    assert!(filtered_energy < original_energy); // Some energy should be removed
    assert!(filtered_energy > 0.0); // But not all energy
}

#[test]
fn test_highpass_filter() {
    let sample_rate = 1000;
    let signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // DC component + high frequency
            1.0 + (2.0 * std::f64::consts::PI * 200.0 * t).sin()
        })
        .collect();

    let filter = signal_processing::HighpassFilter::new()
        .cutoff_freq(100.0)
        .sample_rate(sample_rate as f64)
        .order(4);

    let signal_array = Array1::from_vec(signal.clone());
    let filtered = filter.apply(&signal_array.view()).unwrap();

    assert_eq!(filtered.len(), signal.len());

    for &val in filtered.iter() {
        assert!(val.is_finite());
    }

    // DC component should be removed, high frequency preserved
    let mean_original = signal.iter().sum::<f64>() / signal.len() as f64;
    let mean_filtered = filtered.iter().sum::<f64>() / filtered.len() as f64;

    assert!(mean_original > 0.5); // Original has DC bias
    assert!(mean_filtered.abs() < 0.1); // Filtered should have little DC
}

#[test]
fn test_notch_filter() {
    let sample_rate = 1000;
    let signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Signal with 50Hz noise + 100Hz signal
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + (2.0 * std::f64::consts::PI * 100.0 * t).sin()
        })
        .collect();

    let filter = signal_processing::NotchFilter::new()
        .notch_freq(50.0)
        .bandwidth(10.0)
        .sample_rate(sample_rate as f64);

    let signal_array = Array1::from_vec(signal.clone());
    let filtered = filter.apply(&signal_array.view()).unwrap();

    assert_eq!(filtered.len(), signal.len());

    for &val in filtered.iter() {
        assert!(val.is_finite());
    }

    // 50Hz should be attenuated, 100Hz preserved
    let filtered_energy: f64 = filtered.iter().map(|x| x * x).sum();
    let original_energy: f64 = signal.iter().map(|x| x * x).sum();
    assert!(filtered_energy < original_energy);
    assert!(filtered_energy > 0.1 * original_energy); // 100Hz component should remain
}

#[test]
fn test_signal_resampling() {
    let original_signal: Vec<f64> = (0..100)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
        .collect();

    let resampler = signal_processing::Resampler::new()
        .original_rate(1000.0)
        .target_rate(500.0)
        .method("linear".to_string());

    let original_arr = Array1::from_vec(original_signal.clone());
    let resampled = resampler.resample(&original_arr.view()).unwrap();

    // Should be roughly half the length
    assert!((resampled.len() as f64 - original_signal.len() as f64 / 2.0).abs() < 5.0);

    for &val in resampled.iter() {
        assert!(val.is_finite());
    }

    // Test upsampling
    let upsampler = signal_processing::Resampler::new()
        .original_rate(1000.0)
        .target_rate(2000.0)
        .method("cubic".to_string());

    let upsampled = upsampler.resample(&original_arr.view()).unwrap();

    // Should be roughly double the length
    assert!((upsampled.len() as f64 - original_signal.len() as f64 * 2.0).abs() < 10.0);

    for &val in upsampled.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_signal_windowing() {
    let signal_vec: Vec<f64> = (0..256).map(|i| i as f64).collect();
    let signal = Array1::from_vec(signal_vec);

    // Test different window types
    let windows = vec!["hann", "hamming", "blackman", "rectangular"];

    for window_type in windows {
        let windowed = signal_processing::apply_window(&signal.view(), window_type).unwrap();

        assert_eq!(windowed.len(), signal.len());

        for &val in windowed.iter() {
            assert!(val.is_finite());
        }

        // Windowed signal should taper to zero at edges (except rectangular)
        if window_type != "rectangular" {
            assert!(windowed[0].abs() < 0.1);
            assert!(windowed[windowed.len() - 1].abs() < 0.1);
        }
    }
}

#[test]
fn test_signal_correlation() {
    let signal1: Vec<f64> = (0..100)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin())
        .collect();

    let signal2: Vec<f64> = (0..100)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin())
        .collect();

    // Auto-correlation
    let signal1_arr = Array1::from_vec(signal1.clone());
    let signal2_arr = Array1::from_vec(signal2.clone());
    let autocorr =
        signal_processing::cross_correlate(&signal1_arr.view(), &signal1_arr.view()).unwrap();
    assert_eq!(autocorr.len(), 2 * signal1.len() - 1);

    // Cross-correlation with identical signals should equal auto-correlation
    let crosscorr =
        signal_processing::cross_correlate(&signal1_arr.view(), &signal2_arr.view()).unwrap();
    assert_eq!(crosscorr.len(), autocorr.len());

    for (a, c) in autocorr.iter().zip(crosscorr.iter()) {
        assert!((a - c).abs() < 1e-10);
    }

    // All correlation values should be finite
    for &val in autocorr.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_signal_convolution() {
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = vec![0.2, 0.6, 0.2]; // Simple smoothing kernel
    let signal_arr = Array1::from_vec(signal.clone());
    let kernel_arr = Array1::from_vec(kernel.clone());

    let convolved = signal_processing::convolve(&signal_arr.view(), &kernel_arr.view()).unwrap();

    // Convolution result length depends on mode (full, same, valid)
    assert!(convolved.len() > 0);

    for &val in convolved.iter() {
        assert!(val.is_finite());
    }

    // Test different convolution modes
    let convolved_same =
        signal_processing::convolve_mode(&signal_arr.view(), &kernel_arr.view(), "same").unwrap();
    assert_eq!(convolved_same.len(), signal.len());

    let convolved_valid =
        signal_processing::convolve_mode(&signal_arr.view(), &kernel_arr.view(), "valid").unwrap();
    assert_eq!(convolved_valid.len(), signal.len() - kernel.len() + 1);
}

#[test]
fn test_signal_error_cases() {
    let empty_signal = Array1::from_vec(vec![]);
    let short_signal = Array1::from_vec(vec![1.0]);

    // Test STFT with empty/short signals
    let stft = signal_processing::STFT::new();
    assert!(stft.transform(&empty_signal.view()).is_err());
    assert!(stft.transform(&short_signal.view()).is_err());

    // Test filter with invalid parameters
    let invalid_filter = signal_processing::BandpassFilter::new()
        .low_freq(200.0)
        .high_freq(100.0); // Invalid: high < low
    let normal_signal = Array1::from_vec(vec![1.0; 1000]);
    assert!(invalid_filter.apply(&normal_signal.view()).is_err());

    // Test wavelet with too few samples
    let wavelet = signal_processing::WaveletTransform::new().levels(10);
    let small_signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(wavelet.transform(&small_signal.view()).is_err());
}

#[test]
fn test_signal_performance() {
    // Test with a reasonably large signal to ensure good performance
    let n_samples = 10000;
    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let stft = signal_processing::STFT::new()
        .window_size(512)
        .hop_length(256);

    let start = std::time::Instant::now();
    let spectrogram = stft.transform(&signal.view()).unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time
    assert!(duration.as_secs() < 2);

    // Check output is valid
    assert!(spectrogram.nrows() > 0);
    assert!(spectrogram.ncols() > 0);

    for &val in spectrogram.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}

#[test]
fn test_signal_consistency() {
    let signal_vec: Vec<f64> = (0..512)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin())
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let stft = signal_processing::STFT::new()
        .window_size(128)
        .hop_length(64);

    // Multiple transforms should give same result
    let spec1 = stft.transform(&signal.view()).unwrap();
    let spec2 = stft.transform(&signal.view()).unwrap();

    assert_eq!(spec1.dim(), spec2.dim());

    for (s1, s2) in spec1.iter().zip(spec2.iter()) {
        assert!((s1 - s2).abs() < 1e-10);
    }
}

#[test]
fn test_signal_edge_cases() {
    // Test signal with all zeros
    let zero_signal = vec![0.0; 1000];
    let filter = signal_processing::LowpassFilter::new()
        .cutoff_freq(100.0)
        .sample_rate(1000.0);

    let zero_signal_array = Array1::from_vec(zero_signal.clone());
    let filtered = filter.apply(&zero_signal_array.view()).unwrap();

    // Should remain zeros
    for &val in filtered.iter() {
        assert!(val.abs() < 1e-10);
    }

    // Test signal with single impulse
    let mut impulse_signal_vec = vec![0.0; 100];
    impulse_signal_vec[50] = 1.0;
    let impulse_signal = Array1::from_vec(impulse_signal_vec);

    let hilbert = signal_processing::HilbertTransform::new();
    let result = hilbert.transform(&impulse_signal.view()).unwrap();

    assert_eq!(result.len(), impulse_signal.len());

    for &val in result.iter() {
        assert!(val.is_finite());
    }
}
