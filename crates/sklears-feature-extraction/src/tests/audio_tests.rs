//! Audio feature extraction tests
//!
//! This module contains tests for audio signal processing and feature extraction,
//! including MFCC, spectral features, chroma, tonnetz, tempogram, and RMS energy extraction.

use crate::audio;
use scirs2_core::ndarray::{s, Array1};

#[test]
fn test_mfcc_extractor() {
    // Create a simple sine wave signal
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note

    let n_samples = (sample_rate as f64 * duration) as usize;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (2.0 * std::f64::consts::PI * frequency * t).sin()
        })
        .collect();

    let mfcc = audio::MFCCExtractor::new()
        .n_mfcc(13)
        .n_fft(512)
        .hop_length(160)
        .n_mels(26)
        .sample_rate(sample_rate as f64);

    let signal_array = Array1::from_vec(signal);
    let features = mfcc.extract_features(&signal_array.view()).unwrap();

    // Check dimensions
    let expected_frames = (signal_array.len() - 512) / 160 + 1;
    assert_eq!(features.nrows(), expected_frames);
    assert_eq!(features.ncols(), 13); // n_mfcc

    // Check that all MFCC coefficients are finite
    for &coeff in features.iter() {
        assert!(coeff.is_finite());
    }
}

#[test]
fn test_spectral_centroid_extractor() {
    // Create a signal with known spectral properties
    let sample_rate = 8000.0;
    let n_samples = 1024;
    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * 1000.0 * t).sin() // 1kHz sine wave
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let spectral_centroid = audio::SpectralCentroidExtractor::new()
        .n_fft(512)
        .hop_length(256)
        .sample_rate(sample_rate);

    let features = spectral_centroid.extract_features(&signal.view()).unwrap();

    // Should have one centroid value per frame
    let expected_frames = (signal.len() - 512) / 256 + 1;
    assert_eq!(features.len(), expected_frames);

    // For a 1kHz sine wave, spectral centroid should be around 1000 Hz
    for &centroid in features.iter() {
        assert!(centroid.is_finite());
        assert!(centroid > 0.0);
        // For a pure sine wave, centroid should be close to the frequency
        assert!(centroid > 500.0 && centroid < 1500.0); // Allow some tolerance
    }
}

#[test]
fn test_zero_crossing_rate_extractor() {
    // Create a signal with known zero crossing properties
    let n_samples = 1000;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64;
            (2.0 * std::f64::consts::PI * t / 10.0).sin() // Slow oscillation
        })
        .collect();

    let zcr = audio::ZeroCrossingRateExtractor::new()
        .frame_length(100)
        .hop_length(50);

    let signal_array = Array1::from_vec(signal);
    let features = zcr.extract_features(&signal_array.view()).unwrap();

    // Should have zero crossing rate values per frame
    let expected_frames = (signal_array.len() - 100) / 50 + 1;
    assert_eq!(features.len(), expected_frames);

    // All ZCR values should be between 0 and 1
    for &zcr_val in features.iter() {
        assert!(zcr_val.is_finite());
        assert!(zcr_val >= 0.0 && zcr_val <= 1.0);
    }
}

#[test]
fn test_tonnetz_extractor() {
    // Create a simple signal
    let sample_rate = 22050.0;
    let n_samples = 1024;
    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * 261.6 * t).sin() // C4 note
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let tonnetz = audio::TonnetzExtractor::new()
        .n_fft(512)
        .hop_length(256)
        .sample_rate(sample_rate);

    let features = tonnetz.extract_features(&signal.view()).unwrap();

    // Tonnetz features have 6 dimensions
    let expected_frames = (signal.len() - 512) / 256 + 1;
    assert_eq!(features.nrows(), expected_frames);
    assert_eq!(features.ncols(), 6);

    // All tonnetz values should be finite and bounded
    for &val in features.iter() {
        assert!(val.is_finite());
        assert!(val >= -1.0 && val <= 1.0); // Tonnetz values are normalized
    }
}

#[test]
fn test_chroma_extractor() {
    // Create a signal with harmonic content
    let sample_rate = 22050;
    let n_samples = 2048;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Mixture of C and E (major third)
            (2.0 * std::f64::consts::PI * 261.6 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 329.6 * t).sin()
        })
        .collect();

    let chroma = audio::ChromaExtractor::new()
        .n_fft(1024)
        .hop_length(512)
        .sample_rate(sample_rate as f64)
        .n_chroma(12);

    let signal_array = Array1::from_vec(signal);
    let features = chroma.extract_features(&signal_array.view()).unwrap();

    // Chroma features have 12 dimensions (one per semitone)
    let expected_frames = (signal_array.len() - 1024) / 512 + 1;
    assert_eq!(features.nrows(), expected_frames);
    assert_eq!(features.ncols(), 12);

    // All chroma values should be non-negative
    for &val in features.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}

#[test]
fn test_tempogram_extractor() {
    // Create a signal with rhythmic content (simple beat pattern)
    let sample_rate = 22050.0;
    let beat_frequency = 2.0; // 2 beats per second = 120 BPM
    let n_samples = sample_rate as usize; // 1 second of audio

    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let beat = (2.0 * std::f64::consts::PI * beat_frequency * t).sin();
            if beat > 0.0 {
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() // A4 on beat
            } else {
                0.0
            }
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let tempogram = audio::TempogramExtractor::new()
        .hop_length(512)
        .sample_rate(sample_rate)
        .tempo_min(60)
        .tempo_max(200)
        .n_tempo_bins(40);

    let features = tempogram.extract_features(&signal.view()).unwrap();

    // Tempogram should have tempo bins as columns
    let expected_frames = (signal.len() - 1024) / 512 + 1; // Using default n_fft=1024
    assert_eq!(features.nrows(), expected_frames);
    assert_eq!(features.ncols(), 40); // n_tempo_bins

    // All tempogram values should be non-negative
    for &val in features.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}

#[test]
fn test_rms_energy_extractor() {
    // Create a signal with varying amplitude
    let n_samples = 1000;
    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let amplitude = if i < 500 { 0.5 } else { 1.0 }; // Step change in amplitude
            amplitude * (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin()
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let rms = audio::RMSEnergyExtractor::new()
        .frame_length(100)
        .hop_length(50);

    let features = rms.extract_features(&signal.view()).unwrap();

    // Should have RMS values per frame
    let expected_frames = (signal.len() - 100) / 50 + 1;
    assert_eq!(features.len(), expected_frames);

    // All RMS values should be non-negative
    for &rms_val in features.iter() {
        assert!(rms_val.is_finite());
        assert!(rms_val >= 0.0);
    }

    // The second half should have higher RMS values due to higher amplitude
    let mid_point = features.len() / 2;
    let first_half_sum: f64 = features.slice(s![..mid_point]).iter().sum();
    let second_half_sum: f64 = features.slice(s![mid_point..]).iter().sum();
    let first_half_avg = first_half_sum / mid_point as f64;
    let second_half_avg = second_half_sum / (features.len() - mid_point) as f64;

    assert!(second_half_avg > first_half_avg);
}

#[test]
fn test_spectral_rolloff_extractor() {
    // Create a broadband signal
    let sample_rate = 22050;
    let n_samples = 2048;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Mix of different frequencies
            (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 1760.0 * t).sin()
        })
        .collect();

    let rolloff = audio::SpectralRolloffExtractor::new()
        .n_fft(1024)
        .hop_length(512)
        .sample_rate(sample_rate as f64);
    // Note: rolloff_percent method may need to be implemented

    let signal_array = Array1::from_vec(signal);
    let features = rolloff.extract_features(&signal_array.view()).unwrap();

    let expected_frames = (signal_array.len() - 1024) / 512 + 1;
    assert_eq!(features.len(), expected_frames);

    // All rolloff frequencies should be positive and within Nyquist limit
    for &freq in features.iter() {
        assert!(freq.is_finite());
        assert!(freq > 0.0);
        assert!(freq <= sample_rate as f64 / 2.0);
    }
}

#[test]
fn test_spectral_bandwidth_extractor() {
    // Create a signal with controlled bandwidth
    let sample_rate = 22050;
    let n_samples = 2048;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Narrow bandwidth signal
            (2.0 * std::f64::consts::PI * 1000.0 * t).sin()
        })
        .collect();

    let bandwidth = audio::SpectralBandwidthExtractor::new()
        .n_fft(1024)
        .hop_length(512)
        .sample_rate(sample_rate as f64);

    let signal_array = Array1::from_vec(signal);
    let features = bandwidth.extract_features(&signal_array.view()).unwrap();

    let expected_frames = (signal_array.len() - 1024) / 512 + 1;
    assert_eq!(features.len(), expected_frames);

    // All bandwidth values should be non-negative
    for &bw in features.iter() {
        assert!(bw.is_finite());
        assert!(bw >= 0.0);
    }
}

#[test]
fn test_mel_spectrogram_extractor() {
    let sample_rate = 16000;
    let n_samples = 1024;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let mel_spec = audio::MelSpectrogramExtractor::new()
        .n_fft(512)
        .hop_length(256)
        .n_mels(40)
        .sample_rate(sample_rate as f64);

    let signal_array = Array1::from_vec(signal);
    let features = mel_spec.extract_features(&signal_array.view()).unwrap();

    let expected_frames = (signal_array.len() - 512) / 256 + 1;
    assert_eq!(features.nrows(), expected_frames);
    assert_eq!(features.ncols(), 40); // n_mels

    // All mel spectrogram values should be non-negative
    for &val in features.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}

#[test]
fn test_pitch_extractor() {
    // Create a signal with known pitch
    let sample_rate = 16000.0;
    let frequency = 440.0; // A4
    let n_samples = 2048;

    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * frequency * t).sin()
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let pitch = audio::PitchExtractor::new()
        .method("autocorr".to_string())
        .sample_rate(sample_rate)
        .hop_length(512);

    let features = pitch.extract_features(&signal.view()).unwrap();

    let expected_frames = (signal.len() - 1024) / 512 + 1;
    assert_eq!(features.len(), expected_frames);

    // Check that detected pitch is close to the input frequency
    for &f0 in features.iter() {
        if f0 > 0.0 {
            // Only check voiced frames
            assert!(f0.is_finite());
            assert!((f0 - frequency).abs() < 50.0); // Allow some tolerance
        }
    }
}

#[test]
fn test_onset_detector() {
    // Create a signal with clear onset
    let sample_rate = 22050.0;
    let n_samples = 4410; // 0.2 seconds

    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            if i < n_samples / 2 {
                0.0 // Silence
            } else {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() // Start of tone
            }
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let onset = audio::OnsetDetector::new()
        .hop_length(512)
        .sample_rate(sample_rate)
        .method("energy".to_string());

    let onsets = onset.detect_onsets(&signal.view()).unwrap();

    // Should detect at least one onset around the middle
    assert!(onsets.len() > 0);

    // Check that onset times are reasonable
    for &onset_time in onsets.iter() {
        assert!(onset_time >= 0.0);
        assert!(onset_time < n_samples as f64 / sample_rate as f64);
    }
}

#[test]
fn test_harmonic_percussive_separation() {
    // Create a signal with both harmonic and percussive components
    let sample_rate = 22050.0;
    let n_samples = 2048;

    let signal_vec: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // Harmonic component (sine wave)
            let harmonic = (2.0 * std::f64::consts::PI * 440.0 * t).sin();
            // Percussive component (short burst)
            let percussive = if i % 512 < 10 { 1.0 } else { 0.0 };
            harmonic + 0.5 * percussive
        })
        .collect();
    let signal = Array1::from_vec(signal_vec);

    let hps = audio::HarmonicPercussiveSeparation::new()
        .n_fft(1024)
        .hop_length(256)
        .kernel_size(31)
        .power(2.0);

    let (harmonic, percussive) = hps.separate(&signal.view()).unwrap();

    assert_eq!(harmonic.len(), signal.len());
    assert_eq!(percussive.len(), signal.len());

    // All separated components should be finite
    for &val in harmonic.iter() {
        assert!(val.is_finite());
    }
    for &val in percussive.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_audio_feature_error_cases() {
    let empty_signal: Vec<f64> = vec![];
    let short_signal = vec![1.0, 2.0]; // Too short for most operations

    // Test MFCC with empty signal
    let mfcc = audio::MFCCExtractor::new();
    let empty_arr = Array1::from_vec(empty_signal.clone());
    assert!(mfcc.extract_features(&empty_arr.view()).is_err());
    let short_arr = Array1::from_vec(short_signal.clone());
    assert!(mfcc.extract_features(&short_arr.view()).is_err());

    // Test spectral centroid with empty signal
    let centroid = audio::SpectralCentroidExtractor::new();
    let empty_arr2 = Array1::from_vec(empty_signal.clone());
    assert!(centroid.extract_features(&empty_arr2.view()).is_err());

    // Test with invalid parameters
    let invalid_mfcc = audio::MFCCExtractor::new()
        .n_mfcc(0) // Invalid
        .n_fft(512);
    let normal_signal = vec![0.0; 1024];
    let normal_arr = Array1::from_vec(normal_signal);
    assert!(invalid_mfcc.extract_features(&normal_arr.view()).is_err());
}

#[test]
fn test_audio_feature_consistency() {
    let signal: Vec<f64> = (0..2048)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin())
        .collect();

    let mfcc = audio::MFCCExtractor::new()
        .n_mfcc(13)
        .n_fft(512)
        .hop_length(256);

    // Extract features multiple times - should be consistent
    let signal_arr = Array1::from_vec(signal);
    let features1 = mfcc.extract_features(&signal_arr.view()).unwrap();
    let features2 = mfcc.extract_features(&signal_arr.view()).unwrap();

    assert_eq!(features1.dim(), features2.dim());

    for (f1, f2) in features1.iter().zip(features2.iter()) {
        assert!((f1 - f2).abs() < 1e-10);
    }
}

#[test]
fn test_audio_feature_batch_processing() {
    // Test processing multiple signals in batch
    let signals: Vec<Array1<f64>> = (0..3)
        .map(|i| {
            let signal_vec: Vec<f64> = (0..1024)
                .map(|j| {
                    let freq = 440.0 * (i + 1) as f64; // Different frequencies
                    let t = j as f64 / 16000.0;
                    (2.0 * std::f64::consts::PI * freq * t).sin()
                })
                .collect();
            Array1::from_vec(signal_vec)
        })
        .collect();

    let mfcc = audio::MFCCExtractor::new()
        .n_mfcc(13)
        .n_fft(256)
        .hop_length(128);

    let batch_features = mfcc.extract_features_batch(&signals).unwrap();
    assert_eq!(batch_features.len(), 3);

    for features in batch_features.iter() {
        assert_eq!(features.ncols(), 13);
        assert!(features.nrows() > 0);

        for &val in features.iter() {
            assert!(val.is_finite());
        }
    }
}
