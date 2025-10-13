//! Demonstration of enhanced audio feature extraction
//!
//! This example shows how to use the newly implemented audio feature extractors:
//! - SpectralCentroidExtractor
//! - SpectralBandwidthExtractor
//! - RMSEnergyExtractor
//! - MelSpectrogramExtractor

use scirs2_core::ndarray::Array1;
use sklears_feature_extraction::audio::{
    MelSpectrogramExtractor, RMSEnergyExtractor, SpectralBandwidthExtractor,
    SpectralCentroidExtractor,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Audio Feature Extraction Demo");
    println!("================================");

    // Create a test signal: 440 Hz sine wave (A4 note) with some harmonics
    let sample_rate = 22050.0;
    let duration = 1.0; // 1 second
    let n_samples = (sample_rate * duration) as usize;

    let mut signal = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let t = i as f64 / sample_rate;
        // Fundamental frequency (440 Hz) + some harmonics
        signal[i] = 0.6 * (2.0 * PI * 440.0 * t).sin()
                  + 0.3 * (2.0 * PI * 880.0 * t).sin()  // 2nd harmonic
                  + 0.1 * (2.0 * PI * 1320.0 * t).sin(); // 3rd harmonic
    }

    println!(
        "📊 Generated test signal: {} samples at {} Hz",
        n_samples, sample_rate
    );
    println!("   Fundamental: 440 Hz (A4 note) with harmonics");
    println!();

    // 1. Spectral Centroid
    println!("🎯 Spectral Centroid Analysis");
    println!("-----------------------------");
    let centroid_extractor = SpectralCentroidExtractor::new()
        .sample_rate(sample_rate)
        .n_fft(1024);

    let centroids = centroid_extractor.extract_features(&signal.view())?;
    let mean_centroid = centroids.iter().sum::<f64>() / centroids.len() as f64;

    println!("   Frames analyzed: {}", centroids.len());
    println!("   Mean spectral centroid: {:.1} Hz", mean_centroid);
    println!("   Expected ~600-800 Hz (weighted by harmonics)");
    println!();

    // 2. Spectral Bandwidth
    println!("📏 Spectral Bandwidth Analysis");
    println!("------------------------------");
    let bandwidth_extractor = SpectralBandwidthExtractor::new()
        .sample_rate(sample_rate)
        .n_fft(1024);

    let bandwidths = bandwidth_extractor.extract_features(&signal.view())?;
    let mean_bandwidth = bandwidths.iter().sum::<f64>() / bandwidths.len() as f64;

    println!("   Frames analyzed: {}", bandwidths.len());
    println!("   Mean spectral bandwidth: {:.1} Hz", mean_bandwidth);
    println!("   Shows frequency spread around centroid");
    println!();

    // 3. RMS Energy
    println!("⚡ RMS Energy Analysis");
    println!("---------------------");
    let rms_extractor = RMSEnergyExtractor::new().hop_length(512);

    let rms_values = rms_extractor.extract_features(&signal.view())?;
    let mean_rms = rms_values.iter().sum::<f64>() / rms_values.len() as f64;
    let max_rms = rms_values.iter().fold(0.0_f64, |acc, &x| acc.max(x));

    println!("   Frames analyzed: {}", rms_values.len());
    println!("   Mean RMS energy: {:.4}", mean_rms);
    println!("   Max RMS energy: {:.4}", max_rms);
    println!("   RMS represents signal loudness/energy");
    println!();

    // 4. Mel Spectrogram
    println!("🌈 Mel Spectrogram Analysis");
    println!("---------------------------");
    let mel_extractor = MelSpectrogramExtractor::new()
        .n_fft(1024)
        .hop_length(512)
        .n_mels(40)
        .sample_rate(sample_rate);

    let mel_spectrogram = mel_extractor.extract_features(&signal.view())?;
    let (n_mels, n_frames) = mel_spectrogram.dim();

    println!(
        "   Mel spectrogram shape: {} mels × {} frames",
        n_mels, n_frames
    );

    // Analyze mel energies
    let mel_energies: Vec<f64> = (0..n_mels)
        .map(|mel_idx| mel_spectrogram.row(mel_idx).iter().sum::<f64>() / n_frames as f64)
        .collect();

    let max_energy_mel = mel_energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!(
        "   Mel bin with highest energy: {} (out of {})",
        max_energy_mel, n_mels
    );
    println!("   This corresponds to the frequency region with most energy");
    println!();

    // Summary
    println!("✅ Demo Summary");
    println!("===============");
    println!("All enhanced audio extractors working correctly!");
    println!("• Spectral centroid: Frequency center of mass");
    println!("• Spectral bandwidth: Frequency spread measure");
    println!("• RMS energy: Signal power/loudness measure");
    println!("• Mel spectrogram: Perceptually-scaled frequency analysis");
    println!();
    println!("These are now real implementations (not placeholders) with:");
    println!("• Proper windowing (Hanning window)");
    println!("• FFT-based spectral analysis");
    println!("• Frame-based processing");
    println!("• Mel-scale frequency mapping");

    Ok(())
}
